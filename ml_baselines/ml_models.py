# fraud_detection_ml_fixed.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML библиотеки
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif

# Модели
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Метрики
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve, precision_recall_curve,
                           average_precision_score)

# Балансировка классов
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Оптимизация
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import joblib
import json
import pickle

# Настройки отображения
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
plt.style.use('seaborn-v0_8-darkgrid')

# ======================================================================
# 1. ФУНКЦИИ ДЛЯ СОЗДАНИЯ НОВЫХ ПРИЗНАКОВ (FEATURE ENGINEERING) - ИСПРАВЛЕНЫ
# ======================================================================

def load_and_prepare_data():
    """Загрузка и объединение данных"""
    print("Загрузка данных...")
    
    # Загрузка транзакций
    transactions = pd.read_parquet('transaction_fraud_data.parquet')
    
    # Загрузка курсов валют
    currency_rates = pd.read_parquet('historical_currency_exchange.parquet')
    
    # Базовая предобработка
    transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
    currency_rates['date'] = pd.to_datetime(currency_rates['date'])
    
    print(f"Загружено {len(transactions)} транзакций")
    print(f"Загружено {len(currency_rates)} дней курсов валют")
    
    return transactions, currency_rates

def extract_datetime_features(df):
    """Извлечение признаков из даты и времени"""
    print("Извлечение временных признаков...")
    
    df = df.copy()
    
    # Базовые временные признаки
    df['transaction_date'] = df['timestamp'].dt.date
    df['transaction_hour'] = df['timestamp'].dt.hour
    df['transaction_day'] = df['timestamp'].dt.day
    df['transaction_dow'] = df['timestamp'].dt.dayofweek  # 0=Понедельник
    df['transaction_month'] = df['timestamp'].dt.month
    df['transaction_week'] = df['timestamp'].dt.isocalendar().week
    
    # Часть дня
    def get_time_of_day(hour):
        if 0 <= hour < 6:
            return 'night'
        elif 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        else:
            return 'evening'
    
    df['time_of_day'] = df['transaction_hour'].apply(get_time_of_day)
    
    # Признак времени с полуночи в секундах
    df['seconds_from_midnight'] = df['timestamp'].dt.hour * 3600 + df['timestamp'].dt.minute * 60 + df['timestamp'].dt.second
    
    # Выходной/будний день (уже есть в данных, но пересчитываем для согласованности)
    df['is_weekend_calc'] = df['transaction_dow'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Час в синусоидальном представлении (для учета цикличности)
    df['hour_sin'] = np.sin(2 * np.pi * df['transaction_hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['transaction_hour']/24)
    
    # День недели в синусоидальном представлении
    df['dow_sin'] = np.sin(2 * np.pi * df['transaction_dow']/7)
    df['dow_cos'] = np.cos(2 * np.pi * df['transaction_dow']/7)
    
    return df

def create_behavioral_features(df):
    """Создание поведенческих признаков"""
    print("Создание поведенческих признаков...")
    
    df = df.copy()
    
    # Распаковка last_hour_activity
    if 'last_hour_activity' in df.columns:
        # Проверяем тип first_hour_activity
        first_val = df['last_hour_activity'].iloc[0] if not df['last_hour_activity'].isna().all() else None
        
        if isinstance(first_val, dict):
            # Если это словарь
            activity_cols = ['num_transactions', 'total_amount', 'unique_merchants', 
                            'unique_countries', 'max_single_amount']
            
            for col in activity_cols:
                df[f'last_hour_{col}'] = df['last_hour_activity'].apply(
                    lambda x: x.get(col, np.nan) if isinstance(x, dict) else np.nan
                )
        else:
            # Если это структура pandas/parquet
            try:
                # Пробуем получить доступ через атрибуты
                df['last_hour_num_transactions'] = df['last_hour_activity'].apply(lambda x: x.num_transactions if pd.notnull(x) else np.nan)
                df['last_hour_total_amount'] = df['last_hour_activity'].apply(lambda x: x.total_amount if pd.notnull(x) else np.nan)
                df['last_hour_unique_merchants'] = df['last_hour_activity'].apply(lambda x: x.unique_merchants if pd.notnull(x) else np.nan)
                df['last_hour_unique_countries'] = df['last_hour_activity'].apply(lambda x: x.unique_countries if pd.notnull(x) else np.nan)
                df['last_hour_max_single_amount'] = df['last_hour_activity'].apply(lambda x: x.max_single_amount if pd.notnull(x) else np.nan)
            except:
                # Если не получается, пропускаем
                print("Не удалось распаковать last_hour_activity")
                pass
    
    # Признаки интенсивности транзакций (если созданы соответствующие колонки)
    if 'last_hour_num_transactions' in df.columns and 'last_hour_total_amount' in df.columns:
        df['avg_amount_per_trans_last_hour'] = np.where(
            df['last_hour_num_transactions'] > 0,
            df['last_hour_total_amount'] / df['last_hour_num_transactions'],
            0
        )
        
    if 'last_hour_num_transactions' in df.columns and 'last_hour_unique_merchants' in df.columns:
        df['merchant_diversity_last_hour'] = np.where(
            df['last_hour_num_transactions'] > 0,
            df['last_hour_unique_merchants'] / df['last_hour_num_transactions'],
            0
        )
        
    if 'last_hour_num_transactions' in df.columns and 'last_hour_unique_countries' in df.columns:
        df['country_diversity_last_hour'] = np.where(
            df['last_hour_num_transactions'] > 0,
            df['last_hour_unique_countries'] / df['last_hour_num_transactions'],
            0
        )
    
    # Отношение текущей суммы к средней за последний час
    if 'last_hour_total_amount' in df.columns and 'last_hour_num_transactions' in df.columns:
        df['amount_vs_avg_last_hour'] = np.where(
            (df['last_hour_num_transactions'] > 0) & (df['last_hour_total_amount'] > 0),
            df['amount'] / (df['last_hour_total_amount'] / df['last_hour_num_transactions']),
            np.nan
        )
    
    # Отношение текущей суммы к максимальной за последний час
    if 'last_hour_max_single_amount' in df.columns:
        df['amount_vs_max_last_hour'] = np.where(
            df['last_hour_max_single_amount'] > 0,
            df['amount'] / df['last_hour_max_single_amount'],
            np.nan
        )
    
    return df

def create_customer_features(df):
    """Создание признаков на уровне клиента (исправленная версия)"""
    print("Создание клиентских признаков...")
    
    df = df.copy()
    
    # Убедимся, что нужные колонки существуют
    if 'customer_id' not in df.columns:
        print("Ошибка: нет колонки customer_id")
        return df
    
    # Создаем временную копию для вычислений
    temp_df = df.copy()
    
    # Группировка по клиенту для создания агрегатов
    try:
        customer_stats = temp_df.groupby('customer_id').agg({
            'transaction_id': 'count',
            'amount': ['mean', 'std', 'min', 'max', 'sum']
        }).reset_index()
        
        # Выравниваем multi-index columns
        customer_stats.columns = ['customer_id', 
                                 'customer_total_transactions',
                                 'customer_avg_amount',
                                 'customer_std_amount',
                                 'customer_min_amount',
                                 'customer_max_amount',
                                 'customer_total_amount']
        
        # Вычисляем дополнительные метрики
        customer_stats['customer_std_amount'] = customer_stats['customer_std_amount'].fillna(0)
        customer_stats['customer_avg_transaction_frequency'] = customer_stats['customer_total_transactions'] / 30  # предполагаем 30 дней
        
        # Объединяем обратно
        df = df.merge(customer_stats, on='customer_id', how='left')
        
        # Признаки отклонения от обычного поведения клиента
        df['amount_deviation_from_customer_avg'] = (df['amount'] - df['customer_avg_amount']) / df['customer_std_amount'].replace(0, 1)
        df['is_unusual_amount_for_customer'] = (abs(df['amount_deviation_from_customer_avg']) > 3).astype(int)
        
        # Отношение к максимальной сумме клиента
        df['amount_vs_customer_max'] = df['amount'] / df['customer_max_amount'].replace(0, 1)
        
    except Exception as e:
        print(f"Ошибка при создании клиентских признаков: {e}")
    
    return df

def create_vendor_features(df):
    """Создание признаков на уровне вендора (исправленная версия)"""
    print("Создание признаков вендора...")
    
    df = df.copy()
    
    if 'vendor' not in df.columns:
        print("Ошибка: нет колонки vendor")
        return df
    
    # Частота транзакций у вендора
    try:
        vendor_stats = df.groupby('vendor').agg({
            'transaction_id': 'count',
            'amount': ['mean', 'std']
        }).reset_index()
        
        vendor_stats.columns = ['vendor', 
                               'vendor_transaction_count',
                               'vendor_avg_amount',
                               'vendor_std_amount']
        
        # Объединяем обратно
        df = df.merge(vendor_stats, on='vendor', how='left')
        
        # Отклонение суммы от средней для вендора
        df['amount_deviation_from_vendor_avg'] = (df['amount'] - df['vendor_avg_amount']) / df['vendor_std_amount'].replace(0, 1)
        
    except Exception as e:
        print(f"Ошибка при создании вендорских признаков: {e}")
    
    return df

def create_device_features(df):
    """Создание признаков на уровне устройства (исправленная версия)"""
    print("Создание признаков устройства...")
    
    df = df.copy()
    
    if 'device_fingerprint' in df.columns and not df['device_fingerprint'].isna().all():
        try:
            device_stats = df.groupby('device_fingerprint').agg({
                'transaction_id': 'count',
                'customer_id': 'nunique'
            }).reset_index()
            
            device_stats.columns = ['device_fingerprint',
                                   'device_usage_count',
                                   'unique_customers_per_device']
            
            df = df.merge(device_stats, on='device_fingerprint', how='left')
            
            # Подозрительное устройство (много клиентов)
            df['is_suspicious_device'] = (df['unique_customers_per_device'] > 3).astype(int)
            
        except Exception as e:
            print(f"Ошибка при создании признаков устройства: {e}")
    
    return df

def create_interaction_features(df):
    """Создание взаимодействий между признаками"""
    print("Создание признаков взаимодействия...")
    
    df = df.copy()
    
    try:
        # Взаимодействие типа канала и присутствия карты
        if 'channel' in df.columns and 'is_card_present' in df.columns:
            df['channel_card_present_interaction'] = df['channel'].astype(str) + '_' + df['is_card_present'].astype(str)
        
        # Взаимодействие страны и типа карты
        if 'country' in df.columns and 'card_type' in df.columns:
            df['country_card_type_interaction'] = df['country'].astype(str) + '_' + df['card_type'].astype(str)
        
        # Взаимодействие времени суток и категории вендора
        if 'time_of_day' in df.columns and 'vendor_category' in df.columns:
            df['time_vendor_category_interaction'] = df['time_of_day'].astype(str) + '_' + df['vendor_category'].astype(str)
        
        # Бинарные взаимодействия для высокого риска
        if 'is_high_risk_vendor' in df.columns and 'is_outside_home_country' in df.columns:
            df['high_risk_outside_country'] = (df['is_high_risk_vendor'] & df['is_outside_home_country']).astype(int)
        
        if 'is_high_risk_vendor' in df.columns and 'is_weekend' in df.columns:
            df['high_risk_weekend'] = (df['is_high_risk_vendor'] & df['is_weekend']).astype(int)
        
        if 'is_outside_home_country' in df.columns and 'is_weekend' in df.columns:
            df['outside_country_weekend'] = (df['is_outside_home_country'] & df['is_weekend']).astype(int)
        
        # Взаимодействие суммы и риска
        if 'amount' in df.columns and 'is_high_risk_vendor' in df.columns:
            df['amount_risk_interaction'] = df['amount'] * df['is_high_risk_vendor']
            
    except Exception as e:
        print(f"Ошибка при создании признаков взаимодействия: {e}")
    
    return df

def create_aggregated_features_simple(df, window_sizes=[5, 10, 20]):
    """Упрощенная версия создания агрегированных признаков (без временных окон)"""
    print("Создание упрощенных агрегированных признаков...")
    
    df = df.copy()
    
    # Сортируем по времени
    df = df.sort_values(['customer_id', 'timestamp'])
    
    for window in window_sizes:
        # Простые скользящие агрегаты по транзакциям (не по времени)
        df[f'customer_avg_amount_last_{window}'] = df.groupby('customer_id')['amount'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        
        df[f'customer_trans_count_last_{window}'] = df.groupby('customer_id')['transaction_id'].transform(
            lambda x: x.rolling(window, min_periods=1).count()
        )
        
        # Уникальные вендоры за последние N транзакций
        df[f'customer_unique_vendors_last_{window}'] = df.groupby('customer_id')['vendor'].transform(
            lambda x: x.rolling(window, min_periods=1).apply(lambda y: len(set(y.dropna())))
        )
    
    # Заполняем пропуски
    for window in window_sizes:
        cols = [f'customer_avg_amount_last_{window}', 
                f'customer_trans_count_last_{window}',
                f'customer_unique_vendors_last_{window}']
        
        for col in cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
    
    return df

def create_all_features(transactions, currency_rates=None):
    """Основная функция создания всех признаков (исправленная)"""
    print("=" * 80)
    print("СОЗДАНИЕ ПРИЗНАКОВ ДЛЯ ML МОДЕЛИ")
    print("=" * 80)
    
    df = transactions.copy()
    
    print("1. Извлечение временных признаков...")
    df = extract_datetime_features(df)
    
    print("2. Создание поведенческих признаков...")
    df = create_behavioral_features(df)
    
    # Пропускаем валютные признаки для упрощения
    # if currency_rates is not None:
    #     df = create_currency_features(df, currency_rates)
    
    print("3. Создание клиентских признаков...")
    df = create_customer_features(df)
    
    print("4. Создание признаков вендора...")
    df = create_vendor_features(df)
    
    print("5. Создание признаков устройства...")
    df = create_device_features(df)
    
    print("6. Создание признаков взаимодействия...")
    df = create_interaction_features(df)
    
    print("7. Создание агрегированных признаков...")
    df = create_aggregated_features_simple(df)
    
    print(f"\nИтоговая статистика:")
    print(f"  Исходное количество признаков: {len(transactions.columns)}")
    print(f"  Новое количество признаков: {len(df.columns)}")
    print(f"  Добавлено {len(df.columns) - len(transactions.columns)} новых признаков")
    
    # Проверяем пропуски
    missing_values = df.isnull().sum().sum()
    print(f"  Всего пропущенных значений: {missing_values}")
    
    return df

# ======================================================================
# 2. ПОДГОТОВКА ДАННЫХ ДЛЯ ML (УПРОЩЕННАЯ)
# ======================================================================

def prepare_ml_data_simple(df, target_col='is_fraud', test_size=0.2, random_state=42):
    """Упрощенная подготовка данных для ML"""
    print("\n" + "=" * 80)
    print("УПРОЩЕННАЯ ПОДГОТОВКА ДАННЫХ ДЛЯ ML")
    print("=" * 80)
    
    # Создаем копию
    df_ml = df.copy()
    
    # Определяем колонки для удаления
    columns_to_drop = []
    
    # Удаляем ненужные колонки (только если они существуют)
    potential_drop_cols = [
        'transaction_id', 'customer_id', 'card_number',
        'timestamp', 'vendor', 'device_fingerprint',
        'ip_address', 'last_hour_activity', 'transaction_date',
        'device', 'city'  # добавляем потенциально проблемные колонки
    ]
    
    for col in potential_drop_cols:
        if col in df_ml.columns:
            columns_to_drop.append(col)
    
    print(f"Удаляем колонки: {columns_to_drop}")
    df_ml = df_ml.drop(columns=columns_to_drop, errors='ignore')
    
    # Удаляем колонки с большим количеством пропусков (>50%)
    missing_percentage = df_ml.isnull().sum() / len(df_ml)
    high_missing_cols = missing_percentage[missing_percentage > 0.5].index.tolist()
    
    if high_missing_cols:
        print(f"Удаляем колонки с >50% пропусков: {high_missing_cols}")
        df_ml = df_ml.drop(columns=high_missing_cols, errors='ignore')
    
    # Проверяем, есть ли целевая переменная
    if target_col not in df_ml.columns:
        print(f"ОШИБКА: Целевая переменная '{target_col}' не найдена в данных")
        print(f"Доступные колонки: {list(df_ml.columns)}")
        return None, None, None, None, None, None
    
    # Целевая переменная
    y = df_ml[target_col]
    
    # Преобразуем булевые в int для целевой переменной
    if y.dtype == bool:
        y = y.astype(int)
    
    X = df_ml.drop(columns=[target_col], errors='ignore')
    
    # Разделяем данные
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except:
        # Если не удалось стратифицировать (например, мало фродовых транзакций)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
    print(f"Размеры данных:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    
    if len(y_train.shape) > 0:
        print(f"  y_train распределение: {dict(y_train.value_counts())}")
        print(f"  y_test распределение: {dict(y_test.value_counts())}")
    
    # Определяем типы признаков
    numeric_features = X_train.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()
    
    print(f"\nКоличественные признаки: {len(numeric_features)}")
    print(f"Категориальные признаки: {len(categorical_features)}")
    
    return X_train, X_test, y_train, y_test, numeric_features, categorical_features

def create_preprocessing_pipeline_simple(numeric_features, categorical_features):
    """Упрощенный пайплайн предобработки"""
    print("\nСоздание упрощенного пайплайна предобработки...")
    
    # Препроцессинг для числовых признаков
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Препроцессинг для категориальных признаков
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Объединяем в ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

# ======================================================================
# 3. ОБУЧЕНИЕ ML МОДЕЛЕЙ (УПРОЩЕННАЯ ВЕРСИЯ)
# ======================================================================

def train_simple_models(X_train, X_test, y_train, y_test, preprocessor):
    """Обучение упрощенных ML моделей"""
    print("\n" + "=" * 80)
    print("ОБУЧЕНИЕ УПРОЩЕННЫХ ML МОДЕЛЕЙ")
    print("=" * 80)
    
    # Определяем модели
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
        'XGBoost': XGBClassifier(random_state=42, n_estimators=100, use_label_encoder=False, eval_metric='logloss'),
    }
    
    # Результаты
    results = {}
    
    # Обучаем каждую модель
    for model_name, model in models.items():
        print(f"\nОбучение {model_name}...")
        
        try:
            # Создаем пайплайн
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
            # Обучение
            start_time = datetime.now()
            pipeline.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Предсказания
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            
            # Расчет метрик
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # ROC-AUC (может не работать если только один класс в y_test)
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            except:
                roc_auc = 0.5  # случайное угадывание
            
            # Average Precision
            try:
                avg_precision = average_precision_score(y_test, y_pred_proba)
            except:
                avg_precision = y_test.mean()  # baseline
            
            # Сохраняем результаты
            results[model_name] = {
                'model': pipeline,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'avg_precision': avg_precision,
                'training_time': training_time,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  ROC-AUC: {roc_auc:.4f}")
            print(f"  Время обучения: {training_time:.2f} сек")
            
        except Exception as e:
            print(f"  Ошибка при обучении {model_name}: {e}")
            continue
    
    return results

def evaluate_simple_models(results, y_test):
    """Оценка и сравнение моделей"""
    if not results:
        print("Нет результатов для оценки")
        return None
    
    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 80)
    
    # Создаем DataFrame с результатами
    metrics_data = {}
    
    for model_name, result in results.items():
        metrics_data[model_name] = {
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1-Score': result['f1'],
            'ROC-AUC': result['roc_auc'],
            'Avg Precision': result['avg_precision'],
            'Training Time (s)': result['training_time']
        }
    
    metrics_df = pd.DataFrame(metrics_data).T
    
    # Сортируем по F1-Score
    metrics_df = metrics_df.sort_values('F1-Score', ascending=False)
    
    print("\nСравнение метрик всех моделей:")
    print("-" * 80)
    print(metrics_df.round(4))
    
    # Визуализация сравнения моделей (если есть хотя бы 2 модели)
    if len(metrics_df) >= 2:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Avg Precision']
        
        for idx, metric in enumerate(metrics_to_plot):
            if idx >= len(axes):
                break
                
            if metric in metrics_df.columns:
                axes[idx].barh(range(len(metrics_df)), metrics_df[metric].values)
                axes[idx].set_yticks(range(len(metrics_df)))
                axes[idx].set_yticklabels(metrics_df.index, fontsize=9)
                axes[idx].set_xlabel(metric)
                axes[idx].set_title(f'{metric} по моделям')
                axes[idx].invert_yaxis()
        
        # Удаляем лишние subplots
        for idx in range(len(metrics_to_plot), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.show()
    
    return metrics_df

def plot_simple_confusion_matrices(results, y_test, top_n=2):
    """Визуализация матриц ошибок для лучших моделей"""
    if not results:
        return
    
    print("\n" + "=" * 80)
    print("МАТРИЦЫ ОШИБОК ЛУЧШИХ МОДЕЛЕЙ")
    print("=" * 80)
    
    # Выбираем топ-N моделей по F1-Score
    top_models = sorted(results.keys(), 
                       key=lambda x: results[x]['f1'], 
                       reverse=True)[:min(top_n, len(results))]
    
    if not top_models:
        return
    
    # Создаем подграфики
    n_rows = min(2, len(top_models))
    n_cols = min(2, len(top_models))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))
    
    # Если axes не является массивом, делаем его массивом
    if len(top_models) == 1:
        axes = np.array([axes])
    
    axes = axes.flatten()
    
    for idx, model_name in enumerate(top_models):
        if idx >= len(axes):
            break
        
        y_pred = results[model_name]['y_pred']
        cm = confusion_matrix(y_test, y_pred)
        
        # Нормализованная матрица ошибок
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Legit', 'Fraud'],
                   yticklabels=['Legit', 'Fraud'],
                   ax=axes[idx])
        
        axes[idx].set_title(f'{model_name}\nF1: {results[model_name]["f1"]:.3f}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    # Удаляем лишние subplots
    for idx in range(len(top_models), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()

def analyze_feature_importance_simple(best_model, X_train, categorical_features, numeric_features, top_n=15):
    """Упрощенный анализ важности признаков"""
    print("\n" + "=" * 80)
    print("АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ")
    print("=" * 80)
    
    try:
        # Получаем предобработчик и модель из пайплайна
        preprocessor = best_model.named_steps['preprocessor']
        model = best_model.named_steps['classifier']
        
        # Получаем имена признаков после преобразования
        if hasattr(preprocessor, 'transformers_'):
            # Для старой версии sklearn
            transformers = preprocessor.transformers_
        else:
            # Для новой версии sklearn
            transformers = preprocessor.named_transformers_
        
        # Получаем имена категориальных признаков после one-hot кодирования
        cat_transformer = None
        for name, trans, cols in transformers:
            if name == 'cat':
                cat_transformer = trans
                break
        
        if cat_transformer and hasattr(cat_transformer, 'named_steps'):
            onehot = cat_transformer.named_steps['onehot']
            if hasattr(onehot, 'get_feature_names_out'):
                categorical_feature_names = onehot.get_feature_names_out(categorical_features)
            else:
                # Приблизительное создание имен
                categorical_feature_names = []
                for col in categorical_features:
                    # Получаем уникальные значения
                    unique_vals = X_train[col].dropna().unique()
                    for val in unique_vals:
                        categorical_feature_names.append(f"{col}_{val}")
        else:
            categorical_feature_names = []
        
        all_feature_names = list(numeric_features) + list(categorical_feature_names)
        
        # Проверяем, поддерживает ли модель важность признаков
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Убедимся, что количество признаков совпадает
            n_features = min(len(importances), len(all_feature_names))
            
            # Создаем DataFrame с важностью
            feature_importance_df = pd.DataFrame({
                'feature': all_feature_names[:n_features],
                'importance': importances[:n_features]
            })
            
            # Сортируем по важности
            feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
            
            print(f"\nТоп-{min(top_n, len(feature_importance_df))} самых важных признаков:")
            print("-" * 80)
            print(feature_importance_df.head(min(top_n, len(feature_importance_df))).to_string(index=False))
            
            # Визуализация важности признаков
            plt.figure(figsize=(10, 8))
            top_features = feature_importance_df.head(min(top_n, len(feature_importance_df)))
            plt.barh(range(len(top_features)), top_features['importance'].values)
            plt.yticks(range(len(top_features)), top_features['feature'].values, fontsize=9)
            plt.xlabel('Важность признака')
            plt.title(f'Топ-{len(top_features)} самых важных признаков')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
            return feature_importance_df
        
        else:
            print("Модель не поддерживает важность признаков или их извлечение")
            return None
            
    except Exception as e:
        print(f"Ошибка при анализе важности признаков: {e}")
        return None

# ======================================================================
# 4. ОСНОВНАЯ ФУНКЦИЯ (УПРОЩЕННАЯ)
# ======================================================================

def main_simple():
    """Упрощенная основная функция выполнения ML пайплайна"""
    print("=" * 80)
    print("УПРОЩЕННЫЙ ML ПАЙПЛАЙН ДЛЯ ДЕТЕКЦИИ МОШЕННИЧЕСКИХ ТРАНЗАКЦИЙ")
    print("=" * 80)
    
    try:
        # Шаг 1: Загрузка данных
        transactions, currency_rates = load_and_prepare_data()
        
        # Шаг 2: Создание признаков (упрощенное)
        print("\nСоздание признаков...")
        df_with_features = create_all_features(transactions)
        
        # Шаг 3: Подготовка данных для ML
        X_train, X_test, y_train, y_test, numeric_features, categorical_features = prepare_ml_data_simple(
            df_with_features, test_size=0.2, random_state=42
        )
        
        if X_train is None:
            print("Ошибка при подготовке данных. Завершение работы.")
            return
        
        # Шаг 4: Создание пайплайна предобработки
        preprocessor = create_preprocessing_pipeline_simple(numeric_features, categorical_features)
        
        # Шаг 5: Обучение моделей
        results = train_simple_models(X_train, X_test, y_train, y_test, preprocessor)
        
        if not results:
            print("Не удалось обучить ни одну модель. Завершение работы.")
            return
        
        # Шаг 6: Оценка моделей
        metrics_df = evaluate_simple_models(results, y_test)
        
        if metrics_df is not None and not metrics_df.empty:
            # Шаг 7: Визуализация результатов
            plot_simple_confusion_matrices(results, y_test, top_n=2)
            
            # Шаг 8: Выбор лучшей модели
            best_model_name = metrics_df.index[0]
            best_model = results[best_model_name]['model']
            
            print(f"\n" + "=" * 80)
            print(f"ЛУЧШАЯ МОДЕЛЬ: {best_model_name}")
            print("=" * 80)
            print(f"F1-Score: {results[best_model_name]['f1']:.4f}")
            print(f"Precision: {results[best_model_name]['precision']:.4f}")
            print(f"Recall: {results[best_model_name]['recall']:.4f}")
            
            # Шаг 9: Анализ важности признаков
            analyze_feature_importance_simple(
                best_model, X_train, categorical_features, numeric_features, top_n=15
            )
            
            # Шаг 10: Сохранение лучшей модели
            try:
                joblib.dump(best_model, 'simple_fraud_model.pkl')
                print("\nМодель сохранена в simple_fraud_model.pkl")
            except Exception as e:
                print(f"Ошибка при сохранении модели: {e}")
            
            # Шаг 11: Генерация отчета
            generate_simple_report(results, metrics_df, best_model_name, X_test, y_test)
        
    except Exception as e:
        print(f"Критическая ошибка в основном пайплайне: {e}")
        import traceback
        traceback.print_exc()

def generate_simple_report(results, metrics_df, best_model_name, X_test, y_test):
    """Генерация упрощенного отчета"""
    print("\n" + "=" * 80)
    print("ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 80)
    
    if best_model_name not in results:
        print("Лучшая модель не найдена в результатах")
        return
    
    best_result = results[best_model_name]
    
    # Рассчитываем дополнительные метрики
    cm = confusion_matrix(y_test, best_result['y_pred'])
    tn, fp, fn, tp = cm.ravel()
    
    report = f"""
    ОТЧЕТ О РЕЗУЛЬТАТАХ ML МОДЕЛИ ДЕТЕКЦИИ МОШЕННИЧЕСТВА
    
    1. ЛУЧШАЯ МОДЕЛЬ: {best_model_name}
    
    2. МЕТРИКИ КАЧЕСТВА:
       - Accuracy: {best_result['accuracy']:.4f}
       - Precision: {best_result['precision']:.4f}
       - Recall: {best_result['recall']:.4f}
       - F1-Score: {best_result['f1']:.4f}
       - ROC-AUC: {best_result['roc_auc']:.4f}
    
    3. МАТРИЦА ОШИБОК:
       - True Positives (TP): {tp} - правильно обнаруженный фрод
       - False Positives (FP): {fp} - ложные срабатывания
       - True Negatives (TN): {tn} - правильно пропущенные легитимные транзакции
       - False Negatives (FN): {fn} - пропущенный фрод
    
    4. БИЗНЕС-ИНТЕРПРЕТАЦИЯ:
       - Модель правильно идентифицирует {best_result['recall']*100:.1f}% мошеннических транзакций
       - Из всех транзакций, помеченных как мошеннические, {best_result['precision']*100:.1f}% действительно являются мошенническими
       - Баланс между точностью и полнотой (F1-Score): {best_result['f1']:.3f}
    
    5. РЕКОМЕНДАЦИИ:
       - Для максимизации обнаружения фрода: использовать порог с высоким recall
       - Для минимизации ложных срабатываний: использовать порог с высоким precision
       - Рекомендуемый компромисс: порог, максимизирующий F1-Score
    """
    
    print(report)
    
    # Сохраняем отчет в файл
    try:
        with open('fraud_detection_simple_report.txt', 'w') as f:
            f.write(report)
        print("Отчет сохранен в fraud_detection_simple_report.txt")
    except Exception as e:
        print(f"Ошибка при сохранении отчета: {e}")

# ======================================================================
# 5. АЛЬТЕРНАТИВНЫЙ ПОДХОД: БАЗОВЫЕ МОДЕЛИ БЕЗ СЛОЖНОГО FEATURE ENGINEERING
# ======================================================================

def create_basic_features(df):
    """Создание только базовых признаков"""
    print("Создание базовых признаков...")
    
    df = df.copy()
    
    # Базовые временные признаки
    if 'timestamp' in df.columns:
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['is_night'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
    
    # Бинарные признаки из булевых
    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[f'{col}_int'] = df[col].astype(int)
    
    # Простые взаимодействия
    if 'amount' in df.columns and 'is_high_risk_vendor' in df.columns:
        df['amount_risk'] = df['amount'] * df['is_high_risk_vendor']
    
    if 'is_outside_home_country' in df.columns and 'is_high_risk_vendor' in df.columns:
        df['risk_outside'] = df['is_outside_home_country'] & df['is_high_risk_vendor']
    
    return df

def run_basic_ml():
    """Запуск базового ML без сложного feature engineering"""
    print("=" * 80)
    print("БАЗОВЫЙ ML ПОДХОД")
    print("=" * 80)
    
    # Загрузка данных
    transactions = pd.read_parquet('transaction_fraud_data.parquet')
    
    # Создание базовых признаков
    df = create_basic_features(transactions)
    
    # Удаляем ненужные колонки
    cols_to_drop = ['transaction_id', 'customer_id', 'card_number', 'timestamp', 
                   'vendor', 'device_fingerprint', 'ip_address', 'device', 'city']
    
    if 'last_hour_activity' in df.columns:
        cols_to_drop.append('last_hour_activity')
    
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    
    # Целевая переменная
    if 'is_fraud' not in df.columns:
        print("Ошибка: нет целевой переменной is_fraud")
        return
    
    y = df['is_fraud'].astype(int)
    X = df.drop(columns=['is_fraud'], errors='ignore')
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Размеры: X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"Распределение классов в y_train: {dict(y_train.value_counts())}")
    
    # Определяем числовые и категориальные признаки
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Числовые признаки: {len(numeric_features)}")
    print(f"Категориальные признаки: {len(categorical_features)}")
    
    # Предобработка
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Модели
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    # Обучение и оценка
    results = {}
    for name, model in models.items():
        print(f"\nОбучение {name}...")
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        results[name] = {
            'model': pipeline,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_pred': y_pred
        }
    
    # Сравнение моделей
    metrics_df = pd.DataFrame({
        name: {
            'Accuracy': results[name]['accuracy'],
            'Precision': results[name]['precision'],
            'Recall': results[name]['recall'],
            'F1-Score': results[name]['f1']
        }
        for name in results.keys()
    }).T
    
    metrics_df = metrics_df.sort_values('F1-Score', ascending=False)
    
    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ МОДЕЛЕЙ:")
    print(metrics_df.round(4))
    
    # Матрица ошибок для лучшей модели
    best_model_name = metrics_df.index[0]
    best_model = results[best_model_name]
    
    cm = confusion_matrix(y_test, best_model['y_pred'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Матрица ошибок для {best_model_name}\nF1: {best_model["f1"]:.3f}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    print(f"\nЛучшая модель: {best_model_name} с F1-Score: {best_model['f1']:.4f}")
    
    # Сохранение модели
    joblib.dump(best_model['model'], 'basic_fraud_model.pkl')
    print("Модель сохранена в basic_fraud_model.pkl")

# ======================================================================
# ЗАПУСК ПРОГРАММЫ
# ======================================================================

if __name__ == "__main__":
    print("Выберите вариант запуска:")
    print("1. Упрощенный пайплайн с feature engineering")
    print("2. Базовый ML подход (рекомендуется для начала)")
    
    # choice = input("Введите номер варианта (1 или 2): ").strip()
    choice = "1"
    
    if choice == "1":
        print("\nЗапуск упрощенного пайплайна...")
        main_simple()
    elif choice == "2":
        print("\nЗапуск базового ML подхода...")
        run_basic_ml()
    else:
        print("Неверный выбор. Запускаю базовый подход по умолчанию...")
        run_basic_ml()