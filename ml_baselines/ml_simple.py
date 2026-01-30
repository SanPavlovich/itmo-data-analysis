# ml_models_basic.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML библиотеки
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Модели
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier

# Метрики
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve, precision_recall_curve,
                           average_precision_score)

# Балансировка классов
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

import joblib
import json

# Настройки отображения
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
plt.style.use('seaborn-v0_8-darkgrid')

def load_and_preprocess_data():
    """Загрузка данных и базовая предобработка"""
    print("=" * 80)
    print("ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ")
    print("=" * 80)
    
    # Загрузка транзакций
    print("Загрузка transaction_fraud_data.parquet...")
    trans_path = "/Users/a.kryazhenkov/test/other/itmo/data-analysis/transaction_fraud_data.parquet"
    df = pd.read_parquet(trans_path)
    
    print(f"Исходный размер данных: {df.shape}")
    print(f"Колонки: {list(df.columns)}")
    
    # Базовая предобработка
    df_clean = df.copy()
    
    # 1. Обработка timestamp (используем только час и день недели как числовые признаки)
    if 'timestamp' in df_clean.columns:
        df_clean['hour'] = pd.to_datetime(df_clean['timestamp']).dt.hour
        df_clean['day_of_week'] = pd.to_datetime(df_clean['timestamp']).dt.dayofweek
        # Удаляем исходный timestamp
        df_clean = df_clean.drop(columns=['timestamp'])
    
    # 2. Распаковка last_hour_activity (если это структура)
    if 'last_hour_activity' in df_clean.columns:
        print("Распаковка last_hour_activity...")
        try:
            # Пробуем разные способы распаковки
            if isinstance(df_clean['last_hour_activity'].iloc[0], dict):
                # Если это словарь
                activity_cols = ['num_transactions', 'total_amount', 'unique_merchants', 
                                'unique_countries', 'max_single_amount']
                for col in activity_cols:
                    df_clean[f'last_hour_{col}'] = df_clean['last_hour_activity'].apply(
                        lambda x: x.get(col, np.nan) if isinstance(x, dict) else np.nan
                    )
            else:
                # Если это структура pandas
                try:
                    df_clean['last_hour_num_transactions'] = df_clean['last_hour_activity'].apply(
                        lambda x: x.num_transactions if pd.notnull(x) else np.nan
                    )
                    df_clean['last_hour_total_amount'] = df_clean['last_hour_activity'].apply(
                        lambda x: x.total_amount if pd.notnull(x) else np.nan
                    )
                    df_clean['last_hour_unique_merchants'] = df_clean['last_hour_activity'].apply(
                        lambda x: x.unique_merchants if pd.notnull(x) else np.nan
                    )
                    df_clean['last_hour_unique_countries'] = df_clean['last_hour_activity'].apply(
                        lambda x: x.unique_countries if pd.notnull(x) else np.nan
                    )
                    df_clean['last_hour_max_single_amount'] = df_clean['last_hour_activity'].apply(
                        lambda x: x.max_single_amount if pd.notnull(x) else np.nan
                    )
                except:
                    # Если не получилось, пропускаем
                    pass
        except Exception as e:
            print(f"Ошибка при распаковке last_hour_activity: {e}")
        
        # Удаляем исходную колонку
        df_clean = df_clean.drop(columns=['last_hour_activity'])
    
    # 3. Преобразование булевых колонок в числовые
    bool_columns = df_clean.select_dtypes(include=['bool']).columns
    for col in bool_columns:
        df_clean[col] = df_clean[col].astype(int)
    
    # 4. Удаление колонок с уникальными идентификаторами и текстовыми полями
    columns_to_drop = [
        'transaction_id', 'customer_id', 'card_number',
        'vendor', 'device_fingerprint', 'ip_address',
        'device', 'city'
    ]
    
    # Удаляем только существующие колонки
    columns_to_drop = [col for col in columns_to_drop if col in df_clean.columns]
    df_clean = df_clean.drop(columns=columns_to_drop, errors='ignore')
    
    print(f"Размер данных после очистки: {df_clean.shape}")
    
    # Проверка целевой переменной
    if 'is_fraud' not in df_clean.columns:
        raise ValueError("Целевая переменная 'is_fraud' не найдена в данных")
    
    # Статистика по целевой переменной
    fraud_count = df_clean['is_fraud'].sum()
    total_count = len(df_clean)
    fraud_percentage = fraud_count / total_count * 100
    
    print(f"\nСтатистика целевой переменной:")
    print(f"  Всего транзакций: {total_count:,}")
    print(f"  Мошеннических: {fraud_count:,} ({fraud_percentage:.2f}%)")
    print(f"  Честных: {total_count - fraud_count:,} ({(100 - fraud_percentage):.2f}%)")
    
    return df_clean

def prepare_train_test_split(df, test_size=0.2, random_state=42):
    """Подготовка train/test разделения"""
    print("\n" + "=" * 80)
    print("РАЗДЕЛЕНИЕ ДАННЫХ НА TRAIN/TEST")
    print("=" * 80)
    
    # Целевая переменная
    y = df['is_fraud'].astype(int)
    X = df.drop(columns=['is_fraud'])
    
    # Стратифицированное разделение
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Размеры данных:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train распределение: {dict(y_train.value_counts())}")
    print(f"  y_test распределение: {dict(y_test.value_counts())}")
    
    # Определяем типы признаков
    numeric_features = X_train.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"\nТипы признаков:")
    print(f"  Числовые: {len(numeric_features)}")
    print(f"  Категориальные: {len(categorical_features)}")
    
    if len(categorical_features) > 0:
        print(f"  Примеры категориальных признаков: {categorical_features[:5]}")
    
    return X_train, X_test, y_train, y_test, numeric_features, categorical_features

def create_preprocessing_pipeline(numeric_features, categorical_features):
    """Создание пайплайна предобработки"""
    print("\nСоздание пайплайна предобработки...")
    
    # Для числовых признаков: заполнение пропусков медианой и масштабирование
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Для категориальных признаков: заполнение пропусков константой и one-hot кодирование
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

def balance_dataset(X_train, y_train, method='smote', sampling_strategy='auto'):
    """Балансировка классов в обучающей выборке"""
    print(f"\nБалансировка классов методом: {method}")
    
    original_counts = y_train.value_counts().to_dict()
    
    if method == 'smote':
        balancer = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    elif method == 'adasyn':
        balancer = ADASYN(sampling_strategy=sampling_strategy, random_state=42)
    elif method == 'undersample':
        balancer = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    elif method == 'smoteenn':
        balancer = SMOTEENN(sampling_strategy=sampling_strategy, random_state=42)
    else:
        print("  Используются несбалансированные данные")
        return X_train, y_train
    
    X_train_balanced, y_train_balanced = balancer.fit_resample(X_train, y_train)
    
    balanced_counts = pd.Series(y_train_balanced).value_counts().to_dict()
    
    print(f"  До балансировки: {original_counts}")
    print(f"  После балансировки: {balanced_counts}")
    
    return X_train_balanced, y_train_balanced

def train_ml_models(X_train, X_test, y_train, y_test, preprocessor):
    """Обучение нескольких ML моделей"""
    print("\n" + "=" * 80)
    print("ОБУЧЕНИЕ ML МОДЕЛЕЙ")
    print("=" * 80)
    
    # Определяем модели для обучения
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            'params': {'classifier__C': [0.01, 0.1, 1, 10]}
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'),
            'params': {'classifier__n_estimators': [50, 100, 200], 'classifier__max_depth': [10, 20, None]}
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42, n_estimators=100),
            'params': {'classifier__n_estimators': [50, 100], 'classifier__learning_rate': [0.01, 0.1]}
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, n_estimators=100, use_label_encoder=False, eval_metric='logloss'),
            'params': {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [3, 5]}
        },
        'LightGBM': {
            'model': LGBMClassifier(random_state=42, n_estimators=100, verbose=-1),
            'params': {'classifier__n_estimators': [50, 100], 'classifier__num_leaves': [31, 50]}
        }
    }
    
    results = {}
    
    for model_name, model_info in models.items():
        print(f"\nОбучение {model_name}...")
        
        try:
            # Создаем пайплайн
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model_info['model'])
            ])
            
            # Обучение модели
            start_time = datetime.now()
            pipeline.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Предсказания
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            
            # Вычисление метрик
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # ROC-AUC
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            except:
                roc_auc = 0.5
            
            # Average Precision
            try:
                avg_precision = average_precision_score(y_test, y_pred_proba)
            except:
                avg_precision = y_test.mean()
            
            # Матрица ошибок
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Сохранение результатов
            results[model_name] = {
                'pipeline': pipeline,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'avg_precision': avg_precision,
                'training_time': training_time,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'confusion_matrix': cm,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn
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

def evaluate_and_compare_models(results, y_test):
    """Оценка и сравнение моделей"""
    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 80)
    
    if not results:
        print("Нет обученных моделей для сравнения")
        return None
    
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
            'Training Time (s)': result['training_time'],
            'TP': result['tp'],
            'FP': result['fp'],
            'TN': result['tn'],
            'FN': result['fn']
        }
    
    metrics_df = pd.DataFrame(metrics_data).T
    
    # Сортируем по F1-Score
    metrics_df = metrics_df.sort_values('F1-Score', ascending=False)
    
    print("\nСравнение метрик всех моделей:")
    print("-" * 80)
    print(metrics_df[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Training Time (s)']].round(4))
    
    # Визуализация сравнения моделей
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Avg Precision']
    
    for idx, metric in enumerate(metrics_to_plot):
        if idx >= 6:  # максимум 6 графиков
            break
            
        row = idx // 3
        col = idx % 3
        
        if metric in metrics_df.columns:
            axes[row, col].barh(range(len(metrics_df)), metrics_df[metric].values)
            axes[row, col].set_yticks(range(len(metrics_df)))
            axes[row, col].set_yticklabels(metrics_df.index, fontsize=9)
            axes[row, col].set_xlabel(metric)
            axes[row, col].set_title(f'{metric} по моделям')
            axes[row, col].invert_yaxis()
    
    plt.tight_layout()
    plt.show()
    
    return metrics_df

def plot_confusion_matrices(results, y_test, top_n=3):
    """Визуализация матриц ошибок для лучших моделей"""
    print("\n" + "=" * 80)
    print("МАТРИЦЫ ОШИБОК ЛУЧШИХ МОДЕЛЕЙ")
    print("=" * 80)
    
    if not results:
        return
    
    # Выбираем топ-N моделей по F1-Score
    top_models = sorted(results.keys(), 
                       key=lambda x: results[x]['f1'], 
                       reverse=True)[:min(top_n, len(results))]
    
    if not top_models:
        return
    
    fig, axes = plt.subplots(1, len(top_models), figsize=(5*len(top_models), 4))
    
    if len(top_models) == 1:
        axes = [axes]
    
    for idx, model_name in enumerate(top_models):
        cm = results[model_name]['confusion_matrix']
        
        # Создаем heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Legit', 'Fraud'],
                   yticklabels=['Legit', 'Fraud'],
                   ax=axes[idx])
        
        axes[idx].set_title(f'{model_name}\nF1: {results[model_name]["f1"]:.3f}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()

def plot_roc_curves(results, y_test):
    """Визуализация ROC кривых"""
    print("\n" + "=" * 80)
    print("ROC КРИВЫЕ")
    print("=" * 80)
    
    if not results:
        return
    
    plt.figure(figsize=(10, 8))
    
    # Базовая линия (случайное угадывание)
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
    
    # ROC кривые для каждой модели
    for model_name, result in results.items():
        y_pred_proba = result['y_pred_proba']
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = result['roc_auc']
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Comparison of Models')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_precision_recall_curves(results, y_test):
    """Визуализация Precision-Recall кривых"""
    print("\n" + "=" * 80)
    print("PRECISION-RECALL КРИВЫЕ")
    print("=" * 80)
    
    if not results:
        return
    
    plt.figure(figsize=(10, 8))
    
    # Baseline для несбалансированных данных
    fraud_rate = y_test.mean()
    plt.axhline(y=fraud_rate, color='k', linestyle='--', 
                label=f'Baseline (Precision = {fraud_rate:.3f})')
    
    # Precision-Recall кривые для каждой модели
    for model_name, result in results.items():
        y_pred_proba = result['y_pred_proba']
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = result['avg_precision']
        plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves - Comparison of Models')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.show()

def analyze_feature_importance(results, X_train, categorical_features, numeric_features):
    """Анализ важности признаков для лучшей модели"""
    print("\n" + "=" * 80)
    print("АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ")
    print("=" * 80)
    
    if not results:
        return
    
    # Выбираем лучшую модель по F1-Score
    best_model_name = sorted(results.keys(), 
                           key=lambda x: results[x]['f1'], 
                           reverse=True)[0]
    
    print(f"Анализ важности признаков для модели: {best_model_name}")
    
    best_pipeline = results[best_model_name]['pipeline']
    model = best_pipeline.named_steps['classifier']
    
    # Проверяем, поддерживает ли модель важность признаков
    if not hasattr(model, 'feature_importances_'):
        print(f"Модель {best_model_name} не поддерживает важность признаков")
        return
    
    # Получаем предобработчик
    preprocessor = best_pipeline.named_steps['preprocessor']
    
    # Получаем имена признаков после преобразования
    try:
        # Для категориальных признаков получаем имена после one-hot кодирования
        categorical_transformer = preprocessor.named_transformers_['cat']
        onehot = categorical_transformer.named_steps['onehot']
        
        if hasattr(onehot, 'get_feature_names_out'):
            categorical_feature_names = onehot.get_feature_names_out(categorical_features)
        else:
            # Альтернативный способ
            categorical_feature_names = []
            for col in categorical_features:
                # Получаем уникальные значения
                unique_vals = X_train[col].dropna().unique()
                for val in unique_vals:
                    categorical_feature_names.append(f"{col}_{val}")
        
        # Объединяем имена признаков
        all_feature_names = list(numeric_features) + list(categorical_feature_names)
        
        # Получаем важность признаков
        importances = model.feature_importances_
        
        # Создаем DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': all_feature_names[:len(importances)],
            'importance': importances
        })
        
        # Сортируем по важности
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
        
        print(f"\nТоп-15 самых важных признаков:")
        print("-" * 80)
        print(feature_importance_df.head(15).to_string(index=False))
        
        # Визуализация
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'].values)
        plt.yticks(range(len(top_features)), top_features['feature'].values, fontsize=9)
        plt.xlabel('Важность признака')
        plt.title(f'Топ-15 самых важных признаков ({best_model_name})')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return feature_importance_df
        
    except Exception as e:
        print(f"Ошибка при анализе важности признаков: {e}")
        return None

def save_best_model(results, model_name='best_fraud_model'):
    """Сохранение лучшей модели"""
    print("\n" + "=" * 80)
    print("СОХРАНЕНИЕ МОДЕЛЕЙ")
    print("=" * 80)
    
    if not results:
        print("Нет моделей для сохранения")
        return
    
    # Выбираем лучшую модель по F1-Score
    best_model_name = sorted(results.keys(), 
                           key=lambda x: results[x]['f1'], 
                           reverse=True)[0]
    
    best_pipeline = results[best_model_name]['pipeline']
    
    # Сохраняем лучшую модель
    model_filename = f'{model_name}.pkl'
    joblib.dump(best_pipeline, model_filename)
    print(f"Лучшая модель ({best_model_name}) сохранена в {model_filename}")
    
    # Сохраняем все модели
    all_models_filename = 'all_models.pkl'
    joblib.dump(results, all_models_filename)
    print(f"Все модели сохранены в {all_models_filename}")
    
    # Сохраняем метрики в CSV
    metrics_data = []
    for model_name, result in results.items():
        metrics_data.append({
            'model': model_name,
            'accuracy': result['accuracy'],
            'precision': result['precision'],
            'recall': result['recall'],
            'f1_score': result['f1'],
            'roc_auc': result['roc_auc'],
            'avg_precision': result['avg_precision'],
            'training_time': result['training_time']
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv('model_metrics.csv', index=False)
    print(f"Метрики сохранены в model_metrics.csv")
    
    return best_model_name, model_filename

def generate_report(results, best_model_name):
    """Генерация итогового отчета"""
    print("\n" + "=" * 80)
    print("ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 80)
    
    if best_model_name not in results:
        print("Лучшая модель не найдена в результатах")
        return
    
    best_result = results[best_model_name]
    
    # Рассчитываем бизнес-метрики
    # Предположим, что средняя сумма мошеннической транзакции = 1000
    avg_fraud_amount = 1000
    cost_missed_fraud = avg_fraud_amount * 10  # Пропущенный фрод в 10 раз дороже
    cost_false_alarm = 50  # Стоимость ложного срабатывания
    
    tp_cost_saved = best_result['tp'] * avg_fraud_amount
    fn_cost = best_result['fn'] * cost_missed_fraud
    fp_cost = best_result['fp'] * cost_false_alarm
    
    total_cost = fn_cost + fp_cost
    net_savings = tp_cost_saved - total_cost
    
    report = f"""
    ОТЧЕТ О РЕЗУЛЬТАТАХ ML МОДЕЛИ ДЕТЕКЦИИ МОШЕННИЧЕСТВА
    
    1. ЛУЧШАЯ МОДЕЛЬ: {best_model_name}
    
    2. МЕТРИКИ КАЧЕСТВА:
       - Accuracy: {best_result['accuracy']:.4f}
       - Precision: {best_result['precision']:.4f}
       - Recall: {best_result['recall']:.4f}
       - F1-Score: {best_result['f1']:.4f}
       - ROC-AUC: {best_result['roc_auc']:.4f}
       - Average Precision: {best_result['avg_precision']:.4f}
    
    3. МАТРИЦА ОШИБОК:
       - True Positives (TP): {best_result['tp']} - правильно обнаруженный фрод
       - False Positives (FP): {best_result['fp']} - ложные срабатывания
       - True Negatives (TN): {best_result['tn']} - правильно пропущенные легитимные транзакции
       - False Negatives (FN): {best_result['fn']} - пропущенный фрод
    
    4. БИЗНЕС-МЕТРИКИ (предположительные):
       - Стоимость спасенных средств (TP): ${tp_cost_saved:,.2f}
       - Убытки от пропущенного фрода (FN): ${fn_cost:,.2f}
       - Издержки ложных срабатываний (FP): ${fp_cost:,.2f}
       - Общие убытки: ${total_cost:,.2f}
       - Чистая экономия: ${net_savings:,.2f}
    
    5. РЕКОМЕНДАЦИИ:
       - Текущая модель правильно обнаруживает {best_result['recall']*100:.1f}% мошеннических транзакций
       - {best_result['precision']*100:.1f}% всех помеченных как мошеннические транзакций действительно являются таковыми
       - Для максимизации обнаружения фрода используйте более низкий порог классификации
       - Для минимизации ложных срабатываний используйте более высокий порог классификации
       - Рекомендуемый баланс: порог, максимизирующий F1-Score ({best_result['f1']:.3f})
    
    6. СЛЕДУЮЩИЕ ШАГИ:
       - Тестирование модели на новых данных
       - Мониторинг дрейфа данных и переобучение модели каждые 3 месяца
       - A/B тестирование разных порогов классификации
       - Интеграция с системами реального времени для блокировки подозрительных транзакций
    """
    
    print(report)
    
    # Сохраняем отчет в файл
    with open('fraud_detection_report.txt', 'w') as f:
        f.write(report)
    
    print("Полный отчет сохранен в fraud_detection_report.txt")
    
    return report

def main():
    """Основная функция выполнения ML пайплайна"""
    print("=" * 80)
    print("ML ПАЙПЛАЙН ДЛЯ ДЕТЕКЦИИ МОШЕННИЧЕСТВА")
    print("(БЕЗ СОЗДАНИЯ ДОПОЛНИТЕЛЬНЫХ ПРИЗНАКОВ)")
    print("=" * 80)
    
    try:
        # Шаг 1: Загрузка и предобработка данных
        df = load_and_preprocess_data()
        
        # Шаг 2: Разделение данных
        X_train, X_test, y_train, y_test, numeric_features, categorical_features = prepare_train_test_split(df)
        
        # Шаг 3: Создание пайплайна предобработки
        preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
        
        # Шаг 4: Балансировка классов (опционально)
        balance_method = input("\nВыберите метод балансировки классов (smote/adasyn/undersample/none): ").strip().lower()
        if balance_method not in ['smote', 'adasyn', 'undersample', 'none']:
            balance_method = 'smote'
        
        if balance_method != 'none':
            X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train, method=balance_method)
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
            print("Используются несбалансированные данные")
        
        # Шаг 5: Обучение моделей
        results = train_ml_models(X_train_balanced, X_test, y_train_balanced, y_test, preprocessor)
        
        if not results:
            print("Не удалось обучить ни одну модель")
            return
        
        # Шаг 6: Оценка и сравнение моделей
        metrics_df = evaluate_and_compare_models(results, y_test)
        
        # Шаг 7: Визуализация результатов
        plot_confusion_matrices(results, y_test, top_n=3)
        plot_roc_curves(results, y_test)
        plot_precision_recall_curves(results, y_test)
        
        # Шаг 8: Анализ важности признаков
        analyze_feature_importance(results, X_train, categorical_features, numeric_features)
        
        # Шаг 9: Сохранение моделей
        best_model_name, model_filename = save_best_model(results)
        
        # Шаг 10: Генерация отчета
        generate_report(results, best_model_name)
        
        print("\n" + "=" * 80)
        print("ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nОШИБКА В ПАЙПЛАЙНЕ: {e}")
        import traceback
        traceback.print_exc()

def quick_demo():
    """Быстрая демонстрация с минимальным кодом"""
    print("=" * 80)
    print("БЫСТРАЯ ДЕМОНСТРАЦИЯ ML МОДЕЛЕЙ")
    print("=" * 80)
    
    # Загрузка данных
    trans_path = "/Users/a.kryazhenkov/test/other/itmo/data-analysis/transaction_fraud_data.parquet"
    df = pd.read_parquet(trans_path)
    
    # Базовая предобработка
    df_clean = df.copy()
    
    # Преобразование булевых колонок
    bool_cols = df_clean.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df_clean[col] = df_clean[col].astype(int)
    
    # Удаление ненужных колонок
    cols_to_drop = ['transaction_id', 'customer_id', 'card_number', 'timestamp',
                   'vendor', 'device_fingerprint', 'ip_address', 'city', 'device']
    df_clean = df_clean.drop(columns=[col for col in cols_to_drop if col in df_clean.columns])
    
    # Распаковка last_hour_activity (упрощенная версия)
    if 'last_hour_activity' in df_clean.columns:
        try:
            if isinstance(df_clean['last_hour_activity'].iloc[0], dict):
                activity_cols = ['num_transactions', 'total_amount', 'unique_merchants']
                for col in activity_cols:
                    df_clean[f'last_hour_{col}'] = df_clean['last_hour_activity'].apply(
                        lambda x: x.get(col, np.nan) if isinstance(x, dict) else np.nan
                    )
        except:
            pass
        df_clean = df_clean.drop(columns=['last_hour_activity'])
    
    # Подготовка данных
    X = df_clean.drop(columns=['is_fraud'])
    y = df_clean['is_fraud'].astype(int)
    
    # Разделение
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Предобработка
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Модели
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        # 'Random Forest': RandomForestClassifier(n_estimators=1, max_depth=5, class_weight='balanced'),
        # 'XGBoost': XGBClassifier(n_estimators=1, use_label_encoder=False, eval_metric='logloss')
    }
    
    # Обучение и оценка
    print("\nОбучение моделей...")
    pipelines = {}
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        pipelines[name] = pipeline
        
        print(f"{name}: Accuracy = {accuracy:.4f}, F1-Score = {f1:.4f}")
    
    print("\nДемонстрация завершена!")

    return pipelines

if __name__ == "__main__":
    print("Выберите вариант запуска:")
    print("1. Полный пайплайн")
    print("2. Быстрая демонстрация")
    
    # choice = input("Введите номер (1 или 2): ").strip()
    choice = "2"
    
    if choice == "1":
        main()
    elif choice == "2":
        quick_demo()
    else:
        print("Неверный выбор. Запускаю быструю демонстрацию...")
        quick_demo()