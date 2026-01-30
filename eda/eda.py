# transaction_analysis_eda.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Настройка отображения графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

def load_data():
    """Загрузка данных из parquet файлов"""
    print("=" * 80)
    print("ЗАГРУЗКА ДАННЫХ")
    print("=" * 80)
    
    # Загрузка транзакционных данных
    print("Загрузка transaction_fraud_data.parquet...")
    transactions_df = pd.read_parquet('transaction_fraud_data.parquet')
    
    # Загрузка данных об обменных курсах
    print("Загрузка historical_currency_exchange.parquet...")
    currency_df = pd.read_parquet('historical_currency_exchange.parquet')
    
    print(f"\nТранзакции: {transactions_df.shape[0]} строк, {transactions_df.shape[1]} колонок")
    print(f"Курсы валют: {currency_df.shape[0]} строк, {currency_df.shape[1]} колонок")
    
    return transactions_df, currency_df

def basic_info(transactions_df, currency_df):
    """Базовая информация о данных"""
    print("\n" + "=" * 80)
    print("БАЗОВАЯ ИНФОРМАЦИЯ О ДАННЫХ")
    print("=" * 80)
    
    # Информация о транзакциях
    print("\n1. ИНФОРМАЦИЯ О ТРАНЗАКЦИЯХ:")
    print("-" * 40)
    print("\nПервые 5 строк:")
    print(transactions_df.head())
    
    print("\nТипы данных:")
    print(transactions_df.dtypes)
    
    print("\nБазовая статистика числовых признаков:")
    print(transactions_df.describe())
    
    # Информация о курсах валют
    print("\n\n2. ИНФОРМАЦИЯ О КУРСАХ ВАЛЮТ:")
    print("-" * 40)
    print("\nПервые 5 строк:")
    print(currency_df.head())
    
    print("\nДиапазон дат:")
    print(f"От: {currency_df['date'].min()} до: {currency_df['date'].max()}")
    print(f"Всего дней: {currency_df.shape[0]}")
    
    print("\nДоступные валюты:")
    currencies = [col for col in currency_df.columns if col != 'date']
    print(f"{len(currencies)} валют: {', '.join(currencies)}")

def check_missing_values(transactions_df, currency_df):
    """Проверка пропущенных значений"""
    print("\n" + "=" * 80)
    print("АНАЛИЗ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ")
    print("=" * 80)
    
    print("\n1. ТРАНЗАКЦИИ:")
    print("-" * 40)
    missing_transactions = transactions_df.isnull().sum()
    missing_percentage = (missing_transactions / len(transactions_df)) * 100
    missing_df = pd.DataFrame({
        'Пропущенные значения': missing_transactions,
        'Процент': missing_percentage
    })
    print(missing_df[missing_df['Пропущенные значения'] > 0])
    
    print("\n2. КУРСЫ ВАЛЮТ:")
    print("-" * 40)
    missing_currency = currency_df.isnull().sum()
    missing_percentage_curr = (missing_currency / len(currency_df)) * 100
    missing_df_curr = pd.DataFrame({
        'Пропущенные значения': missing_currency,
        'Процент': missing_percentage_curr
    })
    print(missing_df_curr[missing_df_curr['Пропущенные значения'] > 0])

def analyze_transaction_features(transactions_df):
    """Анализ признаков транзакций"""
    print("\n" + "=" * 80)
    print("АНАЛИЗ ПРИЗНАКОВ ТРАНЗАКЦИЙ")
    print("=" * 80)
    
    # 1. Распределение мошеннических транзакций
    print("\n1. РАСПРЕДЕЛЕНИЕ МОШЕННИЧЕСКИХ ТРАНЗАКЦИЙ:")
    print("-" * 40)
    fraud_counts = transactions_df['is_fraud'].value_counts()
    fraud_percentage = transactions_df['is_fraud'].value_counts(normalize=True) * 100
    
    fraud_df = pd.DataFrame({
        'Количество': fraud_counts,
        'Процент': fraud_percentage
    })
    print(fraud_df)
    
    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Круговая диаграмма
    axes[0].pie(fraud_counts, labels=['Честные', 'Мошеннические'], 
                autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'salmon'])
    axes[0].set_title('Распределение мошеннических транзакций')
    
    # Столбчатая диаграмма
    axes[1].bar(fraud_df.index.astype(str), fraud_df['Количество'], 
                color=['lightgreen', 'salmon'])
    axes[1].set_title('Количество мошеннических транзакций')
    axes[1].set_xlabel('is_fraud')
    axes[1].set_ylabel('Количество')
    plt.tight_layout()
    plt.show()
    
    # 2. Распределение суммы транзакций
    print("\n2. РАСПРЕДЕЛЕНИЕ СУММЫ ТРАНЗАКЦИЙ:")
    print("-" * 40)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Гистограмма
    axes[0].hist(transactions_df['amount'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_title('Распределение суммы транзакций')
    axes[0].set_xlabel('Сумма')
    axes[0].set_ylabel('Частота')
    
    # Boxplot
    axes[1].boxplot(transactions_df['amount'])
    axes[1].set_title('Boxplot суммы транзакций')
    axes[1].set_ylabel('Сумма')
    plt.tight_layout()
    plt.show()
    
    # Сравнение сумм для мошеннических и честных транзакций
    fraud_amounts = transactions_df[transactions_df['is_fraud'] == True]['amount']
    legit_amounts = transactions_df[transactions_df['is_fraud'] == False]['amount']
    
    print(f"Средняя сумма честных транзакций: {legit_amounts.mean():.2f}")
    print(f"Средняя сумма мошеннических транзакций: {fraud_amounts.mean():.2f}")
    print(f"Медиана честных транзакций: {legit_amounts.median():.2f}")
    print(f"Медиана мошеннических транзакций: {fraud_amounts.median():.2f}")
    
    # 3. Категориальные признаки
    print("\n3. КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ:")
    print("-" * 40)
    
    categorical_cols = ['vendor_category', 'vendor_type', 'currency', 'country', 
                       'city_size', 'card_type', 'channel', 'is_high_risk_vendor',
                       'is_outside_home_country', 'is_weekend']
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for idx, col in enumerate(categorical_cols):
        if idx >= len(axes):
            break
            
        value_counts = transactions_df[col].value_counts().head(10)
        if len(value_counts) > 20:  # Если слишком много значений, покажем топ
            value_counts = value_counts.head(20)
        
        if col in ['is_high_risk_vendor', 'is_outside_home_country', 'is_weekend']:
            axes[idx].bar(value_counts.index.astype(str), value_counts.values)
        else:
            axes[idx].barh(range(len(value_counts)), value_counts.values)
            axes[idx].set_yticks(range(len(value_counts)))
            axes[idx].set_yticklabels(value_counts.index)
        
        axes[idx].set_title(f'{col}\n(Всего уникальных: {transactions_df[col].nunique()})')
        axes[idx].set_xlabel('Количество')
    
    # Удаляем лишние subplots
    for idx in range(len(categorical_cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()
    
    # 4. Анализ по времени
    print("\n4. АНАЛИЗ ПО ВРЕМЕНИ:")
    print("-" * 40)
    
    # Извлечение компонентов времени
    transactions_df['date'] = transactions_df['timestamp'].dt.date
    transactions_df['hour'] = transactions_df['timestamp'].dt.hour
    transactions_df['day_of_week'] = transactions_df['timestamp'].dt.day_name()
    transactions_df['month'] = transactions_df['timestamp'].dt.month
    
    # Мошеннические транзакции по часам
    fraud_by_hour = transactions_df.groupby('hour')['is_fraud'].mean() * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(fraud_by_hour.index, fraud_by_hour.values, marker='o')
    axes[0].set_title('Процент мошеннических транзакций по часам')
    axes[0].set_xlabel('Час')
    axes[0].set_ylabel('Процент мошенничества (%)')
    axes[0].grid(True, alpha=0.3)
    
    # Мошеннические транзакции по дням недели
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fraud_by_day = transactions_df.groupby('day_of_week')['is_fraud'].mean() * 100
    fraud_by_day = fraud_by_day.reindex(day_order)
    
    axes[1].bar(fraud_by_day.index, fraud_by_day.values)
    axes[1].set_title('Процент мошеннических транзакций по дням недели')
    axes[1].set_xlabel('День недели')
    axes[1].set_ylabel('Процент мошенничества (%)')
    axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

def analyze_currency_exchange(currency_df):
    """Анализ данных об обменных курсах"""
    print("\n" + "=" * 80)
    print("АНАЛИЗ ОБМЕННЫХ КУРСОВ")
    print("=" * 80)
    
    # 1. Статистика по валютам
    print("\n1. СТАТИСТИКА ПО ВАЛЮТАМ:")
    print("-" * 40)
    
    currency_columns = [col for col in currency_df.columns if col != 'date']
    
    stats = []
    for currency in currency_columns:
        stats.append({
            'Валюта': currency,
            'Среднее': currency_df[currency].mean(),
            'Медиана': currency_df[currency].median(),
            'Мин': currency_df[currency].min(),
            'Макс': currency_df[currency].max(),
            'Std': currency_df[currency].std()
        })
    
    stats_df = pd.DataFrame(stats)
    print(stats_df.to_string(index=False))
    
    # 2. Динамика курсов во времени
    print("\n2. ДИНАМИКА КУРСОВ ВАЛЮТ:")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Выберем несколько ключевых валют для визуализации
    key_currencies = ['EUR', 'GBP', 'JPY', 'RUB'][:4]
    
    for idx, currency in enumerate(key_currencies):
        axes[idx].plot(currency_df['date'], currency_df[currency], marker='o', markersize=2)
        axes[idx].set_title(f'Динамика курса {currency}/USD')
        axes[idx].set_xlabel('Дата')
        axes[idx].set_ylabel(f'{currency} за 1 USD')
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 3. Корреляция между валютами
    print("\n3. КОРРЕЛЯЦИЯ МЕЖДУ ВАЛЮТАМИ:")
    print("-" * 40)
    
    correlation_matrix = currency_df[currency_columns].corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                square=True, cbar_kws={'shrink': 0.8})
    plt.title('Корреляция между курсами валют')
    plt.tight_layout()
    plt.show()

def analyze_last_hour_activity(transactions_df):
    """Анализ активности за последний час"""
    print("\n" + "=" * 80)
    print("АНАЛИЗ АКТИВНОСТИ ЗА ПОСЛЕДНИЙ ЧАС")
    print("=" * 80)
    
    # Распаковка вложенной структуры
    last_hour_cols = ['num_transactions', 'total_amount', 'unique_merchants', 
                      'unique_countries', 'max_single_amount']
    
    for col in last_hour_cols:
        transactions_df[f'last_hour_{col}'] = transactions_df['last_hour_activity'].apply(
            lambda x: x.get(col, np.nan) if pd.notnull(x) else np.nan
        )
    
    # Статистика по распакованным признакам
    print("\nСтатистика активности за последний час:")
    print("-" * 40)
    
    stats_cols = [f'last_hour_{col}' for col in last_hour_cols]
    print(transactions_df[stats_cols].describe())
    
    # Анализ корреляции с мошенничеством
    print("\nКорреляция с мошенничеством:")
    print("-" * 40)
    
    correlations = {}
    for col in stats_cols:
        corr = transactions_df[col].corr(transactions_df['is_fraud'].astype(float))
        correlations[col] = corr
    
    corr_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Корреляция с is_fraud'])
    print(corr_df.sort_values('Корреляция с is_fraud', ascending=False))
    
    # Визуализация распределений для мошеннических и честных транзакций
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(stats_cols[:6]):  # Показываем только первые 6
        if idx >= len(axes):
            break
            
        # Boxplot раздельно для мошеннических и честных транзакций
        data_to_plot = []
        labels = ['Честные', 'Мошеннические']
        
        for fraud_status in [False, True]:
            subset = transactions_df[transactions_df['is_fraud'] == fraud_status][col]
            data_to_plot.append(subset)
        
        axes[idx].boxplot(data_to_plot, labels=labels)
        axes[idx].set_title(f'{col}\nРаспределение')
        axes[idx].set_ylabel('Значение')
        axes[idx].tick_params(axis='x', rotation=45)
    
    # Удаляем лишние subplots
    for idx in range(len(stats_cols[:6]), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()

def analyze_fraud_patterns(transactions_df):
    """Анализ паттернов мошенничества"""
    print("\n" + "=" * 80)
    print("АНАЛИЗ ПАТТЕРНОВ МОШЕННИЧЕСТВА")
    print("=" * 80)
    
    # 1. Мошенничество по категориям вендоров
    print("\n1. МОШЕННИЧЕСТВО ПО КАТЕГОРИЯМ ВЕНДОРОВ:")
    print("-" * 40)
    
    fraud_by_vendor_category = transactions_df.groupby('vendor_category')['is_fraud'].agg(['mean', 'count'])
    fraud_by_vendor_category['mean'] = fraud_by_vendor_category['mean'] * 100
    fraud_by_vendor_category = fraud_by_vendor_category.sort_values('mean', ascending=False)
    
    print("\nТоп категорий по проценту мошенничества:")
    print(fraud_by_vendor_category.head(10).to_string())
    
    # 2. Мошенничество по типу карты
    print("\n\n2. МОШЕННИЧЕСТВО ПО ТИПУ КАРТЫ:")
    print("-" * 40)
    
    fraud_by_card_type = transactions_df.groupby('card_type')['is_fraud'].agg(['mean', 'count'])
    fraud_by_card_type['mean'] = fraud_by_card_type['mean'] * 100
    fraud_by_card_type = fraud_by_card_type.sort_values('mean', ascending=False)
    
    print(fraud_by_card_type.to_string())
    
    # 3. Мошенничество по каналу
    print("\n\n3. МОШЕННИЧЕСТВО ПО КАНАЛУ:")
    print("-" * 40)
    
    fraud_by_channel = transactions_df.groupby('channel')['is_fraud'].agg(['mean', 'count'])
    fraud_by_channel['mean'] = fraud_by_channel['mean'] * 100
    fraud_by_channel = fraud_by_channel.sort_values('mean', ascending=False)
    
    print(fraud_by_channel.to_string())
    
    # 4. Мошенничество по комбинациям признаков
    print("\n\n4. КОМБИНАЦИИ ПРИЗНАКОВ С ВЫСОКИМ РИСКОМ:")
    print("-" * 40)
    
    # Создаем несколько комбинаций
    combinations = [
        ('is_outside_home_country', 'is_high_risk_vendor'),
        ('is_weekend', 'is_high_risk_vendor'),
        ('is_card_present', 'channel')
    ]
    
    for combo in combinations:
        if len(combo) == 2:
            print(f"\nКомбинация: {combo[0]} × {combo[1]}")
            cross_tab = pd.crosstab(
                transactions_df[combo[0]], 
                transactions_df[combo[1]], 
                values=transactions_df['is_fraud'], 
                aggfunc='mean'
            ) * 100
            print(cross_tab.round(2))
    
    # 5. Географический анализ мошенничества
    print("\n\n5. ГЕОГРАФИЧЕСКИЙ АНАЛИЗ МОШЕННИЧЕСТВА:")
    print("-" * 40)
    
    fraud_by_country = transactions_df.groupby('country')['is_fraud'].agg(['mean', 'count'])
    fraud_by_country['mean'] = fraud_by_country['mean'] * 100
    fraud_by_country = fraud_by_country.sort_values('mean', ascending=False)
    
    print("Топ стран по проценту мошенничества:")
    print(fraud_by_country.head(10).to_string())
    
    # Визуализация ключевых паттернов
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Мошенничество по категориям вендоров
    top_categories = fraud_by_vendor_category.head(10).index
    top_data = fraud_by_vendor_category.loc[top_categories]
    axes[0, 0].barh(range(len(top_categories)), top_data['mean'])
    axes[0, 0].set_yticks(range(len(top_categories)))
    axes[0, 0].set_yticklabels(top_categories)
    axes[0, 0].set_xlabel('Процент мошенничества (%)')
    axes[0, 0].set_title('Топ категорий вендоров по мошенничеству')
    
    # Мошенничество по каналам
    axes[0, 1].bar(fraud_by_channel.index, fraud_by_channel['mean'])
    axes[0, 1].set_xlabel('Канал')
    axes[0, 1].set_ylabel('Процент мошенничества (%)')
    axes[0, 1].set_title('Мошенничество по каналам')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Мошенничество по типу карты
    axes[1, 0].bar(fraud_by_card_type.index, fraud_by_card_type['mean'])
    axes[1, 0].set_xlabel('Тип карты')
    axes[1, 0].set_ylabel('Процент мошенничества (%)')
    axes[1, 0].set_title('Мошенничество по типу карты')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Мошенничество по странам (топ-10)
    top_countries = fraud_by_country.head(10).index
    top_countries_data = fraud_by_country.loc[top_countries]
    axes[1, 1].barh(range(len(top_countries)), top_countries_data['mean'])
    axes[1, 1].set_yticks(range(len(top_countries)))
    axes[1, 1].set_yticklabels(top_countries)
    axes[1, 1].set_xlabel('Процент мошенничества (%)')
    axes[1, 1].set_title('Топ стран по мошенничеству')
    
    plt.tight_layout()
    plt.show()

def correlation_analysis(transactions_df):
    """Анализ корреляций"""
    print("\n" + "=" * 80)
    print("АНАЛИЗ КОРРЕЛЯЦИЙ")
    print("=" * 80)
    
    # Создаем копию датафрейма для численных преобразований
    df_numeric = transactions_df.copy()
    
    # Преобразуем булевые колонки в числовые
    bool_cols = df_numeric.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df_numeric[col] = df_numeric[col].astype(int)
    
    # Выбираем числовые колонки для корреляционного анализа
    numeric_cols = df_numeric.select_dtypes(include=[np.number]).columns
    
    # Вычисляем корреляцию с is_fraud
    correlations_with_fraud = df_numeric[numeric_cols].corrwith(df_numeric['is_fraud']).sort_values(ascending=False)
    
    print("\nКорреляция признаков с is_fraud:")
    print("-" * 40)
    for feature, corr in correlations_with_fraud.items():
        if feature != 'is_fraud' and abs(corr) > 0.01:  # Показываем только значимые корреляции
            print(f"{feature:30} : {corr:+.4f}")
    
    # Матрица корреляций для топ признаков
    top_features = correlations_with_fraud.head(15).index.tolist()
    if 'is_fraud' not in top_features:
        top_features.append('is_fraud')
    
    correlation_matrix = df_numeric[top_features].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                square=True, cbar_kws={'shrink': 0.8}, center=0)
    plt.title('Корреляционная матрица (топ признаки)')
    plt.tight_layout()
    plt.show()

def summary_report(transactions_df, currency_df):
    """Сводный отчет по анализу"""
    print("\n" + "=" * 80)
    print("СВОДНЫЙ ОТЧЕТ")
    print("=" * 80)
    
    print("\nОСНОВНЫЕ ВЫВОДЫ:")
    print("-" * 40)
    
    # 1. Общая статистика
    total_transactions = len(transactions_df)
    fraud_transactions = transactions_df['is_fraud'].sum()
    fraud_percentage = (fraud_transactions / total_transactions) * 100
    
    print(f"1. Всего транзакций: {total_transactions:,}")
    print(f"   Мошеннических: {fraud_transactions:,} ({fraud_percentage:.2f}%)")
    
    # 2. Временные характеристики
    date_range = transactions_df['timestamp'].dt.date
    print(f"\n2. Период данных:")
    print(f"   От: {date_range.min()} до: {date_range.max()}")
    print(f"   Всего дней: {(date_range.max() - date_range.min()).days + 1}")
    
    # 3. Географический охват
    unique_countries = transactions_df['country'].nunique()
    unique_cities = transactions_df['city'].nunique()
    print(f"\n3. Географический охват:")
    print(f"   Уникальных стран: {unique_countries}")
    print(f"   Уникальных городов: {unique_cities}")
    
    # 4. Валюты
    unique_currencies = transactions_df['currency'].nunique()
    print(f"\n4. Валюты:")
    print(f"   Уникальных валют в транзакциях: {unique_currencies}")
    print(f"   Валют в курсах обмена: {len(currency_df.columns) - 1}")
    
    # 5. Клиенты и карты
    unique_customers = transactions_df['customer_id'].nunique()
    unique_cards = transactions_df['card_number'].nunique()
    print(f"\n5. Клиенты и карты:")
    print(f"   Уникальных клиентов: {unique_customers}")
    print(f"   Уникальных карт: {unique_cards}")
    
    # 6. Пропущенные значения
    missing_total = transactions_df.isnull().sum().sum()
    missing_percentage_total = (missing_total / (transactions_df.shape[0] * transactions_df.shape[1])) * 100
    print(f"\n6. Качество данных:")
    print(f"   Всего пропущенных значений: {missing_total:,}")
    print(f"   Процент пропусков: {missing_percentage_total:.2f}%")
    
    # 7. Рекомендации для дальнейшего анализа
    print("\n7. РЕКОМЕНДАЦИИ ДЛЯ ДАЛЬНЕЙШЕГО АНАЛИЗА:")
    print("-" * 40)
    print("""
    1. Feature Engineering:
       - Создание новых признаков на основе времени
       - Агрегация по клиентам
       - Нормализация суммы транзакций
    
    2. Подготовка данных для ML:
       - Обработка категориальных признаков (one-hot encoding)
       - Масштабирование числовых признаков
       - Балансировка классов (SMOTE/undersampling)
    
    3. Моделирование:
       - Эксперименты с разными алгоритмами
       - Настройка гиперпараметров
       - Валидация на временных срезах
    """)

def main():
    """Основная функция выполнения EDA"""
    print("НАЧАЛО РАЗВЕДОЧНОГО АНАЛИЗА ДАННЫХ")
    print("=" * 80)
    
    # Загрузка данных
    transactions_df, currency_df = load_data()
    
    # Выполнение анализа
    basic_info(transactions_df, currency_df)
    check_missing_values(transactions_df, currency_df)
    analyze_transaction_features(transactions_df)
    analyze_currency_exchange(currency_df)
    analyze_last_hour_activity(transactions_df)
    analyze_fraud_patterns(transactions_df)
    correlation_analysis(transactions_df)
    summary_report(transactions_df, currency_df)
    
    print("\n" + "=" * 80)
    print("АНАЛИЗ ЗАВЕРШЕН")
    print("=" * 80)

if __name__ == "__main__":
    main()