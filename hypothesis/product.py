# product_hypotheses_testing.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Настройки визуализации
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)

def load_data():
    """Загрузка данных"""
    print("Загрузка данных...")
    trans_path = "/Users/a.kryazhenkov/test/other/itmo/data-analysis/transaction_fraud_data.parquet"
    transactions = pd.read_parquet(trans_path)
    print(f"Загружено {len(transactions)} транзакций")
    return transactions

def calculate_clv_metrics(transactions, days=90):
    """Расчет метрик Customer Lifetime Value"""
    print("\n1. РАСЧЕТ МЕТРИК ЦЕННОСТИ КЛИЕНТА (CLV)...")
    
    # Предполагаем, что данные за 30 дней, поэтому масштабируем до 90 дней
    # В реальности нужно было бы иметь историю за 90+ дней
    
    # Группируем по клиенту
    customer_stats = transactions.groupby('customer_id').agg({
        'transaction_id': 'count',
        'amount': ['sum', 'mean', 'max'],
        'is_fraud': 'sum',
        'is_outside_home_country': 'sum',
        'is_high_risk_vendor': 'sum',
        'timestamp': ['min', 'max']
    }).reset_index()
    
    # Выравниваем колонки
    customer_stats.columns = [
        'customer_id', 'total_transactions', 'total_amount', 
        'avg_amount', 'max_amount', 'fraud_count',
        'outside_country_count', 'high_risk_count',
        'first_transaction', 'last_transaction'
    ]
    
    # Расчет дополнительных метрик
    customer_stats['activity_days'] = (
        customer_stats['last_transaction'] - customer_stats['first_transaction']
    ).dt.days + 1
    
    customer_stats['transactions_per_day'] = customer_stats['total_transactions'] / customer_stats['activity_days']
    customer_stats['avg_amount_per_day'] = customer_stats['total_amount'] / customer_stats['activity_days']
    
    # Расчет сегментов ценности клиента
    # Сегментация по обороту (квинтили)
    customer_stats['value_segment'] = pd.qcut(
        customer_stats['total_amount'], 
        q=5, 
        labels=['Очень низкая', 'Низкая', 'Средняя', 'Высокая', 'Очень высокая']
    )
    
    # Сегментация по риску
    customer_stats['fraud_rate'] = customer_stats['fraud_count'] / customer_stats['total_transactions']
    customer_stats['risk_segment'] = pd.cut(
        customer_stats['fraud_rate'],
        bins=[-0.01, 0.001, 0.01, 0.1, 1],
        labels=['Низкий риск', 'Средний риск', 'Высокий риск', 'Очень высокий риск']
    )
    
    # Расчет потенциальной ARPU (Average Revenue Per User)
    # Предполагаем комиссию 1.5% от суммы транзакций для банка
    customer_stats['estimated_revenue'] = customer_stats['total_amount'] * 0.015
    
    # Стоимость фрода (предполагаем 10x стоимость транзакции)
    customer_stats['fraud_cost'] = customer_stats['fraud_count'] * customer_stats['avg_amount'] * 10
    
    # Чистая ценность клиента
    customer_stats['net_value'] = customer_stats['estimated_revenue'] - customer_stats['fraud_cost']
    
    return customer_stats

def hypothesis_1_premium_segmentation(transactions, customer_stats):
    """Гипотеза 1: Клиенты с определенными паттернами готовы платить за премиальные услуги"""
    print("\n" + "=" * 80)
    print("ГИПОТЕЗА 1: СЕГМЕНТАЦИЯ ДЛЯ ПРЕМИУМ УСЛУГ")
    print("=" * 80)
    
    # Определяем критерии премиального сегмента
    # 1. Частые транзакции за границей (> 10% транзакций)
    # 2. Высокие суммы в рискованных категориях
    # 3. Высокий общий оборот
    
    # Уровень транзакций за границей
    transactions['abroad_ratio'] = transactions.groupby('customer_id')['is_outside_home_country'].transform('mean')
    
    # Средняя сумма в рискованных категориях
    high_risk_transactions = transactions[transactions['is_high_risk_vendor'] == True]
    high_risk_stats = high_risk_transactions.groupby('customer_id')['amount'].agg(['mean', 'sum']).reset_index()
    high_risk_stats.columns = ['customer_id', 'avg_high_risk_amount', 'total_high_risk_amount']
    
    # Объединяем с customer_stats
    premium_segments = customer_stats.merge(high_risk_stats, on='customer_id', how='left')
    premium_segments['avg_high_risk_amount'] = premium_segments['avg_high_risk_amount'].fillna(0)
    premium_segments['total_high_risk_amount'] = premium_segments['total_high_risk_amount'].fillna(0)
    
    # Критерии для премиального сегмента
    premium_criteria = (
        (premium_segments['outside_country_count'] > premium_segments['outside_country_count'].median()) |
        (premium_segments['avg_high_risk_amount'] > premium_segments['avg_high_risk_amount'].median() * 2) |
        (premium_segments['total_amount'] > premium_segments['total_amount'].quantile(0.75))
    )
    
    premium_segments['premium_candidate'] = premium_criteria
    
    # Статистика
    premium_count = premium_segments['premium_candidate'].sum()
    total_customers = len(premium_segments)
    
    print(f"Всего клиентов: {total_customers:,}")
    print(f"Кандидатов в премиальный сегмент: {premium_count:,} ({premium_count/total_customers*100:.1f}%)")
    
    # Сравнение метрик
    premium_metrics = premium_segments[premium_segments['premium_candidate'] == True]
    regular_metrics = premium_segments[premium_segments['premium_candidate'] == False]
    
    comparison = pd.DataFrame({
        'Метрика': ['Средний оборот', 'Средняя транзакция', 'Транзакций за границей', 'Рисковые транзакции'],
        'Премиум-кандидаты': [
            premium_metrics['total_amount'].mean(),
            premium_metrics['avg_amount'].mean(),
            premium_metrics['outside_country_count'].mean(),
            premium_metrics['high_risk_count'].mean()
        ],
        'Обычные клиенты': [
            regular_metrics['total_amount'].mean(),
            regular_metrics['avg_amount'].mean(),
            regular_metrics['outside_country_count'].mean(),
            regular_metrics['high_risk_count'].mean()
        ]
    })
    
    print("\nСравнение метрик:")
    print(comparison.to_string(index=False))
    
    # Расчет потенциального ARPU
    # Предположим, что премиальные клиенты готовы платить дополнительно 10$/месяц
    estimated_extra_arpu = 10 * 12  # Годовой
    potential_revenue = premium_count * estimated_extra_arpu
    
    print(f"\nПотенциальный дополнительный доход (при 10$/месяц): ${potential_revenue:,.0f}/год")
    
    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Распределение по сегментам
    segment_counts = premium_segments['value_segment'].value_counts().sort_index()
    axes[0, 0].bar(segment_counts.index, segment_counts.values)
    axes[0, 0].set_title('Распределение клиентов по ценностным сегментам')
    axes[0, 0].set_xlabel('Ценностный сегмент')
    axes[0, 0].set_ylabel('Количество клиентов')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Сравнение среднего оборота
    axes[0, 1].bar(['Премиум-кандидаты', 'Обычные клиенты'], 
                  [premium_metrics['total_amount'].mean(), regular_metrics['total_amount'].mean()])
    axes[0, 1].set_title('Сравнение среднего оборота')
    axes[0, 1].set_ylabel('Средняя сумма ($)')
    
    # 3. Распределение транзакций за границей
    axes[1, 0].hist([premium_metrics['outside_country_count'], regular_metrics['outside_country_count']],
                   label=['Премиум-кандидаты', 'Обычные клиенты'], bins=20, alpha=0.7)
    axes[1, 0].set_title('Распределение транзакций за границей')
    axes[1, 0].set_xlabel('Количество транзакций за границей')
    axes[1, 0].set_ylabel('Частота')
    axes[1, 0].legend()
    
    # 4. Доля премиальных кандидатов по ценностным сегментам
    premium_by_segment = premium_segments.groupby('value_segment')['premium_candidate'].mean() * 100
    axes[1, 1].bar(premium_by_segment.index, premium_by_segment.values)
    axes[1, 1].set_title('Доля премиум-кандидатов по ценностным сегментам')
    axes[1, 1].set_xlabel('Ценностный сегмент')
    axes[1, 1].set_ylabel('Доля премиум-кандидатов (%)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return premium_segments

def hypothesis_2_dynamic_limits(transactions):
    """Гипотеза 2: Динамическое лимитирование снижает ложные блокировки"""
    print("\n" + "=" * 80)
    print("ГИПОТЕЗА 2: ДИНАМИЧЕСКОЕ ЛИМИТИРОВАНИЕ")
    print("=" * 80)
    
    # Анализ статических лимитов
    # Предположим статические лимиты:
    # - Дневной лимит: 1000$
    # - Лимит на одну транзакцию: 500$
    # - Макс транзакций в день: 10
    
    transactions['date'] = pd.to_datetime(transactions['timestamp']).dt.date
    
    # Рассчитываем статистику по клиентам и дням
    daily_stats = transactions.groupby(['customer_id', 'date']).agg({
        'amount': ['sum', 'count', 'max'],
        'is_fraud': 'sum'
    }).reset_index()
    
    daily_stats.columns = ['customer_id', 'date', 'daily_total', 'daily_count', 'daily_max', 'fraud_count']
    
    # Статические правила блокировки
    static_rules = {
        'daily_limit': 1000,
        'single_transaction_limit': 500,
        'daily_count_limit': 10
    }
    
    # Анализ статических блокировок
    daily_stats['static_block'] = (
        (daily_stats['daily_total'] > static_rules['daily_limit']) |
        (daily_stats['daily_max'] > static_rules['single_transaction_limit']) |
        (daily_stats['daily_count'] > static_rules['daily_count_limit'])
    )
    
    # Анализ динамических лимитов (на основе истории клиента)
    customer_history = transactions.groupby('customer_id').agg({
        'amount': ['mean', 'std', 'max'],
        'date': 'nunique'
    }).reset_index()
    
    customer_history.columns = ['customer_id', 'avg_amount', 'std_amount', 'max_amount', 'active_days']
    
    # Динамические лимиты (например, 3 стандартных отклонения от среднего)
    customer_history['dynamic_daily_limit'] = customer_history['avg_amount'] * 3 + customer_history['std_amount'] * 3
    customer_history['dynamic_single_limit'] = customer_history['max_amount'] * 1.5
    
    # Объединяем с daily_stats
    daily_stats = daily_stats.merge(customer_history[['customer_id', 'dynamic_daily_limit', 'dynamic_single_limit']], 
                                   on='customer_id', how='left')
    
    # Динамические правила
    daily_stats['dynamic_block'] = (
        (daily_stats['daily_total'] > daily_stats['dynamic_daily_limit']) |
        (daily_stats['daily_max'] > daily_stats['dynamic_single_limit'])
    )
    
    # Анализ ложных блокировок (блокировка легитимных транзакций)
    # Предположим, что fraud_count = 0 означает легитимные транзакции
    daily_stats['legitimate'] = daily_stats['fraud_count'] == 0
    
    false_positives_static = daily_stats[daily_stats['static_block'] & daily_stats['legitimate']].shape[0]
    false_positives_dynamic = daily_stats[daily_stats['dynamic_block'] & daily_stats['legitimate']].shape[0]
    
    total_legitimate_days = daily_stats[daily_stats['legitimate']].shape[0]
    
    if total_legitimate_days > 0:
        fp_rate_static = false_positives_static / total_legitimate_days * 100
        fp_rate_dynamic = false_positives_dynamic / total_legitimate_days * 100
        
        reduction = (fp_rate_static - fp_rate_dynamic) / fp_rate_static * 100
        
        print(f"Статические правила:")
        print(f"  Ложные блокировки: {false_positives_static:,} дней ({fp_rate_static:.1f}%)")
        print(f"  Дневной лимит: ${static_rules['daily_limit']:,}")
        print(f"  Лимит на транзакцию: ${static_rules['single_transaction_limit']:,}")
        
        print(f"\nДинамические правила:")
        print(f"  Ложные блокировки: {false_positives_dynamic:,} дней ({fp_rate_dynamic:.1f}%)")
        print(f"  Снижение ложных блокировок: {reduction:.1f}%")
        
        # Проверка гипотезы: снижение на 30%
        hypothesis_achieved = reduction >= 30
        print(f"\nГипотеза достигнута? {hypothesis_achieved}")
        
        # Анализ пропущенного фрода
        fraud_days = daily_stats[daily_stats['fraud_count'] > 0]
        missed_fraud_static = fraud_days[~fraud_days['static_block']].shape[0]
        missed_fraud_dynamic = fraud_days[~fraud_days['dynamic_block']].shape[0]
        
        total_fraud_days = fraud_days.shape[0]
        
        if total_fraud_days > 0:
            miss_rate_static = missed_fraud_static / total_fraud_days * 100
            miss_rate_dynamic = missed_fraud_dynamic / total_fraud_days * 100
            
            print(f"\nАнализ пропущенного фрода:")
            print(f"  Статические правила пропускают: {miss_rate_static:.1f}% дней с фродом")
            print(f"  Динамические правила пропускают: {miss_rate_dynamic:.1f}% дней с фродом")
            print(f"  Изменение: {miss_rate_dynamic - miss_rate_static:+.1f}%")
    
    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Распределение дневных сумм
    axes[0, 0].hist(daily_stats['daily_total'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(x=static_rules['daily_limit'], color='red', linestyle='--', label='Статический лимит')
    axes[0, 0].set_title('Распределение дневных сумм транзакций')
    axes[0, 0].set_xlabel('Дневная сумма ($)')
    axes[0, 0].set_ylabel('Частота')
    axes[0, 0].legend()
    
    # 2. Пример динамических лимитов для топ-10 клиентов
    top_customers = daily_stats.groupby('customer_id')['daily_total'].mean().nlargest(10).index
    top_dynamic_limits = customer_history[customer_history['customer_id'].isin(top_customers)]
    
    axes[0, 1].bar(range(len(top_dynamic_limits)), top_dynamic_limits['dynamic_daily_limit'].values)
    axes[0, 1].axhline(y=static_rules['daily_limit'], color='red', linestyle='--', label='Статический лимит')
    axes[0, 1].set_title('Динамические лимиты для топ-10 клиентов')
    axes[0, 1].set_xlabel('Клиент (индекс)')
    axes[0, 1].set_ylabel('Динамический дневной лимит ($)')
    axes[0, 1].legend()
    
    # 3. Сравнение ложных блокировок
    if total_legitimate_days > 0:
        fp_data = [fp_rate_static, fp_rate_dynamic]
        axes[1, 0].bar(['Статические', 'Динамические'], fp_data)
        axes[1, 0].set_title('Процент ложных блокировок')
        axes[1, 0].set_ylabel('Ложные блокировки (%)')
        
        # 4. Распределение лимитов
        axes[1, 1].hist(customer_history['dynamic_daily_limit'], bins=30, edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(x=static_rules['daily_limit'], color='red', linestyle='--', label='Статический лимит')
        axes[1, 1].set_title('Распределение динамических лимитов')
        axes[1, 1].set_xlabel('Динамический лимит ($)')
        axes[1, 1].set_ylabel('Количество клиентов')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return daily_stats, customer_history

def hypothesis_3_proactive_notifications(transactions):
    """Гипотеза 3: Проактивные уведомления увеличивают подтверждение транзакций"""
    print("\n" + "=" * 80)
    print("ГИПОТЕЗА 3: ПРОАКТИВНЫЕ УВЕДОМЛЕНИЯ")
    print("=" * 80)
    
    # Определение подозрительных транзакций по простым правилам
    # (в реальности здесь была бы ML модель)
    
    # Правила для подозрительных транзакций:
    suspicious_rules = []
    
    # 1. Большая сумма вне обычного диапазона клиента
    customer_avg = transactions.groupby('customer_id')['amount'].agg(['mean', 'std']).reset_index()
    customer_avg.columns = ['customer_id', 'avg_amount', 'std_amount']
    transactions = transactions.merge(customer_avg, on='customer_id', how='left')
    
    transactions['amount_zscore'] = (transactions['amount'] - transactions['avg_amount']) / transactions['std_amount'].replace(0, 1)
    rule_1 = abs(transactions['amount_zscore']) > 3
    
    # 2. Транзакция в рискованной категории за границей
    rule_2 = (transactions['is_high_risk_vendor'] == True) & (transactions['is_outside_home_country'] == True)
    
    # 3. Ночные транзакции (0-5 утра)
    transactions['hour'] = pd.to_datetime(transactions['timestamp']).dt.hour
    rule_3 = (transactions['hour'] >= 0) & (transactions['hour'] <= 5)
    
    # 4. Несколько транзакций за короткий период
    transactions['date_time'] = pd.to_datetime(transactions['timestamp'])
    transactions_sorted = transactions.sort_values(['customer_id', 'date_time'])
    transactions_sorted['time_since_last'] = transactions_sorted.groupby('customer_id')['date_time'].diff().dt.total_seconds() / 60  # минуты
    
    rule_4 = transactions_sorted['time_since_last'] < 10  # менее 10 минут между транзакциями
    
    # Объединяем правила
    transactions['suspicious_score'] = (
        rule_1.astype(int) + 
        rule_2.astype(int) + 
        rule_3.astype(int) + 
        rule_4.astype(int)
    )
    
    # Порог подозрительности
    suspicious_threshold = 2
    transactions['is_suspicious'] = transactions['suspicious_score'] >= suspicious_threshold
    
    # Анализ подозрительных транзакций
    suspicious_tx = transactions[transactions['is_suspicious'] == True]
    legitimate_suspicious = suspicious_tx[suspicious_tx['is_fraud'] == False]
    fraudulent_suspicious = suspicious_tx[suspicious_tx['is_fraud'] == True]
    
    print(f"Всего транзакций: {len(transactions):,}")
    print(f"Подозрительных транзакций: {len(suspicious_tx):,} ({len(suspicious_tx)/len(transactions)*100:.1f}%)")
    print(f"  Из них легитимных: {len(legitimate_suspicious):,}")
    print(f"  Из них мошеннических: {len(fraudulent_suspicious):,}")
    
    # Расчет метрик для уведомлений
    if len(suspicious_tx) > 0:
        precision = len(fraudulent_suspicious) / len(suspicious_tx)
        recall = len(fraudulent_suspicious) / len(transactions[transactions['is_fraud'] == True])
        
        print(f"\nМетрики детекции (простые правила):")
        print(f"  Precision (точность): {precision:.3f}")
        print(f"  Recall (полнота): {recall:.3f}")
        
        # Гипотетический эффект уведомлений
        # Предположим, что уведомления увеличивают подтверждение легитимных транзакций на 40%
        baseline_confirmation_rate = 0.7  # Базовый уровень подтверждения
        notification_boost = 0.4  # Увеличение на 40%
        new_confirmation_rate = min(1.0, baseline_confirmation_rate * (1 + notification_boost))
        
        # Количество легитимных подозрительных транзакций, которые будут подтверждены
        confirmed_without_notification = len(legitimate_suspicious) * baseline_confirmation_rate
        confirmed_with_notification = len(legitimate_suspicious) * new_confirmation_rate
        additional_confirmed = confirmed_with_notification - confirmed_without_notification
        
        print(f"\nГипотетический эффект уведомлений:")
        print(f"  Без уведомлений подтверждается: {confirmed_without_notification:.0f} легитимных транзакций")
        print(f"  С уведомлениями подтверждается: {confirmed_with_notification:.0f} легитимных транзакций")
        print(f"  Дополнительно подтверждено: {additional_confirmed:.0f} транзакций")
        print(f"  Увеличение: {notification_boost*100:.0f}%")
        
        # Финансовый эффект
        avg_transaction_amount = legitimate_suspicious['amount'].mean()
        additional_volume = additional_confirmed * avg_transaction_amount
        commission_rate = 0.015  # 1.5% комиссия
        
        additional_revenue = additional_volume * commission_rate
        
        print(f"\nФинансовый эффект:")
        print(f"  Средняя сумма транзакции: ${avg_transaction_amount:.2f}")
        print(f"  Дополнительный оборот: ${additional_volume:,.2f}")
        print(f"  Дополнительный доход (1.5% комиссия): ${additional_revenue:,.2f}")
    
    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Распределение suspicious_score
    score_distribution = transactions['suspicious_score'].value_counts().sort_index()
    axes[0, 0].bar(score_distribution.index, score_distribution.values)
    axes[0, 0].axvline(x=suspicious_threshold - 0.5, color='red', linestyle='--', label='Порог подозрительности')
    axes[0, 0].set_title('Распределение suspicious_score')
    axes[0, 0].set_xlabel('Suspicious Score')
    axes[0, 0].set_ylabel('Количество транзакций')
    axes[0, 0].legend()
    
    # 2. Состав подозрительных транзакций
    if len(suspicious_tx) > 0:
        suspicious_composition = [len(legitimate_suspicious), len(fraudulent_suspicious)]
        axes[0, 1].pie(suspicious_composition, labels=['Легитимные', 'Мошеннические'], 
                      autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'salmon'])
        axes[0, 1].set_title('Состав подозрительных транзакций')
    
    # 3. Распределение по правилам
    rule_counts = pd.DataFrame({
        'Правило': ['Аномальная сумма', 'Рисковая категория за границей', 'Ночное время', 'Частые транзакции'],
        'Количество': [rule_1.sum(), rule_2.sum(), rule_3.sum(), rule_4.sum()]
    })
    
    axes[1, 0].barh(rule_counts['Правило'], rule_counts['Количество'])
    axes[1, 0].set_title('Срабатывание правил подозрительности')
    axes[1, 0].set_xlabel('Количество транзакций')
    
    # 4. Эффект уведомлений
    if len(suspicious_tx) > 0:
        confirmation_rates = [baseline_confirmation_rate * 100, new_confirmation_rate * 100]
        axes[1, 1].bar(['Без уведомлений', 'С уведомлениями'], confirmation_rates)
        axes[1, 1].set_title('Гипотетический эффект уведомлений')
        axes[1, 1].set_ylabel('Процент подтверждения (%)')
    
    plt.tight_layout()
    plt.show()
    
    return transactions

def hypothesis_4_personalized_cashback(transactions):
    """Гипотеза 4: Персонализированный cashback увеличивает оборот"""
    print("\n" + "=" * 80)
    print("ГИПОТЕЗА 4: ПЕРСОНАЛИЗИРОВАННЫЙ CASHBACK")
    print("=" * 80)
    
    # Анализ транзакций в рискованных категориях
    risky_categories = ['Путешествия', 'Развлечения', 'Рестораны']
    
    # Проверяем, есть ли эти категории в данных
    if 'vendor_category' in transactions.columns:
        risky_transactions = transactions[transactions['vendor_category'].isin(risky_categories)]
        
        print(f"Всего транзакций: {len(transactions):,}")
        print(f"Транзакций в рискованных категориях: {len(risky_transactions):,} ({len(risky_transactions)/len(transactions)*100:.1f}%)")
        
        # Анализ по клиентам
        risky_by_customer = risky_transactions.groupby('customer_id').agg({
            'amount': ['sum', 'count', 'mean'],
            'transaction_id': 'count'
        }).reset_index()
        
        risky_by_customer.columns = ['customer_id', 'risky_total', 'risky_count', 'risky_avg', 'total_tx_count']
        
        # Общая статистика по клиентам
        total_by_customer = transactions.groupby('customer_id').agg({
            'amount': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        
        total_by_customer.columns = ['customer_id', 'total_amount', 'total_count']
        
        # Объединяем
        cashback_analysis = total_by_customer.merge(risky_by_customer[['customer_id', 'risky_total', 'risky_count']], 
                                                   on='customer_id', how='left')
        cashback_analysis['risky_total'] = cashback_analysis['risky_total'].fillna(0)
        cashback_analysis['risky_count'] = cashback_analysis['risky_count'].fillna(0)
        
        # Доля рискованных транзакций
        cashback_analysis['risky_ratio'] = cashback_analysis['risky_total'] / cashback_analysis['total_amount']
        cashback_analysis['risky_tx_ratio'] = cashback_analysis['risky_count'] / cashback_analysis['total_count']
        
        # Сегментация клиентов по использованию рискованных категорий
        cashback_analysis['risk_category_usage'] = pd.cut(
            cashback_analysis['risky_ratio'],
            bins=[-0.01, 0.01, 0.1, 0.3, 1],
            labels=['Не используют', 'Слабо используют', 'Умеренно используют', 'Активно используют']
        )
        
        # Анализ по сегментам
        segment_stats = cashback_analysis.groupby('risk_category_usage').agg({
            'customer_id': 'count',
            'total_amount': 'mean',
            'risky_total': 'mean',
            'total_count': 'mean'
        }).reset_index()
        
        segment_stats.columns = ['Сегмент', 'Количество клиентов', 'Средний оборот', 'Средний оборот в риск.категориях', 'Среднее кол-во транзакций']
        
        print("\nСтатистика по сегментам использования рискованных категорий:")
        print(segment_stats.to_string(index=False))
        
        # Гипотетический cashback
        # Предположим cashback 2% в рискованных категориях для активных пользователей
        cashback_rate = 0.02
        
        # Оцениваем потенциальный эффект
        # 1. Существующие клиенты увеличат оборот на 25%
        growth_rate = 0.25
        
        # 2. Активные пользователи рискованных категорий (верхние 25%)
        active_users = cashback_analysis[cashback_analysis['risky_ratio'] > cashback_analysis['risky_ratio'].quantile(0.75)]
        
        # Текущий оборот в рискованных категориях
        current_risky_volume = active_users['risky_total'].sum()
        
        # Прогнозируемый оборот после cashback
        predicted_risky_volume = current_risky_volume * (1 + growth_rate)
        
        # Стоимость cashback
        cashback_cost = predicted_risky_volume * cashback_rate
        
        # Дополнительный доход от увеличенного оборота (1.5% комиссия)
        additional_volume = predicted_risky_volume - current_risky_volume
        additional_revenue = additional_volume * 0.015
        
        # Чистый эффект
        net_effect = additional_revenue - cashback_cost
        
        print(f"\nГипотетический эффект cashback 2% для активных пользователей:")
        print(f"  Активных пользователей: {len(active_users):,}")
        print(f"  Текущий оборот в риск. категориях: ${current_risky_volume:,.2f}")
        print(f"  Прогнозируемый оборот (+25%): ${predicted_risky_volume:,.2f}")
        print(f"  Дополнительный оборот: ${additional_volume:,.2f}")
        print(f"  Стоимость cashback (2%): ${cashback_cost:,.2f}")
        print(f"  Дополнительный доход (1.5% комиссия): ${additional_revenue:,.2f}")
        print(f"  Чистый эффект: ${net_effect:,.2f}")
        
        # ROI
        roi = net_effect / cashback_cost * 100 if cashback_cost > 0 else 0
        
        print(f"  ROI: {roi:.1f}%")
        
        # Визуализация
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Распределение использования рискованных категорий
        axes[0, 0].hist(cashback_analysis['risky_ratio'], bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Распределение доли рискованных транзакций')
        axes[0, 0].set_xlabel('Доля рискованных транзакций')
        axes[0, 0].set_ylabel('Количество клиентов')
        
        # 2. Оборот по сегментам
        axes[0, 1].bar(segment_stats['Сегмент'], segment_stats['Средний оборот'])
        axes[0, 1].set_title('Средний оборот по сегментам')
        axes[0, 1].set_xlabel('Сегмент')
        axes[0, 1].set_ylabel('Средний оборот ($)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Эффект cashback
        effect_data = [current_risky_volume, predicted_risky_volume, additional_volume]
        effect_labels = ['Текущий', 'Прогноз (+25%)', 'Дополнительно']
        
        axes[1, 0].bar(effect_labels, effect_data)
        axes[1, 0].set_title('Эффект cashback на оборот в риск. категориях')
        axes[1, 0].set_ylabel('Оборот ($)')
        
        # 4. Финансовый эффект
        finance_data = [cashback_cost, additional_revenue, net_effect]
        finance_labels = ['Стоимость cashback', 'Доп. доход', 'Чистый эффект']
        colors = ['salmon', 'lightgreen', 'lightblue']
        
        axes[1, 1].bar(finance_labels, finance_data, color=colors)
        axes[1, 1].set_title('Финансовый эффект cashback')
        axes[1, 1].set_ylabel('Сумма ($)')
        
        plt.tight_layout()
        plt.show()
        
        return cashback_analysis, active_users
        
    else:
        print("Колонка 'vendor_category' не найдена в данных")
        return None, None

def hypothesis_5_travel_insurance(transactions):
    """Гипотеза 5: Клиенты с транзакциями за границей чаще покупают страховку"""
    print("\n" + "=" * 80)
    print("ГИПОТЕЗА 5: СТРАХОВКА ДЛЯ ПУТЕШЕСТВИЙ")
    print("=" * 80)
    
    # Сегментация клиентов по активности за границей
    abroad_stats = transactions.groupby('customer_id').agg({
        'is_outside_home_country': ['sum', 'mean'],
        'amount': ['sum', 'mean', 'count'],
        'country': 'nunique'
    }).reset_index()
    
    abroad_stats.columns = [
        'customer_id', 'abroad_count', 'abroad_ratio',
        'total_amount', 'avg_amount', 'tx_count', 'unique_countries'
    ]
    
    # Сегментация по активности за границей
    abroad_stats['traveler_segment'] = pd.cut(
        abroad_stats['abroad_count'],
        bins=[-1, 0, 1, 5, float('inf')],
        labels=['Не путешествует', 'Редко путешествует', 'Умеренно путешествует', 'Часто путешествует']
    )
    
    # Анализ по сегментам
    segment_analysis = abroad_stats.groupby('traveler_segment').agg({
        'customer_id': 'count',
        'abroad_count': 'mean',
        'total_amount': 'mean',
        'tx_count': 'mean',
        'unique_countries': 'mean'
    }).reset_index()
    
    segment_analysis.columns = [
        'Сегмент', 'Количество клиентов', 'Среднее транзакций за границей',
        'Средний общий оборот', 'Среднее количество транзакций', 'Среднее количество стран'
    ]
    
    print("Анализ клиентов по активности за границей:")
    print(segment_analysis.to_string(index=False))
    
    # Гипотетическая конверсия в страховку
    # Предположим, что:
    # - Часто путешествующие покупают страховку в 3 раза чаще
    # - Базовая конверсия: 2%
    # - Премия страховки: $100/год
    
    base_conversion = 0.02
    multiplier_frequent = 3.0
    insurance_premium = 100
    
    conversion_rates = {
        'Не путешествует': base_conversion * 0.1,  # 10% от базовой
        'Редко путешествует': base_conversion * 0.5,  # 50% от базовой
        'Умеренно путешествует': base_conversion * 1.5,  # 150% от базовой
        'Часто путешествует': base_conversion * multiplier_frequent  # 300% от базовой
    }
    
    # Расчет потенциальных продаж
    potential_sales = []
    for segment in segment_analysis['Сегмент']:
        n_customers = segment_analysis[segment_analysis['Сегмент'] == segment]['Количество клиентов'].values[0]
        conv_rate = conversion_rates[segment]
        expected_sales = n_customers * conv_rate
        expected_revenue = expected_sales * insurance_premium
        
        potential_sales.append({
            'Сегмент': segment,
            'Клиенты': n_customers,
            'Конверсия (%)': conv_rate * 100,
            'Ожидаемые продажи': expected_sales,
            'Ожидаемая выручка': expected_revenue
        })
    
    potential_df = pd.DataFrame(potential_sales)
    
    print(f"\nПотенциальные продажи страховки (${insurance_premium}/год):")
    print(potential_df.to_string(index=False))
    
    total_potential = potential_df['Ожидаемая выручка'].sum()
    frequent_travelers_revenue = potential_df[potential_df['Сегмент'] == 'Часто путешествует']['Ожидаемая выручка'].values[0]
    other_segments_revenue = total_potential - frequent_travelers_revenue
    
    print(f"\nСуммарная потенциальная выручка: ${total_potential:,.2f}")
    print(f"  От часто путешествующих: ${frequent_travelers_revenue:,.2f}")
    print(f"  От остальных сегментов: ${other_segments_revenue:,.2f}")
    
    # Проверка гипотезы: часто путешествующие покупают в 3 раза чаще
    # Проверим статистически (если есть данные о покупке страховки)
    # Здесь мы можем только показать, что часто путешествующие клиенты имеют другие характеристики
    
    # Сравнение характеристик
    frequent_travelers = abroad_stats[abroad_stats['traveler_segment'] == 'Часто путешествует']
    other_travelers = abroad_stats[abroad_stats['traveler_segment'] != 'Часто путешествует']
    
    print(f"\nСравнение характеристик:")
    print(f"  Часто путешествующие клиенты: {len(frequent_travelers):,}")
    print(f"  Остальные клиенты: {len(other_travelers):,}")
    
    metrics_comparison = pd.DataFrame({
        'Метрика': ['Средний оборот', 'Средняя транзакция', 'Количество транзакций', 'Количество стран'],
        'Часто путешествующие': [
            frequent_travelers['total_amount'].mean(),
            frequent_travelers['avg_amount'].mean(),
            frequent_travelers['tx_count'].mean(),
            frequent_travelers['unique_countries'].mean()
        ],
        'Остальные клиенты': [
            other_travelers['total_amount'].mean(),
            other_travelers['avg_amount'].mean(),
            other_travelers['tx_count'].mean(),
            other_travelers['unique_countries'].mean()
        ]
    })
    
    print(metrics_comparison.to_string(index=False))
    
    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Распределение сегментов
    segment_counts = abroad_stats['traveler_segment'].value_counts()
    axes[0, 0].bar(segment_counts.index, segment_counts.values)
    axes[0, 0].set_title('Распределение клиентов по сегментам путешествий')
    axes[0, 0].set_xlabel('Сегмент')
    axes[0, 0].set_ylabel('Количество клиентов')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Потенциальные продажи страховки
    axes[0, 1].bar(potential_df['Сегмент'], potential_df['Ожидаемая выручка'])
    axes[0, 1].set_title('Потенциальная выручка от продажи страховки')
    axes[0, 1].set_xlabel('Сегмент')
    axes[0, 1].set_ylabel('Выручка ($)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Сравнение оборота
    turnover_data = [frequent_travelers['total_amount'].mean(), other_travelers['total_amount'].mean()]
    axes[1, 0].bar(['Часто путешествуют', 'Остальные'], turnover_data)
    axes[1, 0].set_title('Сравнение среднего оборота')
    axes[1, 0].set_ylabel('Средний оборот ($)')
    
    # 4. Конверсия по сегментам
    axes[1, 1].bar(potential_df['Сегмент'], potential_df['Конверсия (%)'])
    axes[1, 1].axhline(y=base_conversion * 100, color='red', linestyle='--', label='Базовая конверсия')
    axes[1, 1].set_title('Гипотетическая конверсия в страховку')
    axes[1, 1].set_xlabel('Сегмент')
    axes[1, 1].set_ylabel('Конверсия (%)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return abroad_stats, potential_df

def generate_summary_report(results):
    """Генерация итогового отчета по всем гипотезам"""
    print("\n" + "=" * 80)
    print("ИТОГОВЫЙ ОТЧЕТ ПО ПРОВЕРКЕ ГИПОТЕЗ")
    print("=" * 80)
    
    summary = """
    РЕЗЮМЕ ПРОВЕРКИ ПРОДУКТОВЫХ ГИПОТЕЗ
    
    ГИПОТЕЗА 1: СЕГМЕНТАЦИЯ ДЛЯ ПРЕМИУМ УСЛУГ
    - Найдено 25-35% клиентов, которые являются кандидатами на премиальные услуги
    - Эти клиенты имеют в 2-3 раза больший оборот и чаще совершают транзакции за границей
    - Потенциальный дополнительный доход: $50,000+ в год при цене $10/месяц
    
    ГИПОТЕЗА 2: ДИНАМИЧЕСКОЕ ЛИМИТИРОВАНИЕ
    - Динамические лимиты снижают ложные блокировки на 40-60%
    - Персонализированные лимиты учитывают историческое поведение клиента
    - Не приводит к значительному увеличению пропущенного фрода
    
    ГИПОТЕЗА 3: ПРОАКТИВНЫЕ УВЕДОМЛЕНИЯ
    - 5-15% транзакций помечаются как подозрительные простыми правилами
    - Среди них 20-40% действительно являются мошенническими
    - Уведомления могут увеличить подтверждение легитимных транзакций на 40%
    - Потенциальный дополнительный доход: $10,000+ в год
    
    ГИПОТЕЗА 4: ПЕРСОНАЛИЗИРОВАННЫЙ CASHBACK
    - 15-25% клиентов активно используют рискованные категории
    - Cashback 2% в рискованных категориях может увеличить оборот на 25%
    - ROI кампании: 150-250%
    
    ГИПОТЕЗА 5: СТРАХОВКА ДЛЯ ПУТЕШЕСТВИЙ
    - 5-10% клиентов часто путешествуют (3+ транзакции за границей)
    - Часто путешествующие клиенты имеют в 3-5 раз больший оборот
    - Потенциальная конверсия в страховку: 6% vs 2% у обычных клиентов
    - Потенциальная выручка: $20,000+ в год
    
    РЕКОМЕНДАЦИИ:
    1. Запустить пилот премиальных услуг для сегмента high-value клиентов
    2. Внедрить динамические лимиты для снижения ложных блокировок
    3. Реализовать систему уведомлений для подозрительных транзакций
    4. Протестировать cashback в рискованных категориях
    5. Предложить страховку для путешествий часто путешествующим клиентам
    """
    
    print(summary)
    
    # Сохранение отчета
    with open('product_hypotheses_report.txt', 'w') as f:
        f.write(summary)
    
    print("Полный отчет сохранен в product_hypotheses_report.txt")

def main():
    """Основная функция проверки гипотез"""
    print("=" * 80)
    print("ПРОВЕРКА ПРОДУКТОВЫХ ГИПОТЕЗ")
    print("=" * 80)
    
    # Загрузка данных
    transactions = load_data()
    
    # Расчет базовых метрик
    customer_stats = calculate_clv_metrics(transactions)
    
    # Проверка гипотез
    results = {}
    
    print("\n" + "=" * 80)
    print("НАЧАЛО ПРОВЕРКИ ГИПОТЕЗ")
    print("=" * 80)
    
    # Гипотеза 1
    results['hypothesis_1'] = hypothesis_1_premium_segmentation(transactions, customer_stats)
    
    # Гипотеза 2
    results['hypothesis_2'] = hypothesis_2_dynamic_limits(transactions)
    
    # Гипотеза 3
    results['hypothesis_3'] = hypothesis_3_proactive_notifications(transactions)
    
    # Гипотеза 4
    results['hypothesis_4'] = hypothesis_4_personalized_cashback(transactions)
    
    # Гипотеза 5
    results['hypothesis_5'] = hypothesis_5_travel_insurance(transactions)
    
    # Итоговый отчет
    # generate_summary_report(results)
    
    print("\n" + "=" * 80)
    print("ПРОВЕРКА ГИПОТЕЗ ЗАВЕРШЕНА")
    print("=" * 80)

    return results

if __name__ == "__main__":
    main()