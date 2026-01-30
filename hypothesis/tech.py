# technical_hypotheses_testing.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ML библиотеки
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Графовые алгоритмы
import networkx as nx
from community import community_louvain

# Временные ряды
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Для ансамблей
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier

import warnings
warnings.filterwarnings('ignore')

# Настройки визуализации
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)

def load_and_preprocess_data():
    """Загрузка и предобработка данных"""
    print("Загрузка данных...")
    transactions = pd.read_parquet('transaction_fraud_data.parquet')
    
    # Базовая предобработка
    transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
    transactions['hour'] = transactions['timestamp'].dt.hour
    transactions['day_of_week'] = transactions['timestamp'].dt.dayofweek
    transactions['date'] = transactions['timestamp'].dt.date
    
    print(f"Загружено {len(transactions)} транзакций")
    return transactions

def hypothesis_1_temporal_patterns(transactions):
    """Гипотеза 1: Мошеннические транзакции группируются во времени"""
    print("\n" + "=" * 80)
    print("ГИПОТЕЗА 1: ВРЕМЕННЫЕ ПАТТЕРНЫ МОШЕННИЧЕСТВА")
    print("=" * 80)
    
    # 1. Анализ по часам
    hourly_fraud = transactions.groupby('hour')['is_fraud'].agg(['mean', 'count']).reset_index()
    hourly_fraud.columns = ['hour', 'fraud_rate', 'total_transactions']
    hourly_fraud['fraud_rate'] = hourly_fraud['fraud_rate'] * 100
    
    # 2. Анализ по дням недели
    daily_fraud = transactions.groupby('day_of_week')['is_fraud'].agg(['mean', 'count']).reset_index()
    daily_fraud.columns = ['day_of_week', 'fraud_rate', 'total_transactions']
    daily_fraud['fraud_rate'] = daily_fraud['fraud_rate'] * 100
    
    # 3. Статистические тесты на кластеризацию
    # Разделяем транзакции на 15-минутные интервалы
    transactions['15min_interval'] = transactions['timestamp'].dt.floor('15min')
    interval_counts = transactions.groupby('15min_interval').agg({
        'transaction_id': 'count',
        'is_fraud': 'sum'
    }).reset_index()
    
    interval_counts['fraud_rate'] = interval_counts['is_fraud'] / interval_counts['transaction_id']
    
    # Проверяем автокорреляцию во временном ряде
    from statsmodels.tsa.stattools import acf
    
    # Создаем временной ряд с почасовой агрегацией
    hourly_series = transactions.groupby(pd.Grouper(key='timestamp', freq='1H')).agg({
        'is_fraud': 'mean'
    }).reset_index()
    
    # Заполняем пропуски
    hourly_series['is_fraud'] = hourly_series['is_fraud'].fillna(0)
    
    # Анализ автокорреляции
    try:
        fraud_acf = acf(hourly_series['is_fraud'], nlags=24)  # 24 часа
        significant_lags = np.where(np.abs(fraud_acf) > 1.96/np.sqrt(len(hourly_series)))[0]
    except:
        fraud_acf = None
        significant_lags = []
    
    # 4. Поиск временных кластеров с помощью DBSCAN
    # Преобразуем время в числовой формат (минуты с начала суток)
    transactions['minutes_from_midnight'] = (
        transactions['timestamp'].dt.hour * 60 + 
        transactions['timestamp'].dt.minute
    )
    
    # Используем только мошеннические транзакции
    fraud_transactions = transactions[transactions['is_fraud'] == True]
    
    if len(fraud_transactions) > 10:
        # Подготавливаем данные для кластеризации
        time_features = fraud_transactions[['minutes_from_midnight', 'day_of_week']].copy()
        time_features_scaled = StandardScaler().fit_transform(time_features)
        
        # Кластеризация DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        fraud_transactions['time_cluster'] = dbscan.fit_predict(time_features_scaled)
        
        # Оценка кластеризации
        unique_clusters = len(set(fraud_transactions['time_cluster'])) - (1 if -1 in fraud_transactions['time_cluster'].values else 0)
        
        print(f"Найдено временных кластеров мошенничества: {unique_clusters}")
        print(f"Транзакций в кластерах: {len(fraud_transactions[fraud_transactions['time_cluster'] != -1])}")
        print(f"Шумовых точек: {len(fraud_transactions[fraud_transactions['time_cluster'] == -1])}")
        
        # Анализ кластеров
        if unique_clusters > 0:
            cluster_stats = fraud_transactions[fraud_transactions['time_cluster'] != -1].groupby('time_cluster').agg({
                'minutes_from_midnight': ['mean', 'std', 'count'],
                'day_of_week': lambda x: x.mode()[0] if len(x.mode()) > 0 else -1
            }).reset_index()
            
            print("\nСтатистика по временным кластерам:")
            for _, row in cluster_stats.iterrows():
                hour = row[('minutes_from_midnight', 'mean')] / 60
                std_hours = row[('minutes_from_midnight', 'std')] / 60
                count = row[('minutes_from_midnight', 'count')]
                day = row[('day_of_week', '<lambda>')]
                
                print(f"  Кластер {int(row['time_cluster'])}: {count} транзакций")
                print(f"    Среднее время: {int(hour)}:{int((hour % 1) * 60):02d} ± {std_hours:.1f} часов")
                print(f"    Типичный день: {day}")
    
    # 5. Проверка гипотезы: точность 85% при обнаружении по времени
    # Простая модель на основе времени
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    
    # Создаем признаки времени
    time_features = transactions[['hour', 'day_of_week']].copy()
    time_features['sin_hour'] = np.sin(2 * np.pi * time_features['hour'] / 24)
    time_features['cos_hour'] = np.cos(2 * np.pi * time_features['hour'] / 24)
    time_features['is_night'] = ((time_features['hour'] >= 0) & (time_features['hour'] <= 5)).astype(int)
    time_features['is_weekend'] = (time_features['day_of_week'] >= 5).astype(int)
    
    y = transactions['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(
        time_features, y, test_size=0.3, random_state=42, stratify=y
    )
    
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nМодель обнаружения по времени:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    # Проверяем гипотезу (F1 > 0.85)
    hypothesis_met = f1 > 0.85
    print(f"\nГипотеза достигнута (F1 > 0.85)? {hypothesis_met}")
    
    # Визуализация
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Мошенничество по часам
    axes[0, 0].plot(hourly_fraud['hour'], hourly_fraud['fraud_rate'], marker='o')
    axes[0, 0].set_title('Процент мошенничества по часам')
    axes[0, 0].set_xlabel('Час')
    axes[0, 0].set_ylabel('Процент мошенничества (%)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Мошенничество по дням недели
    day_names = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
    axes[0, 1].bar(daily_fraud['day_of_week'], daily_fraud['fraud_rate'])
    axes[0, 1].set_title('Процент мошенничества по дням недели')
    axes[0, 1].set_xlabel('День недели')
    axes[0, 1].set_ylabel('Процент мошенничества (%)')
    axes[0, 1].set_xticks(range(7))
    axes[0, 1].set_xticklabels(day_names)
    
    # 3. Автокорреляция
    if fraud_acf is not None:
        axes[0, 2].bar(range(len(fraud_acf)), fraud_acf)
        axes[0, 2].axhline(y=1.96/np.sqrt(len(hourly_series)), color='r', linestyle='--', alpha=0.5)
        axes[0, 2].axhline(y=-1.96/np.sqrt(len(hourly_series)), color='r', linestyle='--', alpha=0.5)
        axes[0, 2].set_title('Автокорреляция мошеннических транзакций')
        axes[0, 2].set_xlabel('Лаг (часы)')
        axes[0, 2].set_ylabel('Автокорреляция')
    
    # 4. Кластеризация по времени (если есть кластеры)
    if len(fraud_transactions) > 10 and unique_clusters > 0:
        scatter = axes[1, 0].scatter(
            fraud_transactions['hour'],
            fraud_transactions['day_of_week'],
            c=fraud_transactions['time_cluster'],
            cmap='tab20',
            alpha=0.6
        )
        axes[1, 0].set_title('Кластеризация мошеннических транзакций')
        axes[1, 0].set_xlabel('Час')
        axes[1, 0].set_ylabel('День недели')
        axes[1, 0].set_yticks(range(7))
        axes[1, 0].set_yticklabels(day_names)
        plt.colorbar(scatter, ax=axes[1, 0])
    
    # 5. Важность признаков времени
    if hasattr(model, 'coef_'):
        feature_importance = pd.DataFrame({
            'feature': time_features.columns,
            'importance': np.abs(model.coef_[0])
        }).sort_values('importance', ascending=False)
        
        axes[1, 1].barh(feature_importance['feature'], feature_importance['importance'])
        axes[1, 1].set_title('Важность временных признаков')
        axes[1, 1].set_xlabel('Важность')
    
    # 6. ROC-кривая для временной модели
    from sklearn.metrics import roc_curve, auc
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    axes[1, 2].plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    axes[1, 2].plot([0, 1], [0, 1], 'k--')
    axes[1, 2].set_title('ROC-кривая временной модели')
    axes[1, 2].set_xlabel('False Positive Rate')
    axes[1, 2].set_ylabel('True Positive Rate')
    axes[1, 2].legend(loc='lower right')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'hourly_fraud': hourly_fraud,
        'daily_fraud': daily_fraud,
        'model_metrics': {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1},
        'hypothesis_met': hypothesis_met
    }

def hypothesis_2_graph_analysis(transactions):
    """Гипотеза 2: Графовые связи увеличивают риск фрода"""
    print("\n" + "=" * 80)
    print("ГИПОТЕЗА 2: ГРАФОВЫЕ СВЯЗИ И РИСК МОШЕННИЧЕСТВА")
    print("=" * 80)
    
    try:
        import networkx as nx
        from community import community_louvain
    except:
        print("Для этой гипотезы необходимы библиотеки networkx и python-louvain")
        print("Установите их: pip install networkx python-louvain")
        return None
    
    # 1. Строим граф связей через общие устройства и IP
    print("Построение графа связей...")
    
    # Создаем граф
    G = nx.Graph()
    
    # Добавляем узлы (клиенты)
    customers = transactions['customer_id'].unique()
    for customer in customers:
        G.add_node(customer, type='customer')
    
    # Добавляем узлы устройств и связи
    if 'device_fingerprint' in transactions.columns and 'ip_address' in transactions.columns:
        # Группируем по устройству
        device_groups = transactions.groupby('device_fingerprint')['customer_id'].unique()
        for device, users in device_groups.items():
            if len(users) > 1:  # Если устройство используют несколько клиентов
                users = list(users)
                for i in range(len(users)):
                    for j in range(i+1, len(users)):
                        if G.has_edge(users[i], users[j]):
                            G[users[i]][users[j]]['weight'] += 1
                            G[users[i]][users[j]]['device_shared'] = True
                        else:
                            G.add_edge(users[i], users[j], weight=1, device_shared=True)
        
        # Группируем по IP
        ip_groups = transactions.groupby('ip_address')['customer_id'].unique()
        for ip, users in ip_groups.items():
            if len(users) > 1:  # Если IP используют несколько клиентов
                users = list(users)
                for i in range(len(users)):
                    for j in range(i+1, len(users)):
                        if G.has_edge(users[i], users[j]):
                            G[users[i]][users[j]]['weight'] += 1
                            G[users[i]][users[j]]['ip_shared'] = True
                        else:
                            G.add_edge(users[i], users[j], weight=1, ip_shared=True)
    
    print(f"Граф построен: {G.number_of_nodes()} узлов, {G.number_of_edges()} ребер")
    
    # 2. Анализ компонент связности
    connected_components = list(nx.connected_components(G))
    print(f"Количество компонент связности: {len(connected_components)}")
    
    # Размеры компонент
    component_sizes = [len(c) for c in connected_components]
    print(f"Размеры компонент: мин={min(component_sizes)}, макс={max(component_sizes)}, среднее={np.mean(component_sizes):.2f}")
    
    # 3. Выявление сообществ (community detection)
    if G.number_of_edges() > 0:
        partition = community_louvain.best_partition(G)
        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node)
        
        print(f"Найдено сообществ: {len(communities)}")
        
        # 4. Анализ риска по сообществам
        # Собираем статистику по мошенничеству для каждого сообщества
        community_fraud_stats = {}
        
        for comm_id, nodes in communities.items():
            # Транзакции клиентов из этого сообщества
            comm_transactions = transactions[transactions['customer_id'].isin(nodes)]
            
            if len(comm_transactions) > 0:
                fraud_rate = comm_transactions['is_fraud'].mean()
                fraud_count = comm_transactions['is_fraud'].sum()
                total_transactions = len(comm_transactions)
                
                community_fraud_stats[comm_id] = {
                    'node_count': len(nodes),
                    'fraud_rate': fraud_rate,
                    'fraud_count': fraud_count,
                    'total_transactions': total_transactions
                }
        
        # Сортируем сообщества по уровню мошенничества
        sorted_communities = sorted(
            community_fraud_stats.items(),
            key=lambda x: x[1]['fraud_rate'],
            reverse=True
        )
        
        print(f"\nТоп-5 самых рискованных сообществ:")
        for i, (comm_id, stats) in enumerate(sorted_communities[:5]):
            print(f"  Сообщество {comm_id}: {stats['node_count']} клиентов, "
                  f"фрод={stats['fraud_count']}/{stats['total_transactions']} "
                  f"({stats['fraud_rate']*100:.1f}%)")
        
        # 5. Проверка гипотезы: клиенты в одном сообществе с мошенниками имеют повышенный риск
        # Находим сообщества с мошенниками
        fraud_communities = set()
        for comm_id, stats in community_fraud_stats.items():
            if stats['fraud_count'] > 0:
                fraud_communities.add(comm_id)
        
        # Собираем статистику
        fraud_neighbors_stats = []
        regular_stats = []
        
        for node in G.nodes():
            # Получаем сообщество узла
            node_community = partition.get(node, -1)
            
            # Находим транзакции этого клиента
            node_transactions = transactions[transactions['customer_id'] == node]
            
            if len(node_transactions) > 0:
                node_fraud_rate = node_transactions['is_fraud'].mean()
                
                if node_community in fraud_communities:
                    fraud_neighbors_stats.append(node_fraud_rate)
                else:
                    regular_stats.append(node_fraud_rate)
        
        # Сравниваем средние значения
        if len(fraud_neighbors_stats) > 0 and len(regular_stats) > 0:
            mean_fraud_neighbors = np.mean(fraud_neighbors_stats)
            mean_regular = np.mean(regular_stats)
            
            print(f"\nСравнение уровня мошенничества:")
            print(f"  Клиенты в сообществах с мошенниками: {mean_fraud_neighbors*100:.2f}%")
            print(f"  Остальные клиенты: {mean_regular*100:.2f}%")
            print(f"  Отношение: {mean_fraud_neighbors/mean_regular if mean_regular > 0 else 'inf':.2f}x")
            
            # Статистический тест
            from scipy.stats import ttest_ind
            t_stat, p_value = ttest_ind(fraud_neighbors_stats, regular_stats, equal_var=False)
            print(f"  t-тест: t={t_stat:.3f}, p={p_value:.5f}")
            
            # Проверяем гипотезу (10x повышение риска)
            hypothesis_met = mean_fraud_neighbors >= mean_regular * 10
            print(f"\nГипотеза достигнута (10x повышение риска)? {hypothesis_met}")
        else:
            print("Недостаточно данных для сравнения")
    
    # 6. Центральность узлов
    if G.number_of_edges() > 0:
        # Вычисляем центральности
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G, k=min(100, len(G.nodes())))
        
        # Сопоставляем с мошенничеством
        centrality_fraud = []
        for node in G.nodes():
            if node in degree_centrality:
                # Находим транзакции этого клиента
                node_transactions = transactions[transactions['customer_id'] == node]
                if len(node_transactions) > 0:
                    fraud_rate = node_transactions['is_fraud'].mean()
                    centrality_fraud.append({
                        'node': node,
                        'degree': degree_centrality[node],
                        'betweenness': betweenness_centrality.get(node, 0),
                        'fraud_rate': fraud_rate
                    })
        
        centrality_df = pd.DataFrame(centrality_fraud)
        
        # Корреляция
        if len(centrality_df) > 0:
            corr_degree = centrality_df['degree'].corr(centrality_df['fraud_rate'])
            corr_betweenness = centrality_df['betweenness'].corr(centrality_df['fraud_rate'])
            
            print(f"\nКорреляция центральности с мошенничеством:")
            print(f"  Degree centrality: {corr_degree:.3f}")
            print(f"  Betweenness centrality: {corr_betweenness:.3f}")
    
    # Визуализация
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Распределение размеров компонент
    if component_sizes:
        axes[0, 0].hist(component_sizes, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Распределение размеров компонент связности')
        axes[0, 0].set_xlabel('Размер компоненты')
        axes[0, 0].set_ylabel('Количество')
    
    # 2. Распределение уровней мошенничества по сообществам
    if 'community_fraud_stats' in locals():
        fraud_rates = [stats['fraud_rate'] for stats in community_fraud_stats.values()]
        axes[0, 1].hist(fraud_rates, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Распределение уровня мошенничества по сообществам')
        axes[0, 1].set_xlabel('Уровень мошенничества')
        axes[0, 1].set_ylabel('Количество сообществ')
    
    # 3. Сравнение клиентов в рискованных и обычных сообществах
    if 'fraud_neighbors_stats' in locals() and 'regular_stats' in locals():
        data_to_plot = [fraud_neighbors_stats, regular_stats]
        axes[0, 2].boxplot(data_to_plot, labels=['В сообществах\nс мошенниками', 'Остальные'])
        axes[0, 2].set_title('Сравнение уровня мошенничества')
        axes[0, 2].set_ylabel('Уровень мошенничества')
    
    # 4. Граф (упрощенная визуализация - только крупные компоненты)
    if G.number_of_nodes() > 0:
        # Берем самую большую компоненту
        largest_component = max(connected_components, key=len)
        subgraph = G.subgraph(list(largest_component)[:50])  # Ограничиваем 50 узлами для читаемости
        
        # Раскрашиваем узлы по мошенничеству
        node_colors = []
        for node in subgraph.nodes():
            node_transactions = transactions[transactions['customer_id'] == node]
            if len(node_transactions) > 0:
                fraud_rate = node_transactions['is_fraud'].mean()
                node_colors.append(fraud_rate)
            else:
                node_colors.append(0)
        
        # Рисуем граф
        pos = nx.spring_layout(subgraph, seed=42)
        nodes = nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, 
                                      cmap='Reds', ax=axes[1, 0], node_size=100)
        nx.draw_networkx_edges(subgraph, pos, ax=axes[1, 0], alpha=0.3)
        axes[1, 0].set_title('Пример графа связей (краснее = выше риск)')
        axes[1, 0].axis('off')
        
        # Цветовая шкала
        plt.colorbar(nodes, ax=axes[1, 0])
    
    # 5. Корреляция центральности с мошенничеством
    if 'centrality_df' in locals() and len(centrality_df) > 0:
        axes[1, 1].scatter(centrality_df['degree'], centrality_df['fraud_rate'], alpha=0.5)
        axes[1, 1].set_title('Зависимость мошенничества от степени центральности')
        axes[1, 1].set_xlabel('Degree Centrality')
        axes[1, 1].set_ylabel('Уровень мошенничества')
        
        # Линия тренда
        if len(centrality_df) > 1:
            z = np.polyfit(centrality_df['degree'], centrality_df['fraud_rate'], 1)
            p = np.poly1d(z)
            axes[1, 1].plot(centrality_df['degree'], p(centrality_df['degree']), "r--", alpha=0.8)
    
    # 6. Связь количества связей с мошенничеством
    if 'centrality_df' in locals() and len(centrality_df) > 0:
        # Подсчитываем количество связей (степень узла)
        degrees = [G.degree(node) for node in centrality_df['node']]
        axes[1, 2].scatter(degrees, centrality_df['fraud_rate'], alpha=0.5)
        axes[1, 2].set_title('Зависимость мошенничества от количества связей')
        axes[1, 2].set_xlabel('Количество связей')
        axes[1, 2].set_ylabel('Уровень мошенничества')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'graph': G,
        'communities': communities if 'communities' in locals() else None,
        'hypothesis_met': hypothesis_met if 'hypothesis_met' in locals() else False
    }

def hypothesis_3_behavioral_biometrics(transactions):
    """Гипотеза 3: Behavioral biometrics для обнаружения ботов"""
    print("\n" + "=" * 80)
    print("ГИПОТЕЗА 3: BEHAVIORAL BIOMETRICS (АНАЛИЗ ПОВЕДЕНИЯ)")
    print("=" * 80)
    
    # Поскольку у нас нет данных о времени между действиями в UI,
    # будем анализировать временные паттерны транзакций
    
    # 1. Анализ скорости транзакций
    transactions_sorted = transactions.sort_values(['customer_id', 'timestamp'])
    
    # Вычисляем время между транзакциями одного клиента
    transactions_sorted['time_diff'] = transactions_sorted.groupby('customer_id')['timestamp'].diff()
    transactions_sorted['time_diff_seconds'] = transactions_sorted['time_diff'].dt.total_seconds()
    
    # Фильтруем разумные значения (от 1 секунды до 1 дня)
    valid_time_diffs = transactions_sorted[
        (transactions_sorted['time_diff_seconds'] >= 1) & 
        (transactions_sorted['time_diff_seconds'] <= 86400)
    ]
    
    # 2. Анализ паттернов времени
    # Группируем по клиентам для анализа
    customer_time_patterns = valid_time_diffs.groupby('customer_id').agg({
        'time_diff_seconds': ['mean', 'std', 'min', 'max', 'count'],
        'is_fraud': 'mean'
    }).reset_index()
    
    customer_time_patterns.columns = [
        'customer_id', 'avg_time_diff', 'std_time_diff', 'min_time_diff',
        'max_time_diff', 'transaction_count', 'fraud_rate'
    ]
    
    # 3. Поиск аномальных паттернов (бот-подобное поведение)
    # Боты часто имеют очень регулярные интервалы (низкое std)
    # или очень высокую скорость транзакций (низкое avg_time_diff)
    
    # Определяем подозрительные паттерны
    customer_time_patterns['low_variability'] = (
        customer_time_patterns['std_time_diff'] < customer_time_patterns['avg_time_diff'] * 0.1
    ) & (customer_time_patterns['transaction_count'] > 3)
    
    customer_time_patterns['high_frequency'] = (
        customer_time_patterns['avg_time_diff'] < 60  # Менее 60 секунд между транзакциями
    ) & (customer_time_patterns['transaction_count'] > 5)
    
    customer_time_patterns['bot_like'] = (
        customer_time_patterns['low_variability'] | customer_time_patterns['high_frequency']
    )
    
    # 4. Анализ корреляции с мошенничеством
    bot_customers = customer_time_patterns[customer_time_patterns['bot_like'] == True]
    human_customers = customer_time_patterns[customer_time_patterns['bot_like'] == False]
    
    print(f"Всего клиентов: {len(customer_time_patterns)}")
    print(f"Клиентов с бот-подобным поведением: {len(bot_customers)} ({len(bot_customers)/len(customer_time_patterns)*100:.1f}%)")
    
    if len(bot_customers) > 0 and len(human_customers) > 0:
        bot_fraud_rate = bot_customers['fraud_rate'].mean()
        human_fraud_rate = human_customers['fraud_rate'].mean()
        
        print(f"\nУровень мошенничества:")
        print(f"  Бот-подобные клиенты: {bot_fraud_rate*100:.2f}%")
        print(f"  Человек-подобные клиенты: {human_fraud_rate*100:.2f}%")
        print(f"  Отношение: {bot_fraud_rate/human_fraud_rate if human_fraud_rate > 0 else 'inf':.2f}x")
        
        # Статистический тест
        from scipy.stats import mannwhitneyu
        stat, p_value = mannwhitneyu(
            bot_customers['fraud_rate'].dropna(),
            human_customers['fraud_rate'].dropna()
        )
        print(f"  Mann-Whitney U test: U={stat:.0f}, p={p_value:.5f}")
    
    # 5. Детектор аномалий на основе временных паттернов
    # Используем Isolation Forest для обнаружения аномальных клиентов
    from sklearn.ensemble import IsolationForest
    
    # Подготавливаем признаки
    time_features = customer_time_patterns[['avg_time_diff', 'std_time_diff', 'transaction_count']].copy()
    time_features['log_avg_time'] = np.log1p(time_features['avg_time_diff'])
    time_features['log_std_time'] = np.log1p(time_features['std_time_diff'])
    time_features['log_count'] = np.log1p(time_features['transaction_count'])
    
    # Заполняем пропуски
    time_features = time_features.fillna(time_features.median())
    
    # Обучение Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    time_features_scaled = StandardScaler().fit_transform(time_features)
    anomalies = iso_forest.fit_predict(time_features_scaled)
    
    customer_time_patterns['time_anomaly'] = anomalies == -1
    
    # Анализ аномалий
    time_anomalies = customer_time_patterns[customer_time_patterns['time_anomaly'] == True]
    normal_customers = customer_time_patterns[customer_time_patterns['time_anomaly'] == False]
    
    if len(time_anomalies) > 0 and len(normal_customers) > 0:
        anomaly_fraud_rate = time_anomalies['fraud_rate'].mean()
        normal_fraud_rate = normal_customers['fraud_rate'].mean()
        
        print(f"\nИзоляционный лес для временных паттернов:")
        print(f"  Обнаружено аномалий: {len(time_anomalies)} ({len(time_anomalies)/len(customer_time_patterns)*100:.1f}%)")
        print(f"  Уровень мошенничества у аномалий: {anomaly_fraud_rate*100:.2f}%")
        print(f"  Уровень мошенничества у нормальных: {normal_fraud_rate*100:.2f}%")
    
    # 6. Проверка гипотезы: точность 95% при обнаружении ботов
    # Обучаем модель классификации на временных паттернах
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    
    # Целевая переменная - мошенничество на уровне клиента
    X = time_features_scaled
    y = (customer_time_patterns['fraud_rate'] > 0).astype(int)  # Бинарная: есть ли фрод у клиента
    
    # Удаляем клиентов без достаточного количества транзакций
    mask = customer_time_patterns['transaction_count'] >= 3
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    if len(X_filtered) > 10:
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        cv_scores = cross_val_score(model, X_filtered, y_filtered, cv=5, scoring='accuracy')
        
        avg_accuracy = cv_scores.mean()
        std_accuracy = cv_scores.std()
        
        print(f"\nМодель на временных паттернах (кросс-валидация):")
        print(f"  Средняя accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        
        # Проверяем гипотезу (accuracy > 0.95)
        hypothesis_met = avg_accuracy > 0.95
        print(f"  Гипотеза достигнута (accuracy > 0.95)? {hypothesis_met}")
    else:
        print("Недостаточно данных для обучения модели")
        hypothesis_met = False
    
    # Визуализация
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Распределение среднего времени между транзакциями
    axes[0, 0].hist(customer_time_patterns['avg_time_diff'].dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Распределение среднего времени между транзакциями')
    axes[0, 0].set_xlabel('Среднее время (секунды)')
    axes[0, 0].set_ylabel('Количество клиентов')
    axes[0, 0].set_xscale('log')
    
    # 2. Распределение стандартного отклонения времени
    axes[0, 1].hist(customer_time_patterns['std_time_diff'].dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Распределение std времени между транзакциями')
    axes[0, 1].set_xlabel('Std времени (секунды)')
    axes[0, 1].set_ylabel('Количество клиентов')
    axes[0, 1].set_xscale('log')
    
    # 3. Сравнение бот-подобных и человеческих паттернов
    if len(bot_customers) > 0 and len(human_customers) > 0:
        box_data = [
            bot_customers['avg_time_diff'].dropna(),
            human_customers['avg_time_diff'].dropna()
        ]
        axes[0, 2].boxplot(box_data, labels=['Бот-подобные', 'Человек-подобные'])
        axes[0, 2].set_title('Сравнение времени между транзакциями')
        axes[0, 2].set_ylabel('Среднее время (секунды)')
        axes[0, 2].set_yscale('log')
    
    # 4. Взаимосвязь времени и мошенничества
    scatter_data = customer_time_patterns.dropna()
    if len(scatter_data) > 0:
        scatter = axes[1, 0].scatter(
            np.log1p(scatter_data['avg_time_diff']),
            np.log1p(scatter_data['std_time_diff']),
            c=scatter_data['fraud_rate'],
            cmap='Reds',
            alpha=0.6,
            s=scatter_data['transaction_count']/10
        )
        axes[1, 0].set_title('Взаимосвязь временных паттернов и мошенничества')
        axes[1, 0].set_xlabel('log(Среднее время)')
        axes[1, 0].set_ylabel('log(Std времени)')
        plt.colorbar(scatter, ax=axes[1, 0])
    
    # 5. PCA визуализация временных паттернов
    from sklearn.decomposition import PCA
    if len(time_features_scaled) > 10:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(time_features_scaled)
        
        scatter = axes[1, 1].scatter(
            pca_result[:, 0],
            pca_result[:, 1],
            c=customer_time_patterns['fraud_rate'],
            cmap='Reds',
            alpha=0.6
        )
        axes[1, 1].set_title('PCA временных паттернов (краснее = выше риск)')
        axes[1, 1].set_xlabel('PCA Component 1')
        axes[1, 1].set_ylabel('PCA Component 2')
        plt.colorbar(scatter, ax=axes[1, 1])
    
    # 6. Распределение мошенничества по сегментам
    if len(customer_time_patterns) > 0:
        # Создаем сегменты по времени и вариативности
        customer_time_patterns['time_segment'] = pd.qcut(
            customer_time_patterns['avg_time_diff'], 
            q=4, 
            labels=['Очень быстро', 'Быстро', 'Медленно', 'Очень медленно']
        )
        
        segment_stats = customer_time_patterns.groupby('time_segment')['fraud_rate'].agg(['mean', 'count']).reset_index()
        
        axes[1, 2].bar(segment_stats['time_segment'], segment_stats['mean'] * 100)
        axes[1, 2].set_title('Уровень мошенничества по скоростным сегментам')
        axes[1, 2].set_xlabel('Скоростной сегмент')
        axes[1, 2].set_ylabel('Процент мошенничества (%)')
        axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'time_patterns': customer_time_patterns,
        'bot_customers': bot_customers if 'bot_customers' in locals() else None,
        'model_accuracy': avg_accuracy if 'avg_accuracy' in locals() else None,
        'hypothesis_met': hypothesis_met
    }