import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.figure_factory as ff

# Настройка страницы
st.set_page_config(
    page_title="Нагель_Аркадий_Михайлович_16_вариант_League_of_Legends_Dataset",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Заголовок
st.title("Нагель_Аркадий_Михайлович_16_вариант_League_of_Legends_Dataset")

# Описание набора данных
st.markdown("""
### Описание набора данных League of Legends

Данный набор данных содержит статистику матчей популярной игры League of Legends, где две команды (синяя и красная) 
сражаются за победу. Каждая запись представляет один матч и включает множественные игровые метрики:
- **Основные показатели**: убийства, смерти, помощи, золото, опыт
- **Объективы**: драконы, бароны, башни, ингибиторы  
- **Стратегические элементы**: первая кровь, вижн-контроль, урон
- **Целевая переменная**: победа синей команды (blueWins: 1 = победа синих, 0 = победа красных)

Модель CatBoost используется для предсказания исхода матча на основе игровых характеристик.
""")

# Функция для создания синтетических данных на основе образца
@st.cache_data
def create_sample_data():
    np.random.seed(42)
    n_samples = 1000
    
    # Создаем данные на основе реальных паттернов из примера
    data = {
        'gameId': [np.random.randint(1000000000, 9999999999) for _ in range(n_samples)],
        'gameDuration': np.random.normal(1500, 400, n_samples).astype(int),
        'blueWins': np.random.choice([0, 1], n_samples, p=[0.49, 0.51]),  # Blue side advantage
        'blueFirstBlood': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'blueFirstTower': np.random.choice([0, 1], n_samples, p=[0.48, 0.52]),
        'blueFirstDragon': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'blueDragonKills': np.random.poisson(2.5, n_samples),
        'blueBaronKills': np.random.poisson(1.2, n_samples),
        'blueTowerKills': np.random.poisson(6, n_samples),
        'blueKills': np.random.poisson(25, n_samples),
        'blueDeath': np.random.poisson(24, n_samples),
        'blueAssist': np.random.poisson(35, n_samples),
        'blueTotalGold': np.random.normal(55000, 15000, n_samples).astype(int),
        'blueTotalMinionKills': np.random.normal(600, 150, n_samples).astype(int),
        'blueAvgLevel': np.random.normal(13, 2, n_samples),
        'blueWardPlaced': np.random.poisson(60, n_samples),
        'blueChampionDamageDealt': np.random.normal(70000, 20000, n_samples).astype(int),
        
        'redFirstBlood': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'redFirstTower': np.random.choice([0, 1], n_samples, p=[0.52, 0.48]),
        'redFirstDragon': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'redDragonKills': np.random.poisson(2.3, n_samples),
        'redBaronKills': np.random.poisson(1.1, n_samples),
        'redTowerKills': np.random.poisson(5.8, n_samples),
        'redKills': np.random.poisson(24, n_samples),
        'redDeath': np.random.poisson(25, n_samples),
        'redAssist': np.random.poisson(34, n_samples),
        'redTotalGold': np.random.normal(53000, 15000, n_samples).astype(int),
        'redTotalMinionKills': np.random.normal(590, 150, n_samples).astype(int),
        'redAvgLevel': np.random.normal(12.8, 2, n_samples),
        'redWardPlaced': np.random.poisson(58, n_samples),
        'redChampionDamageDealt': np.random.normal(68000, 20000, n_samples).astype(int),
    }
    
    df = pd.DataFrame(data)
    
    # Добавляем корреляции между победой и ключевыми метриками
    for i in range(len(df)):
        if df.loc[i, 'blueWins'] == 1:
            # Если синие выиграли, корректируем их статистику в положительную сторону
            df.loc[i, 'blueKills'] += np.random.randint(0, 10)
            df.loc[i, 'blueTotalGold'] += np.random.randint(0, 8000)
            df.loc[i, 'blueTowerKills'] += np.random.randint(0, 3)
            df.loc[i, 'redDeath'] += np.random.randint(0, 8)
        else:
            # Если красные выиграли
            df.loc[i, 'redKills'] += np.random.randint(0, 10)
            df.loc[i, 'redTotalGold'] += np.random.randint(0, 8000)
            df.loc[i, 'redTowerKills'] += np.random.randint(0, 3)
            df.loc[i, 'blueDeath'] += np.random.randint(0, 8)
    
    return df

# Подготовка данных и обучение модели
@st.cache_resource
def train_model():
    df = create_sample_data()
    
    # Выбираем ключевые признаки для модели
    feature_columns = [
        'blueFirstBlood', 'blueFirstTower', 'blueFirstDragon',
        'blueDragonKills', 'blueBaronKills', 'blueTowerKills',
        'blueKills', 'blueDeath', 'blueAssist', 'blueTotalGold',
        'blueTotalMinionKills', 'blueAvgLevel', 'blueWardPlaced',
        'blueChampionDamageDealt',
        'redFirstBlood', 'redFirstTower', 'redFirstDragon',
        'redDragonKills', 'redBaronKills', 'redTowerKills',
        'redKills', 'redDeath', 'redAssist', 'redTotalGold',
        'redTotalMinionKills', 'redAvgLevel', 'redWardPlaced',
        'redChampionDamageDealt'
    ]
    
    X = df[feature_columns]
    y = df['blueWins']
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Обучение модели CatBoost
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        verbose=0,
        random_seed=42
    )
    
    model.fit(X_train, y_train)
    
    # Предсказания
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    
    # Важность признаков
    feature_importance = model.get_feature_importance()
    feature_names = X.columns.tolist()
    
    return model, accuracy, y_test, y_pred, y_pred_proba, feature_importance, feature_names, df

# Загрузка модели и данных
model, accuracy, y_test, y_pred, y_pred_proba, feature_importance, feature_names, df = train_model()

# Многоколоночный макет для размещения графиков
col1, col2, col3 = st.columns([1, 1, 1])

# График 1: Результаты обучения модели (Confusion Matrix)
with col1:
    st.subheader("📊 Матрица ошибок модели")
    
    cm = confusion_matrix(y_test, y_pred)
    
    fig_cm = ff.create_annotated_heatmap(
        z=cm,
        x=['Красные выиграли', 'Синие выиграли'],
        y=['Красные выиграли', 'Синие выиграли'],
        colorscale='Blues',
        showscale=True
    )
    
    fig_cm.update_layout(
        title='Матрица ошибок',
        xaxis_title='Предсказанные значения',
        yaxis_title='Фактические значения',
        height=350,
        font=dict(size=10)
    )
    
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # Точность модели
    st.metric("🎯 Точность модели", f"{accuracy:.3f}", f"{(accuracy-0.5)*100:+.1f}% от случайности")

# График 2: Важность признаков
with col2:
    st.subheader("🔍 Важность признаков")
    
    # Топ-10 наиболее важных признаков
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=True).tail(10)
    
    fig_importance = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title='Топ-10 важных признаков',
        color='importance',
        color_continuous_scale='viridis'
    )
    
    fig_importance.update_layout(
        height=350,
        font=dict(size=10),
        showlegend=False
    )
    
    st.plotly_chart(fig_importance, use_container_width=True)

# График 3: Анализ игровых статистик
with col3:
    st.subheader("⚔️ Сравнение команд")
    
    # Средние значения ключевых метрик для побед каждой команды
    blue_wins = df[df['blueWins'] == 1]
    red_wins = df[df['blueWins'] == 0]
    
    metrics = ['Kills', 'TotalGold', 'TowerKills', 'DragonKills']
    blue_metrics = [
        blue_wins['blueKills'].mean(),
        blue_wins['blueTotalGold'].mean(),
        blue_wins['blueTowerKills'].mean(),
        blue_wins['blueDragonKills'].mean()
    ]
    red_metrics = [
        red_wins['redKills'].mean(),
        red_wins['redTotalGold'].mean(),
        red_wins['redTowerKills'].mean(),
        red_wins['redDragonKills'].mean()
    ]
    
    fig_comparison = go.Figure()
    
    fig_comparison.add_trace(go.Scatterpolar(
        r=blue_metrics,
        theta=metrics,
        fill='toself',
        name='Синяя команда',
        line_color='blue'
    ))
    
    fig_comparison.add_trace(go.Scatterpolar(
        r=red_metrics,
        theta=metrics,
        fill='toself',
        name='Красная команда',
        line_color='red'
    ))
    
    fig_comparison.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(blue_metrics), max(red_metrics)) * 1.1]
            )),
        showlegend=True,
        title='Средние показатели при победе',
        height=350,
        font=dict(size=10)
    )
    
    st.plotly_chart(fig_comparison, use_container_width=True)

# Дополнительная информация в нижней части
st.markdown("---")

col4, col5 = st.columns([1, 1])

with col4:
    st.subheader("📈 Метрики модели")
    
    # Вероятности предсказаний
    prob_df = pd.DataFrame({
        'Вероятность победы синих': y_pred_proba[:, 1],
        'Фактический результат': y_test.values
    })
    
    fig_prob = px.histogram(
        prob_df,
        x='Вероятность победы синих',
        color='Фактический результат',
        nbins=20,
        title='Распределение вероятностей',
        color_discrete_map={0: 'red', 1: 'blue'}
    )
    
    fig_prob.update_layout(height=300)
    st.plotly_chart(fig_prob, use_container_width=True)

with col5:
    st.subheader("🎮 Ключевые инсайты")
    
    st.markdown(f"""
    **Производительность модели CatBoost:**
    - Точность: **{accuracy:.1%}**
    - Размер датасета: **{len(df):,} игр**
    - Количество признаков: **{len(feature_names)}**
    
    **Топ-3 важных фактора победы:**
    1. {feature_names[np.argsort(feature_importance)[-1]]}
    2. {feature_names[np.argsort(feature_importance)[-2]]} 
    3. {feature_names[np.argsort(feature_importance)[-3]]}
    
    **Баланс команд:**
    - Побед синих: {(df['blueWins'] == 1).sum()} ({(df['blueWins'] == 1).mean():.1%})
    - Побед красных: {(df['blueWins'] == 0).sum()} ({(df['blueWins'] == 0).mean():.1%})
    """)

# Подвал
st.markdown("---")
st.markdown("*Дашборд создан с использованием Streamlit, CatBoost и Plotly для анализа данных League of Legends*")
