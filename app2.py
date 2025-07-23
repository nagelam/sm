import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Настройка страницы
st.set_page_config(
    page_title="Иванов_ПИ19-1_Вариант1_League_of_Legends",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок
st.title("🎮 Иванов_ПИ19-1_Вариант1_League_of_Legends")
st.markdown("### Интерактивный анализ данных League of Legends с CatBoost")

# Загрузка данных
@st.cache_data
def load_data():
    """Загрузка и предобработка данных"""
    # В реальном проекте здесь будет загрузка вашего CSV файла
    np.random.seed(42)
    n_samples = 1000
    
    # Генерируем реалистичные данные на основе предоставленного образца
    game_duration = np.random.normal(1500, 400, n_samples).astype(int)
    game_duration = np.clip(game_duration, 300, 3000)
    
    blue_wins = np.random.binomial(1, 0.5, n_samples)
    blue_kills = np.random.poisson(20 + 10 * blue_wins, n_samples)
    blue_deaths = np.random.poisson(20 + 5 * (1 - blue_wins), n_samples)
    blue_assists = np.random.poisson(30 + 15 * blue_wins, n_samples)
    blue_total_gold = np.random.normal(50000 + game_duration * 20 + blue_wins * 10000, 15000, n_samples).astype(int)
    blue_total_gold = np.clip(blue_total_gold, 10000, 150000)
    blue_champion_damage = np.random.normal(blue_total_gold * 1.2 + blue_wins * 20000, 20000, n_samples).astype(int)
    blue_champion_damage = np.clip(blue_champion_damage, 10000, 200000)
    
    red_wins = 1 - blue_wins
    red_kills = np.random.poisson(20 + 10 * red_wins, n_samples)
    red_deaths = np.random.poisson(20 + 5 * (1 - red_wins), n_samples)
    red_assists = np.random.poisson(30 + 15 * red_wins, n_samples)
    red_total_gold = np.random.normal(50000 + game_duration * 20 + red_wins * 10000, 15000, n_samples).astype(int)
    red_total_gold = np.clip(red_total_gold, 10000, 150000)
    red_champion_damage = np.random.normal(red_total_gold * 1.2 + red_wins * 20000, 20000, n_samples).astype(int)
    red_champion_damage = np.clip(red_champion_damage, 10000, 200000)
    
    data = {
        'gameId': np.arange(1000000, 1000000 + n_samples),
        'gameDuration': game_duration,
        'blueWins': blue_wins,
        'blueKills': blue_kills,
        'blueDeaths': blue_deaths,
        'blueAssists': blue_assists,
        'blueTotalGold': blue_total_gold,
        'blueChampionDamageDealt': blue_champion_damage,
        'redKills': red_kills,
        'redDeaths': red_deaths,
        'redAssists': red_assists,
        'redTotalGold': red_total_gold,
        'redChampionDamageDealt': red_champion_damage
    }
    
    return pd.DataFrame(data)

# Обучение модели CatBoost
@st.cache_resource
def train_catboost_model(df):
    """Обучение модели CatBoost для предсказания исходов матчей"""
    # Подготовка признаков
    features = ['gameDuration', 'blueKills', 'blueDeaths', 'blueAssists', 
               'blueTotalGold', 'blueChampionDamageDealt', 'redKills', 
               'redDeaths', 'redAssists', 'redTotalGold', 'redChampionDamageDealt']
    
    X = df[features]
    y = df['blueWins']
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Обучение модели
    model = CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        loss_function='Logloss',
        verbose=False,
        random_seed=42
    )
    
    model.fit(X_train, y_train)
    
    # Предсказания и метрики
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, X_test, y_test, y_pred, features

# Загрузка данных и обучение модели
df = load_data()
model, accuracy, X_test, y_test, y_pred, features = train_catboost_model(df)

# Краткое описание набора данных
st.markdown("""
**Описание набора данных:**
Данные содержат информацию о 1000 матчах League of Legends с детальной статистикой для синей и красной команд. 
Включают ключевые игровые метрики: убийства, смерти, помощи, золото, урон и продолжительность игры, 
которые используются для предсказания победителя матча.
""")

# Боковая панель с фильтрами
st.sidebar.header("🎛️ Интерактивные фильтры")

# Фильтр по продолжительности игры
duration_range = st.sidebar.slider(
    "Продолжительность игры (минуты)",
    min_value=int(df['gameDuration'].min()),
    max_value=int(df['gameDuration'].max()),
    value=(int(df['gameDuration'].min()), int(df['gameDuration'].max())),
    step=50
)

# Фильтр по команде-победителю
team_filter = st.sidebar.multiselect(
    "Команда-победитель",
    options=['Синяя команда', 'Красная команда'],
    default=['Синяя команда', 'Красная команда']
)

# Применение фильтров
filtered_df = df[
    (df['gameDuration'] >= duration_range[0]) & 
    (df['gameDuration'] <= duration_range[1])
]

if 'Синяя команда' not in team_filter:
    filtered_df = filtered_df[filtered_df['blueWins'] == 0]
elif 'Красная команда' not in team_filter:
    filtered_df = filtered_df[filtered_df['blueWins'] == 1]

# Основной контент
col1, col2 = st.columns([2, 1])

with col2:
    # Метрики модели
    st.subheader("📊 Точность модели CatBoost")
    st.metric("Точность", f"{accuracy:.3f}", f"{(accuracy-0.5)*100:+.1f}%")
    
    # Баланс классов
    st.subheader("⚖️ Баланс классов")
    class_counts = filtered_df['blueWins'].value_counts()
    fig_pie = px.pie(
        values=class_counts.values,
        names=['Красная команда', 'Синяя команда'],
        title="Распределение побед",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_pie.update_layout(height=300, showlegend=True)
    st.plotly_chart(fig_pie, use_container_width=True)

with col1:
    # График 1: Зависимость признаков между собой
    st.subheader("🔗 Зависимость признаков")
    
    # Выбор признаков для сравнения
    feature_options = ['blueKills', 'blueDeaths', 'blueAssists', 'blueTotalGold', 
                      'blueChampionDamageDealt', 'redKills', 'redDeaths', 'redAssists', 
                      'redTotalGold', 'redChampionDamageDealt', 'gameDuration']
    
    x_feature = st.selectbox("Признак по оси X:", feature_options, index=0)
    y_feature = st.selectbox("Признак по оси Y:", feature_options, index=4)
    
    fig_scatter = px.scatter(
        filtered_df,
        x=x_feature,
        y=y_feature,
        color='blueWins',
        color_discrete_map={0: 'red', 1: 'blue'},
        title=f"Зависимость {y_feature} от {x_feature}",
        labels={'blueWins': 'Победитель', 0: 'Красная команда', 1: 'Синяя команда'},
        hover_data=['gameDuration']
    )
    fig_scatter.update_layout(height=400)
    st.plotly_chart(fig_scatter, use_container_width=True)

# Вторая строка графиков
col3, col4 = st.columns(2)

with col3:
    # График 2: Зависимость признаков от таргета
    st.subheader("🎯 Анализ признаков по командам")
    
    selected_feature = st.selectbox(
        "Выберите признак для анализа:",
        feature_options,
        index=3
    )
    
    fig_box = px.box(
        filtered_df,
        x='blueWins',
        y=selected_feature,
        color='blueWins',
        color_discrete_map={0: 'red', 1: 'blue'},
        title=f"Распределение {selected_feature} по командам",
        labels={'blueWins': 'Команда-победитель'}
    )
    fig_box.update_xaxes(ticktext=['Красная команда', 'Синяя команда'], tickvals=[0, 1])
    fig_box.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

with col4:
    # График 3: Результаты модели - важность признаков
    st.subheader("🤖 Важность признаков (CatBoost)")
    
    feature_importance = model.get_feature_importance()
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=True)
    
    fig_importance = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Важность признаков в модели",
        color='Importance',
        color_continuous_scale='viridis'
    )
    fig_importance.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_importance, use_container_width=True)

# Дополнительные интерактивные графики
st.subheader("📈 Дополнительный анализ")

col5, col6 = st.columns(2)

with col5:
    # Корреляционная матрица
    st.subheader("🔥 Тепловая карта корреляций")
    
    numeric_features = ['gameDuration', 'blueKills', 'blueDeaths', 'blueAssists', 
                       'blueTotalGold', 'blueChampionDamageDealt', 'blueWins']
    corr_matrix = filtered_df[numeric_features].corr()
    
    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Корреляционная матрица признаков",
        color_continuous_scale='RdBu_r'
    )
    fig_heatmap.update_layout(height=400)
    st.plotly_chart(fig_heatmap, use_container_width=True)

with col6:
    # Гистограмма продолжительности игр
    st.subheader("⏱️ Распределение времени игр")
    
    fig_hist = px.histogram(
        filtered_df,
        x='gameDuration',
        color='blueWins',
        color_discrete_map={0: 'red', 1: 'blue'},
        title="Распределение продолжительности игр",
        labels={'blueWins': 'Победитель'},
        nbins=30,
        opacity=0.7
    )
    fig_hist.update_layout(height=400, barmode='overlay')
    st.plotly_chart(fig_hist, use_container_width=True)

# Интерактивная таблица с данными
st.subheader("📋 Данные матчей")
st.dataframe(
    filtered_df.head(100),
    use_container_width=True,
    height=300
)

# Статистика
col7, col8, col9, col10 = st.columns(4)

with col7:
    st.metric(
        "Всего матчей",
        len(filtered_df),
        delta=f"{len(filtered_df) - len(df)} от общего"
    )

with col8:
    avg_duration = filtered_df['gameDuration'].mean()
    st.metric(
        "Средняя длительность",
        f"{avg_duration:.0f} мин",
        delta=f"{avg_duration - df['gameDuration'].mean():.0f} мин"
    )

with col9:
    blue_win_rate = filtered_df['blueWins'].mean()
    st.metric(
        "Процент побед синей команды",
        f"{blue_win_rate:.1%}",
        delta=f"{(blue_win_rate - 0.5)*100:+.1f}%"
    )

with col10:
    avg_kills = (filtered_df['blueKills'] + filtered_df['redKills']).mean()
    st.metric(
        "Среднее убийств за игру",
        f"{avg_kills:.1f}",
        delta=f"{avg_kills - 40:.1f}"
    )

# Подвал
st.markdown("---")
st.markdown("""
**Технологии:** Streamlit, Plotly, CatBoost, Pandas, NumPy  
**Автор:** Иванов И.И., группа ПИ19-1, вариант 1  
**Набор данных:** League of Legends Match Outcomes
""")
