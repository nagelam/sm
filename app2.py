import streamlit as st
import pandas as pd
import plotly.express as px
import catboost as cb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Заголовок (замените на ваши данные)
FIO = "Нагель Аркадий Михайлович"
GROUP = "2023-ФГИиИБ-ПИ-1б"
VARIANT = "16 вариант"
DATASET_NAME = "LoL"
st.set_page_config(page_title=f"{FIO}_{GROUP}_{VARIANT}_{DATASET_NAME}")

# Загрузка данных
@st.cache_data
def load_data():
    df = pd.read_csv("./temp.csv")
    return df

df = load_data()

# Подготовка данных для модели (пример: выбор ключевых признаков)
features = ['blueKills', 'blueDeath', 'blueTotalGold', 'redKills', 'redDeath', 'redTotalGold',
            'blueTowerKills', 'redTowerKills', 'blueDragonKills', 'redDragonKills']
target = 'blueWins'
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели CatBoost
@st.cache_resource
def train_model():
    model = cb.CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, verbose=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return model, accuracy, shap_values, X_test

model, accuracy, shap_values, X_test = train_model()

# Многостраничная навигация
def page_home():
    st.title("Главная")
    df = load_data('./temp.csv')
    st.header("Обзор набора данных")
    st.markdown(f"**Размер:** {df.shape[0]} строк × {df.shape[1]} столбцов")
    st.dataframe(df.head(), use_container_width=True)
    
    st.write(f"В данной работе я работал с датасетом по игре LOL")
    st.write("""
    В моем датасете следующие важные признаки:
    
    gameDuration — продолжительность игры в секундах.
    
    Целевая переменная: blueWins
    
    Особые первые достижения (флаги):
    
    blueFirstBlood, redFirstBlood — первая нанесенная кровь.
    
    blueFirstTower, redFirstTower — первая разрушенная башня.
    
    blueFirstBaron, redFirstBaron — первый убитый барон.
    
    blueFirstDragon, redFirstDragon — первый убитый дракон.
    
    blueFirstInhibitor, redFirstInhibitor — первый уничтоженный ингибитор (строение, которое блокирует появление вражеских суперминьонов на этой линии).
    
    Количественные показатели команд:
    
    DragonKills — количество убитых драконов.
    
    BaronKills — количество убитых баронов.
    
    TowerKills — количество уничтоженных башен.
    
    InhibitorKills — количество уничтоженных ингибиторов.
    
    WardPlaced — количество поставленных вардингов (тотемов).
    
    Wardkills — количество уничтоженных вардингов соперника.
    
    kills — количество убийств вражеских чемпионов.
    
    Death — количество смертей.
    
    Assist — количество совместных убийств.
    
    ChampionDamageDealt — нанесённый урон чемпионам противника.
    
    TotalGold — общее количество заработанного золота.
    
    TotalMinionKills — количество убитых миньонов.
    
    TotalLevel — суммарный уровень всех чемпионов команды.
    
    AvgLevel — средний уровень чемпионов команды.
    
    JungleMinionKills — количество убитых миньонов в джунглях.
    
    KillingSpree — максимальная серия убийств без смерти.
    
    TotalHeal — общее количество исцеления.
    
    ObjectDamageDealt — урон, нанесённый объектам (например, башням).
    """)
    

def page_features_deps():
    st.title("Зависимости Признаков")
    col1, col2 = st.columns(2)
    
    with col1:
        # Интерактивный scatter: зависимость двух признаков
        feat1 = st.selectbox("Признак X:", features, index=0)
        feat2 = st.selectbox("Признак Y:", features, index=2)
        fig_dep = px.scatter(df, x=feat1, y=feat2, color='blueWins', title=f"{feat1} vs {feat2}")
        fig_dep.update_layout(height=300, width=400)
        st.plotly_chart(fig_dep, use_container_width=True)
        st.write("Зависимости двух выбранных признаков между собой")
    
    with col2:
        # Зависимость признака от таргета
        feat_target = st.selectbox("Признак vs blueWins:", features, index=1)
        fig_target = px.histogram(df, x=feat_target, y='blueWins', title=f"{feat_target} по blueWins")
        fig_target.update_layout(height=300, width=400)
        st.plotly_chart(fig_target, use_container_width=True)
        st.write("Зависимости выбранного признака и blueWins")

def page_distributions():
    st.title("Распределения Признаков")
    col1, col2 = st.columns(2)
    
    with col1:
        # Интерактивная гистограмма распределения
        feat_hist = st.selectbox("Признак для распределения:", features, index=3)
        fig_hist = px.histogram(df, x=feat_hist, color='blueWins', title=f"Распределение {feat_hist}", marginal="box")
        fig_hist.update_layout(height=300, width=400)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Еще одна гистограмма для другого признака
        feat_hist2 = st.selectbox("Второй признак для распределения:", features, index=4)
        fig_hist2 = px.histogram(df, x=feat_hist2, color='blueWins', title=f"Распределение {feat_hist2}", marginal="box")
        fig_hist2.update_layout(height=300, width=400)
        st.plotly_chart(fig_hist2, use_container_width=True)

def page_model_interpret():
    st.write(f"Точность модели CatBoost: {accuracy:.2%} (на тестовых данных).")
    st.title("Интерпретация Модели")
    # График SHAP для интерпретации результатов обучения
    st.write("SHAP-значения для тестовых данных (влияние признаков на предсказания).")
    
    # Исправление: создаем фигуру явно
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Бар важности признаков (без изменений)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
    plt.tight_layout()
    st.pyplot(fig2)

# Навигация
pages = {
    "Главная": page_home,
    "Зависимости": page_features_deps,
    "Распределения": page_distributions,
    "Модель": page_model_interpret
}

st.sidebar.title("Навигация")
selection = st.sidebar.radio("Страницы:", list(pages.keys()))
page = pages[selection]
page()
