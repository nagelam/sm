import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Конфиг страницы
st.set_page_config(
    page_title="LoL Compact Analysis",
    layout="wide"
)

@st.cache_data
def load_data():
    try:
        return pd.read_csv('temp.csv')
    except Exception:
        return pd.DataFrame()

@st.cache_resource
def train_model(df):
    try:
        X = df.drop(['gameId', 'blueWins'], axis=1)
        y = df['blueWins']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = CatBoostClassifier(iterations=60, random_state=42, verbose=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)
        return acc, cm
    except Exception:
        return 0, None

# Основной интерфейс
st.title("LoL Win Prediction — Минимал")
st.caption("League of Legends: компактный анализ побед по признакам")

df = load_data()

with st.container():
    st.subheader("Распределение исхода")
    if not df.empty:
        fig, ax = plt.subplots(figsize=(2, 2))
        df['blueWins'].value_counts().plot.pie(
            autopct='%1.1f%%',
            labels=['Поражение', 'Победа'],
            colors=['#E0E0E0', '#3B8BEB'],
            ax=ax, textprops={'fontsize': 5}, pctdistance=0.8
        )
        plt.ylabel('')
        st.pyplot(fig, use_container_width=True)
    else:
        st.warning("Ошибка: данные не загружены")

with st.container():
    st.subheader("Корреляция")
    try:
        st.image('2_3_2.png', use_column_width=True)
        st.caption("Важность признаков (изображение)")
    except Exception:
        st.warning("Не удалось загрузить изображение корреляций")

acc, cm = train_model(df) if not df.empty else (0, None)

st.subheader("Оценка модели")
st.metric("Точность", f"{acc:.2%}" if acc else "-")

if cm is not None:
    fig2, ax2 = plt.subplots(figsize=(1.5, 1.2))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Поражение', 'Победа'],
        yticklabels=['Поражение', 'Победа'],
        ax=ax2, cbar=False
    )
    plt.xlabel('Предсказание', fontsize=5)
    plt.ylabel('Реальность', fontsize=5)
    st.pyplot(fig2, use_container_width=True)
    st.caption("Матрица ошибок")
else:
    st.warning("Нет данных для оценки модели")
