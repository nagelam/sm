import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

@st.cache_data
def load_data():
    return pd.read_csv('temp.csv')

@st.cache_resource
def train_model(df):
    X = df.drop(['gameId', 'blueWins'], axis=1)
    y = df['blueWins']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = CatBoostClassifier(iterations=100, random_state=42, verbose=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return model, accuracy, cm

st.set_page_config(
    page_title="Нагель.АМ-2023-ФГИиИБ-ПИ-1б_вариант16_LeagueOfLegends",
    layout="wide"
)

st.title("Анализ побед в League of Legends")
st.subheader("Нагель Аркадий ПИ-1б Вариант 16")

df = load_data()

st.write("""
**Описание набора данных:**  
Данные содержат статистику по матчам League of Legends. Целевая переменная - blueWins (победа синей команды).  
Включает 50 признаков: продолжительность матча, первые убийства, убийства драконов/баронов, золото, урон и др.
""")

model, accuracy, cm = train_model(df)

tab1, tab2 = st.tabs(["Анализ признаков", "Результаты модели"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Распределение целевой переменной")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        df['blueWins'].value_counts().plot.pie(
            autopct='%1.1f%%',
            labels=['Поражение', 'Победа'],
            colors=['#ff9999','#66b3ff'],
            ax=ax1
        )
        plt.ylabel('')
        st.pyplot(fig1)
        st.write("датасет сбалансированный по количеству побед синих и красных")
    
    with col2:
        st.header("Корреляция с победой")
        plt.title('Топ-5 признаков влияющих на победу')
        plt.xlabel('Корреляция')
        st.image('2_3_2.png', use_column_width=True)
        st.write("Тут мы видим важность признаков. Наши добавленные признаки хорошо коррелируют с таргетом")

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Матрица ошибок")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Поражение', 'Победа'],
            yticklabels=['Поражение', 'Победа'],
            ax=ax3
        )
        plt.xlabel('Предсказание')
        plt.ylabel('Реальность')
        st.pyplot(fig3)
        st.write("модель хорошо предсказывает результат")
    
    with col2:
        st.header("Оценка модели")
        st.metric("Точность модели", f"{accuracy:.2%}")
        st.write("**CatBoost Classifier**")
        st.write("Параметры: 100 итераций, тестовая выборка 20%")
        st.write("**Интерпретация результатов:**")
        st.write("- Модель хорошо предсказывает поражения (TN)")
        st.write("- Лучше предсказывает победы (TP), чем поражения")