import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")
import os
st.write(os.listdir())

# ───────────────────────  НАСТРОЙКИ СТРАНИЦЫ  ───────────────────────
st.set_page_config(
    page_title="LoL Win Analysis — Compact 1-Screen",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ───────────────────────  ФУНКЦИИ  ───────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    try:
        return pd.read_csv("temp.csv")
    except Exception as e:
        st.error(f"Не удалось загрузить temp.csv: {e}")
        return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def train(df: pd.DataFrame):
    try:
        X = df.drop(columns=["gameId", "blueWins"])
        y = df["blueWins"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        model = CatBoostClassifier(iterations=60, random_state=42, verbose=0)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        return accuracy_score(y_te, preds), confusion_matrix(y_te, preds)
    except Exception as e:
        st.error(f"Ошибка обучения модели: {e}")
        return 0, None

# ───────────────────────  ЗАГОЛОВОК  ───────────────────────
st.markdown(
    "<h2 style='text-align:center;margin-bottom:0.3em'>Анализ побед League of Legends</h2>"
    "<p style='text-align:center;font-size:0.8rem;margin-top:0'>Нагель А.М. • ПИ-1б • Вариант 16</p>",
    unsafe_allow_html=True
)

# ───────────────────────  ДАННЫЕ  ───────────────────────
df = load_data()
if df.empty:
    st.stop()

# ───────────────────────  МОДЕЛЬ  ───────────────────────
acc, cm = train(df)

# ───────────────────────  ВИЗУАЛЬНЫЙ БЛОК (1 ЭКРАН)  ───────────────────────
col1, col2, col3 = st.columns([1.1, 0.9, 1.0], gap="small")

#––– График распределения целевой переменной –––#
with col1:
    st.markdown("##### Распределение исхода", help="Процент побед/поражений синей команды")
    fig, ax = plt.subplots(figsize=(1.9, 1.9))
    df["blueWins"].value_counts().plot.pie(
        autopct="%1.0f%%",
        labels=["Поражение", "Победа"],
        colors=["#FFD6D6", "#ADD8FF"],
        textprops={"fontsize": 6},
        wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
        ax=ax
    )
    ax.set_ylabel("")
    st.pyplot(fig, use_container_width=True)

#––– График матрицы ошибок –––#
with col2:
    st.markdown("##### Матрица ошибок", help="Качество предсказаний модели")
    if cm is not None:
        fig2, ax2 = plt.subplots(figsize=(1.9, 1.6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Пораж", "Победа"],
            yticklabels=["Пораж", "Победа"],
            annot_kws={"fontsize": 6},
            ax=ax2
        )
        ax2.set_xlabel("Предсказание", fontsize=6)
        ax2.set_ylabel("Реальность", fontsize=6)
        st.pyplot(fig2, use_container_width=True)
    else:
        st.write("–")

#––– Блок метрик и изображение корреляций –––#
with col3:
    st.markdown("##### Результаты модели")
    st.metric("Точность", f"{acc:.1%}" if acc else "–")
    st.caption("CatBoost (60 итераций, test 20 %)")
    st.divider()
    st.markdown("##### Корреляции")
    try:
        st.image("2_3_2.png", use_column_width=True)
    except Exception:
        st.write("Корреляционное изображение не найдено")

# –––––––––––––––––––––––––  ФУТЕР ––––––––––––––––––––––––– #
st.caption("© 2025 – Страница оптимизирована под одноэкранный просмотр")
