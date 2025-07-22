import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ───────────────────────  НАСТРОЙКИ СТРАНИЦЫ  ───────────────────────
st.set_page_config(
    page_title="LoL Win Analysis — Compact",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ───────────────────────  ФУНКЦИИ  ───────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    try:
        return pd.read_csv("./temp.csv")
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
        return accuracy_score(y_te, preds), confusion_matrix(y_te, preds), model
    except Exception as e:
        st.error(f"Ошибка обучения модели: {e}")
        return 0, None, None

# ───────────────────────  ЗАГОЛОВОК  ───────────────────────
st.markdown(
    "<h2 style='text-align:center;margin-bottom:0.3em'>Анализ побед League of Legends</h2>"
    "<p style='text-align:center;font-size:0.8rem;margin-top:0'>Нагель А.М. • ПИ-1б • Вариант 16</p>",
    unsafe_allow_html=True
)

# ───────────────────────  ДАННЫЕ  ───────────────────────
df = load_data()
if df.empty:
    uploaded = st.file_uploader("Загрузите temp.csv", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        st.stop()

# ───────────────────────  МОДЕЛЬ  ───────────────────────
acc, cm, model = train(df)

# ───────────────────────  ВКЛАДКИ  ───────────────────────
tab1, tab2 = st.tabs(["📊 Анализ признаков", "🎯 Результаты модели"])

# ───────────────────────  ПЕРВАЯ ВКЛАДКА - АНАЛИЗ ПРИЗНАКОВ  ───────────────────────
with tab1:
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.markdown("##### Распределение исхода", help="Процент побед/поражений синей команды")
        fig, ax = plt.subplots(figsize=(3, 3))
        df["blueWins"].value_counts().plot.pie(
            autopct="%1.1f%%",
            labels=["Поражение", "Победа"],
            colors=["#FFD6D6", "#ADD8FF"],
            textprops={"fontsize": 8},
            wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
            ax=ax
        )
        ax.set_ylabel("")
        st.pyplot(fig, use_container_width=True)
        st.caption("Датасет сбалансирован по исходам матчей")
    
    with col2:
        st.markdown("##### Корреляции признаков")
        try:
            img_path = Path(__file__).parent / "2_3_2.png"
            if img_path.exists():
                st.image(str(img_path), use_column_width=True)
            else:
                st.image("2_3_2.png", use_column_width=True)
            st.caption("Важность признаков для предсказания победы")
        except Exception:
            st.warning("Изображение корреляций не найдено")

# ───────────────────────  ВТОРАЯ ВКЛАДКА - РЕЗУЛЬТАТЫ МОДЕЛИ  ───────────────────────
with tab2:
    st.markdown("### Результаты модели")
    
    # Блок с метриками
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.metric("Точность", f"{acc:.1%}" if acc else "–")
    
    with col2:
        if cm is not None:
            precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
            st.metric("Precision", f"{precision:.1%}")
    
    with col3:
        if cm is not None:
            recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
            st.metric("Recall", f"{recall:.1%}")
    
    st.divider()
    
    # Матрица ошибок и описание модели
    col1, col2 = st.columns([1.2, 1], gap="large")
    
    with col1:
        st.markdown("##### Матрица ошибок", help="Качество предсказаний модели")
        if cm is not None:
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=True,
                xticklabels=["Поражение", "Победа"],
                yticklabels=["Поражение", "Победа"],
                annot_kws={"fontsize": 12},
                ax=ax2
            )
            ax2.set_xlabel("Предсказание", fontsize=10)
            ax2.set_ylabel("Реальность", fontsize=10)
            st.pyplot(fig2, use_container_width=True)
            
            # Интерпретация результатов
            st.markdown("**Интерпретация:**")
            tn, fp, fn, tp = cm.ravel()
            st.write(f"• Верно предсказанные поражения: {tn}")
            st.write(f"• Верно предсказанные победы: {tp}")
            st.write(f"• Ложные срабатывания: {fp}")
            st.write(f"• Пропущенные победы: {fn}")
        else:
            st.write("Матрица ошибок недоступна")
    
    with col2:
        st.markdown("##### Описание модели")
        st.info("""
        **CatBoost Classifier**
        - Итерации: 60
        - Тестовая выборка: 20%
        - Случайное состояние: 42
        """)
        
        if model is not None:
            st.markdown("##### Размер данных")
            st.write(f"• Всего примеров: {len(df):,}")
            st.write(f"• Количество признаков: {len(df.columns) - 2}")
            st.write(f"• Соотношение классов: {df['blueWins'].value_counts()[1]}/{df['blueWins'].value_counts()[0]}")

