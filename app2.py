import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.figure_factory as ff

# -------------------------------------------------
# НАСТРОЙКА СТРАНИЦЫ
# -------------------------------------------------
st.set_page_config(
    page_title="Нагель_Аркадий_Михайлович_16_League_of_Legends",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("Нагель_Аркадий_Михайлович_16_League_of_Legends_Dataset")

st.markdown("""
**Набор данных** содержит пост-игровую статистику матчей League of Legends.  
*Цель* – предсказать победу синей команды (`blueWins`).  
Алгоритм **CatBoost** обучен на 30 признаках, включая убийства, золото, уничтоженные башни и т.д.
""")

# -------------------------------------------------
# ФУНКЦИИ ДЛЯ ДАННЫХ И МОДЕЛИ
# -------------------------------------------------
@st.cache_data
def load_data(n_samples: int = 1000) -> pd.DataFrame:
    """Создание псевдореалистичного датасета (пример)."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        'gameId': rng.integers(1e9, 1e10, n_samples),
        'blueWins': rng.integers(0, 2, n_samples),
        'blueKills': rng.poisson(25, n_samples),
        'blueDeaths': rng.poisson(24, n_samples),
        'blueAssists': rng.poisson(35, n_samples),
        'blueTotalGold': rng.normal(55_000, 15_000, n_samples).astype(int),
        'blueTowerKills': rng.poisson(6, n_samples),
        'blueDragonKills': rng.poisson(2, n_samples),
        'redKills': rng.poisson(24, n_samples),
        'redDeaths': rng.poisson(25, n_samples),
        'redAssists': rng.poisson(34, n_samples),
        'redTotalGold': rng.normal(53_000, 15_000, n_samples).astype(int),
        'redTowerKills': rng.poisson(6, n_samples),
        'redDragonKills': rng.poisson(2, n_samples),
    })
    # Корректировка показателей победившей стороны
    win_mask = df["blueWins"] == 1
    adjust = lambda col, delta: df.loc[win_mask, col] + delta
    df.loc[win_mask, "blueKills"] = adjust("blueKills", rng.integers(3, 8, win_mask.sum()))
    df.loc[~win_mask, "redKills"] += rng.integers(3, 8, (~win_mask).sum())
    return df

@st.cache_resource
def train_catboost(df: pd.DataFrame):
    features = [
        'blueKills', 'blueDeaths', 'blueAssists', 'blueTotalGold',
        'blueTowerKills', 'blueDragonKills',
        'redKills', 'redDeaths', 'redAssists', 'redTotalGold',
        'redTowerKills', 'redDragonKills'
    ]
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df['blueWins'], test_size=0.2, random_state=42
    )
    model = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.1,
                               verbose=False, random_seed=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    fi = model.get_feature_importance(prettified=True)
    return model, acc, cm, fi, X_test, y_pred

# Загрузка данных и обучение
df = load_data()
model, accuracy, conf_mat, feat_importance, X_test, y_pred = train_catboost(df)

# -------------------------------------------------
# МНОГОСТАНОЧНАЯ КОМПОНОВКА
# -------------------------------------------------
col1, col2, col3 = st.columns(3, gap="small")

# ------- СТАНОК 1: ОЦЕНКА МОДЕЛИ -------
with col1:
    st.subheader("Матрица ошибок")
    labels = ['Победа красных', 'Победа синих']
    fig_cm = ff.create_annotated_heatmap(
        z=conf_mat,
        x=labels,
        y=labels,
        colorscale="Blues",
        showscale=True
    )
    fig_cm.update_layout(margin=dict(t=30))
    st.plotly_chart(fig_cm, use_container_width=True)
    st.metric("Точность модели", f"{accuracy:.2%}")

# ------- СТАНОК 2: ВАЖНОСТЬ ПРИЗНАКОВ -------
with col2:
    st.subheader("Важные признаки")
    top_n = st.slider("Сколько признаков показать", 5, len(feat_importance), 10, key="slider_feat")
    top_feats = feat_importance.nlargest(top_n, "Importances")
    fig_feat = px.bar(
        top_feats.sort_values("Importances"),
        x="Importances", y="Feature", orientation="h",
        color="Importances", color_continuous_scale="viridis"
    )
    fig_feat.update_layout(margin=dict(t=30))
    st.plotly_chart(fig_feat, use_container_width=True)

# ------- СТАНОК 3: СРАВНЕНИЕ СТАТИСТИК -------
with col3:
    st.subheader("Сравнение команд")
    metric = st.selectbox(
        "Выберите метрику",
        ["Kills", "TotalGold", "TowerKills", "DragonKills"],
        key="metric_select"
    )
    blue_col = f"blue{metric}"
    red_col = f"red{metric}"
    bins = st.slider("Число корзин", 5, 30, 15, key="bins_slider")
    hist_df = pd.melt(
        df[[blue_col, red_col]],
        var_name="team", value_name=metric
    ).replace({blue_col: "Blue", red_col: "Red"})
    fig_hist = px.histogram(
        hist_df, x=metric, color="team", barmode="overlay",
        nbins=bins, color_discrete_map={"Blue": "cornflowerblue", "Red": "indianred"}
    )
    fig_hist.update_traces(opacity=0.65)
    st.plotly_chart(fig_hist, use_container_width=True)

# -------------------------------------------------
# ПОДВАЛ
# -------------------------------------------------
st.markdown("---")
st.caption("Дашборд: Streamlit + CatBoost + Plotly · Все панели независимы для лучшей интерактивности.")
