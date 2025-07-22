# Отключение всех предупреждений
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import sys

# Конфигурация страницы ДО любого другого вызова st
st.set_page_config(
    page_title="Нагель.АМ-2023-ФГИиИБ-ПИ-1б_вариант16_LeagueOfLegends",
    layout="wide"
)

# Отключение предупреждений matplotlib
plt.rcParams.update({'figure.max_open_warning': 0})

def main():
    try:
        # Проверка версий библиотек
        st.write("Версии библиотек:")
        st.write(f"- Python: {sys.version.split()[0]}")
        st.write(f"- Streamlit: {st.__version__}")
        st.write(f"- Pandas: {pd.__version__}")
        
        @st.cache_data
        def load_data():
            try:
                df = pd.read_csv('temp.csv')
                st.success("Данные успешно загружены")
                return df
            except FileNotFoundError:
                st.error("Файл temp.csv не найден")
                return pd.DataFrame()
            except Exception as e:
                st.error(f"Ошибка загрузки данных: {str(e)}")
                return pd.DataFrame()

        @st.cache_resource
        def train_model(df):
            try:
                if df.empty:
                    return None, 0, None
                    
                X = df.drop(['gameId', 'blueWins'], axis=1)
                y = df['blueWins']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = CatBoostClassifier(iterations=100, random_state=42, verbose=0)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                return model, accuracy, cm
            except Exception as e:
                st.error(f"Ошибка обучения модели: {str(e)}")
                return None, 0, None

        # Основной интерфейс
        st.title("Анализ побед в League of Legends")
        st.subheader("Нагель Аркадий ПИ-1б Вариант 16")
        
        df = load_data()
        
        if not df.empty:
            st.markdown("""
            **Описание набора данных:**  
            Данные содержат статистику по матчам League of Legends. 
            Целевая переменная - blueWins (победа синей команды).  
            Включает 50 признаков: продолжительность матча, первые убийства, 
            убийства драконов/баронов, золото, урон и др.
            """)
        
            model, accuracy, cm = train_model(df)
        
            tab1, tab2 = st.tabs(["Анализ признаков", "Результаты модели"])
        
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.header("Распределение целевой переменной")
                    try:
                        fig1, ax1 = plt.subplots(figsize=(6, 4))
                        df['blueWins'].value_counts().plot.pie(
                            autopct='%1.1f%%',
                            labels=['Поражение', 'Победа'],
                            colors=['#ff9999','#66b3ff'],
                            ax=ax1
                        )
                        plt.ylabel('')
                        st.pyplot(fig1, clear_figure=True)
                        st.write("Датасет сбалансированный по количеству побед синих и красных")
                    except Exception as e:
                        st.error(f"Ошибка построения графика: {str(e)}")
                
                with col2:
                    st.header("Корреляция с победой")
                    try:
                        st.image('2_3_2.png', use_container_width=True)  # Исправленный параметр
                        st.write("Тут мы видим важность признаков. Наши добавленные признаки хорошо коррелируют с таргетом")
                    except FileNotFoundError:
                        st.error("Файл 2_3_2.png не найден")
                    except Exception as e:
                        st.error(f"Ошибка загрузки изображения: {str(e)}")
        
            with tab2:
                if model is not None:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.header("Матрица ошибок")
                        try:
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
                            st.pyplot(fig3, clear_figure=True)
                            st.write("Модель хорошо предсказывает результат")
                        except Exception as e:
                            st.error(f"Ошибка построения матрицы: {str(e)}")
                    
                    with col2:
                        st.header("Оценка модели")
                        st.metric("Точность модели", f"{accuracy:.2%}")
                        st.write("**CatBoost Classifier**")
                        st.write("Параметры: 100 итераций, тестовая выборка 20%")
                        st.write("**Интерпретация результатов:**")
                        st.write("- Модель хорошо предсказывает поражения (TN)")
                        st.write("- Лучше предсказывает победы (TP), чем поражения")

    except Exception as e:
        st.error(f"Критическая ошибка приложения: {str(e)}")
        st.write("Пожалуйста, проверьте:")
        st.write("1. Наличие всех необходимых файлов (temp.csv, 2_3_2.png)")
        st.write("2. Установлены ли все зависимости (pip install -r requirements.txt)")
        st.write("3. Достаточно ли прав у приложения")

if __name__ == "__main__":
    main()
