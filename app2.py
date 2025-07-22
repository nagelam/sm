with col1:
    st.subheader("Распределение целевой переменной")
    try:
        fig1, ax1 = plt.subplots(figsize=(2, 2))
        df['blueWins'].value_counts().plot.pie(
            autopct='%1.1f%%',
            labels=['Поражение', 'Победа'],
            colors=['#ff9999', '#66b3ff'],
            ax=ax1,
            textprops={'fontsize': 5},  # Меньше шрифт
            pctdistance=0.75
        )
        plt.ylabel('')
        st.pyplot(fig1, use_container_width=True)
        st.caption("Датасет сбалансирован")
    except Exception as e:
        st.error(f"Ошибка построения графика: {str(e)}")

with col2:
    st.subheader("Корреляция с победой")
    try:
        st.image('2_3_2.png', use_column_width=True)
        st.caption("Важность признаков")
    except Exception as e:
        st.error(f"Ошибка загрузки изображения: {str(e)}")

with col1:
    st.subheader("Матрица ошибок")
    try:
        fig2, ax2 = plt.subplots(figsize=(1.75, 1.5))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Поражение', 'Победа'],
            yticklabels=['Поражение', 'Победа'],
            ax=ax2,
            cbar=False
        )
        plt.xlabel('Предсказание', fontsize=5)
        plt.ylabel('Реальность', fontsize=5)
        st.pyplot(fig2, use_container_width=True)
        st.caption("Матрица ошибок модели")
    except Exception as e:
        st.error(f"Ошибка построения матрицы: {str(e)}")
