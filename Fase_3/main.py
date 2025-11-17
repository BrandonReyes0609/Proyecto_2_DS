

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os


st.set_page_config(
    page_title="Proyecto 2 - Dashboard JPX",
    layout="wide",
    initial_sidebar_state="expanded"
)


TARGETS = [
    "JPX_Gold_Standard_Futures_Close",
    "JPX_Gold_Standard_Futures_High",
    "JPX_Gold_Standard_Futures_Low",
    "JPX_Gold_Standard_Futures_Open",
    "JPX_Gold_Mini_Futures_settlement_price",
    "JPX_Gold_Mini_Futures_High",
    "JPX_Gold_Mini_Futures_Low",
    "JPX_Gold_Mini_Futures_Close",
    "JPX_Gold_Mini_Futures_Open"
]

MODEL_NAMES = ["ARIMA", "Regresión Lineal", "LSTM", "GRU", "TFT"]



@st.cache_data
def load_train_test():

    df_train = None
    df_test = None

    if os.path.exists("trainlimpio.csv"):
        df_train = pd.read_csv("trainlimpio.csv")
    if os.path.exists("testlimpio.csv"):
        df_test = pd.read_csv("testlimpio.csv")

    return df_train, df_test


def preprocess_data(df, date_col=None):

    df_proc = df.copy()

    if date_col is not None and date_col in df_proc.columns:
        df_proc[date_col] = pd.to_datetime(df_proc[date_col], errors="coerce")
        df_proc = df_proc.sort_values(by=date_col)

    numeric_cols = df_proc.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        mediana = df_proc[col].median()
        df_proc[col] = df_proc[col].fillna(mediana)

    return df_proc, numeric_cols


def example_metrics_dataframe():
    base_mae = {
        "ARIMA": 130,
        "Regresión Lineal": 180,
        "LSTM": 150,
        "GRU": 140,
        "TFT": 120
    }
    base_rmse = {
        "ARIMA": 250,
        "Regresión Lineal": 320,
        "LSTM": 280,
        "GRU": 260,
        "TFT": 220
    }

    rows = []
    for i, target in enumerate(TARGETS):
        for modelo in MODEL_NAMES:
            rows.append({
                "Modelo": modelo,
                "Target": target,
                "MAE": base_mae[modelo] + i * 5,    
                "RMSE": base_rmse[modelo] + i * 8 
            })
    return pd.DataFrame(rows)

@st.cache_data
def load_feature_importance():
    if os.path.exists("feature_importance.csv"):
        df = pd.read_csv("feature_importance.csv")
        return df

    features_demo = [
        "lag_1", "lag_2", "lag_7",
        "rolling_mean_7", "rolling_std_7",
        "volume", "open_interest"
    ]
    rows = []
    for modelo in ["Regresión Lineal", "LSTM", "GRU", "TFT"]:
        for feat in features_demo:
            rows.append({
                "Modelo": modelo,
                "Target": TARGETS[0],   
                "Feature": feat,
                "Importance": np.random.rand()
            })
    return pd.DataFrame(rows)


@st.cache_data
def example_predictions_dataframe():
    rng = np.random.default_rng(42)
    n_points = 150
    rows = []

    modelos_ruido = {
        "ARIMA": 70,
        "Regresión Lineal": 80,
        "LSTM": 60,
        "GRU": 55,
        "TFT": 50
    }

    for modelo, sigma in modelos_ruido.items():
        y_real = np.linspace(1000, 2000, n_points)
        ruido = rng.normal(0, sigma, size=n_points)
        y_pred = y_real + ruido
        for i in range(n_points):
            rows.append({
                "Modelo": modelo,
                "y_real": float(y_real[i]),
                "y_pred": float(y_pred[i])
            })

    df = pd.DataFrame(rows)
    df["error"] = df["y_real"] - df["y_pred"]
    return df


st.sidebar.title("Proyecto 2 - JPX Commodities")
st.sidebar.markdown("**CC3084 – Data Science**")
st.sidebar.markdown("---")

pagina = st.sidebar.radio(
    "Navegación",
    [
        "Inicio",
        "1. Preprocesamiento de datos",
        "2. Resultados de modelos",
        "3. Series de tiempo por target",
        "4. Visualizaciones de eficiencia"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("👥 Equipo: Nancy, Brandon, Santiago, Andre")

df_train, df_test = load_train_test()

if pagina == "Inicio":
    st.title("Dashboard Proyecto 2 – JPX Gold Futures")

    st.markdown(
        """
        Esta aplicación corresponde a la **Parte 1: Diseño y Desarrollo de la Aplicación**  
        del Proyecto 2 de **Data Science**.

        ### Objetivos de esta carcasa

        - Mostrar de forma clara el flujo de:
          - Carga y preprocesamiento de datos (`trainlimpio.csv`, `testlimpio.csv`).
          - Exploración de series de tiempo por *target*.
          - Presentación de resultados de modelos.
          - Visualizaciones comparativas de eficiencia.

        - Dejar listos los espacios para la **Parte 2 (Integración de Modelos)**:
          - Conectar los modelos reales (cargar archivos `.pkl`, `.pt`, etc.).
          - Llenar las métricas con los resultados reales del proyecto.
          - Agregar gráficas específicas: **ARIMA vs GRU vs TFT**, por cada target.

        Use el menú de la izquierda para navegar entre secciones.
        """
    )

    if df_train is None or df_test is None:
        st.error(
            "No se encontraron los archivos `trainlimpio.csv` y/o `testlimpio.csv` "
            "en el directorio actual. Colóquelos junto al `app.py`."
        )
    else:
        st.success("Archivos trainlimpio.csv y testlimpio.csv detectados correctamente.")


elif pagina == "1. Preprocesamiento de datos":
    st.title("1. Preprocesamiento de datos de entrada")

    st.markdown(
        """
        En esta sección se utilizan directamente los archivos:

        - **trainlimpio.csv**: datos de entrenamiento ya limpiados.
        - **testlimpio.csv**: datos de prueba ya limpiados.

        La idea es:
        - Confirmar su estructura.
        - Permitir un preprocesamiento ligero adicional (por ejemplo, fecha).
        - Dejar listo el DataFrame que los modelos usarán en la Parte 2.
        """
    )

    if df_train is None or df_test is None:
        st.error("No se pudieron cargar los archivos limpios. Revise que existan.")
    else:
        st.subheader("Vista previa de trainlimpio.csv")
        st.dataframe(df_train.head())

        st.subheader("Vista previa de testlimpio.csv")
        st.dataframe(df_test.head())

        st.markdown("### Configuración de columna de fecha (opcional)")

        posibles_fechas = ["(ninguna)"] + list(df_train.columns)
        date_col = st.selectbox(
            "Seleccione la columna de fecha si existe en los datos:",
            options=posibles_fechas
        )
        if date_col == "(ninguna)":
            date_col = None

        if st.button("Aplicar preprocesamiento básico"):
            df_train_proc, numeric_cols_train = preprocess_data(df_train, date_col)
            df_test_proc, numeric_cols_test = preprocess_data(df_test, date_col)

            st.success("Preprocesamiento aplicado a train y test.")

            st.subheader("Train preprocesado (primeras filas)")
            st.dataframe(df_train_proc.head())

            st.subheader("Test preprocesado (primeras filas)")
            st.dataframe(df_test_proc.head())

            st.markdown("**Columnas numéricas detectadas (train):**")
            st.write(numeric_cols_train)

            st.info(
                "👉 En la Parte 2, estos DataFrames preprocesados pueden usarse "
                "directamente para alimentar los modelos y generar predicciones."
            )


elif pagina == "2. Resultados de modelos":
    st.title("2. Resultados de modelos (carcasa)")

    st.markdown(
        """
        Esta sección está pensada para **mostrar y comparar los resultados**
        de los diferentes modelos para cada uno de los targets.

        Por ahora, solo se muestran **plantillas y botones**.  
        La Parte 2 debe conectar estos elementos con los modelos reales.
        """
    )

    col_sel_1, col_sel_2 = st.columns(2)

    with col_sel_1:
        target = st.selectbox(
            "Seleccione la variable objetivo (target):",
            options=TARGETS
        )
    with col_sel_2:
        modelo = st.selectbox(
            "Seleccione el modelo:",
            options=MODEL_NAMES
        )

    st.markdown("### Ingreso de datos para predicción (demo)")

    st.info(
        "En la versión final, aquí se deben mostrar los campos relevantes "
        "para el modelo (por ejemplo, últimos lags, indicadores técnicos, etc.)."
    )

    with st.form(key="form_prediccion_demo"):
        st.text("Valores de ejemplo (la Parte 2 definirá las características reales).")
        feature_1 = st.number_input("Feature 1 (ejemplo)", value=0.0)
        feature_2 = st.number_input("Feature 2 (ejemplo)", value=0.0)
        feature_3 = st.number_input("Feature 3 (ejemplo)", value=0.0)

        submitted = st.form_submit_button("Generar predicción (placeholder)")

    if submitted:
        st.warning(
            "TODO (Parte 2): conectar este formulario con el modelo real para generar "
            "una predicción usando el target seleccionado."
        )
        pred_demo = feature_1 + feature_2 + feature_3
        st.metric(
            label="Predicción DEMO para " + modelo + " (" + target + ")",
            value=pred_demo
        )

    st.markdown("---")
    st.subheader("Tabla de métricas por modelo y target (placeholder)")

    st.info(
        "La Parte 2 debe reemplazar la tabla de ejemplo con las métricas reales "
        "del proyecto (MAE, RMSE, R², MAPE, etc.) para train/valid/test."
    )

    df_metrics_demo = example_metrics_dataframe()
    st.dataframe(df_metrics_demo)

    st.markdown(
        """
        - En la integración final, se puede:
          - Filtrar la tabla por `Target`.
          - Resaltar el mejor modelo según MAE o RMSE.
          - Exportar esta tabla como CSV para el reporte.
        """
    )


elif pagina == "3. Series de tiempo por target":
    st.title("3. Series de tiempo por target")

    st.markdown(
        """
        Aquí se pueden visualizar las **series de tiempo reales** para cada target
        utilizando `trainlimpio.csv` y `testlimpio.csv`.

        Más adelante, la Parte 2 puede superponer:
        - La serie **real** vs. la serie **predicha** por cada modelo.
        """
    )

    if df_train is None or df_test is None:
        st.error("No se pudieron cargar los archivos limpios.")
    else:
        target = st.selectbox("Seleccione el target:", options=TARGETS)

        posibles_fechas = [c for c in df_train.columns
                           if "date" in c.lower() or "time" in c.lower()]
        date_col = None
        if len(posibles_fechas) > 0:
            date_col = posibles_fechas[0]

        if date_col is not None:
            df_train_plot = df_train.copy()
            df_test_plot = df_test.copy()
            df_train_plot[date_col] = pd.to_datetime(
                df_train_plot[date_col], errors="coerce"
            )
            df_test_plot[date_col] = pd.to_datetime(
                df_test_plot[date_col], errors="coerce"
            )

            df_train_plot["Conjunto"] = "Train"
            df_test_plot["Conjunto"] = "Test"

            df_all = pd.concat([df_train_plot, df_test_plot], ignore_index=True)

            if target in df_all.columns:
                st.subheader("Serie de tiempo real: " + target)

                chart = (
                    alt.Chart(df_all)
                    .mark_line()
                    .encode(
                        x=alt.X(date_col + ":T", title="Fecha"),
                        y=alt.Y(target + ":Q", title=target),
                        color="Conjunto:N",
                        tooltip=[date_col, target, "Conjunto"]
                    )
                    .properties(height=400)
                )

                st.altair_chart(chart, use_container_width=True)

                st.info(
                    "En la Parte 2 se puede agregar otra línea de color distinto "
                    "para mostrar el valor predicho por cada modelo (ARIMA, GRU, TFT, etc.)."
                )
            else:
                st.error("El target seleccionado no existe como columna numérica.")
        else:
            st.warning(
                "No se detectó automáticamente una columna de fecha. "
                "Si los datos tienen una, puede agregarse su manejo en el código."
            )


elif pagina == "4. Visualizaciones de eficiencia":
    st.title("4. Visualizaciones interactivas de eficiencia")

    st.markdown(
        """
        Esta sección cumple con el **Punto 3: Visualizaciones Interactivas** de la rúbrica.

        - Permite una **exploración profunda** del desempeño de los modelos.
        - Incluye:
          1. Gráficos comparativos de rendimiento por modelo y target.
          2. Gráficos de **importancia de características**.
          3. Visualizaciones de **predicciones vs valores reales** y distribución de errores.
        """
    )

    df_metrics = example_metrics_dataframe()
    df_feat_importance = load_feature_importance()
    df_pred_example = example_predictions_dataframe()

    tab_metrics, tab_features, tab_errors = st.tabs([
        "Comparativo de rendimiento",
        "Importancia de características",
        "Distribución de errores"
    ])

    with tab_metrics:
        st.subheader("Comparativo interactivo de métricas por modelo y target")

        with st.expander("Filtros", expanded=True):
            selected_targets = st.multiselect(
                "Targets a visualizar:",
                options=TARGETS,
                default=[TARGETS[0]]
            )
            if not selected_targets:
                selected_targets = TARGETS

            selected_models = st.multiselect(
                "Modelos a comparar:",
                options=MODEL_NAMES,
                default=MODEL_NAMES
            )
            if not selected_models:
                selected_models = MODEL_NAMES

            metric = st.radio(
                "Métrica a visualizar:",
                options=["MAE", "RMSE"],
                horizontal=True
            )

        df_filt = df_metrics[
            df_metrics["Target"].isin(selected_targets)
            & df_metrics["Modelo"].isin(selected_models)
        ]

        if df_filt.empty:
            st.warning("No hay datos para la combinación de filtros seleccionada.")
        else:
            st.markdown("#### Barras comparativas por target")
            chart_bar = (
                alt.Chart(df_filt)
                .mark_bar()
                .encode(
                    x=alt.X("Modelo:N", title="Modelo", sort=MODEL_NAMES),
                    y=alt.Y(f"{metric}:Q", title=metric),
                    color="Modelo:N",
                    column=alt.Column("Target:N", title="Target"),
                    tooltip=["Modelo", "Target", "MAE", "RMSE"]
                )
                .properties(height=300)
                .interactive()
            )
            st.altair_chart(chart_bar, use_container_width=True)

            st.markdown("#### Mapa de calor de desempeño (modelo vs target)")
            heatmap = (
                alt.Chart(df_filt)
                .mark_rect()
                .encode(
                    x=alt.X("Modelo:N", sort=MODEL_NAMES),
                    y=alt.Y("Target:N", sort=TARGETS),
                    color=alt.Color(
                        f"{metric}:Q",
                        title=metric
                    ),
                    tooltip=["Modelo", "Target", "MAE", "RMSE"]
                )
                .properties(height=300)
                .interactive()
            )
            st.altair_chart(heatmap, use_container_width=True)

            st.caption(
                "En la Parte 2: reemplazar `example_metrics_dataframe()` "
                "con las métricas reales de cada modelo y target."
            )

    with tab_features:
        st.subheader("Importancia de características por modelo y target")

        if df_feat_importance is None or df_feat_importance.empty:
            st.info(
                "No se encontró información de importancia de características. "
                "En la Parte 2 se debe generar un archivo `feature_importance.csv` "
                "con las columnas: Modelo, Target, Feature, Importance."
            )
        else:
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                modelo_f = st.selectbox(
                    "Modelo:",
                    options=sorted(df_feat_importance["Modelo"].unique())
                )
            with col_f2:
                target_f = st.selectbox(
                    "Target:",
                    options=sorted(df_feat_importance["Target"].unique())
                )

            df_sel = df_feat_importance[
                (df_feat_importance["Modelo"] == modelo_f)
                & (df_feat_importance["Target"] == target_f)
            ]

            if df_sel.empty:
                st.warning(
                    "No hay datos de importancia para esa combinación "
                    "(modelo, target)."
                )
            else:
                with col_f3:
                    max_k = min(20, len(df_sel))
                    top_k = st.slider(
                        "Número de características a mostrar:",
                        min_value=3,
                        max_value=max_k,
                        value=min(10, max_k)
                    )

                df_top = df_sel.sort_values(
                    "Importance", ascending=False
                ).head(top_k)

                chart_feat = (
                    alt.Chart(df_top)
                    .mark_bar()
                    .encode(
                        x=alt.X("Importance:Q", title="Importancia relativa"),
                        y=alt.Y(
                            "Feature:N",
                            sort="-x",
                            title="Característica"
                        ),
                        tooltip=["Feature", "Importance"]
                    )
                    .properties(height=25 * len(df_top))
                    .interactive()
                )
                st.altair_chart(chart_feat, use_container_width=True)

                st.caption(
                    "👉 En la Parte 2, estos valores deben provenir de métodos "
                    "como coeficientes de regresión, importancia de árboles, "
                    "SHAP, etc."
                )

    with tab_errors:
        st.subheader("Predicciones vs valores reales y distribución de errores")

        if df_pred_example is None or df_pred_example.empty:
            st.info(
                "No hay predicciones de ejemplo. En la Parte 2 se deben "
                "generar predicciones reales para testlimpio.csv y "
                "alimentar aquí la función `example_predictions_dataframe()`."
            )
        else:
            modelo_sel = st.selectbox(
                "Modelo a visualizar:",
                options=df_pred_example["Modelo"].unique()
            )

            df_model = df_pred_example[
                df_pred_example["Modelo"] == modelo_sel
            ]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Dispersión y_real vs y_pred**")
                scatter = (
                    alt.Chart(df_model)
                    .mark_circle(size=40)
                    .encode(
                        x=alt.X("y_real:Q", title="Valor real"),
                        y=alt.Y("y_pred:Q", title="Valor predicho"),
                        tooltip=["y_real", "y_pred", "error"]
                    )
                    .properties(height=350)
                    .interactive()
                )
                st.altair_chart(scatter, use_container_width=True)

            with col2:
                st.markdown("**Histograma de errores (y_real - y_pred)**")
                hist = (
                    alt.Chart(df_model)
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            "error:Q",
                            bin=alt.Bin(maxbins=30),
                            title="Error"
                        ),
                        y=alt.Y("count():Q", title="Frecuencia"),
                        tooltip=[alt.Tooltip("count():Q", title="N")]
                    )
                    .properties(height=350)
                    .interactive()
                )
                st.altair_chart(hist, use_container_width=True)

            st.caption(
                "En la Parte 2, estos gráficos se alimentan con las "
                "predicciones reales por modelo y target, permitiendo ver "
                "dónde el modelo se equivoca más."
            )

    st.markdown("---")
    st.subheader("Notas para la Parte 2 – Integración de Modelos")

    st.markdown(
        """
        Para completar la parte de **Integración de Modelos y Resultados**:

        1. Reemplazar las funciones de ejemplo:
           - `example_metrics_dataframe()` → métricas reales por modelo/target.
           - `load_feature_importance()` → valores reales de importancia.
           - `example_predictions_dataframe()` → `y_real`, `y_pred` y `error` reales.

        2. Exportar estas tablas desde sus notebooks / scripts de entrenamiento
           y guardarlas como `.csv` para que la app las consuma.

        3. Revisar que los nombres de modelos y targets coincidan con los usados
           en esta app para que los filtros funcionen sin cambios.
        """
    )

