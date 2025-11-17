# app.py
# ============================================================
# Dashboard Proyecto 2 - Data Science (Carcasa)
#
# Parte 1: Diseño y Desarrollo de la Aplicación (40 puntos)
#
# - Usa los datasets: trainlimpio.csv y testlimpio.csv
# - Permite explorar y preprocesar datos de entrada.
# - Deja secciones, botones y estructuras listas para:
#     * Conectar los modelos reales (ARIMA, Regresión, LSTM, GRU, TFT, etc.).
#     * Llenar las métricas con los resultados del proyecto.
#     * Agregar gráficas específicas: ARIMA vs GRU vs TFT, por target.
#
# NOTA:
#   Por ahora esta app es una CARCASA. No carga modelos reales todavía.
#   La Parte 2 (Integración de Modelos en la Aplicación) debe:
#     - Cargar archivos .pkl / .pt con los modelos entrenados.
#     - Usar los datos de testlimpio.csv para generar predicciones.
#     - Completar las tablas de métricas y las gráficas comparativas.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os

# -----------------------------
# Configuración general de la página
# -----------------------------
st.set_page_config(
    page_title="Proyecto 2 - Dashboard JPX",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Constantes del proyecto
# -----------------------------
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


# -----------------------------
# Funciones auxiliares
# -----------------------------
@st.cache_data
def load_train_test():
    """
    Carga trainlimpio.csv y testlimpio.csv si existen en el directorio actual.
    Devuelve:
      - df_train, df_test (o None si no se encuentran).
    """
    df_train = None
    df_test = None

    if os.path.exists("trainlimpio.csv"):
        df_train = pd.read_csv("trainlimpio.csv")
    if os.path.exists("testlimpio.csv"):
        df_test = pd.read_csv("testlimpio.csv")

    return df_train, df_test


def preprocess_data(df, date_col=None):
    """
    Preprocesamiento sencillo:
      - Convierte la columna de fecha a datetime (si se indica).
      - Ordena por fecha (si aplica).
      - Rellena NaN numéricos con la mediana.
    Devuelve el DataFrame procesado y la lista de columnas numéricas.
    """
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
    """
    DataFrame de EJEMPLO para las métricas de modelos.
    La Parte 2 debe reemplazar esta función con las métricas reales
    del proyecto (MAE, RMSE, R2, etc.).
    """
    data = {
        "Modelo": MODEL_NAMES,
        "Target": [
            "Close Std", "Close Std", "Close Std", "Close Std", "Close Std"
        ],
        "MAE": [132, 7000, 9000, 110, 120],
        "RMSE": [262, 7500, 9500, 200, 250]
    }
    return pd.DataFrame(data)


# -----------------------------
# Sidebar de navegación
# -----------------------------
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

# Carga global de datasets
df_train, df_test = load_train_test()


# ============================================================
# PÁGINA: INICIO
# ============================================================
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


# ============================================================
# PÁGINA 1: PREPROCESAMIENTO DE DATOS
# ============================================================
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


# ============================================================
# PÁGINA 2: RESULTADOS DE MODELOS (CARCASA)
# ============================================================
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
        "👉 La Parte 2 debe reemplazar la tabla de ejemplo con las métricas reales "
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


# ============================================================
# PÁGINA 3: SERIES DE TIEMPO POR TARGET
# ============================================================
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

        # Intentar detectar una columna de fecha
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
                    "👉 En la Parte 2 se puede agregar otra línea de color distinto "
                    "para mostrar el valor predicho por cada modelo (ARIMA, GRU, TFT, etc.)."
                )
            else:
                st.error("El target seleccionado no existe como columna numérica.")
        else:
            st.warning(
                "No se detectó automáticamente una columna de fecha. "
                "Si los datos tienen una, puede agregarse su manejo en el código."
            )


# ============================================================
# PÁGINA 4: VISUALIZACIONES DE EFICIENCIA
# ============================================================
elif pagina == "4. Visualizaciones de eficiencia":
    st.title("4. Visualizaciones interactivas de eficiencia")

    st.markdown(
        """
        Esta sección está dedicada a comparar el rendimiento de los modelos.

        Por ahora usa métricas de **ejemplo**, pero la estructura está lista para:
        - Comparar **ARIMA vs GRU vs TFT** y otros modelos.
        - Ver diferencias de MAE y RMSE por target.
        """
    )

    df_metrics = example_metrics_dataframe()  # Placeholder

    st.subheader("MAE por modelo (ejemplo)")

    chart_mae = (
        alt.Chart(df_metrics)
        .mark_bar()
        .encode(
            x=alt.X("Modelo:N", title="Modelo"),
            y=alt.Y("MAE:Q", title="MAE"),
            color="Modelo:N",
            tooltip=["Modelo", "Target", "MAE", "RMSE"]
        )
        .properties(height=350)
    )

    st.altair_chart(chart_mae, use_container_width=True)

    st.subheader("RMSE por modelo (ejemplo)")

    chart_rmse = (
        alt.Chart(df_metrics)
        .mark_bar()
        .encode(
            x=alt.X("Modelo:N", title="Modelo"),
            y=alt.Y("RMSE:Q", title="RMSE"),
            color="Modelo:N",
            tooltip=["Modelo", "Target", "MAE", "RMSE"]
        )
        .properties(height=350)
    )

    st.altair_chart(chart_rmse, use_container_width=True)

    st.markdown("---")
    st.subheader("Notas para la Parte 2 – Integración de Modelos")

    st.markdown(
        """
        Para completar la rúbrica de **Integración de Modelos en la Aplicación (20%)**,
        la siguiente persona debe:

        1. **Conectar los modelos reales**  
           - Cargar archivos de modelos entrenados (`.pkl`, `.pt`, etc.) para:
             - ARIMA  
             - Regresión Lineal  
             - LSTM  
             - GRU  
             - TFT  

        2. **Llenar las métricas con los resultados del Proyecto**  
           - Reemplazar `example_metrics_dataframe()` por un DataFrame construido
             con los resultados reales de cada modelo y cada target
             (MAE, RMSE, R², MAPE, etc.).

        3. **Agregar gráficas específicas: ARIMA vs GRU vs TFT, etc.**  
           - Comparar modelos por target, por ejemplo:
             - Gráfico de barras MAE por modelo para `JPX_Gold_Standard_Futures_Close`.
             - Series de tiempo real vs. predicho para cada modelo.
             - Scatter `y_real` vs. `y_pred` con línea y = x.

        Con estas modificaciones, la app cubrirá tanto la **Parte 1**
        (diseño y desarrollo) como la **Parte 2** (integración de modelos).
        """
    )
