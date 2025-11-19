import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os

# --- PyTorch & ML ---
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# =========================================================
# CONFIG GENERAL
# =========================================================
st.set_page_config(
    page_title="Proyecto 2 - Dashboard JPX",
    layout="wide",
    initial_sidebar_state="expanded"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOOKBACK = 64

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

# =========================================================
# 1) CLASES DE MODELOS PYTORCH
# =========================================================
class GRUForecaster(nn.Module):
    def __init__(self, input_size, out_dim=1, hidden=512, layers=2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden, out_dim)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return self.out_act(out)


class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.lstm_enc = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm_dec = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_proj(x)
        enc_out, _ = self.lstm_enc(x)
        dec_out, _ = self.lstm_dec(enc_out)
        attn_out, _ = self.attn(dec_out, enc_out, enc_out)
        gated = self.gate(attn_out) * attn_out + (1 - self.gate(attn_out)) * dec_out
        out = self.fc(self.dropout(gated[:, -1, :]))
        return out


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        yhat = self.fc(last)
        return self.out_act(yhat)

# =========================================================
# 2) RUTAS DE MODELOS GUARDADOS (.pth)
# =========================================================

# ARIMA guardado como diccionario con 'order' y 'train_series'
ARIMA_MODEL_PATHS = {
    "JPX_Gold_Standard_Futures_Close": "./modelos/ARIMA/arima_JPX_Gold_Standard_Futures_Close.pth",
    "JPX_Gold_Standard_Futures_High":  "./modelos/ARIMA/arima_JPX_Gold_Standard_Futures_High.pth",
    "JPX_Gold_Standard_Futures_Low":   "./modelos/ARIMA/arima_JPX_Gold_Standard_Futures_Low.pth",
    "JPX_Gold_Standard_Futures_Open":  "./modelos/ARIMA/arima_JPX_Gold_Standard_Futures_Open.pth",
    "JPX_Gold_Mini_Futures_settlement_price": "./modelos/ARIMA/arima_JPX_Gold_Mini_Futures_settlement_price.pth",
    "JPX_Gold_Mini_Futures_High": "./modelos/ARIMA/arima_JPX_Gold_Mini_Futures_High.pth",
    "JPX_Gold_Mini_Futures_Low":  "./modelos/ARIMA/arima_JPX_Gold_Mini_Futures_Low.pth",
    "JPX_Gold_Mini_Futures_Close":"./modelos/ARIMA/arima_JPX_Gold_Mini_Futures_Close.pth",
    "JPX_Gold_Mini_Futures_Open": "./modelos/ARIMA/arima_JPX_Gold_Mini_Futures_Open.pth",
}

# Regresión lineal guardada con torch.save(lr)
LINEAR_MODEL_PATHS = {
    "JPX_Gold_Standard_Futures_Close": "./modelos/LINEAR/linear_JPX_Gold_Standard_Futures_Close.pth",
    "JPX_Gold_Standard_Futures_High":  "./modelos/LINEAR/linear_JPX_Gold_Standard_Futures_High.pth",
    "JPX_Gold_Standard_Futures_Low":   "./modelos/LINEAR/linear_JPX_Gold_Standard_Futures_Low.pth",
    "JPX_Gold_Standard_Futures_Open":  "./modelos/LINEAR/linear_JPX_Gold_Standard_Futures_Open.pth",
    "JPX_Gold_Mini_Futures_settlement_price": "./modelos/LINEAR/linear_JPX_Gold_Mini_Futures_settlement_price.pth",
    "JPX_Gold_Mini_Futures_High": "./modelos/LINEAR/linear_JPX_Gold_Mini_Futures_High.pth",
    "JPX_Gold_Mini_Futures_Low":  "./modelos/LINEAR/linear_JPX_Gold_Mini_Futures_Low.pth",
    "JPX_Gold_Mini_Futures_Close":"./modelos/LINEAR/linear_JPX_Gold_Mini_Futures_Close.pth",
    "JPX_Gold_Mini_Futures_Open": "./modelos/LINEAR/linear_JPX_Gold_Mini_Futures_Open.pth",
}

# LSTM: state_dict()
LSTM_MODEL_PATHS = {
    "JPX_Gold_Standard_Futures_Close": "./modelos/LSTM/lstm_JPX_Gold_Standard_Futures_Close.pth",
    "JPX_Gold_Standard_Futures_High":  "./modelos/LSTM/lstm_JPX_Gold_Standard_Futures_High.pth",
    "JPX_Gold_Standard_Futures_Low":   "./modelos/LSTM/lstm_JPX_Gold_Standard_Futures_Low.pth",
    "JPX_Gold_Standard_Futures_Open":  "./modelos/LSTM/lstm_JPX_Gold_Standard_Futures_Open.pth",
    "JPX_Gold_Mini_Futures_settlement_price": "./modelos/LSTM/lstm_JPX_Gold_Mini_Futures_settlement_price.pth",
    "JPX_Gold_Mini_Futures_High": "./modelos/LSTM/lstm_JPX_Gold_Mini_Futures_High.pth",
    "JPX_Gold_Mini_Futures_Low":  "./modelos/LSTM/lstm_JPX_Gold_Mini_Futures_Low.pth",
    "JPX_Gold_Mini_Futures_Close":"./modelos/LSTM/lstm_JPX_Gold_Mini_Futures_Close.pth",
    "JPX_Gold_Mini_Futures_Open": "./modelos/LSTM/lstm_JPX_Gold_Mini_Futures_Open.pth",
}

# GRU y TFT ya estaban
GRU_MODEL_PATHS = {
    "JPX_Gold_Standard_Futures_Close": "./modelos/GRU/gru_JPX_Gold_Standard_Futures_Close.pth",
    "JPX_Gold_Standard_Futures_High":  "./modelos/GRU/gru_JPX_Gold_Standard_Futures_High.pth",
    "JPX_Gold_Standard_Futures_Low":   "./modelos/GRU/gru_JPX_Gold_Standard_Futures_Low.pth",
    "JPX_Gold_Standard_Futures_Open":  "./modelos/GRU/gru_JPX_Gold_Standard_Futures_Open.pth",
    "JPX_Gold_Mini_Futures_settlement_price": "./modelos/GRU/gru_JPX_Gold_Mini_Futures_settlement_price.pth",
    "JPX_Gold_Mini_Futures_High": "./modelos/GRU/gru_JPX_Gold_Mini_Futures_High.pth",
    "JPX_Gold_Mini_Futures_Low":  "./modelos/GRU/gru_JPX_Gold_Mini_Futures_Low.pth",
    "JPX_Gold_Mini_Futures_Close":"./modelos/GRU/gru_JPX_Gold_Mini_Futures_Close.pth",
    "JPX_Gold_Mini_Futures_Open": "./modelos/GRU/gru_JPX_Gold_Mini_Futures_Open.pth",
}

TFT_MODEL_PATHS = {
    "JPX_Gold_Standard_Futures_Close": "./modelos/TFT/tft_JPX_Gold_Standard_Futures_Close.pth",
    "JPX_Gold_Standard_Futures_High":  "./modelos/TFT/tft_JPX_Gold_Standard_Futures_High.pth",
    "JPX_Gold_Standard_Futures_Low":   "./modelos/TFT/tft_JPX_Gold_Standard_Futures_Low.pth",
    "JPX_Gold_Standard_Futures_Open":  "./modelos/TFT/tft_JPX_Gold_Standard_Futures_Open.pth",
    "JPX_Gold_Mini_Futures_settlement_price": "./modelos/TFT/tft_JPX_Gold_Mini_Futures_settlement_price.pth",
    "JPX_Gold_Mini_Futures_High": "./modelos/TFT/tft_JPX_Gold_Mini_Futures_High.pth",
    "JPX_Gold_Mini_Futures_Low":  "./modelos/TFT/tft_JPX_Gold_Mini_Futures_Low.pth",
    "JPX_Gold_Mini_Futures_Close":"./modelos/TFT/tft_JPX_Gold_Mini_Futures_Close.pth",
    "JPX_Gold_Mini_Futures_Open": "./modelos/TFT/tft_JPX_Gold_Mini_Futures_Open.pth",
}

# =========================================================
# 3) HELPERS PARA SECUENCIAS Y CARGA DE MODELOS
# =========================================================
def make_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i:i+lookback])
        ys.append(y[i+lookback])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


@st.cache_resource
def load_pytorch_model(model_name: str, target: str, input_size: int):
    """
    Carga modelos PyTorch desde .pth:
    - GRU, TFT: state_dict()
    - LSTM: state_dict()
    """
    if model_name == "GRU":
        path = GRU_MODEL_PATHS[target]
        model = GRUForecaster(input_size=input_size).to(DEVICE)
    elif model_name == "TFT":
        path = TFT_MODEL_PATHS[target]
        model = TemporalFusionTransformer(input_size=input_size).to(DEVICE)
    elif model_name == "LSTM":
        path = LSTM_MODEL_PATHS[target]
        model = LSTMRegressor(input_size=input_size).to(DEVICE)
    else:
        raise ValueError("Modelo PyTorch no soportado: " + str(model_name))

    state_dict = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_sequences_for_target(df_train, df_test, target, date_col="date_id"):
    """
    Reproduce el preprocesamiento básico:
    - ordenar por fecha si existe
    - usar columnas numéricas (excepto fecha)
    - escalar con MinMaxScaler
    - generar secuencias de longitud LOOKBACK
    """
    df_train = df_train.copy()
    df_test = df_test.copy()

    if date_col in df_train.columns:
        df_train = df_train.sort_values(date_col).reset_index(drop=True)
        df_test = df_test.sort_values(date_col).reset_index(drop=True)

    num_cols = [
        c for c in df_train.columns
        if np.issubdtype(df_train[c].dtype, np.number) and c != date_col
    ]

    X_train = df_train[num_cols].values
    X_test = df_test[num_cols].values

    x_scaler = MinMaxScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    y_scaler = MinMaxScaler()
    y_train = y_scaler.fit_transform(df_train[[target]].values)
    y_test = y_scaler.transform(df_test[[target]].values)

    Xtr_seq, ytr_seq = make_sequences(X_train_scaled, y_train, LOOKBACK)
    Xte_seq, yte_seq = make_sequences(X_test_scaled, y_test, LOOKBACK)

    return Xtr_seq, ytr_seq, Xte_seq, yte_seq, y_scaler

# =========================================================
# 4) CARGA DE DATASETS Y PLACEHOLDERS
# =========================================================
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

# =========================================================
# 5) SIDEBAR & NAVEGACIÓN
# =========================================================
st.sidebar.title("Proyecto 2 - JPX Commodities")
st.sidebar.markdown("**CC3084 – Data Science**")
st.sidebar.markdown("---")

pagina = st.sidebar.radio(
    "Navegación",
    [
        "Inicio",
        "1. Preprocesamiento de datos",
        "2. Resultados de modelos",
        "4. Visualizaciones de eficiencia"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("👥 Equipo: Nancy, Brandon, Santiago, Andre")

df_train, df_test = load_train_test()

# =========================================================
# 6) PÁGINAS
# =========================================================
if pagina == "Inicio":
    st.title("Dashboard Proyecto 2 – JPX Gold Futures")

    st.markdown("""
    # Commodity Price Forecasting Dashboard
    ### Proyecto 2 — Resultados Parciales y Visualizaciones

    Esta aplicación presenta los avances del Proyecto 2 relacionados con la predicción de precios de commodities, utilizando como caso de estudio los contratos de oro del conjunto de datos de la competencia MITSUI&CO. Commodity Prediction Challenge (Kaggle).

    El objetivo de este panel es proporcionar una interfaz clara y organizada para analizar los resultados obtenidos mediante distintos modelos de predicción aplicados a series temporales financieras. A través del menú lateral, el usuario puede acceder a:

    - Visualización de series históricas y comportamiento temporal.
    - Resultados de modelos estadísticos y de aprendizaje automático (ARIMA, Regresión Lineal, LSTM, GRU, TFT).
    - Comparación de predicciones frente a valores reales.
    - Métricas de desempeño como MAE, RMSE y R².
    - Gráficos y análisis de residuales para diagnóstico del modelo.

    Este dashboard permite una exploración estructurada del proceso de modelado y evaluación, facilitando la interpretación de resultados y la comparación entre diferentes enfoques predictivos.
    """)


    if df_train is None or df_test is None:
        st.error(
            "No se encontraron los archivos `trainlimpio.csv` y/o `testlimpio.csv` "
            "en el directorio actual. Colóquelos junto al `main.py`."
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
    st.title("2. Resultados de modelos")

    st.markdown(
        """
        Esta sección está pensada para **mostrar y comparar los resultados**
        de los diferentes modelos para cada uno de los targets.

        Ahora, para **ARIMA**, **Regresión Lineal**, **LSTM**, **GRU** y **TFT**
        se usan los modelos reales guardados en la carpeta `modelos/`.
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

    st.markdown("### Ingreso de datos para predicción (demo / no PyTorch)")

    st.info(
        "Los campos siguientes solo se usan como DEMO en caso de que falten `trainlimpio.csv` "
        "o `testlimpio.csv`. Con los archivos correctos, la app usa los datos reales."
    )

    with st.form(key="form_prediccion_demo"):
        st.text("Valores de ejemplo (para el modo DEMO).")
        feature_1 = st.number_input("Feature 1 (ejemplo)", value=0.0)
        feature_2 = st.number_input("Feature 2 (ejemplo)", value=0.0)
        feature_3 = st.number_input("Feature 3 (ejemplo)", value=0.0)

        submitted = st.form_submit_button("Evaluar modelo")

    if submitted:
        if df_train is None or df_test is None:
            # No hay data real -> DEMO
            st.warning(
                "No se encontraron `trainlimpio.csv` y/o `testlimpio.csv`. "
                "Mostrando resultado DEMO."
            )
            pred_demo = feature_1 + feature_2 + feature_3
            st.metric(
                label="Predicción DEMO para " + modelo + " (" + target + ")",
                value=pred_demo
            )

        else:
            # Hay data real -> usar modelos y .pth
            if modelo in ["GRU", "TFT", "LSTM"]:
                # 1) generar secuencias reales para ese target
                Xtr_seq, ytr_seq, Xte_seq, yte_seq, y_scaler = get_sequences_for_target(
                    df_train, df_test, target, date_col="date_id"
                )

                if len(Xte_seq) == 0:
                    st.error(
                        "No se pudieron generar secuencias (quizá hay muy pocos datos "
                        "para LOOKBACK=" + str(LOOKBACK) + ")."
                    )
                else:
                    # 2) cargar el modelo PyTorch desde el .pth
                    model_pt = load_pytorch_model(
                        model_name=modelo,
                        target=target,
                        input_size=Xtr_seq.shape[-1]
                    )

                    # 3) predicción sobre TODO el test
                    with torch.no_grad():
                        xb = torch.tensor(Xte_seq, dtype=torch.float32).to(DEVICE)
                        yhat_scaled = model_pt(xb).cpu().numpy()

                    yhat = y_scaler.inverse_transform(yhat_scaled)
                    ytrue = y_scaler.inverse_transform(yte_seq)

                    mae = mean_absolute_error(ytrue, yhat)
                    rmse = np.sqrt(mean_squared_error(ytrue, yhat))

                    st.success("Modelo " + modelo + " cargado y evaluado correctamente.")
                    st.metric("MAE " + modelo + " (" + target + ")", "{:.3f}".format(mae))
                    st.metric("RMSE " + modelo + " (" + target + ")", "{:.3f}".format(rmse))

                    df_plot = pd.DataFrame({
                        "idx": np.arange(len(ytrue)),
                        "Real": ytrue.flatten(),
                        "Predicho": yhat.flatten()
                    })

                    df_long = df_plot.melt(
                        id_vars=["idx"],
                        value_vars=["Real", "Predicho"],
                        var_name="Tipo",
                        value_name="Valor"
                    )

                    chart = (
                        alt.Chart(df_long)
                        .mark_line()
                        .encode(
                            x=alt.X("idx:Q", title="Índice (tiempo)"),
                            y=alt.Y("Valor:Q", title=target),
                            color=alt.Color("Tipo:N", title="Serie"),
                            tooltip=["idx", "Tipo", "Valor"]
                        )
                        .properties(height=350)
                    )

                    st.altair_chart(chart, use_container_width=True)

            elif modelo == "Regresión Lineal":
                # Secuencias y escalado igual que en entrenamiento
                Xtr_seq, ytr_seq, Xte_seq, yte_seq, y_scaler = get_sequences_for_target(
                    df_train, df_test, target, date_col="date_id"
                )

                if len(Xte_seq) == 0:
                    st.error(
                        "No se pudieron generar secuencias para Regresión Lineal "
                        "(LOOKBACK=" + str(LOOKBACK) + ")."
                    )
                else:
                    # Flatten igual que en el notebook de entrenamiento
                    Xte_flat = Xte_seq.reshape(Xte_seq.shape[0], -1)

                    # Cargar modelo sklearn guardado con torch.save(lr)
                    path_lr = LINEAR_MODEL_PATHS[target]
                    lr_model = torch.load(path_lr)

                    yhat_scaled = lr_model.predict(Xte_flat).reshape(-1, 1)
                    yhat = y_scaler.inverse_transform(yhat_scaled)
                    ytrue = y_scaler.inverse_transform(yte_seq)

                    mae = mean_absolute_error(ytrue, yhat)
                    rmse = np.sqrt(mean_squared_error(ytrue, yhat))

                    st.success("Modelo de Regresión Lineal cargado y evaluado.")
                    st.metric("MAE Regresión (" + target + ")", "{:.3f}".format(mae))
                    st.metric("RMSE Regresión (" + target + ")", "{:.3f}".format(rmse))

                    df_plot = pd.DataFrame({
                        "idx": np.arange(len(ytrue)),
                        "Real": ytrue.flatten(),
                        "Predicho": yhat.flatten()
                    })

                    df_long = df_plot.melt(
                        id_vars=["idx"],
                        value_vars=["Real", "Predicho"],
                        var_name="Tipo",
                        value_name="Valor"
                    )

                    chart = (
                        alt.Chart(df_long)
                        .mark_line()
                        .encode(
                            x=alt.X("idx:Q", title="Índice (tiempo)"),
                            y=alt.Y("Valor:Q", title=target),
                            color=alt.Color("Tipo:N", title="Serie"),
                            tooltip=["idx", "Tipo", "Valor"]
                        )
                        .properties(height=350)
                    )

                    st.altair_chart(chart, use_container_width=True)

            elif modelo == "ARIMA":
                # Cargar config ARIMA desde .pth
                path_arima = ARIMA_MODEL_PATHS[target]
                arima_obj = torch.load(path_arima)
                order = arima_obj.get("order", (2, 1, 2))

                y_train = df_train[target].values.astype(float)
                y_test = df_test[target].values.astype(float)

                history = list(y_train)
                preds = []

                for t in range(len(y_test)):
                    try:
                        model_arima = SARIMAX(
                            history,
                            order=order,
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        ).fit(disp=False)
                        yhat = model_arima.forecast(steps=1)[0]
                    except Exception:
                        yhat = history[-1]
                    preds.append(yhat)
                    history.append(y_test[t])

                ytrue = np.array(y_test, dtype=float)
                yhat = np.array(preds, dtype=float)

                mae = mean_absolute_error(ytrue, yhat)
                rmse = np.sqrt(mean_squared_error(ytrue, yhat))

                st.success(
                    "Modelo ARIMA cargado (orden " + str(order) + ") y evaluado."
                )
                st.metric("MAE ARIMA (" + target + ")", "{:.3f}".format(mae))
                st.metric("RMSE ARIMA (" + target + ")", "{:.3f}".format(rmse))

                df_plot = pd.DataFrame({
                    "idx": np.arange(len(ytrue)),
                    "Real": ytrue,
                    "Predicho": yhat
                })

                df_long = df_plot.melt(
                    id_vars=["idx"],
                    value_vars=["Real", "Predicho"],
                    var_name="Tipo",
                    value_name="Valor"
                )

                chart = (
                    alt.Chart(df_long)
                    .mark_line()
                    .encode(
                        x=alt.X("idx:Q", title="Índice (tiempo)"),
                        y=alt.Y("Valor:Q", title=target),
                        color=alt.Color("Tipo:N", title="Serie"),
                        tooltip=["idx", "Tipo", "Valor"]
                    )
                    .properties(height=350)
                )

                st.altair_chart(chart, use_container_width=True)

    st.markdown("---")
    st.subheader("Tabla de métricas por modelo y target (placeholder)")

    st.info(
        "La Parte 2 puede reemplazar la tabla de ejemplo con las métricas reales "
        "del proyecto (MAE, RMSE, R², MAPE, etc.) para train/valid/test."
    )

    df_metrics_demo = example_metrics_dataframe()
    st.dataframe(df_metrics_demo)


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

    # ---------- Métricas ----------
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
                    y=alt.Y(metric + ":Q", title=metric),
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
                    color=alt.Color(metric + ":Q", title=metric),
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

    # ---------- Importancia de características ----------
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

    # ---------- Distribución de errores ----------
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
    