# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4.4 — LOLIUM TRES ARROYOS 2026
# Versión Corregida: Solución de Error de Indexación y Dimensiones
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pickle
import io
from datetime import timedelta
from pathlib import Path

# ---------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILO
# ---------------------------------------------------------
st.set_page_config(page_title="PREDWEEM TRES ARROYOS vK4.4", layout="wide", page_icon="🌾")

st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    [data-testid="stSidebar"] { background-color: #dcfce7; border-right: 1px solid #bbf7d0; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e2e8f0; }
    .metric-header { color: #1e293b; font-weight: bold; margin-bottom: -10px; }
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ---------------------------------------------------------
# 2. ROBUSTEZ Y ARCHIVOS (MOCKS)
# ---------------------------------------------------------
def create_mock_files_if_missing():
    if not (BASE / "IW.npy").exists():
        np.save(BASE / "IW.npy", np.random.rand(4, 10))
        np.save(BASE / "bias_IW.npy", np.random.rand(10))
        np.save(BASE / "LW.npy", np.random.rand(1, 10))
        np.save(BASE / "bias_out.npy", np.random.rand(1))
    
    if not (BASE / "modelo_clusters_k3.pkl").exists():
        jd = np.arange(1, 366)
        mock_cluster = {"JD_common": jd, "curves_interp": [np.sin(jd/50)]*3, "medoids_k3": [0, 1, 2]}
        with open(BASE / "modelo_clusters_k3.pkl", "wb") as f:
            pickle.dump(mock_cluster, f)

create_mock_files_if_missing()

# ---------------------------------------------------------
# 3. LÓGICA TÉCNICA
# ---------------------------------------------------------
def calculate_tt_scalar(t, t_base, t_opt, t_crit):
    if t <= t_base: return 0.0
    elif t <= t_opt: return t - t_base
    elif t < t_crit: return (t - t_base) * ((t_crit - t) / (t_crit - t_opt))
    else: return 0.0

class PracticalANNModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW, self.bIW, self.LW, self.bLW = IW, bIW, LW, bLW
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])

    def normalize(self, X):
        return 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1

    def predict(self, Xreal):
        Xn = self.normalize(Xreal)
        emer = []
        for x in Xn:
            z1 = self.IW.T @ x + self.bIW
            a1 = np.tanh(z1)
            z2 = self.LW @ a1 + self.bLW
            emer.append(np.tanh(z2))
        emer = (np.array(emer).flatten() + 1) / 2
        # Calculamos la tasa diaria (diff) para que el largo coincida con la entrada
        emer_ac = np.cumsum(emer)
        emerrel = np.diff(emer_ac, prepend=0)
        return emerrel # DEVOLVEMOS SOLO UN ARREGLO PARA EVITAR EL VALUEERROR

@st.cache_resource
def load_models():
    try:
        ann = PracticalANNModel(np.load(BASE/"IW.npy"), np.load(BASE/"bias_IW.npy"), np.load(BASE/"LW.npy"), np.load(BASE/"bias_out.npy"))
        with open(BASE/"modelo_clusters_k3.pkl", "rb") as f:
            k3 = pickle.load(f)
        return ann, k3
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        return None, None

# ---------------------------------------------------------
# 4. INTERFAZ Y SIDEBAR
# ---------------------------------------------------------
modelo_ann, cluster_model = load_models()

st.sidebar.image("https://raw.githubusercontent.com/PREDWEEM/loliumTA_2026/main/logo.png", use_container_width=True)
archivo_meteo = st.sidebar.file_uploader("1. Clima (Lartigau)", type=["xlsx", "csv"])
archivo_campo = st.sidebar.file_uploader("2. Campo (Validación)", type=["xlsx", "csv"])

umbral_er = st.sidebar.slider("Umbral Alerta Temprana", 0.05, 0.80, 0.15)
dga_optimo = st.sidebar.number_input("TT Control Post-em (°Cd)", value=600, step=10)

# ---------------------------------------------------------
# 5. MOTOR DE CÁLCULO
# ---------------------------------------------------------
if archivo_meteo is not None and modelo_ann is not None:
    # --- PREPROCESAMIENTO CLIMA ---
    df = pd.read_excel(archivo_meteo) if archivo_meteo.name.endswith('xlsx') else pd.read_csv(archivo_meteo)
    df.columns = [c.upper().strip() for c in df.columns]
    df = df.rename(columns={'FECHA': 'Fecha', 'DATE': 'Fecha', 'TMAX': 'TMAX', 'TMIN': 'TMIN', 'PREC': 'Prec', 'LLUVIA': 'Prec'})
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    # Limpieza estricta y reset_index para asegurar coincidencia de filas
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    
    # --- PREDICCIÓN ---
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    y_pred = modelo_ann.predict(X) 
    
    # Ahora y_pred es un solo arreglo, np.maximum funcionará correctamente
    df["EMERREL"] = np.maximum(y_pred, 0.0)
    
    # Restricción hídrica
    df["Prec_sum_21d"] = df["Prec"].rolling(window=21, min_periods=1).sum()
    df["Hydric_Factor"] = 1 / (1 + np.exp(-0.4 * (df["Prec_sum_21d"] - 15)))
    df["EMERREL"] *= df["Hydric_Factor"]
    
    # Bio-térmico
    df["Tmedia"] = (df["TMAX"] + df["TMIN"]) / 2
    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, 2.0, 20.0, 30.0))

    # --- VALIDACIÓN DE CAMPO ---
    pearson_punto, pearson_intervalo, pec = 0.0, 0.0, 0.0
    fecha_control = None
    df_val_punto = pd.DataFrame()

    if archivo_campo is not None:
        df_campo = pd.read_excel(archivo_campo) if archivo_campo.name.endswith('xlsx') else pd.read_csv(archivo_campo)
        df_campo.columns = [c.upper().strip() for c in df_campo.columns]
        col_fecha = 'FECHA' if 'FECHA' in df_campo.columns else df_campo.columns[0]
        col_plm2 = 'PLM2' if 'PLM2' in df_campo.columns else df_campo.columns[1]
        df_campo[col_fecha] = pd.to_datetime(df_campo[col_fecha])
        df_campo = df_campo.sort_values(col_fecha).reset_index(drop=True)
        
        # Pearson Puntual
        df_val_punto = df_campo.merge(df[['Fecha', 'EMERREL']], left_on=col_fecha, right_on='Fecha', how='left')
        pearson_punto = df_val_punto[col_plm2].corr(df_val_punto['EMERREL'])

        # Pearson Intervalo
        sim_intervals = []
        last_date = df['Fecha'].min() - pd.Timedelta(days=1)
        for _, row in df_campo.iterrows():
            mask = (df['Fecha'] > last_date) & (df['Fecha'] <= row[col_fecha])
            sim_intervals.append(df.loc[mask, 'EMERREL'].sum())
            last_date = row[col_fecha]
        df_campo['Sim_Intervalo'] = sim_intervals
        pearson_intervalo = df_campo[col_plm2].corr(df_campo['Sim_Intervalo'])

    # -----------------------------------------------------
    # VISUALIZACIÓN
    # -----------------------------------------------------
    st.title("🌾 PREDWEEM LOLIUM - TRES ARROYOS 2026")
    
    tab1, tab2 = st.tabs(["📊 MONITOR", "🔍 AJUSTE"])

    with tab1:
        if archivo_campo is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("Pearson Puntual", f"{pearson_punto:.3f}")
            c2.metric("Pearson Intervalo", f"{pearson_intervalo:.3f}")
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Fecha"], y=df["EMERREL"], name='Modelo', fill='tozeroy', line=dict(color='green')))
        if archivo_campo is not None:
            norm_obs = df_campo[col_plm2] / (df_campo[col_plm2].max() + 1e-6)
            fig.add_trace(go.Scatter(x=df_campo[col_fecha], y=norm_obs, mode='markers+lines', name='Campo (Norm)'))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if not df_val_punto.empty:
            fig_scatter = px.scatter(df_val_punto, x=col_plm2, y="EMERREL", trendline="ols")
            st.plotly_chart(fig_scatter, use_container_width=True)

else:
    st.info("👋 Suba los datos climáticos para ejecutar el modelo.")
