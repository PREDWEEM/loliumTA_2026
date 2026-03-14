
# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4.4 — LOLIUM TRES ARROYOS 2026
# Actualización: Dual Pearson (Puntual vs Intervalos) + Scatter Plot
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
# 2. MOCKS Y FUNCIONES TÉCNICAS
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

def calculate_tt_scalar(t, t_base, t_opt, t_crit):
    if t <= t_base: return 0.0
    elif t <= t_opt: return t - t_base
    elif t < t_crit: return (t - t_base) * ((t_crit - t) / (t_crit - t_opt))
    return 0.0

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
        return (np.array(emer).flatten() + 1) / 2

@st.cache_resource
def load_models():
    try:
        ann = PracticalANNModel(np.load(BASE/"IW.npy"), np.load(BASE/"bias_IW.npy"), np.load(BASE/"LW.npy"), np.load(BASE/"bias_out.npy"))
        with open(BASE/"modelo_clusters_k3.pkl", "rb") as f:
            k3 = pickle.load(f)
        return ann, k3
    except: return None, None

# ---------------------------------------------------------
# 3. INTERFAZ Y PROCESAMIENTO
# ---------------------------------------------------------
modelo_ann, cluster_model = load_models()

st.sidebar.image("https://raw.githubusercontent.com/PREDWEEM/loliumTA_2026/main/logo.png", use_container_width=True)
archivo_meteo = st.sidebar.file_uploader("1. Clima", type=["xlsx", "csv"])
archivo_campo = st.sidebar.file_uploader("2. Campo", type=["xlsx", "csv"])

umbral_er = st.sidebar.slider("Umbral Alerta", 0.05, 0.80, 0.15)
dga_optimo = st.sidebar.number_input("TT Control (°Cd)", value=600)

if archivo_meteo is not None and modelo_ann is not None:
    # Procesamiento Clima
    df = pd.read_excel(archivo_meteo) if archivo_meteo.name.endswith('xlsx') else pd.read_csv(archivo_meteo)
    df.columns = [c.upper().strip() for c in df.columns]
    df = df.rename(columns={'FECHA': 'Fecha', 'TMAX': 'TMAX', 'TMIN': 'TMIN', 'PREC': 'Prec'})
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df.dropna(subset=["TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    
    # Predicción
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].values.astype(float)
    df["EMERREL"] = np.maximum(modelo_ann.predict(X), 0.0)
    
    # Bio-Térmico
    df["Tmedia"] = (df["TMAX"] + df["TMIN"]) / 2
    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, 2.0, 20.0, 30.0))

    # --- VALIDACIÓN DUAL PEARSON ---
    pearson_punto, pearson_intervalo = 0.0, 0.0
    df_val_punto = pd.DataFrame()

    if archivo_campo is not None:
        df_campo = pd.read_excel(archivo_campo) if archivo_campo.name.endswith('xlsx') else pd.read_csv(archivo_campo)
        df_campo.columns = [c.upper().strip() for c in df_campo.columns]
        col_fecha = 'FECHA' if 'FECHA' in df_campo.columns else df_campo.columns[0]
        col_plm2 = 'PLM2' if 'PLM2' in df_campo.columns else df_campo.columns[1]
        df_campo[col_fecha] = pd.to_datetime(df_campo[col_fecha])
        df_campo = df_campo.sort_values(col_fecha).reset_index(drop=True)

        # 1. Pearson Puntual (Sin intervalos)
        df_val_punto = df_campo.merge(df[['Fecha', 'EMERREL']], left_on=col_fecha, right_on='Fecha', how='left')
        df_val_punto['Obs_Norm'] = df_val_punto[col_plm2] / (df_val_punto[col_plm2].max() + 1e-6)
        pearson_punto = df_val_punto['Obs_Norm'].corr(df_val_punto['EMERREL'])

        # 2. Pearson por Intervalos
        sim_intervals = []
        last_date = df['Fecha'].min() - pd.Timedelta(days=1)
        for _, row in df_campo.iterrows():
            mask = (df['Fecha'] > last_date) & (df['Fecha'] <= row[col_fecha])
            sim_intervals.append(df.loc[mask, 'EMERREL'].sum())
            last_date = row[col_fecha]
        
        df_campo['Sim_Intervalo'] = sim_intervals
        pearson_intervalo = df_campo[col_plm2].corr(df_campo['Sim_Intervalo'])

    # -----------------------------------------------------
    # 4. FRONT-END
    # -----------------------------------------------------
    st.title("🌾 PREDWEEM LOLIUM - TRES ARROYOS 2026")
    
    tab1, tab2, tab3 = st.tabs(["📊 MONITOR", "🔍 CALIDAD DE AJUSTE", "🧪 BIO-LAB"])

    with tab1:
        if archivo_campo is not None:
            st.markdown("<p class='metric-header'>🚜 SINCRONÍA DE CAMPO</p>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Pearson (Puntual)", f"{pearson_punto:.3f}", "Sensibilidad a la intensidad")
            m2.metric("Pearson (Intervalo)", f"{pearson_intervalo:.3f}", "Sincronía de flujos")
        
        fig_main = go.Figure()
        fig_main.add_trace(go.Scatter(x=df["Fecha"], y=df["EMERREL"], name="Modelo", fill='tozeroy', line=dict(color='green')))
        if archivo_campo is not None:
            fig_main.add_trace(go.Scatter(x=df_campo[col_fecha], y=df_campo[col_plm2]/df_campo[col_plm2].max(), name="Campo", mode='markers'))
        st.plotly_chart(fig_main, use_container_width=True)

    with tab2:
        if not df_val_punto.empty:
            st.header("📈 Análisis de Correlación Puntual")
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.write("""
                **Interpretación:**
                Este gráfico compara la emergencia observada vs. la simulada en la misma fecha. 
                - La **línea punteada** es el ajuste ideal (1:1).
                - Si los puntos están por encima, el modelo subestima.
                - Si están por debajo, el modelo sobreestima.
                """)
                st.metric("Coeficiente R", f"{pearson_punto:.4f}")
            
            with col_b:
                fig_scatter = px.scatter(
                    df_val_punto, x="Obs_Norm", y="EMERREL", 
                    labels={"Obs_Norm": "Observado (Normalizado)", "EMERREL": "Simulado (Tasa)"},
                    hover_data=[col_fecha], trendline="ols", template="plotly_white"
                )
                # Añadir línea 1:1
                fig_scatter.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="Red", dash="dash"))
                st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Suba datos de campo para ver el análisis de dispersión.")

    with tab3:
        st.write("Configuración de curvas de respuesta térmica...")

else:
    st.info("👋 Cargue archivos para comenzar.")
