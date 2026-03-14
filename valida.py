
# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4.4 — LOLIUM TRES ARROYOS 2026
# Actualización: Pearson Puntual + Pearson por Intervalos
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
st.set_page_config(
    page_title="PREDWEEM TRES ARROYOS vK4.4", 
    layout="wide",
    page_icon="🌾"
)

st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    [data-testid="stSidebar"] {
        background-color: #dcfce7; 
        border-right: 1px solid #bbf7d0;
    }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p {
        color: #166534 !important;
    }
    .stMetric { 
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .metric-header { color: #1e293b; font-weight: bold; margin-bottom: -10px; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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
        p1 = np.exp(-((jd - 100)**2)/600)
        p2 = np.exp(-((jd - 160)**2)/900) + 0.3*np.exp(-((jd - 260)**2)/1200)
        p3 = np.exp(-((jd - 230)**2)/1500)
        mock_cluster = {"JD_common": jd, "curves_interp": [p2, p1, p3], "medoids_k3": [0, 1, 2]}
        with open(BASE / "modelo_clusters_k3.pkl", "wb") as f:
            pickle.dump(mock_cluster, f)

create_mock_files_if_missing()

# ---------------------------------------------------------
# 3. LÓGICA TÉCNICA (ANN + BIO)
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
        # Escalamiento y retorno de arreglo único (Tasa Diaria)
        emer = (np.array(emer).flatten() + 1) / 2
        emer_ac = np.cumsum(emer)
        emerrel = np.diff(emer_ac, prepend=0)
        return emerrel

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

def load_data(file_uploader, default_name):
    if file_uploader:
        return pd.read_excel(file_uploader) if file_uploader.name.endswith(('.xlsx', '.xls')) else pd.read_csv(file_uploader)
    return None

# ---------------------------------------------------------
# 4. INTERFAZ Y SIDEBAR
# ---------------------------------------------------------
modelo_ann, cluster_model = load_models()

st.sidebar.image("https://raw.githubusercontent.com/PREDWEEM/loliumTA_2026/main/logo.png", use_container_width=True)
st.sidebar.markdown("## 📂 1. Datos del Lote")
archivo_meteo = st.sidebar.file_uploader("1. Clima (Excel/CSV)", type=["xlsx", "csv"])
archivo_campo = st.sidebar.file_uploader("2. Campo (Excel/CSV)", type=["xlsx", "csv"])

df_meteo_raw = load_data(archivo_meteo, "TRES ARROYOS")
df_campo_raw = load_data(archivo_campo, "TRES ARROYOS_campo")

st.sidebar.divider()
umbral_er = st.sidebar.slider("Umbral Alerta Temprana", 0.05, 0.80, 0.15)
residualidad = st.sidebar.number_input("Residualidad Herbicida (días)", 0, 60, 20)
t_base_val = st.sidebar.number_input("T Base", value=2.0)
t_opt_max = st.sidebar.number_input("T Óptima Max", value=20.0)
dga_optimo = st.sidebar.number_input("TT Control Post-em (°Cd)", value=600)

# ---------------------------------------------------------
# 5. MOTOR DE CÁLCULO
# ---------------------------------------------------------
if df_meteo_raw is not None and modelo_ann is not None:
    
    # --- PREPROCESAMIENTO CLIMA ---
    df = df_meteo_raw.copy()
    df.columns = [c.upper().strip() for c in df.columns]
    df = df.rename(columns={'FECHA': 'Fecha', 'DATE': 'Fecha', 'TMAX': 'TMAX', 'TMIN': 'TMIN', 'PREC': 'Prec', 'LLUVIA': 'Prec'})
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    # LIMPIEZA Y RESET DE ÍNDICE (CRÍTICO PARA DIMENSIONES)
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    
    # --- PREDICCIÓN ---
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    y_pred = modelo_ann.predict(X) 
    df["EMERREL"] = np.maximum(y_pred, 0.0)
    
    # Factores hídricos y térmicos
    df["Prec_sum_21d"] = df["Prec"].rolling(window=21, min_periods=1).sum()
    df["Hydric_Factor"] = 1 / (1 + np.exp(-0.4 * (df["Prec_sum_21d"] - 15)))
    df["EMERREL"] *= df["Hydric_Factor"]
    df["Tmedia"] = (df["TMAX"] + df["TMIN"]) / 2
    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, t_base_val, t_opt_max, 30.0))

    # --- VALIDACIÓN DE CAMPO ---
    pearson_punto, pearson_intervalo, pec, peak_lag, lead_time = 0.0, 0.0, 0.0, 0, 0
    fecha_control, fecha_inicio_ventana = None, None
    df_val_punto = pd.DataFrame()
    df_campo = None

    if df_campo_raw is not None:
        df_campo = df_campo_raw.copy()
        df_campo.columns = [c.upper().strip() for c in df_campo.columns]
        col_fecha = 'FECHA' if 'FECHA' in df_campo.columns else df_campo.columns[0]
        col_plm2 = 'PLM2' if 'PLM2' in df_campo.columns else df_campo.columns[1]
        df_campo[col_fecha] = pd.to_datetime(df_campo[col_fecha])
        df_campo = df_campo.sort_values(col_fecha).reset_index(drop=True)
        
        # Pearson Puntual (Sin intervalos)
        df_val_punto = df_campo.merge(df[['Fecha', 'EMERREL']], left_on=col_fecha, right_on='Fecha', how='left')
        df_val_punto['Obs_Norm'] = df_val_punto[col_plm2] / (df_val_punto[col_plm2].max() + 1e-6)
        pearson_punto = df_val_punto['Obs_Norm'].corr(df_val_punto['EMERREL'])

        # Pearson por Intervalo
        sim_intervals = []
        last_date = df['Fecha'].min() - pd.Timedelta(days=1)
        for _, row in df_campo.iterrows():
            mask = (df['Fecha'] > last_date) & (df['Fecha'] <= row[col_fecha])
            sim_intervals.append(df.loc[mask, 'EMERREL'].sum())
            last_date = row[col_fecha]
        df_campo['Sim_Intervalo'] = sim_intervals
        pearson_intervalo = df_campo[col_plm2].corr(df_campo['Sim_Intervalo'])

        # Lógica de Ventana de Control
        indices_pulso = df.index[df["EMERREL"] >= umbral_er].tolist()
        if indices_pulso:
            fecha_inicio_ventana = df.loc[indices_pulso[0], "Fecha"]
            df_v = df[df["Fecha"] >= fecha_inicio_ventana].copy()
            df_v["DGA_cum"] = df_v["DG"].cumsum()
            df_target = df_v[df_v["DGA_cum"] >= dga_optimo]
            if not df_target.empty:
                fecha_control = df_target.iloc[0]["Fecha"]
                m_total = df_campo[col_plm2].sum()
                m_ctrl = df_campo.loc[df_campo[col_fecha] <= fecha_control, col_plm2].sum()
                pec = (m_ctrl / m_total * 100) if m_total > 0 else 0
                peak_lag = (fecha_control - df_campo.loc[df_campo[col_plm2].idxmax(), col_fecha]).days
                lead_time = (fecha_control - fecha_inicio_ventana).days

    # -----------------------------------------------------
    # 6. FRONT-END
    # -----------------------------------------------------
    st.title("🌾 PREDWEEM LOLIUM - TRES ARROYOS 2026")
    
    tab1, tab2, tab3 = st.tabs(["📊 MONITOR", "🔍 CALIDAD DE AJUSTE", "🧪 BIO-LAB"])

    with tab1:
        if df_campo is not None:
            st.markdown("<p class='metric-header'>🚜 DIAGNÓSTICO DE PRECISIÓN</p>", unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Pearson (Puntual)", f"{pearson_punto:.3f}")
            m2.metric("Pearson (Intervalo)", f"{pearson_intervalo:.3f}")
            m3.metric("Control (PEC)", f"{pec:.1f}%")
            m4.metric("Anticipación", f"{lead_time} d")
            st.divider()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Fecha"], y=df["EMERREL"], name="Simulación", fill='tozeroy', line=dict(color='green')))
        if df_campo is not None:
            fig.add_trace(go.Scatter(x=df_campo[col_fecha], y=df_campo[col_plm2]/df_campo[col_plm2].max(), name="Campo (Norm)", mode='markers+lines', marker=dict(color='red')))
        if fecha_control:
            fig.add_vline(x=fecha_control.timestamp()*1000, line_dash="dash", line_color="orange", annotation_text="Momento Control")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if not df_val_punto.empty:
            st.subheader("Dispersión Observado vs Simulado (Punto a Punto)")
            fig_scatter = px.scatter(df_val_punto, x="Obs_Norm", y="EMERREL", trendline="ols", labels={"Obs_Norm": "Observado (Normalizado)", "EMERREL": "Simulado (Tasa)"})
            fig_scatter.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="Red", dash="dash"))
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Suba datos de campo para ver la comparativa de ajuste.")

    with tab3:
        st.subheader("🧪 Curva de Respuesta Fisiológica")
        x_temps = np.linspace(0, 45, 100)
        y_tt = [calculate_tt_scalar(t, t_base_val, t_opt_max, 30.0) for t in x_temps]
        st.plotly_chart(go.Figure(go.Scatter(x=x_temps, y=y_tt, fill='tozeroy', name='TT')), use_container_width=True)

else:
    st.info("👋 Por favor, cargue un archivo de clima para procesar la simulación.")
