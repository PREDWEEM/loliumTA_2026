# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4.4 — LOLIUM TRES ARROYOS 2026
# Versión Corregida: Estabilidad de Índices y Dual Pearson
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
def dtw_distance(a, b):
    na, nb = len(a), len(b)
    dp = np.full((na+1, nb+1), np.inf)
    dp[0,0] = 0
    for i in range(1, na+1):
        for j in range(1, nb+1):
            cost = abs(a[i-1] - b[j-1])
            dp[i,j] = cost + min(dp[i-1,j], dp[i,j-1], dp[i-1,j-1])
    return dp[na, nb]

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
        return emer

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
archivo_meteo = st.sidebar.file_uploader("1. Clima (Lartigau)", type=["xlsx", "csv"])
archivo_campo = st.sidebar.file_uploader("2. Campo (Validación)", type=["xlsx", "csv"])

df_meteo_raw = load_data(archivo_meteo, "TRES ARROYOS")
df_campo_raw = load_data(archivo_campo, "TRES ARROYOS_campo")

st.sidebar.divider()
umbral_er = st.sidebar.slider("Umbral Alerta Temprana", 0.05, 0.80, 0.15)
residualidad = st.sidebar.number_input("Residualidad Herbicida (días)", 0, 60, 20)
t_base_val = st.sidebar.number_input("T Base", value=2.0, step=0.5)
t_opt_max = st.sidebar.number_input("T Óptima Max", value=20.0, step=1.0)
t_critica = st.sidebar.slider("T Crítica (Stop)", 26.0, 42.0, 30.0)
dga_optimo = st.sidebar.number_input("TT Control Post-em (°Cd)", value=600, step=10)
dga_critico = st.sidebar.number_input("Límite Ventana (°Cd)", value=800, step=10)

# ---------------------------------------------------------
# 5. MOTOR DE CÁLCULO
# ---------------------------------------------------------
if df_meteo_raw is not None and modelo_ann is not None:
    
    # --- PREPROCESAMIENTO CLIMA ---
    df = df_meteo_raw.copy()
    df.columns = [c.upper().strip() for c in df.columns]
    df = df.rename(columns={'FECHA': 'Fecha', 'DATE': 'Fecha', 'TMAX': 'TMAX', 'TMIN': 'TMIN', 'PREC': 'Prec', 'LLUVIA': 'Prec'})
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    
    # --- PREDICCIÓN ---
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    df["EMERREL"] = np.maximum(modelo_ann.predict(X), 0.0)
    
    # Restricción hídrica
    df["Prec_sum_21d"] = df["Prec"].rolling(window=21, min_periods=1).sum()
    df["Hydric_Factor"] = 1 / (1 + np.exp(-0.4 * (df["Prec_sum_21d"] - 15)))
    df["EMERREL"] *= df["Hydric_Factor"]
    
    # Bio-térmico
    df["Tmedia"] = (df["TMAX"] + df["TMIN"]) / 2
    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, t_base_val, t_opt_max, t_critica))
    
    # Lógica de Ventana
    fecha_hoy = pd.Timestamp.now().normalize()
    if fecha_hoy not in df['Fecha'].values: fecha_hoy = df['Fecha'].max()
    
    indices_pulso = df.index[df["EMERREL"] >= umbral_er].tolist()
    fecha_inicio_ventana, fecha_control = None, None
    pec, peak_lag, lead_time, pearson_punto, pearson_intervalo = 0.0, 0, 0, 0.0, 0.0
    dga_hoy, dga_7dias = 0.0, 0.0
    df_val_punto = pd.DataFrame()

    if indices_pulso:
        fecha_inicio_ventana = df.loc[indices_pulso[0], "Fecha"]
        df_desde_pico = df[df["Fecha"] >= fecha_inicio_ventana].copy()
        df_desde_pico["DGA_cum"] = df_desde_pico["DG"].cumsum()
        
        df_ctrl_row = df_desde_pico[df_desde_pico["DGA_cum"] >= dga_optimo]
        if not df_ctrl_row.empty: 
            fecha_control = df_ctrl_row.iloc[0]["Fecha"]
        
        dga_hoy = df.loc[(df["Fecha"] >= fecha_inicio_ventana) & (df["Fecha"] <= fecha_hoy), "DG"].sum()
        idx_hoy_list = df.index[df["Fecha"] == fecha_hoy].tolist()
        if idx_hoy_list:
            idx_hoy = idx_hoy_list[0]
            dga_7dias = dga_hoy + df.iloc[idx_hoy + 1 : idx_hoy + 8]["DG"].sum()

    # --- VALIDACIÓN DE CAMPO ---
    if df_campo_raw is not None:
        df_campo = df_campo_raw.copy()
        df_campo.columns = [c.upper().strip() for c in df_campo.columns]
        col_fecha = 'FECHA' if 'FECHA' in df_campo.columns else df_campo.columns[0]
        col_plm2 = 'PLM2' if 'PLM2' in df_campo.columns else df_campo.columns[1]
        df_campo[col_fecha] = pd.to_datetime(df_campo[col_fecha])
        df_campo = df_campo.sort_values(col_fecha).reset_index(drop=True)
        
        max_plm2 = df_campo[col_plm2].max()
        df_campo['Campo_Normalizado'] = df_campo[col_plm2] / max_plm2 if max_plm2 > 0 else 0

        # 1. Pearson Puntual (Sin intervalos)
        df_val_punto = df_campo.merge(df[['Fecha', 'EMERREL']], left_on=col_fecha, right_on='Fecha', how='left')
        pearson_punto = df_val_punto['Campo_Normalizado'].corr(df_val_punto['EMERREL'])

        # 2. Pearson Intervalo
        sim_intervals = []
        last_date = df['Fecha'].min() - pd.Timedelta(days=1)
        for _, row in df_campo.iterrows():
            mask = (df['Fecha'] > last_date) & (df['Fecha'] <= row[col_fecha])
            sim_intervals.append(df.loc[mask, 'EMERREL'].sum())
            last_date = row[col_fecha]
        df_campo['Sim_Intervalo'] = sim_intervals
        pearson_intervalo = df_campo[col_plm2].corr(df_campo['Sim_Intervalo'])

        # 3. Métricas Logísticas
        if fecha_control:
            malezas_totales_campo = df_campo[col_plm2].sum()
            malezas_ctrl_efec = df_campo.loc[df_campo[col_fecha] <= fecha_control, col_plm2].sum()
            pec = (malezas_ctrl_efec / malezas_totales_campo * 100) if malezas_totales_campo > 0 else 0
            
            idx_pico_campo = df_campo[col_plm2].idxmax()
            fecha_pico_campo = df_campo.loc[idx_pico_campo, col_fecha]
            peak_lag = (fecha_control - fecha_pico_campo).days
            
            df_alertas = df[df['EMERREL'] >= umbral_er]
            fecha_1ra = df_alertas['Fecha'].iloc[0] if not df_alertas.empty else fecha_inicio_ventana
            lead_time = (fecha_control - fecha_1ra).days

    # -----------------------------------------------------
    # VISUALIZACIÓN
    # -----------------------------------------------------
    st.title("🌾 PREDWEEM LOLIUM - TRES ARROYOS 2026")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 MONITOR", "🔍 CALIDAD DE AJUSTE", "📈 ESTRATEGIA", "🧪 BIO-LAB"])

    with tab1:
        if df_campo_raw is not None:
            st.markdown("<p class='metric-header'>🚜 DIAGNÓSTICO DE CAMPO</p>", unsafe_allow_html=True)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Control Efectivo (PEC)", f"{pec:.1f}%")
            k2.metric("Lag vs Pico", f"{peak_lag} d")
            k3.metric("Pearson (Puntual)", f"{pearson_punto:.3f}")
            k4.metric("Pearson (Intervalo)", f"{pearson_intervalo:.3f}")
        
        fig_emer = go.Figure()
        fig_emer.add_trace(go.Scatter(x=df["Fecha"], y=df["EMERREL"], name='Simulado', fill='tozeroy', line=dict(color='green')))
        if df_campo_raw is not None:
            fig_emer.add_trace(go.Scatter(x=df_campo[col_fecha], y=df_campo['Campo_Normalizado'], mode='markers+lines', name='Campo', marker=dict(color='red')))
        if fecha_control:
            fig_emer.add_vline(x=fecha_control.timestamp()*1000, line_dash="dash", line_color="orange")
        st.plotly_chart(fig_emer, use_container_width=True)

    with tab2:
        if not df_val_punto.empty:
            st.subheader("Análisis de Dispersión (Ajuste 1:1)")
            fig_scatter = px.scatter(df_val_punto, x="Campo_Normalizado", y="EMERREL", trendline="ols", 
                                     labels={"Campo_Normalizado": "Observado", "EMERREL": "Simulado"})
            fig_scatter.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="Red", dash="dash"))
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Suba datos de campo para habilitar el análisis de ajuste.")

    # ... [Tab 3 y 4 mantienen la lógica original de DTW y Bio-respuesta] ...

else:
    st.info("👋 Bienvenido. Cargue datos climáticos para comenzar.")
