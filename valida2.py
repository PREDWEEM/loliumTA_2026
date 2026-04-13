# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4.9.9 — LOLIUM TRES ARROYOS 2026
# Actualización: 
# - Validación: Match estricto de valores observados > 0.
# - Métricas: Pearson y Dispersión basados en eventos reales.
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import io
import time
from datetime import timedelta
from pathlib import Path
from scipy.signal import find_peaks
import base64

# ---------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ---------------------------------------------------------
if 'arranque_fase' not in st.session_state:
    st.set_page_config(page_title="PREDWEEM INTEGRAL", layout="wide", page_icon="🌾")
    st.session_state.arranque_fase = 1

st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    [data-testid="stSidebar"] { background-color: #dcfce7; border-right: 1px solid #bbf7d0; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .metric-header { color: #1e293b; font-weight: bold; margin-bottom: -10px; margin-top: 15px; }
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ---------------------------------------------------------
# 2. FUNCIONES TÉCNICAS (MODELO Y FÍSICA)
# ---------------------------------------------------------
def calculate_tt_scalar(t, t_base, t_opt, t_crit):
    if t <= t_base: return 0.0
    elif t <= t_opt: return t - t_base
    elif t < t_crit: return (t - t_base) * ((t_crit - t) / (t_crit - t_opt))
    else: return 0.0

def calcular_et0_hargreaves(jday, tmax, tmin, latitud=-38.37):
    lat_rad = np.radians(latitud)
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * jday)
    dec = 0.409 * np.sin(2 * np.pi / 365 * jday - 1.39)
    ws = np.arccos(-np.tan(lat_rad) * np.tan(dec))
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (ws * np.sin(lat_rad) * np.sin(dec) + np.cos(lat_rad) * np.cos(dec) * np.sin(ws))
    ra_mm = ra / 2.45
    tmean = (tmax + tmin) / 2.0
    trange = np.maximum(tmax - tmin, 0)
    return np.maximum(0.0023 * ra_mm * (tmean + 17.8) * np.sqrt(trange), 0)

def balance_hidrico_superficial(prec, et0, w_max=20.0, ke_suelo=0.4):
    n = len(prec)
    w = np.zeros(n)
    w[0] = w_max / 2.0 
    for i in range(1, n):
        evaporacion_real = et0[i] * ke_suelo
        w[i] = max(0.0, min(w_max, w[i-1] + prec[i] - evaporacion_real))
    return w

class PracticalANNModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW, self.bIW, self.LW, self.bLW = IW, bIW, LW, bLW
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])
    def normalize(self, X): return 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1
    def predict(self, Xreal):
        Xn = self.normalize(Xreal)
        a1 = np.tanh(Xn @ self.IW + self.bIW)
        emerrel = (np.tanh((a1 @ self.LW.T).flatten() + self.bLW) + 1) / 2
        return emerrel, np.cumsum(emerrel)

# ---------------------------------------------------------
# 3. LÓGICA DE VALIDACIÓN (AJUSTADA A VALORES > 0)
# ---------------------------------------------------------
def sincronizar_series_por_intervalos(df_sim, df_campo, col_fecha, col_plm2):
    df_sync = df_campo.copy()
    total_campo = df_sync[col_plm2].sum()
    df_sync['Campo_Relativo'] = df_sync[col_plm2] / total_campo if total_campo > 0 else 0
    
    sim_acumulada_intervalos = []
    fecha_anterior = df_sim["Fecha"].min() - pd.Timedelta(days=1)
    
    for _, row in df_sync.iterrows():
        fecha_actual = row[col_fecha]
        mask_ventana = (df_sim["Fecha"] > fecha_anterior) & (df_sim["Fecha"] <= fecha_actual)
        sim_acumulada_intervalos.append(df_sim.loc[mask_ventana, "EMERREL"].sum())
        fecha_anterior = fecha_actual
        
    df_sync['Simulado_Intervalo'] = sim_acumulada_intervalos
    total_sim = df_sync['Simulado_Intervalo'].sum()
    df_sync['Sim_Relativo'] = df_sync['Simulado_Intervalo'] / total_sim if total_sim > 0 else 0.0
    df_sync['Campo_Acumulado'] = df_sync['Campo_Relativo'].cumsum()
    df_sync['Sim_Acumulado'] = df_sync['Sim_Relativo'].cumsum()
    return df_sync

def calcular_metricas_validacion_integral(df_sync):
    # FILTRO: Solo considerar donde hubo capturas reales para correlación de flujos
    df_pos = df_sync[df_sync['Campo_Relativo'] > 0].copy()
    
    if len(df_pos) < 2:
        pearson_r = 0.0
    else:
        pearson_r = np.corrcoef(df_pos['Campo_Relativo'], df_pos['Sim_Relativo'])[0, 1]
    
    # Trayectoria completa para RMSE y CCC
    obs_acum, sim_acum = df_sync['Campo_Acumulado'].values, df_sync['Sim_Acumulado'].values
    rmse_acumulado = np.sqrt(np.mean((obs_acum - sim_acum)**2))
    
    mean_obs, mean_sim = np.mean(obs_acum), np.mean(sim_acum)
    var_obs, var_sim = np.var(obs_acum), np.var(sim_acum)
    covar = np.mean((obs_acum - mean_obs) * (sim_acum - mean_sim))
    ccc_acumulado = (2 * covar) / (var_obs + var_sim + (mean_obs - mean_sim)**2) if (var_obs + var_sim) > 0 else 0.0
    
    return {"Pearson_Flujos": pearson_r, "RMSE_Acumulado": rmse_acumulado, "CCC_Acumulado": ccc_acumulado}

# ---------------------------------------------------------
# 4. CARGA DE MODELOS Y DATOS
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        ann = PracticalANNModel(np.load(BASE / "IW.npy"), np.load(BASE / "bias_IW.npy"), np.load(BASE / "LW.npy"), np.load(BASE / "bias_out.npy"))
        with open(BASE / "modelo_clusters_k3.pkl", "rb") as f: k3 = pickle.load(f)
        return ann, k3
    except: return None, None

def load_data(file_uploader, default_name):
    if file_uploader:
        return pd.read_excel(file_uploader) if file_uploader.name.endswith((".xlsx", ".xls")) else pd.read_csv(file_uploader)
    return None

# ---------------------------------------------------------
# 5. INTERFAZ Y SIDEBAR
# ---------------------------------------------------------
modelo_ann, cluster_model = load_models()
st.title("🌾 PREDWEEM LOLIUM - TRES ARROYOS 2026")

with st.expander("📂 1. Datos del Lote y Manejo", expanded=True):
    col_u1, col_u2 = st.columns(2)
    archivo_meteo = col_u1.file_uploader("1. Clima (Requerido)", type=["xlsx", "csv"])
    archivo_campo = col_u1.file_uploader("2. Campo (Opcional)", type=["xlsx", "csv"])
    cobertura_pct = col_u2.slider("Cobertura de Rastrojo (%)", 0, 100, 50, 5)
    
    ke_val = float(np.interp(cobertura_pct, [0, 100], [0.95, 0.10]))
    mod_termico = float(np.interp(cobertura_pct, [0, 100], [1.00, 0.80]))

# Sidebar params (resumidos para el script)
t_base_val = st.sidebar.number_input("T Base", 0.0, 10.0, 2.0)
t_opt_max = st.sidebar.number_input("T Opt", 10.0, 30.0, 20.0)
t_critica = st.sidebar.number_input("T Crit", 25.0, 45.0, 30.0)
w_max_val = st.sidebar.number_input("Cap. Campo (mm)", 5.0, 50.0, 15.0)
umbral_er = st.sidebar.slider("Umbral Alerta", 0.1, 0.9, 0.5)
dga_optimo = st.sidebar.number_input("TT Control (°Cd)", 100, 1500, 600)

# ---------------------------------------------------------
# 6. MOTOR DE CÁLCULO Y VISUALIZACIÓN
# ---------------------------------------------------------
if archivo_meteo is not None and modelo_ann is not None:
    df = load_data(archivo_meteo, "meteo")
    df.columns = [c.upper().strip() for c in df.columns]
    df = df.rename(columns={'FECHA':'Fecha', 'TMAX':'TMAX', 'TMIN':'TMIN', 'PREC':'Prec', 'LLUVIA':'Prec'})
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df.sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear

    # Cálculos físicos
    df["Tmedia"] = (df["TMAX"] + df["TMIN"]) / 2
    df["TMAX_suelo"] = df["Tmedia"] + ((df["TMAX"] - df["TMIN"]) / 2 * mod_termico)
    df["TMIN_suelo"] = df["Tmedia"] - ((df["TMAX"] - df["TMIN"]) / 2 * mod_termico)
    
    X = df[["Julian_days", "TMAX_suelo", "TMIN_suelo", "Prec"]].to_numpy(float)
    df["EMERREL"], _ = modelo_ann.predict(X)
    
    # Balance Hídrico
    df["ET0"] = calcular_et0_hargreaves(df["Julian_days"].values, df["TMAX"].values, df["TMIN"].values)
    df["W_superficial"] = balance_hidrico_superficial(df["Prec"].values, df["ET0"].values, w_max=w_max_val, ke_suelo=ke_val)
    df["EMERREL"] = df["EMERREL"] * (df["W_superficial"] / w_max_val)
    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, t_base_val, t_opt_max, t_critica))

    # Validación con Campo
    df_campo = None
    if archivo_campo:
        df_campo = load_data(archivo_campo, "campo")
        col_f = df_campo.columns[0]; col_p = df_campo.columns[1]
        df_campo[col_f] = pd.to_datetime(df_campo[col_f])
        df_sync = sincronizar_series_por_intervalos(df, df_campo, col_f, col_p)
        met = calcular_metricas_validacion_integral(df_sync)

        # UI METRICS
        st.markdown("<p class='metric-header'>📊 VALIDACIÓN (Solo valores > 0)</p>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Pearson (Sincronía)", f"{met['Pearson_Flujos']:.3f}")
        c2.metric("CCC (Trayectoria)", f"{met['CCC_Acumulado']:.3f}")
        c3.metric("RMSE", f"{met['RMSE_Acumulado']:.3f}", delta_color="inverse")

        # TABS
        t1, t2 = st.tabs(["📈 Dinámica", "🎯 Dispersión 1:1"])
        
        with t1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["Fecha"], y=df["EMERREL"], name="Simulado", line=dict(color="green")))
            fig.add_trace(go.Scatter(x=df_sync[col_f], y=df_sync["Campo_Relativo"], name="Campo", mode="markers+lines", marker=dict(color="red")))
            st.plotly_chart(fig, use_container_width=True)

        with t2:
            # Match estricto en gráfico
            df_pos = df_sync[df_sync['Campo_Relativo'] > 0]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='1:1', line=dict(color='gray', dash='dash')))
            fig2.add_trace(go.Scatter(x=df_pos['Campo_Relativo'], y=df_pos['Sim_Relativo'], mode='markers', 
                                     marker=dict(size=12, color='blue'), text=df_pos[col_f]))
            fig2.update_layout(title="Ajuste de Intensidad (Valores > 0)", xaxis_title="Obs", yaxis_title="Sim")
            st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("Cargue archivos para iniciar simulación.")
