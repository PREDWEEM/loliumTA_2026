# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4.4 — LOLIUM TRES ARROYOS 2026
# Versión Final Corregida: Soporte KGE y Estabilidad de Dimensiones
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
    .metric-header { color: #1e293b; font-weight: bold; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ---------------------------------------------------------
# 2. MOCKS DE SEGURIDAD (Si faltan archivos .npy/.pkl)
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
# 3. CLASES Y MÉTODOS TÉCNICOS
# ---------------------------------------------------------
class PracticalANNModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW, self.bIW, self.LW, self.bLW = IW, bIW, LW, bLW
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])

    def normalize(self, X):
        return 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1

    def predict(self, Xreal):
        # Aseguramos que Xreal sea float y 2D
        Xn = self.normalize(Xreal)
        emer = []
        for x in Xn:
            z1 = self.IW.T @ x + self.bIW
            a1 = np.tanh(z1)
            z2 = self.LW @ a1 + self.bLW
            emer.append(np.tanh(z2))
        # Retornamos vector 1D exacto para Pandas
        return (np.array(emer).flatten() + 1) / 2

def calculate_kge(sim, obs):
    if len(obs) < 2 or np.all(obs == 0): return 0.0
    r = np.corrcoef(sim, obs)[0, 1]
    if np.isnan(r): r = 0.0
    beta = np.mean(sim) / (np.mean(obs) + 1e-6)
    cv_sim = np.std(sim) / (np.mean(sim) + 1e-6)
    cv_obs = np.std(obs) / (np.mean(obs) + 1e-6)
    gamma = cv_sim / (cv_obs + 1e-6)
    return 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)

def calculate_tt_scalar(t, t_base, t_opt, t_crit):
    if t <= t_base: return 0.0
    elif t <= t_opt: return t - t_base
    elif t < t_crit: return (t - t_base) * ((t_crit - t) / (t_crit - t_opt))
    return 0.0

@st.cache_resource
def load_models():
    try:
        ann = PracticalANNModel(np.load(BASE/"IW.npy"), np.load(BASE/"bias_IW.npy"), np.load(BASE/"LW.npy"), np.load(BASE/"bias_out.npy"))
        with open(BASE/"modelo_clusters_k3.pkl", "rb") as f:
            k3 = pickle.load(f)
        return ann, k3
    except: return None, None

# ---------------------------------------------------------
# 4. SIDEBAR Y CARGA
# ---------------------------------------------------------
modelo_ann, cluster_model = load_models()

st.sidebar.markdown("## 📂 Datos del Lote")
archivo_meteo = st.sidebar.file_uploader("1. Clima (Excel/CSV)", type=["xlsx", "csv"])
archivo_campo = st.sidebar.file_uploader("2. Campo (Validación)", type=["xlsx", "csv"])

# Parámetros Fisiológicos
st.sidebar.divider()
umbral_er = st.sidebar.slider("Umbral Alerta", 0.05, 0.80, 0.15)
dga_optimo = st.sidebar.number_input("TT Control (°Cd)", value=600)
t_base_val = st.sidebar.number_input("T Base", value=2.0)
t_opt_max = st.sidebar.number_input("T Optima", value=20.0)
t_critica = st.sidebar.slider("T Critica", 26.0, 42.0, 30.0)

# ---------------------------------------------------------
# 5. PROCESAMIENTO PRINCIPAL (CORRECCIÓN DIMENSIONES)
# ---------------------------------------------------------
if archivo_meteo is not None and modelo_ann is not None:
    # Carga y limpieza estricta
    df = pd.read_excel(archivo_meteo) if archivo_meteo.name.endswith('xlsx') else pd.read_csv(archivo_meteo)
    df.columns = [c.upper().strip() for c in df.columns]
    df = df.rename(columns={'FECHA': 'Fecha', 'TMAX': 'TMAX', 'TMIN': 'TMIN', 'PREC': 'Prec', 'LLUVIA': 'Prec'})
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    # ELIMINACIÓN DE NA Y RESET DE ÍNDICE (Clave para evitar el ValueError)
    df = df.dropna(subset=["TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    
    # Construcción de X y Predicción
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].values.astype(float)
    y_pred = modelo_ann.predict(X)
    
    # Asignación segura
    df["EMERREL"] = np.maximum(y_pred, 0.0)
    
    # Factores adicionales
    df["Prec_sum_21d"] = df["Prec"].rolling(window=21, min_periods=1).sum()
    df["Hydric_Factor"] = 1 / (1 + np.exp(-0.4 * (df["Prec_sum_21d"] - 15)))
    df["EMERREL"] *= df["Hydric_Factor"]
    
    # Cálculo Térmico
    df["Tmedia"] = (df["TMAX"] + df["TMIN"]) / 2
    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, t_base_val, t_opt_max, t_critica))

    # --- VALIDACIÓN DE CAMPO ---
    kge_score, pearson_r, pec = 0.0, 0.0, 0.0
    fecha_control = None
    
    if archivo_campo is not None:
        df_campo = pd.read_excel(archivo_campo) if archivo_campo.name.endswith('xlsx') else pd.read_csv(archivo_campo)
        df_campo.columns = [c.upper().strip() for c in df_campo.columns]
        col_fecha = 'FECHA' if 'FECHA' in df_campo.columns else df_campo.columns[0]
        col_plm2 = 'PLM2' if 'PLM2' in df_campo.columns else df_campo.columns[1]
        df_campo[col_fecha] = pd.to_datetime(df_campo[col_fecha])
        df_campo = df_campo.sort_values(col_fecha).reset_index(drop=True)
        
        # Sincronía por intervalos
        sim_intervals = []
        last_date = df['Fecha'].min() - pd.Timedelta(days=1)
        for _, row in df_campo.iterrows():
            mask = (df['Fecha'] > last_date) & (df['Fecha'] <= row[col_fecha])
            sim_intervals.append(df.loc[mask, 'EMERREL'].sum())
            last_date = row[col_fecha]
        
        df_campo['Sim_Intervalo'] = sim_intervals
        
        # MÉTRICAS
        pearson_r = df_campo[col_plm2].corr(df_campo['Sim_Intervalo'])
        kge_score = calculate_kge(df_campo['Sim_Intervalo'].values, df_campo[col_plm2].values)

        # Cálculo de Momento de Control
        indices_pulso = df.index[df["EMERREL"] >= umbral_er].tolist()
        if indices_pulso:
            f_ini = df.loc[indices_pulso[0], "Fecha"]
            df_v = df[df["Fecha"] >= f_ini].copy()
            df_v["DGA_cum"] = df_v["DG"].cumsum()
            df_target = df_v[df_v["DGA_cum"] >= dga_optimo]
            if not df_target.empty:
                fecha_control = df_target.iloc[0]["Fecha"]
                m_total = df_campo[col_plm2].sum()
                m_ctrl = df_campo.loc[df_campo[col_fecha] <= fecha_control, col_plm2].sum()
                pec = (m_ctrl / m_total * 100) if m_total > 0 else 0

    # -----------------------------------------------------
    # 6. FRONT-END
    # -----------------------------------------------------
    st.title("🌾 PREDWEEM LOLIUM - TRES ARROYOS 2026")
    
    if archivo_campo is not None:
        st.markdown("<p class='metric-header'>🚜 DIAGNÓSTICO DE PRECISIÓN (KGE)</p>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("KGE Score (Eficiencia)", f"{kge_score:.3f}", "Ajuste Sim vs Obs")
        c2.metric("Control Efectivo (PEC)", f"{pec:.1f}%", "Plantas bajo ventana")
        c3.metric("Pearson (r)", f"{pearson_r:.2f}", "Sincronía Temporal")
        st.divider()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Fecha"], y=df["EMERREL"], name="Simulación", fill='tozeroy', line=dict(color='green')))
    if archivo_campo is not None:
        norm_campo = df_campo[col_plm2] / (df_campo[col_plm2].max() + 1e-6)
        fig.add_trace(go.Scatter(x=df_campo[col_fecha], y=norm_campo, name="Campo (Normalizado)", mode='markers+lines', marker=dict(color='red')))
    
    if fecha_control:
        fig.add_vline(x=fecha_control.timestamp()*1000, line_dash="dash", line_color="orange", annotation_text="Momento Control")

    fig.update_layout(title="Dinámica de Emergencia", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Exportación
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Simulacion')
    st.sidebar.download_button("📥 Reporte Excel", output.getvalue(), "PREDWEEM_Lartigau.xlsx")

else:
    st.info("👋 Por favor, cargue un archivo de clima para iniciar la simulación.")
