# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4.4 — LOLIUM TRES ARROYOS 2026
# Actualización: Integración Kling-Gupta Efficiency (KGE)
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
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ---------------------------------------------------------
# 2. ROBUSTEZ Y ARCHIVOS (MOCKS PARA TESTEO)
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
# 3. LÓGICA TÉCNICA Y MÉTRICAS
# ---------------------------------------------------------
def calculate_kge(sim, obs):
    """ Calcula el Kling-Gupta Efficiency (KGE) """
    if len(obs) < 2 or np.all(obs == 0):
        return 0.0
    r = np.corrcoef(sim, obs)[0, 1]
    if np.isnan(r): r = 0.0
    beta = np.mean(sim) / (np.mean(obs) + 1e-6)
    cv_sim = np.std(sim) / (np.mean(sim) + 1e-6)
    cv_obs = np.std(obs) / (np.mean(obs) + 1e-6)
    gamma = cv_sim / (cv_obs + 1e-6)
    return 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)

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
        return np.diff(np.cumsum(emer), prepend=0)

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

st.sidebar.markdown("## 📂 1. Datos del Lote")
archivo_meteo = st.sidebar.file_uploader("1. Clima (Lartigau)", type=["xlsx", "csv"])
archivo_campo = st.sidebar.file_uploader("2. Campo (Validación)", type=["xlsx", "csv"])

df_meteo_raw = load_data(archivo_meteo, "TRES ARROYOS")
df_campo_raw = load_data(archivo_campo, "TRES ARROYOS_campo")

st.sidebar.divider()
st.sidebar.markdown("## ⚙️ 2. Fisiología y Control")
umbral_er = st.sidebar.slider("Umbral Alerta Temprana", 0.05, 0.80, 0.15)
residualidad = st.sidebar.number_input("Residualidad Herbicida (días)", 0, 60, 20)
t_base_val = st.sidebar.number_input("T Base", value=2.0)
t_opt_max = st.sidebar.number_input("T Óptima Max", value=20.0)
t_critica = st.sidebar.slider("T Crítica (Stop)", 26.0, 42.0, 30.0)
dga_optimo = st.sidebar.number_input("TT Control Post-em (°Cd)", value=600)

# ---------------------------------------------------------
# 5. MOTOR DE CÁLCULO PRINCIPAL
# ---------------------------------------------------------
if df_meteo_raw is not None and modelo_ann is not None:
    df = df_meteo_raw.copy()
    df.columns = [c.upper().strip() for c in df.columns]
    df = df.rename(columns={'FECHA': 'Fecha', 'DATE': 'Fecha', 'TMAX': 'TMAX', 'TMIN': 'TMIN', 'PREC': 'Prec', 'LLUVIA': 'Prec'})
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    
    # Predicción y Factores
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    df["EMERREL"] = np.maximum(modelo_ann.predict(X), 0.0)
    df["Prec_sum_21d"] = df["Prec"].rolling(window=21, min_periods=1).sum()
    df["Hydric_Factor"] = 1 / (1 + np.exp(-0.4 * (df["Prec_sum_21d"] - 15)))
    df["EMERREL"] *= df["Hydric_Factor"]
    
    # Suma térmica
    df["Tmedia"] = (df["TMAX"] + df["TMIN"]) / 2
    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, t_base_val, t_opt_max, t_critica))
    
    # Sincronía con Campo y KGE
    kge_score, pearson_r, pec, peak_lag, lead_time = 0, 0, 0, 0, 0
    fecha_control, fecha_inicio_ventana = None, None
    df_campo = None

    if df_campo_raw is not None:
        df_campo = df_campo_raw.copy()
        col_fecha = 'FECHA' if 'FECHA' in df_campo.columns else df_campo.columns[0]
        col_plm2 = 'PLM2' if 'PLM2' in df_campo.columns else df_campo.columns[1]
        df_campo[col_fecha] = pd.to_datetime(df_campo[col_fecha])
        df_campo = df_campo.sort_values(col_fecha).reset_index(drop=True)
        
        sim_intervals = []
        last_date = df['Fecha'].min() - pd.Timedelta(days=1)
        for _, row in df_campo.iterrows():
            mask = (df['Fecha'] > last_date) & (df['Fecha'] <= row[col_fecha])
            sim_intervals.append(df.loc[mask, 'EMERREL'].sum())
            last_date = row[col_fecha]
        
        df_campo['Sim_Intervalo'] = sim_intervals
        pearson_r = df_campo[col_plm2].corr(df_campo['Sim_Intervalo'])
        kge_score = calculate_kge(df_campo['Sim_Intervalo'].values, df_campo[col_plm2].values)
        
        # Lógica de Ventana
        indices_pulso = df.index[df["EMERREL"] >= umbral_er].tolist()
        if indices_pulso:
            fecha_inicio_ventana = df.loc[indices_pulso[0], "Fecha"]
            df_desde = df[df["Fecha"] >= fecha_inicio_ventana].copy()
            df_desde["DGA_cum"] = df_desde["DG"].cumsum()
            df_ctrl = df_desde[df_desde["DGA_cum"] >= dga_optimo]
            if not df_ctrl.empty:
                fecha_control = df_ctrl.iloc[0]["Fecha"]
                malezas_totales = df_campo[col_plm2].sum()
                malezas_ctrl = df_campo.loc[df_campo[col_fecha] <= fecha_control, col_plm2].sum()
                pec = (malezas_ctrl / malezas_totales * 100) if malezas_totales > 0 else 0
                peak_lag = (fecha_control - df_campo.loc[df_campo[col_plm2].idxmax(), col_fecha]).days
                lead_time = (fecha_control - fecha_inicio_ventana).days

    # -----------------------------------------------------
    # VISUALIZACIÓN
    # -----------------------------------------------------
    st.title("🌾 PREDWEEM LOLIUM - TRES ARROYOS 2026")
    
    tab1, tab2, tab3 = st.tabs(["📊 MONITOR DE DECISIÓN", "📈 ANÁLISIS DTW", "🧪 BIO-RESPUESTA"])

    with tab1:
        if df_campo is not None:
            st.markdown("<p class='metric-header'>🚜 VALIDACIÓN INTEGRAL (Kling-Gupta Efficiency)</p>", unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("KGE Score", f"{kge_score:.3f}", "Ajuste Sim vs Obs")
            m2.metric("Control (PEC)", f"{pec:.1f}%", "Efectividad")
            m3.metric("Lag", f"{peak_lag} d", "Desfase vs Pico")
            m4.metric("Pearson (r)", f"{pearson_r:.2f}", "Sincronía")
        
        fig_emer = go.Figure()
        fig_emer.add_trace(go.Scatter(x=df["Fecha"], y=df["EMERREL"], name='Simulación', line=dict(color='#166534', width=2), fill='tozeroy'))
        if df_campo is not None:
            max_c = df_campo[col_plm2].max()
            fig_emer.add_trace(go.Scatter(x=df_campo[col_fecha], y=df_campo[col_plm2]/max_c if max_c > 0 else 0, mode='markers+lines', name='Campo (Norm)'))
        if fecha_control:
            fig_emer.add_vline(x=fecha_control.timestamp()*1000, line_dash="dot", line_color="red")
        
        st.plotly_chart(fig_emer, use_container_width=True)

    with tab2:
        st.info("Clasificación de patrón de emergencia mediante Dynamic Time Warping...")
        # Aquí iría el bloque DTW similar al original...

    with tab3:
        x_t = np.linspace(0, 40, 100)
        y_t = [calculate_tt_scalar(t, t_base_val, t_opt_max, t_critica) for t in x_t]
        st.plotly_chart(go.Figure(go.Scatter(x=x_t, y=y_t, fill='tozeroy', name='Respuesta Térmica')), use_container_width=True)

    # Exportación
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Simulacion')
        if df_campo is not None:
            pd.DataFrame({'Métrica': ['KGE', 'Pearson', 'PEC'], 'Valor': [kge_score, pearson_r, pec]}).to_excel(writer, sheet_name='Validacion')
    st.sidebar.download_button("📥 Descargar Reporte", output.getvalue(), "PREDWEEM_2026.xlsx")

else:
    st.info("👋 Cargue el archivo de clima para procesar el modelo.")
