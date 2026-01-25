
# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io
from pathlib import Path

# ---------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILO PROFESIONAL (CSS)
# ---------------------------------------------------------
st.set_page_config(
    page_title="PREDWEEM | Decision Support", 
    layout="wide", 
    page_icon="🌾"
)

def apply_custom_style():
    st.markdown("""
    <style>
    /* Importar fuente Inter */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #f8fafc; }

    /* Estilo Sidebar Ejecutivo */
    [data-testid="stSidebar"] { background-color: #064e3b !important; border-right: 1px solid #065f46; }
    [data-testid="stSidebar"] * { color: #ecfdf5 !important; }
    
    /* Tarjetas de Métricas */
    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e2e8f0;
        padding: 20px !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Contenedores de Gráficos */
    .plot-card {
        background-color: white;
        padding: 24px;
        border-radius: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }

    /* Botones Modernos */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        background-image: linear-gradient(to right, #059669, #10b981);
        color: white;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4); }
    
    /* Ocultar elementos nativos */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

apply_custom_style()

# ---------------------------------------------------------
# 2. CLASE DEL MODELO NEURONAL (ANN)
# ---------------------------------------------------------
class PracticalANNModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW, self.bIW, self.LW, self.bLW = IW, bIW, LW, bLW
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])

    def predict(self, Xreal):
        Xn = 2 * (Xreal - self.input_min) / (self.input_max - self.input_min) - 1
        emer = []
        for x in Xn:
            z1 = self.IW.T @ x + self.bIW
            a1 = np.tanh(z1)
            z2 = self.LW @ a1 + self.bLW
            emer.append((np.tanh(z2) + 1) / 2)
        emer = np.array(emer).flatten()
        emer_ac = np.cumsum(emer)
        emerrel = np.diff(emer_ac, prepend=0)
        return emerrel, emer_ac

@st.cache_resource
def load_models():
    BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    try:
        return PracticalANNModel(
            np.load(BASE/"IW.npy"), np.load(BASE/"bias_IW.npy"),
            np.load(BASE/"LW.npy"), np.load(BASE/"bias_out.npy")
        )
    except: return None

# ---------------------------------------------------------
# 3. LÓGICA DE NEGOCIO (BIO-MOTOR)
# ---------------------------------------------------------
def calculate_tt_scalar(t, t_base, t_opt, t_crit):
    if t <= t_base or t >= t_crit: return 0.0
    if t <= t_opt: return t - t_base
    return (t - t_base) * (t_crit - t) / (t_crit - t_opt)

# ---------------------------------------------------------
# 4. INTERFAZ Y PROCESAMIENTO
# ---------------------------------------------------------
model = load_models()

# SIDEBAR
with st.sidebar:
    st.markdown("## 🌾 PREDWEEM")
    st.markdown("### Decision Support System")
    archivo = st.file_uploader("Subir Clima (CSV/Excel)", type=["xlsx", "csv"])
    
    st.divider()
    st.markdown("#### Parámetros Biológicos")
    t_base = st.number_input("T Base (°C)", value=2.0)
    t_opt = st.number_input("T Óptima (°C)", value=25.0)
    t_crit = st.number_input("T Crítica (°C)", value=30.0)
    
    st.divider()
    umbral_er = st.slider("Umbral Alerta Emergencia", 0.1, 0.9, 0.5)
    dga_optimo = st.number_input("Objetivo Control (°Cd)", value=600)
    dga_critico = st.number_input("Límite Crítico (°Cd)", value=700)

# CUERPO PRINCIPAL
if archivo is not None and model is not None:
    # Procesamiento de datos
    df = pd.read_csv(archivo) if archivo.name.endswith('csv') else pd.read_excel(archivo)
    df.columns = [c.upper().strip() for c in df.columns]
    df = df.rename(columns={'FECHA':'Fecha', 'TMAX':'TMAX', 'TMIN':'TMIN', 'PREC':'Prec'})
    df["Julian_days"] = pd.to_datetime(df["Fecha"]).dt.dayofyear
    
    # Predicción
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    emerrel, _ = model.predict(X)
    df["EMERREL"] = np.maximum(emerrel, 0.0)
    
    # Tiempo Térmico
    df["Tmedia"] = (df["TMAX"] + df["TMIN"]) / 2
    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, t_base, t_opt, t_crit))

    # Identificar Ventana
    indices_pulso = df.index[df["EMERREL"] >= umbral_er].tolist()
    dga_actual = 0.0
    if indices_pulso:
        f_inicio = df.loc[indices_pulso[0], "Fecha"]
        df_v = df[df["Fecha"] >= f_inicio].copy()
        df_v["DGA_cum"] = df_v["DG"].cumsum()
        dga_actual = df_v["DGA_cum"].iloc[-1]

    # --- DASHBOARD UI ---
    st.title("🚜 Panel de Control Estratégico")
    
    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Emergencia Actual", f"{df['EMERREL'].iloc[-1]:.2f}")
    c2.metric("Acumulación Bio", f"{dga_actual:.1f} °Cd")
    c3.metric("Riesgo", "ALTO" if dga_actual > dga_optimo else "OPTIMO")

    

    # Gráfico Principal en Card
    st.markdown('<div class="plot-card">', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Fecha"], y=df["EMERREL"], fill='tozeroy', line_color='#059669', name="Emergencia"))
    fig.add_hline(y=umbral_er, line_dash="dash", line_color="orange", annotation_text="Umbral Alerta")
    fig.update_layout(title="Dinámica de Emergencia de Lolium", height=400, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Lógica de Semáforo
    if dga_actual > dga_critico:
        st.error(f"🚫 **VENTANA CERRADA:** Eficacia mínima comprometida. Acumulado: {dga_actual:.1f} °Cd")
    elif dga_actual > dga_optimo:
        st.warning(f"⚠️ **VENTANA CRÍTICA:** Aplicar inmediatamente. Límite óptimo superado.")
    else:
        st.success(f"✅ **VENTANA ÓPTIMA:** Condiciones ideales para el control químico.")

    # Exportación
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    st.sidebar.download_button("📥 Descargar Reporte de Decisión", output.getvalue(), "PREDWEEM_Report.xlsx")

else:
    st.info("👋 Bienvendido. Por favor sube el archivo de clima para iniciar el análisis de Lolium.")
