# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io
from pathlib import Path

# ---------------------------------------------------------
# 1. CONFIGURACIÓN, ESTILO Y LOGO
# ---------------------------------------------------------
st.set_page_config(page_title="PREDWEEM – LOLIUM TRES ARROYOS 2026", layout="wide")

# Dirección proporcionada (convertida a formato raw para Streamlit)
LOGO_URL = "https://raw.githubusercontent.com/PREDWEEM/loliumTA_2026/main/logo.png"

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
    }
</style>
""", unsafe_allow_html=True)

# --- LOGO EN LA PARTE SUPERIOR ---
# En el Sidebar
st.sidebar.image(LOGO_URL, use_container_width=True)

# En el cuerpo principal (Centrado)
col_l, col_c, col_r = st.columns([1, 2, 1])
with col_c:
    st.image(LOGO_URL, use_container_width=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ---------------------------------------------------------
# 2. MODELOS Y FUNCIONES TÉCNICAS
# ---------------------------------------------------------
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
        emer = (np.array(emer) + 1) / 2
        emer_ac = np.cumsum(emer)
        emerrel = np.diff(emer_ac, prepend=0)
        return emerrel, emer_ac

@st.cache_resource
def load_models():
    try:
        ann = PracticalANNModel(
            np.load(BASE/"IW.npy"), np.load(BASE/"bias_IW.npy"),
            np.load(BASE/"LW.npy"), np.load(BASE/"bias_out.npy")
        )
        return ann
    except Exception as e:
        st.error(f"Error cargando archivos de modelo: {e}")
        return None

def get_data(file_input):
    try:
        if file_input is not None:
            df = pd.read_csv(file_input, parse_dates=["Fecha"]) if file_input.name.endswith('.csv') else pd.read_excel(file_input, parse_dates=["Fecha"])
        else:
            path_github = BASE / "meteo_daily.csv"
            if path_github.exists():
                df = pd.read_csv(path_github, parse_dates=["Fecha"])
            else: return None
        
        df.columns = [c.upper().strip() for c in df.columns]
        mapeo = {'FECHA': 'Fecha', 'DATE': 'Fecha', 'TMAX': 'TMAX', 'TMIN': 'TMIN', 'PREC': 'Prec', 'LLUVIA': 'Prec'}
        df = df.rename(columns=mapeo)
        return df
    except Exception as e:
        st.error(f"Error en datos: {e}"); return None

# ---------------------------------------------------------
# 3. INTERFAZ Y PROCESAMIENTO
# ---------------------------------------------------------
modelo_ann = load_models()

st.sidebar.markdown("### PANEL DE CONTROL")
df = get_data(st.sidebar.file_uploader("Subir Clima Manual (Opcional)", type=["xlsx", "csv"]))

st.sidebar.divider()
umbral_er = st.sidebar.slider("Sensibilidad de Detección", 0.05, 0.80, 0.45)
dga_optimo = st.sidebar.slider("Umbral Óptimo (°Cd)", 50, 800, 600)
dga_critico = st.sidebar.slider("Umbral Crítico (°Cd)", 600, 1200, 850)

if df is not None and modelo_ann is not None:
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    emerrel, _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(emerrel, 0.0)
    df.loc[df["Julian_days"] <= 15, "EMERREL"] = 0.0
    
    df["DG"] = np.maximum(((df["TMAX"] + df["TMIN"]) / 2) - 2.0, 0) 
    max_er = df["EMERREL"].max()
    df["Riesgo"] = df["EMERREL"] / max_er if max_er > 0 else 0.0

    st.title("🌾 PREDWEEM | LOLIUM TRES ARROYOS 2026")

    # --- VISUALIZACIONES ---
    fig_risk = go.Figure(data=go.Heatmap(
        z=[df["Riesgo"].values], x=df["Fecha"], y=["Riesgo"],
        colorscale=[[0, 'green'], [0.5, 'yellow'], [1, 'red']],
        zmin=0, zmax=1, showscale=False))
    st.plotly_chart(fig_risk, use_container_width=True)

    fig_emer = go.Figure()
    fig_emer.add_trace(go.Scatter(x=df["Fecha"], y=df["EMERREL"], mode='lines', fill='tozeroy'))
    st.plotly_chart(fig_emer, use_container_width=True)

    # --- DESCARGA ---
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    st.sidebar.download_button("📥 Descargar Reporte Profesional", output.getvalue(), "PREDWEEM_2026.xlsx")

else:
    st.warning("⚠️ Esperando datos para procesar la simulación.")

st.sidebar.caption("PREDWEEM vK3 | Tres Arroyos 2026")
