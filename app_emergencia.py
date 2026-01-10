# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io
from pathlib import Path

# ---------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILO (SEGURIDAD DE INTERFAZ)
# ---------------------------------------------------------
st.set_page_config(
    page_title="PREDWEEM – LOLIUM TRES ARROYOS 2026", 
    layout="wide",
    page_icon="🌾"
)

# Inyección de CSS para Personalización y Bloqueo de Menús "View Source"
st.markdown("""
<style>
    /* BLOQUEO DE INTERFAZ GITHUB / STREAMLIT */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stAppDeployButton {display:none;}
    [data-testid="stStatusWidget"] {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
    
    /* ESTILO VISUAL PREDWEEM */
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
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ---------------------------------------------------------
# 2. CLASE DEL MODELO NEURONAL (ANN)
# ---------------------------------------------------------
class PracticalANNModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW = IW
        self.bIW = bIW
        self.LW = LW
        self.bLW = bLW
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
            np.load(BASE/"IW.npy"), 
            np.load(BASE/"bias_IW.npy"),
            np.load(BASE/"LW.npy"), 
            np.load(BASE/"bias_out.npy")
        )
        return ann
    except Exception as e:
        st.error(f"Error crítico cargando archivos del modelo: {e}")
        return None

def get_data(file_input):
    try:
        if file_input is not None:
            if file_input.name.endswith('.csv'):
                df = pd.read_csv(file_input, parse_dates=["Fecha"]) 
            else:
                df = pd.read_excel(file_input, parse_dates=["Fecha"])
        else:
            path_github = BASE / "meteo_daily.csv"
            if path_github.exists():
                df = pd.read_csv(path_github, parse_dates=["Fecha"])
            else: 
                return None
        
        df.columns = [c.upper().strip() for c in df.columns]
        mapeo = {'FECHA': 'Fecha', 'DATE': 'Fecha', 'TMAX': 'TMAX', 'TMIN': 'TMIN', 'PREC': 'Prec', 'LLUVIA': 'Prec'}
        df = df.rename(columns=mapeo)
        
        required_cols = ["Fecha", "TMAX", "TMIN", "Prec"]
        if not all(col in df.columns for col in required_cols):
            st.error(f"El archivo debe contener: {required_cols}")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error procesando datos: {e}")
        return None

# ---------------------------------------------------------
# 3. INTERFAZ Y PROCESAMIENTO
# ---------------------------------------------------------
modelo_ann = load_models()

# LOGO Y SIDEBAR
LOGO_URL = "https://raw.githubusercontent.com/PREDWEEM/loliumTA_2026/main/logo.png"
st.sidebar.image(LOGO_URL, use_container_width=True)
st.sidebar.markdown("## 🌾 PREDWEEM")
st.sidebar.caption("vK3 | Tres Arroyos 2026")

archivo_usuario = st.sidebar.file_uploader("Subir Clima Manual (Opcional)", type=["xlsx", "csv"])
df = get_data(archivo_usuario)

st.sidebar.divider()
st.sidebar.markdown("**Parámetros de Simulación**")
umbral_er = st.sidebar.slider("Umbral de Alerta (Emergencia Diaria)", 0.05, 0.80, 0.50)
dga_optimo = st.sidebar.slider("Umbral Térmico Óptimo (°Cd)", 50, 800, 600)
dga_critico = st.sidebar.slider("Umbral Térmico Crítico (°Cd)", 600, 1200, 850)

if df is not None and modelo_ann is not None:
    # 1. Procesamiento
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    
    # 2. Predicción
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    emerrel, _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(emerrel, 0.0)
    df.loc[df["Julian_days"] <= 30, "EMERREL"] = 0.0
    df["DG"] = np.maximum(((df["TMAX"] + df["TMIN"]) / 2) - 2.0, 0)

    st.title("🌾 PREDWEEM | LOLIUM TRES ARROYOS 2026")

    # VISUALIZACIÓN A: HEATMAP
    colorscale_hard = [[0.0, "green"], [0.49, "green"], [0.49, "yellow"], [0.90, "yellow"], [0.90, "red"], [1.0, "red"]]
    fig_risk = go.Figure(data=go.Heatmap(
        z=[df["EMERREL"].values], x=df["Fecha"], y=["Emergencia"],
        colorscale=colorscale_hard, zmin=0, zmax=1, showscale=False,
        hovertemplate="<b>%{x|%d-%b}</b><br>Tasa: %{z:.3f}<extra></extra>"
    ))
    fig_risk.update_layout(height=130, margin=dict(t=30, b=0, l=10, r=10), title="Mapa de Intensidad: Emergencia Relativa")
    st.plotly_chart(fig_risk, use_container_width=True)

    # VISUALIZACIÓN B: SERIE TIEMPO
    fig_emer = go.Figure()
    fig_emer.add_trace(go.Scatter(x=df["Fecha"], y=df["EMERREL"], mode='lines', name='Emergencia', line=dict(color='#166534', width=2.5), fill='tozeroy'))
    fig_emer.add_hline(y=umbral_er, line_dash="dash", line_color="orange", annotation_text="Umbral Alerta")
    fig_emer.update_layout(title="Dinámica de Emergencia Relativa", height=350)
    st.plotly_chart(fig_emer, use_container_width=True)

    # MONITOR SEMÁFORO
    indices_pulso = df.index[df["EMERREL"] >= umbral_er].tolist()
    fecha_inicio_ventana = None
    for i in range(len(indices_pulso) - 1):
        if (df.loc[indices_pulso[i+1], "Fecha"] - df.loc[indices_pulso[i], "Fecha"]).days <= 5:
            fecha_inicio_ventana = df.loc[indices_pulso[i], "Fecha"]
            break

    dga_actual_acumulado = 0.0
    if fecha_inicio_ventana:
        df_ventana = df[df["Fecha"] >= fecha_inicio_ventana].copy()
        df_ventana["DGA_cum"] = df_ventana["DG"].cumsum()
        dga_actual_acumulado = df_ventana["DGA_cum"].iloc[-1]

    st.divider()
    st.header("🗓️ Monitor de Ventana de Aplicación")
    col_info, col_gauge = st.columns([1.5, 1])

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta", value = dga_actual_acumulado,
            title = {'text': "<b>ACUMULACIÓN TÉRMICA</b><br><span style='font-size:0.8em'>Grados Días (°Cd)</span>"},
            delta = {'reference': dga_optimo},
            gauge = {
                'axis': {'range': [None, dga_critico*1.2]},
                'bar': {'color': "black", 'thickness': 0.05},
                'steps': [
                    {'range': [0, dga_optimo], 'color': "#4ade80"},
                    {'range': [dga_optimo, dga_critico], 'color': "#facc15"},
                    {'range': [dga_critico, dga_critico*1.5], 'color': "#f87171"}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'value': dga_actual_acumulado}
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_info:
        if fecha_inicio_ventana:
            def obtener_estado(obj):
                if dga_actual_acumulado >= obj:
                    return df_ventana[df_ventana["DGA_cum"] >= obj].iloc[0]["Fecha"].strftime("%d-%m-%Y"), "PASADO"
                return "Proyección Futura", "PENDIENTE"
            f_opt, status_opt = obtener_estado(dga_optimo)
            f_cri, status_cri = obtener_estado(dga_critico)

            st.metric("Inicio de Cohorte", fecha_inicio_ventana.strftime("%d-%b"))
            datos_tabla = {
                "Zona": ["🟢 VERDE", "🟡 AMARILLA", "🔴 ROJA"],
                "Fase": ["Ventana Óptima", "Ventana Crítica", "Fuera de Ventana"],
                "Situación": [
                    "✅ ACTIVO" if dga_actual_acumulado <= dga_optimo else "",
                    "⚠️ ACTIVO" if dga_optimo < dga_actual_acumulado <= dga_critico else "",
                    "🚫 ACTIVO" if dga_actual_acumulado > dga_critico else ""
                ]
            }
            st.table(pd.DataFrame(datos_tabla))
            if status_opt == "PENDIENTE": st.success("✅ CONDICIÓN IDEAL")
            elif status_cri == "PENDIENTE": st.warning("⚠️ ATENCIÓN: VENTANA CRÍTICA")
            else: st.error("🚫 ALERTA ROJA: LÍMITE SUPERADO")
        else:
            st.info(f"⏳ Sistema en Espera de Pulso (>= {umbral_er*100:.0f}%)")

    # EXPORTACIÓN
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    st.sidebar.download_button(label="📥 Descargar Resultados", data=output.getvalue(), file_name="PREDWEEM_2026.xlsx")
else:
    st.warning("⚠️ Esperando datos (meteo_daily.csv).")
