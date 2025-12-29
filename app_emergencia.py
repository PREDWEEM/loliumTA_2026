# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle, io
from pathlib import Path

# ---------------------------------------------------------
# 1. CONFIGURACIÓN Y ESTILO
# ---------------------------------------------------------
st.set_page_config(page_title="PREDWEEM – LOLIUM TRES ARROYOS 2026", layout="wide")

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

# SIDEBAR
st.sidebar.markdown("## 🌾 PREDWEEM")
st.sidebar.markdown("### LOLIUM TRES ARROYOS 2026")
df = get_data(st.sidebar.file_uploader("Subir Clima Manual (Opcional)", type=["xlsx", "csv"]))

st.sidebar.divider()
umbral_er = st.sidebar.slider("Sensibilidad de Detección", 0.05, 0.80, 0.45)
dga_optimo = st.sidebar.slider("Umbral Óptimo (°Cd)", 50, 800, 600)
dga_critico = st.sidebar.slider("Umbral Crítico (°Cd)", 600, 1200, 850)

if df is not None and modelo_ann is not None:
    # Cálculos Técnicos
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    emerrel, _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(emerrel, 0.0)
    df.loc[df["Julian_days"] <= 15, "EMERREL"] = 0.0
    
    # Grados Día (Base 2.0°C)
    df["DG"] = np.maximum(((df["TMAX"] + df["TMIN"]) / 2) - 2.0, 0) 
    max_er = df["EMERREL"].max()
    df["Riesgo"] = df["EMERREL"] / max_er if max_er > 0 else 0.0

    st.title("🌾 PREDWEEM | LOLIUM TRES ARROYOS 2026")
    st.caption("Validación de ventanas de control basada en acumulación de tiempo térmico.")

    # 1. VISUALIZACIÓN DE RIESGO
    fig_risk = go.Figure(data=go.Heatmap(
        z=[df["Riesgo"].values], x=df["Fecha"], y=["Riesgo"],
        colorscale=[[0, 'green'], [0.5, 'yellow'], [1, 'red']],
        zmin=0, zmax=1, showscale=False))
    fig_risk.update_layout(height=120, margin=dict(t=30, b=0, l=10, r=10), title="Mapa de Calor: Intensidad de Riesgo")
    st.plotly_chart(fig_risk, use_container_width=True)

    # 2. VENTANA DE ACCIÓN (LÓGICA PEDIDA)
    indices_pulso = df.index[df["EMERREL"] >= umbral_er].tolist()
    fecha_inicio_ventana = None
    
    # Detectar inicio de ventana por pulsos sostenidos
    for i in range(len(indices_pulso) - 1):
        if (df.loc[indices_pulso[i+1], "Fecha"] - df.loc[indices_pulso[i], "Fecha"]).days <= 5:
            fecha_inicio_ventana = df.loc[indices_pulso[i], "Fecha"]
            break

    if fecha_inicio_ventana:
        st.divider()
        st.header("🗓️ Cronograma y Fechas Límite")
        
        df_ventana = df[df["Fecha"] >= fecha_inicio_ventana].copy()
        df_ventana["DGA_cum"] = df_ventana["DG"].cumsum()
        dga_actual = df_ventana["DGA_cum"].iloc[-1]

        # REGLA: Solo estimar si se acumuló el tiempo térmico requerido
        def calc_limite_estricto(objetivo):
            if dga_actual >= objetivo:
                fecha_alcanzada = df_ventana[df_ventana["DGA_cum"] >= objetivo]["Fecha"].iloc[0]
                return fecha_alcanzada.strftime("%d-%m-%Y"), "CUMPLIDO"
            else:
                return "Sin dato", "PENDIENTE"

        f_opt, s_opt = calc_limite_estricto(dga_optimo)
        f_cri, s_cri = calc_limite_estricto(dga_critico)

        # Métricas principales
        c1, c2, c3 = st.columns(3)
        c1.metric("Inicio de Ventana", fecha_inicio_ventana.strftime("%d-%b"))
        c2.metric("Acumulado Térmico", f"{dga_actual:.1f} °Cd")
        c3.metric("Límite Óptimo", f_opt if f_opt != "Sin dato" else "---")

        # Tabla de Seguimiento
        df_tabla = pd.DataFrame({
            "Nivel de Alerta": ["🟢 ÓPTIMO", "🟡 LÍMITE CRÍTICO", "🔴 POST-CRÍTICO"],
            "Estado Fenológico": ["EMERGENCIA / PRE-MACOLLAJE", "MACOLLAJE INICIAL", "MACOLLAJE AVANZADO"],
            "Fecha Límite (Térmica)": [f_opt, f_cri, "Control No Recomendado"],
            "Estatus Requisito": [s_opt, s_cri, "N/A"]
        })
        st.table(df_tabla)

        # Alertas dinámicas
        if f_opt == "Sin dato":
            st.info(f"💡 El sistema está esperando a que se acumulen {dga_optimo} °Cd para fijar la fecha límite óptima.")
        elif dga_actual <= dga_critico:
            st.warning(f"⚠️ Atención: Se ha superado el tiempo térmico óptimo ({f_opt}). Ventana de control cerrándose.")
        else:
            st.error(f"❗ Alerta: Se ha superado el tiempo térmico crítico ({f_cri}). El control puede ser ineficiente.")

    else:
        st.info(f"⏳ Monitoreando... No se ha detectado un pulso de emergencia suficiente (Umbral: {umbral_er}) para iniciar el cronograma.")

    # Descarga de Datos
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='PREDWEEM_DATA')
    st.sidebar.download_button("📥 Descargar Datos Procesados", output.getvalue(), "predweem_report.xlsx")

else:
    st.warning("⚠️ Cargue un archivo de clima o asegúrese de que 'meteo_daily.csv' esté en la carpeta raíz.")
