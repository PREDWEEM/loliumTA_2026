# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM vK3 — LOLIUM TRES ARROYOS 2026
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, io
from pathlib import Path
import plotly.graph_objects as go

# ---------------------------------------------------------
# CONFIG STREAMLIT + ESTILO
# ---------------------------------------------------------
st.set_page_config(page_title="PREDWEEM vK3 – LOLIUM TRES ARROYOS 2026", layout="wide")

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header [data-testid="stToolbar"] {visibility: hidden;}
.stAppDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ===============================================================
# 🔧 MODELOS Y FUNCIONES TÉCNICAS
# ===============================================================
def dtw_distance(a, b):
    na, nb = len(a), len(b)
    dp = np.full((na+1, nb+1), np.inf)
    dp[0,0] = 0
    for i in range(1, na+1):
        for j in range(1, nb+1):
            cost = abs(a[i-1] - b[j-1])
            dp[i,j] = cost + min(dp[i-1,j], dp[i,j-1], dp[i-1,j-1])
    return dp[na, nb]

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
        with open(BASE/"modelo_clusters_k3.pkl", "rb") as f:
            k3 = pickle.load(f)
        return ann, k3
    except Exception as e:
        st.error(f"Error cargando archivos de modelo: {e}")
        return None, None

# ===============================================================
# ⚙️ CONFIGURACIÓN DE UMBRALES (BARRA LATERAL)
# ===============================================================
st.sidebar.header("📂 Gestión de Datos")
uploaded_file = st.sidebar.file_uploader("Subir Clima (Excel o CSV)", type=["xlsx", "csv"])

st.sidebar.divider()
st.sidebar.header("⚙️ Ajustes de Control")

umbral_rel_input = st.sidebar.slider(
    "Sensibilidad de detección", 
    0.05, 0.80, 0.50, 0.05,
    help="Define el tamaño del pulso de emergencia necesario para considerar un evento de nacimiento."
)

st.sidebar.subheader("Límites de Tiempo Térmico (°Cd)")
dga_optimo = st.sidebar.slider(
    "Límite Estado Óptimo", 
    50, 300, 180, 10,
    help="DGA máximos para una maleza de 1 a 3 hojas. Absorción de herbicidas máxima."
)

dga_critico = st.sidebar.slider(
    "Límite Estado Crítico", 
    500, 1000, 800, 10,
    help="DGA a partir de los cuales se considera riesgo alto de macollaje y fallas de control."
)

# ===============================================================
# 📊 PROCESAMIENTO DE DATOS
# ===============================================================
def get_data(file_input):
    try:
        if file_input is not None:
            df = pd.read_csv(file_input, parse_dates=["Fecha"]) if file_input.name.endswith('.csv') else pd.read_excel(file_input, parse_dates=["Fecha"])
        else:
            path_fixed = BASE / "meteo_daily.csv"
            if path_fixed.exists():
                df = pd.read_csv(path_fixed, parse_dates=["Fecha"])
            else: return None
        
        df.columns = [c.upper().strip() for c in df.columns]
        mapeo = {
            'FECHA': 'Fecha', 'DATE': 'Fecha',
            'TMAX': 'TMAX', 'TMIN': 'TMIN', 'PREC': 'Prec', 'LLUVIA': 'Prec'
        }
        return df.rename(columns=mapeo)
    except Exception as e:
        st.error(f"Error al leer datos: {e}"); return None

modelo_ann, cluster_model = load_models()
df = get_data(uploaded_file)

if df is not None and modelo_ann is not None:
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear

    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    emerrel, _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(emerrel, 0.0)
    df.loc[df["Julian_days"] <= 15, "EMERREL"] = 0.0
    
    T_BASE = 2.0
    df["DG"] = np.maximum(((df["TMAX"] + df["TMIN"]) / 2) - T_BASE, 0)
    
    max_er = df["EMERREL"].max()
    df["Riesgo"] = df["EMERREL"] / max_er if max_er > 0 else 0.0

    # ===============================================================
    # 🖥️ VISUALIZACIÓN
    # ===============================================================
    st.title("🌾 PREDWEEM vK3 — LOLIUM TRES ARROYOS 2026")
    
    fig_risk = go.Figure(data=go.Heatmap(
        z=[df["Riesgo"].values], x=df["Fecha"], y=["Riesgo"],
        colorscale='Viridis', zmin=0, zmax=1,
        hovertemplate="<b>%{x|%d-%b}</b><br>Riesgo: %{z:.2f}<extra></extra>"))
    fig_risk.update_layout(height=180, title="Evolución del Riesgo de Emergencia", margin=dict(t=40, b=0))
    st.plotly_chart(fig_risk, use_container_width=True)

    st.divider()

    # --- LÓGICA DE VALIDACIÓN: 2 PULSOS EN 5 DÍAS ---
    indices_pulso = df.index[df["EMERREL"] >= umbral_rel_input].tolist()
    fecha_inicio_ventana = None
    
    for i in range(len(indices_pulso) - 1):
        idx1 = indices_pulso[i]
        idx2 = indices_pulso[i+1]
        if (df.loc[idx2, "Fecha"] - df.loc[idx1, "Fecha"]).days <= 5:
            fecha_inicio_ventana = df.loc[idx1, "Fecha"]
            break

    if fecha_inicio_ventana:
        # 1. Análisis de Patrón
        JD_COMMON = cluster_model["JD_common"]
        curves_interp = cluster_model["curves_interp"]
        meds_idx = cluster_model["medoids_k3"]
        
        emer_norm = df["EMERREL"].to_numpy() / max_er
        curve_year_interp = np.interp(JD_COMMON, df["Julian_days"], emer_norm)
        
        meds = [curves_interp[i] for i in meds_idx]
        dists = [dtw_distance(curve_year_interp, m) for m in meds]
        cluster_pred = np.argmin(dists)

        names = {0: "🌾 Intermedio / Bimodal", 1: "🌱 Temprano / Compacto", 2: "🍂 Tardío / Extendido"}
        colors = {0: "blue", 1: "green", 2: "orange"}
        
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.header("🎯 Patrón Detectado")
            st.markdown(f"<h2 style='color:{colors[cluster_pred]};'>{names[cluster_pred]}</h2>", unsafe_allow_html=True)
            cert = 1 - (min(dists) / sum(dists))
            st.metric("Confianza", f"{cert:.1%}")
        with c2:
            fig_cmp, ax = plt.subplots(figsize=(7, 3))
            ax.plot(JD_COMMON, curve_year_interp, label="Datos Actuales", color="black", lw=2)
            ax.plot(JD_COMMON, meds[cluster_pred], label="Referencia", color=colors[cluster_pred], ls="--")
            ax.legend(); st.pyplot(fig_cmp)

        # 2. Ventana de Acción
        st.divider()
        st.header("🗓️ Ventana de Acción Agronómica")
        
        
        
        dga = df[df["Fecha"] >= fecha_inicio_ventana]["DG"].cumsum().iloc[-1]
        
        v1, v2, v3 = st.columns(3)
        v1.metric("Inicio (Confirmado)", fecha_inicio_ventana.strftime("%d-%b"))
        v2.metric("Suma Térmica", f"{dga:.1f} °Cd")
        
        if dga <= dga_optimo:
            v3.success(f"🟢 ÓPTIMO: < {dga_optimo} °Cd")
            st.info("✅ **Diagnóstico:** Emergencia confirmada. Máxima sensibilidad.")
        elif dga <= dga_critico:
            v3.warning(f"🟡 LÍMITE: {dga_optimo}-{dga_critico} °Cd")
            st.warning("⚠️ **Diagnóstico:** INICIO MACOLLAJE 3/4 HOJAS")
        else:
            v3.error(f"🔴 CRÍTICO: > {dga_critico} °Cd")
            st.error("❗ **Alerta:** MACOLLAJE AVANZADO. Posibles fallas de control.")
    else:
        st.info(f"Esperando emergencia sostenida (2 pulsos ≥ {umbral_rel_input} en 5 días) para activar alertas.")

    # Descarga
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Predicciones')
    st.sidebar.download_button("📥 Descargar Excel", output.getvalue(), "reporte_tres_arroyos.xlsx")

    with st.expander("🔍 Ver tabla de datos"):
        st.dataframe(df.style.format(precision=3))
else:
    st.warning("👈 Por favor, sube un archivo de clima para comenzar.")

st.sidebar.markdown("---")
st.sidebar.caption("PREDWEEM vK3 | Tres Arroyos 2026")
