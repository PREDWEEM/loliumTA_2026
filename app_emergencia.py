# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM vK3 — LOLIUM TRES ARROYOS 2026 (Proyectivo)
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, io
from pathlib import Path
import plotly.graph_objects as go
from datetime import timedelta

# ---------------------------------------------------------
# CONFIG STREAMLIT + ESTILO
# ---------------------------------------------------------
st.set_page_config(page_title="PREDWEEM vK3 – TRES ARROYOS 2026", layout="wide")

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

def proyectar_fecha(df_v, target_dga, dga_actual):
    if dga_actual >= target_dga:
        # Ya ocurrió: buscar fecha exacta
        return df_v[df_v["DGA_sum"] >= target_dga]["Fecha"].min(), False
    
    # Proyectar: usar promedio de DG de los últimos 7 días
    ultimos_dg = df_v["DG"].tail(7).mean()
    if ultimos_dg <= 0.2: ultimos_dg = 4.5 # Fallback para días muy fríos
    
    faltante = target_dga - dga_actual
    dias_estimados = int(np.ceil(faltante / ultimos_dg))
    fecha_proyectada = df_v["Fecha"].max() + timedelta(days=dias_estimados)
    return fecha_proyectada, True

# ===============================================================
# ⚙️ CONFIGURACIÓN (SIDEBAR)
# ===============================================================
st.sidebar.header("📂 Gestión de Datos")
uploaded_file = st.sidebar.file_uploader("Subir Clima (Excel o CSV)", type=["xlsx", "csv"])

st.sidebar.divider()
st.sidebar.header("⚙️ Ajustes de Control")

umbral_rel_input = st.sidebar.slider("Sensibilidad de detección", 0.05, 1.00, 0.20, 0.05)
dga_optimo = st.sidebar.slider("Límite Estado Óptimo (°Cd)", 50, 400, 180, 10)
dga_critico = st.sidebar.slider("Límite Estado Crítico (°Cd)", 401, 1000, 600, 10)

# ===============================================================
# 📊 PROCESAMIENTO
# ===============================================================
modelo_ann, cluster_model = load_models()

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["Fecha"]) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file, parse_dates=["Fecha"])
    df.columns = [c.upper().strip() for c in df.columns]
    mapeo = {'FECHA': 'Fecha', 'TMAX': 'TMAX', 'TMIN': 'TMIN', 'PREC': 'Prec'}
    df = df.rename(columns=mapeo)
    
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    emerrel, _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(emerrel, 0.0)
    df.loc[df["Julian_days"] <= 15, "EMERREL"] = 0.0
    df["DG"] = np.maximum(((df["TMAX"] + df["TMIN"]) / 2) - 2.0, 0)
    df["Riesgo"] = df["EMERREL"] / df["EMERREL"].max() if df["EMERREL"].max() > 0 else 0

    st.title("🌾 PREDWEEM vK3 — TRES ARROYOS")

    # --- LÓGICA 2 PULSOS / 5 DÍAS ---
    indices_pulso = df.index[df["EMERREL"] >= umbral_rel_input].tolist()
    fecha_inicio = None
    for i in range(len(indices_pulso) - 1):
        if (df.loc[indices_pulso[i+1], "Fecha"] - df.loc[indices_pulso[i], "Fecha"]).days <= 5:
            fecha_inicio = df.loc[indices_pulso[i], "Fecha"]
            break

    if fecha_inicio:
        df_v = df[df["Fecha"] >= fecha_inicio].copy()
        df_v["DGA_sum"] = df_v["DG"].cumsum()
        dga_act = df_v["DGA_sum"].iloc[-1]

        # Fechas Hito
        f_opt, proy_opt = proyectar_fecha(df_v, dga_optimo, dga_act)
        f_crit, proy_crit = proyectar_fecha(df_v, dga_critico, dga_act)

        # --- DASHBOARD DE ESTADO ---
        st.divider()
        st.header("⏱️ Ventana de Acción Agronómica")
        
        col_gauge, col_dates = st.columns([1, 1])

        with col_gauge:
            fig_g = go.Figure(go.Indicator(
                mode = "gauge+number", value = dga_act,
                title = {'text': "DGA Acumulados (°Cd)", 'font': {'size': 20}},
                gauge = {
                    'axis': {'range': [0, dga_critico + 150]},
                    'bar': {'color': "#2c3e50"},
                    'steps': [
                        {'range': [0, dga_optimo], 'color': "#2ecc71"},
                        {'range': [dga_optimo, dga_critico], 'color': "#f1c40f"},
                        {'range': [dga_critico, dga_critico + 150], 'color': "#e74c3c"}]}))
            fig_g.update_layout(height=350, margin=dict(t=50, b=0))
            st.plotly_chart(fig_g, use_container_width=True)

        with col_dates:
            st.markdown(f"**Inicio Emergencia Sostenida:** {fecha_inicio.strftime('%d-%b-%Y')}")
            
            def label(proy): return "📅 Estimado" if proy else "✅ Alcanzado"
            
            st.info(f"### 🟢 Límite Óptimo\n**{f_opt.strftime('%d-%b')}** \n*{label(proy_opt)}*")
            st.error(f"### 🔴 Límite Crítico\n**{f_crit.strftime('%d-%b')}** \n*{label(proy_crit)}*")

            if dga_act > dga_critico:
                st.error("❗ **ALERTA:** Maleza en macollaje. Control difícil.")
            elif dga_act > dga_optimo:
                st.warning("⚠️ **AVISO:** Ventana óptima cerrada. Aumentar dosis/vigilancia.")
            else:
                st.success("✨ **ESTADO ÓPTIMO:** Máxima sensibilidad a herbicidas.")

        st.divider()
        st.subheader("📈 Progresión del Desarrollo")
        
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=df_v["Fecha"], y=df_v["DGA_sum"], name="DGA Real", line=dict(color='black', width=3)))
        fig_line.add_hline(y=dga_optimo, line_dash="dot", line_color="green", annotation_text="Límite 1-3 Hojas")
        fig_line.add_hline(y=dga_critico, line_dash="dot", line_color="red", annotation_text="Inicio Macollaje")
        st.plotly_chart(fig_line, use_container_width=True)

    else:
        st.warning("🔎 No se detecta emergencia sostenida aún (2 pulsos ≥ umbral en 5 días).")

    # Botón de Descarga
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Analisis')
    st.sidebar.download_button("📥 Descargar Reporte", output.getvalue(), "reporte_lolium.xlsx")

else:
    st.info("👈 Por favor, cargue un archivo de clima para activar el sistema.")
