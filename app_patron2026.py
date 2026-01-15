# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM vK3 — LOLIUM TRES ARROYOS 2026
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle, io
from pathlib import Path

# Configuración de la página (opcional, pero recomendada)
st.set_page_config(page_title="Mi App", layout="wide")

# CSS para ocultar el menú hamburguesa, el header y el footer
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


# ---------------------------------------------------------
# 0. ROBUSTNESS: GENERADOR DE ARCHIVOS MOCK (Para pruebas)
# ---------------------------------------------------------
BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

def create_mock_files_if_missing():
    """Genera archivos base si no existen para que el script sea ejecutable de inmediato."""
    if not (BASE / "IW.npy").exists():
        np.save(BASE / "IW.npy", np.random.rand(4, 10))
        np.save(BASE / "bias_IW.npy", np.random.rand(10))
        np.save(BASE / "LW.npy", np.random.rand(1, 10))
        np.save(BASE / "bias_out.npy", np.random.rand(1))
    
    if not (BASE / "modelo_clusters_k3.pkl").exists():
        jd = np.arange(1, 366)
        # Crear 3 patrones sintéticos (Temprano, Intermedio, Tardío)
        p1 = np.exp(-((jd - 100)**2)/600)  # Temprano
        p2 = np.exp(-((jd - 160)**2)/900) + 0.3*np.exp(-((jd - 260)**2)/1200) # Bimodal
        p3 = np.exp(-((jd - 230)**2)/1500) # Tardío
        mock_cluster = {
            "JD_common": jd,
            "curves_interp": [p2, p1, p3],
            "medoids_k3": [0, 1, 2]
        }
        with open(BASE / "modelo_clusters_k3.pkl", "wb") as f:
            pickle.dump(mock_cluster, f)

    if not (BASE / "meteo_daily.csv").exists():
        # Generamos datos hasta mayo para permitir el análisis funcional de inmediato
        dates = pd.date_range(start="2026-01-01", periods=150)
        data = {
            "Fecha": dates,
            "TMAX": np.random.uniform(25, 35, size=150) - (np.arange(150)*0.1),
            "TMIN": np.random.uniform(10, 18, size=150) - (np.arange(150)*0.06),
            "Prec": np.random.choice([0, 0, 5, 15, 45], size=150)
        }
        pd.DataFrame(data).to_csv(BASE / "meteo_daily.csv", index=False)

create_mock_files_if_missing()

# ---------------------------------------------------------
# 1. MODELOS Y LÓGICA TÉCNICA
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
        st.error(f"Error cargando modelos: {e}")
        return None, None

# ---------------------------------------------------------
# 2. CONFIGURACIÓN DE PÁGINA Y DATOS
# ---------------------------------------------------------
st.set_page_config(page_title="PREDWEEM vK3 – TRES ARROYOS", layout="wide", page_icon="🌾")

st.sidebar.title("🌾 PREDWEEM vK3")
st.sidebar.caption("Lolium TRES ARROYOS 2026")

# Parámetros en Sidebar
umbral_alerta = st.sidebar.slider("Umbral de Alerta (Emergencia)", 0.1, 1.0, 0.5, 0.05)
archivo_subido = st.sidebar.file_uploader("Subir Clima (Excel/CSV)", type=["xlsx", "csv"])

def get_data(file_input):
    if file_input:
        df = pd.read_csv(file_input, parse_dates=["Fecha"]) if file_input.name.endswith('.csv') else pd.read_excel(file_input, parse_dates=["Fecha"])
    else:
        path = BASE / "meteo_daily.csv"
        df = pd.read_csv(path, parse_dates=["Fecha"]) if path.exists() else None
    
    if df is not None:
        df.columns = [c.upper().strip() for c in df.columns]
        mapeo = {'FECHA': 'Fecha', 'TMAX': 'TMAX', 'TMIN': 'TMIN', 'PREC': 'Prec', 'LLUVIA': 'Prec'}
        df = df.rename(columns=mapeo)
    return df

modelo_ann, cluster_model = load_models()
df = get_data(archivo_subido)

# ---------------------------------------------------------
# 3. PROCESAMIENTO Y DASHBOARD
# ---------------------------------------------------------
if df is not None and modelo_ann is not None:
    # Limpieza
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear

    # Predicción ANN
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    emerrel, _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(emerrel, 0.0)
    df.loc[df["Julian_days"] <= 30, "EMERREL"] = 0.0 # Filtro biológico inicial
    
    # --- UI ---
    st.title("🌾 PREDWEEM vK3 — TRES ARROYOS 2026")

    # A. MAPA SEMAFÓRICO
    colorscale = [[0, "#dcfce7"], [0.49, "#16a34a"], [0.49, "#facc15"], [0.9, "#eab308"], [0.9, "#ef4444"], [1, "#b91c1c"]]
    fig_h = go.Figure(data=go.Heatmap(z=[df["EMERREL"]], x=df["Fecha"], y=["Emergencia"], colorscale=colorscale, zmin=0, zmax=1, showscale=False))
    fig_h.update_layout(height=130, margin=dict(t=30, b=0, l=10, r=10), title="Intensidad de Emergencia Diaria")
    st.plotly_chart(fig_h, use_container_width=True)

    # B. MONITOREO DE PULSOS
    fig_m = go.Figure()
    fig_m.add_trace(go.Scatter(x=df["Fecha"], y=df["EMERREL"], fill='tozeroy', line_color='#15803d', name="Tasa Diaria"))
    fig_m.add_hline(y=umbral_alerta, line_dash="dash", line_color="red", annotation_text="Umbral Crítico")
    fig_m.update_layout(height=300, title="Dinámica de Emergencia Relativa", margin=dict(t=30, b=10))
    st.plotly_chart(fig_m, use_container_width=True)

    # C. ANÁLISIS FUNCIONAL (CORTE 1 DE MAYO)
    st.divider()
    st.header("📊 Análisis Funcional de Patrones")
    
    # Punto de corte: 1 de Mayo (Incluye información de Ene, Feb, Mar y Abr)
    fecha_corte = pd.Timestamp("2026-05-01")
    df_cuatrimestre = df[df["Fecha"] < fecha_corte].copy()

    if df_cuatrimestre.empty:
        st.info("ℹ️ El análisis funcional se activará cuando existan datos colectados previos al 1 de MAYO.")
    else:
        st.success(f"🔍 Clasificación activa: Analizando datos de Enero a Abril (Corte al 1 de MAYO).")
        
        # Lógica de Clasificación DTW
        jd_corte = df_cuatrimestre["Julian_days"].max()
        max_e_obs = df_cuatrimestre["EMERREL"].max() if df_cuatrimestre["EMERREL"].max() > 0 else 1
        
        # Curva actual interpolada a la grilla común hasta el día de corte
        JD_COMMON = cluster_model["JD_common"]
        jd_obs_grid = JD_COMMON[JD_COMMON <= jd_corte]
        curva_obs_norm = np.interp(jd_obs_grid, df_cuatrimestre["Julian_days"], df_cuatrimestre["EMERREL"] / max_e_obs)
        
        # Comparación con patrones históricos (medoides)
        dists = []
        meds = cluster_model["curves_interp"]
        for m in meds:
            # Recortamos el patrón histórico al mismo punto para comparar
            m_slice = m[JD_COMMON <= jd_corte]
            # Re-normalizamos el trozo para que la escala sea comparable en forma
            m_slice_norm = m_slice / m_slice.max() if m_slice.max() > 0 else m_slice
            dists.append(dtw_distance(curva_obs_norm, m_slice_norm))
            
        cluster_pred = np.argmin(dists)
        nombres = {0: "🌾 Intermedio / Bimodal", 1: "🌱 Temprano / Compacto", 2: "🍂 Tardío / Extendido"}
        colores = {0: "#0284c7", 1: "#16a34a", 2: "#ea580c"}

        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"#### Patrón Detectado: <span style='color:{colores[cluster_pred]}'>{nombres[cluster_pred]}</span>", unsafe_allow_html=True)
            fig_p, ax = plt.subplots(figsize=(10, 4))
            
            # Graficamos el patrón completo para mostrar la proyección futura
            ax.plot(JD_COMMON, meds[cluster_pred], color=colores[cluster_pred], ls="--", alpha=0.4, label="Proyección Patrón Anual")
            
            # Graficamos los datos reales observados
            # Escalamos para que coincidan visualmente con el patrón
            factor_escala = meds[cluster_pred].max() if meds[cluster_pred].max() > 0 else 1
            ax.plot(jd_obs_grid, curva_obs_norm * factor_escala, color="black", lw=2.5, label="Observado (Ene-Abr)")
            
            ax.axvline(jd_corte, color="red", ls=":", label="Corte 1 de Mayo")
            ax.set_title("Ajuste de Campaña Actual vs. Patrones Históricos")
            ax.set_xlabel("Día Juliano")
            ax.legend()
            ax.grid(True, alpha=0.2)
            st.pyplot(fig_p)
            
        with c2:
            st.metric("Distancia DTW", f"{min(dists):.2f}")
            st.caption("Menor distancia = Mayor similitud.")
            st.info("El sistema evalúa la 'forma' de la emergencia acumulada durante el primer cuatrimestre para proyectar el comportamiento más probable del resto de la campaña.")

    # Exportar
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    st.sidebar.download_button("📥 Descargar Datos y Predicciones", output.getvalue(), "predweem_bordenave_2026.xlsx")

else:
    st.warning("👈 Cargue un archivo de clima o use los datos por defecto para visualizar el dashboard.")
