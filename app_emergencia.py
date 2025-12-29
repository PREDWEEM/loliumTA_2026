# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, io
from pathlib import Path
import plotly.graph_objects as go

# ---------------------------------------------------------
# CONFIGURACIÓN Y ESTILO
# ---------------------------------------------------------
st.set_page_config(page_title="PREDWEEM vK3 – LOLIUM 2026", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    [data-testid="stSidebar"] { background-color: #1e293b; color: white; }
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ---------------------------------------------------------
# FUNCIONES TÉCNICAS Y MODELOS
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

# ---------------------------------------------------------
# GESTIÓN DE DATOS (GitHub Auto-update)
# ---------------------------------------------------------
def get_data(file_input):
    """Carga datos del archivo subido o, por defecto, del repositorio local."""
    df = None
    try:
        if file_input is not None:
            # Opción 1: Archivo subido por el usuario
            df = pd.read_csv(file_input, parse_dates=["Fecha"]) if file_input.name.endswith('.csv') else pd.read_excel(file_input, parse_dates=["Fecha"])
            st.sidebar.success("Usando archivo subido manualmente.")
        else:
            # Opción 2: Archivo automático en la carpeta de GitHub
            path_fixed = BASE / "meteo_daily.csv"
            if path_fixed.exists():
                df = pd.read_csv(path_fixed, parse_dates=["Fecha"])
                st.sidebar.info("Cargando datos automáticos desde GitHub (meteo_daily.csv)")
            else:
                return None
        
        # Estandarización de columnas
        df.columns = [c.upper().strip() for c in df.columns]
        mapeo = {
            'FECHA': 'Fecha', 'DATE': 'Fecha', 
            'TMAX': 'TMAX', 'TMIN': 'TMIN', 
            'PREC': 'Prec', 'LLUVIA': 'Prec'
        }
        df = df.rename(columns=mapeo)
        return df
    except Exception as e:
        st.error(f"Error al procesar datos: {e}")
        return None

# ---------------------------------------------------------
# INTERFAZ LATERAL
# ---------------------------------------------------------
st.sidebar.title("🌾 PREDWEEM vK3")
uploaded_file = st.sidebar.file_uploader("Subir Clima alternativo (Excel/CSV)", type=["xlsx", "csv"])

if st.sidebar.button("🔄 Actualizar Datos Ahora"):
    st.rerun()

st.sidebar.divider()
umbral_rel_input = st.sidebar.slider("Sensibilidad Detección", 0.05, 0.80, 0.50)
dga_optimo = st.sidebar.slider("Límite Óptimo (°Cd)", 50, 800, 600)
dga_critico = st.sidebar.slider("Límite Crítico (°Cd)", 600, 1200, 850)

# ---------------------------------------------------------
# PROCESAMIENTO PRINCIPAL
# ---------------------------------------------------------
modelo_ann, cluster_model = load_models()
df = get_data(uploaded_file)

if df is not None and modelo_ann is not None:
    # Limpieza básica
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    
    # Predicción ANN
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    emerrel, _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(emerrel, 0.0)
    df.loc[df["Julian_days"] <= 15, "EMERREL"] = 0.0 # Filtro ruido enero
    
    # Cálculo Térmico (Base 2.0°C)
    T_BASE = 2.0
    df["DG"] = np.maximum(((df["TMAX"] + df["TMIN"]) / 2) - T_BASE, 0)
    max_er = df["EMERREL"].max()
    df["Riesgo"] = df["EMERREL"] / max_er if max_er > 0 else 0.0

    st.title("🌾 PREDWEEM vK3 — LOLIUM TRES ARROYOS 2026")

    # Mapa de Calor de Riesgo
    fig_risk = go.Figure(data=go.Heatmap(
        z=[df["Riesgo"].values], x=df["Fecha"], y=["Riesgo"],
        colorscale='YlOrRd', zmin=0, zmax=1,
        hovertemplate="<b>%{x|%d-%b}</b><br>Riesgo: %{z:.2f}<extra></extra>"))
    fig_risk.update_layout(height=180, title="Evolución del Riesgo de Emergencia", margin=dict(t=40, b=0))
    st.plotly_chart(fig_risk, use_container_width=True)

    # Detección de Pulso: 2 eventos en 5 días
    indices_pulso = df.index[df["EMERREL"] >= umbral_rel_input].tolist()
    fecha_inicio_ventana = None
    for i in range(len(indices_pulso) - 1):
        idx1, idx2 = indices_pulso[i], indices_pulso[i+1]
        if (df.loc[idx2, "Fecha"] - df.loc[idx1, "Fecha"]).days <= 5:
            fecha_inicio_ventana = df.loc[idx1, "Fecha"]
            break

    if fecha_inicio_ventana:
        st.divider()
        
        # 1. Comparativa de Patrones (Clustering)
        JD_COMMON = cluster_model["JD_common"]
        curves_interp = cluster_model["curves_interp"]
        meds_idx = cluster_model["medoids_k3"]
        emer_norm = df["EMERREL"].to_numpy() / max_er
        curve_year_interp = np.interp(JD_COMMON, df["Julian_days"], emer_norm)
        meds = [curves_interp[i] for i in meds_idx]
        dists = [dtw_distance(curve_year_interp, m) for m in meds]
        cluster_pred = np.argmin(dists)

        names = {0: "🌾 Intermedio / Bimodal", 1: "🌱 Temprano / Compacto", 2: "🍂 Tardío / Extendido"}
        colors = {0: "#2E86C1", 1: "#28B463", 2: "#E67E22"}
        
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.subheader("🎯 Patrón de Emergencia")
            st.markdown(f"<h2 style='color:{colors[cluster_pred]};'>{names[cluster_pred]}</h2>", unsafe_allow_html=True)
            cert = 1 - (min(dists) / sum(dists))
            st.metric("Confianza de Ajuste", f"{cert:.1%}")
        with c2:
            fig_cmp, ax = plt.subplots(figsize=(7, 3), facecolor='#f5f7f9')
            ax.plot(JD_COMMON, curve_year_interp, label="Año Actual", color="black", lw=2)
            ax.plot(JD_COMMON, meds[cluster_pred], label="Referencia", color=colors[cluster_pred], ls="--")
            ax.legend(); st.pyplot(fig_cmp)

        # 2. Ventana de Acción y Proyección de Fechas
        st.divider()
        st.subheader("🗓️ Proyección de la Ventana de Acción")
        
        mask_v = df["Fecha"] >= fecha_inicio_ventana
        dga_actual = df[mask_v]["DG"].sum()
        
        # Tasa de avance: promedio térmico últimos 7 días
        tasa_diaria = df["DG"].tail(7).mean()
        if tasa_diaria < 1.0: tasa_diaria = 5.5 # Valor defensivo para días fríos

        def estimar_fecha(objetivo):
            faltante = objetivo - dga_actual
            if faltante <= 0: return "ALCANZADO"
            dias_extra = int(faltante / tasa_diaria)
            return df["Fecha"].max() + pd.Timedelta(days=dias_extra)

        f_optima = estimar_fecha(dga_optimo)
        f_critica = estimar_fecha(dga_critico)

        v1, v2, v3 = st.columns(3)
        v1.metric("Inicio Emergencia", fecha_inicio_ventana.strftime("%d-%b"))
        v2.metric("Térmico Acumulado", f"{dga_actual:.1f} °Cd")
        v3.metric("Velocidad Est.", f"{tasa_diaria:.1f} °Cd/día")

        # Tabla Resumen
        def fmt(f): return f.strftime("%d-%m-%Y") if isinstance(f, pd.Timestamp) else f
        
        resumen_data = {
            "Nivel": ["🟢 Óptimo", "🟡 Límite Crítico", "🔴 Post-Crítico"],
            "Fenología": ["1-3 hojas (Sin macollo)", "Inicio de Macollaje", "Macollaje Avanzado"],
            "Umbral (°Cd)": [f"Hasta {dga_optimo}", f"{dga_optimo} a {dga_critico}", f"Más de {dga_critico}"],
            "Fecha Estimada": [fmt(f_optima), fmt(f_critica), "Fuera de Ventana"]
        }
        st.table(pd.DataFrame(resumen_data))

        # Alertas de Estado
        if dga_actual <= dga_optimo:
            st.success(f"✅ **ESTADO ÓPTIMO:** Tienes tiempo hasta el **{fmt(f_optima)}** para aplicar.")
        elif dga_actual <= dga_critico:
            st.warning(f"⚠️ **ESTADO LÍMITE:** El Lolium está iniciando macollaje. Fecha crítica: **{fmt(f_critica)}**.")
        else:
            st.error(f"❗ **ESTADO CRÍTICO:** Se superaron los {dga_critico} °Cd. Alta probabilidad de fallas de control.")

    else:
        st.info(f"Detección: Esperando 2 pulsos significativos (≥ {umbral_rel_input}) en 5 días para fijar inicio.")

    # Descarga de Reporte
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Predicciones')
    st.sidebar.download_button("📥 Descargar Reporte .xlsx", output.getvalue(), "predweem_reporte.xlsx")

else:
    st.warning("⚠️ No se encontró 'meteo_daily.csv' ni se subió un archivo. Por favor, verifica tu repositorio.")

st.sidebar.caption("PREDWEEM vK3 | Tres Arroyos 2026")
