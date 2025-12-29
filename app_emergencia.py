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
st.set_page_config(page_title="PREDWEEM vK3 – LOLIUM 2026", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    /* Barra lateral verde claro */
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
# 3. GESTIÓN DE DATOS (LECTURA AUTOMÁTICA GITHUB)
# ---------------------------------------------------------
def get_data(file_input):
    df = None
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
# 4. INTERFAZ Y CÁLCULOS
# ---------------------------------------------------------
modelo_ann, cluster_model = load_models()
df = get_data(st.sidebar.file_uploader("Subir Clima Manual", type=["xlsx", "csv"]))

# Configuración de umbrales en Sidebar
st.sidebar.title("🌿 PREDWEEM vK3")
if st.sidebar.button("🔄 Actualizar App"): st.rerun()
st.sidebar.divider()
umbral_er = st.sidebar.slider("Sensibilidad Detección (Umbral)", 0.05, 0.80, 0.45)
dga_optimo = st.sidebar.slider("Límite Óptimo (°Cd)", 50, 800, 650)
dga_critico = st.sidebar.slider("Límite Crítico (°Cd)", 600, 1200, 850)

if df is not None and modelo_ann is not None:
    # Procesamiento Clima
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    
    # Predicción ANN y Tiempo Térmico
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    emerrel, _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(emerrel, 0.0)
    df.loc[df["Julian_days"] <= 15, "EMERREL"] = 0.0
    
    df["DG"] = np.maximum(((df["TMAX"] + df["TMIN"]) / 2) - 2.0, 0) # Base 2.0°C
    max_er = df["EMERREL"].max()
    df["Riesgo"] = df["EMERREL"] / max_er if max_er > 0 else 0.0

    st.title("🌾 PREDWEEM vK3 — LOLIUM TRES ARROYOS 2026")

    # --- GRÁFICOS PRINCIPALES ---
    
    # 1. Mapa de Riesgo Semafórico
    fig_risk = go.Figure(data=go.Heatmap(
        z=[df["Riesgo"].values], x=df["Fecha"], y=["Riesgo"],
        colorscale=[[0, 'green'], [0.5, 'yellow'], [1, 'red']],
        zmin=0, zmax=1,
        hovertemplate="<b>%{x|%d-%b}</b><br>Riesgo: %{z:.2f}<extra></extra>"))
    fig_risk.update_layout(height=180, margin=dict(t=40, b=0, l=10, r=10), title="Intensidad de Riesgo de Emergencia")
    st.plotly_chart(fig_risk, use_container_width=True)

    # 2. Gráfico de Emergencia Relativa (EMERREL)
    fig_emer = go.Figure()
    fig_emer.add_trace(go.Scatter(
        x=df["Fecha"], y=df["EMERREL"], 
        mode='lines', name='Emergencia Diaria',
        line=dict(color='#166534', width=2),
        fill='tozeroy', fillcolor='rgba(22, 101, 52, 0.1)'
    ))
    # Línea de umbral de detección
    fig_emer.add_hline(y=umbral_er, line_dash="dash", line_color="orange", 
                       annotation_text=f"Umbral: {umbral_er}", annotation_position="top right")
    
    fig_emer.update_layout(
        title="Predicción de Pulsos de Emergencia (EMERREL)",
        xaxis_title="Fecha", yaxis_title="Emergencia Relativa",
        height=350, margin=dict(t=40, b=40, l=10, r=10),
        hovermode="x unified"
    )
    st.plotly_chart(fig_emer, use_container_width=True)


    # --- LÓGICA DE VENTANA Y FECHAS LÍMITE ---
    
    indices_pulso = df.index[df["EMERREL"] >= umbral_er].tolist()
    fecha_inicio_ventana = None
    for i in range(len(indices_pulso) - 1):
        if (df.loc[indices_pulso[i+1], "Fecha"] - df.loc[indices_pulso[i], "Fecha"]).days <= 5:
            fecha_inicio_ventana = df.loc[indices_pulso[i], "Fecha"]
            break

    if fecha_inicio_ventana:
        # Clasificación de Patrón
        JD_COMMON = cluster_model["JD_common"]
        curves_interp = cluster_model["curves_interp"]
        meds_idx = cluster_model["medoids_k3"]
        curve_curr = np.interp(JD_COMMON, df["Julian_days"], df["EMERREL"]/max_er)
        dists = [dtw_distance(curve_curr, curves_interp[i]) for i in meds_idx]
        cluster_pred = np.argmin(dists)
        
        names = {0: "🌾 Intermedio", 1: "🌱 Temprano", 2: "🍂 Tardío"}

        st.divider()
        st.header("🗓️ Cronograma y Fechas Límite de Acción")
        
        # Cálculos de Tiempo Térmico Proyectado
        df_ventana = df[df["Fecha"] >= fecha_inicio_ventana].copy()
        df_ventana["DGA_cum"] = df_ventana["DG"].cumsum()
        dga_actual = df_ventana["DGA_cum"].iloc[-1]
        
        tasa_proy = df["DG"].tail(7).mean() 
        if tasa_proy < 1.0: tasa_proy = 5.5 # Valor de seguridad por defecto

        def calcular_fecha_limite(objetivo):
            if dga_actual >= objetivo:
                match = df_ventana[df_ventana["DGA_cum"] >= objetivo]
                return match["Fecha"].iloc[0], "PASADO"
            else:
                faltante = objetivo - dga_actual
                dias_proy = int(faltante / tasa_proy)
                fecha_est = df["Fecha"].max() + pd.Timedelta(days=dias_proy)
                return fecha_est, "ESTIMADO"

        f_optima, status_opt = calcular_fecha_limite(dga_optimo)
        f_critica, status_crit = calcular_fecha_limite(dga_critico)

        # Métricas
        m1, m2, m3 = st.columns(3)
        m1.metric("Inicio de Ventana", fecha_inicio_ventana.strftime("%d-%b"))
        m2.metric("Acumulado (°Cd)", f"{dga_actual:.1f}")
        m3.metric("Patrón Detectado", names[cluster_pred])

        # Tabla de Fechas Límite
        fmt = "%d-%m-%Y"
        data_limites = {
            "Nivel de Alerta": ["🟢 ÓPTIMO", "🟡 LÍMITE", "🔴 CRÍTICO"],
            "G. Día Objetivo": [f"{dga_optimo} °Cd", f"{dga_critico} °Cd", "Final de Emergencia"],
            "Fecha Límite": [f_optima.strftime(fmt), f_critica.strftime(fmt), "Sin Proyección"],
            "Estado del Dato": [status_opt, status_crit, "RIESGO ALTO"]
        }
        st.table(pd.DataFrame(data_limites))

        # Alertas de Acción
        if dga_actual <= dga_optimo:
            st.success(f"✅ **ESTADO ÓPTIMO:** Tienes hasta el **{f_optima.strftime(fmt)}** para control de máxima eficiencia.")
        elif dga_actual <= dga_critico:
            st.warning(f"⚠️ **ESTADO LÍMITE:** La ventana óptima cerró el {f_optima.strftime(fmt)}. Límite crítico: **{f_critica.strftime(fmt)}**.")
        else:
            st.error(f"❗ **ESTADO CRÍTICO:** Se superó el límite térmico el {f_critica.strftime(fmt)}. Eficacia de control reducida.")

    else:
        st.info(f"⏳ Esperando detección de pulso: Se requieren 2 eventos significativos (≥ {umbral_er}) cercanos para iniciar el cronograma.")

    # Exportación
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Predicciones')
    st.sidebar.download_button("📥 Descargar Reporte Excel", output.getvalue(), "predweem_2026.xlsx")

else:
    st.warning("⚠️ Cargando datos... Asegúrate de tener 'meteo_daily.csv' en el repositorio o subir un archivo manualmente.")
