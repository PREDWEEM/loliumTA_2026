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
df_raw = get_data(st.sidebar.file_uploader("Subir Clima Manual (Opcional)", type=["xlsx", "csv"]))

st.sidebar.divider()
umbral_er = st.sidebar.slider("Sensibilidad de Detección", 0.05, 0.80, 0.45)
dga_optimo = st.sidebar.slider("Umbral Óptimo (°Cd)", 50, 800, 600)
dga_critico = st.sidebar.slider("Umbral Crítico (°Cd)", 600, 1200, 850)

if df_raw is not None and modelo_ann is not None:
    # 1. Limpieza y Fecha Límite de Registro
    df = df_raw.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    ultima_fecha_registro = df["Fecha"].max()
    
    # 2. Cálculos Técnicos
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    emerrel, _ = modelo_ann.predict(X)
    
    df["EMERREL"] = np.maximum(emerrel, 0.0)
    df.loc[df["Julian_days"] <= 15, "EMERREL"] = 0.0
    
    df["DG"] = np.maximum(((df["TMAX"] + df["TMIN"]) / 2) - 2.0, 0) 
    max_er = df["EMERREL"].max()
    df["Riesgo"] = df["EMERREL"] / max_er if max_er > 0 else 0.0

    st.title("🌾 PREDWEEM | LOLIUM TRES ARROYOS 2026")
    st.info(f"📅 Datos visualizados hasta el último registro: **{ultima_fecha_registro.strftime('%d-%m-%Y')}**")

    # --- 1. MAPA DE CALOR (Limitado a registro) ---
    fig_risk = go.Figure(data=go.Heatmap(
        z=[df["Riesgo"].values], x=df["Fecha"], y=["Riesgo"],
        colorscale=[[0, 'green'], [0.5, 'yellow'], [1, 'red']],
        zmin=0, zmax=1, showscale=False))
    fig_risk.update_layout(height=120, margin=dict(t=30, b=0, l=10, r=10), title="Mapa de Calor: Intensidad de Riesgo")
    st.plotly_chart(fig_risk, use_container_width=True)

    # --- 2. GRÁFICO DE EMERGENCIA (Limitado a registro) ---
    fig_emer = go.Figure()
    fig_emer.add_trace(go.Scatter(
        x=df["Fecha"], y=df["EMERREL"], 
        mode='lines', name='Emergencia',
        line=dict(color='#166534', width=2.5),
        fill='tozeroy', fillcolor='rgba(22, 101, 52, 0.1)'
    ))
    fig_emer.add_hline(y=umbral_er, line_dash="dash", line_color="orange", 
                       annotation_text="Umbral de Alerta")
    fig_emer.update_layout(title="Dinámica de Emergencia Relativa Diaria", height=300)
    st.plotly_chart(fig_emer, use_container_width=True)

    # --- 3. CRONOGRAMA Y VENTANA DE ACCIÓN ---
    indices_pulso = df.index[df["EMERREL"] >= umbral_er].tolist()
    fecha_inicio_ventana = None
    
    for i in range(len(indices_pulso) - 1):
        if (df.loc[indices_pulso[i+1], "Fecha"] - df.loc[indices_pulso[i], "Fecha"]).days <= 5:
            fecha_inicio_ventana = df.loc[indices_pulso[i], "Fecha"]
            break

    if fecha_inicio_ventana:
        st.divider()
        st.header("🗓️ Cronograma y Fechas Límite de Acción")
        
        df_ventana = df[df["Fecha"] >= fecha_inicio_ventana].copy()
        df_ventana["DGA_cum"] = df_ventana["DG"].cumsum()
        dga_actual = df_ventana["DGA_cum"].iloc[-1]

        # REGLA: Si no se acumuló el tiempo térmico requerido, "Sin dato"
        def calc_limite_estricto(objetivo):
            # Solo buscamos dentro de la ventana de datos reales acumulados
            cumplimiento = df_ventana[df_ventana["DGA_cum"] >= objetivo]
            if not cumplimiento.empty:
                fecha_fase = cumplimiento["Fecha"].iloc[0]
                return fecha_fase.strftime("%d-%m-%Y"), "CUMPLIDO"
            else:
                return "Sin dato", "PENDIENTE"

        f_opt, s_opt = calc_limite_estricto(dga_optimo)
        f_cri, s_cri = calc_limite_estricto(dga_critico)

        c1, c2, c3 = st.columns(3)
        c1.metric("Inicio Confirmado", fecha_inicio_ventana.strftime("%d-%b"))
        c2.metric("Suma Térmica Real", f"{dga_actual:.1f} °Cd")
        c3.metric("Fecha Límite Óptima", f_opt if f_opt != "Sin dato" else "Sin dato")

        st.table(pd.DataFrame({
            "Nivel de Alerta": ["🟢 ÓPTIMO", "🟡 LÍMITE CRÍTICO", "🔴 POST-CRÍTICO"],
            "Estatus Térmico": [s_opt, s_cri, "TOLERANTE"],
            "Fecha Límite (Datos Reales)": [f_opt, f_cri, "Control Comprometido"]
        }))
    else:
        st.info("⏳ Monitoreando... No se han detectado pulsos de emergencia en el registro actual.")

    # Descarga
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='PREDWEEM_2026')
    st.sidebar.download_button("📥 Descargar Reporte", output.getvalue(), "predweem_2026.xlsx")

else:
    st.warning("⚠️ Cargue el archivo climático para visualizar la simulación.")
