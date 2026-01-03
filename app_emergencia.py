# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io
from pathlib import Path

# ---------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILO
# ---------------------------------------------------------
st.set_page_config(
    page_title="PREDWEEM – LOLIUM TRES ARROYOS 2026", 
    layout="wide",
    page_icon="🌾"
)

# URL del Logo (Formato Raw)
LOGO_URL = "https://raw.githubusercontent.com/PREDWEEM/loliumTA_2026/main/logo.png"
st.sidebar.image(LOGO_URL, use_container_width=True)

# Inyección de CSS para personalizar la apariencia
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
        # Valores de normalización fijos (entrenamiento original)
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])

    def normalize(self, X):
        # Escalado min-max entre -1 y 1
        return 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1

    def predict(self, Xreal):
        Xn = self.normalize(Xreal)
        emer = []
        # Propagación hacia adelante (Feed-forward)
        for x in Xn:
            z1 = self.IW.T @ x + self.bIW
            a1 = np.tanh(z1)               # Capa Oculta
            z2 = self.LW @ a1 + self.bLW
            emer.append(np.tanh(z2))       # Capa Salida
        
        # Desnormalizar salida
        emer = (np.array(emer) + 1) / 2
        
        # Calcular acumulada y relativa
        emer_ac = np.cumsum(emer)
        emerrel = np.diff(emer_ac, prepend=0)
        return emerrel, emer_ac

@st.cache_resource
def load_models():
    """Carga los pesos y bias de la red neuronal desde archivos .npy"""
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
    """Procesa el archivo de clima (Usuario o Default)"""
    try:
        if file_input is not None:
            # Detectar formato
            if file_input.name.endswith('.csv'):
                df = pd.read_csv(file_input, parse_dates=["Fecha"]) 
            else:
                df = pd.read_excel(file_input, parse_dates=["Fecha"])
        else:
            # Carga por defecto desde repo/local
            path_github = BASE / "meteo_daily.csv"
            if path_github.exists():
                df = pd.read_csv(path_github, parse_dates=["Fecha"])
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
        
        # Validación mínima
        required_cols = ["Fecha", "TMAX", "TMIN", "Prec"]
        if not all(col in df.columns for col in required_cols):
            st.error(f"El archivo debe contener las columnas: {required_cols}")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error procesando datos: {e}")
        return None

# ---------------------------------------------------------
# 3. INTERFAZ Y PROCESAMIENTO PRINCIPAL
# ---------------------------------------------------------
modelo_ann = load_models()

# --- SIDEBAR ---
st.sidebar.markdown("## 🌾 PREDWEEM")
st.sidebar.markdown("### LOLIUM TRES ARROYOS 2026")

archivo_usuario = st.sidebar.file_uploader("Subir Clima Manual (Opcional)", type=["xlsx", "csv"])
df = get_data(archivo_usuario)

st.sidebar.divider()
st.sidebar.markdown("**Parámetros de Simulación**")
umbral_er = st.sidebar.slider("Umbral de Alerta (Emergencia Diaria)", 0.05, 0.80, 0.50)
dga_optimo = st.sidebar.slider("Umbral Térmico Óptimo (°Cd)", 50, 800, 600)
dga_critico = st.sidebar.slider("Umbral Térmico Crítico (°Cd)", 600, 1200, 850)

# --- CUERPO PRINCIPAL ---
if df is not None and modelo_ann is not None:
    
    # 1. Preprocesamiento
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    
    # 2. Predicción con la Red Neuronal
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    emerrel, _ = modelo_ann.predict(X)
    
    # Limpieza post-predicción
    df["EMERREL"] = np.maximum(emerrel, 0.0)
    # Forzamos cero los primeros 15 días del año (ruido biológico)
    df.loc[df["Julian_days"] <= 15, "EMERREL"] = 0.0
    
    # 3. Cálculo de Grados Día (Base 2.0°C para Lolium)
    df["DG"] = np.maximum(((df["TMAX"] + df["TMIN"]) / 2) - 2.0, 0)
    
    # Variable 'Riesgo' relativa al máximo (Opcional, pero calculada por si se usa luego)
    max_er = df["EMERREL"].max()
    df["Riesgo_Norm"] = df["EMERREL"] / max_er if max_er > 0 else 0.0

    st.title("🌾 PREDWEEM | LOLIUM TRES ARROYOS 2026")

    # -----------------------------------------------------
    # VISUALIZACIÓN A: MAPA DE CALOR (Heatmap)
    # -----------------------------------------------------
    # Lógica de color basada estrictamente en valor de EMERREL:
    # 0.00 - 0.49: Verde
    # 0.50 - 0.90: Amarillo
    # > 0.90     : Rojo
    
    colorscale_hard = [
        [0.00, "green"],  
        [0.49, "green"],  
        [0.49, "yellow"], # Corte duro
        [0.90, "yellow"], 
        [0.90, "red"],    # Corte duro
        [1.00, "red"]     
    ]

    fig_risk = go.Figure(data=go.Heatmap(
        z=[df["EMERREL"].values],  # Usamos el valor directo de emergencia
        x=df["Fecha"], 
        y=["Emergencia"],
        colorscale=colorscale_hard,
        zmin=0, zmax=1,            # Fijamos escala 0-1 para que los cortes % funcionen
        showscale=False,
        hovertemplate="<b>%{x|%d-%b}</b><br>Tasa Emergencia: %{z:.3f}<extra></extra>"
    ))
    fig_risk.update_layout(
        height=130, 
        margin=dict(t=30, b=0, l=10, r=10), 
        title="Mapa de Intensidad: Emergencia Relativa Diaria"
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    # -----------------------------------------------------
    # VISUALIZACIÓN B: SERIE DE TIEMPO
    # -----------------------------------------------------
    fig_emer = go.Figure()
    fig_emer.add_trace(go.Scatter(
        x=df["Fecha"], y=df["EMERREL"], 
        mode='lines', name='Emergencia Diaria',
        line=dict(color='#166534', width=2.5),
        fill='tozeroy', fillcolor='rgba(22, 101, 52, 0.1)'
    ))
    fig_emer.add_hline(
        y=umbral_er, 
        line_dash="dash", line_color="orange", 
        annotation_text=f"Umbral Alerta ({umbral_er})", 
        annotation_position="top right"
    )
    fig_emer.update_layout(
        title="Dinámica de Emergencia Relativa", 
        yaxis_title="Emergencia Relativa (%)",
        height=350, 
        margin=dict(t=40, b=40)
    )
    st.plotly_chart(fig_emer, use_container_width=True)

    # -----------------------------------------------------
    # LÓGICA DE DECISIÓN Y CRONOGRAMA
    # -----------------------------------------------------
    # Detectar inicio de ventana: Primer pulso sostenido sobre el umbral
    indices_pulso = df.index[df["EMERREL"] >= umbral_er].tolist()
    fecha_inicio_ventana = None
    
    # Buscamos dos días cercanos con alta emergencia para confirmar inicio
    for i in range(len(indices_pulso) - 1):
        delta_dias = (df.loc[indices_pulso[i+1], "Fecha"] - df.loc[indices_pulso[i], "Fecha"]).days
        if delta_dias <= 5:
            fecha_inicio_ventana = df.loc[indices_pulso[i], "Fecha"]
            break

    if fecha_inicio_ventana:
        st.divider()
        st.header("🗓️ Cronograma y Ventana de Acción")
        
        # Filtramos datos desde el inicio de la ventana biológica
        df_ventana = df[df["Fecha"] >= fecha_inicio_ventana].copy()
        
        # Acumulación de tiempo térmico desde ese momento
        df_ventana["DGA_cum"] = df_ventana["DG"].cumsum()
        dga_actual_acumulado = df_ventana["DGA_cum"].iloc[-1]

        # Función auxiliar para determinar fechas límite
        def obtener_estado(objetivo_termico):
            # Si ya acumulamos el calor suficiente en los datos históricos/actuales
            if dga_actual_acumulado >= objetivo_termico:
                # Buscar el día exacto donde se cruzó el umbral
                row_cruce = df_ventana[df_ventana["DGA_cum"] >= objetivo_termico].iloc[0]
                return row_cruce["Fecha"].strftime("%d-%m-%Y"), "PASADO"
            else:
                return "Proyección Futura", "PENDIENTE"

        f_opt, status_opt = obtener_estado(dga_optimo)
        f_cri, status_cri = obtener_estado(dga_critico)

        # 1. Métricas KPI
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Inicio de Cohorte", fecha_inicio_ventana.strftime("%d-%b"))
        kpi2.metric("Suma Térmica Actual", f"{dga_actual_acumulado:.1f} °Cd")
        kpi3.metric("Fecha Límite Óptima", f_opt)

        # 2. Tabla de Estados
        st.subheader("Estados Fenológicos")
        datos_tabla = {
            "Nivel de Alerta": ["🟢 ÓPTIMO", "🟡 CRÍTICO", "🔴 TARDIÓ"],
            "Fenología Estimada": ["Emergencia - 1 Hoja", "Macollaje Inicio", "Macollaje Avanzado"],
            "Umbral Térmico": [f"Hasat {dga_optimo} °Cd", f"{dga_optimo} - {dga_critico} °Cd", f"> {dga_critico} °Cd"],
            "Fecha Límite": [f_opt, f_cri, "Control Comprometido"],
            "Estado": [status_opt, status_cri, "NO RECOMENDADO"]
        }
        st.table(pd.DataFrame(datos_tabla))

        # 3. Mensajes Contextuales
        if status_opt == "PENDIENTE":
            st.info(f"⏳ La ventana está abierta. Faltan {dga_optimo - dga_actual_acumulado:.1f} °Cd para el límite óptimo.")
        elif status_cri == "PENDIENTE":
            st.warning(f"⚠️ **ATENCIÓN:** Se ha superado la fecha óptima ({f_opt}). Estás en la ventana crítica hasta {f_cri}.")
        else:
            st.error(f"🚫 **ALERTA ROJA:** Se ha superado el límite crítico ({f_cri}). Eficacia de control reducida.")
            
    else:
        st.info(f"⏳ **Sistema en Espera:** No se han detectado pulsos de emergencia significativos (>= {umbral_er}) para iniciar el conteo térmico.")

    # -----------------------------------------------------
    # EXPORTACIÓN
    # -----------------------------------------------------
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='PREDWEEM_Data')
        # Hoja extra con metadatos
        pd.DataFrame({'Variable':['Umbral Alerta', 'Optimo', 'Critico'], 'Valor':[umbral_er, dga_optimo, dga_critico]}).to_excel(writer, sheet_name='Params', index=False)
        
    st.sidebar.download_button(
        label="📥 Descargar Datos Completos",
        data=output.getvalue(),
        file_name="PREDWEEM_Resultados.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    # Pantalla de bienvenida / espera
    st.warning("⚠️ **Esperando datos.** Por favor sube un archivo CSV/Excel o asegúrate que 'meteo_daily.csv' esté en el repositorio.")

st.sidebar.caption("PREDWEEM vK3 | Tres Arroyos 2026")
