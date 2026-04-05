# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM OPERATIVO vK4.9.8 — LOLIUM TRES ARROYOS 2026
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import io
import base64
import time 
from pathlib import Path

# ---------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILO (Debe ser lo primero)
# ---------------------------------------------------------
st.set_page_config(
    page_title="PREDWEEM TRES ARROYOS vK4.9.8", 
    layout="wide",
    page_icon="🌾"
)

st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    [data-testid="stSidebar"] { background-color: #dcfce7; border-right: 1px solid #bbf7d0; }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p { color: #166534 !important; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .bio-alert { padding: 10px; border-radius: 5px; background-color: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; margin-bottom: 10px; font-size: 0.9em; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ---------------------------------------------------------
# 2. FUNCIONES BASE Y MODELOS
# ---------------------------------------------------------
@st.cache_data
def get_base64_image(main_bg_file):
    try:
        with open(main_bg_file, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        return ""

def set_bg_hack(main_bg_file):
    encoded_string = get_base64_image(main_bg_file)
    if encoded_string:
        st.markdown(
            f"<style>.stApp {{ background-image: url(data:image/png;base64,{encoded_string}); background-size: cover; background-position: center; background-repeat: no-repeat; background-attachment: fixed; }}</style>",
            unsafe_allow_html=True
        )

set_bg_hack("fondo_predweem_v3.png")

def create_mock_files_if_missing():
    if not (BASE / "IW.npy").exists():
        np.save(BASE / "IW.npy", np.random.rand(4, 10))
        np.save(BASE / "bias_IW.npy", np.random.rand(10))
        np.save(BASE / "LW.npy", np.random.rand(1, 10))
        np.save(BASE / "bias_out.npy", np.random.rand(1))
    
    if not (BASE / "modelo_clusters_k3.pkl").exists():
        jd = np.arange(1, 366)
        p1, p2, p3 = np.exp(-((jd - 100)**2)/600), np.exp(-((jd - 160)**2)/900) + 0.3*np.exp(-((jd - 260)**2)/1200), np.exp(-((jd - 230)**2)/1500)
        with open(BASE / "modelo_clusters_k3.pkl", "wb") as f:
            pickle.dump({"JD_common": jd, "curves_interp": [p2, p1, p3], "medoids_k3": [0, 1, 2]}, f)

create_mock_files_if_missing()

def dtw_distance(a, b):
    na, nb = len(a), len(b)
    dp = np.full((na+1, nb+1), np.inf)
    dp[0,0] = 0
    for i in range(1, na+1):
        for j in range(1, nb+1):
            cost = abs(a[i-1] - b[j-1])
            dp[i,j] = cost + min(dp[i-1,j], dp[i,j-1], dp[i-1,j-1])
    return dp[na, nb]

def calculate_tt_scalar(t, t_base, t_opt, t_crit):
    if t <= t_base: return 0.0
    elif t <= t_opt: return t - t_base
    elif t < t_crit: return (t - t_base) * ((t_crit - t) / (t_crit - t_opt))
    else: return 0.0

def calcular_et0_hargreaves(jday, tmax, tmin, latitud=-38.45):
    lat_rad = np.radians(latitud)
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * jday)
    dec = 0.409 * np.sin(2 * np.pi / 365 * jday - 1.39)
    ws = np.arccos(-np.tan(lat_rad) * np.tan(dec))
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (ws * np.sin(lat_rad) * np.sin(dec) + np.cos(lat_rad) * np.cos(dec) * np.sin(ws))
    et0 = 0.0023 * (ra / 2.45) * (((tmax + tmin) / 2.0) + 17.8) * np.sqrt(np.maximum(tmax - tmin, 0))
    return np.maximum(et0, 0)

def balance_hidrico_superficial(prec, et0, w_max=20.0, ke_suelo_max=0.4):
    n = len(prec)
    w = np.zeros(n)
    w[0] = w_max / 2.0  
    for i in range(1, n):
        w[i] = max(0.0, min(w_max, w[i-1] + prec[i] - (et0[i] * (ke_suelo_max * (w[i-1] / w_max)))))
    return w

class PracticalANNModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW, self.bIW, self.LW, self.bLW = IW, bIW, LW, bLW
        self.input_min, self.input_max = np.array([1, 0, -7, 0]), np.array([300, 41, 25.5, 84])

    def predict(self, Xreal):
        Xn = 2 * (Xreal - self.input_min) / (self.input_max - self.input_min) - 1
        emerrel = (np.tanh((np.tanh(Xn @ self.IW + self.bIW) @ self.LW.T).flatten() + self.bLW) + 1) / 2
        return emerrel, np.cumsum(emerrel)

@st.cache_resource
def load_models():
    try:
        ann = PracticalANNModel(np.load(BASE/"IW.npy"), np.load(BASE/"bias_IW.npy"), np.load(BASE/"LW.npy"), np.load(BASE/"bias_out.npy"))
        with open(BASE/"modelo_clusters_k3.pkl", "rb") as f:
            k3 = pickle.load(f)
        return ann, k3
    except Exception as e:
        return None, None

@st.cache_data
def get_data(file_input):
    try:
        if file_input:
            df = pd.read_csv(file_input, parse_dates=["Fecha"]) if file_input.name.endswith('.csv') else pd.read_excel(file_input, parse_dates=["Fecha"])
        else:
            try: df = pd.read_csv("https://raw.githubusercontent.com/PREDWEEM/LOLIUM_TA2026/main/meteo_daily.csv", parse_dates=["Fecha"])
            except: 
                if (BASE / "meteo_daily.csv").exists(): df = pd.read_csv(BASE / "meteo_daily.csv", parse_dates=["Fecha"])
                else: return None
        df.columns = [c.upper().strip() for c in df.columns]
        return df.rename(columns={'FECHA': 'Fecha', 'DATE': 'Fecha', 'TMAX': 'TMAX', 'TMIN': 'TMIN', 'PREC': 'Prec', 'LLUVIA': 'Prec'})
    except: return None

@st.cache_data
def generar_reporte_excel(df, params):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Data_Diaria')
        pd.DataFrame({'Configuracion': list(params.keys()), 'Valor': list(params.values())}).to_excel(writer, sheet_name='Bio_Params', index=False)
    return output.getvalue()


# ---------------------------------------------------------
# 3. PANTALLA DE CARGA CON RE-RUN (Garantiza visibilidad)
# ---------------------------------------------------------
if 'app_cargada' not in st.session_state:
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    st.success("### 🚜 Bienvenido a PREDWEEM Operativo")
    st.info("⏳ **Conectando con el servidor...** Descargando datos climáticos y configurando redes neuronales. Por favor espere unos segundos.")
    
    barra = st.progress(10)
    
    # Renderizamos barra y pre-cargamos en caché
    time.sleep(0.5) 
    barra.progress(40)
    load_models() 
    
    barra.progress(70)
    get_data(None) 
    
    barra.progress(100)
    time.sleep(0.5) 
    
    # Marcamos como cargada y REINICIAMOS la aplicación
    st.session_state.app_cargada = True
    st.rerun() 
    
    # El código se detiene acá en la primera pasada y vuelve al inicio de la página.
    # Como los modelos ya se metieron en caché en la línea 170 y 173, la segunda pasada
    # será instantánea.


# ---------------------------------------------------------
# A PARTIR DE AQUÍ SOLO SE EJECUTA CUANDO YA ESTÁ TODO CARGADO
# ---------------------------------------------------------
modelo_ann, cluster_model = load_models()

# ---------------------------------------------------------
# 4. INTERFAZ Y SIDEBAR
# ---------------------------------------------------------
st.title("🌾 PREDWEEM LOLIUM - TRES ARROYOS (BA) lat=-38.378223 lon=-60.276321 ")

with st.expander("📂 1. Datos del Lote", expanded=True):
    col_upload, col_rastrojo = st.columns(2)
    
    with col_upload:
        archivo_usuario = st.file_uploader("Subir Clima Manual (TRES ARROYOS)", type=["xlsx", "csv"])
        with st.spinner("⏳ Analizando datos meteorológicos..."):
            df = get_data(archivo_usuario)
        
    with col_rastrojo:
        tipo_manejo = st.selectbox(
            "Nivel de Rastrojo",
            options=["Cobertura Muy Densa (SD - Extra Rastrojo/CS)", "Alta Cobertura (SD - Rastrojo Trigo/Maíz)", "Cobertura Media (SD - Rastrojo Soja)", "Baja Cobertura / Labranza Convencional"],
            index=1 
        )
        if "Muy Densa" in tipo_manejo: ke_val, mod_termico = 0.10, 0.80 
        elif "Alta" in tipo_manejo: ke_val, mod_termico = 0.25, 0.90 
        elif "Media" in tipo_manejo: ke_val, mod_termico = 0.50, 0.95 
        else: ke_val, mod_termico = 0.95, 1.00 
        st.caption(f"Coeficiente Ke interno aplicado: **{ke_val:.2f}** | Modulador Térmico Suelo: **{mod_termico:.2f}**")

LOGO_URL = "https://raw.githubusercontent.com/PREDWEEM/LOLIUM_TA2026/main/logo.png"
st.sidebar.image(LOGO_URL, use_container_width=True)

st.sidebar.markdown("## ⚙️ 2. Fisiología y Logística")
umbral_er = st.sidebar.slider("Umbral Tasa Diaria (Detección pico)", 0.05, 0.80, 0.30)
st.sidebar.markdown("**Ruptura de Dormición Estival (Escudo)**")
umbral_termoinhibicion = st.sidebar.number_input("Umbral Termoinhibición (°C)", min_value=15.0, max_value=35.0, value=24.0, step=0.5)
umbral_choque_hidrico = st.sidebar.slider("Choque Hídrico 3 días (mm)", 20.0, 100.0, 45.0)

col_t1, col_t2 = st.sidebar.columns(2)
with col_t1: t_base_val = st.number_input("T Base", value=2.0, step=0.5)
with col_t2: t_opt_max = st.number_input("T Óptima Max", value=20.0, step=1.0)
t_critica = st.sidebar.slider("T Crítica (Stop)", 26.0, 42.0, 30.0)

st.sidebar.markdown("**Objetivos (°Cd)**")
dga_optimo = st.sidebar.number_input("Objetivo Control", value=600, step=50)
dga_critico = st.sidebar.number_input("Límite Ventana", value=800, step=50)

st.sidebar.divider()
st.sidebar.markdown("## 💧 3. Balance Hídrico (Suelo)")
w_max_val = st.sidebar.number_input("Cap. de Campo Superficial (mm)", value=20.0, step=1.0)


# ---------------------------------------------------------
# 5. MOTOR DE CÁLCULO (LÓGICA 100% MECANÍSTICA)
# ---------------------------------------------------------
if df is not None and modelo_ann is not None:
    with st.spinner("⚙️ Recalculando balance hídrico y redes neuronales..."):
        df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
        df["Julian_days"] = df["Fecha"].dt.dayofyear
        df["Tmedia_aire"] = (df["TMAX"] + df["TMIN"]) / 2
        amplitud_termica = (df["TMAX"] - df["TMIN"]) / 2
        df["TMAX_suelo"] = df["Tmedia_aire"] + (amplitud_termica * mod_termico)
        df["TMIN_suelo"] = df["Tmedia_aire"] - (amplitud_termica * mod_termico)

        X = df[["Julian_days", "TMAX_suelo", "TMIN_suelo", "Prec"]].to_numpy(float)
        emerrel_raw, _ = modelo_ann.predict(X)
        df["EMERREL"] = np.maximum(emerrel_raw, 0.0)
        
        df["Prec_3d"] = df["Prec"].rolling(window=3, min_periods=1).sum()
        mask_ruptura = (df["Julian_days"] <= 110) & (df["Prec_3d"] >= umbral_choque_hidrico)
        df.loc[mask_ruptura, "EMERREL"] = np.maximum(df.loc[mask_ruptura, "EMERREL"], 0.75)

        df["ET0"] = calcular_et0_hargreaves(df["Julian_days"].values, df["TMAX"].values, df["TMIN"].values, latitud=-38.45)
        df["W_superficial"] = balance_hidrico_superficial(df["Prec"].values, df["ET0"].values, w_max=w_max_val, ke_suelo_max=ke_val)
        
        humedad_relativa = df["W_superficial"] / w_max_val
        df["Hydric_Factor"] = 1 / (1 + np.exp(-10 * (humedad_relativa - 0.3)))
        df["EMERREL"] = df["EMERREL"] * df["Hydric_Factor"]

        df.loc[humedad_relativa < 0.20, "EMERREL"] = 0.0
        df['Lluvia_Recarga'] = (df['Prec'] >= w_max_val).cummax()
        df.loc[~df['Lluvia_Recarga'], "EMERREL"] = 0.0

        df["Tmedia"] = df["Tmedia_aire"]
        df["Tmedia_10d"] = df["Tmedia"].rolling(window=10, min_periods=1).mean()
        df.loc[df["Tmedia_10d"] >= umbral_termoinhibicion, "EMERREL"] = 0.0
        
        df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, t_base_val, t_opt_max, t_critica))
        
        fecha_hoy = pd.Timestamp.now().normalize() 
        if fecha_hoy not in df['Fecha'].values: fecha_hoy = df['Fecha'].max()
        
        indices_pulso = df.index[df["EMERREL"] >= umbral_er].tolist()
        dga_hoy, dga_7dias, fecha_inicio_ventana, msg_estado = 0.0, 0.0, None, "Esperando pico de emergencia..."

        if indices_pulso:
            fecha_inicio_ventana = df.loc[indices_pulso[0], "Fecha"]
            df_desde_pico = df[df["Fecha"] >= fecha_inicio_ventana].copy()
            df_desde_pico["DGA_cum"] = df_desde_pico["DG"].cumsum()
            dga_hoy = df.loc[(df["Fecha"] >= fecha_inicio_ventana) & (df["Fecha"] <= fecha_hoy), "DG"].sum()
            idx_hoy = df[df["Fecha"] == fecha_hoy].index[0]
            dga_7dias = dga_hoy + df.iloc[idx_hoy + 1 : idx_hoy + 8]["DG"].sum()
            msg_estado = f"Pico detectado el {fecha_inicio_ventana.strftime('%d/%m')}"
            dias_stress = len(df_desde_pico[df_desde_pico["Tmedia"] > t_opt_max])

    # -----------------------------------------------------
    # VISUALIZACIÓN FRONT-END
    # -----------------------------------------------------
    colorscale_hard = [[0.0, "green"], [0.29, "green"], [0.30, "red"], [1.0, "red"]]
    fig_risk = go.Figure(data=go.Heatmap(z=[df["EMERREL"].values], x=df["Fecha"], y=["Emergencia"], colorscale=colorscale_hard, zmin=0, zmax=1, showscale=False))
    fig_risk.update_layout(height=120, margin=dict(t=30, b=0, l=10, r=10), title="Mapa de Riesgo (Tasa Diaria)")
    st.plotly_chart(fig_risk, use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 MONITOR DE DECISIÓN", "💧 PRECIPITACIONES Y SUELO", "📈 ANÁLISIS ESTRATÉGICO", "🧪 BIO-CALIBRACIÓN"])

    with tab1:
        col_main, col_gauge = st.columns([2, 1])
        with col_main:
            fig_emer = go.Figure()
            fig_emer.add_trace(go.Scatter(x=df["Fecha"], y=df["EMERREL"], mode='lines', name='Tasa Diaria Simulada', line=dict(color='#166534', width=2.5), fill='tozeroy', fillcolor='rgba(22, 101, 52, 0.1)'))
            fig_emer.add_hline(y=umbral_er, line_dash="dash", line_color="orange", annotation_text=f"Umbral Alerta ({umbral_er})")
            fig_emer.update_layout(title="Dinámica de Emergencia y Detección de Picos", height=350, hovermode="x unified")
            st.plotly_chart(fig_emer, use_container_width=True)

            if fecha_inicio_ventana:
                st.success(f"📅 **Inicio de Conteo Térmico:** {fecha_inicio_ventana.strftime('%d-%m-%Y')} (Primer pico detectado)")
                if dias_stress > 0: st.markdown(f"""<div class="bio-alert">🔥 <b>Estrés Térmico:</b> {dias_stress} días con T > {t_opt_max}°C desde el inicio.</div>""", unsafe_allow_html=True)
            else: st.warning(f"⏳ Esperando primera alerta (Tasa diaria >= {umbral_er}).")

        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = dga_hoy, domain = {'x': [0, 1], 'y': [0, 1]}, title = {'text': f"<b>TT ACUMULADO (°Cd)</b>", 'font': {'size': 18}},
                gauge = {'axis': {'range': [None, dga_critico * 1.2]}, 'bar': {'color': "#1e293b", 'thickness': 0.3},
                         'steps': [{'range': [0, dga_optimo], 'color': "#4ade80"}, {'range': [dga_optimo, dga_critico], 'color': "#facc15"}, {'range': [dga_critico, dga_critico * 1.2], 'color': "#f87171"}],
                         'threshold': {'line': {'color': "#2563eb", 'width': 6}, 'thickness': 0.8, 'value': dga_7dias}}
            ))
            fig_gauge.add_annotation(x=0.5, y=-0.1, text=f"{msg_estado}<br>Pronóstico +7d: <b>{dga_7dias:.1f} °Cd</b>", showarrow=False, font=dict(size=14, color="#1e3a8a"), align="center")
            fig_gauge.update_layout(height=350, margin=dict(t=80, b=50, l=30, r=30))
            st.plotly_chart(fig_gauge, use_container_width=True)

    with tab2:
        st.header("💧 Dinámica Hídrica del Suelo (Balance Superficial)")
        fig_hidrico = go.Figure()
        fig_hidrico.add_trace(go.Bar(x=df["Fecha"], y=df["Prec"], name='Lluvia Diaria (mm)', marker_color='#93c5fd', opacity=0.7))
        fig_hidrico.add_trace(go.Scatter(x=df["Fecha"], y=df["W_superficial"], name='Agua en Suelo (0-10cm)', mode='lines', line=dict(color='#0284c7', width=3), fill='tozeroy', fillcolor='rgba(2, 132, 199, 0.2)'))
        fig_hidrico.add_hline(y=w_max_val, line_dash="dot", line_color="#334155", annotation_text=f"Capacidad Máx. ({w_max_val} mm)", annotation_position="top left")
        fig_hidrico.update_layout(title="Precipitación vs. Retención Real de Humedad", xaxis_title="Fecha", yaxis_title="Milímetros (mm)", height=450, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_hidrico, use_container_width=True)
                    
    with tab3:
        st.header("🔍 Clasificación DTW (Tres Arroyos)")
        df_obs = df[df["Fecha"] < pd.Timestamp("2026-05-01")].copy()
        if not df_obs.empty and df_obs["EMERREL"].sum() > 0:
            jd_corte = df_obs["Julian_days"].max()
            JD_COM = cluster_model["JD_common"]
            obs_norm = np.interp(JD_COM[JD_COM <= jd_corte], df_obs["Julian_days"], df_obs["EMERREL"] / (df_obs["EMERREL"].max() if df_obs["EMERREL"].max() > 0 else 1.0))
            dists = [dtw_distance(obs_norm, m[JD_COM <= jd_corte] / (m[JD_COM <= jd_corte].max() if m[JD_COM <= jd_corte].max() > 0 else m[JD_COM <= jd_corte])) for m in cluster_model["curves_interp"]]
            pred = int(np.argmin(dists))
            names, cols = {0: "🌾 Bimodal", 1: "🌱 Temprano", 2: "🍂 Tardío"}, {0: "#0284c7", 1: "#16a34a", 2: "#ea580c"}
            c1, c2 = st.columns([3, 1])
            with c1:
                fp = go.Figure()
                fp.add_trace(go.Scatter(x=JD_COM, y=cluster_model["curves_interp"][pred], name="Patrón Histórico", line=dict(dash='dash', color=cols.get(pred))))
                fp.add_trace(go.Scatter(x=JD_COM[JD_COM <= jd_corte], y=obs_norm * cluster_model["curves_interp"][pred].max(), name="2026", line=dict(color='black', width=3)))
                st.plotly_chart(fp, use_container_width=True)
            with c2: st.success(f"### {names.get(pred)}"); st.metric("DTW Score", f"{min(dists):.2f}")
        else: st.info("Datos insuficientes para clasificación DTW.")

    with tab4:
        st.subheader("🧪 Curva de Respuesta Fisiológica")
        x_temps = np.linspace(0, 45, 200)
        fig_bio = go.Figure(go.Scatter(x=x_temps, y=[calculate_tt_scalar(t, t_base_val, t_opt_max, t_critica) for t in x_temps], mode='lines', line=dict(color='#2563eb', width=4), fill='tozeroy'))
        st.plotly_chart(fig_bio, use_container_width=True)

    excel_data = generar_reporte_excel(df, {'T_Base': t_base_val, 'T_Optima': t_opt_max, 'T_Critica': t_critica, 'W_Max': w_max_val, 'Ke': ke_val, 'Mod_Termico': mod_termico, 'Umbral_Termoinhibicion': umbral_termoinhibicion})
    st.sidebar.download_button("📥 Descargar Reporte", excel_data, "PREDWEEM_Operativo_TresArroyos_vK4_9_8.xlsx")

else:
    st.info("👋 Bienvenido a PREDWEEM. Cargue datos meteorológicos para comenzar.")
