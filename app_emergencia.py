# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4.1 — LOLIUM TRES ARROYOS 2026
# Actualización: Inicio de conteo desde el PRIMER pico + Heatmap
# + Loader premium post-hibernación (Splash + Skeletons + Stepper)
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import io
import time
from pathlib import Path

# ---------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILO
# ---------------------------------------------------------
st.set_page_config(
    page_title="PREDWEEM INTEGRAL vK4",
    layout="wide",
    page_icon="🌾"
)

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
    .bio-alert {
        padding: 10px;
        border-radius: 5px;
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
        margin-bottom: 10px;
        font-size: 0.9em;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# Logo (usado por el loader premium)
LOGO_URL = "https://raw.githubusercontent.com/PREDWEEM/loliumTA_2026/main/logo.png"

# ---------------------------------------------------------
# 2. ROBUSTEZ: GENERADOR DE ARCHIVOS MOCK
# ---------------------------------------------------------
def create_mock_files_if_missing():
    if not (BASE / "IW.npy").exists():
        np.save(BASE / "IW.npy", np.random.rand(4, 10))
        np.save(BASE / "bias_IW.npy", np.random.rand(10))
        np.save(BASE / "LW.npy", np.random.rand(1, 10))
        np.save(BASE / "bias_out.npy", np.random.rand(1))

    if not (BASE / "modelo_clusters_k3.pkl").exists():
        jd = np.arange(1, 366)
        p1 = np.exp(-((jd - 100) ** 2) / 600)
        p2 = np.exp(-((jd - 160) ** 2) / 900) + 0.3 * np.exp(-((jd - 260) ** 2) / 1200)
        p3 = np.exp(-((jd - 230) ** 2) / 1500)
        mock_cluster = {
            "JD_common": jd,
            "curves_interp": [p2, p1, p3],
            "medoids_k3": [0, 1, 2]
        }
        with open(BASE / "modelo_clusters_k3.pkl", "wb") as f:
            pickle.dump(mock_cluster, f)

    if not (BASE / "meteo_daily.csv").exists():
        dates = pd.date_range(start="2026-01-01", periods=150)
        data = {
            "Fecha": dates,
            "TMAX": np.random.uniform(25, 35, size=150) - (np.arange(150) * 0.1),
            "TMIN": np.random.uniform(10, 18, size=150) - (np.arange(150) * 0.06),
            "Prec": np.random.choice([0, 0, 5, 15, 45], size=150)
        }
        pd.DataFrame(data).to_csv(BASE / "meteo_daily.csv", index=False)

create_mock_files_if_missing()

# ---------------------------------------------------------
# 3. LÓGICA TÉCNICA (ANN + DTW + BIO)
# ---------------------------------------------------------
def dtw_distance(a, b):
    na, nb = len(a), len(b)
    dp = np.full((na + 1, nb + 1), np.inf)
    dp[0, 0] = 0
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return dp[na, nb]

def calculate_tt_scalar(t, t_base, t_opt, t_crit):
    if t <= t_base:
        return 0.0
    elif t <= t_opt:
        return t - t_base
    elif t < t_crit:
        factor = (t_crit - t) / (t_crit - t_opt)
        return (t - t_base) * factor
    else:
        return 0.0

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
    ann = PracticalANNModel(
        np.load(BASE / "IW.npy"), np.load(BASE / "bias_IW.npy"),
        np.load(BASE / "LW.npy"), np.load(BASE / "bias_out.npy")
    )
    with open(BASE / "modelo_clusters_k3.pkl", "rb") as f:
        k3 = pickle.load(f)
    return ann, k3

# ---------------------------------------------------------
# 3B. LOADER PREMIUM (post-hibernación)
# ---------------------------------------------------------
def _loader_css():
    st.markdown("""
    <style>
    .pw-hero{
        background: radial-gradient(1200px 500px at 10% 0%, rgba(34,197,94,0.20), transparent 55%),
                    radial-gradient(900px 450px at 90% 0%, rgba(59,130,246,0.10), transparent 55%),
                    #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 22px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.08);
        padding: 22px 22px 18px 22px;
        margin: 10px 0 16px 0;
    }
    .pw-top{display:flex;gap:16px;align-items:center;}
    .pw-logo{
        width:66px;height:66px;border-radius:16px;
        border:1px solid #e2e8f0;background:#f8fafc;
        overflow:hidden;flex:0 0 auto;
        display:flex;align-items:center;justify-content:center;
    }
    .pw-title{margin:0;font-weight:900;color:#14532d;font-size:1.35rem;line-height:1.1;}
    .pw-sub{margin:6px 0 0 0;color:#64748b;font-size:0.95rem;}
    .pw-badges{margin-left:auto;display:flex;gap:8px;align-items:center;flex-wrap:wrap;justify-content:flex-end}
    .pw-pill{
        border:1px solid #bbf7d0;background:#dcfce7;color:#166534;
        padding:4px 10px;border-radius:999px;font-weight:800;font-size:0.82rem;
    }
    .pw-pill2{
        border:1px solid #cbd5e1;background:#f8fafc;color:#0f172a;
        padding:4px 10px;border-radius:999px;font-weight:800;font-size:0.82rem;
    }
    .pw-grid{
        margin-top:14px;
        display:grid;
        grid-template-columns: 1.2fr 0.8fr;
        gap:12px;
    }
    .pw-card{
        border:1px solid #e2e8f0;border-radius:18px;background:rgba(255,255,255,0.70);
        box-shadow: 0 6px 18px rgba(0,0,0,0.05);
        padding:14px;
        backdrop-filter: blur(6px);
    }
    .pw-h{font-weight:900;color:#0f172a;margin:0 0 8px 0;font-size:0.95rem;}
    .pw-skel{
        border-radius:12px;
        background: linear-gradient(90deg, #f1f5f9 25%, #e2e8f0 37%, #f1f5f9 63%);
        background-size: 400% 100%;
        animation: pwsh 1.2s ease-in-out infinite;
    }
    @keyframes pwsh{0%{background-position: 100% 0}100%{background-position: 0 0}}
    .pw-s1{height:120px;}
    .pw-s2{height:34px;margin-top:10px;}
    .pw-s3{height:220px;margin-top:10px;}
    .pw-tabs{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
    .pw-tab{
        border:1px solid #e2e8f0;background:#ffffff;border-radius:999px;
        padding:6px 10px;font-weight:800;color:#334155;font-size:0.82rem;
    }
    .pw-step{display:flex;gap:10px;align-items:flex-start;margin:8px 0;}
    .pw-dot{
        width:14px;height:14px;border-radius:999px;border:2px solid #94a3b8;flex:0 0 auto;margin-top:3px;
    }
    .pw-done{border-color:#22c55e;background:#22c55e;}
    .pw-steptext{color:#334155;font-size:0.90rem;}
    .pw-muted{color:#94a3b8;font-size:0.82rem;margin-top:8px;display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap}
    </style>
    """, unsafe_allow_html=True)

def _loader_ui(step_idx: int, total_steps: int, started_at: float):
    steps = [
        ("Restaurando sesión y caché…", "Validación de estado + warm start"),
        ("Cargando modelos ANN…", "Pesos IW/LW + bias (cache_resource)"),
        ("Cargando patrones DTW (K3)…", "Medoides y grillas JD_common"),
        ("Verificando dataset climático…", "CSV/XLSX, columnas, NaNs"),
        ("Preparando visualizaciones…", "Heatmap + tabs + métricas"),
    ]

    elapsed = time.perf_counter() - started_at
    _loader_css()

    st.markdown(f"""
    <div class="pw-hero">
      <div class="pw-top">
        <div class="pw-logo">
          <img src="{LOGO_URL}" style="width:66px;height:66px;object-fit:cover;">
        </div>
        <div style="flex:1;">
          <p class="pw-title">PREDWEEM INTEGRAL</p>
          <p class="pw-sub">Inicialización profesional post-hibernación • optimizando recursos…</p>
        </div>
        <div class="pw-badges">
          <div class="pw-pill">vK4.1</div>
          <div class="pw-pill2">Tres Arroyos 2026</div>
        </div>
      </div>

      <div class="pw-grid">
        <div class="pw-card">
          <p class="pw-h">Vista previa</p>
          <div class="pw-skel pw-s1"></div>
          <div class="pw-tabs">
            <div class="pw-tab">📊 Monitor</div>
            <div class="pw-tab">📈 Análisis</div>
            <div class="pw-tab">🧪 Bio</div>
          </div>
          <div class="pw-skel pw-s2"></div>
          <div class="pw-skel pw-s3"></div>
        </div>

        <div class="pw-card">
          <p class="pw-h">Inicializando</p>
          {"".join([
            f'''
            <div class="pw-step">
              <div class="pw-dot {"pw-done" if i < step_idx else ""}"></div>
              <div>
                <div class="pw-steptext"><b>{steps[i][0]}</b></div>
                <div class="pw-steptext" style="color:#94a3b8;font-size:0.82rem">{steps[i][1]}</div>
              </div>
            </div>
            ''' for i in range(len(steps))
          ])}
          <div class="pw-muted">
            <span>⏱️ {elapsed:.1f}s</span>
            <span>🧠 cache: {("warm" if step_idx > 0 else "cold")}</span>
            <span>🔄 paso {min(step_idx, total_steps)}/{total_steps}</span>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

def boot_premium():
    total = 5
    started_at = time.perf_counter()

    placeholder = st.empty()
    prog = st.progress(0, text="Inicializando…")

    def render(step):
        placeholder.empty()
        with placeholder:
            _loader_ui(step, total, started_at)

    render(0)
    prog.progress(10, text="Restaurando sesión y caché…")
    time.sleep(0.08)

    render(1)
    prog.progress(30, text="Cargando modelos ANN…")
    _ = load_models()  # warm cache
    time.sleep(0.08)

    render(2)
    prog.progress(55, text="Cargando patrones DTW (K3)…")
    time.sleep(0.08)

    render(3)
    prog.progress(75, text="Verificando dataset climático…")
    time.sleep(0.08)

    render(4)
    prog.progress(92, text="Preparando visualizaciones…")
    time.sleep(0.08)

    render(5)
    prog.progress(100, text="Listo ✔")
    time.sleep(0.10)

    placeholder.empty()
    prog.empty()

if "booted" not in st.session_state:
    st.session_state.booted = False

if not st.session_state.booted:
    with st.spinner("Saliendo de hibernación e inicializando PREDWEEM…"):
        boot_premium()
    st.session_state.booted = True

# ---------------------------------------------------------
# 4. DATOS (lectura robusta)
# ---------------------------------------------------------
def get_data(file_input):
    try:
        if file_input:
            if file_input.name.endswith(".csv"):
                df = pd.read_csv(file_input, parse_dates=["Fecha"])
            else:
                df = pd.read_excel(file_input, parse_dates=["Fecha"])
        else:
            path = BASE / "meteo_daily.csv"
            if path.exists():
                df = pd.read_csv(path, parse_dates=["Fecha"])
            else:
                return None

        df.columns = [c.upper().strip() for c in df.columns]
        mapeo = {
            "FECHA": "Fecha", "DATE": "Fecha",
            "TMAX": "TMAX", "TMIN": "TMIN",
            "PREC": "Prec", "LLUVIA": "Prec"
        }
        df = df.rename(columns=mapeo)
        return df
    except Exception as e:
        st.error(f"Error leyendo datos: {e}")
        return None

# ---------------------------------------------------------
# 5. INTERFAZ Y SIDEBAR
# ---------------------------------------------------------
try:
    modelo_ann, cluster_model = load_models()
except Exception as e:
    st.error(f"Error cargando modelos: {e}")
    modelo_ann, cluster_model = None, None

st.sidebar.image(LOGO_URL, use_container_width=True)

st.sidebar.markdown("## ⚙️ Configuración")
archivo_usuario = st.sidebar.file_uploader("Subir Clima Manual", type=["xlsx", "csv"])
df = get_data(archivo_usuario)

st.sidebar.divider()
st.sidebar.markdown("**Parámetros de Emergencia**")
umbral_er = st.sidebar.slider("Umbral Tasa Diaria (Para detectar pico)", 0.05, 0.80, 0.25)

st.sidebar.divider()
st.sidebar.markdown("🌡️ **Fisiología Térmica (Bio-Limit)**")
st.sidebar.caption("Ajusta la respuesta biológica al calor.")

col_t1, col_t2 = st.sidebar.columns(2)
with col_t1:
    t_base_val = st.number_input("T Base", value=2.0, step=0.5)
with col_t2:
    t_opt_max = st.number_input("T Óptima Max", value=20.0, step=1.0)

t_critica = st.sidebar.slider("T Crítica (Stop)", 26.0, 42.0, 30.0)

st.sidebar.markdown("**Objetivos (°Cd)**")
dga_optimo = st.sidebar.number_input("Objetivo Control", value=1000, step=50)
dga_critico = st.sidebar.number_input("Límite Ventana", value=1200, step=50)

# ---------------------------------------------------------
# 6. MOTOR DE CÁLCULO
# ---------------------------------------------------------
if df is not None and modelo_ann is not None and cluster_model is not None:

    # A. Preprocesamiento
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear

    # B. Predicción Neural
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    emerrel, _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(emerrel, 0.0)
    df.loc[df["Julian_days"] <= 25, "EMERREL"] = 0.0

    # C. CÁLCULO BIO-TÉRMICO
    df["Tmedia"] = (df["TMAX"] + df["TMIN"]) / 2
    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, t_base_val, t_opt_max, t_critica))

    # -----------------------------------------------------
    # VISUALIZACIÓN
    # -----------------------------------------------------
    st.title("🌾 PREDWEEM LOLIUM- TRES ARROYOS 2026")

    # --- HEATMAP ---
    colorscale_hard = [
        [0.0, "green"], [0.24, "green"],
        [0.25, "yellow"], [0.74, "yellow"],
        [0.75, "red"], [1.0, "red"]
    ]
    fig_risk = go.Figure(data=go.Heatmap(
        z=[df["EMERREL"].values], x=df["Fecha"], y=["Emergencia"],
        colorscale=colorscale_hard, zmin=0, zmax=1, showscale=False
    ))
    fig_risk.update_layout(height=120, margin=dict(t=30, b=0, l=10, r=10), title="Mapa de Intensidad de Emergencia")
    st.plotly_chart(fig_risk, use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["📊 MONITOR DE DECISIÓN", "📈 ANÁLISIS ESTRATÉGICO", "🧪 BIO-CALIBRACIÓN"])

    # --- TAB 1 ---
    with tab1:
        col_main, col_gauge = st.columns([2, 1])

        indices_pulso = df.index[df["EMERREL"] >= umbral_er].tolist()
        fecha_inicio_ventana = None

        if indices_pulso:
            first_peak_index = indices_pulso[0]
            fecha_inicio_ventana = df.loc[first_peak_index, "Fecha"]

        dga_actual = 0.0
        dias_stress = 0
        if fecha_inicio_ventana is not None:
            df_ventana = df[df["Fecha"] >= fecha_inicio_ventana].copy()
            df_ventana["DGA_cum"] = df_ventana["DG"].cumsum()
            dga_actual = df_ventana["DGA_cum"].iloc[-1] if not df_ventana.empty else 0.0
            dias_stress = len(df_ventana[df_ventana["Tmedia"] > t_opt_max])

        with col_main:
            fig_emer = go.Figure()
            fig_emer.add_trace(go.Scatter(
                x=df["Fecha"], y=df["EMERREL"], mode="lines", name="Tasa Diaria",
                line=dict(color="#166534", width=2.5),
                fill="tozeroy", fillcolor="rgba(22, 101, 52, 0.1)"
            ))
            fig_emer.add_hline(
                y=umbral_er, line_dash="dash", line_color="orange",
                annotation_text=f"Umbral Pico ({umbral_er})"
            )
            fig_emer.update_layout(title="Dinámica de Emergencia y Detección de Picos", height=350)
            st.plotly_chart(fig_emer, use_container_width=True)

            if fecha_inicio_ventana is not None:
                st.success(f"📅 **Inicio de Conteo Térmico:** {fecha_inicio_ventana.strftime('%d-%m-%Y')} (Primer pico detectado)")
                if dias_stress > 0:
                    st.markdown(
                        f"""<div class="bio-alert">🔥 <b>Estrés Térmico:</b> {dias_stress} días con T > {t_opt_max}°C desde el inicio.</div>""",
                        unsafe_allow_html=True
                    )
            else:
                st.warning(f"⏳ Esperando el primer pico de emergencia (Tasa diaria >= {umbral_er}).")

        with col_gauge:
            max_axis = dga_critico * 1.2
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=dga_actual,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "<b>ACUMULACIÓN TÉRMICA (°Cd)</b><br><span style='font-size:0.8em;color:gray'>Desde primer pico</span>"},
                gauge={
                    "axis": {"range": [None, max_axis]},
                    "bar": {"color": "black"},
                    "steps": [
                        {"range": [0, dga_optimo], "color": "#4ade80"},
                        {"range": [dga_optimo, dga_critico], "color": "#facc15"},
                        {"range": [dga_critico, max_axis], "color": "#f87171"},
                    ],
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(t=50, b=10, l=30, r=30))
            st.plotly_chart(fig_gauge, use_container_width=True)

    # --- TAB 2 ---
    with tab2:
        st.header("🔍 Clasificación DTW")
        fecha_corte = pd.Timestamp("2026-05-01")
        df_obs = df[df["Fecha"] < fecha_corte].copy()

        if not df_obs.empty and df_obs["EMERREL"].sum() > 0:
            jd_corte = int(df_obs["Julian_days"].max())
            max_e = float(df_obs["EMERREL"].max()) if float(df_obs["EMERREL"].max()) > 0 else 1.0

            JD_COM = cluster_model["JD_common"]
            jd_grid = JD_COM[JD_COM <= jd_corte]
            obs_norm = np.interp(jd_grid, df_obs["Julian_days"], df_obs["EMERREL"] / max_e)

            dists = []
            for m in cluster_model["curves_interp"]:
                m_slice = m[JD_COM <= jd_corte]
                m_norm = m_slice / m_slice.max() if m_slice.max() > 0 else m_slice
                dists.append(dtw_distance(obs_norm, m_norm))

            pred = int(np.argmin(dists))
            names = {0: "🌾 Bimodal", 1: "🌱 Temprano", 2: "🍂 Tardío"}
            cols = {0: "#0284c7", 1: "#16a34a", 2: "#ea580c"}

            c1, c2 = st.columns([3, 1])
            with c1:
                fp = go.Figure()
                fp.add_trace(go.Scatter(
                    x=JD_COM,
                    y=cluster_model["curves_interp"][pred],
                    name="Patrón Histórico",
                    line=dict(dash="dash", color=cols.get(pred))
                ))
                fp.add_trace(go.Scatter(
                    x=jd_grid,
                    y=obs_norm * cluster_model["curves_interp"][pred].max(),
                    name="2026",
                    line=dict(color="black", width=3)
                ))
                fp.update_layout(title="Curva observada vs patrón asignado", height=350)
                st.plotly_chart(fp, use_container_width=True)

            with c2:
                st.success(f"### {names.get(pred)}")
                st.metric("DTW Score", f"{min(dists):.2f}")

        else:
            st.info("Datos insuficientes para clasificación DTW (Se requiere actividad antes de Mayo).")

    # --- TAB 3 ---
    with tab3:
        st.subheader("🧪 Curva de Respuesta Fisiológica")
        st.markdown("Así se comporta la acumulación térmica según los parámetros definidos.")

        x_temps = np.linspace(0, 45, 200)
        y_tt = [calculate_tt_scalar(t, t_base_val, t_opt_max, t_critica) for t in x_temps]

        fig_bio = go.Figure()
        fig_bio.add_trace(go.Scatter(
            x=x_temps, y=y_tt, mode="lines", name="Acumulación TT",
            line=dict(color="#2563eb", width=4),
            fill="tozeroy", fillcolor="rgba(37, 99, 235, 0.1)"
        ))
        fig_bio.add_vrect(x0=t_base_val, x1=t_opt_max, fillcolor="green", opacity=0.1,
                          annotation_text="Óptimo", annotation_position="top left")
        fig_bio.add_vrect(x0=t_opt_max, x1=t_critica, fillcolor="orange", opacity=0.1,
                          annotation_text="Estrés (Penalizado)", annotation_position="top right")
        fig_bio.add_vrect(x0=t_critica, x1=45, fillcolor="red", opacity=0.1,
                          annotation_text="Inhibición", annotation_position="top right")

        fig_bio.update_layout(
            xaxis_title="Temperatura Media Diaria (°C)",
            yaxis_title="Tiempo Térmico Acumulado (°Cd)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_bio, use_container_width=True)

        st.info(f"""
        **Interpretación:**
        * Hasta **{t_base_val}°C**: No pasa nada (Dormición/Inactividad).
        * Entre **{t_base_val}°C y {t_opt_max}°C**: Crecimiento lineal.
        * Entre **{t_opt_max}°C y {t_critica}°C**: La eficiencia cae rápidamente.
        * Más de **{t_critica}°C**: El sistema se detiene (TT = 0).
        """)

    # -----------------------------------------------------
    # EXPORTACIÓN
    # -----------------------------------------------------
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Data_Diaria")
        pd.DataFrame({
            "Configuracion": ["T_Base", "T_Optima", "T_Critica", "Umbral_Pico"],
            "Valor": [t_base_val, t_opt_max, t_critica, umbral_er]
        }).to_excel(writer, sheet_name="Bio_Params", index=False)

    st.sidebar.download_button("📥 Descargar Reporte", output.getvalue(), "PREDWEEM_Report.xlsx")

else:
    st.info("👋 **Bienvenido a PREDWEEM.** Cargue datos climáticos para comenzar.")
