# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4 — LOLIUM TRES ARROYOS 2026
# UI PRO: header + cards + sidebar form (Aplicar) + modo experto
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import io
import hashlib
from pathlib import Path

# ---------------------------------------------------------
# 1) CONFIG + ESTILO (UI PRO)
# ---------------------------------------------------------
st.set_page_config(
    page_title="PREDWEEM INTEGRAL vK4",
    layout="wide",
    page_icon="🌾"
)

st.markdown("""
<style>
    .main { background-color: #f8fafc; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #dcfce7;
        border-right: 1px solid #bbf7d0;
    }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
        color: #166534 !important;
    }

    /* Métricas (selector correcto) */
    [data-testid="stMetric"]{
        background:#ffffff;
        padding:14px;
        border-radius:14px;
        border:1px solid #e2e8f0;
        box-shadow:0 2px 10px rgba(2,6,23,.06);
    }

    /* Card genérica */
    .pw-card{
        background:#fff;
        border:1px solid #e2e8f0;
        border-radius:16px;
        padding:16px;
        box-shadow:0 2px 10px rgba(2,6,23,.06);
        margin-bottom:12px;
    }

    /* Alert */
    .bio-alert {
        padding: 10px;
        border-radius: 10px;
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
        margin-top: 10px;
        font-size: 0.95em;
    }

    /* Compact tabs spacing */
    [data-baseweb="tab-list"] { gap: 8px; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ---------------------------------------------------------
# 2) ROBUSTEZ: GENERADOR DE ARCHIVOS MOCK
# ---------------------------------------------------------
def create_mock_files_if_missing():
    if not (BASE / "IW.npy").exists():
        np.save(BASE / "IW.npy", np.random.rand(4, 10))
        np.save(BASE / "bias_IW.npy", np.random.rand(10))
        np.save(BASE / "LW.npy", np.random.rand(1, 10))
        np.save(BASE / "bias_out.npy", np.random.rand(1))

    if not (BASE / "modelo_clusters_k3.pkl").exists():
        jd = np.arange(1, 366)
        p1 = np.exp(-((jd - 100)**2)/600)
        p2 = np.exp(-((jd - 160)**2)/900) + 0.3*np.exp(-((jd - 260)**2)/1200)
        p3 = np.exp(-((jd - 230)**2)/1500)
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
            "TMAX": np.random.uniform(25, 35, size=150) - (np.arange(150)*0.1),
            "TMIN": np.random.uniform(10, 18, size=150) - (np.arange(150)*0.06),
            "Prec": np.random.choice([0, 0, 5, 15, 45], size=150)
        }
        pd.DataFrame(data).to_csv(BASE / "meteo_daily.csv", index=False)

create_mock_files_if_missing()

# ---------------------------------------------------------
# 3) LÓGICA TÉCNICA (ANN + DTW + BIO)
# ---------------------------------------------------------
def dtw_distance(a, b):
    na, nb = len(a), len(b)
    dp = np.full((na+1, nb+1), np.inf)
    dp[0, 0] = 0
    for i in range(1, na+1):
        for j in range(1, nb+1):
            cost = abs(a[i-1] - b[j-1])
            dp[i, j] = cost + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
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

def polish(fig, title=None, height=None):
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(t=55, b=20, l=20, r=20),
        hovermode="x unified",
        font=dict(size=14),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,.25)")
    return fig

def data_signature_from_upload(file_input):
    # firma estable para cache/estado
    if not file_input:
        p = BASE / "meteo_daily.csv"
        if p.exists():
            b = p.read_bytes()
            return hashlib.md5(b).hexdigest()
        return "no-data"
    b = file_input.getvalue()
    return hashlib.md5(b).hexdigest()

def get_data(file_input):
    try:
        if file_input:
            if file_input.name.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(file_input.getvalue()), parse_dates=["Fecha"])
            else:
                df = pd.read_excel(io.BytesIO(file_input.getvalue()), parse_dates=["Fecha"])
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

@st.cache_data(show_spinner=False)
def run_pipeline(df_raw: pd.DataFrame, t_base_val, t_opt_max, t_critica):
    df = df_raw.copy()
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    df["Tmedia"] = (df["TMAX"] + df["TMIN"]) / 2
    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, t_base_val, t_opt_max, t_critica))
    return df

# ---------------------------------------------------------
# 4) SIDEBAR PRO (FORM + MODO EXPERTO)
# ---------------------------------------------------------
modelo_ann, cluster_model = load_models()

LOGO_URL = "https://raw.githubusercontent.com/PREDWEEM/loliumTA_2026/main/logo.png"

if "cfg" not in st.session_state:
    st.session_state.cfg = dict(
        umbral_er=0.50,
        t_base_val=2.0,
        t_opt_max=25.0,
        t_critica=30.0,
        dga_optimo=600,
        dga_critico=700,
        modo_experto=False
    )

with st.sidebar:
    st.image(LOGO_URL, use_container_width=True)
    st.markdown("## ⚙️ Configuración")

    with st.form("cfg_form", border=False):
        archivo_usuario = st.file_uploader("Subir Clima Manual", type=["xlsx", "csv"])

        st.session_state.cfg["modo_experto"] = st.toggle("Modo experto", value=st.session_state.cfg["modo_experto"])
        st.divider()

        st.markdown("**Parámetros de Emergencia**")
        umbral_er = st.slider("Umbral Tasa Diaria", 0.05, 0.80, st.session_state.cfg["umbral_er"])

        st.divider()
        st.markdown("🌡️ **Fisiología Térmica (Bio-Limit)**")
        c1, c2 = st.columns(2)
        with c1:
            t_base_val = st.number_input("T Base", value=float(st.session_state.cfg["t_base_val"]), step=0.5)
        with c2:
            t_opt_max = st.number_input("T Óptima Max", value=float(st.session_state.cfg["t_opt_max"]), step=1.0)

        t_critica = st.slider("T Crítica (Stop)", 26.0, 42.0, float(st.session_state.cfg["t_critica"]))

        if st.session_state.cfg["modo_experto"]:
            st.markdown("**Objetivos (°Cd)**")
            dga_optimo = st.number_input("Objetivo Control", value=int(st.session_state.cfg["dga_optimo"]), step=50)
            dga_critico = st.number_input("Límite Ventana", value=int(st.session_state.cfg["dga_critico"]), step=50)
        else:
            dga_optimo = st.session_state.cfg["dga_optimo"]
            dga_critico = st.session_state.cfg["dga_critico"]
            st.caption(f"Objetivo: {dga_optimo} °Cd · Límite: {dga_critico} °Cd")

        aplicar = st.form_submit_button("✅ Aplicar cambios")

    if aplicar:
        st.session_state.cfg.update(
            umbral_er=float(umbral_er),
            t_base_val=float(t_base_val),
            t_opt_max=float(t_opt_max),
            t_critica=float(t_critica),
            dga_optimo=int(dga_optimo),
            dga_critico=int(dga_critico),
        )
        st.toast("Configuración aplicada")

# Leer datos + firma
df_raw = get_data(archivo_usuario)
sig = data_signature_from_upload(archivo_usuario)

# ---------------------------------------------------------
# 5) MOTOR (con UI PRO arriba)
# ---------------------------------------------------------
if df_raw is None:
    st.info("👋 **Bienvenido a PREDWEEM.** Cargue datos climáticos para comenzar.")
    st.stop()

# Header pro
st.markdown("""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
  <div style="font-size:34px;">🌾</div>
  <div>
    <div style="font-size:28px;font-weight:850;line-height:1.1;">PREDWEEM LOLIUM – Tres Arroyos 2026</div>
    <div style="color:#64748b;font-size:14px;">INTEGRAL vK4 · Monitor + Patrones + Bio-calibración</div>
  </div>
</div>
""", unsafe_allow_html=True)

cfg = st.session_state.cfg
df = run_pipeline(df_raw, cfg["t_base_val"], cfg["t_opt_max"], cfg["t_critica"])

# Predicción ANN (no la cacheo adentro de cache_data para evitar objetos)
X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
emerrel, _ = modelo_ann.predict(X)
df["EMERREL"] = np.maximum(emerrel, 0.0)
df.loc[df["Julian_days"] <= 30, "EMERREL"] = 0.0  # regla

# KPIs arriba
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Última fecha", df["Fecha"].max().strftime("%d-%m-%Y"))
with k2:
    st.metric("EMERREL máx.", f"{df['EMERREL'].max():.2f}")
with k3:
    st.metric("DG acumulado", f"{df['DG'].sum():.0f} °Cd")
with k4:
    st.metric("Datos", f"{len(df)} días")

# Heatmap superior
colorscale_hard = [
    [0.0, "green"], [0.49, "green"],
    [0.49, "yellow"], [0.90, "yellow"],
    [0.90, "red"], [1.0, "red"]
]
fig_risk = go.Figure(data=go.Heatmap(
    z=[df["EMERREL"].values],
    x=df["Fecha"],
    y=["Emergencia"],
    colorscale=colorscale_hard,
    zmin=0, zmax=1,
    showscale=False
))
fig_risk = polish(fig_risk, "Mapa de Intensidad de Emergencia", 130)
st.plotly_chart(fig_risk, use_container_width=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 MONITOR DE DECISIÓN", "📈 ANÁLISIS ESTRATÉGICO", "🧪 BIO-CALIBRACIÓN"])

# ---------------------------------------------------------
# TAB 1: Monitor
# ---------------------------------------------------------
with tab1:
    col_main, col_gauge = st.columns([2, 1])

    indices_pulso = df.index[df["EMERREL"] >= cfg["umbral_er"]].tolist()
    fecha_inicio_ventana = None
    for i in range(len(indices_pulso) - 1):
        if (df.loc[indices_pulso[i+1], "Fecha"] - df.loc[indices_pulso[i], "Fecha"]).days <= 5:
            fecha_inicio_ventana = df.loc[indices_pulso[i], "Fecha"]
            break

    dga_actual = 0.0
    dias_stress = 0
    if fecha_inicio_ventana is not None:
        df_ventana = df[df["Fecha"] >= fecha_inicio_ventana].copy()
        df_ventana["DGA_cum"] = df_ventana["DG"].cumsum()
        dga_actual = float(df_ventana["DGA_cum"].iloc[-1])
        dias_stress = int((df_ventana["Tmedia"] > cfg["t_opt_max"]).sum())

    # Card recomendación
    if fecha_inicio_ventana is None:
        estado_txt = "⏳ Esperando pulsos consistentes"
    else:
        if dga_actual < cfg["dga_optimo"]:
            estado_txt = "🟢 En ventana (óptimo)"
        elif dga_actual < cfg["dga_critico"]:
            estado_txt = "🟡 Ventana avanzada (precaución)"
        else:
            estado_txt = "🔴 Fuera de ventana (límite excedido)"

    st.markdown(f"""
    <div class="pw-card">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div style="font-weight:850;font-size:16px;">📌 Recomendación operativa</div>
        <div style="font-weight:850;">{estado_txt}</div>
      </div>
      <div style="color:#475569;margin-top:6px;">
        Cohorte: <b>{fecha_inicio_ventana.strftime('%d-%m-%Y') if fecha_inicio_ventana else '—'}</b> ·
        DGA actual: <b>{dga_actual:.0f}</b> ·
        Estrés térmico: <b>{dias_stress}</b> días (T &gt; {cfg["t_opt_max"]}°C)
      </div>
    </div>
    """, unsafe_allow_html=True)

    with col_main:
        fig_emer = go.Figure()
        fig_emer.add_trace(go.Scatter(
            x=df["Fecha"], y=df["EMERREL"],
            mode="lines",
            name="Tasa diaria",
            line=dict(color="#166534", width=2.8),
            fill="tozeroy",
            fillcolor="rgba(22, 101, 52, 0.12)"
        ))
        fig_emer.add_hline(y=cfg["umbral_er"], line_dash="dash", line_color="orange")
        fig_emer = polish(fig_emer, "Dinámica de Emergencia", 360)
        st.plotly_chart(fig_emer, use_container_width=True)

        if fecha_inicio_ventana:
            st.info(f"📅 **Inicio cohorte:** {fecha_inicio_ventana.strftime('%d-%m-%Y')}")
            if dias_stress > 0:
                st.markdown(
                    f"""<div class="bio-alert">🔥 <b>Estrés térmico:</b> {dias_stress} días con T &gt; {cfg["t_opt_max"]}°C.</div>""",
                    unsafe_allow_html=True
                )
        else:
            st.warning("⏳ Aún no se detectó una cohorte consistente según el umbral seleccionado.")

    with col_gauge:
        max_axis = cfg["dga_critico"] * 1.2
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=dga_actual,
            title={"text": "<b>ACUMULACIÓN TÉRMICA</b>"},
            gauge={
                "axis": {"range": [None, max_axis]},
                "bar": {"color": "black"},
                "steps": [
                    {"range": [0, cfg["dga_optimo"]], "color": "#4ade80"},
                    {"range": [cfg["dga_optimo"], cfg["dga_critico"]], "color": "#facc15"},
                    {"range": [cfg["dga_critico"], max_axis], "color": "#f87171"},
                ],
            }
        ))
        fig_gauge = polish(fig_gauge, None, 320)
        st.plotly_chart(fig_gauge, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: Análisis (DTW)
# ---------------------------------------------------------
with tab2:
    st.markdown('<div class="pw-card">', unsafe_allow_html=True)
    st.subheader("🔍 Clasificación DTW (histórica)")
    st.caption("Comparación parcial hasta fecha de corte con los patrones (medoides) históricos.")
    st.markdown("</div>", unsafe_allow_html=True)

    fecha_corte = pd.Timestamp("2026-05-01")
    df_obs = df[df["Fecha"] < fecha_corte].copy()

    if df_obs.empty:
        st.warning("No hay datos suficientes antes de la fecha de corte para clasificar.")
    else:
        jd_corte = int(df_obs["Julian_days"].max())
        max_e = float(df_obs["EMERREL"].max()) if df_obs["EMERREL"].max() > 0 else 1.0

        JD_COM = np.array(cluster_model["JD_common"])
        jd_grid = JD_COM[JD_COM <= jd_corte]

        obs_norm = np.interp(jd_grid, df_obs["Julian_days"], (df_obs["EMERREL"] / max_e))

        dists = []
        for m in cluster_model["curves_interp"]:
            m = np.array(m)
            m_slice = m[JD_COM <= jd_corte]
            m_norm = (m_slice / m_slice.max()) if m_slice.max() > 0 else m_slice
            dists.append(float(dtw_distance(obs_norm, m_norm)))

        pred = int(np.argmin(dists))
        names = {0: "🌾 Bimodal", 1: "🌱 Temprano", 2: "🍂 Tardío"}
        cols = {0: "#0284c7", 1: "#16a34a", 2: "#ea580c"}

        c1, c2 = st.columns([3, 1])
        with c1:
            fp = go.Figure()
            fp.add_trace(go.Scatter(
                x=JD_COM, y=np.array(cluster_model["curves_interp"][pred]),
                name="Patrón histórico",
                line=dict(dash="dash", color=cols.get(pred))
            ))
            fp.add_trace(go.Scatter(
                x=jd_grid,
                y=obs_norm * float(np.array(cluster_model["curves_interp"][pred]).max()),
                name="2026 (observado hasta corte)",
                line=dict(color="black", width=3)
            ))
            fp = polish(fp, "Curva 2026 vs. patrón asignado", 420)
            st.plotly_chart(fp, use_container_width=True)

        with c2:
            st.success(f"### {names.get(pred)}")
            st.metric("DTW score", f"{min(dists):.2f}")

# ---------------------------------------------------------
# TAB 3: Bio-calibración (curva respuesta)
# ---------------------------------------------------------
with tab3:
    st.markdown('<div class="pw-card">', unsafe_allow_html=True)
    st.subheader("🧪 Curva de respuesta fisiológica (Bio-Limit)")
    st.caption("Visualiza cómo cambia el TT diario según T Base, T Óptima y T Crítica.")
    st.markdown("</div>", unsafe_allow_html=True)

    x_temps = np.linspace(0, 45, 200)
    y_tt = [calculate_tt_scalar(t, cfg["t_base_val"], cfg["t_opt_max"], cfg["t_critica"]) for t in x_temps]

    fig_bio = go.Figure()
    fig_bio.add_trace(go.Scatter(
        x=x_temps, y=y_tt,
        mode="lines",
        name="TT diario",
        line=dict(color="#2563eb", width=4),
        fill="tozeroy",
        fillcolor="rgba(37, 99, 235, 0.12)"
    ))

    fig_bio.add_vrect(x0=cfg["t_base_val"], x1=cfg["t_opt_max"],
                      fillcolor="green", opacity=0.10,
                      annotation_text="Óptimo", annotation_position="top left")
    fig_bio.add_vrect(x0=cfg["t_opt_max"], x1=cfg["t_critica"],
                      fillcolor="orange", opacity=0.10,
                      annotation_text="Estrés (penalizado)", annotation_position="top right")
    fig_bio.add_vrect(x0=cfg["t_critica"], x1=45,
                      fillcolor="red", opacity=0.10,
                      annotation_text="Inhibición", annotation_position="top right")

    fig_bio.update_layout(xaxis_title="Temperatura media diaria (°C)",
                          yaxis_title="TT diario (°Cd)")
    fig_bio = polish(fig_bio, "Curva TT (Bio-Limit)", 420)
    st.plotly_chart(fig_bio, use_container_width=True)

    st.info(f"""
    **Interpretación:**
    - Hasta **{cfg["t_base_val"]}°C**: TT = 0 (inactividad).
    - **{cfg["t_base_val"]}–{cfg["t_opt_max"]}°C**: crecimiento lineal.
    - **{cfg["t_opt_max"]}–{cfg["t_critica"]}°C**: eficiencia penalizada.
    - **>{cfg["t_critica"]}°C**: TT = 0 (stop térmico).
    """)

# ---------------------------------------------------------
# EXPORTACIÓN (manteniendo tu salida)
# ---------------------------------------------------------
output = io.BytesIO()
with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
    df.to_excel(writer, index=False, sheet_name="Data_Diaria")
    pd.DataFrame({
        "Configuracion": ["T_Base", "T_Optima", "T_Critica", "Umbral_ER", "DGA_Optimo", "DGA_Critico", "Data_Signature"],
        "Valor": [cfg["t_base_val"], cfg["t_opt_max"], cfg["t_critica"], cfg["umbral_er"], cfg["dga_optimo"], cfg["dga_critico"], sig]
    }).to_excel(writer, sheet_name="Config", index=False)

st.sidebar.download_button("📥 Descargar Reporte", output.getvalue(), "PREDWEEM_Report.xlsx")

