
# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM v7.3 PRO — LOLIUM TRES ARROYOS 2026
# - ANN → EMERREL diaria
# - Post-proceso: recorte negativos, suavizado opcional, acumulado
# - Percentiles d25–d95 sobre curva truncada
# - Clasificación Temprano / Extendido + confianza (ALTA / MEDIA / BAJA)
# - Momento crítico en fecha calendario real
# - Fuente de datos FIJA: meteo_daily.csv
# - UI profesional con navegación por secciones
# ===============================================================

import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import requests
import xml.etree.ElementTree as ET
import streamlit as st

# ---------------------------------------------------------------
# 🖥️ CONFIG STREAMLIT + ESTILO GLOBAL
# ---------------------------------------------------------------
st.set_page_config(
    page_title="PREDWEEM v7.3 PRO – LOLIUM TA 2026",
    layout="wide",
)

# 🎨 Estilo visual más profesional
st.markdown("""
<style>
/* Ocultar menús de Streamlit para modo "app" */
#MainMenu, footer, header [data-testid="stToolbar"], .stAppDeployButton {
    visibility: hidden !important;
}

/* Contenedor central más angosto y centrado */
.block-container {
    padding-top: 0.5rem;
    padding-bottom: 1.0rem;
    max-width: 1350px;
}

/* Títulos principales */
h1, h2, h3 {
    color: #004D40;
}

/* Tarjetas suaves para separar secciones */
.st-emotion-cache-1r6slb0, .st-emotion-cache-1wmy9hl {
    border-radius: 0.6rem;
}

/* Pequeño ajuste de tablas */
[data-testid="stDataFrame"] {
    border-radius: 0.5rem;
}

/* Colores auxiliares */
.badge {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    color: #fff;
}
.badge-green { background-color: #2E7D32; }
.badge-orange { background-color: #EF6C00; }
.badge-red { background-color: #C62828; }
</style>
""", unsafe_allow_html=True)

# Estilo matplotlib
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["legend.frameon"] = False

# Colores "institucionales"
COL_TEMP      = "#2E7D32"   # Temprano
COL_EXT       = "#EF6C00"   # Extendido
COL_EVAL      = "#0277BD"   # Año Evaluado
COL_CRITICO   = "#C62828"   # Momento crítico
COL_MAX       = "#303F9F"   # Máxima certeza

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ---------------------------------------------------------------
# 🔧 FUNCIONES SEGURAS
# ---------------------------------------------------------------
def safe(fn, msg):
    try:
        return fn()
    except Exception as e:
        st.error(f"{msg}: {e}")
        return None

# ---------------------------------------------------------------
# 🔧 API METEOBAHIA (no utilizada, pero lista si se quiere extender)
# ---------------------------------------------------------------
API_URL = "https://meteobahia.com.ar/scripts/forecast/for-ta.xml"

def _to_float(x):
    try:
        return float(str(x).replace(",", "."))
    except:
        return None

@st.cache_data(ttl=900)
def fetch_forecast():
    r = requests.get(API_URL, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.content)

    rows = []
    for d in root.findall(".//forecast/tabular/day"):
        fecha  = d.find("fecha").get("value")
        tmax   = d.find("tmax").get("value")
        tmin   = d.find("tmin").get("value")
        prec   = d.find("precip").get("value")
        rows.append({
            "Fecha": pd.to_datetime(fecha),
            "TMAX": _to_float(tmax),
            "TMIN": _to_float(tmin),
            "Prec": _to_float(prec),
        })

    df = pd.DataFrame(rows).sort_values("Fecha").head(7)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    return df

# ---------------------------------------------------------------
# 🔧 ANN — Modelo de predicción de emergencia
# ---------------------------------------------------------------
class PracticalANNModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW = IW
        self.bIW = bIW
        self.LW = LW
        self.bLW = bLW
        # rango de entrenamiento original
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])

    def normalize(self, X):
        return 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1

    def predict(self, Xreal):
        """
        Devuelve EMERREL cruda de la ANN y EMERAC cruda (cumsum).
        El post-procesamiento se hace por fuera.
        """
        Xn = self.normalize(Xreal)
        emer = []
        for x in Xn:
            z1 = self.IW.T @ x + self.bIW
            a1 = np.tanh(z1)
            z2 = self.LW @ a1 + self.bLW
            emer.append(np.tanh(z2))
        emer = (np.array(emer) + 1) / 2    # 0–1 (diario, crudo)
        emer_ac = np.cumsum(emer)          # acumulada cruda
        emerrel = np.diff(emer_ac, prepend=0)
        return emerrel, emer_ac

@st.cache_resource
def load_ann():
    IW  = np.load(BASE/"IW.npy")
    bIW = np.load(BASE/"bias_IW.npy")
    LW  = np.load(BASE/"LW.npy")
    bLW = np.load(BASE/"bias_out.npy")
    return PracticalANNModel(IW, bIW, LW, bLW)

modelo_ann = safe(lambda: load_ann(), "Error cargando pesos ANN")
if modelo_ann is None:
    st.stop()

# ---------------------------------------------------------------
# 🔧 POST-PROCESO EMERGENCIA
# ---------------------------------------------------------------
def postprocess_emergence(emerrel_raw,
                          smooth=True,
                          window=3,
                          clip_zero=True):
    """
    Toma EMERREL cruda de la ANN y devuelve:
    - emerrel_proc: EMERREL suavizada / recortada
    - emerac_proc : EMERAC acumulada (no forzada a terminar en 1)
    """
    emer = np.array(emerrel_raw, dtype=float)

    # 1) Recortar posibles negativos
    if clip_zero:
        emer = np.maximum(emer, 0.0)

    # 2) Suavizado por media móvil
    if smooth and len(emer) > 1 and window > 1:
        window = int(window)
        window = max(1, min(window, len(emer)))
        if window > 1:
            kernel = np.ones(window, dtype=float) / window
            emer = np.convolve(emer, kernel, mode="same")

    # 3) EMERAC acumulada
    emerac = np.cumsum(emer)

    return emer, emerac

# ---------------------------------------------------------------
# 🔧 CARGAR MODELO DE CLUSTERS
# ---------------------------------------------------------------
def load_cluster_model():
    local_path = BASE/"modelo_cluster_d25_d50_d75_d95.pkl"
    alt_path   = Path("/mnt/data/modelo_cluster_d25_d50_d75_d95.pkl")

    if local_path.exists():
        path = local_path
    elif alt_path.exists():
        path = alt_path
    else:
        raise FileNotFoundError("modelo_cluster_d25_d50_d75_d95.pkl no encontrado")

    with open(path, "rb") as f:
        data = pickle.load(f)

    scaler        = data["scaler"]
    model         = data["model"]
    centroides    = data["centroides"]       # numpy (2,4)
    metricas_hist = data.get("metricas_hist", data.get("metricas", {}))
    labels_hist   = data.get("labels_hist",  data.get("labels", {}))

    return scaler, model, metricas_hist, labels_hist, centroides

cluster_pack = safe(lambda: load_cluster_model(),
    "Error cargando modelo_cluster_d25_d50_d75_d95.pkl")

if cluster_pack is None:
    st.stop()
else:
    scaler_cl, model_cl, metricas_hist, labels_hist, centroides = cluster_pack

# ---------------------------------------------------------------
# 🔧 FUNCIONES D25–D95 (sobre curva truncada)
# ---------------------------------------------------------------
def calc_percentiles_trunc(dias, emerac):
    """
    Calcula d25–d95 tomando como referencia el máximo disponible
    (curva potencialmente truncada).
    """
    emerac = np.asarray(emerac)
    if emerac.size == 0 or emerac.max() == 0:
        return None
    y = emerac / emerac.max()   # normaliza respecto a lo emergido hasta la fecha
    d25 = np.interp(0.25, y, dias)
    d50 = np.interp(0.50, y, dias)
    d75 = np.interp(0.75, y, dias)
    d95 = np.interp(0.95, y, dias)
    return d25, d50, d75, d95

def curva(vals):
    d25, d50, d75, d95 = vals
    x = np.array([d25, d50, d75, d95])
    y = np.array([0.25, 0.50, 0.75, 0.95])
    dias = np.arange(20, 200)
    curva = np.interp(dias, x, y)
    return dias, curva

# ---------------------------------------------------------------
# 🔧 RADAR MULTISERIES
# ---------------------------------------------------------------
def radar_multiseries(values_dict, labels, title):

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)

    colors = {
        "Año evaluado": COL_EVAL,
        "Temprano": COL_TEMP,
        "Extendido": COL_EXT
    }

    for name, vals in values_dict.items():
        vals2 = list(vals) + [vals[0]]
        c = colors.get(name, None)
        ax.plot(angles, vals2, lw=2.5, label=name, color=c)
        ax.fill(angles, vals2, alpha=0.15, color=c)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", bbox_to_anchor=(1.3, 0.1))

    return fig

# ---------------------------------------------------------------
# 📂 CARGA DE DATOS METEOROLÓGICOS (FIJO: meteo_daily.csv)
# ---------------------------------------------------------------
@st.cache_data
def load_meteo_daily():
    path_daily = BASE / "meteo_daily.csv"
    if not path_daily.exists():
        raise FileNotFoundError("No se encontró meteo_daily.csv en el directorio de la app.")
    df = pd.read_csv(path_daily, parse_dates=["Fecha"])
    df = df.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    return df

df = safe(load_meteo_daily, "Error cargando meteo_daily.csv")
if df is None or df.empty:
    st.stop()

# ---------------------------------------------------------------
# 🔧 CÁLCULOS PRINCIPALES (ANN + POSTPROCESO + CLASIFICACIÓN)
# ---------------------------------------------------------------
# Controles principales en sidebar
st.sidebar.title("PREDWEEM v7.3 PRO")
modo = st.sidebar.radio("Modo de visualización", ["Usuario", "Desarrollador"], index=0)

st.sidebar.markdown("---")
st.sidebar.header("Ajustes de emergencia")
use_smoothing = st.sidebar.checkbox("Suavizar EMERREL", value=True)
window_size   = st.sidebar.slider("Ventana de suavizado (días)", min_value=1, max_value=9, value=3, step=1)
clip_zero     = st.sidebar.checkbox("Recortar negativos a 0", value=True)

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navegación",
    [
        "Dashboard",
        "Curvas de emergencia",
        "Clasificación y momento crítico",
        "Comparación de patrones"
    ],
)

# ANN
X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
emerrel_raw, emerac_raw = modelo_ann.predict(X)
emerrel, emerac = postprocess_emergence(
    emerrel_raw,
    smooth=use_smoothing,
    window=window_size,
    clip_zero=clip_zero,
)

df["EMERREL"] = emerrel
df["EMERAC"]  = emerac

dias   = df["Julian_days"].to_numpy()
fechas = df["Fecha"].to_numpy()

# Cobertura temporal
JD_START = int(dias.min())
JD_END   = int(dias.max())
TEMPORADA_MAX = 274  # 1-ene → 1-oct, aprox. temporada completa
cobertura = (JD_END - JD_START + 1) / TEMPORADA_MAX

# Percentiles sobre curva truncada
res = calc_percentiles_trunc(dias, emerac)
if res is None:
    st.error("No se pudieron calcular percentiles sobre la curva emergida hasta la fecha.")
    st.stop()

d25, d50, d75, d95 = res

# Clasificación global
entrada_sc = scaler_cl.transform([[d25, d50, d75, d95]])
cl = int(cluster_pack[1].predict(entrada_sc)[0])  # model_cl

nombres = {1: "Temprano / Compacto", 0: "Extendido / Lento"}
colors_html  = {1: "#2E7D32", 0: "#EF6C00"}

nombre_clase = nombres[cl]
color_clase_html = colors_html[cl]

# Curvas comparativas (d25–d95)
dias_x,   curva_x    = curva([d25, d50, d75, d95])
dias_ext, curva_ext  = curva(centroides[0])
dias_temp, curva_temp = curva(centroides[1])

# ---------------------------------------------------------------
# 🔧 CERTEZA TEMPORAL DEL PATRÓN + MOMENTO CRÍTICO
# ---------------------------------------------------------------
probs_temp = []
probs_ext  = []
dias_eval  = []
fechas_eval = []

for i in range(5, len(df)):

    dias_parc   = dias[:i]
    emerac_parc = emerac[:i]
    fechas_parc = df["Fecha"].iloc[:i]

    res_parc = calc_percentiles_trunc(dias_parc, emerac_parc)
    if res_parc is None:
        continue

    d25_p, d50_p, d75_p, d95_p = res_parc

    entrada_sc_parc = scaler_cl.transform([[d25_p, d50_p, d75_p, d95_p]])

    # Distancias a centroides en espacio escaleado
    d_ext = np.linalg.norm(
        entrada_sc_parc - model_cl.cluster_centers_[0].reshape(1, -1)
    )
    d_temp = np.linalg.norm(
        entrada_sc_parc - model_cl.cluster_centers_[1].reshape(1, -1)
    )

    # Probabilidades ~ inverso de la distancia
    if d_ext == 0 and d_temp == 0:
        prob_temp = 0.5
        prob_ext  = 0.5
    else:
        w_ext  = 1.0 / (d_ext + 1e-9)
        w_temp = 1.0 / (d_temp + 1e-9)
        s = w_ext + w_temp
        prob_temp = w_temp / s
        prob_ext  = w_ext / s

    dias_eval.append(dias_parc[-1])
    fechas_eval.append(fechas_parc.iloc[-1])
    probs_temp.append(prob_temp)
    probs_ext.append(prob_ext)

# Determinar serie según clase resultante
if cl == 1:
    probs_clase   = probs_temp
    nombre_clase  = "Temprano / Compacto"
    color_clase   = COL_TEMP
else:
    probs_clase   = probs_ext
    nombre_clase  = "Extendido / Lento"
    color_clase   = COL_EXT

UMBRAL = 0.8  # umbral de decisión

idx_crit = next((i for i, p in enumerate(probs_clase) if p >= UMBRAL), None)
idx_max  = int(np.argmax(probs_clase)) if len(probs_clase) > 0 else None

fecha_crit = None
prob_crit  = None
if idx_crit is not None:
    fecha_crit = fechas_eval[idx_crit]
    prob_crit  = probs_clase[idx_crit]

fecha_max = None
prob_max  = None
if idx_max is not None:
    fecha_max = fechas_eval[idx_max]
    prob_max  = probs_clase[idx_max]

# Nivel de confianza global
nivel_conf = None
color_conf = None
if prob_max is not None:
    if cobertura >= 0.7 and prob_max >= 0.8:
        nivel_conf = "ALTA"
        color_conf = "green"
    elif cobertura >= 0.4 and prob_max >= 0.65:
        nivel_conf = "MEDIA"
        color_conf = "orange"
    else:
        nivel_conf = "BAJA"
        color_conf = "red"

# ===============================================================
# 🌐 PÁGINA: DASHBOARD
# ===============================================================
if page == "Dashboard":

    st.title("🌾 PREDWEEM v7.3 PRO — LOLIUM TRES ARROYOS 2026")

    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.markdown(
            f"""
            ### Patrón estimado del año  
            <span style='color:{color_clase_html}; font-size:28px; font-weight:700;'>
            {nombre_clase}
            </span>
            """,
            unsafe_allow_html=True
        )

        if nivel_conf is not None:
            col_conf = {
                "ALTA": "badge-green",
                "MEDIA": "badge-orange",
                "BAJA": "badge-red"
            }.get(nivel_conf, "badge-red")

            st.markdown(
                f"""
                <span class="badge {col_conf}">
                    Confianza {nivel_conf}
                </span>
                """,
                unsafe_allow_html=True
            )

        st.markdown("#### Resumen rápido")

        items = []
        items.append(f"- **Cobertura temporal:** {cobertura*100:.1f} % (JD {JD_START}–{JD_END})")
        if prob_max is not None:
            items.append(f"- **Probabilidad máxima del patrón resultante:** {prob_max:.2f}")
        if fecha_crit is not None:
            items.append(
                f"- **Momento crítico (prob ≥ {UMBRAL:.0%}):** "
                f"{fecha_crit.strftime('%d-%b')}"
            )
        if fecha_max is not None:
            items.append(
                f"- **Fecha de máxima certeza del patrón:** "
                f"{fecha_max.strftime('%d-%b')} (prob = {prob_max:.2f})"
            )

        st.markdown("\n".join(items))

        st.markdown("#### Percentiles simulados (curva truncada)")
        st.write({
            "d25": round(d25, 1),
            "d50": round(d50, 1),
            "d75": round(d75, 1),
            "d95": round(d95, 1),
        })

    with col2:
        # Radar resumen
        vals_year = [d25, d50, d75, d95]
        vals_temp = list(centroides[1])
        vals_ext  = list(centroides[0])

        fig_rad = radar_multiseries(
            {
                "Año evaluado": vals_year,
                "Temprano": vals_temp,
                "Extendido": vals_ext
            },
            labels=["d25", "d50", "d75", "d95"],
            title="Radar — Año (parcial) vs Temprano vs Extendido"
        )
        st.pyplot(fig_rad)

        # Descarga PNG del radar
        buf_rad = io.BytesIO()
        fig_rad.savefig(buf_rad, format="png", dpi=150, bbox_inches="tight")
        st.download_button(
            "⬇️ Descargar radar (PNG)",
            data=buf_rad.getvalue(),
            file_name="predweem_radar_patrones.png",
            mime="image/png",
        )

    st.markdown("---")
    st.markdown("##### Nota metodológica")
    st.caption(
        "La clasificación se basa en los percentiles d25–d95 calculados sobre la "
        "curva emergida hasta la fecha, comparados contra centroides históricos "
        "de patrones Temprano y Extendido."
    )

    if modo == "Desarrollador":
        with st.expander("Ver datos internos (modo desarrollador)"):
            st.write(df.head())
            st.write("Centroides d25–d95:", centroides)

# ===============================================================
# 🌐 PÁGINA: CURVAS DE EMERGENCIA
# ===============================================================
elif page == "Curvas de emergencia":

    st.title("📈 Curvas de emergencia — LOLIUM TA 2026")

    st.subheader("🔍 EMERGENCIA diaria y acumulada — Cruda vs Procesada")

    col_er, col_ac = st.columns(2)

    # --------------------- EMERREL ---------------------
    with col_er:
        fig_er, ax_er = plt.subplots(figsize=(5,4))

        ax_er.plot(
            fechas, emerrel_raw,
            label="EMERREL cruda (ANN)",
            color="red",
            alpha=0.6
        )

        ax_er.plot(
            fechas, emerrel,
            label="EMERREL procesada",
            color=COL_EVAL,
            linewidth=2
        )

        ax_er.set_xlabel("Fecha calendario real")
        ax_er.set_ylabel("EMERREL (fracción diaria)")
        ax_er.set_title("EMERREL: ANN vs post-proceso")
        ax_er.legend()
        fig_er.autofmt_xdate()

        st.pyplot(fig_er)

        buf_er = io.BytesIO()
        fig_er.savefig(buf_er, format="png", dpi=150, bbox_inches="tight")
        st.download_button(
            "⬇️ Descargar gráfico EMERREL (PNG)",
            data=buf_er.getvalue(),
            file_name="predweem_emerrel.png",
            mime="image/png",
        )

        st.caption(
            "EMERREL procesada incluye recorte de negativos y suavizado por ventana "
            f"móvil de {window_size} días (configurable en el panel lateral)."
        )

    # --------------------- EMERAC ---------------------
    with col_ac:
        fig_ac, ax_ac = plt.subplots(figsize=(5,4))

        # Curva cruda normalizada cuando corresponde
        if emerac_raw[-1] > 0:
            ax_ac.plot(
                fechas, emerac_raw / emerac_raw[-1],
                label="EMERAC cruda (normalizada)",
                color=COL_EXT,
                alpha=0.6
            )
        else:
            ax_ac.plot(
                fechas, emerac_raw,
                label="EMERAC cruda",
                color=COL_EXT,
                alpha=0.6
            )

        # Curva procesada normalizada
        if emerac[-1] > 0:
            ax_ac.plot(
                fechas, emerac / emerac[-1],
                label="EMERAC procesada (normalizada)",
                color=COL_TEMP,
                linewidth=2
            )
        else:
            ax_ac.plot(
                fechas, emerac,
                label="EMERAC procesada",
                color=COL_TEMP,
                linewidth=2
            )

        ax_ac.set_xlabel("Fecha calendario real")
        ax_ac.set_ylabel("EMERAC (0–1 relativo al período observado)")
        ax_ac.set_title("EMERAC: ANN vs post-proceso")
        ax_ac.legend()
        fig_ac.autofmt_xdate()

        st.pyplot(fig_ac)

        buf_ac = io.BytesIO()
        fig_ac.savefig(buf_ac, format="png", dpi=150, bbox_inches="tight")
        st.download_button(
            "⬇️ Descargar gráfico EMERAC (PNG)",
            data=buf_ac.getvalue(),
            file_name="predweem_emerac.png",
            mime="image/png",
        )

    if modo == "Desarrollador":
        with st.expander("Datos numéricos de EMERREL / EMERAC"):
            st.write(df[["Fecha", "Julian_days", "EMERREL", "EMERAC"]].head(20))

# ===============================================================
# 🌐 PÁGINA: CLASIFICACIÓN Y MOMENTO CRÍTICO
# ===============================================================
elif page == "Clasificación y momento crítico":

    st.title("🎯 Clasificación del patrón y momento crítico")

    st.markdown(
        f"""
        ### Patrón resultante  
        <span style='color:{color_clase_html}; font-size:26px; font-weight:700;'>
        {nombre_clase}
        </span>
        """,
        unsafe_allow_html=True
    )

    if nivel_conf is not None:
        col_conf = {
            "ALTA": "badge-green",
            "MEDIA": "badge-orange",
            "BAJA": "badge-red"
        }.get(nivel_conf, "badge-red")

        st.markdown(
            f"""
            <span class="badge {col_conf}">
                Confianza {nivel_conf}
            </span>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    col1, col2 = st.columns([1.3, 1])

    # ---------- Curva d25-d95 vs centroides ----------
    with col1:
        st.subheader("Curva normalizada del año vs centroides históricos")

        fig_c, ax_c = plt.subplots(figsize=(7,4))
        ax_c.plot(dias_x, curva_x, lw=3, label="Año evaluado (parcial)", color=COL_EVAL)
        ax_c.plot(dias_temp, curva_temp, lw=2, label="Centroide Temprano", color=COL_TEMP)
        ax_c.plot(dias_ext,  curva_ext,  lw=2, label="Centroide Extendido", color=COL_EXT)
        ax_c.set_xlabel("Día juliano (escala normalizada d25–d95)")
        ax_c.set_ylabel("EMERAC relativa (0–1)")
        ax_c.legend()
        st.pyplot(fig_c)

        buf_c = io.BytesIO()
        fig_c.savefig(buf_c, format="png", dpi=150, bbox_inches="tight")
        st.download_button(
            "⬇️ Descargar curva año vs centroides (PNG)",
            data=buf_c.getvalue(),
            file_name="predweem_curva_vs_centroides.png",
            mime="image/png",
        )

    # ---------- Probabilidad temporal ----------
    with col2:
        st.subheader("Evolución temporal de la certeza del patrón")

        figp, axp = plt.subplots(figsize=(7,4))

        axp.plot(fechas_eval, probs_temp, label="Probabilidad Temprano",  color=COL_TEMP,  lw=2.0)
        axp.plot(fechas_eval, probs_ext,  label="Probabilidad Extendido", color=COL_EXT, lw=2.0)

        if fecha_crit is not None:
            axp.axvline(fecha_crit, color=COL_CRITICO, linestyle="--", linewidth=2,
                        label=f"Momento crítico ({nombre_clase})")

        if fecha_max is not None and (fecha_crit is None or fecha_max != fecha_crit):
            axp.axvline(fecha_max, color=COL_MAX, linestyle=":", linewidth=2,
                        label="Fecha máxima certeza")

        axp.set_ylim(0,1)
        axp.set_xlabel("Fecha calendario real")
        axp.set_ylabel("Probabilidad")
        axp.set_title("Evolución de la certeza del patrón")
        axp.legend()
        figp.autofmt_xdate()
        st.pyplot(figp)

        buf_p = io.BytesIO()
        figp.savefig(buf_p, format="png", dpi=150, bbox_inches="tight")
        st.download_button(
            "⬇️ Descargar gráfico de probabilidades (PNG)",
            data=buf_p.getvalue(),
            file_name="predweem_probabilidades_patron.png",
            mime="image/png",
        )

    st.markdown("---")
    st.markdown("### 🧠 Momento crítico de definición del patrón")

    if fecha_crit is not None:
        st.write(
            f"- **Patrón resultante:** {nombre_clase}  \n"
            f"- **Momento crítico (primer día con prob ≥ {UMBRAL:.0%}):** "
            f"**{fecha_crit.strftime('%d-%b')}**  \n"
            f"- **Probabilidad en ese día:** {prob_crit:.2f}  \n"
            f"- **Fecha de máxima certeza:** {fecha_max.strftime('%d-%b')} "
            f"(prob = {prob_max:.2f})"
        )
    elif fecha_max is not None:
        st.write(
            f"- **Patrón resultante:** {nombre_clase}  \n"
            f"- No se alcanza el umbral de {UMBRAL:.0%}, "
            f"pero la máxima certeza se logra el "
            f"**{fecha_max.strftime('%d-%b')}** con probabilidad **{prob_max:.2f}**."
        )
    else:
        st.info("No se pudo calcular la evolución de probabilidad del patrón.")

    st.markdown("#### Interpretación agronómica (guía rápida)")
    if cl == 1:
        st.markdown(
            "- Emergencia muy concentrada y temprana.  \n"
            "- Ventanas de control **tempranas** críticas (residuales y postemergentes "
            "precoces).  \n"
            "- Riesgo alto si el cultivo no está implantado y cubriendo rápido."
        )
    else:
        st.markdown(
            "- Emergencia más extendida hacia otoño.  \n"
            "- Necesidad de **monitoreo prolongado** y flexibilidad en postemergentes.  \n"
            "- Mayor superposición con siembras de cultivos de invierno."
        )

    if modo == "Desarrollador":
        with st.expander("Valores internos de probabilidad por fecha"):
            tmp = pd.DataFrame({
                "Fecha": fechas_eval,
                "Prob_Temprano": probs_temp,
                "Prob_Extendido": probs_ext
            })
            st.dataframe(tmp)

# ===============================================================
# 🌐 PÁGINA: COMPARACIÓN DE PATRONES (TABLA)
# ===============================================================
elif page == "Comparación de patrones":

    st.title("🌾 Comparación de Patrones de Emergencia (Temprano vs Extendido)")

    # TABLA FINAL (editable en código)
    data = {
        "Rasgo": [
            "Inicio",
            "Velocidad",
            "Concentración",
            "Fecha 80% EMERAC",
            "Fecha d95 (fin del proceso)",
            "Implicancias de manejo"
        ],
        "Patrón Temprano": [
            "Febrero – inicio de marzo",
            "Muy rápida",
            "1–2 pulsos concentrados",
            "Abril",
            "Fin de abril – inicios de mayo",
            "Residuales + control temprano (antes del 10 de marzo)"
        ],
        "Patrón Tardío / Extendido": [
            "Mitad de marzo – abril",
            "Lenta y escalonada",
            "2–4 pulsos, forma extendida",
            "Mayo – junio",
            "Junio – agosto",
            "Monitoreo prolongado + postemergente tardío"
        ]
    }

    df_comp = pd.DataFrame(data)

    st.subheader("📊 Tabla comparativa de patrones de emergencia")
    st.dataframe(df_comp, use_container_width=True)

    # DESCARGA DE LA TABLA
    csv = df_comp.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Descargar tabla comparativa en CSV",
        data=csv,
        file_name="comparacion_patrones_predweem_v7_3.csv",
        mime="text/csv"
    )

    st.markdown("### 🟦🟥 Codificación histórica utilizada")
    st.markdown("""
    - **🟦 Tempranos:** 2008, 2012, 2013, 2025  
    - **🟥 Tardíos / Extendidos:** 2009, 2010, 2011, 2014, 2015, 2023, 2024  

    Esta clasificación proviene del modelo `modelo_cluster_d25_d50_d75_d95.pkl`
    (Cluster 1 = Temprano, Cluster 0 = Tardío).
    """)

    st.subheader("📝 Descripción agronómica sintetizada")

    texto = """
    Los **patrones tempranos** muestran una emergencia concentrada entre febrero y abril,
    con más del 80% del total emergido antes del 20 de abril. Estos años suelen requerir
    **control temprano**, idealmente con residuales previos al 10 de marzo, y monitoreo intensivo
    en la primera quincena de marzo.

    Los **patrones tardíos y extendidos** desplazan la emergencia hacia abril–junio, con colas
    que pueden prolongarse hasta agosto. Esto obliga a **mantener estrategias de control
    postemergente tardías** y ampliar la ventana de monitoreo hasta finales de otoño.
    """

    st.markdown(texto)
