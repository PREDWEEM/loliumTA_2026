# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM vK3 UNIFICADO — LOLIUM TRES ARROYOS 2026
# - ANN → EMERREL diaria + acumulada
# - Mapa de riesgo (0–1) con umbrales (nulo/bajo/medio/alto)
# - Serie temporal + detección de pulsos
# - Monitor térmico (DG base 2°C) + semáforo (óptimo/crítico)
# - Clasificación funcional K=3 (DTW vs medoides) con confianza
# - Exportación a Excel
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import io
from pathlib import Path

# ---------------------------------------------------------
# CONFIGURACIÓN STREAMLIT
# ---------------------------------------------------------
st.set_page_config(
    page_title="PREDWEEM vK3 — TRES ARROYOS 2026",
    layout="wide",
    page_icon="🌾",
)

st.markdown(
    """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

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
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
</style>
""",
    unsafe_allow_html=True,
)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ---------------------------------------------------------
# ROBUSTEZ (MOCKS) — tomado del script de patrón
# ---------------------------------------------------------

def create_mock_files_if_missing():
    """Genera archivos base si no existen para evitar crash en primera ejecución."""
    if not (BASE / "IW.npy").exists():
        np.save(BASE / "IW.npy", np.random.rand(4, 10))
        np.save(BASE / "bias_IW.npy", np.random.rand(10))
        np.save(BASE / "LW.npy", np.random.rand(1, 10))
        np.save(BASE / "bias_out.npy", np.random.rand(1))

    if not (BASE / "modelo_clusters_k3.pkl").exists():
        jd = np.arange(1, 366)
        # 3 patrones sintéticos
        p1 = np.exp(-((jd - 100) ** 2) / 600)  # Temprano
        p2 = np.exp(-((jd - 160) ** 2) / 900) + 0.3 * np.exp(-((jd - 260) ** 2) / 1200)  # Bimodal
        p3 = np.exp(-((jd - 230) ** 2) / 1500)  # Tardío
        mock_cluster = {
            "JD_common": jd,
            "curves_interp": [p2, p1, p3],
            "medoids_k3": [0, 1, 2],
        }
        with open(BASE / "modelo_clusters_k3.pkl", "wb") as f:
            pickle.dump(mock_cluster, f)

    if not (BASE / "meteo_daily.csv").exists():
        dates = pd.date_range(start="2026-01-01", periods=180)
        data = {
            "Fecha": dates,
            "TMAX": np.random.uniform(25, 35, size=len(dates)) - (np.arange(len(dates)) * 0.07),
            "TMIN": np.random.uniform(10, 18, size=len(dates)) - (np.arange(len(dates)) * 0.05),
            "Prec": np.random.choice([0, 0, 0, 5, 15, 45], size=len(dates)),
        }
        pd.DataFrame(data).to_csv(BASE / "meteo_daily.csv", index=False)


create_mock_files_if_missing()

# ---------------------------------------------------------
# MODELO ANN (idéntico a tus apps)
# ---------------------------------------------------------

class PracticalANNModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW, self.bIW, self.LW, self.bLW = IW, bIW, LW, bLW
        # normalización del entrenamiento original
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


def dtw_distance(a, b):
    """DTW simple (L1)."""
    na, nb = len(a), len(b)
    dp = np.full((na + 1, nb + 1), np.inf)
    dp[0, 0] = 0
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[na, nb])


@st.cache_resource
def load_models():
    """Carga ANN + modelo de clusters K3."""
    try:
        ann = PracticalANNModel(
            np.load(BASE / "IW.npy"),
            np.load(BASE / "bias_IW.npy"),
            np.load(BASE / "LW.npy"),
            np.load(BASE / "bias_out.npy"),
        )
        with open(BASE / "modelo_clusters_k3.pkl", "rb") as f:
            k3 = pickle.load(f)
        return ann, k3
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        return None, None


def get_data(file_input):
    """Lee meteo (subido o meteo_daily.csv). Estandariza columnas."""
    try:
        if file_input is not None:
            if file_input.name.endswith(".csv"):
                df = pd.read_csv(file_input, parse_dates=["Fecha"])
            else:
                df = pd.read_excel(file_input, parse_dates=["Fecha"])
        else:
            path = BASE / "meteo_daily.csv"
            df = pd.read_csv(path, parse_dates=["Fecha"]) if path.exists() else None

        if df is None:
            return None

        df.columns = [c.upper().strip() for c in df.columns]
        mapeo = {
            "FECHA": "Fecha",
            "DATE": "Fecha",
            "TMAX": "TMAX",
            "TMIN": "TMIN",
            "PREC": "Prec",
            "LLUVIA": "Prec",
        }
        df = df.rename(columns=mapeo)

        required_cols = ["Fecha", "TMAX", "TMIN", "Prec"]
        if not all(col in df.columns for col in required_cols):
            st.error(f"El archivo debe contener las columnas: {required_cols}")
            return None

        df = df[required_cols].copy()
        df = df.dropna(subset=required_cols).sort_values("Fecha").reset_index(drop=True)
        return df

    except Exception as e:
        st.error(f"Error procesando datos: {e}")
        return None


def safe_norm01(x: pd.Series) -> pd.Series:
    m = float(x.max()) if len(x) else 0.0
    if m <= 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x / m).clip(0, 1)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
LOGO_URL = "https://raw.githubusercontent.com/PREDWEEM/loliumTA_2026/main/logo.png"
st.sidebar.image(LOGO_URL, use_container_width=True)

st.sidebar.markdown("## 🌾 PREDWEEM")
st.sidebar.markdown("### LOLIUM TRES ARROYOS 2026")

archivo_usuario = st.sidebar.file_uploader("Subir Clima (Excel/CSV)", type=["xlsx", "csv"])

st.sidebar.divider()
st.sidebar.markdown("**Emergencia & Riesgo**")

umbral_er = st.sidebar.slider("Umbral de alerta (EMERREL)", 0.05, 0.80, 0.50, 0.01)

jd_inactivo = st.sidebar.slider("Forzar EMERREL=0 hasta JD", 0, 80, 30, 1)

usar_suavizado = st.sidebar.checkbox("Suavizar EMERREL (media móvil)", value=False)
win_suav = st.sidebar.slider("Ventana suavizado (días)", 3, 21, 7, 2, disabled=not usar_suavizado)

st.sidebar.caption("Riesgo = EMERREL / max(EMERREL)")

st.sidebar.divider()
st.sidebar.markdown("**Monitor térmico (Grados Día)**")
dga_optimo = st.sidebar.slider("Umbral térmico óptimo (°Cd)", 50, 800, 600, 10)
dga_critico = st.sidebar.slider("Umbral térmico crítico (°Cd)", 600, 1200, 850, 10)

st.sidebar.divider()
st.sidebar.markdown("**Clasificación de patrón (K=3, DTW)**")
fechas_corte = [
    "2026-04-01",
    "2026-04-15",
    "2026-05-01",
    "2026-05-15",
    "2026-06-01",
    "2026-06-15",
]
fecha_corte_str = st.sidebar.selectbox("Fecha de corte", fechas_corte, index=2)

st.sidebar.divider()

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

ann, k3 = load_models()
df = get_data(archivo_usuario)

st.title("🌾 PREDWEEM vK3 — LOLIUM TRES ARROYOS 2026")

if df is None or ann is None:
    st.warning("⚠️ Cargue un archivo de clima (CSV/Excel) o verifique que exista 'meteo_daily.csv' junto al script.")
    st.stop()

# --- PREPROCESAMIENTO ---
df["Julian_days"] = df["Fecha"].dt.dayofyear

# --- PREDICCIÓN ANN ---
X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
emerrel_raw, emerac = ann.predict(X)

df["EMERREL_raw"] = np.maximum(emerrel_raw, 0.0)
df.loc[df["Julian_days"] <= jd_inactivo, "EMERREL_raw"] = 0.0

df["EMERREL"] = df["EMERREL_raw"].copy()
if usar_suavizado:
    df["EMERREL"] = df["EMERREL"].rolling(int(win_suav), center=True, min_periods=1).mean()

# grados-día (base 2°C)
df["DG"] = np.maximum(((df["TMAX"] + df["TMIN"]) / 2) - 2.0, 0.0)

# riesgo 0–1 (norm por máximo)
df["Riesgo"] = safe_norm01(df["EMERREL"])

# ---------------------------------------------------------
# 1) MAPA DE RIESGO (discreto)
# ---------------------------------------------------------
st.subheader("🗺️ Mapa diario de riesgo de emergencia")

# Umbrales requeridos
# nulo: 0–0.10 (blanco)
# bajo: 0.11–0.33 (verde)
# medio: 0.34–0.66 (amarillo)
# alto: >0.66 (rojo)

def riesgo_a_nivel(r: float) -> str:
    if r <= 0.10:
        return "Nulo"
    if r <= 0.33:
        return "Bajo"
    if r <= 0.66:
        return "Medio"
    return "Alto"


def riesgo_a_color(r: float) -> str:
    if r <= 0.10:
        return "#ffffff"  # blanco
    if r <= 0.33:
        return "#16a34a"  # verde
    if r <= 0.66:
        return "#facc15"  # amarillo
    return "#ef4444"  # rojo


# Heatmap con cortes "duros" (manteniendo z en 0–1)
colorscale_hard = [
    [0.00, "#ffffff"],
    [0.10, "#ffffff"],
    [0.10, "#16a34a"],
    [0.33, "#16a34a"],
    [0.33, "#facc15"],
    [0.66, "#facc15"],
    [0.66, "#ef4444"],
    [1.00, "#ef4444"],
]

hover = (
    "<b>%{x|%d-%b-%Y}</b><br>"
    "Riesgo: %{z:.2f}<br>"
    "Nivel: %{customdata}<extra></extra>"
)

fig_risk = go.Figure(
    data=go.Heatmap(
        z=[df["Riesgo"].values],
        x=df["Fecha"],
        y=["Riesgo"],
        customdata=[[riesgo_a_nivel(float(r)) for r in df["Riesgo"].values]],
        colorscale=colorscale_hard,
        zmin=0,
        zmax=1,
        showscale=False,
        hovertemplate=hover,
    )
)
fig_risk.update_layout(height=140, margin=dict(t=30, b=0, l=10, r=10))
st.plotly_chart(fig_risk, use_container_width=True)

# ---------------------------------------------------------
# 2) SERIE TEMPORAL EMERREL
# ---------------------------------------------------------
st.subheader("📈 Dinámica de EMERREL")

fig_emer = go.Figure()
fig_emer.add_trace(
    go.Scatter(
        x=df["Fecha"],
        y=df["EMERREL"],
        mode="lines",
        name="EMERREL",
        fill="tozeroy",
    )
)
fig_emer.add_hline(
    y=umbral_er,
    line_dash="dash",
    annotation_text=f"Umbral alerta ({umbral_er:.2f})",
    annotation_position="top right",
)
fig_emer.update_layout(height=320, margin=dict(t=30, b=10, l=10, r=10), yaxis_title="EMERREL")
st.plotly_chart(fig_emer, use_container_width=True)

# ---------------------------------------------------------
# 3) MONITOR DE VENTANA (pulso + DG acumulado)
# ---------------------------------------------------------
st.subheader("🗓️ Monitor de ventana de aplicación (DG)")

indices_pulso = df.index[df["EMERREL"] >= umbral_er].tolist()
fecha_inicio_ventana = None

for i in range(len(indices_pulso) - 1):
    delta_dias = (df.loc[indices_pulso[i + 1], "Fecha"] - df.loc[indices_pulso[i], "Fecha"]).days
    if delta_dias <= 5:
        fecha_inicio_ventana = df.loc[indices_pulso[i], "Fecha"]
        break

dga_actual_acumulado = 0.0
df_ventana = pd.DataFrame()

if fecha_inicio_ventana is not None:
    df_ventana = df[df["Fecha"] >= fecha_inicio_ventana].copy()
    df_ventana["DGA_cum"] = df_ventana["DG"].cumsum()
    dga_actual_acumulado = float(df_ventana["DGA_cum"].iloc[-1]) if len(df_ventana) else 0.0

col_info, col_gauge = st.columns([1.6, 1])

with col_gauge:
    max_axis = dga_critico * 1.2
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=dga_actual_acumulado,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "<b>ACUMULACIÓN TÉRMICA</b><br><span style='font-size:0.8em;color:gray'>Grados Días (°Cd)</span>"},
            delta={"reference": dga_optimo},
            gauge={
                "axis": {"range": [None, max_axis]},
                "bar": {"thickness": 0.06},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, dga_optimo], "color": "#4ade80"},
                    {"range": [dga_optimo, dga_critico], "color": "#facc15"},
                    {"range": [dga_critico, max_axis], "color": "#f87171"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": dga_actual_acumulado},
            },
        )
    )
    fig_gauge.update_layout(height=290, margin=dict(t=50, b=10, l=30, r=30))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_info:
    if fecha_inicio_ventana is None:
        st.info(f"⏳ Sistema en espera: no se detectaron pulsos (EMERREL >= {umbral_er:.2f}).")
    else:
        def obtener_estado(objetivo):
            if len(df_ventana) == 0:
                return "-", "PENDIENTE"
            if dga_actual_acumulado >= objetivo:
                row = df_ventana[df_ventana["DGA_cum"] >= objetivo].iloc[0]
                return row["Fecha"].strftime("%d-%m-%Y"), "PASADO"
            return "Proyección futura", "PENDIENTE"

        f_opt, status_opt = obtener_estado(dga_optimo)
        f_cri, status_cri = obtener_estado(dga_critico)

        c1, c2, c3 = st.columns(3)
        c1.metric("Inicio de cohorte", fecha_inicio_ventana.strftime("%d-%b"))
        c2.metric("Límite óptimo", f_opt)
        c3.metric("Límite crítico", f_cri)

        st.markdown("**Estados fenológicos (DG acumulado):**")
        tabla = pd.DataFrame(
            {
                "Zona": ["🟢 VERDE", "🟡 AMARILLA", "🔴 ROJA"],
                "Fase": ["Ventana óptima", "Ventana crítica", "Fuera de ventana"],
                "Rango": [f"0–{dga_optimo} °Cd", f"{dga_optimo}–{dga_critico} °Cd", f"> {dga_critico} °Cd"],
                "Situación": [
                    "✅ ACTIVO" if dga_actual_acumulado <= dga_optimo else "",
                    "⚠️ ACTIVO" if (dga_optimo < dga_actual_acumulado <= dga_critico) else "",
                    "🚫 ACTIVO" if dga_actual_acumulado > dga_critico else "",
                ],
            }
        )
        st.table(tabla)

        if status_opt == "PENDIENTE":
            st.success(f"✅ Ventana ideal abierta. Faltan {dga_optimo - dga_actual_acumulado:.1f} °Cd para el óptimo.")
        elif status_cri == "PENDIENTE":
            st.warning(f"⚠️ Ventana crítica: se superó el óptimo ({f_opt}).")
        else:
            st.error(f"🚫 Límite crítico superado ({f_cri}).")

# ---------------------------------------------------------
# 4) CLASIFICACIÓN FUNCIONAL K=3 (DTW)
# ---------------------------------------------------------
st.divider()
st.subheader("📊 Clasificación funcional del patrón (K=3)")

if k3 is None or "JD_common" not in k3 or "curves_interp" not in k3:
    st.warning("No se encontró un modelo K=3 válido (modelo_clusters_k3.pkl).")
else:
    fecha_corte = pd.Timestamp(fecha_corte_str)
    df_trunc = df[df["Fecha"] <= fecha_corte].copy()

    if df_trunc.empty:
        st.info("ℹ️ La clasificación se activará cuando existan datos hasta la fecha de corte seleccionada.")
    else:
        JD_COMMON = np.array(k3["JD_common"], dtype=int)
        meds = [np.array(x, dtype=float) for x in k3["curves_interp"]]

        jd_corte = int(df_trunc["Julian_days"].max())
        jd_obs_grid = JD_COMMON[JD_COMMON <= jd_corte]

        # Observado normalizado a su propio máximo (forma)
        max_obs = float(df_trunc["EMERREL"].max())
        if max_obs <= 0:
            st.info("ℹ️ EMERREL aún sin señal (máximo = 0). Clasificación no informativa.")
        else:
            curva_obs = np.interp(jd_obs_grid, df_trunc["Julian_days"], df_trunc["EMERREL"])
            curva_obs_norm = (curva_obs / max_obs).clip(0, 1)

            # DTW contra cada medoide (también normalizado en el tramo)
            dists = []
            for m in meds:
                m_slice = m[JD_COMMON <= jd_corte]
                mmax = float(np.max(m_slice)) if len(m_slice) else 0.0
                m_norm = (m_slice / mmax) if mmax > 0 else m_slice
                dists.append(dtw_distance(curva_obs_norm, m_norm))

            order = np.argsort(dists)
            best, second = int(order[0]), int(order[1])

            # Confianza simple por separación relativa
            d1, d2 = float(dists[best]), float(dists[second])
            gap = (d2 - d1) / (d2 + 1e-9)
            if gap >= 0.25:
                conf = "ALTA"
            elif gap >= 0.10:
                conf = "MEDIA"
            else:
                conf = "BAJA"

            nombres = {0: "🌾 Intermedio / Bimodal", 1: "🌱 Temprano / Compacto", 2: "🍂 Tardío / Extendido"}
            nombre_final = nombres.get(best, f"Patrón {best}")

            cA, cB, cC = st.columns([2.2, 1, 1])
            cA.markdown(f"#### Patrón detectado: **{nombre_final}**")
            cB.metric("Confianza", conf)
            cC.metric("DTW (mejor)", f"{d1:.2f}")

            st.caption(f"Corte: {fecha_corte.strftime('%d-%b-%Y')} (JD={jd_corte}). Gap relativo vs 2º: {gap:.2f}.")

            # Gráfica: observado vs medoides (toggle)
            st.markdown("**Comparación (observado vs medoides):**")
            colL, colR = st.columns([1, 3])
            with colL:
                show_all = st.checkbox("Mostrar los 3 medoides", value=True)
                show_best = st.checkbox("Resaltar medoide asignado", value=True)

            with colR:
                fig_p = go.Figure()

                # observado (escala 0-1)
                fig_p.add_trace(
                    go.Scatter(
                        x=jd_obs_grid,
                        y=curva_obs_norm,
                        mode="lines",
                        name="Observado (norm 0–1)",
                        line=dict(width=3),
                    )
                )

                if show_all:
                    for i, m in enumerate(meds):
                        m_slice = m[JD_COMMON <= jd_corte]
                        mmax = float(np.max(m_slice)) if len(m_slice) else 0.0
                        m_norm = (m_slice / mmax) if mmax > 0 else m_slice
                        fig_p.add_trace(
                            go.Scatter(
                                x=jd_obs_grid,
                                y=m_norm,
                                mode="lines",
                                name=f"Medoide {i}",
                                line=dict(dash="dash", width=2),
                                opacity=0.55,
                            )
                        )

                if show_best:
                    m = meds[best]
                    m_slice = m[JD_COMMON <= jd_corte]
                    mmax = float(np.max(m_slice)) if len(m_slice) else 0.0
                    m_norm = (m_slice / mmax) if mmax > 0 else m_slice
                    fig_p.add_trace(
                        go.Scatter(
                            x=jd_obs_grid,
                            y=m_norm,
                            mode="lines",
                            name=f"Asignado (medoide {best})",
                            line=dict(width=4),
                        )
                    )

                fig_p.add_vline(x=jd_corte, line_width=1, line_dash="dot", annotation_text="Corte")

                fig_p.update_layout(
                    height=360,
                    margin=dict(t=20, b=20, l=10, r=10),
                    xaxis_title="Día Juliano",
                    yaxis_title="Curva normalizada (0–1)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_p, use_container_width=True)

# ---------------------------------------------------------
# 5) EXPORTACIÓN
# ---------------------------------------------------------
st.divider()
st.subheader("📦 Exportación")

# Agregamos nivel/categoría diaria de riesgo
niveles = [riesgo_a_nivel(float(r)) for r in df["Riesgo"].values]
df_export = df.copy()
df_export["Riesgo_Nivel"] = niveles

output = io.BytesIO()
with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
    df_export.to_excel(writer, index=False, sheet_name="PREDWEEM_Data")
    params = pd.DataFrame(
        {
            "Variable": [
                "umbral_er",
                "jd_inactivo",
                "suavizado",
                "win_suav",
                "dga_optimo",
                "dga_critico",
                "fecha_corte",
            ],
            "Valor": [
                umbral_er,
                jd_inactivo,
                bool(usar_suavizado),
                int(win_suav) if usar_suavizado else 0,
                dga_optimo,
                dga_critico,
                fecha_corte_str,
            ],
        }
    )
    params.to_excel(writer, index=False, sheet_name="Params")

st.download_button(
    label="📥 Descargar Excel",
    data=output.getvalue(),
    file_name="PREDWEEM_unificado_TA_2026.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.sidebar.caption("PREDWEEM vK3 | Unificado")
