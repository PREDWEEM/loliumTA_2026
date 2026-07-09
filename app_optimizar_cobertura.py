# -*- coding: utf-8 -*-
"""
🌾 PREDWEEM — Optimización de Ke y Modulador Térmico según cobertura

App Streamlit autosuficiente para buscar, para un nivel de cobertura dado,
la combinación de:
    - Coeficiente Hídrico Suelo (Ke)
    - Modulador Térmico Suelo
que mejor ajusta los flujos simulados PREDWEEM a los observados a campo.

Datos por defecto:
    - meteo_daily.csv
    - validacion.xlsx

Ejecución:
    streamlit run app_optimizar_cobertura.py

Nota técnica:
    Esta versión NO importa calibrar_suelo_desnudo_bimodal.py. Incluye dentro
    del mismo archivo el modelo ANN, ET0 Hargreaves, balance hídrico,
    sincronización por intervalos reales y métricas de validación.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
DEFAULT_METEO = "meteo_daily.csv"
DEFAULT_CAMPO = "validacion.xlsx"
DEFAULT_SALIDA = "resultados_optimizacion_cobertura_ke_modtermico.csv"
LAT_TRES_ARROYOS = -38.4500
CRITERIOS_MENOR_ES_MEJOR = {"RMSE_Campo", "MAE_Campo", "Abs_Bias_Campo"}


st.set_page_config(
    page_title="PREDWEEM Optimización Cobertura",
    page_icon="🌾",
    layout="wide",
)


# ---------------------------------------------------------------------
# Modelo ANN y motor biofísico autosuficiente
# ---------------------------------------------------------------------
class PracticalANNModel:
    def __init__(self, IW: np.ndarray, bIW: np.ndarray, LW: np.ndarray, bLW: np.ndarray):
        self.IW = IW
        self.bIW = bIW
        self.LW = LW
        self.bLW = bLW
        self.input_min = np.array([1, 0, -7, 0], dtype=float)
        self.input_max = np.array([300, 41, 25.5, 84], dtype=float)

    def normalize(self, X: np.ndarray) -> np.ndarray:
        return 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1

    def predict(self, Xreal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xn = self.normalize(Xreal)
        a1 = np.tanh(Xn @ self.IW + self.bIW)
        emerrel = (np.tanh((a1 @ self.LW.T).flatten() + self.bLW) + 1) / 2
        return emerrel, np.cumsum(emerrel)


@st.cache_resource
def cargar_modelo_ann_cache() -> PracticalANNModel:
    requeridos = ["IW.npy", "bias_IW.npy", "LW.npy", "bias_out.npy"]
    faltantes = [nombre for nombre in requeridos if not (BASE / nombre).exists()]
    if faltantes:
        raise FileNotFoundError(f"Faltan archivos ANN en el repositorio: {faltantes}")

    return PracticalANNModel(
        np.load(BASE / "IW.npy"),
        np.load(BASE / "bias_IW.npy"),
        np.load(BASE / "LW.npy"),
        np.load(BASE / "bias_out.npy"),
    )


def calcular_et0_hargreaves(jday: np.ndarray, tmax: np.ndarray, tmin: np.ndarray, latitud: float) -> np.ndarray:
    lat_rad = np.radians(latitud)
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * jday)
    dec = 0.409 * np.sin(2 * np.pi / 365 * jday - 1.39)
    ws = np.arccos(-np.tan(lat_rad) * np.tan(dec))
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(lat_rad) * np.sin(dec)
        + np.cos(lat_rad) * np.cos(dec) * np.sin(ws)
    )
    ra_mm = ra / 2.45
    tmean = (tmax + tmin) / 2.0
    trange = np.maximum(tmax - tmin, 0)
    return np.maximum(0.0023 * ra_mm * (tmean + 17.8) * np.sqrt(trange), 0)


def balance_hidrico_superficial(prec: np.ndarray, et0: np.ndarray, w_max: float, ke_suelo: float) -> np.ndarray:
    n = len(prec)
    w = np.zeros(n, dtype=float)
    if n == 0:
        return w
    w[0] = w_max / 2.0
    for i in range(1, n):
        kr = w[i - 1] / w_max if w_max > 0 else 0.0
        ke_dinamico = ke_suelo * kr
        evaporacion_real = et0[i] * ke_dinamico
        w[i] = max(0.0, min(w_max, w[i - 1] + prec[i] - evaporacion_real))
    return w


# ---------------------------------------------------------------------
# Utilidades de datos
# ---------------------------------------------------------------------
def materializar_archivo(uploaded_file, default_filename: str) -> Path:
    if uploaded_file is None:
        path = BASE / default_filename
        if not path.exists():
            raise FileNotFoundError(f"No se encontró {path}")
        return path

    tmp_dir = BASE / ".streamlit_tmp"
    tmp_dir.mkdir(exist_ok=True)
    suffix = Path(uploaded_file.name).suffix or Path(default_filename).suffix
    tmp_path = tmp_dir / f"uploaded_{Path(default_filename).stem}{suffix}"
    tmp_path.write_bytes(uploaded_file.getbuffer())
    return tmp_path


def leer_tabla(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)


def normalizar_columnas_meteo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).upper().strip() for c in df.columns]
    df = df.rename(
        columns={
            "FECHA": "Fecha",
            "DATE": "Fecha",
            "PREC": "Prec",
            "PP": "Prec",
            "LLUVIA": "Prec",
            "PRECIPITACION": "Prec",
            "PRECIPITACIÓN": "Prec",
        }
    )
    requeridas = ["Fecha", "TMAX", "TMIN", "Prec"]
    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas meteorológicas requeridas: {faltantes}")

    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df = df.dropna(subset=requeridas).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    df["Tmedia_aire"] = (df["TMAX"] + df["TMIN"]) / 2.0
    return df


def preparar_meteo_con_modulador(path: Path, mod_termico: float) -> pd.DataFrame:
    df = normalizar_columnas_meteo(leer_tabla(path))
    amp = (df["TMAX"] - df["TMIN"]) / 2.0
    df["TMAX_suelo"] = df["Tmedia_aire"] + amp * mod_termico
    df["TMIN_suelo"] = df["Tmedia_aire"] - amp * mod_termico
    df["ET0"] = calcular_et0_hargreaves(
        df["Julian_days"].to_numpy(float),
        df["TMAX"].to_numpy(float),
        df["TMIN"].to_numpy(float),
        latitud=LAT_TRES_ARROYOS,
    )
    return df


def cargar_campo(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    campo = leer_tabla(path)
    if campo.empty or len(campo.columns) < 2:
        raise ValueError("El archivo de campo debe tener al menos columnas de fecha y conteo/PLM2.")
    col_fecha = "FECHA" if "FECHA" in campo.columns else campo.columns[0]
    campo[col_fecha] = pd.to_datetime(campo[col_fecha])
    return campo


def columnas_campo(df_campo: pd.DataFrame) -> tuple[str, str]:
    col_fecha = "FECHA" if "FECHA" in df_campo.columns else df_campo.columns[0]
    col_plm2 = "PLM2" if "PLM2" in df_campo.columns else df_campo.columns[1]
    return col_fecha, col_plm2


def rango_float(inicio: float, fin: float, paso: float) -> np.ndarray:
    vals = np.arange(float(inicio), float(fin) + paso * 0.5, float(paso))
    return np.round(vals, 6)


# ---------------------------------------------------------------------
# Relación inicial cobertura -> valores esperados
# ---------------------------------------------------------------------
def estimar_ke_por_cobertura(cobertura_pct: float) -> float:
    """Priori agronómica: mayor cobertura reduce evaporación y, por tanto, Ke."""
    x = [0, 30, 70, 100]
    y = [1.25, 0.70, 0.30, 0.10]
    return float(np.interp(cobertura_pct, x, y))


def estimar_modtermico_por_cobertura(cobertura_pct: float) -> float:
    """Mayor cobertura amortigua la amplitud térmica del suelo."""
    x = [0, 30, 70, 100]
    y = [1.00, 0.95, 0.90, 0.80]
    return float(np.interp(cobertura_pct, x, y))


def rangos_automaticos(cobertura_pct: float) -> tuple[tuple[float, float], tuple[float, float]]:
    ke0 = estimar_ke_por_cobertura(cobertura_pct)
    mod0 = estimar_modtermico_por_cobertura(cobertura_pct)
    ke_range = (max(0.05, ke0 - 0.35), min(1.60, ke0 + 0.35))
    mod_range = (max(0.60, mod0 - 0.10), min(1.15, mod0 + 0.10))
    return ke_range, mod_range


# ---------------------------------------------------------------------
# Sincronización y métricas
# ---------------------------------------------------------------------
def sincronizar_intervalos_variables(df_sim: pd.DataFrame, df_campo: pd.DataFrame, col_fecha: str, col_plm2: str) -> pd.DataFrame:
    df_campo = df_campo.sort_values(col_fecha).copy()
    df_campo["Campo_Acum_Abs"] = df_campo[col_plm2].cumsum()
    fechas = df_campo[col_fecha].tolist()
    registros = []

    for i in range(1, len(fechas)):
        f_ini = fechas[i - 1]
        f_fin = fechas[i]
        dias_intervalo = (f_fin - f_ini).days
        obs_ini = df_campo.loc[df_campo[col_fecha] == f_ini, "Campo_Acum_Abs"].values[0]
        obs_fin = df_campo.loc[df_campo[col_fecha] == f_fin, "Campo_Acum_Abs"].values[0]
        flujo_obs = max(0.0, obs_fin - obs_ini)
        flujo_sim = df_sim.loc[(df_sim["Fecha"] > f_ini) & (df_sim["Fecha"] <= f_fin), "EMERREL"].sum()
        acum_sim_fin = df_sim.loc[df_sim["Fecha"] <= f_fin, "EMERREL"].sum()
        registros.append(
            {
                "Fecha": f_fin,
                "Dias_Intervalo": dias_intervalo,
                "Flujo_Obs_Abs": flujo_obs,
                "Flujo_Sim_Abs": flujo_sim,
                "Acum_Obs_Abs": obs_fin,
                "Acum_Sim_Abs": acum_sim_fin,
            }
        )

    out = pd.DataFrame(registros)
    if out.empty:
        return out

    total_obs = out["Flujo_Obs_Abs"].sum()
    total_sim = df_sim.loc[df_sim["Fecha"] <= fechas[-1], "EMERREL"].sum()
    out["Campo_Relativo"] = out["Flujo_Obs_Abs"] / total_obs if total_obs > 0 else 0.0
    out["Sim_Relativo"] = out["Flujo_Sim_Abs"] / total_sim if total_sim > 0 else 0.0
    return out


def metricas_evento(df_sync: pd.DataFrame, umbral_deteccion: float = 0.05) -> dict:
    if df_sync.empty or len(df_sync) < 2:
        return {"F1_Campo": np.nan, "NSE_Campo": np.nan, "Pearson_Campo": np.nan}

    obs = df_sync["Campo_Relativo"].to_numpy(float)
    sim = df_sync["Sim_Relativo"].to_numpy(float)
    active = (obs > 0) | (sim > 0)

    if active.sum() >= 2 and np.std(obs[active]) > 0 and np.std(sim[active]) > 0:
        pearson = float(np.corrcoef(obs[active], sim[active])[0, 1])
    else:
        pearson = 0.0

    denom = np.sum((obs[active] - np.mean(obs[active])) ** 2) if active.sum() >= 2 else 0.0
    nse = float(1 - np.sum((sim[active] - obs[active]) ** 2) / denom) if denom > 0 else 0.0

    obs_evt = df_sync["Campo_Relativo"] > umbral_deteccion
    sim_evt = df_sync["Sim_Relativo"] > umbral_deteccion
    hits = int((obs_evt & sim_evt).sum())
    fp = int((~obs_evt & sim_evt).sum())
    miss = int((obs_evt & ~sim_evt).sum())
    precision = hits / (hits + fp) if hits + fp > 0 else 0.0
    recall = hits / (hits + miss) if hits + miss > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"F1_Campo": f1, "NSE_Campo": nse, "Pearson_Campo": pearson}


def metricas_ajuste_campo(df_sync: pd.DataFrame, umbral_deteccion: float) -> dict:
    if df_sync.empty or len(df_sync) < 2:
        return {
            "Ajuste_Campo_Compuesto": np.nan,
            "RMSE_Campo": np.nan,
            "MAE_Campo": np.nan,
            "Bias_Campo": np.nan,
            "Abs_Bias_Campo": np.nan,
            "NSE_Campo": np.nan,
            "Pearson_Campo": np.nan,
            "R2_Campo": np.nan,
            "F1_Campo": np.nan,
        }

    base = metricas_evento(df_sync, umbral_deteccion=umbral_deteccion)
    obs = df_sync["Campo_Relativo"].to_numpy(float)
    sim = df_sync["Sim_Relativo"].to_numpy(float)
    active = (obs > 0) | (sim > 0)

    if active.sum() == 0:
        return {
            **base,
            "Ajuste_Campo_Compuesto": 0.0,
            "RMSE_Campo": 1.0,
            "MAE_Campo": 1.0,
            "Bias_Campo": 0.0,
            "Abs_Bias_Campo": 0.0,
            "R2_Campo": 0.0,
        }

    obs_a = obs[active]
    sim_a = sim[active]
    resid = sim_a - obs_a
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    bias = float(np.mean(resid))
    pearson = float(base.get("Pearson_Campo", 0.0)) if pd.notna(base.get("Pearson_Campo", np.nan)) else 0.0
    nse = float(base.get("NSE_Campo", 0.0)) if pd.notna(base.get("NSE_Campo", np.nan)) else 0.0
    f1 = float(base.get("F1_Campo", 0.0)) if pd.notna(base.get("F1_Campo", np.nan)) else 0.0
    r2 = max(0.0, pearson) ** 2

    nse_score = float(np.clip((nse + 1.0) / 2.0, 0.0, 1.0))
    pearson_score = float(np.clip((pearson + 1.0) / 2.0, 0.0, 1.0))
    rmse_score = float(1.0 / (1.0 + rmse))
    mae_score = float(1.0 / (1.0 + mae))
    ajuste = 0.35 * nse_score + 0.25 * rmse_score + 0.20 * pearson_score + 0.10 * mae_score + 0.10 * f1

    return {
        **base,
        "Ajuste_Campo_Compuesto": ajuste,
        "RMSE_Campo": rmse,
        "MAE_Campo": mae,
        "Bias_Campo": bias,
        "Abs_Bias_Campo": abs(bias),
        "R2_Campo": r2,
    }


# ---------------------------------------------------------------------
# Simulación y optimización
# ---------------------------------------------------------------------
def simular_ke_modtermico(
    meteo_path: Path,
    ke_suelo: float,
    mod_termico: float,
    w_max: float,
    humedad_mid: float,
    corte_seco: float,
    umbral_choque_hidrico: float,
    umbral_termoinhibicion: float,
    umbral_primer_pico: Optional[float],
) -> tuple[pd.DataFrame, Optional[pd.Timestamp]]:
    df = preparar_meteo_con_modulador(meteo_path, mod_termico=mod_termico)
    modelo_ann = cargar_modelo_ann_cache()

    X = df[["Julian_days", "TMAX_suelo", "TMIN_suelo", "Prec"]].to_numpy(float)
    emerrel_raw, _ = modelo_ann.predict(X)
    df["EMERREL_RAW"] = np.maximum(emerrel_raw, 0.0)
    df.loc[df["Julian_days"] <= 45, "EMERREL_RAW"] = 0.0

    df["Prec_3d"] = df["Prec"].rolling(window=3, min_periods=1).sum()
    mask_ruptura = (
        (df["Julian_days"] > 45)
        & (df["Julian_days"] <= 110)
        & (df["Prec_3d"] >= umbral_choque_hidrico)
    )
    df.loc[mask_ruptura, "EMERREL_RAW"] = np.maximum(df.loc[mask_ruptura, "EMERREL_RAW"], 0.75)

    df["W_superficial"] = balance_hidrico_superficial(
        df["Prec"].to_numpy(float),
        df["ET0"].to_numpy(float),
        w_max=w_max,
        ke_suelo=ke_suelo,
    )
    humedad_relativa = df["W_superficial"] / w_max
    df["Humedad_Relativa"] = humedad_relativa
    df["Hydric_Factor"] = 1.0 / (1.0 + np.exp(-10.0 * (humedad_relativa - humedad_mid)))
    df["EMERREL"] = df["EMERREL_RAW"] * df["Hydric_Factor"]
    df.loc[humedad_relativa < corte_seco, "EMERREL"] = 0.0

    df["Lluvia_Recarga"] = (df["Prec"] >= w_max).cummax()
    df.loc[~df["Lluvia_Recarga"], "EMERREL"] = 0.0

    df["Tmedia_5d"] = df["Tmedia_aire"].rolling(window=5, min_periods=1).mean()
    df.loc[df["Tmedia_5d"] >= umbral_termoinhibicion, "EMERREL"] = 0.0
    df.loc[df["Julian_days"] <= 45, "EMERREL"] = 0.0
    df["EMERREL"] = np.clip(df["EMERREL"], 0.0, 1.0)

    fecha_primer_pico = None
    if umbral_primer_pico is not None:
        candidatos = df.index[df["EMERREL"] > umbral_primer_pico].tolist()
        if candidatos:
            idx = candidatos[0]
            fecha_primer_pico = df.loc[idx, "Fecha"]
            df["Primer_Pico_Habilitado"] = df.index >= idx
            df.loc[df.index < idx, "EMERREL"] = 0.0
        else:
            df["Primer_Pico_Habilitado"] = False
            df["EMERREL"] = 0.0
    else:
        df["Primer_Pico_Habilitado"] = True

    return df, fecha_primer_pico


def evaluar_combinacion(
    meteo_path: Path,
    df_campo: pd.DataFrame,
    ke_suelo: float,
    mod_termico: float,
    w_max: float,
    humedad_mid: float,
    corte_seco: float,
    umbral_choque_hidrico: float,
    umbral_termoinhibicion: float,
    umbral_primer_pico: Optional[float],
    umbral_deteccion: float,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    df_sim, fecha_primer_pico = simular_ke_modtermico(
        meteo_path=meteo_path,
        ke_suelo=ke_suelo,
        mod_termico=mod_termico,
        w_max=w_max,
        humedad_mid=humedad_mid,
        corte_seco=corte_seco,
        umbral_choque_hidrico=umbral_choque_hidrico,
        umbral_termoinhibicion=umbral_termoinhibicion,
        umbral_primer_pico=umbral_primer_pico,
    )
    col_fecha, col_plm2 = columnas_campo(df_campo)
    df_sync = sincronizar_intervalos_variables(df_sim, df_campo, col_fecha, col_plm2)
    metricas = metricas_ajuste_campo(df_sync, umbral_deteccion=umbral_deteccion)
    fila = {
        "Ke_Suelo": float(ke_suelo),
        "Mod_Termico": float(mod_termico),
        "W_Max_mm": float(w_max),
        "Humedad_mid": float(humedad_mid),
        "Corte_seco": float(corte_seco),
        "Fecha_Primer_Pico": fecha_primer_pico,
        **metricas,
    }
    return fila, df_sim, df_sync


def ordenar_ranking(df: pd.DataFrame, criterio: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if criterio in CRITERIOS_MENOR_ES_MEJOR:
        out[criterio] = out[criterio].fillna(np.inf)
        return out.sort_values([criterio, "Ajuste_Campo_Compuesto"], ascending=[True, False]).reset_index(drop=True)
    out[criterio] = out[criterio].fillna(-np.inf)
    return out.sort_values([criterio, "Ajuste_Campo_Compuesto"], ascending=[False, False]).reset_index(drop=True)


def ejecutar_barrido(
    meteo_path: Path,
    df_campo: pd.DataFrame,
    ke_values: np.ndarray,
    mod_values: np.ndarray,
    w_max: float,
    humedad_mid: float,
    corte_seco: float,
    umbral_choque_hidrico: float,
    umbral_termoinhibicion: float,
    umbral_primer_pico: Optional[float],
    umbral_deteccion: float,
    criterio: str,
) -> pd.DataFrame:
    resultados: list[dict] = []
    total = len(ke_values) * len(mod_values)
    done = 0
    progress = st.progress(0, text="Preparando barrido Ke × modulador térmico...")

    for ke in ke_values:
        for mod in mod_values:
            fila, _, _ = evaluar_combinacion(
                meteo_path=meteo_path,
                df_campo=df_campo,
                ke_suelo=float(ke),
                mod_termico=float(mod),
                w_max=w_max,
                humedad_mid=humedad_mid,
                corte_seco=corte_seco,
                umbral_choque_hidrico=umbral_choque_hidrico,
                umbral_termoinhibicion=umbral_termoinhibicion,
                umbral_primer_pico=umbral_primer_pico,
                umbral_deteccion=umbral_deteccion,
            )
            resultados.append(fila)
            done += 1
            if done % 10 == 0 or done == total:
                progress.progress(done / max(total, 1), text=f"Evaluando {done}/{total} combinaciones")

    progress.empty()
    return ordenar_ranking(pd.DataFrame(resultados), criterio)


# ---------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------
def grafico_intervalos(df_sync: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df_sync.empty:
        return fig
    fig.add_trace(
        go.Scatter(
            x=df_sync["Fecha"],
            y=df_sync["Campo_Relativo"],
            mode="markers+lines",
            name="Observado campo",
            marker=dict(size=10, symbol="diamond"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_sync["Fecha"],
            y=df_sync["Sim_Relativo"],
            mode="markers+lines",
            name="Simulado agregado",
            line=dict(dash="dot"),
        )
    )
    fig.update_layout(
        title="Flujo observado vs flujo simulado agregado por intervalo",
        xaxis_title="Fecha de muestreo",
        yaxis_title="Flujo relativo",
        height=460,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def grafico_diario(df_sim: pd.DataFrame, df_campo: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    y_sim = df_sim["EMERREL"] / df_sim["EMERREL"].max() if df_sim["EMERREL"].max() > 0 else df_sim["EMERREL"]
    fig.add_trace(
        go.Scatter(
            x=df_sim["Fecha"],
            y=y_sim,
            mode="lines",
            name="Simulado diario normalizado",
            fill="tozeroy",
        )
    )
    col_fecha, col_plm2 = columnas_campo(df_campo)
    campo = df_campo.copy()
    campo[col_fecha] = pd.to_datetime(campo[col_fecha])
    max_obs = campo[col_plm2].max()
    campo["Campo_norm"] = campo[col_plm2] / max_obs if max_obs > 0 else 0.0
    fig.add_trace(
        go.Scatter(
            x=campo[col_fecha],
            y=campo["Campo_norm"],
            mode="markers+lines",
            name="Campo normalizado",
            marker=dict(size=10, symbol="diamond"),
        )
    )
    fig.update_layout(
        title="Emergencia diaria simulada y observaciones de campo",
        xaxis_title="Fecha",
        yaxis_title="Emergencia normalizada 0–1",
        height=460,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def grafico_heatmap(ranking: pd.DataFrame, criterio: str) -> go.Figure:
    fig = go.Figure()
    if ranking.empty or criterio not in ranking.columns:
        return fig
    pivot = ranking.pivot_table(index="Mod_Termico", columns="Ke_Suelo", values=criterio, aggfunc="mean")
    fig.add_trace(go.Heatmap(x=pivot.columns, y=pivot.index, z=pivot.values, colorbar=dict(title=criterio)))
    fig.update_layout(
        title=f"Superficie de respuesta — {criterio}",
        xaxis_title="Ke suelo",
        yaxis_title="Modulador térmico",
        height=480,
    )
    return fig


def grafico_uno_a_uno(df_sync: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df_sync.empty:
        return fig
    max_xy = float(max(df_sync["Campo_Relativo"].max(), df_sync["Sim_Relativo"].max(), 0.01))
    fig.add_trace(
        go.Scatter(
            x=df_sync["Campo_Relativo"],
            y=df_sync["Sim_Relativo"],
            mode="markers+text",
            text=df_sync["Fecha"].dt.strftime("%d/%m"),
            textposition="top center",
            name="Intervalos",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, max_xy],
            y=[0, max_xy],
            mode="lines",
            name="Línea 1:1",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(
        title="Línea 1:1 — observado vs simulado",
        xaxis_title="Observado relativo",
        yaxis_title="Simulado relativo",
        height=460,
    )
    return fig


def fmt(valor: float, dec: int = 3) -> str:
    if pd.isna(valor):
        return "NA"
    return f"{valor:.{dec}f}"


# ---------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------
st.title("🌾 PREDWEEM — Optimización de Ke y Modulador Térmico por cobertura")
st.caption("Tres Arroyos 2026 · Ajuste contra observaciones de campo · Barrido 2D Ke × modulador térmico")

with st.sidebar:
    st.header("📂 Datos")
    meteo_upload = st.file_uploader("Meteorología", type=["csv", "xlsx", "xls"], help=f"Por defecto: {DEFAULT_METEO}")
    campo_upload = st.file_uploader("Validación de campo", type=["xlsx", "xls", "csv"], help=f"Por defecto: {DEFAULT_CAMPO}")

    st.divider()
    st.header("🌾 Cobertura")
    cobertura_pct = st.slider("Cobertura de suelo / rastrojo (%)", 0, 100, 0, step=5)
    ke_prior = estimar_ke_por_cobertura(cobertura_pct)
    mod_prior = estimar_modtermico_por_cobertura(cobertura_pct)
    st.metric("Ke esperado", f"{ke_prior:.2f}")
    st.metric("Modulador térmico esperado", f"{mod_prior:.2f}")

    ke_auto, mod_auto = rangos_automaticos(cobertura_pct)

    st.divider()
    st.header("🔎 Rangos a optimizar")
    usar_rangos_auto = st.checkbox("Usar rangos automáticos según cobertura", value=True)

    if usar_rangos_auto:
        ke_min, ke_max = ke_auto
        mod_min, mod_max = mod_auto
        st.caption(f"Ke: {ke_min:.2f}–{ke_max:.2f} · Mod: {mod_min:.2f}–{mod_max:.2f}")
    else:
        ke_rango = st.slider("Rango Ke", 0.05, 1.60, (float(ke_auto[0]), float(ke_auto[1])), step=0.05)
        mod_rango = st.slider("Rango modulador térmico", 0.60, 1.15, (float(mod_auto[0]), float(mod_auto[1])), step=0.01)
        ke_min, ke_max = ke_rango
        mod_min, mod_max = mod_rango

    ke_step = st.number_input("Paso Ke", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
    mod_step = st.number_input("Paso modulador térmico", min_value=0.005, max_value=0.10, value=0.02, step=0.005, format="%.3f")

    st.divider()
    st.header("💧 Parámetros fijos del suelo")
    w_max = st.number_input("W_Max superficial (mm)", min_value=1.0, max_value=60.0, value=26.0, step=1.0)
    humedad_mid = st.number_input("Humedad_mid", min_value=0.05, max_value=0.80, value=0.36, step=0.01, format="%.2f")
    corte_seco = st.number_input("Corte seco HR", min_value=0.01, max_value=0.80, value=0.25, step=0.01, format="%.2f")

    st.divider()
    st.header("⚙️ Filtros fisiológicos")
    umbral_choque_hidrico = st.slider("Choque hídrico 3 días (mm)", 10.0, 100.0, 45.0, step=1.0)
    umbral_termoinhibicion = st.number_input("Termoinhibición media 5 días (°C)", 15.0, 35.0, 24.0, step=0.5)
    usar_filtro_primer_pico = st.checkbox("Usar filtro de primer pico", value=True)
    umbral_primer_pico = st.number_input("Umbral primer pico", 0.001, 0.80, 0.05, step=0.005, format="%.3f")
    if not usar_filtro_primer_pico:
        umbral_primer_pico = None

    st.divider()
    st.header("🎯 Criterio de ajuste")
    criterio = st.selectbox(
        "Optimizar según",
        [
            "Ajuste_Campo_Compuesto",
            "RMSE_Campo",
            "NSE_Campo",
            "Pearson_Campo",
            "F1_Campo",
            "MAE_Campo",
            "Abs_Bias_Campo",
            "R2_Campo",
        ],
        index=0,
    )
    umbral_deteccion = st.slider("Umbral evento para F1", 0.01, 0.30, 0.05, step=0.01)

    st.divider()
    ejecutar = st.button("🚀 Optimizar Ke y modulador", type="primary")

try:
    meteo_path = materializar_archivo(meteo_upload, DEFAULT_METEO)
    campo_path = materializar_archivo(campo_upload, DEFAULT_CAMPO)
    df_campo = cargar_campo(campo_path)
    if df_campo is None:
        raise FileNotFoundError("No se pudo cargar el archivo de validación de campo.")

    col_fecha, col_plm2 = columnas_campo(df_campo)
    df_campo[col_fecha] = pd.to_datetime(df_campo[col_fecha])
    df_campo = df_campo.sort_values(col_fecha).reset_index(drop=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cobertura", f"{cobertura_pct} %")
    c2.metric("Ke esperado", f"{ke_prior:.2f}")
    c3.metric("Mod térmico esperado", f"{mod_prior:.2f}")
    c4.metric("Intervalos campo", f"{max(len(df_campo)-1, 0)}")

    st.success(f"Meteorología: `{meteo_path.name}` · Campo: `{campo_path.name}`")

    if ejecutar:
        ke_values = rango_float(ke_min, ke_max, ke_step)
        mod_values = rango_float(mod_min, mod_max, mod_step)

        if len(ke_values) == 0 or len(mod_values) == 0:
            st.error("Los rangos seleccionados no generan combinaciones válidas.")
            st.stop()

        with st.spinner("Optimizando Ke y modulador térmico contra observaciones de campo..."):
            ranking = ejecutar_barrido(
                meteo_path=meteo_path,
                df_campo=df_campo,
                ke_values=ke_values,
                mod_values=mod_values,
                w_max=w_max,
                humedad_mid=humedad_mid,
                corte_seco=corte_seco,
                umbral_choque_hidrico=umbral_choque_hidrico,
                umbral_termoinhibicion=umbral_termoinhibicion,
                umbral_primer_pico=umbral_primer_pico,
                umbral_deteccion=umbral_deteccion,
                criterio=criterio,
            )

        if ranking.empty:
            st.error("El barrido no produjo resultados.")
            st.stop()

        best = ranking.iloc[0]
        _, df_best, df_sync_best = evaluar_combinacion(
            meteo_path=meteo_path,
            df_campo=df_campo,
            ke_suelo=float(best["Ke_Suelo"]),
            mod_termico=float(best["Mod_Termico"]),
            w_max=w_max,
            humedad_mid=humedad_mid,
            corte_seco=corte_seco,
            umbral_choque_hidrico=umbral_choque_hidrico,
            umbral_termoinhibicion=umbral_termoinhibicion,
            umbral_primer_pico=umbral_primer_pico,
            umbral_deteccion=umbral_deteccion,
        )

        st.subheader("🏆 Parámetros óptimos para la cobertura seleccionada")
        st.caption(f"Criterio ganador: `{criterio}` · Cobertura: {cobertura_pct} %")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Ke óptimo", f"{best['Ke_Suelo']:.2f}")
        m2.metric("Mod térmico óptimo", f"{best['Mod_Termico']:.2f}")
        m3.metric("Ajuste compuesto", fmt(best.get("Ajuste_Campo_Compuesto", np.nan)))
        m4.metric("RMSE campo", fmt(best.get("RMSE_Campo", np.nan)))
        m5.metric("NSE campo", fmt(best.get("NSE_Campo", np.nan)))

        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Pearson", fmt(best.get("Pearson_Campo", np.nan)))
        v2.metric("R²", fmt(best.get("R2_Campo", np.nan)))
        v3.metric("F1", fmt(best.get("F1_Campo", np.nan)))
        v4.metric("Bias", fmt(best.get("Bias_Campo", np.nan)))

        st.code(
            f"""
# Parámetros óptimos para app_emergencia.py
cobertura_pct = {cobertura_pct}
ke_val = {best['Ke_Suelo']:.2f}
mod_termico = {best['Mod_Termico']:.2f}
w_max_val = {w_max:.1f}
humedad_mid = {humedad_mid:.2f}
corte_seco = {corte_seco:.2f}
umbral_choque_hidrico = {umbral_choque_hidrico:.1f}
umbral_termoinhibicion = {umbral_termoinhibicion:.1f}
""".strip(),
            language="python",
        )

        st.plotly_chart(grafico_intervalos(df_sync_best), width="stretch")
        st.plotly_chart(grafico_uno_a_uno(df_sync_best), width="stretch")
        st.plotly_chart(grafico_diario(df_best, df_campo), width="stretch")
        st.plotly_chart(grafico_heatmap(ranking, criterio), width="stretch")

        st.subheader("📊 Ranking de combinaciones Ke × modulador térmico")
        cols = [
            "Ke_Suelo",
            "Mod_Termico",
            "Ajuste_Campo_Compuesto",
            "RMSE_Campo",
            "MAE_Campo",
            "NSE_Campo",
            "Pearson_Campo",
            "R2_Campo",
            "F1_Campo",
            "Bias_Campo",
            "Fecha_Primer_Pico",
        ]
        cols = [c for c in cols if c in ranking.columns]
        st.dataframe(ranking[cols].head(80), width="stretch", height=430)
        st.download_button(
            "📥 Descargar ranking completo CSV",
            data=ranking.to_csv(index=False).encode("utf-8"),
            file_name=DEFAULT_SALIDA,
            mime="text/csv",
        )

        with st.expander("Ver tabla sincronizada del mejor ajuste"):
            st.dataframe(df_sync_best, width="stretch", height=360)
        with st.expander("Ver simulación diaria completa"):
            st.dataframe(df_best, width="stretch", height=420)

    else:
        st.info("Seleccione un nivel de cobertura y ejecute la optimización desde la barra lateral.")
        st.markdown(
            """
**Qué hace esta app**

1. Toma un porcentaje de cobertura dado.
2. Define valores esperados de `Ke` y `mod_termico` según esa cobertura.
3. Ejecuta un barrido 2D alrededor de esos valores.
4. Agrega la emergencia diaria simulada a los intervalos reales de muestreo.
5. Elige la combinación que mejor ajusta los datos observados a campo.
"""
        )

except Exception as exc:
    st.error("No se pudo ejecutar la app de optimización.")
    st.exception(exc)
