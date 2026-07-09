# -*- coding: utf-8 -*-
"""
Miniapp Streamlit para calibrar PREDWEEM Tres Arroyos con datos reales
`validacion.xlsx`, meteorologia `meteo_daily.csv` y cobertura de suelo = 0 %.

Ejecucion local:

    streamlit run calibrar_validacion_suelo_desnudo.py

Objetivo agronomico:
    Buscar automaticamente los parametros edaficos que mejor ajusten
    los flujos simulados diarios, agregados por intervalos de muestreo,
    a los datos observados a campo.

La app mantiene fijo:
    - Cobertura de suelo: 0 %
    - Modulador termico: 1.00

Y calibra:
    - W_Max_mm
    - Ke_Suelo
    - Humedad_mid
    - Corte_seco
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from calibrar_suelo_desnudo_bimodal import (
    BASE,
    LAT_TRES_ARROYOS,
    MOD_TERMICO_SUELO_DESNUDO,
    cargar_campo,
    load_ann,
    metricas_evento,
    preparar_meteo,
    score_dos_picos,
    simular,
    sincronizar_intervalos_variables,
)


DEFAULT_METEO = "meteo_daily.csv"
DEFAULT_CAMPO = "validacion.xlsx"
DEFAULT_SALIDA = "resultados_calibracion_validacion_suelo_desnudo.csv"
COBERTURA_PCT = 0
TARGET_PEAKS = [pd.Timestamp("2026-05-24"), pd.Timestamp("2026-06-28")]

CRITERIOS_MAYOR_ES_MEJOR = {
    "Ajuste_Campo_Compuesto",
    "NSE_Campo",
    "Pearson_Campo",
    "F1_Campo",
    "R2_Campo",
    "Score_Dos_Picos",
}
CRITERIOS_MENOR_ES_MEJOR = {"RMSE_Campo", "MAE_Campo", "Abs_Bias_Campo"}


st.set_page_config(
    page_title="PREDWEEM Calibracion Campo",
    page_icon="🌾",
    layout="wide",
)


# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------
def materializar_archivo(uploaded_file, default_filename: str) -> Path:
    """Devuelve el archivo por defecto o guarda temporalmente el upload."""
    if uploaded_file is None:
        path = BASE / default_filename
        if not path.exists():
            raise FileNotFoundError(f"No se encontro {path}")
        return path

    tmp_dir = BASE / ".streamlit_tmp"
    tmp_dir.mkdir(exist_ok=True)
    suffix = Path(uploaded_file.name).suffix or Path(default_filename).suffix
    tmp_path = tmp_dir / f"uploaded_{Path(default_filename).stem}{suffix}"
    tmp_path.write_bytes(uploaded_file.getbuffer())
    return tmp_path


def rango_float(inicio: float, fin: float, paso: float) -> np.ndarray:
    vals = np.arange(float(inicio), float(fin) + paso * 0.5, float(paso))
    return np.round(vals, 6)


def columnas_campo(df_campo: pd.DataFrame) -> tuple[str, str]:
    col_fecha = "FECHA" if "FECHA" in df_campo.columns else df_campo.columns[0]
    col_plm2 = "PLM2" if "PLM2" in df_campo.columns else df_campo.columns[1]
    return col_fecha, col_plm2


def metricas_ajuste_observado(df_sync: pd.DataFrame, umbral_deteccion: float = 0.05) -> dict:
    """
    Calcula metricas continuas y binarias sobre los intervalos de campo.

    Importante: df_sync ya contiene el flujo simulado diario agregado entre
    dos fechas consecutivas de observacion. Por eso la comparacion es campo
    semanal/intervalar vs simulacion acumulada en el mismo intervalo.
    """
    if df_sync.empty or len(df_sync) < 2:
        return {
            "F1_Campo": np.nan,
            "NSE_Campo": np.nan,
            "Pearson_Campo": np.nan,
            "R2_Campo": np.nan,
            "RMSE_Campo": np.nan,
            "MAE_Campo": np.nan,
            "Bias_Campo": np.nan,
            "Abs_Bias_Campo": np.nan,
            "Ajuste_Campo_Compuesto": np.nan,
        }

    base = metricas_evento(df_sync, umbral_deteccion=umbral_deteccion)
    obs = df_sync["Campo_Relativo"].to_numpy(float)
    sim = df_sync["Sim_Relativo"].to_numpy(float)
    active = (obs > 0) | (sim > 0)

    if active.sum() == 0:
        return {
            **base,
            "R2_Campo": 0.0,
            "RMSE_Campo": 1.0,
            "MAE_Campo": 1.0,
            "Bias_Campo": 0.0,
            "Abs_Bias_Campo": 0.0,
            "Ajuste_Campo_Compuesto": 0.0,
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

    # Normalizacion robusta a 0-1 para un criterio compuesto.
    nse_score = float(np.clip((nse + 1.0) / 2.0, 0.0, 1.0))
    pearson_score = float(np.clip((pearson + 1.0) / 2.0, 0.0, 1.0))
    rmse_score = float(1.0 / (1.0 + rmse))
    mae_score = float(1.0 / (1.0 + mae))

    ajuste_compuesto = (
        0.35 * nse_score
        + 0.25 * rmse_score
        + 0.20 * pearson_score
        + 0.10 * mae_score
        + 0.10 * f1
    )

    return {
        **base,
        "R2_Campo": r2,
        "RMSE_Campo": rmse,
        "MAE_Campo": mae,
        "Bias_Campo": bias,
        "Abs_Bias_Campo": abs(bias),
        "Ajuste_Campo_Compuesto": ajuste_compuesto,
    }


def ordenar_ranking(ranking: pd.DataFrame, criterio: str) -> pd.DataFrame:
    """Ordena el ranking usando como criterio principal el ajuste a campo."""
    if ranking.empty:
        return ranking

    if criterio not in ranking.columns:
        criterio = "Ajuste_Campo_Compuesto" if "Ajuste_Campo_Compuesto" in ranking.columns else "Score_Dos_Picos"

    ranking = ranking.copy()
    if criterio in CRITERIOS_MENOR_ES_MEJOR:
        ranking[criterio] = ranking[criterio].fillna(np.inf)
        orden = [criterio, "Score_Dos_Picos"] if "Score_Dos_Picos" in ranking.columns else [criterio]
        asc = [True, False] if len(orden) == 2 else [True]
    else:
        ranking[criterio] = ranking[criterio].fillna(-np.inf)
        orden = [criterio, "Score_Dos_Picos"] if "Score_Dos_Picos" in ranking.columns else [criterio]
        asc = [False, False] if len(orden) == 2 else [False]

    return ranking.sort_values(orden, ascending=asc).reset_index(drop=True)


def ejecutar_barrido_streamlit(
    meteo_path: Path,
    campo_path: Optional[Path],
    w_values: np.ndarray,
    ke_values: np.ndarray,
    humedad_mid_values: list[float],
    corte_seco_values: list[float],
    umbral_choque_hidrico: float,
    umbral_termoinhibicion: float,
    umbral_primer_pico: Optional[float],
    umbral_deteccion: float,
    criterio_ajuste: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Ejecuta el barrido y devuelve ranking, meteo preparada y campo."""
    df_meteo = preparar_meteo(meteo_path)
    modelo_ann = load_ann(BASE)
    df_campo = cargar_campo(campo_path) if campo_path is not None else None

    total_iter = 0
    for _w in w_values:
        for _ke in ke_values:
            for hum in humedad_mid_values:
                for corte in corte_seco_values:
                    if corte < hum:
                        total_iter += 1

    resultados: list[dict] = []
    progress = st.progress(0, text="Preparando barrido edafico...")
    done = 0

    for w_max in w_values:
        for ke in ke_values:
            for humedad_mid in humedad_mid_values:
                for corte_seco in corte_seco_values:
                    if corte_seco >= humedad_mid:
                        continue

                    df_sim, fecha_inicio = simular(
                        df_meteo,
                        modelo_ann,
                        w_max=float(w_max),
                        ke_suelo=float(ke),
                        humedad_mid=float(humedad_mid),
                        corte_seco=float(corte_seco),
                        umbral_choque_hidrico=float(umbral_choque_hidrico),
                        umbral_termoinhibicion=float(umbral_termoinhibicion),
                        umbral_primer_pico=umbral_primer_pico,
                    )

                    fila = {
                        "Cobertura_%": COBERTURA_PCT,
                        "W_Max_mm": float(w_max),
                        "Ke_Suelo": float(ke),
                        "Humedad_mid": float(humedad_mid),
                        "Corte_seco": float(corte_seco),
                        "Mod_Termico": MOD_TERMICO_SUELO_DESNUDO,
                        "Umbral_Choque_3d_mm": float(umbral_choque_hidrico),
                        "Umbral_Termoinhibicion_C": float(umbral_termoinhibicion),
                        "Umbral_Primer_Pico": umbral_primer_pico if umbral_primer_pico is not None else "OFF",
                        "Fecha_Inicio_Filtro": fecha_inicio,
                        **score_dos_picos(df_sim, TARGET_PEAKS),
                    }

                    if df_campo is not None:
                        col_fecha, col_plm2 = columnas_campo(df_campo)
                        sync = sincronizar_intervalos_variables(df_sim, df_campo, col_fecha, col_plm2)
                        fila.update(metricas_ajuste_observado(sync, umbral_deteccion=umbral_deteccion))

                    resultados.append(fila)
                    done += 1
                    if done % 10 == 0 or done == total_iter:
                        progress.progress(
                            done / max(total_iter, 1),
                            text=f"Barrido edafico: {done}/{total_iter} combinaciones",
                        )

    progress.empty()
    ranking = pd.DataFrame(resultados)
    ranking = ordenar_ranking(ranking, criterio_ajuste if df_campo is not None else "Score_Dos_Picos")
    return ranking, df_meteo, df_campo


def simular_mejor(df_meteo: pd.DataFrame, best: pd.Series) -> pd.DataFrame:
    modelo_ann = load_ann(BASE)
    df_best, _ = simular(
        df_meteo,
        modelo_ann,
        w_max=float(best["W_Max_mm"]),
        ke_suelo=float(best["Ke_Suelo"]),
        humedad_mid=float(best["Humedad_mid"]),
        corte_seco=float(best["Corte_seco"]),
        umbral_choque_hidrico=float(best["Umbral_Choque_3d_mm"]),
        umbral_termoinhibicion=float(best["Umbral_Termoinhibicion_C"]),
        umbral_primer_pico=(None if best["Umbral_Primer_Pico"] == "OFF" else float(best["Umbral_Primer_Pico"])),
    )
    return df_best


def sincronizar_mejor(df_best: pd.DataFrame, df_campo: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df_campo is None:
        return pd.DataFrame()
    col_fecha, col_plm2 = columnas_campo(df_campo)
    return sincronizar_intervalos_variables(df_best, df_campo, col_fecha, col_plm2)


def fmt_metric(valor: float, decimales: int = 3) -> str:
    if pd.isna(valor):
        return "NA"
    return f"{valor:.{decimales}f}"


def grafico_emergencia(df_best: pd.DataFrame, df_campo: Optional[pd.DataFrame]) -> go.Figure:
    fig = go.Figure()

    sim_max = df_best["EMERREL"].max()
    y_sim = df_best["EMERREL"] / sim_max if sim_max > 0 else df_best["EMERREL"]
    fig.add_trace(
        go.Scatter(
            x=df_best["Fecha"],
            y=y_sim,
            mode="lines",
            name="Simulado PREDWEEM diario normalizado",
            line=dict(width=3),
            fill="tozeroy",
        )
    )

    if df_campo is not None:
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
                name="Campo validacion.xlsx normalizado",
                marker=dict(size=10, symbol="diamond"),
                line=dict(width=2, dash="dot"),
            )
        )

    fig.update_layout(
        title="Emergencia diaria simulada vs observaciones de campo",
        xaxis_title="Fecha",
        yaxis_title="Emergencia normalizada 0–1",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def grafico_ajuste_intervalos(df_sync: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df_sync.empty:
        return fig

    fig.add_trace(
        go.Scatter(
            x=df_sync["Fecha"],
            y=df_sync["Campo_Relativo"],
            mode="markers+lines",
            name="Observado campo por intervalo",
            marker=dict(size=10, symbol="diamond"),
            line=dict(width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_sync["Fecha"],
            y=df_sync["Sim_Relativo"],
            mode="markers+lines",
            name="Simulado agregado al intervalo",
            marker=dict(size=8),
            line=dict(width=3, dash="dot"),
        )
    )
    fig.update_layout(
        title="Ajuste directo: flujo observado vs flujo simulado agregado por intervalo de muestreo",
        xaxis_title="Fecha de muestreo",
        yaxis_title="Flujo relativo por intervalo",
        height=470,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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
            marker=dict(size=11),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, max_xy],
            y=[0, max_xy],
            mode="lines",
            name="Linea 1:1",
            line=dict(width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title="Linea 1:1 — observado vs simulado",
        xaxis_title="Observado campo relativo",
        yaxis_title="Simulado relativo",
        height=470,
        hovermode="closest",
    )
    return fig


def grafico_hidrico(df_best: pd.DataFrame, w_max: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df_best["Fecha"],
            y=df_best["Prec"],
            name="Precipitacion diaria",
            opacity=0.65,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_best["Fecha"],
            y=df_best["W_superficial"],
            mode="lines",
            name="Agua superficial simulada",
            line=dict(width=3),
        )
    )
    fig.add_hline(y=w_max, line_dash="dot", annotation_text=f"W_Max = {w_max:.1f} mm")
    fig.update_layout(
        title="Balance hidrico superficial calibrado",
        xaxis_title="Fecha",
        yaxis_title="mm",
        height=430,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ---------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------
st.title("🌾 PREDWEEM — Calibracion edafica contra observaciones de campo")
st.caption("Tres Arroyos 2026 · Cobertura de suelo fija en 0 % · Optimiza simulado vs validacion.xlsx")

with st.sidebar:
    st.header("📂 Datos")
    meteo_upload = st.file_uploader("Meteorologia", type=["csv", "xlsx", "xls"], help=f"Por defecto: {DEFAULT_METEO}")
    campo_upload = st.file_uploader("Validacion de campo", type=["xlsx", "xls", "csv"], help=f"Por defecto: {DEFAULT_CAMPO}")

    st.divider()
    st.header("🎯 Criterio de ajuste")
    criterio_ajuste = st.selectbox(
        "Parametro ganador segun",
        options=[
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
        help="Por defecto usa un indice compuesto campo: NSE alto, RMSE/MAE bajos, Pearson alto y F1 alto.",
    )
    umbral_deteccion = st.slider("Umbral evento para F1", 0.01, 0.30, 0.05, step=0.01)

    st.divider()
    st.header("🌱 Supuesto fijo")
    st.metric("Cobertura", "0 %")
    st.metric("Modulador termico", f"{MOD_TERMICO_SUELO_DESNUDO:.2f}")
    st.caption(f"Latitud ET0: {LAT_TRES_ARROYOS:.4f}")

    st.divider()
    st.header("💧 Barrido edafico")
    w_range = st.slider("Rango W_Max superficial (mm)", 6.0, 40.0, (8.0, 26.0), step=1.0)
    w_step = st.number_input("Paso W_Max", min_value=0.5, max_value=5.0, value=1.0, step=0.5)

    ke_range = st.slider("Rango Ke suelo desnudo", 0.40, 1.60, (0.75, 1.25), step=0.05)
    ke_step = st.number_input("Paso Ke", min_value=0.01, max_value=0.20, value=0.05, step=0.01)

    humedad_mid_values = st.multiselect(
        "Humedad_mid sigmoide",
        options=[0.20, 0.22, 0.24, 0.27, 0.30, 0.33, 0.36, 0.40],
        default=[0.24, 0.27, 0.30, 0.33, 0.36],
    )
    corte_seco_values = st.multiselect(
        "Corte seco HR",
        options=[0.10, 0.12, 0.15, 0.18, 0.20, 0.23, 0.25, 0.28, 0.30],
        default=[0.15, 0.18, 0.20, 0.23, 0.25],
    )

    st.divider()
    st.header("⚙️ Filtros fisiologicos")
    umbral_choque_hidrico = st.slider("Choque hidrico 3 dias (mm)", 10.0, 100.0, 45.0, step=1.0)
    umbral_termoinhibicion = st.number_input("Umbral termoinhibicion 5d (°C)", 15.0, 35.0, 24.0, step=0.5)
    usar_filtro_primer_pico = st.checkbox("Usar filtro de primer pico", value=True)
    umbral_primer_pico = st.number_input("Umbral primer pico", 0.001, 0.80, 0.05, step=0.005, format="%.3f")
    if not usar_filtro_primer_pico:
        umbral_primer_pico = None

    st.divider()
    ejecutar = st.button("🚀 Buscar parametros optimos", type="primary")

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.info("**Criterio principal:** simulado vs observado")
with col_b:
    st.info("**Simulado diario:** agregado a intervalos reales")
with col_c:
    st.info("**Cobertura:** 0 % suelo desnudo")

try:
    meteo_path = materializar_archivo(meteo_upload, DEFAULT_METEO)
    campo_path = materializar_archivo(campo_upload, DEFAULT_CAMPO)

    st.success(f"Meteorologia: `{meteo_path.name}` · Validacion: `{campo_path.name}`")

    if ejecutar:
        if not humedad_mid_values or not corte_seco_values:
            st.error("Debe seleccionar al menos un valor de Humedad_mid y Corte_seco.")
            st.stop()

        w_values = rango_float(w_range[0], w_range[1], w_step)
        ke_values = rango_float(ke_range[0], ke_range[1], ke_step)

        with st.spinner("Buscando parametros que maximizan el ajuste contra observaciones de campo..."):
            ranking, df_meteo, df_campo = ejecutar_barrido_streamlit(
                meteo_path=meteo_path,
                campo_path=campo_path,
                w_values=w_values,
                ke_values=ke_values,
                humedad_mid_values=humedad_mid_values,
                corte_seco_values=corte_seco_values,
                umbral_choque_hidrico=umbral_choque_hidrico,
                umbral_termoinhibicion=umbral_termoinhibicion,
                umbral_primer_pico=umbral_primer_pico,
                umbral_deteccion=umbral_deteccion,
                criterio_ajuste=criterio_ajuste,
            )

        if ranking.empty:
            st.error("El barrido no produjo resultados. Revisar rangos de parametros.")
            st.stop()

        best = ranking.iloc[0]
        df_best = simular_mejor(df_meteo, best)
        df_sync_best = sincronizar_mejor(df_best, df_campo)
        csv_bytes = ranking.to_csv(index=False).encode("utf-8")

        st.subheader("🏆 Parametros ganadores por ajuste a campo")
        st.caption(f"Criterio principal usado: `{criterio_ajuste}`")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("W_Max", f"{best['W_Max_mm']:.1f} mm")
        m2.metric("Ke", f"{best['Ke_Suelo']:.2f}")
        m3.metric("Humedad_mid", f"{best['Humedad_mid']:.2f}")
        m4.metric("Corte_seco", f"{best['Corte_seco']:.2f}")
        m5.metric("Ajuste campo", fmt_metric(best.get("Ajuste_Campo_Compuesto", np.nan)))

        v1, v2, v3, v4, v5 = st.columns(5)
        v1.metric("RMSE campo", fmt_metric(best.get("RMSE_Campo", np.nan)))
        v2.metric("MAE campo", fmt_metric(best.get("MAE_Campo", np.nan)))
        v3.metric("NSE campo", fmt_metric(best.get("NSE_Campo", np.nan)))
        v4.metric("Pearson campo", fmt_metric(best.get("Pearson_Campo", np.nan)))
        v5.metric("F1 campo", fmt_metric(best.get("F1_Campo", np.nan)))

        st.code(
            f"""
# Parametros sugeridos para app_emergencia.py
# Criterio ganador: {criterio_ajuste}
cobertura_pct = 0
w_max_val = {best['W_Max_mm']:.1f}
ke_val = {best['Ke_Suelo']:.2f}
mod_termico = {MOD_TERMICO_SUELO_DESNUDO:.2f}
humedad_mid = {best['Humedad_mid']:.2f}
corte_seco = {best['Corte_seco']:.2f}
umbral_choque_hidrico = {best['Umbral_Choque_3d_mm']:.1f}
umbral_termoinhibicion = {best['Umbral_Termoinhibicion_C']:.1f}
""".strip(),
            language="python",
        )

        st.plotly_chart(grafico_ajuste_intervalos(df_sync_best), width="stretch")
        st.plotly_chart(grafico_uno_a_uno(df_sync_best), width="stretch")
        st.plotly_chart(grafico_emergencia(df_best, df_campo), width="stretch")
        st.plotly_chart(grafico_hidrico(df_best, float(best["W_Max_mm"])), width="stretch")

        st.subheader("📊 Ranking de calibracion contra campo")
        columnas_preferidas = [
            "Ajuste_Campo_Compuesto",
            "RMSE_Campo",
            "MAE_Campo",
            "NSE_Campo",
            "Pearson_Campo",
            "R2_Campo",
            "F1_Campo",
            "Bias_Campo",
            "W_Max_mm",
            "Ke_Suelo",
            "Humedad_mid",
            "Corte_seco",
            "Score_Dos_Picos",
            "Pico1_Fecha",
            "Pico1_Valor",
            "Pico1_Lag_d",
            "Pico2_Fecha",
            "Pico2_Valor",
            "Pico2_Lag_d",
            "Relacion_P2_P1",
            "Penalidad_Valle",
            "Penalidad_Extra",
        ]
        columnas_presentes = [c for c in columnas_preferidas if c in ranking.columns]
        st.dataframe(ranking[columnas_presentes].head(50), width="stretch", height=420)
        st.download_button(
            "📥 Descargar ranking completo CSV",
            data=csv_bytes,
            file_name=DEFAULT_SALIDA,
            mime="text/csv",
        )

        with st.expander("Ver tabla sincronizada del mejor ajuste"):
            st.dataframe(df_sync_best, width="stretch", height=360)

        with st.expander("Ver simulacion diaria completa"):
            st.dataframe(df_best, width="stretch", height=420)

    else:
        st.warning("Ejecutar la busqueda desde el boton de la barra lateral.")
        st.markdown(
            """
**Uso operativo recomendado**

1. Mantener cobertura en **0 %**.
2. Ejecutar la busqueda con `validacion.xlsx`.
3. La app suma la emergencia simulada diaria dentro de cada intervalo real de muestreo.
4. El ranking se ordena por el criterio de ajuste elegido contra los datos observados.
5. Luego aplicar en `app_emergencia.py` los parametros ganadores.
"""
        )

except Exception as exc:
    st.error("No se pudo ejecutar la calibracion.")
    st.exception(exc)
