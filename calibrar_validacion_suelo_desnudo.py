# -*- coding: utf-8 -*-
"""
Miniapp Streamlit para calibrar PREDWEEM Tres Arroyos con datos reales
`validacion.xlsx`, meteorologia `meteo_daily.csv` y cobertura de suelo = 0 %.

Ejecucion local:

    streamlit run calibrar_validacion_suelo_desnudo.py

Objetivo agronomico:
    Ajustar parametros edaficos superficiales para captar dos picos de campo:
    - Pico 1 cercano al 24/05/2026
    - Pico 2 cercano al 28/06/2026

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


st.set_page_config(
    page_title="PREDWEEM Calibracion Suelo Desnudo",
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
                        fila.update(metricas_evento(sync))

                    resultados.append(fila)
                    done += 1
                    if done % 10 == 0 or done == total_iter:
                        progress.progress(
                            done / max(total_iter, 1),
                            text=f"Barrido edafico: {done}/{total_iter} combinaciones",
                        )

    progress.empty()
    ranking = pd.DataFrame(resultados)

    if df_campo is not None:
        ranking = ranking.sort_values(
            ["F1_Campo", "NSE_Campo", "Score_Dos_Picos"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    else:
        ranking = ranking.sort_values("Score_Dos_Picos", ascending=False).reset_index(drop=True)

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


def grafico_emergencia(df_best: pd.DataFrame, df_campo: Optional[pd.DataFrame]) -> go.Figure:
    fig = go.Figure()

    sim_max = df_best["EMERREL"].max()
    y_sim = df_best["EMERREL"] / sim_max if sim_max > 0 else df_best["EMERREL"]
    fig.add_trace(
        go.Scatter(
            x=df_best["Fecha"],
            y=y_sim,
            mode="lines",
            name="Simulado PREDWEEM normalizado",
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

    for fecha, etiqueta in zip(TARGET_PEAKS, ["Pico campo 24/05", "Pico campo 28/06"]):
        fig.add_vline(
            x=fecha,
            line_dash="dash",
            annotation_text=etiqueta,
            annotation_position="top",
        )

    fig.update_layout(
        title="Emergencia simulada vs validacion de campo — suelo desnudo 0 %",
        xaxis_title="Fecha",
        yaxis_title="Emergencia normalizada 0–1",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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
st.title("🌾 PREDWEEM — Calibracion edafica con validacion.xlsx")
st.caption("Tres Arroyos 2026 · Cobertura de suelo fija en 0 % · Busqueda de dos picos: 24/05 y 28/06")

with st.sidebar:
    st.header("📂 Datos")
    meteo_upload = st.file_uploader("Meteorologia", type=["csv", "xlsx", "xls"], help=f"Por defecto: {DEFAULT_METEO}")
    campo_upload = st.file_uploader("Validacion de campo", type=["xlsx", "xls", "csv"], help=f"Por defecto: {DEFAULT_CAMPO}")

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
    ejecutar = st.button("🚀 Ejecutar calibracion", type="primary")

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.info(f"**Pico objetivo 1:** {TARGET_PEAKS[0].strftime('%d/%m/%Y')}")
with col_b:
    st.info(f"**Pico objetivo 2:** {TARGET_PEAKS[1].strftime('%d/%m/%Y')}")
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

        with st.spinner("Ejecutando calibracion edafica contra validacion.xlsx..."):
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
            )

        if ranking.empty:
            st.error("El barrido no produjo resultados. Revisar rangos de parametros.")
            st.stop()

        best = ranking.iloc[0]
        df_best = simular_mejor(df_meteo, best)
        csv_bytes = ranking.to_csv(index=False).encode("utf-8")

        st.subheader("🏆 Mejor combinacion edafica")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("W_Max", f"{best['W_Max_mm']:.1f} mm")
        m2.metric("Ke", f"{best['Ke_Suelo']:.2f}")
        m3.metric("Humedad_mid", f"{best['Humedad_mid']:.2f}")
        m4.metric("Corte_seco", f"{best['Corte_seco']:.2f}")
        m5.metric("Score 2 picos", f"{best['Score_Dos_Picos']:.3f}")

        if "F1_Campo" in ranking.columns:
            v1, v2, v3 = st.columns(3)
            v1.metric("F1 campo", f"{best['F1_Campo']:.3f}")
            v2.metric("NSE campo", f"{best['NSE_Campo']:.3f}")
            v3.metric("Pearson campo", f"{best['Pearson_Campo']:.3f}")

        st.code(
            f"""
# Parametros sugeridos para app_emergencia.py
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

        st.plotly_chart(grafico_emergencia(df_best, df_campo), width="stretch")
        st.plotly_chart(grafico_hidrico(df_best, float(best["W_Max_mm"])), width="stretch")

        st.subheader("📊 Ranking de calibracion")
        columnas_preferidas = [
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
        if "F1_Campo" in ranking.columns:
            columnas_preferidas = ["F1_Campo", "NSE_Campo", "Pearson_Campo"] + columnas_preferidas

        st.dataframe(ranking[columnas_preferidas].head(50), width="stretch", height=420)
        st.download_button(
            "📥 Descargar ranking completo CSV",
            data=csv_bytes,
            file_name=DEFAULT_SALIDA,
            mime="text/csv",
        )

        with st.expander("Ver simulacion diaria completa"):
            st.dataframe(df_best, width="stretch", height=420)

    else:
        st.warning("Ejecutar la calibracion desde el boton de la barra lateral.")
        st.markdown(
            """
**Uso operativo recomendado**

1. Mantener cobertura en **0 %**.
2. Ejecutar el barrido con `validacion.xlsx`.
3. Tomar la mejor combinacion de `W_Max`, `Ke`, `Humedad_mid` y `Corte_seco`.
4. Luego aplicar esos valores en `app_emergencia.py`.
"""
        )

except Exception as exc:
    st.error("No se pudo ejecutar la calibracion.")
    st.exception(exc)
