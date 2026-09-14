# -*- coding: utf-8 -*-
"""Decaimiento tardío de PREDWEEM Tres Arroyos desde el 15 de abril.

Replica el criterio operativo aplicado en Bordenave y Lartigau:
- hasta el 14 de abril inclusive, EMERREL queda sin cambios;
- desde el 15 de abril, los pulsos quedan limitados inicialmente al 50 %
  del máximo simulado previo al 15 de abril;
- ese techo disminuye después con una función Weibull/exponencial.

Formulación
-----------
Sea M_pre el máximo EMERREL previo al 15-abr y t los días desde 15-abr:

    techo_0 = 0.50 * M_pre
    F(t) = (1-I) + I*exp(-(t/tau)**beta)
    techo(t) = techo_0 * F(t)
    EMERREL_final(t) = min(EMERREL_original(t), techo(t))

Parámetros iniciales:
    tau = 60 días
    beta = 1.0
    intensidad = 0.75
    fracción máxima inicial = 0.50

La modificación se aplica antes de EMERAC, Event-to-Event, T50 y las métricas.
No altera los pesos ni las entradas de la ANN.
"""

from __future__ import annotations

import numpy as np


TAU_DECAIMIENTO_15ABR_DEFAULT = 60.0
BETA_DECAIMIENTO_15ABR_DEFAULT = 1.0
INTENSIDAD_DECAIMIENTO_15ABR_DEFAULT = 0.75
FRACCION_MAX_15ABR = 0.50


def factor_decaimiento_15abril(
    dias_desde_inicio,
    activo,
    tau: float = TAU_DECAIMIENTO_15ABR_DEFAULT,
    beta: float = BETA_DECAIMIENTO_15ABR_DEFAULT,
    intensidad: float = INTENSIDAD_DECAIMIENTO_15ABR_DEFAULT,
):
    """Factor 1 antes del 15-abr y Weibull desde esa fecha."""
    dias = np.maximum(np.asarray(dias_desde_inicio, dtype=float), 0.0)
    activo_arr = np.asarray(activo, dtype=bool)
    tau = max(float(tau), 1e-9)
    beta = max(float(beta), 1e-9)
    intensidad = float(np.clip(intensidad, 0.0, 1.0))

    factor = np.ones_like(dias, dtype=float)
    factor[activo_arr] = (
        (1.0 - intensidad)
        + intensidad * np.exp(-((dias[activo_arr] / tau) ** beta))
    )
    return np.clip(factor, 0.0, 1.0)


def _reemplazar_unico(source: str, old: str, new: str, etiqueta: str) -> str:
    cantidad = source.count(old)
    if cantidad != 1:
        raise RuntimeError(
            f"Parche de decaimiento Tres Arroyos no aplicado: '{etiqueta}' aparece "
            f"{cantidad} veces; se esperaba exactamente una coincidencia."
        )
    return source.replace(old, new, 1)


def parchear_modelo_decaimiento_15abril(source: str) -> str:
    """Inserta techo 50 % + decaimiento desde 15-abr dentro del motor."""

    motor_old = '''    df, idx_primer_pico = aplicar_filtro_primer_pico(
        df,
        umbral=UMBRAL_PRIMER_PICO,
    )
    df["EMERAC"] = df["EMERREL"].cumsum()'''

    motor_new = '''    df, idx_primer_pico = aplicar_filtro_primer_pico(
        df,
        umbral=UMBRAL_PRIMER_PICO,
    )

    # Decaimiento tardío Tres Arroyos: hasta 14-abr EMERREL queda intacto.
    df["EMERREL_ANTES_DECAIMIENTO_15ABR"] = df["EMERREL"].copy()
    df["Dias_Desde_15Abr"] = 0.0
    df["Factor_Decaimiento_15Abr"] = 1.0
    df["Techo_EMERREL_15Abr"] = np.nan

    tau_d = 60.0
    beta_d = 1.0
    intensidad_d = 0.75
    fraccion_max_d = 0.50

    inicio_decaimiento = pd.to_datetime({
        "year": df["Fecha"].dt.year,
        "month": np.full(len(df), 4),
        "day": np.full(len(df), 15),
    })
    mascara_decay = df["Fecha"] >= inicio_decaimiento
    dias_decay = (df["Fecha"] - inicio_decaimiento).dt.days.clip(lower=0).astype(float)

    max_pre_por_anio = {}
    for anio in sorted(df["Fecha"].dt.year.dropna().unique()):
        fecha_inicio_anio = pd.Timestamp(year=int(anio), month=4, day=15)
        mascara_pre = (df["Fecha"].dt.year == anio) & (df["Fecha"] < fecha_inicio_anio)
        max_pre = (
            float(df.loc[mascara_pre, "EMERREL"].clip(lower=0.0).max())
            if mascara_pre.any()
            else 0.0
        )
        max_pre_por_anio[int(anio)] = max_pre

    factor_decay = np.ones(len(df), dtype=float)
    factor_decay[mascara_decay.to_numpy()] = (
        (1.0 - intensidad_d)
        + intensidad_d * np.exp(
            -((dias_decay[mascara_decay].to_numpy() / tau_d) ** beta_d)
        )
    )

    techo_decay = np.full(len(df), np.nan, dtype=float)
    for pos, (_, fila) in enumerate(df.iterrows()):
        if bool(mascara_decay.iloc[pos]):
            max_pre = max_pre_por_anio.get(int(fila["Fecha"].year), 0.0)
            if max_pre > 0.0:
                techo_decay[pos] = fraccion_max_d * max_pre * factor_decay[pos]

    df["Dias_Desde_15Abr"] = dias_decay
    df["Factor_Decaimiento_15Abr"] = np.clip(factor_decay, 0.0, 1.0)
    df["Techo_EMERREL_15Abr"] = techo_decay

    mascara_con_techo = mascara_decay & df["Techo_EMERREL_15Abr"].notna()
    if mascara_con_techo.any():
        valores_originales = df.loc[mascara_con_techo, "EMERREL"].clip(lower=0.0).to_numpy()
        techos_activos = df.loc[mascara_con_techo, "Techo_EMERREL_15Abr"].to_numpy()
        df.loc[mascara_con_techo, "EMERREL"] = np.minimum(
            valores_originales,
            techos_activos,
        )

    df["Tau_Decaimiento_15Abr_d"] = tau_d
    df["Beta_Decaimiento_15Abr"] = beta_d
    df["Intensidad_Decaimiento_15Abr"] = intensidad_d
    df["Fraccion_Maxima_15Abr"] = fraccion_max_d

    df["EMERAC"] = df["EMERREL"].cumsum()'''

    return _reemplazar_unico(source, motor_old, motor_new, "motor desde 15-abr")
