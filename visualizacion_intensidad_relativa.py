# -*- coding: utf-8 -*-
"""Parche Tres Arroyos: decaimiento desde 15-abr + visualización relativa 0–100 %.

La normalización porcentual es exclusivamente visual. El decaimiento sí forma
parte del modelo y modifica EMERREL antes de la acumulación y la validación.
"""

from __future__ import annotations

from modelo_decaimiento_15abril import parchear_modelo_decaimiento_15abril


def _reemplazar_unico(source: str, old: str, new: str, etiqueta: str) -> str:
    cantidad = source.count(old)
    if cantidad != 1:
        raise RuntimeError(
            f"Parche visual Tres Arroyos no aplicado: '{etiqueta}' aparece {cantidad} veces; "
            "se esperaba exactamente una coincidencia."
        )
    return source.replace(old, new, 1)


def _reemplazar_n(source: str, old: str, new: str, cantidad_esperada: int, etiqueta: str) -> str:
    cantidad = source.count(old)
    if cantidad != cantidad_esperada:
        raise RuntimeError(
            f"Parche visual Tres Arroyos no aplicado: '{etiqueta}' aparece {cantidad} veces; "
            f"se esperaban {cantidad_esperada} coincidencias."
        )
    return source.replace(old, new)


def parchear_visualizacion_intensidad_relativa(source: str) -> str:
    """Aplica el decaimiento del motor y expresa el gráfico principal en 0–100 %."""

    source = parchear_modelo_decaimiento_15abril(source)

    transformacion_old = '''    # Transformación Logarítmica Analítica
    c_log = 0.01
    df["EMERREL_LOG"] = np.log10(df["EMERREL"] + c_log)
    umbral_er_log = np.log10(umbral_er + c_log)
    if df_campo is not None:
        df_campo['Campo_Normalizado_LOG'] = np.log10(df_campo['Campo_Normalizado'] + c_log)'''

    transformacion_new = '''    # Escala visual relativa 0–100 %; EMERREL del motor no se renormaliza.
    max_emerrel_visual = float(df["EMERREL"].clip(lower=0.0).max())
    if max_emerrel_visual > 0.0:
        df["EMERREL_REL_PCT"] = (
            df["EMERREL"].clip(lower=0.0) / max_emerrel_visual * 100.0
        )
        umbral_er_pct = float(umbral_er) / max_emerrel_visual * 100.0
    else:
        df["EMERREL_REL_PCT"] = 0.0
        umbral_er_pct = 0.0
    if df_campo is not None:
        df_campo["Campo_Normalizado_PCT"] = (
            df_campo["Campo_Normalizado"].clip(lower=0.0) * 100.0
        )'''

    source = _reemplazar_unico(
        source,
        transformacion_old,
        transformacion_new,
        "transformación visual",
    )

    source = _reemplazar_unico(
        source,
        '                    y=df["EMERREL_LOG"],',
        '                    y=df["EMERREL_REL_PCT"],',
        "serie simulada",
    )
    source = _reemplazar_unico(
        source,
        '                    name="Tasa diaria simulada (log)",',
        '                    name="Intensidad relativa simulada (%)",',
        "nombre serie simulada",
    )
    source = _reemplazar_unico(
        source,
        '                        "Simulado: %{y:.3f}<extra></extra>"',
        '                        "Intensidad relativa: %{y:.1f}%<extra></extra>"',
        "hover simulado",
    )

    source = _reemplazar_unico(
        source,
        '                        y=df_campo["Campo_Normalizado_LOG"],',
        '                        y=df_campo["Campo_Normalizado_PCT"],',
        "serie campo",
    )
    source = _reemplazar_unico(
        source,
        '                        name="Campo normalizado (log)",',
        '                        name="Campo normalizado (%)",',
        "nombre serie campo",
    )
    source = _reemplazar_unico(
        source,
        '                            "Campo: %{y:.3f}<extra></extra>"',
        '                            "Campo: %{y:.1f}%<extra></extra>"',
        "hover campo",
    )

    source = _reemplazar_n(
        source,
        "umbral_er_log",
        "umbral_er_pct",
        2,
        "umbral gráfico",
    )

    source = _reemplazar_unico(
        source,
        '                        text="Log10(EMERREL + 0,01)",',
        '                        text="Intensidad relativa de emergencia (%)",',
        "título eje Y",
    )
    source = _reemplazar_unico(
        source,
        "                    range=[-2.18, 0.12],",
        "                    range=[0.0, 105.0],",
        "rango eje Y",
    )
    source = _reemplazar_unico(
        source,
        "                    tickvals=[-2.0, -1.5, -1.0, -0.5, 0.0],",
        "                    tickvals=[0, 20, 40, 60, 80, 100],",
        "ticks eje Y",
    )

    return source
