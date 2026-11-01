# -*- coding: utf-8 -*-
"""Calendario propio de emergencia acumulada de Tres Arroyos.

Se construye con los conteos de ``data/campanias/<año>/emergencia_tres_arroyos_<año>.csv`` (columnas ``Fecha`` y
``PLM2``, plántulas nuevas por conteo). Cada campaña se expresa como fracción acumulada de su total: un cero el
14 de enero y luego el acumulado en cada fecha de conteo, interpolado por día del año. El calendario es la curva
media y el rango mínimo–máximo de las campañas.

Para sumar una campaña, agregue su carpeta con el CSV de conteos. La campaña en curso (el año del clima que muestra
la app) se excluye, porque su total todavía no se conoce.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DIRECTORIO = Path(__file__).with_name("data") / "campanias"
DIAS = np.arange(1, 367)


def curva_campania(archivo: Path) -> tuple[int, np.ndarray]:
    c = pd.read_csv(archivo)
    c["Fecha"] = pd.to_datetime(c["Fecha"], errors="coerce")
    c["PLM2"] = pd.to_numeric(c["PLM2"], errors="coerce")
    c = c.dropna(subset=["Fecha", "PLM2"]).sort_values("Fecha")
    if c.empty or c["PLM2"].sum() <= 0:
        raise ValueError(f"{archivo.name}: sin conteos válidos")
    anio = int(c["Fecha"].dt.year.iloc[0])
    acum = pd.Series(c["PLM2"].cumsum().to_numpy() / c["PLM2"].sum(), index=c["Fecha"])
    acum = pd.concat([pd.Series([0.0], index=[pd.Timestamp(anio, 1, 14)]), acum])
    acum = acum[~acum.index.duplicated(keep="last")]
    return anio, np.interp(DIAS, acum.index.dayofyear, acum.to_numpy())


def construir_calendario(excluir: tuple[int, ...] = (), directorio: Path = DIRECTORIO) -> dict | None:
    curvas = {}
    for archivo in sorted(directorio.glob("*/emergencia_*.csv")):
        try:
            anio, curva = curva_campania(archivo)
        except (ValueError, KeyError):
            continue
        if anio not in excluir:
            curvas[anio] = curva
    if len(curvas) < 2:
        return None
    m = np.array([curvas[a] for a in sorted(curvas)])
    return {"dia": DIAS.tolist(), "media": m.mean(0).tolist(), "bajo": m.min(0).tolist(), "alto": m.max(0).tolist(),
            "campanias": sorted(curvas), "rango": "mínimo–máximo"}
