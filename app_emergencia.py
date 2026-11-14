# -*- coding: utf-8 -*-
"""
PREDWEEM Lolium — Tres Arroyos.

La página muestra la alerta de grandes flujos de Lolium del paquete ``lolium-comun`` (probabilidad de gran flujo en
los próximos 7 días, qué hacer esta semana, ventana del graminicida de barbecho, residual a la siembra y flujos tardíos)
con el clima diario del repositorio (``meteo_daily.csv``).

El motor PREDWEEM anterior (red neuronal y balance hídrico, ``app_emergencia_core.py`` y sus modelos) ya no se
ejecuta: su inicio estimado de la emergencia erró 12–14 días en promedio y no mejoraba la alerta. Sus archivos quedan
en el repositorio solo como referencia.
"""
import importlib
from pathlib import Path

import pandas as pd
import streamlit as st

# Control de acceso (ingreso con Google y usuarios aprobados). Inactivo hasta configurar los secretos [acceso] y [auth].
from acceso_predweem import mostrar_sesion, requerir_acceso
from lolium_comun import pagina

BASE = Path(__file__).resolve().parent

requerir_acceso("Tres Arroyos")
pagina.encabezado("🌾 PREDWEEM LOLIUM — TRES ARROYOS (BA) lat=-38.4500 lon=-60.2763", "PREDWEEM TRES ARROYOS")
parametros = pagina.barra_lateral(BASE, "Tres Arroyos")
meteo = pagina.cargar_meteo(BASE, None)

NOTA_TRES_ARROYOS = (
    "**Validación en Tres Arroyos.** El modelo se probó, sin reajustarlo, con los conteos de "
    "Tres Arroyos 2023 (29 visitas semanales a quincenales), 2024 (10 conteos cada 2–4 semanas) y 2026 (17 conteos), con "
    "el clima de cada campaña: 53 intervalos entre conteos, 9 con emergencia ≥ 5 % de la campaña por semana. Emitida al "
    "comienzo de cada intervalo, la alerta separó bien los intervalos con gran flujo: AUC 0,93 solo con clima y 0,94 con la "
    "fecha de primeras plántulas. Los 9 grandes flujos quedaron en *Alerta*, que se emitió en el 40 % de los intervalos. "
    "Por campaña: 2023 AUC 0,97; 2024 0,95; 2026 0,82.\n\n"
    "**Límites en Tres Arroyos.** Tres campañas y 9 grandes flujos. En 2023 y 2024 el primer conteo ya tenía plántulas (no hay "
    "cero inicial), y gran parte de los conteos de 2024 y 2026 son cada 2–4 semanas: la alerta con conteos está calibrada "
    "con conteos semanales, por eso conviene usar sobre todo la fecha de primeras plántulas.\n\n"
    "**Curva acumulada.** El panel inferior usa el calendario propio de Tres Arroyos (conteos de `data/campanias/`), "
    "sin la campaña en curso. Prediciendo cada campaña con las otras dos, el error medio de la curva en las fechas de "
    "conteo fue 2,2 / 4,4 / 8,9 puntos (2023 / 2024 / 2026), contra 7,6 / 6,9 / 13,3 con el calendario de referencia regional. "
    "Recalibrar la probabilidad de gran flujo con estas campañas no la mejoró, y usar el calendario local dentro de la alerta "
    "la empeoró levemente (AUC 0,92 contra 0,94): por eso la alerta sigue con el calendario de referencia regional. Con tres "
    "campañas, la banda del panel es el mínimo y el máximo observados; se afirma a medida que se suman campañas."
)

from calendario_local import construir_calendario

# calendario propio del sitio, sin la campaña en curso (su total todavía no se conoce)
_anio_campania = int(pd.to_datetime(meteo["Fecha"], errors="coerce").max().year) if meteo is not None else None
_calendario_local = construir_calendario(excluir=(_anio_campania,)) if _anio_campania else None

_resultados_alerta = None
if meteo is not None and not meteo.empty:
    try:
        import alerta_lolmu as _alerta_lolmu
        import visualizacion_alerta_lolmu as _vis_alerta

        # Streamlit conserva en memoria los módulos ya importados entre ejecuciones: tras una actualización del
        # repositorio, recargarlos evita usar una versión vieja
        importlib.reload(_alerta_lolmu)
        importlib.reload(_vis_alerta)
        _resultados_alerta = _vis_alerta.mostrar_alerta_lolmu(
            meteo,
            **parametros,
    tope_hasta_ultimo_cero=True,  # semanas ya cerradas por un conteo posterior en cero: probabilidad 0
    curva_desde_ultimo_cero=True,  # la curva acumulada arranca a mitad de camino entre el último cero y el primer conteo con plántulas
    curva_desde_primeras=True,  # con primeras plántulas, la curva del calendario propio arranca ese día
    localidad="Tres Arroyos",
    nota_sitio=NOTA_TRES_ARROYOS,
    calendario_local=_calendario_local,
        )
    except Exception as _error_alerta:  # noqa: BLE001
        st.warning(f"No se pudo calcular la alerta: {_error_alerta}")

pagina.descarga_excel(_resultados_alerta, "TresArroyos", "descarga_excel_resultados_final_tres_arroyos")
mostrar_sesion()
