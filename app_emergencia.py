# -*- coding: utf-8 -*-
"""
Punto de entrada de PREDWEEM Tres Arroyos.

La aplicación científica original se conserva en ``app_emergencia_core.py``. Antes de ejecutarla:

* ``parchear_visualizacion_intensidad_relativa`` incorpora al motor el decaimiento desde el 15 de abril
  (y ajustes de formato de su gráfico);
* se retiran de la página los elementos que reemplaza la alerta de flushes del final: mapa de riesgo,
  gráfico de dinámica principal, mensajes de inicio y momento de control del motor, la leyenda de versión,
  las pestañas (Monitor de decisión, Precipitaciones y suelo, Análisis estratégico y Bio-calibración), la
  sección "Datos del lote" (se usa el clima SIGA del repositorio y la cobertura de rastrojo por defecto,
  20 %), la imagen de fondo (reemplazada por un fondo liso claro) y, en la barra lateral, los parámetros del
  motor (quedan en sus valores por defecto), sus avisos, el calibrador 2D y el botón del reporte del motor;
  la barra lateral conserva los parámetros de la ventana de control.

El motor sigue calculando como antes: su primer pico se usa como inicio estimado del reloj térmico hasta
que se informa la fecha de las primeras plántulas. La alerta (``alerta_lolmu.py``) es la calibrada con
17 campañas de Bordenave y validada con las campañas 2023, 2024 y 2026 de Tres Arroyos.
"""
from pathlib import Path

from visualizacion_horizonte_pronostico import mostrar_horizonte_pronostico
from visualizacion_intensidad_relativa import parchear_visualizacion_intensidad_relativa

_CORE_APP = Path(__file__).with_name("app_emergencia_core.py")
_core_source = _CORE_APP.read_text(encoding="utf-8")
_core_source = parchear_visualizacion_intensidad_relativa(_core_source)


def _reemplazar(origen: str, viejo: str, nuevo: str) -> str:
    if viejo not in origen:
        raise RuntimeError(f"No se encontró el bloque a reemplazar en el motor: {viejo[:70]!r}")
    return origen.replace(viejo, nuevo, 1)


def _cortar(origen: str, desde: str, hasta: str, nuevo: str) -> str:
    """Reemplaza el tramo [desde, hasta) del motor."""
    ini = origen.find(desde)
    fin = origen.find(hasta, ini) if ini >= 0 else -1
    if ini < 0 or fin < ini:
        raise RuntimeError(f"No se encontró el tramo a retirar del motor: {desde[:70]!r}")
    return origen[:ini] + nuevo + origen[fin:]


_FONDO_CSS = """<style>
.stApp { background: #f4f6f3; }
[data-testid="stAppViewContainer"] > .main, [data-testid="stMain"] { background: transparent; }
[data-testid="stSidebar"] { background-color: #e8f3ea; border-right: 1px solid #cfe3d3; }
[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color: #14532d !important; }
[data-testid="stCaptionContainer"], .stCaption { color: #4b5563 !important; }
h1, h2, h3, h4 { color: #1f2937; }
</style>"""

# gráficos del motor que reemplaza la alerta (el cálculo no cambia)
for _llamada in ("st.plotly_chart(go.Figure(data=go.Heatmap(", "st.plotly_chart(\n                fig_emer,"):
    _core_source = _reemplazar(_core_source, _llamada, _llamada.replace("st.plotly_chart(", "_omitir_grafico("))

# la regla "header {visibility: hidden;}" ocultaba también el encabezado de los calendarios desplegables
_core_source = _reemplazar(_core_source, "    header {visibility: hidden;}", '    header[data-testid="stHeader"] {visibility: hidden;}')
# fondo liso claro en lugar de la imagen de plantas (mejor contraste)
_core_source = _reemplazar(_core_source, 'set_bg_hack("fondo_predweem_v3.png")', "st.markdown(_FONDO_CSS, unsafe_allow_html=True)")
# leyenda de versión del motor
_core_source = _reemplazar(
    _core_source,
    'st.caption("Tres Arroyos · VISUAL V3 · ventana eficiente verde · exportación PNG de alta resolución")',
    "pass",
)

# sección "Datos del lote": clima SIGA del repositorio, sin conteos del formato anterior, cobertura por defecto (20 %)
_core_source = _cortar(
    _core_source,
    'with st.expander("📂 1. Datos del Lote", expanded=True):',
    'df_meteo_raw = load_data(archivo_meteo, "meteo_daily")',
    "archivo_meteo = None\narchivo_campo = None\ncobertura_pct = 20\nke_val, mod_termico = parametros_superficie(cobertura_pct)\n\n",
)
_core_source = _reemplazar(_core_source, 'df_campo_raw = load_data(archivo_campo, "tres_arroyos_campo")', "df_campo_raw = None")

# barra lateral: solo la ventana de control; los parámetros del motor quedan en sus valores por defecto
_core_source = _cortar(
    _core_source,
    'st.sidebar.markdown("## ⚙️ 2. Fisiología y Logística")',
    "# ---------------------------------------------------------\n# 6. MOTOR DE CÁLCULO",
    '''st.sidebar.markdown("## ⚙️ Ventana de control")
st.sidebar.caption("Tiempo térmico sobre la temperatura media del aire y objetivos de la ventana post-emergente.")
col_t1, col_t2 = st.sidebar.columns(2)
with col_t1: t_base_val = st.number_input("T Base", value=2.0, step=0.5)
with col_t2: t_opt_max = st.number_input("T Óptima Max", value=20.0, step=1.0)
t_critica = st.sidebar.slider("T Crítica (Stop)", 26.0, 42.0, 30.0)
st.sidebar.markdown("**Objetivos (°Cd)**")
dga_optimo = st.sidebar.number_input("TT Control Post-emergente (°Cd)", value=600, step=10)
dga_critico = st.sidebar.number_input("Límite Ventana (°Cd)", value=800, step=10)

# parámetros del motor PREDWEEM en sus valores por defecto (controles retirados de la página)
umbral_er = 0.0001
umbral_termoinhibicion = 24.0
umbral_choque_hidrico = 45.0
residualidad = 0
w_max_val = 18.81
exponente_kr = 0.0

''',
)

# botón lateral del reporte del motor: la descarga final reúne los resultados de la alerta
_core_source = _reemplazar(_core_source, 'st.sidebar.download_button("📥 Descargar Reporte Tres Arroyos"', '_omitir_grafico("📥 Descargar Reporte Tres Arroyos"')

# mensajes de inicio de conteo térmico y momento crítico de control del motor
_core_source = _cortar(_core_source, "            if fecha_inicio_ventana:\n                st.success(", "        col_gauge = col_gauge", "            pass\n\n")

# pestañas del motor (Monitor de decisión, Precipitaciones y suelo, Análisis estratégico, Bio-calibración)
_core_source = _cortar(_core_source, "    tab1, tab2, tab3, tab4 = st.tabs(", "    output = io.BytesIO()", "")


def _omitir_grafico(*_args, **_kwargs):
    """No dibuja el elemento (se conservan la figura y todos los cálculos del motor)."""
    return None


exec(
    compile(_core_source, str(_CORE_APP), "exec"),
    globals(),
)

if "df" in globals() and isinstance(df, pd.DataFrame) and not df.empty:
    st.divider()
    mostrar_horizonte_pronostico(df, "Tres Arroyos")

NOTA_TRES_ARROYOS = (
    "**Validación en Tres Arroyos.** El modelo se calibró en Bordenave y se probó, sin reajustarlo, con los conteos de "
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
    "conteo fue 2,2 / 4,4 / 8,9 puntos (2023 / 2024 / 2026), contra 7,6 / 6,9 / 13,3 con el calendario de Bordenave. "
    "Recalibrar la probabilidad de gran flujo con estas campañas no la mejoró, y usar el calendario local dentro de la alerta "
    "la empeoró levemente (AUC 0,92 contra 0,94): por eso la alerta sigue con el calendario de Bordenave. Con tres "
    "campañas, la banda del panel es el mínimo y el máximo observados; se afirma a medida que se suman campañas."
)

# Alerta de flushes (independiente del motor científico; no modifica sus resultados)
if "df_meteo_raw" in globals() and isinstance(df_meteo_raw, pd.DataFrame) and not df_meteo_raw.empty:
    st.divider()
    try:
        from calendario_local import construir_calendario
        import importlib

        import alerta_lolmu as _alerta_lolmu
        import visualizacion_alerta_lolmu as _vis_alerta

        # Streamlit conserva en memoria los módulos ya importados entre ejecuciones: tras una actualización del
        # repositorio, recargarlos evita usar una versión vieja
        importlib.reload(_alerta_lolmu)
        importlib.reload(_vis_alerta)
        mostrar_alerta_lolmu = _vis_alerta.mostrar_alerta_lolmu

        # calendario propio de Tres Arroyos, sin la campaña en curso (su total todavía no se conoce)
        _anio_campania = int(pd.to_datetime(df_meteo_raw["Fecha"], errors="coerce").max().year)
        _calendario_local = construir_calendario(excluir=(_anio_campania,))

        _idx_pico = globals().get("idx_primer_pico")
        _inicio_predicho = (
            df.loc[_idx_pico, "Fecha"]
            if _idx_pico is not None and "df" in globals() and isinstance(df, pd.DataFrame) and _idx_pico in df.index
            else globals().get("fecha_inicio_ventana")
        )
        _resultados_alerta = mostrar_alerta_lolmu(
            df_meteo_raw,
            inicio_predicho=_inicio_predicho,
            t_base=float(globals().get("t_base_val", 2.0)),
            t_opt=float(globals().get("t_opt_max", 20.0)),
            t_crit=float(globals().get("t_critica", 30.0)),
            tt_control=float(globals().get("dga_optimo", 600.0)),
            tt_limite=float(globals().get("dga_critico", 800.0)),
            tope_hasta_ultimo_cero=True,  # semanas ya cerradas por un conteo posterior en cero: probabilidad 0
            curva_desde_ultimo_cero=True,  # la curva acumulada arranca a mitad de camino entre el último cero y el primer conteo con plántulas
            localidad="Tres Arroyos",
            nota_sitio=NOTA_TRES_ARROYOS,
            calendario_local=_calendario_local,
        )
    except Exception as _error_alerta:  # noqa: BLE001
        st.warning(f"No se pudo calcular la alerta: {_error_alerta}")


def _escribir_hoja(writer, dataframe, nombre):
    """Escribe una hoja y aplica un formato básico de lectura."""
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        return
    dataframe.to_excel(writer, sheet_name=nombre, index=False)
    hoja = writer.sheets[nombre]
    ultima_columna = max(0, len(dataframe.columns) - 1)
    hoja.freeze_panes(1, 0)
    hoja.autofilter(0, 0, len(dataframe), ultima_columna)
    hoja.set_column(0, ultima_columna, 18)


# Descarga final: los resultados que muestra la app (alerta, emergencia acumulada y ventana de control)
_resultados_alerta = globals().get("_resultados_alerta")
if isinstance(_resultados_alerta, dict) and _resultados_alerta:
    reporte_excel_final = io.BytesIO()
    with pd.ExcelWriter(reporte_excel_final, engine="xlsxwriter", datetime_format="dd/mm/yyyy", date_format="dd/mm/yyyy") as writer:
        for _hoja, _tabla in _resultados_alerta.items():
            _escribir_hoja(writer, _tabla, _hoja)
    reporte_excel_final.seek(0)
    _emision = _resultados_alerta["Resumen"].set_index("Indicador")["Valor"].get("Fecha de emisión", "")

    st.divider()
    st.subheader("📥 Descarga de resultados")
    st.caption(
        "Excel con los resultados que muestra la app para la fecha de emisión elegida: resumen (nivel y probabilidad de "
        "gran flujo, inicio y ventana de control 600–800 °Cd, emergencia acumulada esperada), alerta semanal, emergencia "
        "acumulada con su rango, tiempo térmico diario desde el inicio, conteos usados, clima, parámetros y validación."
    )
    st.download_button(
        label="📊 Descargar resultados en Excel",
        data=reporte_excel_final.getvalue(),
        file_name=f"PREDWEEM_Lolium_TresArroyos_alerta_{str(_emision).replace('/', '-') or 'resultados'}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
        key="descarga_excel_resultados_final_tres_arroyos",
    )
