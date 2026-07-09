# -*- coding: utf-8 -*-
"""
🌾 PREDWEEM LOLIUM — TRES ARROYOS 2026
APP OPERATIVA CALIBRADA CONTRA validacion.xlsx

Esta variante ejecuta `app_emergencia.py` inyectando los parámetros ganadores
obtenidos con `app_optimizar_cobertura.py`.

Parámetros calibrados:
    cobertura_pct = 0
    ke_val = 1.60
    mod_termico = 0.90
    w_max_val = 26.0
    humedad_mid = 0.36
    corte_seco = 0.25
    umbral_choque_hidrico = 45.0
    umbral_termoinhibicion = 24.0

Ejecución:
    streamlit run app_emergencia_calibrada.py

Nota agronómica:
    Ke=1.60 quedó en el límite superior del último barrido. Conviene repetir
    `app_optimizar_cobertura.py` con un máximo de Ke mayor, por ejemplo 2.00,
    para confirmar si el óptimo real no queda por encima del rango evaluado.
"""

from __future__ import annotations

from pathlib import Path


BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
APP_ORIGINAL = BASE / "app_emergencia.py"


PARAMETROS_CALIBRADOS = {
    "cobertura_pct": 0,
    "ke_val": 1.60,
    "mod_termico": 0.90,
    "w_max_val": 26.0,
    "humedad_mid": 0.36,
    "corte_seco": 0.25,
    "umbral_choque_hidrico": 45.0,
    "umbral_termoinhibicion": 24.0,
}


if not APP_ORIGINAL.exists():
    raise FileNotFoundError(f"No se encontró la app original: {APP_ORIGINAL}")

source = APP_ORIGINAL.read_text(encoding="utf-8")

# ---------------------------------------------------------------------
# 1) Identificación visual y carga de validación
# ---------------------------------------------------------------------
source = source.replace(
    'st.title("🌾 PREDWEEM LOLIUM — TRES ARROYOS (BA) lat=-38.4500 lon=-60.2763")',
    'st.title("🌾 PREDWEEM LOLIUM — TRES ARROYOS CALIBRADO CAMPO (BA) lat=-38.4500 lon=-60.2763")',
)

source = source.replace(
    'df_campo_raw = load_data(archivo_campo, "tres_arroyos_campo")',
    'df_campo_raw = load_data(archivo_campo, "validacion")',
)

# ---------------------------------------------------------------------
# 2) Superficie: fijar cobertura 0 %, Ke y modulador térmico calibrados
# ---------------------------------------------------------------------
source = source.replace(
    'min_value=0, max_value=100, value=70, step=5,',
    'min_value=0, max_value=100, value=0, step=5,',
)

source = source.replace(
    'ke_val = float(np.interp(cobertura_pct, x_cobertura, [0.85, 0.50, 0.25, 0.10]))\n            mod_termico = float(np.interp(cobertura_pct, x_cobertura, [1.00, 0.95, 0.90, 0.80]))',
    '# Parámetros calibrados contra validacion.xlsx — app_optimizar_cobertura.py\n            ke_val = 1.60\n            mod_termico = 0.90',
)

# ---------------------------------------------------------------------
# 3) Sidebar: defaults calibrados
# ---------------------------------------------------------------------
source = source.replace(
    'w_max_val = st.sidebar.number_input("Cap. de Campo Superficial (mm)", value=30.0, step=1.0)',
    'w_max_val = st.sidebar.number_input("Cap. de Campo Superficial (mm)", value=26.0, step=1.0)',
)

source = source.replace(
    'w_max_val = st.sidebar.number_input("Cap. de Campo Superficial (mm)", value=26.0, step=1.0)\n'
    'st.sidebar.divider()',
    'w_max_val = st.sidebar.number_input("Cap. de Campo Superficial (mm)", value=26.0, step=1.0)\n'
    'humedad_mid = st.sidebar.number_input("Humedad_mid calibrada", value=0.36, step=0.01, format="%.2f")\n'
    'corte_seco = st.sidebar.number_input("Corte seco HR calibrado", value=0.25, step=0.01, format="%.2f")\n'
    'st.sidebar.caption("Calibrado contra validacion.xlsx · Ke=1.60 · Mod térmico=0.90")\n'
    'st.sidebar.divider()',
)

# ---------------------------------------------------------------------
# 4) Motor hídrico: usar humedad_mid y corte_seco calibrados
# ---------------------------------------------------------------------
source = source.replace(
    'df["Hydric_Factor"] = 1 / (1 + np.exp(-10 * (humedad_relativa - 0.3)))',
    'df["Hydric_Factor"] = 1 / (1 + np.exp(-10 * (humedad_relativa - humedad_mid)))',
)

source = source.replace(
    'df.loc[humedad_relativa < 0.20, "EMERREL"] = 0.0',
    'df.loc[humedad_relativa < corte_seco, "EMERREL"] = 0.0',
)

# ---------------------------------------------------------------------
# 5) Ajuste de controles fisiológicos por defecto
# ---------------------------------------------------------------------
source = source.replace(
    'umbral_termoinhibicion = st.sidebar.number_input("Umbral Termoinhibición (°C)", 15.0, 35.0, 24.0, 0.5)',
    'umbral_termoinhibicion = st.sidebar.number_input("Umbral Termoinhibición (°C)", 15.0, 35.0, 24.0, 0.5)',
)

# ---------------------------------------------------------------------
# 6) Tarjeta visible de calibración
# ---------------------------------------------------------------------
source = source.replace(
    'st.sidebar.info("🔬 **Modo Event-to-Event Habilitado**: Los desvíos se calculan de manera dinámica adaptándose al intervalo real de muestreo a campo (7 a 21 días), protegiendo la varianza pura del flujo.")',
    'st.sidebar.success("✅ **App calibrada**: cobertura 0 %, W_Max=26 mm, Ke=1.60, mod térmico=0.90, humedad_mid=0.36, corte_seco=0.25. Validación: validacion.xlsx.")\n'
    'st.sidebar.warning("⚠️ Ke=1.60 quedó en el límite superior del barrido. Repetir optimización con Ke máximo > 1.60 para confirmar estabilidad.")\n'
    'st.sidebar.info("🔬 **Modo Event-to-Event Habilitado**: Los desvíos se calculan de manera dinámica adaptándose al intervalo real de muestreo a campo (7 a 21 días), protegiendo la varianza pura del flujo.")',
)

compiled = compile(source, str(APP_ORIGINAL), "exec")
exec(compiled, globals(), globals())
