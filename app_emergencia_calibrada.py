# -*- coding: utf-8 -*-
"""
🌾 PREDWEEM LOLIUM — TRES ARROYOS 2026
APP OPERATIVA CALIBRADA CONTRA validacion.xlsx

Esta variante ejecuta `app_emergencia.py` inyectando los parametros ganadores
obtenidos con `calibrar_validacion_suelo_desnudo.py`.

Parametros calibrados:
    cobertura_pct = 0
    w_max_val = 26.0
    ke_val = 1.25
    mod_termico = 1.00
    humedad_mid = 0.36
    corte_seco = 0.25
    umbral_choque_hidrico = 45.0
    umbral_termoinhibicion = 24.0

Ejecucion:
    streamlit run app_emergencia_calibrada.py

Motivo:
    Mantener `app_emergencia.py` intacta y disponer de una version calibrada,
    reversible y auditable para comparar con la app original.
"""

from __future__ import annotations

from pathlib import Path


BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
APP_ORIGINAL = BASE / "app_emergencia.py"


PARAMETROS_CALIBRADOS = {
    "cobertura_pct": 0,
    "w_max_val": 26.0,
    "ke_val": 1.25,
    "mod_termico": 1.00,
    "humedad_mid": 0.36,
    "corte_seco": 0.25,
    "umbral_choque_hidrico": 45.0,
    "umbral_termoinhibicion": 24.0,
}


if not APP_ORIGINAL.exists():
    raise FileNotFoundError(f"No se encontro la app original: {APP_ORIGINAL}")

source = APP_ORIGINAL.read_text(encoding="utf-8")

# ---------------------------------------------------------------------
# 1) Identificacion visual y documentacion interna
# ---------------------------------------------------------------------
source = source.replace(
    "# - OPTIMIZADOR 2D: Barrido enfocado puramente en la física del suelo (W_Max y Ke) usando ventanas reales.",
    "# - CALIBRACION CAMPO: Parametros edaficos fijados por Ajuste_Campo_Compuesto contra validacion.xlsx.",
)

source = source.replace(
    'st.title("🌾 PREDWEEM LOLIUM — TRES ARROYOS (BA) lat=-38.4500 lon=-60.2763")',
    'st.title("🌾 PREDWEEM LOLIUM — TRES ARROYOS CALIBRADO CAMPO (BA) lat=-38.4500 lon=-60.2763")',
)

# ---------------------------------------------------------------------
# 2) Carga por defecto de validacion.xlsx como archivo de campo
# ---------------------------------------------------------------------
source = source.replace(
    'df_campo_raw = load_data(archivo_campo, "tres_arroyos_campo")',
    'df_campo_raw = load_data(archivo_campo, "validacion")',
)

# ---------------------------------------------------------------------
# 3) Superficie: fijar suelo desnudo 0 %, Ke y modulador calibrados
# ---------------------------------------------------------------------
source = source.replace(
    'min_value=0, max_value=100, value=70, step=5,',
    'min_value=0, max_value=100, value=0, step=5,',
)

source = source.replace(
    'ke_val = float(np.interp(cobertura_pct, x_cobertura, [0.85, 0.50, 0.25, 0.10]))\n            mod_termico = float(np.interp(cobertura_pct, x_cobertura, [1.00, 0.95, 0.90, 0.80]))',
    '# Parametros calibrados contra validacion.xlsx — criterio Ajuste_Campo_Compuesto\n            ke_val = 1.25\n            mod_termico = 1.00',
)

# ---------------------------------------------------------------------
# 4) Sidebar: defaults calibrados
# ---------------------------------------------------------------------
source = source.replace(
    'umbral_termoinhibicion = st.sidebar.number_input("Umbral Termoinhibición (°C)", 15.0, 35.0, 24.0, 0.5)',
    'umbral_termoinhibicion = st.sidebar.number_input("Umbral Termoinhibición (°C)", 15.0, 35.0, 24.0, 0.5)',
)

source = source.replace(
    'value=45.0,\n    step=1.0\n)',
    'value=45.0,\n    step=1.0\n)',
    1,
)

source = source.replace(
    'w_max_val = st.sidebar.number_input("Cap. de Campo Superficial (mm)", value=30.0, step=1.0)',
    'w_max_val = st.sidebar.number_input("Cap. de Campo Superficial (mm)", value=26.0, step=1.0)',
)

# Insertar controles calibrados para el factor hidrico, inmediatamente despues de W_Max.
source = source.replace(
    'w_max_val = st.sidebar.number_input("Cap. de Campo Superficial (mm)", value=26.0, step=1.0)\n'
    'st.sidebar.divider()',
    'w_max_val = st.sidebar.number_input("Cap. de Campo Superficial (mm)", value=26.0, step=1.0)\n'
    'humedad_mid = st.sidebar.number_input("Humedad_mid calibrada", value=0.36, step=0.01, format="%.2f")\n'
    'corte_seco = st.sidebar.number_input("Corte seco HR calibrado", value=0.25, step=0.01, format="%.2f")\n'
    'st.sidebar.caption("Calibrado contra validacion.xlsx · Ajuste_Campo_Compuesto")\n'
    'st.sidebar.divider()',
)

# ---------------------------------------------------------------------
# 5) Motor hidrico: usar humedad_mid y corte_seco calibrados
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
# 6) Tarjeta visible de calibracion
# ---------------------------------------------------------------------
source = source.replace(
    'st.sidebar.info("🔬 **Modo Event-to-Event Habilitado**: Los desvíos se calculan de manera dinámica adaptándose al intervalo real de muestreo a campo (7 a 21 días), protegiendo la varianza pura del flujo.")',
    'st.sidebar.success("✅ **App calibrada**: suelo desnudo 0 %, W_Max=26 mm, Ke=1.25, humedad_mid=0.36, corte_seco=0.25. Criterio: Ajuste_Campo_Compuesto contra validacion.xlsx.")\n'
    'st.sidebar.info("🔬 **Modo Event-to-Event Habilitado**: Los desvíos se calculan de manera dinámica adaptándose al intervalo real de muestreo a campo (7 a 21 días), protegiendo la varianza pura del flujo.")',
)

# ---------------------------------------------------------------------
# Ejecutar la app original ya modificada en memoria.
# ---------------------------------------------------------------------
compiled = compile(source, str(APP_ORIGINAL), "exec")
exec(compiled, globals(), globals())
