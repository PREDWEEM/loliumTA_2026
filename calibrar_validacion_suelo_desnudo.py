# -*- coding: utf-8 -*-
"""
Runner directo para calibrar PREDWEEM Tres Arroyos con el archivo real
`validacion.xlsx` del repositorio y cobertura de suelo = 0 %.

Ejecutar desde la raiz del repo:

    python calibrar_validacion_suelo_desnudo.py

Equivale a:

    python calibrar_suelo_desnudo_bimodal.py --campo validacion.xlsx --salida resultados_calibracion_validacion_suelo_desnudo.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

from calibrar_suelo_desnudo_bimodal import main


BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--campo" not in args:
        args.extend(["--campo", "validacion.xlsx"])

    if "--salida" not in args:
        args.extend(["--salida", "resultados_calibracion_validacion_suelo_desnudo.csv"])

    if "--meteo" not in args:
        args.extend(["--meteo", "meteo_daily.csv"])

    sys.argv = [sys.argv[0]] + args
    main()
