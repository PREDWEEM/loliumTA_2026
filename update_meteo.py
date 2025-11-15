# update_meteo.py
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

URL = "https://meteobahia.com.ar/scripts/forecast/for-ta.xml"
OUT = Path("meteo_daily.csv")

def to_float(x):
    try:
        return float(str(x).replace(",", "."))
    except:
        return None

def fetch_meteobahia():
    r = requests.get(URL, timeout=20)
    r.raise_for_status()
    root = ET.fromstring(r.content)

    rows = []
    for d in root.findall(".//forecast/tabular/day"):
        fecha = d.find("fecha").get("value")
        tmax  = d.find("tmax").get("value")
        tmin  = d.find("tmin").get("value")
        prec  = d.find("precip").get("value")

        rows.append({
            "Fecha": pd.to_datetime(fecha),
            "TMAX": to_float(tmax),
            "TMIN": to_float(tmin),
            "Prec": to_float(prec),
        })

    df = pd.DataFrame(rows).sort_values("Fecha")
    return df

def update_file():
    df_new = fetch_meteobahia()

    if OUT.exists():
        df_old = pd.read_csv(OUT, parse_dates=["Fecha"])
        df_all = pd.concat([df_old, df_new]).drop_duplicates("Fecha").sort_values("Fecha")
    else:
        df_all = df_new

    df_all.to_csv(OUT, index=False)
    print(f"[OK] Archivo actualizado: {len(df_all)} registros.")

if __name__ == "__main__":
    update_file()

