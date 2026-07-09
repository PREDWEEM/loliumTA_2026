# -*- coding: utf-8 -*-
"""
Calibrador edafico para PREDWEEM LOLIUM — Tres Arroyos 2026
Caso especifico: cobertura de suelo = 0 %, con dos pulsos observados
cerca del 24 de mayo y 28 de junio.

Uso basico:
    python calibrar_suelo_desnudo_bimodal.py

Con archivo de campo propio:
    python calibrar_suelo_desnudo_bimodal.py --campo tres_arroyos_campo.csv

Salida:
    resultados_calibracion_suelo_desnudo_bimodal.csv

El barrido mantiene cobertura = 0 %, por lo tanto mod_termico = 1.0.
Se calibran parametros edaficos superficiales:
    - W_Max: capacidad de almacenamiento superficial efectiva (mm)
    - Ke_Suelo: coeficiente de evaporacion/secado de suelo desnudo
    - Humedad_mid: punto medio de la respuesta hidrica sigmoide
    - Corte_seco: humedad relativa minima para permitir emergencia
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
LAT_TRES_ARROYOS = -38.4500
COBERTURA_PCT = 0
MOD_TERMICO_SUELO_DESNUDO = 1.00
TBASE, TOPT, TCRIT = 2.0, 20.0, 30.0

# Fechas reportadas por observacion de campo.
TARGET_PEAKS = [pd.Timestamp("2026-05-24"), pd.Timestamp("2026-06-28")]


class PracticalANNModel:
    def __init__(self, IW: np.ndarray, bIW: np.ndarray, LW: np.ndarray, bLW: np.ndarray):
        self.IW, self.bIW, self.LW, self.bLW = IW, bIW, LW, bLW
        self.input_min = np.array([1, 0, -7, 0], dtype=float)
        self.input_max = np.array([300, 41, 25.5, 84], dtype=float)

    def normalize(self, X: np.ndarray) -> np.ndarray:
        return 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1

    def predict(self, Xreal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xn = self.normalize(Xreal)
        a1 = np.tanh(Xn @ self.IW + self.bIW)
        emerrel = (np.tanh((a1 @ self.LW.T).flatten() + self.bLW) + 1) / 2
        return emerrel, np.cumsum(emerrel)


def load_ann(base: Path = BASE) -> PracticalANNModel:
    return PracticalANNModel(
        np.load(base / "IW.npy"),
        np.load(base / "bias_IW.npy"),
        np.load(base / "LW.npy"),
        np.load(base / "bias_out.npy"),
    )


def calcular_et0_hargreaves(jday: np.ndarray, tmax: np.ndarray, tmin: np.ndarray, latitud: float) -> np.ndarray:
    lat_rad = np.radians(latitud)
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * jday)
    dec = 0.409 * np.sin(2 * np.pi / 365 * jday - 1.39)
    ws = np.arccos(-np.tan(lat_rad) * np.tan(dec))
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(lat_rad) * np.sin(dec)
        + np.cos(lat_rad) * np.cos(dec) * np.sin(ws)
    )
    ra_mm = ra / 2.45
    tmean = (tmax + tmin) / 2.0
    trange = np.maximum(tmax - tmin, 0)
    return np.maximum(0.0023 * ra_mm * (tmean + 17.8) * np.sqrt(trange), 0)


def balance_hidrico_superficial(prec: np.ndarray, et0: np.ndarray, w_max: float, ke_suelo: float) -> np.ndarray:
    """Balance superficial con secado exponencial Kr, igual al motor principal."""
    n = len(prec)
    w = np.zeros(n, dtype=float)
    w[0] = w_max / 2.0
    for i in range(1, n):
        kr = w[i - 1] / w_max if w_max > 0 else 0.0
        ke_dinamico = ke_suelo * kr
        evaporacion_real = et0[i] * ke_dinamico
        w[i] = max(0.0, min(w_max, w[i - 1] + prec[i] - evaporacion_real))
    return w


def preparar_meteo(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    df.columns = [c.upper().strip() for c in df.columns]
    df = df.rename(columns={"FECHA": "Fecha", "DATE": "Fecha", "PREC": "Prec", "LLUVIA": "Prec"})
    requeridas = ["Fecha", "TMAX", "TMIN", "Prec"]
    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas meteorologicas requeridas: {faltantes}")
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df = df.dropna(subset=requeridas).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    df["Tmedia_aire"] = (df["TMAX"] + df["TMIN"]) / 2.0
    amp = (df["TMAX"] - df["TMIN"]) / 2.0
    df["TMAX_suelo"] = df["Tmedia_aire"] + amp * MOD_TERMICO_SUELO_DESNUDO
    df["TMIN_suelo"] = df["Tmedia_aire"] - amp * MOD_TERMICO_SUELO_DESNUDO
    df["ET0"] = calcular_et0_hargreaves(
        df["Julian_days"].values,
        df["TMAX"].values,
        df["TMIN"].values,
        latitud=LAT_TRES_ARROYOS,
    )
    return df


def aplicar_filtro_primer_pico(df: pd.DataFrame, umbral: Optional[float]) -> tuple[pd.DataFrame, Optional[pd.Timestamp]]:
    if umbral is None:
        df = df.copy()
        df["Primer_Pico_Habilitado"] = True
        return df, None

    df = df.copy()
    candidatos = df.index[df["EMERREL"] > umbral].tolist()
    if candidatos:
        idx = candidatos[0]
        fecha = df.loc[idx, "Fecha"]
        df["Primer_Pico_Habilitado"] = df.index >= idx
        df.loc[df.index < idx, "EMERREL"] = 0.0
        return df, fecha

    df["Primer_Pico_Habilitado"] = False
    df["EMERREL"] = 0.0
    return df, None


def simular(
    df_meteo: pd.DataFrame,
    modelo_ann: PracticalANNModel,
    w_max: float,
    ke_suelo: float,
    humedad_mid: float,
    corte_seco: float,
    umbral_choque_hidrico: float = 45.0,
    umbral_termoinhibicion: float = 24.0,
    umbral_primer_pico: Optional[float] = 0.05,
) -> tuple[pd.DataFrame, Optional[pd.Timestamp]]:
    df = df_meteo.copy()

    X = df[["Julian_days", "TMAX_suelo", "TMIN_suelo", "Prec"]].to_numpy(float)
    emerrel_raw, _ = modelo_ann.predict(X)
    df["EMERREL_RAW"] = np.maximum(emerrel_raw, 0.0)
    df.loc[df["Julian_days"] <= 45, "EMERREL_RAW"] = 0.0

    # Choque hidrico temprano: se mantiene igual al motor actual.
    df["Prec_3d"] = df["Prec"].rolling(window=3, min_periods=1).sum()
    mask_ruptura = (
        (df["Julian_days"] > 45)
        & (df["Julian_days"] <= 110)
        & (df["Prec_3d"] >= umbral_choque_hidrico)
    )
    df.loc[mask_ruptura, "EMERREL_RAW"] = np.maximum(df.loc[mask_ruptura, "EMERREL_RAW"], 0.75)

    df["W_superficial"] = balance_hidrico_superficial(
        df["Prec"].values,
        df["ET0"].values,
        w_max=w_max,
        ke_suelo=ke_suelo,
    )
    humedad_relativa = df["W_superficial"] / w_max
    df["Humedad_Relativa"] = humedad_relativa
    df["Hydric_Factor"] = 1 / (1 + np.exp(-10 * (humedad_relativa - humedad_mid)))
    df["EMERREL"] = df["EMERREL_RAW"] * df["Hydric_Factor"]
    df.loc[humedad_relativa < corte_seco, "EMERREL"] = 0.0

    # Recarga inicial por evento: se conserva, pero con W_Max bajo permite suelo desnudo mas reactivo.
    df["Lluvia_Recarga"] = (df["Prec"] >= w_max).cummax()
    df.loc[~df["Lluvia_Recarga"], "EMERREL"] = 0.0

    df["Tmedia_5d"] = df["Tmedia_aire"].rolling(window=5, min_periods=1).mean()
    df.loc[df["Tmedia_5d"] >= umbral_termoinhibicion, "EMERREL"] = 0.0
    df.loc[df["Julian_days"] <= 45, "EMERREL"] = 0.0
    df["EMERREL"] = np.clip(df["EMERREL"], 0, 1.0)

    df, fecha_primer_pico = aplicar_filtro_primer_pico(df, umbral_primer_pico)
    return df, fecha_primer_pico


def score_dos_picos(df: pd.DataFrame, target_peaks: Iterable[pd.Timestamp], ventana_dias: int = 7) -> dict:
    total = float(df["EMERREL"].sum())
    if total <= 0:
        return {
            "Score_Dos_Picos": -999.0,
            "Pico1_Fecha": pd.NaT,
            "Pico1_Valor": 0.0,
            "Pico1_Lag_d": 999,
            "Pico2_Fecha": pd.NaT,
            "Pico2_Valor": 0.0,
            "Pico2_Lag_d": 999,
            "Relacion_P2_P1": 0.0,
            "Penalidad_Extra": 1.0,
            "Penalidad_Valle": 1.0,
        }

    detalles = []
    mask_objetivo_total = pd.Series(False, index=df.index)
    for target in target_peaks:
        mask = (df["Fecha"] >= target - pd.Timedelta(days=ventana_dias)) & (
            df["Fecha"] <= target + pd.Timedelta(days=ventana_dias)
        )
        mask_objetivo_total = mask_objetivo_total | mask
        ventana = df.loc[mask].copy()
        if ventana.empty or ventana["EMERREL"].max() <= 0:
            detalles.append((pd.NaT, 0.0, 999))
            continue
        row = ventana.loc[ventana["EMERREL"].idxmax()]
        lag = abs((row["Fecha"] - target).days)
        detalles.append((row["Fecha"], float(row["EMERREL"]), int(lag)))

    (f1, p1, lag1), (f2, p2, lag2) = detalles
    fuera = float(df.loc[~mask_objetivo_total, "EMERREL"].sum() / total)

    # Penaliza un flujo continuo entre picos. Queremos dos pulsos separados.
    valle_mask = (df["Fecha"] > TARGET_PEAKS[0] + pd.Timedelta(days=ventana_dias)) & (
        df["Fecha"] < TARGET_PEAKS[1] - pd.Timedelta(days=ventana_dias)
    )
    valle = float(df.loc[valle_mask, "EMERREL"].sum() / total) if total > 0 else 1.0

    if p1 > 0 and p2 > 0:
        balance = min(p1, p2) / max(p1, p2)
    else:
        balance = 0.0

    score = (
        1.50 * p1
        + 1.50 * p2
        + 0.75 * balance
        - 0.06 * (lag1 + lag2)
        - 0.85 * valle
        - 0.45 * fuera
    )

    return {
        "Score_Dos_Picos": score,
        "Pico1_Fecha": f1,
        "Pico1_Valor": p1,
        "Pico1_Lag_d": lag1,
        "Pico2_Fecha": f2,
        "Pico2_Valor": p2,
        "Pico2_Lag_d": lag2,
        "Relacion_P2_P1": balance,
        "Penalidad_Extra": fuera,
        "Penalidad_Valle": valle,
    }


def sincronizar_intervalos_variables(df_sim: pd.DataFrame, df_campo: pd.DataFrame, col_fecha: str, col_plm2: str) -> pd.DataFrame:
    df_campo = df_campo.sort_values(col_fecha).copy()
    df_campo["Campo_Acum_Abs"] = df_campo[col_plm2].cumsum()
    fechas = df_campo[col_fecha].tolist()
    registros = []
    for i in range(1, len(fechas)):
        f_ini, f_fin = fechas[i - 1], fechas[i]
        obs_ini = df_campo.loc[df_campo[col_fecha] == f_ini, "Campo_Acum_Abs"].values[0]
        obs_fin = df_campo.loc[df_campo[col_fecha] == f_fin, "Campo_Acum_Abs"].values[0]
        flujo_obs = max(0.0, obs_fin - obs_ini)
        flujo_sim = df_sim.loc[(df_sim["Fecha"] > f_ini) & (df_sim["Fecha"] <= f_fin), "EMERREL"].sum()
        acum_sim_fin = df_sim.loc[df_sim["Fecha"] <= f_fin, "EMERREL"].sum()
        registros.append({
            "Fecha": f_fin,
            "Flujo_Obs_Abs": flujo_obs,
            "Flujo_Sim_Abs": flujo_sim,
            "Acum_Obs_Abs": obs_fin,
            "Acum_Sim_Abs": acum_sim_fin,
        })
    out = pd.DataFrame(registros)
    if out.empty:
        return out
    total_obs = out["Flujo_Obs_Abs"].sum()
    total_sim = df_sim.loc[df_sim["Fecha"] <= fechas[-1], "EMERREL"].sum()
    out["Campo_Relativo"] = out["Flujo_Obs_Abs"] / total_obs if total_obs > 0 else 0.0
    out["Sim_Relativo"] = out["Flujo_Sim_Abs"] / total_sim if total_sim > 0 else 0.0
    return out


def metricas_evento(df_sync: pd.DataFrame, umbral_deteccion: float = 0.05) -> dict:
    if df_sync.empty or len(df_sync) < 2:
        return {"F1_Campo": np.nan, "NSE_Campo": np.nan, "Pearson_Campo": np.nan}
    obs = df_sync["Campo_Relativo"].values
    sim = df_sync["Sim_Relativo"].values
    active = (obs > 0) | (sim > 0)
    if active.sum() >= 2 and np.std(obs[active]) > 0 and np.std(sim[active]) > 0:
        pearson = float(np.corrcoef(obs[active], sim[active])[0, 1])
    else:
        pearson = 0.0
    denom = np.sum((obs[active] - np.mean(obs[active])) ** 2) if active.sum() >= 2 else 0.0
    nse = float(1 - np.sum((sim[active] - obs[active]) ** 2) / denom) if denom > 0 else 0.0
    obs_evt = df_sync["Campo_Relativo"] > umbral_deteccion
    sim_evt = df_sync["Sim_Relativo"] > umbral_deteccion
    hits = int((obs_evt & sim_evt).sum())
    fp = int((~obs_evt & sim_evt).sum())
    miss = int((obs_evt & ~sim_evt).sum())
    precision = hits / (hits + fp) if hits + fp > 0 else 0.0
    recall = hits / (hits + miss) if hits + miss > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"F1_Campo": f1, "NSE_Campo": nse, "Pearson_Campo": pearson}


def cargar_campo(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    campo = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    col_fecha = "FECHA" if "FECHA" in campo.columns else campo.columns[0]
    campo[col_fecha] = pd.to_datetime(campo[col_fecha])
    return campo


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibracion edafica suelo desnudo bimodal PREDWEEM Tres Arroyos 2026")
    parser.add_argument("--meteo", default="meteo_daily.csv", help="Archivo meteorologico CSV/XLSX")
    parser.add_argument("--campo", default=None, help="Archivo de campo opcional CSV/XLSX")
    parser.add_argument("--salida", default="resultados_calibracion_suelo_desnudo_bimodal.csv")
    parser.add_argument("--umbral_primer_pico", type=float, default=0.05, help="Use -1 para desactivar el filtro")
    args = parser.parse_args()

    meteo_path = BASE / args.meteo
    campo_path = BASE / args.campo if args.campo else None
    umbral_primer_pico = None if args.umbral_primer_pico < 0 else args.umbral_primer_pico

    df_meteo = preparar_meteo(meteo_path)
    modelo_ann = load_ann(BASE)
    df_campo = cargar_campo(campo_path)

    resultados = []
    for w_max in np.arange(8.0, 27.0, 1.0):
        for ke in np.round(np.arange(0.75, 1.26, 0.05), 2):
            for humedad_mid in [0.24, 0.27, 0.30, 0.33, 0.36]:
                for corte_seco in [0.15, 0.18, 0.20, 0.23, 0.25]:
                    if corte_seco >= humedad_mid:
                        continue
                    df_sim, fecha_inicio = simular(
                        df_meteo,
                        modelo_ann,
                        w_max=w_max,
                        ke_suelo=ke,
                        humedad_mid=humedad_mid,
                        corte_seco=corte_seco,
                        umbral_primer_pico=umbral_primer_pico,
                    )
                    s = score_dos_picos(df_sim, TARGET_PEAKS)
                    fila = {
                        "Cobertura_%": COBERTURA_PCT,
                        "W_Max_mm": w_max,
                        "Ke_Suelo": ke,
                        "Humedad_mid": humedad_mid,
                        "Corte_seco": corte_seco,
                        "Mod_Termico": MOD_TERMICO_SUELO_DESNUDO,
                        "Fecha_Inicio_Filtro": fecha_inicio,
                        **s,
                    }

                    if df_campo is not None:
                        col_fecha = "FECHA" if "FECHA" in df_campo.columns else df_campo.columns[0]
                        col_plm2 = "PLM2" if "PLM2" in df_campo.columns else df_campo.columns[1]
                        sync = sincronizar_intervalos_variables(df_sim, df_campo, col_fecha, col_plm2)
                        fila.update(metricas_evento(sync))

                    resultados.append(fila)

    res = pd.DataFrame(resultados)
    orden = ["Score_Dos_Picos"]
    asc = [False]
    if df_campo is not None:
        orden = ["F1_Campo", "NSE_Campo", "Score_Dos_Picos"]
        asc = [False, False, False]
    res = res.sort_values(orden, ascending=asc).reset_index(drop=True)
    res.to_csv(BASE / args.salida, index=False)

    top = res.head(20).copy()
    cols = [
        "W_Max_mm", "Ke_Suelo", "Humedad_mid", "Corte_seco", "Score_Dos_Picos",
        "Pico1_Fecha", "Pico1_Valor", "Pico1_Lag_d", "Pico2_Fecha", "Pico2_Valor", "Pico2_Lag_d",
        "Relacion_P2_P1", "Penalidad_Valle", "Penalidad_Extra",
    ]
    if df_campo is not None:
        cols = ["F1_Campo", "NSE_Campo", "Pearson_Campo"] + cols

    print("\n=== TOP 20 combinaciones suelo desnudo bimodal ===")
    print(top[cols].to_string(index=False))
    best = res.iloc[0]
    print("\n=== PARAMETROS SUGERIDOS PARA LA APP ===")
    print(f"Cobertura de rastrojo: 0 %")
    print(f"Cap. de Campo Superficial W_Max: {best['W_Max_mm']:.1f} mm")
    print(f"Ke_Suelo: {best['Ke_Suelo']:.2f}")
    print(f"Humedad_mid sigmoide: {best['Humedad_mid']:.2f}")
    print(f"Corte seco humedad relativa: {best['Corte_seco']:.2f}")
    print(f"Archivo guardado: {BASE / args.salida}")


if __name__ == "__main__":
    main()
