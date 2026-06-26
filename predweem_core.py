# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 API CLOUD PREDWEEM INTEGRAL vK4.9.15 — LOLIUM TRES ARROYOS 2026
# ===============================================================

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import io
import base64

app = FastAPI(
    title="PREDWEEM Cloud Engine",
    description="API de simulación termo-hidrómica para Lolium - Tres Arroyos",
    version="4.9.15"
)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ---------------------------------------------------------
# LÓGICA TÉCNICA, MATEMÁTICA Y MODELOS NATIVOS
# ---------------------------------------------------------
def dtw_distance(a, b):
    na, nb = len(a), len(b)
    dp = np.full((na + 1, nb + 1), np.inf)
    dp[0, 0] = 0
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return dp[na, nb]

def calculate_tt_scalar(t, t_base, t_opt, t_crit):
    if t <= t_base: return 0.0
    elif t <= t_opt: return t - t_base
    elif t < t_crit: return (t - t_base) * ((t_crit - t) / (t_crit - t_opt))
    else: return 0.0

def calcular_et0_hargreaves(jday, tmax, tmin, latitud=-38.45):
    lat_rad = np.radians(latitud)
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * jday)
    dec = 0.409 * np.sin(2 * np.pi / 365 * jday - 1.39)
    ws = np.arccos(-np.tan(lat_rad) * np.tan(dec))
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (ws * np.sin(lat_rad) * np.sin(dec) + np.cos(lat_rad) * np.cos(dec) * np.sin(ws))
    ra_mm = ra / 2.45
    tmean = (tmax + tmin) / 2.0
    trange = np.maximum(tmax - tmin, 0)
    return np.maximum(0.0023 * ra_mm * (tmean + 17.8) * np.sqrt(trange), 0)

def balance_hidrico_superficial(prec, et0, w_max=15.0, ke_suelo=0.4):
    n = len(prec)
    w = np.zeros(n)
    w[0] = w_max / 2.0 
    for i in range(1, n):
        kr = w[i-1] / w_max  
        ke_dinamico = ke_suelo * kr
        evaporacion_real = et0[i] * ke_dinamico
        w[i] = max(0.0, min(w_max, w[i-1] + prec[i] - evaporacion_real))
    return w

class PracticalANNModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW, self.bIW, self.LW, self.bLW = IW, bIW, LW, bLW
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])
    def normalize(self, X): return 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1
    def predict(self, Xreal):
        Xn = self.normalize(Xreal)
        a1 = np.tanh(Xn @ self.IW + self.bIW)
        emerrel = (np.tanh((a1 @ self.LW.T).flatten() + self.bLW) + 1) / 2
        return emerrel, np.cumsum(emerrel)

# Carga segura en memoria del servidor al arrancar
try:
    modelo_ann = PracticalANNModel(np.load(BASE / "IW.npy"), np.load(BASE / "bias_IW.npy"), np.load(BASE / "LW.npy"), np.load(BASE / "bias_out.npy"))
    with open(BASE / "modelo_clusters_k3.pkl", "rb") as f: 
        cluster_model = pickle.load(f)
except Exception as e:
    print(f"⚠️ Alerta inicial: Componentes de pesos npy/pkl ausentes en raíz. Asegurar despliegue de artefactos.")
    modelo_ann, cluster_model = None, None

# ---------------------------------------------------------
# FUNCIONES AUXILIARES DE PROCESAMIENTO
# ---------------------------------------------------------
def sincronizar_intervalos_variables(df_sim, df_campo, col_fecha, col_plm2):
    df_campo = df_campo.sort_values(col_fecha).copy()
    df_campo['Campo_Acum_Abs'] = df_campo[col_plm2].cumsum()
    fechas_reales = df_campo[col_fecha].tolist()
    registros = []
    
    for i in range(1, len(fechas_reales)):
        f_inicio = fechas_reales[i-1]
        f_fin = fechas_reales[i]
        dias_intervalo = (f_fin - f_inicio).days
        obs_inicio = df_campo.loc[df_campo[col_fecha] == f_inicio, 'Campo_Acum_Abs'].values[0]
        obs_fin = df_campo.loc[df_campo[col_fecha] == f_fin, 'Campo_Acum_Abs'].values[0]
        flujo_obs = max(0.0, obs_fin - obs_inicio)
        
        mask_sim = (df_sim['Fecha'] > f_inicio) & (df_sim['Fecha'] <= f_fin)
        flujo_sim = df_sim.loc[mask_sim, 'EMERREL'].sum()
        acum_sim_fin = df_sim.loc[df_sim['Fecha'] <= f_fin, 'EMERREL'].sum()
        
        registros.append({
            'Fecha': f_fin,
            'Dias_Intervalo': dias_intervalo,
            'Flujo_Obs_Abs': flujo_obs,
            'Flujo_Sim_Abs': flujo_sim,
            'Acum_Obs_Abs': obs_fin,
            'Acum_Sim_Abs': acum_sim_fin
        })
        
    df_res = pd.DataFrame(registros)
    if df_res.empty: return pd.DataFrame()
        
    total_obs = df_res['Flujo_Obs_Abs'].sum()
    total_sim = df_sim.loc[df_sim['Fecha'] <= fechas_reales[-1], 'EMERREL'].sum()
    
    df_res['Campo_Relativo'] = df_res['Flujo_Obs_Abs'] / total_obs if total_obs > 0 else 0.0
    df_res['Sim_Relativo'] = df_res['Flujo_Sim_Abs'] / total_sim if total_sim > 0 else 0.0
    df_res['Campo_Acumulado'] = df_res['Acum_Obs_Abs'] / df_campo['Campo_Acum_Abs'].max() if df_campo['Campo_Acum_Abs'].max() > 0 else 0.0
    df_res['Sim_Acumulado'] = df_res['Acum_Sim_Abs'] / df_sim['EMERREL'].sum() if df_sim['EMERREL'].sum() > 0 else 0.0
    return df_res

def calcular_metricas_validacion_integral(df_sync):
    if df_sync.empty or len(df_sync) < 2:
        return {"Pearson_Flujos": 0.0, "NSE_Flujos": 0.0, "KGE_Flujos": 0.0, "RMSE_Acumulado": 0.0, "CCC_Acumulado": 0.0, "R2_Acumulado": 0.0}

    mask_activos = (df_sync['Campo_Relativo'] > 0) | (df_sync['Sim_Relativo'] > 0)
    df_activos = df_sync[mask_activos].copy()
    
    if len(df_activos) < 2:
        pearson_r, nse_flujos, kge_flujos = 0.0, 0.0, 0.0
    else:
        obs, sim = df_activos['Campo_Relativo'].values, df_activos['Sim_Relativo'].values
        std_obs, std_sim = np.std(obs), np.std(sim)
        pearson_r = np.corrcoef(obs, sim)[0, 1] if std_obs > 0 and std_sim > 0 else 0.0
        var_obs_sum = np.sum((obs - np.mean(obs))**2)
        nse_flujos = 1 - (np.sum((sim - obs)**2) / var_obs_sum) if var_obs_sum > 0 else 0.0
        
        if np.mean(obs) > 0 and std_obs > 0:
            r = pearson_r
            alpha = std_sim / std_obs
            beta = np.mean(sim) / np.mean(obs)
            kge_flujos = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        else:
            kge_flujos = 0.0

    obs_acum, sim_acum = df_sync['Campo_Acumulado'].values, df_sync['Sim_Acumulado'].values
    rmse_acumulado = np.sqrt(np.mean((obs_acum - sim_acum)**2))
    
    mean_obs_ac, mean_sim_ac = np.mean(obs_acum), np.mean(sim_acum)
    var_obs_ac, var_sim_ac = np.var(obs_acum), np.var(sim_acum)
    covar_ac = np.mean((obs_acum - mean_obs_ac) * (sim_acum - mean_sim_ac))
    
    denominador_ccc = var_obs_ac + var_sim_ac + (mean_obs_ac - mean_sim_ac)**2
    ccc_acumulado = (2 * covar_ac) / denominador_ccc if denominador_ccc > 0 else 0.0
    
    ss_res_ac = np.sum((obs_acum - sim_acum)**2)
    ss_tot_ac = np.sum((obs_acum - mean_obs_ac)**2)
    r2_acumulado = 1 - (ss_res_ac / ss_tot_ac) if ss_tot_ac > 0 else 0.0
    
    return {
        "Pearson_Flujos": pearson_r, "NSE_Flujos": nse_flujos, "KGE_Flujos": kge_flujos,
        "RMSE_Acumulado": rmse_acumulado, "CCC_Acumulado": ccc_acumulado, "R2_Acumulado": r2_acumulado
    }

# ---------------------------------------------------------
# ENDPOINT DE SIMULACIÓN CENTRAL
# ---------------------------------------------------------
@app.post("/simulate")
async def run_simulation(
    archivo_clima: UploadFile = File(...),
    archivo_campo: UploadFile = File(None),
    cobertura_pct: int = Form(0),
    umbral_er: float = Form(0.005),
    umbral_termoinhibicion: float = Form(24.0),
    umbral_choque_hidrico: float = Form(30.0),
    residualidad: int = Form(0),
    t_base_val: float = Form(2.0),
    t_opt_max: float = Form(20.0),
    t_critica: float = Form(30.0),
    dga_optimo: int = Form(600),
    dga_critico: int = Form(800),
    w_max_val: float = Form(20.0)
):
    if modelo_ann is None:
        raise HTTPException(status_code=500, detail="Los archivos de pesos (.npy) del modelo ANN no están inicializados en el servidor.")

    # 1. Lectura del archivo climático
    try:
        content = await archivo_clima.read()
        if archivo_clima.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando archivo_clima: {str(e)}")

    df.columns = [c.upper().strip() for c in df.columns]
    df = df.rename(columns={'FECHA': 'Fecha', 'DATE': 'Fecha', 'TMAX': 'TMAX', 'TMIN': 'TMIN', 'PREC': 'Prec', 'LLUVIA': 'Prec'})
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear

    # Moduladores derivados de la cobertura por rastrojo
    x_cobertura = [0, 30, 70, 100]
    ke_val = float(np.interp(cobertura_pct, x_cobertura, [0.95, 0.50, 0.25, 0.10]))
    mod_termico = float(np.interp(cobertura_pct, x_cobertura, [1.00, 0.95, 0.90, 0.80]))

    # Simulación Térmica
    df["Tmedia_aire"] = (df["TMAX"] + df["TMIN"]) / 2
    amplitud_termica = (df["TMAX"] - df["TMIN"]) / 2
    df["TMAX_suelo"] = df["Tmedia_aire"] + (amplitud_termica * mod_termico)
    df["TMIN_suelo"] = df["Tmedia_aire"] - (amplitud_termica * mod_termico)

    # 2. Lectura opcional de datos de Validación de Campo
    df_campo, col_fecha, col_plm2 = None, None, None
    if archivo_campo:
        try:
            content_campo = await archivo_campo.read()
            if archivo_campo.filename.endswith(('.xlsx', '.xls')):
                df_campo = pd.read_excel(io.BytesIO(content_campo))
            else:
                df_campo = pd.read_csv(io.BytesIO(content_campo))
            
            col_fecha = 'FECHA' if 'FECHA' in df_campo.columns else df_campo.columns[0]
            col_plm2 = 'PLM2' if 'PLM2' in df_campo.columns else df_campo.columns[1]
            df_campo[col_fecha] = pd.to_datetime(df_campo[col_fecha])
            df_campo = df_campo.sort_values(col_fecha).reset_index(drop=True)
            max_plm2 = df_campo[col_plm2].max()
            df_campo['Campo_Normalizado'] = df_campo[col_plm2] / max_plm2 if max_plm2 > 0 else 0
        except Exception as e:
            df_campo = None # Fallback resiliente si el archivo estructural está corrupto

    # 3. Motor Fisiológico de Emergencia (PREDWEEM Core)
    X = df[["Julian_days", "TMAX_suelo", "TMIN_suelo", "Prec"]].to_numpy(float)
    emerrel_raw, _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(emerrel_raw, 0.0)

    df["Prec_3d"] = df["Prec"].rolling(window=3, min_periods=1).sum()
    mask_ruptura = (df["Julian_days"] > 25) & (df["Julian_days"] <= 110) & (df["Prec_3d"] >= umbral_choque_hidrico)
    df.loc[mask_ruptura, "EMERREL"] = np.maximum(df.loc[mask_ruptura, "EMERREL"], 0.75)

    df["ET0"] = calcular_et0_hargreaves(df["Julian_days"].values, df["TMAX"].values, df["TMIN"].values, latitud=-38.4500)
    df["W_superficial"] = balance_hidrico_superficial(df["Prec"].values, df["ET0"].values, w_max=w_max_val, ke_suelo=ke_val)
    humedad_relativa = df["W_superficial"] / w_max_val
    df["Hydric_Factor"] = 1 / (1 + np.exp(-10 * (humedad_relativa - 0.3)))
    df["EMERREL"] = df["EMERREL"] * df["Hydric_Factor"]

    df.loc[humedad_relativa < 0.20, "EMERREL"] = 0.0
    df['Lluvia_Recarga'] = (df['Prec'] >= w_max_val).cummax()
    df.loc[~df['Lluvia_Recarga'], "EMERREL"] = 0.0

    df["Tmedia"] = df["Tmedia_aire"]
    df["Tmedia_10d"] = df["Tmedia"].rolling(window=10, min_periods=1).mean()
    df.loc[df["Tmedia_10d"] >= umbral_termoinhibicion, "EMERREL"] = 0.0
    df["EMERREL"] = np.clip(df["EMERREL"], 0, 1.0)
    df.loc[df["Julian_days"] <= 25, "EMERREL"] = 0.0

    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, t_base_val, t_opt_max, t_critica))

    # 4. Cálculo de Ventanas Logísticas y de Control
    fecha_hoy = pd.Timestamp.now().normalize()
    if fecha_hoy not in df['Fecha'].values: fecha_hoy = df['Fecha'].max()
    indices_pulso = df.index[df["EMERREL"] >= umbral_er].tolist()

    dga_hoy, dga_7dias = 0.0, 0.0
    fecha_inicio_ventana, fecha_control, fecha_limite = None, None, None
    
    if indices_pulso:
        fecha_inicio_ventana = df.loc[indices_pulso[0], "Fecha"]
        df_desde_pico = df[df["Fecha"] >= fecha_inicio_ventana].copy()
        df_desde_pico["DGA_cum"] = df_desde_pico["DG"].cumsum()
        
        df_control = df_desde_pico[df_desde_pico["DGA_cum"] >= dga_optimo]
        if not df_control.empty: fecha_control = df_control.iloc[0]["Fecha"]
        
        df_limite = df_desde_pico[df_desde_pico["DGA_cum"] >= dga_critico]
        if not df_limite.empty: fecha_limite = df_limite.iloc[0]["Fecha"]
        
        dga_hoy = df.loc[(df["Fecha"] >= fecha_inicio_ventana) & (df["Fecha"] <= fecha_hoy), "DG"].sum()
        idx_hoy = df[df["Fecha"] == fecha_hoy].index[0]
        dga_7dias = dga_hoy + df.iloc[idx_hoy + 1: idx_hoy + 8]["DG"].sum() if idx_hoy + 8 <= len(df) else dga_hoy

    # Sincronización e Índices Robustos Event-to-Event
    metricas = {}
    pec, peak_lag, lead_time, desfase_t50 = 0.0, 0, 0, 0
    df_sincronizado = pd.DataFrame()

    if df_campo is not None:
        df_sincronizado = sincronizar_intervalos_variables(df, df_campo, col_fecha, col_plm2)
        if not df_sincronizado.empty:
            metricas = calcular_metricas_validacion_integral(df_sincronizado)
            tot_plm2 = df_campo[col_plm2].sum()
            if tot_plm2 > 0:
                df_campo['cum_plm2_norm'] = df_campo[col_plm2].cumsum() / tot_plm2
                t50_obs_date = df_campo[df_campo['cum_plm2_norm'] >= 0.5].iloc[0][col_fecha]
                df_sim_trunc = df[df['Fecha'] <= df_campo[col_fecha].max()].copy()
                tot_emer = df_sim_trunc['EMERREL'].sum()
                if tot_emer > 0:
                    df_sim_trunc['cum_emer_norm'] = df_sim_trunc['EMERREL'].cumsum() / tot_emer
                    t50_sim_date = df_sim_trunc[df_sim_trunc['cum_emer_norm'] >= 0.5].iloc[0]['Fecha']
                    desfase_t50 = (t50_sim_date - t50_obs_date).days

            if fecha_control:
                malezas_totales_campo = df_campo[col_plm2].sum()
                pec = ((df_campo.loc[df_campo[col_fecha] <= fecha_control, col_plm2].sum() / malezas_totales_campo) * 100 if malezas_totales_campo > 0 else 0)
                peak_lag = (fecha_control - df_campo.loc[df_campo[col_plm2].idxmax(), col_fecha]).days
                df_alertas = df[df['EMERREL'] >= umbral_er]
                lead_time = (fecha_control - (df_alertas['Fecha'].iloc[0] if not df_alertas.empty else fecha_inicio_ventana)).days

    # 5. Análisis Estratégico Temporal mediante DTW
    clasificacion_dtw = {"categoria": "Insuficiente", "score": 0.0}
    df_obs = df[df["Fecha"] < pd.Timestamp("2026-05-01")].copy()
    if cluster_model and not df_obs.empty and df_obs["EMERREL"].sum() > 0:
        jd_corte = df_obs["Julian_days"].max()
        max_e = df_obs["EMERREL"].max() if df_obs["EMERREL"].max() > 0 else 1.0
        JD_COM = cluster_model["JD_common"]
        jd_grid = JD_COM[JD_COM <= jd_corte]
        obs_norm = np.interp(jd_grid, df_obs["Julian_days"], df_obs["EMERREL"] / max_e)
        dists = [dtw_distance(obs_norm, m[JD_COM <= jd_corte] / m[JD_COM <= jd_corte].max() if m[JD_COM <= jd_corte].max() > 0 else m[JD_COM <= jd_corte]) for m in cluster_model["curves_interp"]]
        pred = int(np.argmin(dists))
        nombres_patrones = {0: "Bimodal", 1: "Temprano", 2: "Tardío"}
        clasificacion_dtw = {"categoria": nombres_patrones.get(pred, "Desconocido"), "score": float(min(dists))}

    # 6. Generación del libro Excel en memoria (ExcelWriter)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_excel = df.copy()
        df_excel['Fecha'] = df_excel['Fecha'].dt.strftime('%Y-%m-%d')
        df_excel.to_excel(writer, index=False, sheet_name='Data_Diaria')
        if df_campo is not None and not df_sincronizado.empty:
            df_campo_exc = df_campo.copy()
            df_campo_exc[col_fecha] = df_campo_exc[col_fecha].dt.strftime('%Y-%m-%d')
            df_campo_exc.to_excel(writer, index=False, sheet_name='Campo_Validacion')
            pd.DataFrame({
                'Métrica de Validación': ['PEC (%)', 'Lag Control (días)', 'Lead Time Control (días)', 'Pearson (Flujos)', 'NSE (Flujos Reales)', 'KGE (Flujos)', 'RMSE (Acumulado)', 'R2 (Acumulado)', 'CCC (Acumulado)', 'Desfase T50 (días)'], 
                'Valor': [pec, peak_lag, lead_time, metricas.get("Pearson_Flujos", 0), metricas.get("NSE_Flujos", 0), metricas.get("KGE_Flujos", 0), metricas.get("RMSE_Acumulado", 0), metricas.get("R2_Acumulado", 0), metricas.get("CCC_Acumulado", 0), desfase_t50]
            }).to_excel(writer, sheet_name='Validacion_Campo', index=False)
        pd.DataFrame({'Configuracion': ['T_Base', 'T_Optima', 'T_Critica', 'W_Max', 'Ke_Calculado', 'Mod_Termico', 'Umbral_Termoinhibicion'], 'Valor': [t_base_val, t_opt_max, t_critica, w_max_val, ke_val, mod_termico, umbral_termoinhibicion]}).to_excel(writer, sheet_name='Bio_Params', index=False)
    
    excel_base64 = base64.b64encode(output.getvalue()).decode('utf-8')

    # 7. Respuesta JSON Estructurada para consumo Frontend
    return JSONResponse(content={
        "logistica": {
            "fecha_inicio_ventana": fecha_inicio_ventana.strftime('%Y-%m-%d') if fecha_inicio_ventana else None,
            "fecha_critica_control": fecha_control.strftime('%Y-%m-%d') if fecha_control else None,
            "fecha_limite_ventana": fecha_limite.strftime('%Y-%m-%d') if fecha_limite else None,
            "dga_hoy": float(dga_hoy),
            "dga_pronostico_7d": float(dga_7dias)
        },
        "metricas_validacion": {
            "pec_pct": float(pec),
            "lag_dias": int(peak_lag),
            "lead_time_dias": int(lead_time),
            "desfase_t50_dias": int(desfase_t50),
            "indices_fidelidad": metricas
        },
        "estrategia_dtw": clasificacion_dtw,
        "vectores_graficas": {
            "fechas": df["Fecha"].dt.strftime('%Y-%m-%d').tolist(),
            "emerrel": df["EMERREL"].tolist(),
            "w_superficial": df["W_superficial"].tolist(),
            "precipitacion": df["Prec"].tolist()
        },
        "reporte_excel_base64": excel_base64
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
