# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM vK3 — LOLIUM TRES ARROYOS 2026 (Proyectivo)
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, io
from pathlib import Path
import plotly.graph_objects as go
from datetime import timedelta

# [Mantenemos las funciones dtw_distance, PracticalANNModel y load_models]

# ===============================================================
# 📊 LÓGICA DE PROYECCIÓN TÉRMICA
# ===============================================================
def proyectar_fecha(df_v, target_dga, dga_actual):
    if dga_actual >= target_dga:
        return df_v[df_v["DGA_sum"] >= target_dga]["Fecha"].min(), False
    
    # Calcular promedio de DG de los últimos 7 días con datos
    ultimos_dg = df_v["DG"].tail(7).mean()
    if ultimos_dg <= 0.1: ultimos_dg = 5.0 # Valor de seguridad si hace mucho frío
    
    faltante = target_dga - dga_actual
    dias_estimados = int(faltante / ultimos_dg)
    fecha_proyectada = df_v["Fecha"].max() + timedelta(days=dias_estimados)
    return fecha_proyectada, True

# ... [Procesamiento de datos y ANN igual al anterior] ...

if df is not None and modelo_ann is not None:
    # [Cálculos base de EMERREL, Riesgo y DG]
    
    # --- LÓGICA DE VALIDACIÓN: 2 PULSOS EN 5 DÍAS ---
    indices_pulso = df.index[df["EMERREL"] >= umbral_rel_input].tolist()
    fecha_inicio_ventana = None
    
    for i in range(len(indices_pulso) - 1):
        idx1, idx2 = indices_pulso[i], indices_pulso[i+1]
        if (df.loc[idx2, "Fecha"] - df.loc[idx1, "Fecha"]).days <= 5:
            fecha_inicio_ventana = df.loc[idx1, "Fecha"]
            break

    st.title("🌾 PREDWEEM vK3 — TRES ARROYOS")

    if fecha_inicio_ventana:
        df_v = df[df["Fecha"] >= fecha_inicio_ventana].copy()
        df_v["DGA_sum"] = df_v["DG"].cumsum()
        dga_actual = df_v["DGA_sum"].iloc[-1]

        # Proyecciones
        f_opt, es_proy_opt = proyectar_fecha(df_v, dga_optimo, dga_actual)
        f_crit, es_proy_crit = proyectar_fecha(df_v, dga_critico, dga_actual)

        # --- SECCIÓN: TACÓMETRO ---
        st.divider()
        col_gauge, col_info = st.columns([1, 1])

        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = dga_actual,
                title = {'text': "Días Grado Acumulados", 'font': {'size': 20}},
                gauge = {
                    'axis': {'range': [0, dga_critico + 100]},
                    'bar': {'color': "#34495e"},
                    'steps': [
                        {'range': [0, dga_optimo], 'color': "#2ecc71"},
                        {'range': [dga_optimo, dga_critico], 'color': "#f1c40f"},
                        {'range': [dga_critico, dga_critico + 100], 'color': "#e74c3c"}]}))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_info:
            st.subheader("📅 Cronograma Estimado")
            
            def fmt_hizo(f, proy):
                txt = "Estimada" if proy else "Alcanzada"
                color = "orange" if proy else "gray"
                return f"<small style='color:{color}'>{txt}: {f.strftime('%d-%b')}</small>"

            st.markdown(f"**Inicio de Emergencia:** {fecha_inicio_ventana.strftime('%d-%b-%Y')}")
            st.markdown(f"### 🟢 Límite Óptimo\n{fmt_hizo(f_opt, es_proy_opt)}", unsafe_allow_html=True)
            st.markdown(f"### 🔴 Límite Crítico (Macollaje)\n{fmt_hizo(f_crit, es_proy_crit)}", unsafe_allow_html=True)

            if dga_actual > dga_critico:
                st.error("⚠️ ESTADO CRÍTICO: La maleza ha superado el límite de macollaje.")
            elif dga_actual > dga_optimo:
                st.warning("🟡 VENTANA CERRÁNDOSE: Maleza en crecimiento activo.")
            else:
                st.success("🟢 ESTADO ÓPTIMO: Alta sensibilidad a herbicidas.")

        # --- SECCIÓN: DIAGRAMA DE CRECIMIENTO ---
        st.divider()
        st.subheader("📈 Progresión del Desarrollo")
        
        
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=df_v["Fecha"], y=df_v["DGA_sum"], name="DGA Real", line=dict(color='black', width=3)))
        fig_line.add_hline(y=dga_optimo, line_dash="dot", line_color="green", annotation_text="Límite 1-3 Hojas")
        fig_line.add_hline(y=dga_critico, line_dash="dot", line_color="red", annotation_text="Inicio Macollaje")
        st.plotly_chart(fig_line, use_container_width=True)

    else:
        st.info("🔎 El sistema está monitoreando el clima. Se activará el tacómetro cuando se detecte emergencia sostenida (2 pulsos).")

    # [Resto del código: Mapa de Riesgo Heatmap, Análisis DTW y Descarga de Excel]
