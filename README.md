# PREDWEEM — Lolium Tres Arroyos 2026

Repositorio correspondiente a la implementación de **PREDWEEM** para la predicción de la emergencia y la dinámica fenológica de *Lolium multiflorum* en Tres Arroyos, provincia de Buenos Aires, Argentina.

> **Propiedad intelectual**  
> Copyright © 2026 Guillermo R. Chantre / PREDWEEM.  
> Todos los derechos reservados.
>
> Este repositorio constituye software propietario. Su disponibilidad pública no concede autorización para utilizar, copiar, modificar, redistribuir, sublicenciar, realizar ingeniería inversa ni explotar comercialmente el código, los modelos, los parámetros, los pesos neuronales, la documentación o los datos incluidos.
>
> Consulte el aviso completo en [COPYRIGHT.md](COPYRIGHT.md).

## Finalidad

PREDWEEM es una herramienta de apoyo a la toma de decisiones agronómicas basada en la integración de datos meteorológicos, modelos predictivos y filtros ecofisiológicos para anticipar los flujos de emergencia de raigrás anual.

La implementación de este repositorio está orientada a **Tres Arroyos** y debe utilizarse considerando el dominio geográfico, climático y agronómico para el cual fue configurada, así como su estado específico de validación.

## Estrategia meteorológica

La serie operativa `meteo_daily.csv` utiliza las coordenadas de INTA Barrow (`-38.388, -60.346`) y aplica una jerarquía explícita de fuentes:

1. **SIGA–INTA Tres Arroyos / Barrow:** fuente observada prioritaria y definitiva para todas las fechas publicadas por la estación.
2. **ECMWF IFS histórico:** puente provisional para los días vencidos que SIGA todavía no publicó. Estas filas se identifican con `TipoDato=Provisional` y son reemplazadas automáticamente cuando aparece la observación SIGA.
3. **ECMWF IFS ENS 0,25°:** pronóstico desde el día actual hasta seis días posteriores.

La serie debe permanecer diaria y continua desde el 1 de enero de 2026 hasta el final del horizonte de pronóstico. Un hueco interior dentro del período ya publicado por SIGA provoca la detención de la actualización; no se inventan observaciones.

### Pronóstico por ensamble

Temperatura y precipitación se emparejan mediante el identificador real de cada miembro del ensamble. Cada miembro y día debe contener 24 valores horarios válidos. Un miembro incompleto se descarta para ese día y nunca se reemplaza una precipitación ausente por cero.

La actualización exige al menos 30 miembros válidos y, simultáneamente, el 80 % de los miembros emparejados disponibles. Si no se cumple esta condición, `meteo_daily.csv` conserva su versión anterior.

Para las filas de pronóstico se utiliza de manera coherente la **mediana P50** como serie operativa:

- `TMAX = TMAX_P50`
- `TMIN = TMIN_P50`
- `TMEDIA = TMEDIA_P50`
- `Prec = Prec_P50`

Las medias del ensamble se conservan por separado en `TMAX_Media_Ens`, `TMIN_Media_Ens`, `TMEDIA_Media_Ens` y `Prec_Media_Ens`. También se guardan P10, P50, P90 y probabilidades de superar 1, 5, 10 y 30 mm de precipitación diaria.

### Trazabilidad

Las columnas `Fuente`, `TipoDato`, `CalidadDato`, `N_miembros` y `Emision_UTC` permiten distinguir:

- observación de estación;
- dato provisional pendiente de SIGA;
- pronóstico por ensamble.

El archivo `data/estado_actualizacion_meteo.json` registra la última observación SIGA, el tramo provisional, el horizonte del pronóstico, el estadístico operativo y la cantidad mínima de miembros válidos.

La actualización automática se ejecuta a las **07:30** y **15:30**, hora argentina, y también puede iniciarse manualmente mediante GitHub Actions.

## Despliegue privado

La aplicación está preparada para utilizar los datos, el logo y los activos del modelo desde el checkout local. En Streamlit Community Cloud debe mantenerse la rama `main` y el archivo principal `app_emergencia.py`. Consulte [PRIVATE_REPOSITORY.md](PRIVATE_REPOSITORY.md).

## Condiciones de uso

No se concede licencia de uso por el solo hecho de acceder al repositorio. Cualquier utilización académica, técnica, institucional o comercial que exceda la visualización del contenido requiere autorización previa y escrita del titular de los derechos correspondientes.

Las solicitudes de autorización deben canalizarse mediante los medios de contacto del titular del repositorio PREDWEEM.

## Limitación de responsabilidad

PREDWEEM es una herramienta de soporte para decisiones y no sustituye el diagnóstico profesional, el monitoreo a campo ni la evaluación agronómica específica de cada lote. Las decisiones de manejo deben ser adoptadas por profesionales responsables considerando las condiciones locales y la normativa aplicable.

## Autoría

**PREDWEEM by Guillermo R. Chantre**
