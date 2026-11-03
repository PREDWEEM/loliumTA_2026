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

## Alerta de flushes

La página muestra la **alerta de flushes** de *Lolium multiflorum*: la probabilidad de que en los próximos 7 días emerja
≥ 5 % de la campaña, con dos niveles (*Atención* desde 15 %, *Alerta* desde 25 %), la emergencia acumulada esperada y la
ventana de control post-emergente (600–800 °Cd desde el inicio). El modelo (`alerta_lolmu.py`, `alerta_lolmu_params.json`,
`visualizacion_alerta_lolmu.py`) es el calibrado con 17 campañas de Bordenave y combina, con la información disponible a la
fecha de emisión:

1. el calendario medio de emergencia de LOLMU;
2. la respuesta a las lluvias ya registradas (balance de agua y tiempo térmico de suelo);
3. opcionalmente, la fecha en que se vieron las primeras plántulas del lote, que además ancla el reloj térmico (sin ese dato,
   se usa el primer pico del motor PREDWEEM como inicio estimado);
4. opcionalmente, conteos de plántulas (tabla o planilla Excel/CSV con una columna «Fecha» y una por marco).

**Validación en Tres Arroyos.** Sin reajustar el modelo, con las campañas 2023, 2024 y 2026 publicadas en el gemelo
digital de Tres Arroyos (conteos y clima de cada año; 53 intervalos entre conteos, 9 con flush):

| Información disponible | AUC | Flushes en *Alerta* | Intervalos en *Alerta* |
|---|---|---|---|
| Solo clima | 0,93 | 9 de 9 | 43 % |
| + fecha de primeras plántulas | 0,94 | 9 de 9 | 40 % |
| + conteos | 0,93 | 9 de 9 | 38 % |

Por campaña (con primeras plántulas): 2023 AUC 0,97; 2024 0,95; 2026 0,82. Con los mismos intervalos, el gemelo digital
dio AUC 0,79 (0,96 en 2023, 0,55 en 2024 y 0,96 en 2026; 2026 se usó en su calibración). 2025 no se evaluó: el gemelo
solo tiene la curva procesada, sin clima. Límites: tres campañas y 9 flushes; en 2023 y 2024 el primer conteo ya tenía
plántulas; buena parte de los conteos de 2024 y 2026 son cada 2–4 semanas (la alerta con conteos está calibrada con conteos
semanales). La curva acumulada es la de Bordenave; en 2026 la emergencia de Tres Arroyos se concentró más tarde.

**Curva acumulada con calendario propio.** El panel de emergencia acumulada usa el calendario de Tres Arroyos
(`calendario_local.py`, con los conteos de `data/campanias/`; curva media y rango mínimo–máximo, sin la campaña en curso).
Prediciendo cada campaña con las otras dos, el error medio de la curva en las fechas de conteo fue 2,2 / 4,4 / 8,9 puntos
(2023 / 2024 / 2026), contra 7,6 / 6,9 / 13,3 con el calendario de Bordenave. La probabilidad de flush sigue con el
calendario de Bordenave: recalibrarla con estas campañas no la mejoró (AUC 0,937 contra 0,944), y usar el calendario
local dentro de la alerta la empeoró levemente (0,924, 8 de 9 flushes).

La barra lateral conserva solo los parámetros que usa la alerta (temperaturas base, óptima y crítica, y objetivos 600 y
800 °Cd); los del motor PREDWEEM quedan en sus valores por defecto (cobertura de rastrojo 20 %).

## Campaña vigente y actualización automática

La campaña meteorológica va del **1 de enero al 1 de octubre** de cada año (`_campania_vigente()` en
`actualizar_meteo_tres_arroyos.py`; las variables `CAMPANIA_INICIO` y `CAMPANIA_FIN` la fijan a mano). Después del
1 de octubre la serie queda cerrada; el **1 de enero siguiente** la carga se reinicia sola. Durante los primeros 21 días se
admite una serie sin observaciones SIGA del año (SIGA publica con demora): se usan el puente provisional y el pronóstico ECMWF.

## Despliegue privado

La aplicación está preparada para utilizar los datos, el logo y los activos del modelo desde el checkout local. En Streamlit Community Cloud debe mantenerse la rama `main` y el archivo principal `app_emergencia.py`. Consulte [PRIVATE_REPOSITORY.md](PRIVATE_REPOSITORY.md).

### Fin de carga de la campaña 2026

La última fecha meteorológica es el **1 de octubre de 2026, inclusive**.
El pronóstico de siete días se recorta al alcanzar ese límite, tanto en la
serie operativa como en los nuevos archivos de pronóstico. Desde el 2 de
octubre no se consulta el ensamble; las actualizaciones de SIGA y del puente
provisional quedan limitadas al 1 de octubre, permitiendo incorporar la
observación de ese día cuando se publique y reemplazar datos provisionales.
La validación admite una serie cerrada sin pronóstico y rechaza fechas
posteriores al cierre. Los archivos históricos existentes se conservan.

## Aviso de flujos tardíos (ventana de siembra)

Del 1 de mayo al 1 de julio la app muestra además la **probabilidad de flujos tardíos**: que en los próximos 7 días
emerja al menos 2 % de la campaña, con aviso desde 20 %. En esa época los grandes flujos (≥ 5 %) son raros, pero pocas plantas
por semana importan para decidir la fecha de siembra. Es la misma alerta recalibrada por nivel de información (`tardia`
en `alerta_lolmu_params.json`). Aparece en un recuadro debajo de la alerta, en el **semáforo
semanal** del gráfico (fila «Flujos tardíos»: violeta fuerte = flujo tardío probable; la fila «Grandes flujos» repite el nivel de cada semana) y en
la descarga en Excel.

| Información | AUC Bordenave (17 campañas, 132 semanas, 42 con ≥ 2 %) | Detectadas con el aviso | Semanas marcadas |
|---|---|---|---|
| Solo clima | 0,60 | 76 % | 78 % |
| + primeras plántulas | 0,78 | 86 % | 60 % |
| + conteos semanales | 0,87 | 83 % | 47 % |

Cada campaña se predijo con el ajuste sin ella. En otras seis localidades (59 intervalos del 1/05 al 1/07, 10 con ≥ 2 %
por semana) la AUC fue 0,88 con primeras plántulas y 0,93 con conteos, con 100 % y 90 % de detección; allí la probabilidad
tiende a ser alta (25 % en promedio contra 17 % observado). Solo con clima el aviso marca casi todas las semanas: conviene
usarlo con la fecha de primeras plántulas o con conteos.

## Condiciones de uso

No se concede licencia de uso por el solo hecho de acceder al repositorio. Cualquier utilización académica, técnica, institucional o comercial que exceda la visualización del contenido requiere autorización previa y escrita del titular de los derechos correspondientes.

Las solicitudes de autorización deben canalizarse mediante los medios de contacto del titular del repositorio PREDWEEM.

## Limitación de responsabilidad

PREDWEEM es una herramienta de soporte para decisiones y no sustituye el diagnóstico profesional, el monitoreo a campo ni la evaluación agronómica específica de cada lote. Las decisiones de manejo deben ser adoptadas por profesionales responsables considerando las condiciones locales y la normativa aplicable.

## Autoría

**PREDWEEM by Guillermo R. Chantre**
