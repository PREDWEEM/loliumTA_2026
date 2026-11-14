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
ventana del graminicida de barbecho (600–800 °Cd, raigrás de 4 a 6 macollos desde el inicio). El modelo (`alerta_lolmu.py`, `alerta_lolmu_params.json`,
`visualizacion_alerta_lolmu.py`) es el calibrado con 17 campañas de Bordenave y combina, con la información disponible a la
fecha de emisión:

1. el calendario medio de emergencia de LOLMU;
2. la respuesta a las lluvias ya registradas (balance de agua y tiempo térmico de suelo);
3. opcionalmente, la fecha en que se vieron las primeras plántulas del lote, que además ancla el reloj térmico (sin ese dato,
   la ventana del graminicida de barbecho no se calcula: el inicio no se estima con el clima);
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

**Inicio de la curva.** Cuando se informa la fecha de las primeras plántulas (o hay un primer conteo con plántulas), la curva
del calendario propio vale 0 antes de esa fecha, arranca ese día en la fracción típica ya emergida al ver las primeras plantas
(~18 %) y desde ahí sigue el calendario reescalado (error medio de la curva desde las primeras plántulas: 6,5 puntos, contra
6,1 con el calendario por fecha).

## Señal de lluvia (actualización de noviembre de 2026)

La señal de lluvia usa ahora el **modelo de pulsos ajustado con las 17 campañas de *Lolium multiflorum* de Bordenave**.
El tamaño de cada pulso depende del agua que entró al suelo, de cuán seco estaba y de la época del año (máximo para
lluvias de fines de mayo); la mitad de la respuesta emerge dentro de 114 °Cd de tiempo térmico de suelo y el 90 %
dentro de 252 °Cd. Con el nuevo núcleo se reajustaron los modelos logísticos, los percentiles de referencia y la
recalibración de flujos tardíos (`alerta_lolmu_params.json`). El balance de agua y la temperatura de suelo no cambian.

Validación con cada campaña de Bordenave predicha con el ajuste sin ella (incluido el núcleo de lluvia): AUC 0,77 solo
con clima, 0,83 con primeras plántulas y 0,87 con conteos. En las ocho localidades de validación, sin reajustar: 0,89,
0,92 y 0,93 (calendario solo, 0,89). En Tres Arroyos: 0,90, 0,93 y 0,90 (antes 0,93, 0,94 y 0,93); las tablas de
validación de este sitio que figuran más arriba corresponden a la versión anterior.

## Flujos tardíos

Del **1 de agosto al 1 de octubre** la app recomienda **recorrer los lotes cada 2 semanas** para definir uso de graminicida
postemergente (`tardia.recorrida`): en esa época la probabilidad de flujo tardío no distinguió las semanas con emergencia
(AUC 0,41–0,50) y bajar el umbral solo marcaba más semanas al azar. Del 1/05 al 31/07 la emergencia que sigue a la siembra
se evalúa con el **riesgo para el residual** (ver más abajo), por eso la app ya no muestra la probabilidad de flujos tardíos
de mayo–junio (`tardia` sigue en los parámetros, sin uso en la pantalla).

## Qué hacer esta semana

Arriba de la alerta, un recuadro resume las acciones de la semana según la fecha y los datos cargados: recorrer el lote ante
un gran flujo probable o hasta ver las primeras plántulas; el momento del **graminicida de barbecho**; si conviene un
**residual** a la siembra prevista y si hay lluvia para activarlo; y la recorrida de flujos tardíos (1/08–1/10). El mismo resumen va en el Excel.

- **Graminicida de barbecho:** la ventana de 600–800 °Cd desde las primeras plántulas corresponde al raigrás de **4 a 6
  macollos**, momento óptimo para intervenir con un graminicida de barbecho.
- **Activación del residual:** suma la lluvia registrada y pronosticada en los 7 días desde la siembra; con 10 mm o más la
  activación es probable (el mínimo exacto depende del producto).
- **Flujos tardíos:** solo del 1/08 al 1/10 (recorrida cada 2 semanas); antes, la emergencia posterior a la siembra se
  evalúa con el riesgo para el residual.

## Residual a la siembra (guía de riesgo)

Debajo de la alerta, un recuadro siempre visible muestra, para la **fecha de siembra prevista** (1/05–31/07; si no se
carga, el 1/06 como referencia), el riesgo de nacimientos en la **ventana de acción del residual: de 10 días antes a 45 días
después de la siembra**, el período de acción promedio de un herbicida residual. Tiene tres tarjetas: riesgo, nacimientos en
la ventana (mediana de las campañas, en plantas/m² y en % de la campaña) y recomendación. Se calcula con las 3 campañas de Tres Arroyos (`residual` en `alerta_lolmu_params.json`). Las plantas/m² usan la densidad de cada campaña con conteos; las campañas que solo
tienen la curva relativa usan la densidad de referencia del sitio (7947 pl/m² por campaña, mediana de las
campañas con conteos):

| Siembra | Nacimientos de −10 a +45 días (mediana) | Riesgo para el residual | Campañas del sitio |
|---|---|---|---|
| 1/05 | 707 pl/m² (13,5 %) | Muy alto | 2023: 358 pl/m² · 2024: 1171 pl/m² · 2026: 707 pl/m² |
| 15/05 | 515 pl/m² (6,6 %) | Muy alto | 2023: 174 pl/m² · 2024: 703 pl/m² · 2026: 515 pl/m² |
| 1/06 | 320 pl/m² (2,6 %) | Muy alto | 2023: 64 pl/m² · 2024: 620 pl/m² · 2026: 320 pl/m² |
| 15/06 | 228 pl/m² (1,9 %) | Muy alto | 2023: 26 pl/m² · 2024: 533 pl/m² · 2026: 228 pl/m² |
| 1/07 | 141 pl/m² (1,2 %) | Muy alto | 2023: 13 pl/m² · 2024: 367 pl/m² · 2026: 141 pl/m² |

El nivel de riesgo surge de las plantas/m² que nacieron en la ventana en una campaña típica (mediana). Se usan plantas
y no el % de la campaña porque en sitios con mucha densidad una fracción chica son muchas plantas (en San Pedro, 0,8 % de
la campaña fueron 29–42 pl/m²):

| Riesgo | Nacimientos en la ventana | Recomendación |
|---|---|---|
| 🔴 Muy alto | ≥ 50 pl/m² | Usar residual |
| 🟠 Alto | 20–50 pl/m² | Residual recomendado |
| 🟡 Medio | 5–20 pl/m² | Según infestación: residual si el lote tiene alta infestación o antecedentes de raigrás |
| 🟢 Bajo | < 5 pl/m² | Residual solo en lotes muy infestados |

Sin otro dato, las densidades son las registradas en los ensayos del sitio. El usuario puede indicar la **infestación de raigrás de su lote** (opcional): baja (unas 200 pl/m² en toda la campaña), media (unas 1.000) o alta (unas 4.000). Con ese dato las plantas/m² se calculan con la densidad del lote y, como la infestación ya está considerada, las recomendaciones de los niveles medio y bajo pasan a «residual opcional» y «no hace falta residual».

Debajo de las tarjetas se indica el valor de cada campaña del sitio y qué proporción de las campañas cayó en cada nivel. Es
un **riesgo histórico, no una predicción de la campaña**: ni la fecha de primeras plántulas ni el clima anticiparon cuánta
emergencia queda después de la siembra.

En el gráfico, la segunda fila del **semáforo semanal** («Residual (siembra)») muestra, para cada semana del 1/05 al
31/07, el riesgo para el residual si se siembra el primer día de esa semana: rojo, alto o muy alto (≥ 20 pl/m²); amarillo, medio
(5–20 pl/m²); verde, bajo (< 5 pl/m²), con la infestación del lote si se indicó. Con fecha de siembra cargada, las semanas empiezan ese día, así la celda
de la siembra coincide con el recuadro del residual. La fecha de siembra prevista se marca
con una línea verde y una flecha con cartel.

## Condiciones de uso

No se concede licencia de uso por el solo hecho de acceder al repositorio. Cualquier utilización académica, técnica, institucional o comercial que exceda la visualización del contenido requiere autorización previa y escrita del titular de los derechos correspondientes.

Las solicitudes de autorización deben canalizarse mediante los medios de contacto del titular del repositorio PREDWEEM.

## Limitación de responsabilidad

PREDWEEM es una herramienta de soporte para decisiones y no sustituye el diagnóstico profesional, el monitoreo a campo ni la evaluación agronómica específica de cada lote. Las decisiones de manejo deben ser adoptadas por profesionales responsables considerando las condiciones locales y la normativa aplicable.

## Autoría

**PREDWEEM by Guillermo R. Chantre**

## Módulo común (LOLIUM-COMUN)

La alerta, su visualización y el control de acceso son comunes a las 8 apps de Lolium y están en el paquete
[PREDWEEM/LOLIUM-COMUN](https://github.com/PREDWEEM/LOLIUM-COMUN), fijado por versión en `requirements.txt`
(`lolium-comun @ git+https://github.com/PREDWEEM/LOLIUM-COMUN@<commit>  # v1.4.1`). En esta app, `alerta_lolmu.py`,
`visualizacion_alerta_lolmu.py` y `acceso_predweem.py` son archivos puente que cargan ese paquete con los parámetros de este
sitio (`alerta_lolmu_params.json`), que siguen acá. Los cambios a la alerta se hacen en LOLIUM-COMUN y llegan a la app al
actualizar la versión en `requirements.txt`.
