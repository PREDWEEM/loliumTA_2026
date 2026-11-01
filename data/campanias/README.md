# Campañas de Tres Arroyos: emergencia y clima

Son los datos de las campañas de *Lolium multiflorum* de Tres Arroyos con que se validó la alerta y se construye el
**calendario propio** del panel de emergencia acumulada (`calendario_local.py`). Se copiaron del gemelo digital
(`PREDWEEM/TREASA_digitaltween`, `data/reference/` y `data/calibration/`) sin cambios en los valores.
La actualización automática no modifica estas carpetas.

| Campaña | Conteos | Clima |
|---|---|---|
| 2023 | 29 visitas del 27/02 al 09/10, cinco réplicas (`REP1`–`REP5`) y promedio `PLM2`; 2658.8 pl/m². El primer conteo ya tenía plántulas (sin cero inicial). | 283 días (2023-01-01 – 2023-10-10), planilla aportada «meteo 2023 tres arroyos.xls»; falta TMAX/TMIN del 17/06/2023 (no se imputó). Estación no identificada en el archivo. |
| 2024 | 10 conteos cada 2–4 semanas, del 15/02 al 30/07; 7946.7 pl/m². Sin cero inicial. | 335 días (2024-01-01 – 2024-11-30), archivo aportado «meteo TRES ARROYOS 2024.csv». Estación no identificada en el archivo. |
| 2026 | 17 conteos del 05/02 al 16/09, con cero inicial; 12089.7 pl/m². | 274 días (2026-01-01 – 2026-10-01), copia de `meteo_daily.csv` al cierre: SIGA–INTA Barrow observado. |

En cada carpeta:
- `emergencia_tres_arroyos_<año>.csv`: `Fecha`, `PLM2` (plántulas nuevas por m² en el intervalo que termina en esa fecha),
  días desde el conteo anterior, fracción de cada intervalo y emergencia acumulada (% del total registrado).
- `emergencia_tres_arroyos_<año>_original.xlsx`: planilla original.
- `clima_tres_arroyos_<año>.csv` (Fecha, TMAX, TMIN, Prec y procedencia) y, para 2023 y 2024, el archivo original.

**Para sumar una campaña**, cree la carpeta `<año>/` con `emergencia_tres_arroyos_<año>.csv` (al menos `Fecha` y `PLM2`) una vez
cerrada; el calendario la incorpora solo. La campaña en curso se excluye del calendario.

## Huellas SHA-256

| Archivo | SHA-256 |
|---|---|
| `2023/clima_tres_arroyos_2023.csv` | `9d53675afd67d08675a9ae712768cc1286cc3b4c9c3e9abae19e1cd3d1f272f3` |
| `2023/clima_tres_arroyos_2023_original.xls` | `a915f3eb6d939e24b0a7cb1b1c9d88c956cc5849b22743ddb322fc55f4f187d6` |
| `2023/emergencia_tres_arroyos_2023.csv` | `6eb7dea067c4cb698ceaffcb980d3c65c86c9bc5d899dbef9b411ff52e97f4e1` |
| `2023/emergencia_tres_arroyos_2023_original.xlsx` | `51eae9e32c3544a1688a7c45cf8343b3486c5af29a168aea30ce9abc52244425` |
| `2024/clima_tres_arroyos_2024.csv` | `78e36167eaf65819928c94c5e4c45bc95f4927b02240213f08019faabd520d2f` |
| `2024/clima_tres_arroyos_2024_original.csv` | `42d534e9863f59fe238af080ff204857c02c9ce73962f309c7ff49661dd5088a` |
| `2024/emergencia_tres_arroyos_2024.csv` | `2739f3c5f0a39aced8e7f448181f1b46be576b2538c77412e00c59208d600674` |
| `2024/emergencia_tres_arroyos_2024_original.xlsx` | `b480daf0dd201b52945b5a70b695c27f240ab8cc96d16af81d759f15cf58fb33` |
| `2026/clima_tres_arroyos_2026.csv` | `9dd2b5d5e3a7778f4cd0ab5225104f278f029c6738d408b53c39dabf7c77bd68` |
| `2026/emergencia_tres_arroyos_2026.csv` | `c679ce034abdf58633fbe5d8200ba9a9493263dc6e46aba89b32617737096006` |
| `2026/emergencia_tres_arroyos_2026_original.xlsx` | `021eb652657495cdf873ed6e120bd69565c497c0bde603bb3769205b91e87fca` |
