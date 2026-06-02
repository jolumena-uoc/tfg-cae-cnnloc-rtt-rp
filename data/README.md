# Datos

Este repositorio **no redistribuye** el conjunto de datos utilizado en el TFG, porque la licencia ODbL del paquete original así lo requiere. Para reproducir los experimentos hay que obtenerlo de su fuente oficial y colocarlo en esta carpeta.

## Fuente original

Matey-Sanz, M.; Torres-Sospedra, J. (2025). *Comparative Analysis of Indoor Positioning Approaches with Wi-Fi RTT from Android Devices* — Reproducible package.

- Repositorio: <https://github.com/matey97/comparative-analysis-android-rtt>
- DOI Zenodo: [10.5281/zenodo.15391797](https://doi.org/10.5281/zenodo.15391797)
- DOI del artículo: [10.1109/IPIN66788.2025.11212914](https://doi.org/10.1109/IPIN66788.2025.11212914)

## Cómo descargar e instalar los datos

Hay dos formas equivalentes de dejar los datos disponibles:

### Opción A — clonar el paquete original como carpeta hermana (recomendada)

Es la opción más sencilla y la que usan los notebooks por defecto. Desde el directorio padre del repositorio:

```bash
cd ..
git clone https://github.com/matey97/comparative-analysis-android-rtt.git Comparative_Analysis_Android_RTT_main
```

Tras esto, la estructura del directorio padre debe quedar:

```text
<directorio padre>/
├── tfg-cae-cnnloc-rtt-rp/                    # este repositorio
└── Comparative_Analysis_Android_RTT_main/    # paquete original
    └── 01_DATA/
        ├── poco_f2pro/
        ├── samsung_s24ultra/
        ├── aps.csv
        ├── locations.csv
        └── ...
```

Los notebooks resuelven la ruta a `../Comparative_Analysis_Android_RTT_main/01_DATA/` automáticamente.

### Opción B — descargar el ZIP de Zenodo y colocarlo dentro de `data/`

```bash
cd data/
# Descarga manual desde https://doi.org/10.5281/zenodo.15391797
unzip <fichero_zenodo>.zip
# Verifica que existe data/01_DATA/poco_f2pro/, data/01_DATA/samsung_s24ultra/, etc.
```

Si optas por esta opción, ajusta la variable `DATA_DIR` al inicio de cada notebook para apuntar a `data/01_DATA/`.

## Verificación rápida

Una vez instalado el dataset, puedes verificar que todo está en su sitio ejecutando el cuaderno de humo:

```bash
jupyter lab ../notebooks/00_smoke.ipynb
```

## Aviso de licencia

Los datos están licenciados bajo **Open Data Commons Attribution License (ODbL)**. Cualquier uso o redistribución debe respetar los términos originales. Consulta la licencia en el repositorio fuente.
