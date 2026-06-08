# tfg-cae-cnnloc-rtt-rp

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Data License: ODbL](https://img.shields.io/badge/Data%20License-ODbL-brightgreen.svg)](https://opendatacommons.org/licenses/odbl/)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)](https://www.tensorflow.org/)

Paquete reproducible asociado al Trabajo Final de Grado **"Posicionamiento indoor con WiFi RTT en dispositivos Android: adaptación de CAE-CNNLoc y evaluación comparativa frente a centroide ponderado y KNN-DtC"** (Universitat Oberta de Catalunya, 2026).

> **Autor:** José Luis Mena Maestro · **Tutor:** Joaquín Torres-Sospedra · **PRA:** Antoni Pérez-Navarro.

Este repositorio adapta la arquitectura **CAE-CNNLoc 2D-Temporal** (Kargar-Barzi et al., 2024), pensada originalmente para *fingerprinting* basado en RSSI, al caso de WiFi RTT con cuatro puntos de acceso. La compara contra dos *baselines* clásicos (centroide ponderado y *fingerprinting* k-NN con representación DtC) sobre el conjunto de datos público de Matey-Sanz y Torres-Sospedra (2025).

---

## Tabla de contenidos

- [Estructura del repositorio](#estructura-del-repositorio)
- [Requisitos previos](#requisitos-previos)
- [Instalación](#instalación)
- [Datos](#datos)
- [Reproducir los experimentos](#reproducir-los-experimentos)
- [Resultados versionados](#resultados-versionados)
- [Cómo citar](#cómo-citar)
- [Licencia](#licencia)

---

## Estructura del repositorio

```text
tfg-cae-cnnloc-rtt-rp/
├── cnnloc_rtt/                    # paquete con la implementación CAE-CNNLoc 2D-Temporal
│   ├── data.py                    # construcción de matrices fingerprint 4xN
│   ├── models.py                  # arquitectura Keras (CAE + cabeza de regresión)
│   ├── train.py                   # entrenamiento en dos fases con multi-semilla
│   ├── eval.py                    # evaluación cross-device / cross-pose
│   └── utils.py
├── notebooks/
│   ├── 00_smoke.ipynb             # comprobación rápida de instalación
│   ├── 01_baselines.ipynb         # centroide ponderado y KNN-DtC
│   ├── 02_cnnloc.ipynb            # CAE-CNNLoc 2D-Temporal
│   └── 03_comparativa.ipynb       # tabla unificada y CDFs comparativas
├── scripts/                       # scripts reproducibles fuera de los notebooks
│   ├── run_cnnloc_one.py          # CAE-CNNLoc para una (subconjunto, semilla)
│   ├── run_cnnloc_full.py         # orquestador 4 subconjuntos x 5 semillas
│   ├── aggregate_summary.py       # CSVs resumen de baselines y CAE-CNNLoc
│   ├── tune_nwindow.py            # barrido del hiperparámetro N (8/16/32)
│   ├── rerun_knn_dtc_per_pair.py  # re-ejecución de KNN-DtC por (train, test)
│   ├── regenerate_baselines_cdf.py
│   ├── regenerate_cnnloc_curves.py
│   ├── regenerate_cdf_gen.py
│   └── regenerate_anx_figures.py
├── results/                       # CSVs agregados y figuras del documento
├── tests/                         # smoke tests del paquete
├── data/                          # placeholder con instrucciones de descarga (NO contiene datos)
├── requirements.txt
├── setup.ps1                      # creación del venv en Windows (.\setup.ps1)
├── LICENSE                        # Apache License 2.0
├── NOTICE                         # atribuciones
└── README.md                      # este fichero
```

## Requisitos previos

- Python **3.11** o **3.12** (probado con 3.12 en Windows 11).
- Aproximadamente **2 GB** libres en disco para el entorno virtual y los resultados regenerables.

## Instalación

### Opción A — Windows (PowerShell)

Desde la raíz del repositorio:

```powershell
.\setup.ps1
.\.venv\Scripts\Activate.ps1
```

### Opción B — Linux / macOS o instalación manual

```bash
python3.12 -m venv .venv
source .venv/bin/activate    # en Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Datos

El conjunto de datos **no se redistribuye** en este repositorio (lo prohíbe la licencia ODbL del paquete original). Para reproducir los experimentos hay que descargarlo aparte y colocarlo en `data/`. Las instrucciones detalladas están en [`data/README.md`](data/README.md).

## Reproducir los experimentos

1. Activar el entorno virtual (si no está activo).
2. Verificar la instalación con el cuaderno de humo:
   ```bash
   jupyter lab notebooks/00_smoke.ipynb
   ```
3. Ejecutar los cuadernos en orden:
   - `01_baselines.ipynb` — centroide ponderado y KNN-DtC.
   - `02_cnnloc.ipynb` — entrenamiento y evaluación de CAE-CNNLoc 2D-Temporal.
   - `03_comparativa.ipynb` — tabla unificada y CDFs comparativas.

> **Nota.** Las semillas están fijadas (`utils.set_global_seed`) para garantizar repetibilidad, pero TensorFlow puede producir variaciones marginales entre máquinas distintas (CPU vs GPU, distintas versiones de cuDNN). El comportamiento global y la jerarquía de errores se mantienen.

### Versión "todo de golpe"

Si solo interesa regenerar resultados y figuras sin abrir Jupyter, el flujo completo (cinco semillas, agregación temporal de 5 s) se reproduce con la siguiente secuencia. Los pasos 1 y 2 generan los CSVs crudos de los baselines (no se versionan por tamaño); los pasos 3 y 4 entrenan y agregan CAE-CNNLoc; los pasos 5–8 regeneran las figuras del documento.

```bash
# 1. Baselines: centroide ponderado y KNN-DtC (todas las ventanas y k)
jupyter nbconvert --to notebook --execute notebooks/01_baselines.ipynb \
    --output notebooks/01_baselines.ipynb
python scripts/rerun_knn_dtc_per_pair.py     # KNN-DtC por par (train, test)

# 2. Barrido del hiperparámetro N (tabla y figura del anexo)
python scripts/tune_nwindow.py

# 3. CAE-CNNLoc 2D-Temporal: 4 subconjuntos x 5 semillas
python scripts/run_cnnloc_full.py

# 4. Consolida resúmenes y produce tabla_global.csv
python scripts/aggregate_summary.py

# 5-8. Figuras de la memoria
python scripts/regenerate_baselines_cdf.py
python scripts/regenerate_cnnloc_curves.py
python scripts/regenerate_cdf_gen.py
python scripts/regenerate_anx_figures.py
```

El orquestador `run_cnnloc_full.py` invoca internamente a `run_cnnloc_one.py` (que también puede ejecutarse de manera aislada para una sola combinación, p. ej. `python scripts/run_cnnloc_one.py --train POCO#STANDING --seed 0`). El paquete reproducible original de Matey-Sanz y Torres-Sospedra debe estar disponible como carpeta hermana del repositorio (o indicada con la variable de entorno `MATEY_RTT_DIR`); de allí provienen los enumerados `Device` y `MeasuringType` y el dataset.

## Resultados versionados

Para mantener el repositorio ligero, sólo se versionan los **CSVs agregados** (`*_summary.csv`, `*_per_sample.csv`, `*_history.csv`, `*_curves.csv`, `tabla_global.csv`) y las **figuras** (PDF) que aparecen en la memoria del TFG. Los volcados intermedios por (subconjunto, semilla) se regeneran al ejecutar los scripts y están listados en `.gitignore` para no versionarse.

## Cómo citar

Si este código te resulta útil, te agradeceríamos que cites el TFG:

```bibtex
@misc{menamaestro2026tfg,
  author       = {Mena Maestro, Jos\'e Luis},
  title        = {{Posicionamiento indoor con WiFi RTT en dispositivos Android: adaptaci\'on de CAE-CNNLoc y evaluaci\'on comparativa frente a centroide ponderado y KNN-DtC}},
  year         = {2026},
  note         = {Trabajo Final de Grado, Universitat Oberta de Catalunya},
  howpublished = {\url{https://github.com/jolumena-uoc/tfg-cae-cnnloc-rtt-rp}}
}
```

Y, por descontado, las dos referencias de las que parte el trabajo:

- A. Kargar-Barzi, E. Farahmand, N. Taheri-Chatrudi, A. Mahani, M. Shafique. **"An Edge-Based WiFi Fingerprinting Indoor Localization Using Convolutional Neural Network and Convolutional Auto-Encoder"**. *IEEE Access*, vol. 12, pp. 85050–85060, 2024.
- M. Matey-Sanz, J. Torres-Sospedra. **"Comparative Analysis of Indoor Positioning Approaches with Wi-Fi RTT from Android Devices"**. *IPIN 2025*. Reproducible package: <https://github.com/matey97/comparative-analysis-android-rtt> · Zenodo: <https://doi.org/10.5281/zenodo.15391797>.

## Licencia

Código distribuido bajo **Apache License 2.0** (ver [`LICENSE`](LICENSE) y [`NOTICE`](NOTICE)).

Los datos del dataset original heredan la licencia **Open Data Commons Attribution License (ODbL)** y se obtienen aparte (ver [`data/README.md`](data/README.md)).
