# 🫀 Dashboard Web — Predicción de Mortalidad en UCI

Versión web del notebook `mortality_prediction.ipynb` (WiDS Datathon — mortalidad
hospitalaria en UCI con datos de las primeras 24 h). Es **el mismo pipeline y el
mismo modelo final del notebook** (XGBoost); aquí solo se traslada a una interfaz
web tipo dashboard con KPIs, filtros en el sidebar y dos paneles.

## ¿Qué hace?

- **Panel A — Análisis de Datos:** KPIs (registros filtrados, % de mortalidad,
  edad promedio), filtros en el sidebar (rango de edad, sexo, tipo de UCI,
  desenlace) y gráficos interactivos (histograma de edad por desenlace, dispersión
  IMC vs frecuencia cardíaca, distribución del target y mapa de calor de
  correlaciones), más la tabla de datos filtrados.
- **Panel B — Análisis Predictivo:** formulario con las 25 variables del modelo
  final, botón *Predecir*, resultado con probabilidad y la comparación de los 5
  modelos entrenados.

## Estructura

```
uci_dashboard/
├── app.py                  # La app Streamlit (Panel A + Panel B)
├── train_pipeline.py       # Reproduce el pipeline del notebook y genera /artifacts
├── requirements.txt        # Dependencias para ejecutar la app
├── requirements-train.txt  # Dependencias extra solo para re-entrenar
└── artifacts/
    ├── modelo_final.json    # XGBoost final (formato nativo, portable)
    ├── metadata.json        # variables, métricas y comparación de modelos
    └── panelA.parquet       # dataset limpio para el Panel A
```

Los modelos vienen **pre-entrenados** en `artifacts/` para que la app cargue al
instante y despliegue sin problemas. Se generaron ejecutando `train_pipeline.py`
sobre `training_v2.csv` (idéntico al notebook).

## Ejecutar en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Desplegar en Streamlit Community Cloud

1. Sube esta carpeta a un repo de GitHub (incluye la carpeta `artifacts/`).
2. En https://share.streamlit.io conecta el repo, archivo principal `app.py`.
3. Deploy. (Streamlit instala `requirements.txt` automáticamente.)

## Regenerar los artefactos (opcional)

Solo si quieres volver a entrenar desde cero con el CSV original:

```bash
pip install -r requirements-train.txt
python train_pipeline.py   # ajusta la ruta SRC al training_v2.csv
```

## Resultados del modelo (conjunto de prueba)

| Modelo              | Accuracy | F1     | Precision | Recall | AUC    |
|---------------------|----------|--------|-----------|--------|--------|
| **XGBoost** (final) | 0.8441   | 0.4160 | 0.2983    | 0.6874 | 0.8686 |
| Regresión Logística | 0.7985   | 0.3790 | 0.2524    | 0.7610 | 0.8657 |
| Random Forest       | 0.8212   | 0.3891 | 0.2687    | 0.7048 | 0.8594 |
| CatBoost            | 0.8891   | 0.4280 | 0.3670    | 0.5134 | 0.8505 |
| Árbol de Decisión   | 0.7605   | 0.3222 | 0.2089    | 0.7048 | 0.7871 |

Se eligió **XGBoost** (mayor AUC) como en el notebook.
