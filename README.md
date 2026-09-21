# Clasificador de sentimiento de reseñas de películas + API

Clasificación de sentimiento (positiva/negativa) de reseñas de IMDb usando
regresión logística sobre representaciones Bag-of-Words (BoW). El proyecto
incluye el análisis completo en un notebook, el modelo entrenado y una API en
FastAPI para predecir nuevas reseñas.

## Contenido

- Análisis y experimentos (`ml/train_eval.ipynb`): preprocesamiento de
  texto, baseline con BoW, manejo de negaciones y n-gramas (1–2).
- Modelo final (`ml/sentiment_model.pkl`): pipeline
  `CountVectorizer(1–2 gramas) + LogisticRegression` con aproximadamente 89.5% de accuracy en el conjunto de test.
- API (`app/`): endpoint `POST /predict/` que recibe una reseña y devuelve
  la etiqueta y la probabilidad de que sea positiva.

## Estructura

```
.
├── app/
│   ├── main.py            # API FastAPI
│   ├── preprocess.py      # limpieza de texto (inferencia)
│   ├── requirements.txt
│   └── pyproject.toml
└── ml/
    ├── train_eval.ipynb   # análisis y experimentos
    ├── preprocessor.py    # limpieza de texto (entrenamiento)
    ├── retrain_model.py   # reentrena el modelo
    └── sentiment_model.pkl
```

## Requisitos

Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecutar la API

```bash
cd app
uvicorn main:app --reload
```

Documentación interactiva en <http://127.0.0.1:8000/docs>.

## Uso

```bash
curl -X POST http://127.0.0.1:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{"review": "This movie is not good."}'
```

Respuesta:

```json
{
  "review": "This movie is not good.",
  "label": "negative",
  "label_id": 0,
  "probability_positive": 0.306819
}
```

## Reentrenar el modelo

```bash
python ml/retrain_model.py
```

El script descarga el dataset de IMDb y sobrescribe `ml/sentiment_model.pkl`.

## Nota

El modelo se desarrollo en el contexto de la Tarea 1 del curso `CC6104 - Procesamiento de Lenguage Natural`.