# Loan Default Prediction System

**Domain:** Banking  
**Use Case:** Bank Muscat  
**Project Type:** Machine Learning + Web Application + MLOps Prototype

## Overview
An end-to-end ML application for estimating the probability of serious borrower delinquency using the public **Give Me Some Credit** dataset. Bank Muscat is the use-case context; the model is not trained on proprietary Bank Muscat customer data.

## Stack
- Python 3.13
- Pandas, NumPy, scikit-learn
- XGBoost
- SHAP
- FastAPI + Uvicorn
- PostgreSQL / Neon
- MLflow
- Evidently AI
- Streamlit
- Docker / Docker Compose
- AWS ECS (planned deployment)

## Current Model Results

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8205 | 0.3599 | 0.5928 | 0.1561 | 0.2471 |
| Logistic Regression (Balanced) | 0.8255 | 0.3629 | 0.2708 | 0.6304 | 0.3788 |
| Random Forest | 0.8469 | 0.3712 | 0.5762 | 0.1810 | 0.2755 |
| Random Forest (Balanced) | 0.8468 | 0.3468 | 0.4266 | 0.3636 | 0.3926 |
| XGBoost | 0.8693 | 0.4100 | 0.6051 | ~0.2000 | 0.3018 |
| XGBoost (Tuned) | 0.8692 | 0.4095 | 0.6065 | 0.1875 | 0.2865 |

Current validation-stage XGBoost ROC-AUC: **0.8693**.

The API currently uses threshold **0.20**, selected during validation as the maximum F1 among the tested thresholds. This is not a finalized banking policy.

## Architecture

```text
Streamlit (:8501)
        |
        v
FastAPI (:8000)
        |
   +----+---------+-------------+
   |              |             |
 XGBoost + SHAP  PostgreSQL   Evidently
                 / Neon       Monitoring

MLflow -> Experiment / model tracking
Docker -> Containerization
AWS ECS -> Planned deployment
```

## Project Structure

```text
Loan-Default-Predictor/
├── data/raw/
├── docs/
├── frontend/
│   ├── app.py
│   └── Dockerfile
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── main.py
│   ├── model_service.py
│   ├── schema.py
│   ├── database.py
│   └── monitoring.py
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
├── requirements-docker.txt
└── .env
```

## API

### Health
`GET /api/health`

### Prediction
`POST /api/predict`

Returns default probability, prediction, risk level, threshold, and SHAP explanations.

### Prediction History
`GET /api/predictions`

Returns recent records stored in PostgreSQL.

### Monitoring
`GET /api/monitoring/drift`

Returns the current Evidently drift summary.

## Local Development

Activate the environment:

```powershell
.venv\Scripts\Activate.ps1
```

FastAPI:

```powershell
uvicorn src.main:app --reload
```

Streamlit, in a second terminal:

```powershell
streamlit run frontend/app.py
```

MLflow, in a third terminal:

```powershell
mlflow server --host 127.0.0.1 --port 5000
```

URLs:

```text
FastAPI:  http://127.0.0.1:8000
Swagger:  http://127.0.0.1:8000/docs
Streamlit: http://localhost:8501
MLflow:   http://127.0.0.1:5000
```

## Docker

Build and start:

```powershell
docker compose up --build
```

After images are built:

```powershell
docker compose up
```

Check status:

```powershell
docker compose ps
```

Stop:

```powershell
docker compose down
```

Docker service URLs:

```text
Frontend: http://localhost:8501
Backend:  http://localhost:8000
Swagger:  http://localhost:8000/docs
```

Inside Docker Compose, the frontend reaches the backend through:

```text
http://backend:8000
```

## Database

PostgreSQL is hosted through Neon.

The connection string is supplied with:

```text
DATABASE_URL
```

Never commit `.env` to GitHub.

Prediction records are stored in the `predictions` table.

## Dataset

- Training file: `cs-training.csv`
- Training rows: 150,000
- Inference/test rows: 101,503
- Positive class rate: approximately 6.68%

`cs-test.csv` contains the target column name but its target values are unlabeled (`NaN`), so it is not used for local ROC-AUC calculation.

## Explainability

SHAP is used to explain individual XGBoost predictions. Explanations describe model behavior and are not causal conclusions.

## Monitoring

Evidently compares reference and current feature distributions and generates:

```text
reports/data_drift_report.html
```

Observed drift does not by itself prove that model accuracy has changed.

## MLOps

MLflow tracks:

- parameters
- metrics
- model artifacts

Experiment:

```text
Loan Default Prediction
```

Logged full pipeline:

```text
xgb_full_pipeline
```

## Security

Do not commit `.env`.

Database URLs, API secrets, and cloud credentials must be supplied through environment variables or secure cloud configuration.

## Current Status

Completed:
- EDA and cleaning
- preprocessing
- Logistic Regression
- Random Forest
- XGBoost
- tuning
- threshold analysis
- SHAP
- MLflow
- FastAPI
- PostgreSQL/Neon
- prediction history
- Evidently monitoring
- Streamlit
- Docker
- end-to-end Docker integration tests

Next:
- deployment preparation
- AWS ECR/ECS
- cloud deployment testing
- final integration and documentation

## Academic Context

3rd-year B.Tech project at ABES Engineering College / AKTU.

Technical target: validation ROC-AUC >= 0.85. The NPA-reduction objective is a business objective, not a guaranteed model outcome.
