# Loan Default Prediction — Day 4 Summary

**Domain:** Banking  
**Use Case:** Bank Muscat  
**Project:** Loan Default Prediction System  
**Date:** 2026-10-01

---

# Part 1 — Concise Summary

## 1. Backend Development

- Built the reusable FastAPI backend around the trained XGBoost pipeline.
- Added:
  - `src/main.py`
  - `src/model_service.py`
  - `src/schema.py`
  - `src/database.py`
  - `src/monitoring.py`
- Implemented:
  - `GET /api/health`
  - `POST /api/predict`
  - `GET /api/predictions`
  - `GET /api/monitoring/drift`
- Added Pydantic input validation for borrower information.
- Loaded the saved XGBoost preprocessing + model pipeline from `models/xgb_pipeline.joblib`.
- Applied the validation-selected threshold of `0.20`.
- Added human-readable risk levels.
- Added SHAP-based individual prediction explanations.

## 2. PostgreSQL Integration

- Connected the backend to PostgreSQL using the `DATABASE_URL` environment variable.
- Used Neon PostgreSQL for the project database.
- Created the `predictions` table.
- Stored:
  - borrower input values
  - default probability
  - prediction
  - risk level
  - threshold
  - model version
  - prediction timestamp
- Added a prediction-history API so previously generated predictions can be retrieved.

## 3. Model Serving + Explainability

- Integrated the trained XGBoost model into the API.
- Reproduced the same feature engineering used during training:
  - `IncomeMissing`
  - `DelinquencyAnomaly`
  - anomaly handling for `96` and `98`
  - invalid `age = 0` handling
- Reordered inference features to match the trained pipeline.
- Added SHAP explanations to the prediction response.

Example response structure:

```json
{
  "default_probability": 0.01798,
  "prediction": 0,
  "risk_level": "Low",
  "threshold": 0.2,
  "explanations": []
}
```

## 4. Evidently Monitoring

- Added Evidently AI monitoring to the backend.
- Used the training data as the reference distribution and the available unlabeled test/inference data as the current distribution.
- Added cleaning logic so monitoring data matches the expected feature schema.
- Generated a data-drift HTML report.
- Added `/api/monitoring/drift` to return drift summary information.

The current comparison did not detect significant feature drift in the checked dataset.

Important limitation:

- This is a development/reference-data comparison, not evidence of production drift behavior.

## 5. Streamlit Frontend

- Connected the Streamlit frontend to the FastAPI backend.
- Added a more polished banking-style interface.
- Added separate sections/pages for:
  - Prediction
  - Prediction History
  - Monitoring
- Added visual presentation for:
  - default probability
  - prediction result
  - risk level
  - threshold
  - SHAP explanations
  - prediction history
  - drift monitoring

The frontend communicates with the backend through HTTP rather than accessing the model directly.

## 6. Docker Containerization

- Installed and configured Docker Desktop with WSL2.
- Verified the Docker Engine is running correctly.
- Created:
  - `Dockerfile`
  - `frontend/Dockerfile`
  - `docker-compose.yml`
  - `.dockerignore`
  - `requirements-docker.txt`
- Separated Linux runtime dependencies from the Windows development `requirements.txt`.
- Successfully built both Docker images:
  - backend
  - frontend
- Configured Docker Compose networking so the frontend reaches the backend using:
  - `http://backend:8000`

A local port conflict on `8501` was identified as being caused by an older Python/Streamlit process running outside Docker.

## 7. SRS and Project Documentation

- Updated the Software Requirements Specification to reflect the current multi-tier architecture and implemented components.
- The architecture now clearly represents:
  - Streamlit frontend
  - FastAPI API layer
  - ML prediction pipeline
  - SHAP
  - PostgreSQL
  - Evidently AI
  - MLflow
  - Docker
  - AWS ECS as the deployment target

The SRS continues to identify Bank Muscat as the use-case context while using the public Give Me Some Credit dataset for development.

## 8. Git/GitHub Preparation

- Reviewed the project files before the next GitHub push.
- Confirmed that `.env` must remain untracked because it contains secrets such as the PostgreSQL connection string.
- Prepared the current project state for a Git commit and push.

---

# Part 2 — Detailed Explanation

## 1. Transition from Notebook to Application

The project has now moved beyond notebook-only experimentation.

Earlier stages focused on:

```text
Dataset
   ↓
EDA
   ↓
Preprocessing
   ↓
Model training
   ↓
Evaluation
   ↓
SHAP
   ↓
MLflow
```

The current stage converts those experiments into an actual application:

```text
User
 ↓
Streamlit
 ↓
FastAPI
 ↓
Saved ML Pipeline
 ↓
XGBoost Prediction
 ↓
SHAP Explanation
 ↓
PostgreSQL Storage
```

This separation is important because the frontend should not contain model logic. The model is accessed through the API.

---

## 2. FastAPI Backend

The main application entry point is:

```text
src/main.py
```

The API provides health checking:

```http
GET /api/health
```

This endpoint is intentionally simple. It allows us to verify that the FastAPI service is running without loading a borrower prediction request.

The main prediction endpoint is:

```http
POST /api/predict
```

The request contains borrower features. FastAPI first validates them through the Pydantic schema before passing them to the model service.

The prediction process is:

```text
Request
  ↓
Pydantic validation
  ↓
Feature engineering
  ↓
Feature ordering
  ↓
Saved pipeline
  ↓
Probability
  ↓
Threshold
  ↓
SHAP
  ↓
Database storage
  ↓
JSON response
```

---

## 3. Why Pydantic Validation Is Used

The backend receives external input, so it should not assume that the incoming values are correct.

The schema validates properties such as:

- age must be greater than zero
- utilization cannot be negative
- delinquency counts cannot be negative
- income cannot be negative
- number of dependents cannot be negative

Optional fields such as `MonthlyIncome` and `NumberOfDependents` can remain missing.

This prevents invalid values from silently reaching the model.

---

## 4. Reproducing Training-Time Feature Engineering

The trained model was not trained only on the original ten borrower features.

Two engineered features were also used:

```text
IncomeMissing
DelinquencyAnomaly
```

Therefore the API must reproduce the same logic during inference.

For example:

```text
MonthlyIncome = missing
        ↓
IncomeMissing = 1
```

and:

```text
Any delinquency field = 96 or 98
        ↓
DelinquencyAnomaly = 1
        ↓
96/98 converted to missing
```

This is necessary because a deployed model must receive data in the same feature representation used during training.

---

## 5. Saved Pipeline

The model is stored as:

```text
models/xgb_pipeline.joblib
```

The saved pipeline contains the preprocessing and XGBoost model together.

This is safer than manually rebuilding transformations inside the API because the same trained preprocessing logic can be reused.

The pipeline performs the required imputation before the XGBoost model receives the data.

The API additionally reorders columns using the trained feature names so that inference does not accidentally pass features in the wrong order.

---

## 6. Threshold Logic

The XGBoost model produces a probability.

For example:

```text
default_probability = 0.01798
```

The API then applies the selected threshold:

```text
threshold = 0.20
```

Conceptually:

```python
prediction = 1 if probability >= 0.20 else 0
```

The threshold of `0.20` was selected during validation as the best F1 among the tested thresholds.

It is currently an engineering/model-analysis choice, not a finalized banking policy.

---

## 7. SHAP Integration

SHAP was integrated into the backend so that the API does more than return a probability.

A prediction can now contain feature-level contributions such as:

```text
Feature
Contribution
Direction
```

This lets the frontend explain which borrower characteristics contributed most strongly to the model's output.

For example:

```text
Higher revolving utilization
        ↓
increased predicted risk
```

The explanation describes model behavior. It does not establish that the feature causally produced the outcome.

---

## 8. PostgreSQL Prediction Storage

The backend now stores every successful prediction in PostgreSQL.

The current table contains fields for:

```text
id
borrower features
default_probability
prediction
risk_level
threshold
model_version
created_at
```

The database uses parameterized SQL queries rather than constructing SQL strings from raw input.

This reduces the risk of SQL injection and keeps the database layer separate from the prediction logic.

---

## 9. Prediction History

The endpoint:

```http
GET /api/predictions
```

retrieves recent prediction records.

This allows the Streamlit application to show previously generated predictions rather than treating every API call as an isolated event.

The general flow is:

```text
Prediction request
       ↓
FastAPI
       ↓
PostgreSQL INSERT
       ↓
Prediction History
       ↓
Streamlit table
```

---

## 10. Evidently Drift Monitoring

The project also includes a monitoring layer.

The monitoring module compares:

```text
Reference data
      VS
Current/incoming data
```

The current implementation uses the training/reference dataset and the available unlabeled test/inference data.

The backend generates an HTML report and returns a summary such as:

```text
dataset_drift
drifted_columns
total_columns
share_drifted_columns
report_path
```

A useful distinction was established:

```text
Data drift
    ≠
Model accuracy
```

A feature distribution can change without us knowing whether predictions became more or less accurate.

Therefore monitoring is an early-warning mechanism rather than proof of model correctness.

---

## 11. Streamlit Frontend

The frontend now acts as a presentation layer.

It does not directly load the XGBoost model.

Instead:

```text
Streamlit
    ↓ HTTP
FastAPI
    ↓
Model
```

This makes the system easier to maintain and later deploy as separate services.

The interface currently contains three main areas:

### Prediction

Collect borrower information and display the prediction.

### Prediction History

Display recently stored prediction records from PostgreSQL.

### Monitoring

Display the drift-monitoring summary returned by the backend.

---

## 12. Docker Architecture

The application is now split into two containers.

### Backend container

```text
Python 3.13
FastAPI
XGBoost
SHAP
PostgreSQL client
Evidently
```

### Frontend container

```text
Python 3.13
Streamlit
Requests
```

Docker Compose connects them on an internal network:

```text
frontend
   │
   │ http://backend:8000
   ▼
backend
```

This is different from local development where the frontend may use:

```text
127.0.0.1:8000
```

Inside Docker, `localhost` would refer to the frontend container itself, so the Docker service name `backend` must be used.

---

## 13. Linux-Compatible Docker Dependencies

The original `requirements.txt` was generated on Windows and contained Windows-specific packages.

For example, `pywin32` cannot be installed in the Linux Python image used by Docker.

A separate file was therefore created:

```text
requirements-docker.txt
```

It contains only the runtime packages needed by the containers.

This keeps the deployment environment cleaner and avoids installing notebook/development packages unnecessarily.

---

## 14. Docker Build Verification

Docker Compose successfully built both images:

```text
loan-default-predictor-backend
loan-default-predictor-frontend
```

The Docker Engine was also verified through Docker Desktop/WSL2.

The first attempt to run the containers encountered:

```text
port 8501:
Only one usage of each socket address
```

Investigation showed that PID `9840` was an existing `python.exe` process listening on port `8501`.

The conflict was therefore with the host machine, not the Docker image.

---

## 15. Project Architecture After Day 4

The system can now be represented as:

```text
                  ┌──────────────────────┐
                  │      Streamlit       │
                  │      Frontend        │
                  │       :8501          │
                  └──────────┬───────────┘
                             │ HTTP
                             ▼
                  ┌──────────────────────┐
                  │       FastAPI        │
                  │        :8000         │
                  └──────────┬───────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
       XGBoost + SHAP   PostgreSQL       Evidently
            │             / Neon          Monitoring
            │
            ▼
       Prediction API

        MLflow
          │
          ▼
   Experiment / Model Tracking

       Docker
          │
          ▼
      AWS ECS
```

This architecture is consistent with the current SRS, which describes Streamlit, FastAPI, the ML pipeline, PostgreSQL, SHAP, Evidently, MLflow, and Docker/ECS as separate layers/components.

---

# End-of-Day Status

## Completed

- FastAPI backend
- Pydantic request validation
- XGBoost model loading
- Inference-time feature engineering
- Threshold-based prediction
- SHAP individual explanations
- PostgreSQL/Neon connection
- Prediction persistence
- Prediction history API
- Evidently drift monitoring
- Drift report generation
- Streamlit API integration
- Streamlit UI improvements
- Docker Desktop + WSL2 setup
- Backend Dockerfile
- Frontend Dockerfile
- Docker Compose configuration
- Docker runtime requirements
- Successful backend image build
- Successful frontend image build
- SRS architecture/documentation update
- GitHub push preparation

## Current Project Structure

```text
Loan-Default-Predictor/
│
├── data/
│   └── raw/
│       ├── cs-training.csv
│       └── cs-test.csv
│
├── docs/
│   ├── SRS.md
│   ├── day1_summary.md
│   └── day4.md
│
├── frontend/
│   ├── app.py
│   └── Dockerfile
│
├── models/
│   ├── xgb_pipeline.joblib
│   └── threshold.json
│
├── reports/
│   └── data_drift_report.html
│
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── model_service.py
│   ├── schema.py
│   ├── database.py
│   └── monitoring.py
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_baseline_model.ipynb
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-docker.txt
├── .dockerignore
├── .env
└── synopsis.docx
```

## Next Session

### Day 5 — Full Docker Run + Integration Testing

1. Stop the old local Streamlit process occupying port `8501`.
2. Start the complete stack with Docker Compose.
3. Verify `/api/health`.
4. Test `/api/predict`.
5. Verify predictions are inserted into PostgreSQL.
6. Verify Prediction History in Streamlit.
7. Verify Monitoring in Streamlit.
8. Test invalid API inputs.
9. Test backend/frontend failure handling.
10. Commit and push the completed Day 4 state to GitHub.
11. Begin preparing the Dockerized application for AWS ECS deployment.

---

**Current overall project status:** The project has moved from ML experimentation into an integrated application stage, with the trained model now connected to an API, database, explainability, monitoring, frontend, and containerization layers.
