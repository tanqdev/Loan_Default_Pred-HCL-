# Loan Default Prediction System

**Project domain:** Banking / Credit Risk  
**Use-case context:** Bank Muscat (academic prototype)  
**Institution:** ABES Engineering College (AKTU)  
**Project type:** Machine Learning + Web Application + MLOps prototype

## Overview
An end-to-end application prototype that estimates serious borrower delinquency risk from borrower credit-profile attributes. The model uses the public **Give Me Some Credit** dataset. Bank Muscat is the use-case context only; the project does not use proprietary Bank Muscat customer data.

## Key results
- 150,000 labeled training rows; positive class prevalence approximately 6.68%.
- Best documented ROC-AUC: **0.8693** (XGBoost validation run).
- Corresponding documented PR-AUC: **0.4100**.
- API threshold: **0.20**, selected for maximum F1 among tested validation thresholds; not a bank-approved policy.

## Features
- EDA, data cleaning, missing-value handling and delinquency anomaly feature engineering.
- Logistic Regression, Random Forest and XGBoost experiments.
- SHAP explanations for individual model predictions.
- FastAPI endpoints: `/api/health`, `/api/predict`, `/api/predictions`, `/api/monitoring/drift`.
- PostgreSQL/Neon prediction persistence.
- Streamlit Prediction, Prediction History and Monitoring views.
- MLflow experiment/model tracking.
- Evidently feature drift report at `reports/data_drift_report.html`.
- Docker and Docker Compose local deployment configuration.

## Tech stack
Python 3.13, Pandas, NumPy, scikit-learn, XGBoost, SHAP, FastAPI, Pydantic, PostgreSQL, MLflow, Evidently, Streamlit, Docker, Docker Compose.

## Run locally
```powershell
.venv\Scripts\Activate.ps1
uvicorn src.main:app --reload
```
In a second terminal:
```powershell
streamlit run frontend/app.py
```
Docker:
```powershell
docker compose up --build
```

Frontend: `http://localhost:8501`  
API docs: `http://localhost:8000/docs`

Set `DATABASE_URL` securely in the environment. Never commit `.env` or credentials.

## Important limitations
- The supplied `cs-test.csv` target is unlabeled and is for inference only; it cannot be used to calculate local ROC-AUC.
- Metrics above are validation results, not production results.
- SHAP is descriptive, not causal.
- Drift detection does not imply that predictive performance has degraded.
- AWS ECS deployment is planned, not claimed as completed.
- This academic prototype must not be used as an autonomous lending decision system.

## Repository
https://github.com/tanqdev/Loan_Default_Pred-HCL-
