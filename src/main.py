from fastapi import FastAPI
from fastapi import FastAPI, Query
from src.model_service import predict_default
from src.schemas import BorrowerInput
from src.database import save_prediction, get_predictions
from src.monitoring import run_drift_report

app = FastAPI(
    title="Loan Default Prediction API",
    description="API for predicting borrower default risk.",
    version="1.0.0"
)


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy"
    }

@app.get("/api/predictions")
def prediction_history(
    limit: int = Query(default=20, ge=1, le=100)
):
    return get_predictions(limit)

@app.get("/api/monitoring/drift")
def drift_monitoring():
    return run_drift_report()


@app.post("/api/predict")
def predict(borrower: BorrowerInput):
    data = borrower.model_dump()

    # Generate ML prediction
    result = predict_default(data)

    # Save prediction to PostgreSQL
    save_prediction(data, result)

    # Return prediction to API caller
    return result