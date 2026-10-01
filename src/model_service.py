from pathlib import Path
import json

import joblib
import pandas as pd
import shap


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "xgb_pipeline.joblib"
THRESHOLD_PATH = BASE_DIR / "models" / "threshold.json"


# Load trained pipeline once when the API starts
model = joblib.load(MODEL_PATH)

# Load threshold
with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
    threshold_config = json.load(f)

THRESHOLD = float(threshold_config["threshold"])


# Extract the XGBoost model and preprocessing step
xgb_model = model.named_steps["model"]
imputer = model.named_steps["imputer"]

# Create SHAP explainer once
explainer = shap.TreeExplainer(xgb_model)


FEATURE_LABELS = {
    "RevolvingUtilizationOfUnsecuredLines": "Revolving Utilization",
    "age": "Age",
    "NumberOfTime30-59DaysPastDueNotWorse": "30–59 Days Past Due",
    "DebtRatio": "Debt Ratio",
    "MonthlyIncome": "Monthly Income",
    "NumberOfOpenCreditLinesAndLoans": "Open Credit Lines and Loans",
    "NumberOfTimes90DaysLate": "90+ Days Late",
    "NumberRealEstateLoansOrLines": "Real Estate Loans / Lines",
    "NumberOfTime60-89DaysPastDueNotWorse": "60–89 Days Past Due",
    "NumberOfDependents": "Number of Dependents",
    "IncomeMissing": "Income Missing",
    "DelinquencyAnomaly": "Delinquency Anomaly",
}


def predict_default(data: dict) -> dict:
    """
    Convert API input into the format expected by the ML pipeline,
    generate the prediction, and calculate SHAP contributions.
    """

    # Original delinquency values
    past_due_30_59 = data["NumberOfTime30_59DaysPastDueNotWorse"]
    past_due_60_89 = data["NumberOfTime60_89DaysPastDueNotWorse"]
    past_due_90 = data["NumberOfTimes90DaysLate"]

    # Detect delinquency anomalies
    delinquency_anomaly = int(
        past_due_30_59 in [96, 98]
        or past_due_60_89 in [96, 98]
        or past_due_90 in [96, 98]
    )

    # Replace anomalous values with missing values
    if past_due_30_59 in [96, 98]:
        past_due_30_59 = None

    if past_due_60_89 in [96, 98]:
        past_due_60_89 = None

    if past_due_90 in [96, 98]:
        past_due_90 = None

    # Missing-income indicator
    income_missing = int(data["MonthlyIncome"] is None)

    model_input = {
        "RevolvingUtilizationOfUnsecuredLines":
            data["RevolvingUtilizationOfUnsecuredLines"],

        "age":
            data["age"],

        "NumberOfTime30-59DaysPastDueNotWorse":
            past_due_30_59,

        "DebtRatio":
            data["DebtRatio"],

        "MonthlyIncome":
            data["MonthlyIncome"],

        "NumberOfOpenCreditLinesAndLoans":
            data["NumberOfOpenCreditLinesAndLoans"],

        "NumberOfTimes90DaysLate":
            past_due_90,

        "NumberRealEstateLoansOrLines":
            data["NumberRealEstateLoansOrLines"],

        "NumberOfTime60-89DaysPastDueNotWorse":
            past_due_60_89,

        "NumberOfDependents":
            data["NumberOfDependents"],

        "IncomeMissing":
            income_missing,

        "DelinquencyAnomaly":
            delinquency_anomaly,
    }

    # Create one-row DataFrame
    df = pd.DataFrame([model_input])

    # Match the exact feature order used during training
    df = df.reindex(columns=model.feature_names_in_)

    # Prediction probability
    probability = float(
        model.predict_proba(df)[0, 1]
    )

    # Apply selected threshold
    prediction = int(probability >= THRESHOLD)

    # Risk level
    risk_level = "High" if prediction == 1 else "Low"

    # ---------------------------------
    # SHAP explanation
    # ---------------------------------

    # Apply the same imputer used inside the pipeline
    X_transformed = imputer.transform(df)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_transformed)

    # Handle different SHAP return formats
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    contributions = shap_values[0]

    explanations = []

    for feature, contribution in zip(
        model.feature_names_in_,
        contributions
    ):
        actual_value = df.iloc[0][feature]

        if pd.isna(actual_value):
            actual_value = None
        elif hasattr(actual_value, "item"):
            actual_value = actual_value.item()

        contribution = float(contribution)

        explanations.append({
            "feature": FEATURE_LABELS.get(feature, feature),
            "value": actual_value,
            "contribution": round(contribution, 6),
            "direction": (
                "increases risk"
                if contribution > 0
                else "decreases risk"
            )
        })

    # Show the most influential features first
    explanations.sort(
        key=lambda x: abs(x["contribution"]),
        reverse=True
    )

    # Return top 5 contributors
    explanations = explanations[:5]

    return {
        "default_probability": probability,
        "prediction": prediction,
        "risk_level": risk_level,
        "threshold": THRESHOLD,
        "explanations": explanations
    }
