from pathlib import Path

import numpy as np
import pandas as pd

from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset


BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "data" / "raw" / "cs-training.csv"
TEST_PATH = BASE_DIR / "data" / "raw" /"cs-test.csv"

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)


FEATURE_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same deterministic cleaning used during model development.
    """

    df = df.copy()

    # Remove ID column
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Missing income indicator
    df["IncomeMissing"] = (
        df["MonthlyIncome"].isna().astype(int)
    )

    # Delinquency anomaly indicator
    delinquency_columns = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]

    df["DelinquencyAnomaly"] = (
        df[delinquency_columns]
        .isin([96, 98])
        .any(axis=1)
        .astype(int)
    )

    # Replace anomaly values with missing
    df[delinquency_columns] = (
        df[delinquency_columns]
            .replace([96, 98], np.nan)
    )   

    df.loc[df["age"] == 0, "age"] = np.nan

    # Ensure all monitored columns are numeric
    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def run_drift_report():
    # Load reference and current data
    reference_raw = pd.read_csv(TRAIN_PATH)
    current_raw = pd.read_csv(TEST_PATH)

    # Apply same deterministic cleaning
    reference = clean_data(reference_raw)
    current = clean_data(current_raw)

    # Keep only model input features
    reference = reference[FEATURE_COLUMNS]
    current = current[FEATURE_COLUMNS]

    # Define feature types
    data_definition = DataDefinition(
        numerical_columns=FEATURE_COLUMNS
    )

    # Convert pandas DataFrames into Evidently datasets
    reference_dataset = Dataset.from_pandas(
        reference,
        data_definition=data_definition
    )

    current_dataset = Dataset.from_pandas(
        current,
        data_definition=data_definition
    )

    # Create drift report
    report = Report([
        DataDriftPreset()
    ])

    # Compare current data against reference data
    evaluation = report.run(
        current_dataset,
        reference_dataset
    )

    # Save HTML report
    report_path = REPORT_DIR / "data_drift_report.html"

    evaluation.save_html(str(report_path))

    # Get report results
    result = evaluation.dict()

    # Find the DriftedColumnsCount metric
    drift_metric = None

    for metric in result.get("metrics", []):
        metric_name = metric.get("metric_name", "")

        if metric_name.startswith("DriftedColumnsCount"):
            drift_metric = metric
            break

    if drift_metric is None:
        raise RuntimeError(
            "Could not find DriftedColumnsCount in Evidently results."
        )

    # Extract count and share
    drift_value = drift_metric.get("value", {})

    drifted_columns = int(drift_value.get("count", 0))
    share_drifted_columns = float(
        drift_value.get("share", 0.0)
    )

    total_columns = len(FEATURE_COLUMNS)

    dataset_drift = share_drifted_columns >= 0.5

    summary = {
        "dataset_drift": dataset_drift,
        "drifted_columns": drifted_columns,
        "total_columns": total_columns,
        "share_drifted_columns": share_drifted_columns,
        "report_path": str(report_path),
    }

    print("Drift report generated:")
    print(report_path)

    return summary

if __name__ == "__main__":
    run_drift_report()