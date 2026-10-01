import os

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in the .env file")


def get_connection():
    return psycopg.connect(DATABASE_URL)


def save_prediction(data: dict, result: dict):
    query = """
        INSERT INTO predictions (
            revolving_utilization,
            age,
            number_of_time_30_59_days_past_due,
            debt_ratio,
            monthly_income,
            number_of_open_credit_lines_and_loans,
            number_of_times_90_days_late,
            number_real_estate_loans_or_lines,
            number_of_time_60_89_days_past_due,
            number_of_dependents,
            default_probability,
            prediction,
            risk_level,
            threshold,
            model_version
        )
        VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s
        )
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                query,
                (
                    data["RevolvingUtilizationOfUnsecuredLines"],
                    data["age"],
                    data["NumberOfTime30_59DaysPastDueNotWorse"],
                    data["DebtRatio"],
                    data["MonthlyIncome"],
                    data["NumberOfOpenCreditLinesAndLoans"],
                    data["NumberOfTimes90DaysLate"],
                    data["NumberRealEstateLoansOrLines"],
                    data["NumberOfTime60_89DaysPastDueNotWorse"],
                    data["NumberOfDependents"],
                    result["default_probability"],
                    result["prediction"],
                    result["risk_level"],
                    result["threshold"],
                    "xgb_baseline_v1"
                )
            )

def get_predictions(limit: int = 20):
    query = """
        SELECT
            id,
            revolving_utilization,
            age,
            monthly_income,
            default_probability,
            prediction,
            risk_level,
            threshold,
            model_version,
            created_at
        FROM predictions
        ORDER BY created_at DESC
        LIMIT %s
    """

    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query, (limit,))
            return cur.fetchall()
        

if __name__ == "__main__":
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT NOW();")
            print("Database connected:", cur.fetchone()[0])