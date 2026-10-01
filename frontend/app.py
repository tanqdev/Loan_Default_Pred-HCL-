import os

import requests
import streamlit as st


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Loan Default Risk Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1rem;
        opacity: 0.7;
        margin-bottom: 2rem;
    }

    .risk-card {
        padding: 1.5rem;
        border-radius: 14px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        text-align: center;
        margin-bottom: 1rem;
    }

    .risk-value {
        font-size: 2.4rem;
        font-weight: 700;
    }

    .risk-label {
        font-size: 1rem;
        opacity: 0.7;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown("## 🏦 Loan Risk Platform")

    st.caption(
        "ML-powered credit risk assessment"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Prediction",
            "Prediction History",
            "Monitoring"
        ]
    )

    st.divider()

    st.caption("Model")
    st.write("XGBoost")

    st.caption("Validation ROC-AUC")
    st.write("0.8693")

    st.caption("Selected Threshold")
    st.write("0.20")


# ==================================================
# HEADER
# ==================================================

st.markdown(
    '<div class="main-title">'
    'Loan Default Risk Platform'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Assess borrower default risk using the trained XGBoost model.'
    '</div>',
    unsafe_allow_html=True
)


# ==================================================
# PREDICTION PAGE
# ==================================================

if page == "Prediction":

    st.subheader("Borrower Assessment")

    st.write(
        "Enter the borrower's credit profile below."
    )

    col1, col2 = st.columns(2)

    # --------------------------------------------------
    # LEFT COLUMN
    # --------------------------------------------------

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=45
        )

        monthly_income = st.number_input(
            "Monthly Income",
            min_value=0.0,
            value=50000.0,
            step=500.0
        )

        debt_ratio = st.number_input(
            "Debt Ratio",
            min_value=0.0,
            value=0.35,
            step=0.01
        )

        revolving_utilization = st.number_input(
            "Revolving Utilization",
            min_value=0.0,
            value=0.25,
            step=0.01
        )

        number_of_dependents = st.number_input(
            "Number of Dependents",
            min_value=0,
            value=2
        )

    # --------------------------------------------------
    # RIGHT COLUMN
    # --------------------------------------------------

    with col2:

        open_credit_lines = st.number_input(
            "Open Credit Lines and Loans",
            min_value=0,
            value=8
        )

        past_due_30_59 = st.number_input(
            "30–59 Days Past Due",
            min_value=0,
            value=0
        )

        past_due_60_89 = st.number_input(
            "60–89 Days Past Due",
            min_value=0,
            value=0
        )

        times_90_late = st.number_input(
            "90+ Days Late",
            min_value=0,
            value=0
        )

        real_estate_loans = st.number_input(
            "Real Estate Loans / Lines",
            min_value=0,
            value=1
        )

    st.divider()

    predict_clicked = st.button(
        "🔍 Assess Default Risk",
        type="primary",
        use_container_width=True
    )

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    if predict_clicked:

        payload = {
            "RevolvingUtilizationOfUnsecuredLines":
                revolving_utilization,

            "age":
                age,

            "NumberOfTime30_59DaysPastDueNotWorse":
                past_due_30_59,

            "DebtRatio":
                debt_ratio,

            "MonthlyIncome":
                monthly_income,

            "NumberOfOpenCreditLinesAndLoans":
                open_credit_lines,

            "NumberOfTimes90DaysLate":
                times_90_late,

            "NumberRealEstateLoansOrLines":
                real_estate_loans,

            "NumberOfTime60_89DaysPastDueNotWorse":
                past_due_60_89,

            "NumberOfDependents":
                number_of_dependents
        }

        try:

            with st.spinner(
                "Running risk assessment..."
            ):

                response = requests.post(
                    f"{API_URL}/api/predict",
                    json=payload,
                    timeout=30
                )

            response.raise_for_status()

            result = response.json()

            probability = result["default_probability"]
            prediction = result["prediction"]
            risk_level = result["risk_level"]
            threshold = result["threshold"]

            # Save result so it survives Streamlit reruns
            st.session_state["last_result"] = result

            # --------------------------------------------------
            # RESULT
            # --------------------------------------------------

            st.subheader("Assessment Result")

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:

                st.markdown(
                    f"""
                    <div class="risk-card">
                        <div class="risk-value">
                            {probability * 100:.2f}%
                        </div>
                        <div class="risk-label">
                            Default Probability
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with result_col2:

                if prediction == 1:

                    st.error(
                        f"⚠️ {risk_level} Risk"
                    )

                else:

                    st.success(
                        f"✅ {risk_level} Risk"
                    )

            with result_col3:

                st.metric(
                    "Decision Threshold",
                    f"{threshold:.2f}"
                )

            # --------------------------------------------------
            # PROBABILITY BAR
            # --------------------------------------------------

            st.write("Risk Probability")

            st.progress(
                min(max(probability, 0.0), 1.0)
            )

            st.caption(
                f"Classification threshold: {threshold:.2f}"
            )

            # --------------------------------------------------
            # SHAP
            # --------------------------------------------------

            st.divider()

            st.subheader(
                "Why did the model make this prediction?"
            )

            st.caption(
                "These are the features with the largest "
                "influence on this individual prediction."
            )

            explanations = result.get(
                "explanations",
                []
            )

            if explanations:

                for item in explanations:

                    feature = item["feature"]
                    value = item["value"]
                    contribution = item["contribution"]
                    direction = item["direction"]

                    if contribution > 0:
                        icon = "🔴"
                    else:
                        icon = "🟢"

                    col1, col2 = st.columns(
                        [3, 2]
                    )

                    with col1:

                        st.write(
                            f"{icon} **{feature}**"
                        )

                    with col2:

                        st.write(
                            f"{direction} "
                            f"({contribution:+.4f})"
                        )

                    st.caption(
                        f"Input value: {value}"
                    )

                    st.divider()

            else:

                st.info(
                    "No explanation data was returned."
                )

        except requests.exceptions.RequestException as e:

            st.error(
                f"Unable to connect to the prediction API: {e}"
            )


# ==================================================
# PREDICTION HISTORY
# ==================================================

elif page == "Prediction History":

    st.subheader("Prediction History")

    try:

        response = requests.get(
            f"{API_URL}/api/predictions?limit=100",
            timeout=10
        )

        response.raise_for_status()

        history = response.json()

        if not history:

            st.info(
                "No prediction history available yet."
            )

        else:

            display_data = []

            for row in history:

                display_data.append(
                    {
                        "ID": row["id"],
                        "Age": row["age"],
                        "Monthly Income": row["monthly_income"],
                        "Default Probability":
                            f"{row['default_probability'] * 100:.2f}%",
                        "Prediction": row["prediction"],
                        "Risk": row["risk_level"],
                        "Model": row["model_version"],
                        "Created": row["created_at"]
                    }
                )

            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=True
            )

    except requests.exceptions.RequestException as e:

        st.error(
            f"Unable to load prediction history: {e}"
        )


# ==================================================
# MONITORING
# ==================================================

elif page == "Monitoring":

    st.subheader("📊 Data Drift Monitoring")

    st.write(
        "Compare current borrower data with the "
        "reference dataset used during model development."
    )

    if st.button(
        "🔄 Run Drift Analysis",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Running Evidently drift analysis..."
            ):

                response = requests.get(
                    f"{API_URL}/api/monitoring/drift",
                    timeout=120
                )

            response.raise_for_status()

            drift = response.json()

            if drift["dataset_drift"]:

                st.error(
                    "⚠️ Significant data drift detected"
                )

            else:

                st.success(
                    "✅ No significant data drift detected"
                )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Drifted Columns",
                    drift["drifted_columns"]
                )

            with col2:

                st.metric(
                    "Total Columns",
                    drift["total_columns"]
                )

            with col3:

                st.metric(
                    "Drift Share",
                    f"{drift['share_drifted_columns'] * 100:.1f}%"
                )

            st.divider()

            st.write("Generated report:")

            st.code(
                drift["report_path"]
            )

        except requests.exceptions.RequestException as e:

            st.error(
                f"Unable to run drift monitoring: {e}"
            )