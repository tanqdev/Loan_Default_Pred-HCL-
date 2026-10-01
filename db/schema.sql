CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,

    revolving_utilization DOUBLE PRECISION NOT NULL,
    age DOUBLE PRECISION NOT NULL,

    number_of_time_30_59_days_past_due INTEGER,
    debt_ratio DOUBLE PRECISION NOT NULL,
    monthly_income DOUBLE PRECISION,

    number_of_open_credit_lines_and_loans INTEGER NOT NULL,
    number_of_times_90_days_late INTEGER,
    number_real_estate_loans_or_lines INTEGER,
    number_of_time_60_89_days_past_due INTEGER,

    number_of_dependents DOUBLE PRECISION,

    default_probability DOUBLE PRECISION NOT NULL,
    prediction INTEGER NOT NULL,
    risk_level VARCHAR(20) NOT NULL,

    threshold DOUBLE PRECISION NOT NULL,
    model_version VARCHAR(50) NOT NULL,

    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
