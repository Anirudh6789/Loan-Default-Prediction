"""
Generates a realistic synthetic loan default dataset.
"""
import numpy as np
import pandas as pd

np.random.seed(7)
N = 6000

age = np.random.randint(21, 65, N)
annual_income = np.round(np.random.normal(650000, 250000, N).clip(150000, 2000000), 0)
loan_amount = np.round(np.random.normal(400000, 200000, N).clip(50000, 1500000), 0)
credit_score = np.random.randint(300, 900, N)
employment_type = np.random.choice(["Salaried", "Self-Employed", "Business"], N, p=[0.55, 0.25, 0.20])
existing_loans = np.random.poisson(1.0, N)
loan_term_months = np.random.choice([12, 24, 36, 48, 60], N)
missed_payments_last_year = np.random.poisson(0.6, N)

debt_to_income = loan_amount / annual_income

default_logit = (
    -3.0
    + 3.5 * (debt_to_income > 0.8)
    - 0.004 * (credit_score - 600)
    + 0.5 * missed_payments_last_year
    + 0.3 * existing_loans
    - 0.3 * (employment_type == "Salaried")
    + 0.6 * (employment_type == "Self-Employed")
    + 0.15 * (loan_term_months / 12)
)
default_prob = 1 / (1 + np.exp(-default_logit))
defaulted = np.random.binomial(1, default_prob)

df = pd.DataFrame({
    "age": age,
    "annual_income": annual_income,
    "loan_amount": loan_amount,
    "credit_score": credit_score,
    "employment_type": employment_type,
    "existing_loans": existing_loans,
    "loan_term_months": loan_term_months,
    "missed_payments_last_year": missed_payments_last_year,
    "defaulted": defaulted,
})

df.to_csv("data/loan_data.csv", index=False)
print(f"Generated {len(df)} rows. Default rate: {df['defaulted'].mean():.2%}")
