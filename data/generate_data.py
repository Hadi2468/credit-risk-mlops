import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N_USERS = 2000
os.makedirs("data/raw", exist_ok=True)


# 1 users: each user has a hidden risk level between 0 and 1
users = pd.DataFrame({
    "user_id": np.arange(1, N_USERS + 1),
    "age": rng.integers(21, 65, N_USERS),
    "monthly_income": rng.normal(3200, 900, N_USERS).clip(800).round(0),
    "risk": rng.beta(2, 5, N_USERS),
})


# 2 payments: riskier users pay later and miss more often
rows = []
for u in users.itertuples():
    n_payments = rng.integers(3, 13)
    days = rng.integers(0, 270, n_payments)
    for day in days:
        payment_date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(day))
        missed = rng.random() < 0.25 * u.risk
        delay_days = None if missed else int(rng.poisson(1 + 14 * u.risk))
        rows.append((u.user_id, payment_date, delay_days))
payments = pd.DataFrame(rows, columns=["user_id", "payment_date", "delay_days"])
payments["payment_date"] = payments["payment_date"].astype("datetime64[us]")

# primary key: one unique id per payment, as the first column
payments.insert(0, "payment_id", range(1, len(payments) + 1))
assert payments["payment_id"].is_unique, "payment_id must be unique"


# 3 loans: one application per user, riskier users default more often
application_days = rng.integers(0, 210, N_USERS)
loans = pd.DataFrame({
    "loan_id": np.arange(1, N_USERS + 1),
    "user_id": users["user_id"],
    "application_date": pd.Timestamp("2026-04-01") + pd.to_timedelta(application_days, unit="D"),
    "loan_amount": rng.choice([100, 200, 300, 400, 500], N_USERS),
})
loans["application_date"] = loans["application_date"].astype("datetime64[us]")
p_default = (0.05 + 0.6 * users["risk"]).clip(0, 0.95)
loans["defaulted"] = (rng.random(N_USERS) < p_default).astype(int)


# 4 save: the hidden risk column is NOT saved, the model must learn it from behavior
users.drop(columns="risk").to_parquet("data/raw/users.parquet", index=False)
payments.to_parquet("data/raw/payments.parquet", index=False)
loans.to_parquet("data/raw/loans.parquet", index=False)


print("users   ", users.shape)
print("payments", payments.shape)
print("loans   ", loans.shape)
print("default rate:", loans["defaulted"].mean().round(3))
print(users.head())
print(payments.head())
print(loans.head())