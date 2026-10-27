"""Training and inference helpers for the MUC02 Streamlit demo."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_VERSION = "muc02-logistic-prescreener-v1.0"

FEATURE_COLS = [
    "credit_score",
    "is_ntc",
    "foir",
    "loan_to_income",
    "delinquencies_24m",
    "num_active_loans",
    "years_in_current_job",
    "age",
    "tenure_months",
    "city_tier",
    "employment_type",
    "loan_purpose",
]
NUMERIC_FEATURES = [
    "credit_score",
    "is_ntc",
    "foir",
    "loan_to_income",
    "delinquencies_24m",
    "num_active_loans",
    "years_in_current_job",
    "age",
    "tenure_months",
    "city_tier",
]
CATEGORICAL_FEATURES = ["employment_type", "loan_purpose"]
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]

EMPLOYMENT_TYPES = [
    "Salaried",
    "Self-Employed Professional",
    "Business Owner",
    "Gig / Contract",
]
LOAN_PURPOSES = [
    "Debt Consolidation",
    "Home Renovation",
    "Wedding",
    "Education",
    "Consumer Durables",
    "Medical",
    "Travel",
]


def correct_documented_income_units(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the assignment's documented correction on a copy of the source data.

    For historical training records only: suspected annual income is corrected when
    employment is Salaried, the reported monthly income is above ₹600,000, and the
    value is divisible by 12. Live applicant values are never silently corrected.
    """
    clean = frame.copy()
    income = pd.to_numeric(clean["monthly_income"], errors="coerce")
    mask = (
        clean["employment_type"].eq("Salaried")
        & income.gt(600_000)
        & np.isclose(income.fillna(0) % 12, 0)
    )
    clean["income_unit_suspected"] = mask
    clean["monthly_income_original"] = income
    clean.loc[mask, "monthly_income"] = income.loc[mask] / 12
    clean["is_ntc"] = clean["credit_score"].isna().astype(int)
    return clean


def add_model_features(
    partition: pd.DataFrame,
    income_medians_by_employment: dict[str, float],
    training_global_income_median: float,
) -> pd.DataFrame:
    """Create model features using income medians learned from training data only."""
    frame = partition.copy()
    employment_medians = frame["employment_type"].map(income_medians_by_employment)
    frame["monthly_income_model"] = (
        pd.to_numeric(frame["monthly_income"], errors="coerce")
        .fillna(employment_medians)
        .fillna(training_global_income_median)
    )
    frame["foir"] = (
        frame["existing_emi"] + frame["proposed_emi"]
    ) / frame["monthly_income_model"]
    frame["loan_to_income"] = frame["loan_amount"] / (
        frame["monthly_income_model"] * 12
    )
    frame["is_ntc"] = frame["credit_score"].isna().astype(int)
    return frame


def calculate_emi(loan_amount: float, tenure_months: int, annual_rate: float = 0.14) -> float:
    """Calculate monthly EMI at 14% p.a. and round to nearest ₹100, as in the data."""
    principal = float(loan_amount)
    months = int(tenure_months)
    monthly_rate = annual_rate / 12
    if principal <= 0 or months <= 0:
        raise ValueError("Loan amount and tenure must be positive.")
    if monthly_rate == 0:
        raw_emi = principal / months
    else:
        factor = (1 + monthly_rate) ** months
        raw_emi = principal * monthly_rate * factor / (factor - 1)
    return float(round(raw_emi / 100) * 100)


def make_single_applicant_features(
    applicant: dict[str, Any],
    income_medians_by_employment: dict[str, float],
    training_global_income_median: float,
) -> tuple[pd.DataFrame, float, bool]:
    """Create a one-row feature frame matching the training feature definitions.

    Returns (model_features, effective_income_for_ratios, income_was_imputed).
    Gender, city name, ID, application date, and historical decision are not used.
    """
    employment_type = str(applicant["employment_type"])
    raw_income = applicant.get("monthly_income")
    income_was_imputed = raw_income is None or pd.isna(raw_income)

    if income_was_imputed:
        effective_income = income_medians_by_employment.get(
            employment_type, training_global_income_median
        )
    else:
        effective_income = float(raw_income)

    if not np.isfinite(effective_income) or effective_income <= 0:
        raise ValueError("Monthly income used for feature engineering must be positive.")

    no_credit_score = bool(applicant.get("no_credit_score", False))
    score_value = np.nan if no_credit_score else float(applicant["credit_score"])
    proposed_emi = calculate_emi(
        applicant["loan_amount"], applicant["tenure_months"]
    )
    foir = (float(applicant["existing_emi"]) + proposed_emi) / effective_income
    loan_to_income = float(applicant["loan_amount"]) / (effective_income * 12)

    row = {
        "credit_score": score_value,
        "is_ntc": int(no_credit_score),
        "foir": float(foir),
        "loan_to_income": float(loan_to_income),
        "delinquencies_24m": int(applicant["delinquencies_24m"]),
        "num_active_loans": int(applicant["num_active_loans"]),
        "years_in_current_job": float(applicant["years_in_current_job"]),
        "age": int(applicant["age"]),
        "tenure_months": int(applicant["tenure_months"]),
        "city_tier": int(applicant["city_tier"]),
        "employment_type": employment_type,
        "loan_purpose": str(applicant["loan_purpose"]),
    }
    feature_frame = pd.DataFrame([row], columns=FEATURE_COLS)
    return feature_frame, float(effective_income), bool(income_was_imputed)


def train_bundle(csv_path: str | Path) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit the notebook-aligned logistic model and return bundle and threshold table."""
    raw = pd.read_csv(csv_path)
    required = set([
        "credit_score", "delinquencies_24m", "num_active_loans", "years_in_current_job",
        "age", "tenure_months", "city_tier", "employment_type", "loan_purpose",
        "monthly_income", "existing_emi", "proposed_emi", "loan_amount", "decision",
    ])
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    model_source = correct_documented_income_units(raw)
    y_all = model_source["decision"].map({"Approved": 0, "Declined": 1})
    if y_all.isna().any():
        raise ValueError("Unexpected values in decision; expected Approved or Declined.")
    y_all = y_all.astype(int)

    train_idx, test_idx = train_test_split(
        model_source.index,
        test_size=0.20,
        stratify=y_all,
        random_state=42,
    )
    train_base = model_source.loc[train_idx].copy()
    test_base = model_source.loc[test_idx].copy()
    y_train = y_all.loc[train_idx]
    y_test = y_all.loc[test_idx]

    learned_income_medians = train_base.groupby("employment_type")["monthly_income"].median()
    income_medians = {str(k): float(v) for k, v in learned_income_medians.items()}
    global_income_median = float(train_base["monthly_income"].median())

    train_features = add_model_features(train_base, income_medians, global_income_median)
    test_features = add_model_features(test_base, income_medians, global_income_median)

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    model.fit(train_features[FEATURE_COLS], y_train)

    y_pred = model.predict(test_features[FEATURE_COLS])
    y_proba = model.predict_proba(test_features[FEATURE_COLS])[:, 1]
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_declined": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall_declined": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "tn": int(cm[0, 0]),
        "fp_historically_approved_flagged": int(cm[0, 1]),
        "fn_historical_declines_missed": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "train_applications": int(len(train_base)),
        "test_applications": int(len(test_base)),
    }

    y_test_arr = y_test.to_numpy()
    test_records = []
    for threshold in THRESHOLDS:
        mask = y_proba >= threshold
        selected_count = int(mask.sum())
        wrong = int(((y_test_arr == 0) & mask).sum())
        wrong_share = wrong / selected_count if selected_count else np.nan
        test_records.append({
            "threshold": float(threshold),
            "auto_declined_count": selected_count,
            "auto_declined_percent_of_test": selected_count / len(y_test) * 100,
            "historically_approved_wrongly_auto_declined": wrong,
            "historically_approved_share_of_auto_declines_percent": wrong_share * 100 if np.isfinite(wrong_share) else np.nan,
            "meets_5pct_guardrail": bool(selected_count > 0 and np.isfinite(wrong_share) and wrong_share <= 0.05),
        })
    threshold_table = pd.DataFrame(test_records)
    eligible = threshold_table.loc[threshold_table["meets_5pct_guardrail"]].sort_values(
        ["auto_declined_count", "threshold"], ascending=[False, False]
    )
    if eligible.empty:
        chosen_threshold = None
        selected = None
    else:
        selected = eligible.iloc[0]
        chosen_threshold = float(selected["threshold"])

    # Group accuracy is useful for the app's limitations panel.
    test_is_ntc = test_features["is_ntc"].to_numpy().astype(bool)
    scored_mask = ~test_is_ntc
    ntc_mask = test_is_ntc
    y_pred_array = np.asarray(y_pred)
    ntc_metrics = {
        "ntc_n": int(ntc_mask.sum()),
        "ntc_accuracy": float(accuracy_score(y_test_arr[ntc_mask], y_pred_array[ntc_mask])) if ntc_mask.any() else None,
        "scored_n": int(scored_mask.sum()),
        "scored_accuracy": float(accuracy_score(y_test_arr[scored_mask], y_pred_array[scored_mask])) if scored_mask.any() else None,
    }

    # Store raw training ranges to signal predictions made outside the observed data ranges.
    range_source = train_base.copy()
    range_source["monthly_income"] = pd.to_numeric(range_source["monthly_income"], errors="coerce")
    input_columns = [
        "age", "years_in_current_job", "monthly_income", "existing_emi",
        "credit_score", "num_active_loans", "delinquencies_24m", "loan_amount", "tenure_months",
    ]
    training_ranges = {}
    for column in input_columns:
        series = range_source[column].dropna()
        if not series.empty:
            training_ranges[column] = [float(series.min()), float(series.max())]

    threshold_records = threshold_table.replace({np.nan: None}).to_dict(orient="records")
    bundle = {
        "model": model,
        "model_version": MODEL_VERSION,
        "feature_cols": FEATURE_COLS,
        "income_medians_by_employment": income_medians,
        "training_global_income_median": global_income_median,
        "recommended_threshold": chosen_threshold,
        "metrics": metrics,
        "ntc_metrics": ntc_metrics,
        "threshold_table": threshold_records,
        "training_ranges": training_ranges,
        "training_cleaning_rule": "Salaried + reported income > ₹600,000 + value divisible by 12 => divide by 12 for historical training records only.",
        "target_description": "Historical analyst decision: Declined=1, Approved=0; not repayment/default prediction.",
    }
    return bundle, threshold_table
