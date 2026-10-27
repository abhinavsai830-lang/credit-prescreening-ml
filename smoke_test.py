"""Quick test that the saved model bundle can make representative predictions."""
from pathlib import Path
import joblib
import numpy as np

from model_utils import FEATURE_COLS, make_single_applicant_features

ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "artifacts" / "credit_prescreener_bundle.joblib"


def main():
    bundle = joblib.load(BUNDLE_PATH)
    assert bundle["recommended_threshold"] == 0.90
    assert set(["gender", "city", "application_id", "application_date", "decision"]).isdisjoint(FEATURE_COLS)

    examples = [
        {
            "age": 33, "city_tier": 2, "employment_type": "Salaried", "years_in_current_job": 3.0,
            "monthly_income": 50000, "existing_emi": 4000, "credit_score": 720, "no_credit_score": False,
            "num_active_loans": 1, "delinquencies_24m": 0, "loan_purpose": "Debt Consolidation",
            "loan_amount": 300000, "tenure_months": 36,
        },
        {
            "age": 26, "city_tier": 3, "employment_type": "Gig / Contract", "years_in_current_job": 1.0,
            "monthly_income": None, "existing_emi": 2000, "credit_score": None, "no_credit_score": True,
            "num_active_loans": 0, "delinquencies_24m": 0, "loan_purpose": "Education",
            "loan_amount": 150000, "tenure_months": 36,
        },
    ]
    for idx, applicant in enumerate(examples, start=1):
        X, used_income, imputed = make_single_applicant_features(
            applicant,
            bundle["income_medians_by_employment"],
            bundle["training_global_income_median"],
        )
        probability = float(bundle["model"].predict_proba(X[FEATURE_COLS])[:, 1][0])
        assert np.isfinite(probability) and 0 <= probability <= 1
        assert len(X) == 1 and list(X.columns) == FEATURE_COLS
        print(f"Example {idx}: p(historical Declined)={probability:.3f}; used income=₹{used_income:,.0f}; imputed_income={imputed}")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
