from __future__ import annotations

from pathlib import Path
import math

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from model_utils import (
    EMPLOYMENT_TYPES,
    LOAN_PURPOSES,
    calculate_emi,
    make_single_applicant_features,
)

ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "artifacts" / "credit_prescreener_bundle.joblib"

st.set_page_config(
    page_title="Credit Pre-Screening | MUC02",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1240px;}
      .hero {padding: 1.35rem 1.55rem; border-radius: 18px; background: linear-gradient(120deg, #10233F 0%, #1E4B62 100%); color: white; margin-bottom: 1rem;}
      .hero-kicker {font-size: .78rem; letter-spacing: .14em; text-transform: uppercase; opacity: .82; font-weight: 700;}
      .hero h1 {font-size: 2.05rem; line-height: 1.15; margin: .35rem 0 .55rem 0; color: white;}
      .hero p {font-size: 1rem; opacity: .92; margin: 0; max-width: 900px;}
      .small-note {font-size: .85rem; color: #526273;}
      div[data-testid="stMetric"] {border: 1px solid rgba(120,130,145,.22); padding: 1rem; border-radius: 14px; background: rgba(128,128,128,.035);}
      .result-label {font-size: .84rem; text-transform: uppercase; letter-spacing: .08em; color: #637386; font-weight: 700;}
      .result-score {font-size: 2.2rem; line-height: 1.15; font-weight: 750; margin: .25rem 0 .5rem 0;}
      [data-testid="stAlert"] {border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the trained logistic regression model…")
def load_bundle():
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(
            "Model artifact is missing. In this project folder, run `python train_model.py` first."
        )
    return joblib.load(BUNDLE_PATH)


try:
    bundle = load_bundle()
except Exception as exc:  # show actionable app setup guidance, not a raw traceback
    st.error("The model could not be loaded.")
    st.code(str(exc))
    st.info("Run `python train_model.py`, confirm artifacts/credit_prescreener_bundle.joblib exists, then restart Streamlit.")
    st.stop()

model = bundle["model"]
threshold = bundle["recommended_threshold"]
metrics = bundle["metrics"]
ntc_metrics = bundle["ntc_metrics"]
income_medians = bundle["income_medians_by_employment"]
global_income_median = bundle["training_global_income_median"]

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">MUC02 · Supervised Learning · BFSI</div>
      <h1>Credit Pre-Screening Engine</h1>
      <p>A transparent decision-support demo that estimates whether an application resembles cases historically marked Declined by the credit team.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "**Academic prototype — not for real lending decisions.** This model predicts historical analyst decisions in a simulated dataset, "
    "not whether a person will repay a loan. Do not automatically approve or reject anyone from this output; keep a qualified human reviewer involved."
)

with st.sidebar:
    st.title("Workspace")
    selected_view = st.radio(
        "Navigate",
        ["Applicant pre-screen", "Model performance", "Model card & limitations"],
        index=0,
    )
    st.divider()
    st.caption("**Model version**")
    st.code(bundle.get("model_version", "unknown"), language=None)
    st.metric("Assignment threshold", f"{threshold:.2f}" if threshold is not None else "Not available")
    st.caption("Input values are processed in this app session only; this demo does not write applications to a database.")


def format_inr(value: float) -> str:
    return f"₹{value:,.0f}"


def training_range_warning(column: str, value: float | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    bounds = bundle.get("training_ranges", {}).get(column)
    if not bounds:
        return None
    low, high = bounds
    if value < low or value > high:
        return f"{column.replace('_', ' ').title()} ({value:,.2f}) is outside the observed training range ({low:,.2f}–{high:,.2f}). Predictions may be less reliable."
    return None


if selected_view == "Applicant pre-screen":
    st.subheader("Applicant information")
    st.write("Enter an illustrative application. Required model features are derived from these fields; gender, city name, application ID, and application date are not used for prediction.")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=33, step=1)
        city_tier_label = st.selectbox("City tier", ["Tier 1 — Metro", "Tier 2 — Large city", "Tier 3 — Smaller city"], index=1)
        city_tier = {"Tier 1 — Metro": 1, "Tier 2 — Large city": 2, "Tier 3 — Smaller city": 3}[city_tier_label]
    with col_b:
        employment_type = st.selectbox("Employment type", EMPLOYMENT_TYPES, index=0)
        years_in_current_job = st.number_input("Years in current job/business", min_value=0.0, max_value=60.0, value=3.0, step=0.5)
    with col_c:
        loan_purpose = st.selectbox("Loan purpose", LOAN_PURPOSES, index=0)
        loan_amount = st.number_input("Requested loan amount (₹)", min_value=50_000, max_value=2_500_000, value=300_000, step=25_000)

    st.divider()
    st.subheader("Income and existing obligations")
    income_unknown = st.checkbox("Monthly income is unavailable", value=False, help="If checked, the model uses the training median for this employment type to calculate affordability features.")
    monthly_income = None
    if income_unknown:
        imputed_preview = income_medians.get(employment_type, global_income_median)
        st.info(f"For this demo, the model will use an employment-type median of approximately {format_inr(float(imputed_preview))}/month to construct FOIR and loan-to-income. Treat this as an estimation warning, not verified income.")
    else:
        monthly_income = st.number_input("Gross monthly income (₹)", min_value=1_000, max_value=5_000_000, value=50_000, step=1_000)

    annual_income_misentered = False
    if (
        not income_unknown
        and employment_type == "Salaried"
        and monthly_income is not None
        and monthly_income > 600_000
        and math.isclose(monthly_income % 12, 0.0, abs_tol=1e-6)
    ):
        st.warning("This value matches a pattern that appeared as a possible annual/monthly unit error in the historical training data.")
        annual_income_misentered = st.checkbox("This figure is actually annual income; divide it by 12 for this assessment", value=False)
        if annual_income_misentered:
            monthly_income = float(monthly_income) / 12
            st.caption(f"Income used for this assessment: {format_inr(monthly_income)} per month.")

    col_d, col_e, col_f = st.columns(3)
    with col_d:
        existing_emi = st.number_input("Existing monthly EMIs (₹)", min_value=0, max_value=1_000_000, value=4_000, step=500)
    with col_e:
        tenure_months = st.selectbox("Requested tenure", [12, 24, 36, 48, 60], index=2, format_func=lambda x: f"{x} months")
    with col_f:
        proposed_emi = calculate_emi(float(loan_amount), int(tenure_months))
        st.metric("Estimated proposed EMI", format_inr(proposed_emi))
        st.caption("Calculated at 14% p.a.; rounded to the nearest ₹100 to match the training data's convention.")

    st.divider()
    st.subheader("Credit history")
    has_credit_score = st.checkbox("Applicant has a recorded bureau credit score", value=True)
    credit_score = None
    if has_credit_score:
        credit_score = st.slider("Credit score", min_value=300, max_value=900, value=720, step=1)
    else:
        st.warning("No recorded score selected. The app will set `is_ntc = 1`, impute the score inside the model pipeline, and show an additional human-review warning.")

    col_g, col_h = st.columns(2)
    with col_g:
        num_active_loans = st.number_input("Number of active loans", min_value=0, max_value=50, value=1, step=1)
    with col_h:
        delinquencies_24m = st.number_input("Late/missed repayments in past 24 months", min_value=0, max_value=50, value=0, step=1)

    st.caption("Gender is deliberately not requested as a model input. Fairness should be checked separately using governed, appropriate data.")
    predict_clicked = st.button("Evaluate application", type="primary", use_container_width=True)

    if predict_clicked:
        if monthly_income is not None and monthly_income <= 0:
            st.error("Monthly income must be greater than zero.")
        elif not has_credit_score and credit_score is not None:
            st.error("Credit-score inputs are inconsistent. Please review the credit-history selection.")
        else:
            applicant = {
                "age": int(age),
                "city_tier": int(city_tier),
                "employment_type": employment_type,
                "years_in_current_job": float(years_in_current_job),
                "monthly_income": None if income_unknown else float(monthly_income),
                "existing_emi": float(existing_emi),
                "credit_score": np.nan if not has_credit_score else float(credit_score),
                "no_credit_score": not has_credit_score,
                "num_active_loans": int(num_active_loans),
                "delinquencies_24m": int(delinquencies_24m),
                "loan_purpose": loan_purpose,
                "loan_amount": float(loan_amount),
                "tenure_months": int(tenure_months),
            }
            feature_row, effective_income, income_was_imputed = make_single_applicant_features(
                applicant, income_medians, float(global_income_median)
            )
            decline_probability = float(model.predict_proba(feature_row[bundle["feature_cols"]])[:, 1][0])
            triggered = threshold is not None and decline_probability >= float(threshold)
            foir_value = float(feature_row.loc[0, "foir"])
            lti_value = float(feature_row.loc[0, "loan_to_income"])

            st.divider()
            st.subheader("Pre-screen result")
            result_col, probability_col, threshold_col = st.columns([1.25, 1, 1])
            with result_col:
                if triggered:
                    st.warning("**High-confidence historical-decline signal**\n\nFlag for specialist manual review. This is not a final rejection.")
                else:
                    st.info("**Below the high-confidence flag threshold**\n\nContinue the normal human review process. This is not an approval.")
            with probability_col:
                st.metric("Predicted probability of historical Declined", f"{decline_probability:.1%}")
                st.progress(min(max(decline_probability, 0.0), 1.0))
            with threshold_col:
                st.metric("Configured assignment threshold", f"{float(threshold):.2f}" if threshold is not None else "Unavailable")
                st.caption("Threshold chosen from the seven assignment values in the original test-set analysis.")

            detail_a, detail_b, detail_c = st.columns(3)
            detail_a.metric("FOIR", f"{foir_value:.1%}")
            detail_b.metric("Loan amount / annual income", f"{lti_value:.2f}×")
            detail_c.metric("Income used for affordability", format_inr(effective_income))

            if foir_value >= 0.50:
                st.warning("FOIR is at least 50%. In the assignment's exploratory analysis, higher FOIR was associated with lower historical approval. Treat this as an enhanced-review cue, not an automatic policy rule.")
            if not has_credit_score:
                ntc_accuracy = ntc_metrics.get("ntc_accuracy")
                accuracy_text = f"{ntc_accuracy:.1%}" if ntc_accuracy is not None else "not available"
                st.error(f"**New-to-credit safeguard:** the model's test-set accuracy for this group was {accuracy_text} ({ntc_metrics.get('ntc_n', 0)} NTC test cases). Accuracy is only historical-decision agreement, but this weaker result is a strong reason to retain human review.")
            if income_was_imputed:
                st.warning("Income was unavailable and an employment-type training median was used to calculate affordability features. The output is less informative until verified income is available.")

            range_warnings = []
            for column, value in {
                "age": age,
                "years_in_current_job": years_in_current_job,
                "monthly_income": monthly_income,
                "existing_emi": existing_emi,
                "credit_score": None if not has_credit_score else credit_score,
                "num_active_loans": num_active_loans,
                "delinquencies_24m": delinquencies_24m,
                "loan_amount": loan_amount,
                "tenure_months": tenure_months,
            }.items():
                warning = training_range_warning(column, value)
                if warning:
                    range_warnings.append(warning)
            if range_warnings:
                with st.expander("Inputs outside the historical training ranges"):
                    for warning in range_warnings:
                        st.write(f"- {warning}")

            with st.expander("See the exact features passed to the model"):
                readable = feature_row.copy()
                readable["credit_score"] = readable["credit_score"].apply(lambda v: "Missing / NTC" if pd.isna(v) else f"{v:.0f}")
                st.dataframe(readable, use_container_width=True, hide_index=True)
                st.caption("The model's internal imputer fills a missing credit score using the training median; `is_ntc` preserves the missing-score indicator.")

elif selected_view == "Model performance":
    st.subheader("Held-out evaluation summary")
    st.caption("These are reference results on the assignment's 2,400-row test split. They measure agreement with historical decisions in a simulated dataset.")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Accuracy", f"{metrics['accuracy']:.1%}")
    metric_cols[1].metric("Precision · Declined", f"{metrics['precision_declined']:.1%}")
    metric_cols[2].metric("Recall · Declined", f"{metrics['recall_declined']:.1%}")
    metric_cols[3].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    st.markdown("#### Confusion matrix at the default 0.50 cutoff")
    cm_df = pd.DataFrame(
        [[metrics["tn"], metrics["fp_historically_approved_flagged"]],
         [metrics["fn_historical_declines_missed"], metrics["tp"]]],
        index=["Actual Approved", "Actual Declined"],
        columns=["Predicted Approved", "Predicted Declined"],
    )
    st.dataframe(cm_df, use_container_width=True)
    st.info(f"At 0.50, {metrics['fp_historically_approved_flagged']:,} historically Approved test applications were predicted as Declined. This is why the default cutoff is not used for the high-confidence flag.")

    st.markdown("#### Threshold trade-off")
    table = pd.DataFrame(bundle["threshold_table"])
    display_columns = [
        "threshold", "auto_declined_count", "auto_declined_percent_of_test",
        "historically_approved_wrongly_auto_declined",
        "historically_approved_share_of_auto_declines_percent", "meets_5pct_guardrail",
    ]
    st.dataframe(
        table[display_columns].rename(columns={
            "threshold": "Threshold",
            "auto_declined_count": "Auto-decline candidates",
            "auto_declined_percent_of_test": "Test set flagged (%)",
            "historically_approved_wrongly_auto_declined": "Historical approvals flagged",
            "historically_approved_share_of_auto_declines_percent": "Historical approvals among flagged (%)",
            "meets_5pct_guardrail": "Passes 5% rule",
        }).style.format({
            "Threshold": "{:.2f}",
            "Test set flagged (%)": "{:.2f}",
            "Historical approvals among flagged (%)": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    if table["threshold"].eq(float(threshold)).any():
        chart = table.set_index("threshold")[[
            "auto_declined_percent_of_test",
            "historically_approved_share_of_auto_declines_percent",
        ]].rename(columns={
            "auto_declined_percent_of_test": "Share of applications flagged (%)",
            "historically_approved_share_of_auto_declines_percent": "Historical approvals among flagged (%)",
        })
        st.line_chart(chart)
        st.caption("The 5% guardrail applies to the historical-approval share within the flagged group. The chart uses the seven thresholds required by the assignment.")

    st.markdown("#### New-to-credit performance caveat")
    ntc_col1, ntc_col2 = st.columns(2)
    ntc_col1.metric("NTC test accuracy", f"{ntc_metrics['ntc_accuracy']:.1%}", help="Agreement with historical analyst labels, not repayment performance.")
    ntc_col2.metric("Scored-applicant test accuracy", f"{ntc_metrics['scored_accuracy']:.1%}")
    st.warning("The NTC accuracy gap supports a human-review safeguard. Group accuracy is a simple diagnostic and does not, by itself, establish fairness or model suitability.")

elif selected_view == "Model card & limitations":
    st.subheader("Model card")
    st.markdown("#### Intended use")
    st.write("Educational demonstration of a logistic-regression pre-screener for a supervised-learning assignment. It provides a high-confidence historical-decline signal for discussion and review.")
    st.markdown("#### Not intended for")
    st.write("Real credit underwriting, final loan approval, automated rejection, eligibility guarantees, or predicting default / repayment. The data is simulated and the label is the historical credit team's decision.")

    st.markdown("#### Target and features")
    st.write("Target: `Declined = 1`, `Approved = 0`. The model uses credit score, new-to-credit flag, FOIR, loan-to-income, delinquency count, active loans, job tenure, age, requested tenure, city tier, employment type, and loan purpose.")
    st.write("Excluded from prediction: gender, city name, application ID, application date, and the target `decision`. Gender can be used for a separate governed fairness audit, not as a model feature in this demo.")

    st.markdown("#### Main limitations and safeguards")
    st.markdown(
        "- The dataset is simulated and covers six months of historical decisions.\n"
        "- The model imitates past analyst decisions, including potential historical bias; it does not estimate repayment probability.\n"
        "- Test accuracy for new-to-credit applicants is substantially lower than for scored applicants. Keep these cases with a human reviewer.\n"
        "- Threshold 0.90 was selected using the assignment's held-out test set. Before any real use, select thresholds on validation data and evaluate on a separate untouched test set.\n"
        "- The 5% guardrail passes overall in this test sample, but the historically approved share among Male auto-decline candidates was above 5%; investigate subgroup performance.\n"
        "- Real credit use would require representative real outcomes, independent validation, calibration, monitoring, fairness/legal review, security controls, and a documented human-appeal process."
    )
    st.markdown("#### Privacy")
    st.write("This sample app does not save the entered application to a database. Do not enter real customer personal or financial data into this public educational demo.")

st.divider()
st.caption("MUC02 Credit Pre-Screening · Logistic Regression · Simulated educational dataset · Human review required")
