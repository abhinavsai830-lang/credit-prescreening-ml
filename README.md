# MUC02 Credit Pre-Screening — Streamlit Demo

A Streamlit interface for the MUC02 simulated credit pre-screening assignment. It uses the same logistic-regression feature definitions and the required 0.90 threshold selected in the notebook analysis.

> **Important:** This is an educational prototype, not a real lending tool. It predicts the historical analyst decision in a simulated dataset, not whether an applicant will repay. It must not automatically approve or reject applicants. A qualified human must review any flagged case.

## Project files

```text
MUC02_Streamlit_Credit_PreScreening/
├── app.py
├── model_utils.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── artifacts/
│   └── credit_prescreener_bundle.joblib
└── data/
    └── MUC02_Credit_PreScreening_Dataset.csv
```

The app loads the pre-trained bundle in `artifacts/` and does not need the CSV at runtime. The CSV is included only so the model can be reproduced locally.

## Run locally on Windows

1. Extract the ZIP file.
2. Open a terminal/PowerShell in the extracted `MUC02_Streamlit_Credit_PreScreening` folder.
3. Create and activate a virtual environment:

   ```powershell
   py -3.13 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If Python 3.13 is not installed, install a Python version compatible with the pinned dependency wheels, then update the model/dependency versions together if needed.
4. Install packages:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. (Optional/reproducibility) Rebuild the model artifact using the provided simulated CSV:

   ```powershell
   python train_model.py
   ```
6. Start Streamlit:

   ```powershell
   streamlit run app.py
   ```
7. Open the local URL printed in the terminal, normally `http://localhost:8501`.

## Deploy free with Streamlit Community Cloud

1. Create a new GitHub repository, e.g. `muc02-credit-prescreener`.
2. Upload/push the files and folders in this project to the repository root. Include `app.py`, `model_utils.py`, `requirements.txt`, `.streamlit/config.toml`, and `artifacts/credit_prescreener_bundle.joblib`. Keep `data/` if you want retraining to be reproducible; the live app does not need it to make predictions.
3. Open <https://share.streamlit.io/> and sign in with GitHub.
4. Select **Create app**, choose the repository and `main` branch, and set the app file path to `app.py`.
5. In **Advanced settings**, select Python 3.13 to match the environment used to build the included model artifact (if that version is available in the deployment selector).
6. Click **Deploy**. If you change dependency versions, retrain/rebuild the artifact under the same versions before pushing.
7. After deployment, test with fictional data only and inspect the app logs if startup fails.

Official deployment instructions: <https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy>

## Model logic preserved from the notebook

- Target is `Declined = 1`, `Approved = 0`.
- Stratified 80/20 split with `random_state=42` for the reproduced baseline.
- Income unit correction on historical data follows the assignment's documented rule; missing income medians are learned on training data only.
- Derived features: `foir`, `loan_to_income`, `is_ntc`.
- Numeric imputation/scaling and categorical imputation/one-hot encoding are in a scikit-learn pipeline.
- Features exclude gender, city name, application ID, application date and decision.
- The UI's 0.90 threshold is the assignment's maximum-coverage threshold among the seven tested values that met the overall 5% historical-approval guardrail in the notebook's test sample. It is not a production-certified threshold.

## Safety and limitations

- The sample is simulated and its target represents historical analyst choices, not default or repayment.
- The assignment's overall 5% threshold result does not guarantee that every subgroup meets 5%; the notebook identified a Male subgroup result requiring further investigation.
- New-to-credit performance was materially weaker in the test set, so the app explicitly warns and requires human review for such cases.
- Never upload real customer information to this public demo. A real system needs independent validation, representative repayment outcomes, privacy/security assessment, fair-lending/legal review, monitoring, and human appeal/override processes.
