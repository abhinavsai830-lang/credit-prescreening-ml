"""Train the MUC02 demo model and save its inference bundle.

Run from this project folder:
    python train_model.py
"""
from pathlib import Path
import json

import joblib

from model_utils import train_bundle

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "MUC02_Credit_PreScreening_Dataset.csv"
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "credit_prescreener_bundle.joblib"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Place the provided CSV in the data/ folder."
        )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    bundle, threshold_table = train_bundle(DATA_PATH)
    joblib.dump(bundle, ARTIFACT_PATH, compress=3)

    print(f"Saved model bundle: {ARTIFACT_PATH}")
    print("Model metrics:")
    print(json.dumps(bundle["metrics"], indent=2))
    print("New-to-credit / scored group metrics:")
    print(json.dumps(bundle["ntc_metrics"], indent=2))
    print("Threshold comparison:")
    print(threshold_table.to_string(index=False))
    print(f"Recommended threshold: {bundle['recommended_threshold']}")


if __name__ == "__main__":
    main()
