import json
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "train_clean.csv"
EXAMPLES = ROOT / "examples" / "example_applicants.csv"
OUT_DIR = ROOT / "artifacts"

TARGET = "loan_paid_back"
NUM = ["annual_income", "debt_to_income_ratio", "credit_score", "loan_amount", "interest_rate"]
CAT = ["marital_status", "education_level", "employment_status", "loan_purpose"]
GRADES = [f"{letter}{num}" for letter in "ABCDEF" for num in range(1, 6)]
INPUT_COLUMNS = NUM + ["grade_subgrade"] + CAT
CUTOFFS = {"decline_below": 0.77, "approve_at_or_above": 0.89}


def build_pipeline(categories):
    prep = ColumnTransformer(
        [
            ("num", "passthrough", NUM),
            ("grade", OrdinalEncoder(categories=[GRADES], handle_unknown="use_encoded_value", unknown_value=np.nan), ["grade_subgrade"]),
            ("cat", OrdinalEncoder(categories=categories, handle_unknown="use_encoded_value", unknown_value=np.nan), CAT),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    model = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
    return Pipeline([("prep", prep), ("model", model)])


def tier(prob):
    if prob >= CUTOFFS["approve_at_or_above"]:
        return "Low risk: approve"
    if prob >= CUTOFFS["decline_below"]:
        return "Medium risk: review"
    return "High risk: decline"


def main():
    data = pd.read_csv(DATA)
    train_df, test_df = train_test_split(data, test_size=0.20, stratify=data[TARGET], random_state=42)
    train_df, _ = train_test_split(train_df, test_size=0.25, stratify=train_df[TARGET], random_state=42)

    categories = [sorted(train_df[col].unique()) for col in CAT]
    pipeline = build_pipeline(categories)
    pipeline.fit(train_df[INPUT_COLUMNS], train_df[TARGET], model__categorical_feature=[6, 7, 8, 9])

    test_auc = roc_auc_score(test_df[TARGET], pipeline.predict_proba(test_df[INPUT_COLUMNS])[:, 1])

    examples = pd.read_csv(EXAMPLES)
    example_probs = pipeline.predict_proba(examples[INPUT_COLUMNS])[:, 1]

    OUT_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, OUT_DIR / "model.joblib")

    metadata = {
        "model": "LightGBM in a scikit-learn Pipeline (encoding + model)",
        "trained_on": f"{len(train_df):,} loans (60% training set, random_state=42)",
        "input_columns": INPUT_COLUMNS,
        "output": "predict_proba(...)[:, 1] = probability that the loan is paid back",
        "tier_cutoffs": CUTOFFS,
        "test_roc_auc": round(test_auc, 4),
        "library_versions": {"scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__,
                             "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
        "reference_predictions": {
            row.applicant: round(float(p), 6) for row, p in zip(examples.itertuples(), example_probs)
        },
    }
    (OUT_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Saved {OUT_DIR / 'model.joblib'}")
    print(f"Test ROC AUC: {test_auc:.4f}")
    for row, p in zip(examples.itertuples(), example_probs):
        print(f"Applicant {row.applicant}: payback probability {p:.3f} -> {tier(p)}")


if __name__ == "__main__":
    main()
