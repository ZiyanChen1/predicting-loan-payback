import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def tier(prob, cutoffs):
    if prob >= cutoffs["approve_at_or_above"]:
        return "Low risk: approve"
    if prob >= cutoffs["decline_below"]:
        return "Medium risk: review"
    return "High risk: decline"


def main():
    model = joblib.load(ROOT / "artifacts" / "model.joblib")
    metadata = json.loads((ROOT / "artifacts" / "model_metadata.json").read_text())
    examples = pd.read_csv(ROOT / "examples" / "example_applicants.csv")

    probs = model.predict_proba(examples[metadata["input_columns"]])[:, 1]
    reference = np.array([metadata["reference_predictions"][a] for a in examples["applicant"]])

    assert np.all((probs >= 0) & (probs <= 1)), "probabilities must be between 0 and 1"
    assert np.allclose(probs, reference, atol=1e-6), "reloaded model does not match the reference predictions"

    for applicant, p in zip(examples["applicant"], probs):
        print(f"Applicant {applicant}: payback probability {p:.3f} -> {tier(p, metadata['tier_cutoffs'])}")

    try:
        model.predict_proba(examples[metadata["input_columns"][:-1]])
        raise AssertionError("a missing input column should raise an error")
    except ValueError as error:
        print(f"Missing column rejected: {type(error).__name__}")

    print("All checks passed.")


if __name__ == "__main__":
    main()
