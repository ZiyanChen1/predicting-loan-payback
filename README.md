# Predicting Loan Payback

MSBA 265 Final Project · Ziyan Chen
Data: Kaggle Playground Series S5E11, [Predicting Loan Payback](https://www.kaggle.com/competitions/playground-series-s5e11)

## Project question
Lenders already rank borrowers with their own risk grade. This project asks:

> **Can machine learning rank approved borrowers by default risk better than the lender's own grade, by enough to matter, while staying explainable?**

- **Stakeholder:** a lender's credit risk team.
- **How the result is used:** to group similar future applicants into low, medium, and high risk tiers, and to explain the main drivers of each borrower's default risk.
- **Success measures:**
  1. Cross-validated ROC AUC above the lender's grade (0.655) and a classic credit score + DTI scorecard (0.768).
  2. Less defaulted principal than the lender's grade when only the safest share of loans is kept.
  3. Borrower-level risk drivers that a loan officer can read.

## Current status (P1)
| Stage | File | Status |
|---|---|---|
| Data quality audit and cleaning | `src/eda_audit.py`, `01_eda.ipynb` | Done |
| Baseline and first models | `02_modeling.ipynb` | Done (proof of concept) |
| Risk drivers, risk tiers, sensitivity | `02_modeling.ipynb` Steps 7c–7d | First version done |
| Tuning, saved model, fairness check | — | Next (P2) |

P1 proposal (topic examples, infographic, proposal, EDA, baseline): `docs/P1_Proposal_Ziyan_Chen.pdf`

## Setup
Python 3.11 or later.

```
git clone https://github.com/ZiyanChen1/msba265-final-project.git
cd msba265-final-project
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac / Linux
pip install -r requirements.txt
```

## Data (not included in this repository)
The Kaggle competition rules do not allow sharing the competition data with people outside the competition,
so **no data files are stored here**. To reproduce the project:

1. Sign in to Kaggle and open the competition [Data page](https://www.kaggle.com/competitions/playground-series-s5e11/data).
2. Click **Join Competition / I Understand and Accept** to accept the rules.
3. Download `train.csv`, `test.csv`, and `sample_submission.csv` and put them in the `data/` folder:
   ```
   data/train.csv
   data/test.csv
   data/sample_submission.csv
   ```
   (Or, with the Kaggle CLI: `kaggle competitions download -c playground-series-s5e11 -p data` and unzip.)

`data/*.csv` is listed in `.gitignore`, so data files and generated files (cleaned data, submission) are never uploaded.

- **One row** = one loan that was already approved, with the borrower's finances, loan terms, and profile.
- **Target:** `loan_paid_back` (1 = repaid, 0 = not repaid). About 80% of loans were repaid.
- **Size:** 593,994 training rows and 254,569 test rows; no missing values or duplicates.
- **Limits:** the data is synthetic (generated from an original loan dataset), so results show direction rather than real-bank dollars. It contains approved loans only; declined applicants are not observed (reject inference).

## How to run (in this order)
1. `python src/eda_audit.py`: audit and clean the data
   - writes audit tables and figures to `reports/`
   - writes `data/train_clean.csv` and `data/test_clean.csv`
2. `01_eda.ipynb`: exploratory data analysis with interpretations (Run All)
3. `02_modeling.ipynb`: models, business impact, risk drivers, risk tiers, Kaggle submission (Run All; about 7 minutes)

## Expected outputs
| Output | Where | Key result |
|---|---|---|
| Model comparison | `reports/model_comparison.csv` | ROC AUC: baseline 0.500 · lender's grade 0.655 · scorecard 0.768 · logistic regression 0.908 · LightGBM 0.922 |
| Business impact | `reports/business_impact.csv`, `.png` | Keeping the safest 80% of approved loans: $97.1M defaulted principal with LightGBM vs. $242.0M with the lender's grade |
| Risk tiers and sensitivity | `reports/tier_sensitivity.csv` | Default rate 3.7% (low) · 15.4% (medium) · 67.1% (high) |
| Kaggle submission | `data/submission.csv` | Payback probability for each test `id` |

## Project structure
```
data/                 put the Kaggle files here (not tracked; see data/README.md)
docs/                 P1 proposal PDF
reports/              audit tables, figures, model results
src/eda_audit.py      reproducible audit and cleaning pipeline
01_eda.ipynb          exploratory data analysis
02_modeling.ipynb     models, business impact, risk drivers, risk tiers
requirements.txt      Python dependencies
```

## Key EDA findings
- 79.9% of loans were repaid, so the classes are imbalanced; ROC AUC is used instead of accuracy.
- Employment status is the strongest signal: only 7.8% of unemployed borrowers repaid, versus 89.4% of employed.
- Payback drops sharply in the top two DTI deciles (from about 78% to 49%), so DTI risk is nonlinear.
- Credit score and loan grade are highly correlated (r = −0.91), so logistic regression keeps only credit score.
- Outliers were kept: high-DTI outliers are real high-risk borrowers (45% payback vs. 81% for others).

## Known limitations
- Synthetic data: dollar amounts are illustrative.
- Data is not redistributed: reviewers need a Kaggle account to download it.
- Approved loans only: model explanations are risk drivers, not real decline reasons.
- Cost assumptions for the risk tiers (loss given default, review cost, review capacity) are stated assumptions, tested with a sensitivity analysis.
