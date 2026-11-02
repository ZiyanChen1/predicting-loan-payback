# Predicting Loan Payback

**Can machine learning rank approved borrowers by default risk better than the lender's own grade, by enough to matter, while staying explainable?**

Ziyan Chen · MSBA 265 Applied Analytics, University of the Pacific · Final project
Data: Kaggle Playground Series S5E11, [Predicting Loan Payback](https://www.kaggle.com/competitions/playground-series-s5e11)

---

## Why this project
Lenders already score borrowers. In this data, every loan carries the lender's own risk grade (A1 to F5), yet about 1 in 5 approved loans was not repaid. This project treats the lender's grade as the **current model (champion)** and tests whether machine learning is a better **challenger**. It measures the answer in dollars, not only in accuracy.

| | |
|---|---|
| **Stakeholder** | A consumer lender's credit risk team: the credit risk manager and the loan officers who review borderline files |
| **Label → action** | `loan_paid_back` (1 = repaid, 0 = not repaid) → route similar future applicants: approve / review / reject |
| **Method** | Binary classification. Shortlist for a table: logistic regression + a tree ensemble (LightGBM), compared with three baselines |
| **Success measures** | ROC AUC and recall above every baseline; lower operational cost (less defaulted principal); readable risk drivers |

## Key results (preliminary)
Mean of stratified 5-fold cross-validation on the training data:

| Model | ROC AUC | Recall on loans not repaid |
|---|---|---|
| Majority baseline: always predict "repaid" | 0.500 | 0.0% |
| Lender's grade only (company baseline) | 0.655 | 1.8% |
| Classic scorecard: credit score + DTI | 0.768 | 19.3% |
| Logistic regression | 0.908 | 59.8% |
| **LightGBM** | **0.922** | **60.9%** |

- **Dollar impact:** if the lender kept only the safest 80% of its approved loans, ranking by LightGBM would leave **$97.1M** of defaulted principal, versus **$242.0M** ranking by the lender's grade, about 60% less. The gap holds at 70% and 90%. (20% held-out validation set, 118,799 loans.)
- **Risk drivers:** the riskiest borrowers share the same top drivers: unemployed, high debt-to-income ratio, and low credit score.
- **Approve / review / reject:** cutoffs chosen with business costs give default rates of **3.7% / 15.4% / 67.1%**. Net value stays positive in all nine cost scenarios tested.

Recall is at the default 0.5 threshold. The majority baseline is 79.9% accurate but catches no defaults. Dollar figures come from synthetic data: the direction matters more than the exact amount.

## Approach
1. **Audit:** schema, target, missing values, duplicates, ranges, categories, outliers (Tukey 1.5 × IQR).
2. **Explore:** payback rates by category and DTI decile, correlations, skewness.
3. **Prepare:** learn the preparation once, from training rows, and reuse it (ColumnTransformer → Pipeline → cross-validation). Log transforms for income and DTI; one-hot encoding for small categories; ordinal grades (A1 = 1 … F5 = 30); gender excluded.
4. **Model:** majority baseline, lender's grade, classic scorecard → logistic regression → LightGBM.
5. **Evaluate:** fit on training data, choose on validation, report on untouched test data. Stratified 5-fold CV for ROC AUC and recall; a fixed 20% validation set for the dollar analysis and cutoffs.
6. **Decide:** defaulted principal at 70 / 80 / 90% of loans kept; risk drivers from LightGBM feature contributions; approve / review / reject cutoffs chosen with business costs, plus a sensitivity check.

## Repository structure
```
predicting-loan-payback/
├── README.md
├── requirements.txt          Python dependencies (pinned versions)
├── .gitignore                keeps data files out of the repository
├── 01_eda.ipynb              data audit and exploratory analysis, with interpretations
├── 02_modeling.ipynb         baselines, models, business impact, risk drivers, risk tiers
├── src/
│   └── eda_audit.py          reproducible audit and cleaning pipeline
├── data/
│   └── README.md             where to download the data (data files are not tracked)
├── reports/                  audit tables, figures, and model results
└── docs/
    └── P1_Proposal_Ziyan_Chen.pdf   project proposal (P1)
```

## How to reproduce

### 1. Set up the environment
Python 3.11 or later.

```
git clone https://github.com/ZiyanChen1/predicting-loan-payback.git
cd predicting-loan-payback
python -m venv venv
venv\Scripts\activate          # Windows (PowerShell)
source venv/bin/activate       # Mac / Linux
pip install -r requirements.txt
```

### 2. Get the data (not included)
The Kaggle competition rules do not allow sharing the competition data with people outside the competition, so **no data files are stored in this repository**.

1. Sign in to Kaggle and open the competition's [Data page](https://www.kaggle.com/competitions/playground-series-s5e11/data).
2. Accept the competition rules.
3. Download `train.csv`, `test.csv`, and `sample_submission.csv` into the `data/` folder.

Or, with the Kaggle CLI: `kaggle competitions download -c playground-series-s5e11 -p data`, then unzip.

### 3. Run, in this order
| Step | Command or file | What it does |
|---|---|---|
| 1 | `python src/eda_audit.py` | Audits and cleans the data; writes audit tables and figures to `reports/` and `train_clean.csv` / `test_clean.csv` to `data/` |
| 2 | `01_eda.ipynb` (Run All) | Exploratory analysis with interpretations |
| 3 | `02_modeling.ipynb` (Run All, several minutes) | Baselines, models, business impact, risk drivers, risk tiers, Kaggle submission |

### 4. Expected outputs
| Output | File | What to check |
|---|---|---|
| Model comparison | `reports/model_comparison.csv` | LightGBM ROC AUC ≈ 0.922 |
| Business impact | `reports/business_impact.csv`, `reports/business_impact.png` | ≈ $97.1M (LightGBM) vs. $242.0M (lender's grade) at 80% kept |
| Risk tier sensitivity | `reports/tier_sensitivity.csv` | Nine scenarios, all with positive net value |
| Kaggle submission | `data/submission.csv` (not tracked) | One payback probability per test `id` |

## Data
- **One row** = one loan that was already approved: the borrower's finances, loan terms, and profile.
- **Target:** `loan_paid_back` (1 = repaid, 0 = not repaid); 79.9% repaid.
- **Size:** 593,994 training rows and 254,569 test rows; no missing values or duplicates.
- **Key fields:** `employment_status`, `debt_to_income_ratio`, `credit_score`, `grade_subgrade` (the lender's grade), `annual_income`, `loan_amount`, `interest_rate`, plus marital status, education, loan purpose, and gender.
- A full data dictionary is in `reports/data_dictionary.csv`.

## Key EDA findings
- Classes are imbalanced (80 / 20), so ROC AUC is used instead of accuracy.
- Employment status is the strongest signal: 7.8% of unemployed borrowers repaid, versus 89.4% of employed borrowers.
- Debt-to-income risk is nonlinear: payback falls from about 78% to 49% across the top three DTI deciles.
- Credit score and grade are highly correlated (r = −0.91), so logistic regression keeps only credit score.
- Outliers were kept: high-DTI outliers are real high-risk borrowers (45% payback vs. 81% for everyone else).

## Limitations
- **Synthetic data:** the rows were generated from an original loan dataset, so dollar amounts are illustrative.
- **Approved loans only:** declined applicants are not observed (reject inference). Model explanations are risk drivers, not real decline reasons.
- **Assumed costs:** the risk tiers use stated assumptions (loss given default, review cost, review capacity), tested with a sensitivity analysis.
- **Data access:** reviewers need a Kaggle account to download the data.

## Project status
| Stage | Status |
|---|---|
| P1: proposal, data audit, EDA, baseline and first models | ✅ Done |
| P2: confusion matrix, final cutoffs, calibration and fairness checks, saved model with a load-and-test demo, FastAPI + Docker service, draft paper and slides | ⏳ Next |
| Final delivery | ⏳ |
