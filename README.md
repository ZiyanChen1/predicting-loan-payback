# Predicting Loan Payback

Ziyan Chen · MSBA 265 Applied Analytics, University of the Pacific · Final project
Data: Kaggle Playground Series S5E11, [Predicting Loan Payback](https://www.kaggle.com/competitions/playground-series-s5e11)

## Overview
This project asks whether a machine learning model can rank borrowers by default risk better than the lender's own grade, so a lender can approve more loans without higher default losses. It compares three baselines (always "repaid", the lender's grade, and a credit score + debt-to-income scorecard) with logistic regression and LightGBM. On a held-out test set, LightGBM reaches a ROC AUC of 0.921, versus 0.658 for the lender's grade, and at the same default losses it approves 93.4% of loans instead of 80%.

**To read the analysis without installing anything,** open [`01_eda.ipynb`](01_eda.ipynb) and [`02_modeling.ipynb`](02_modeling.ipynb) on this page; GitHub shows all saved outputs. The steps below are only for rerunning the project.

## Repository structure
```
predicting-loan-payback/
├── README.md
├── requirements.txt          
├── .gitignore               
├── 01_eda.ipynb              data audit and exploratory analysis, with interpretations
├── 02_modeling.ipynb         60/20/20 split, baselines, models, test-set checks, business impact, risk tiers
├── src/
│   ├── eda_audit.py          reproducible audit and cleaning pipeline
│   └── train.py              trains the final model and saves it to artifacts/
├── tests/
│   └── test_model.py         reloads the saved model and checks its predictions
├── artifacts/
│   ├── model.joblib          saved model: encoding + LightGBM in one pipeline
│   └── model_metadata.json   input columns, output meaning, tier cutoffs, versions, reference predictions
├── examples/
│   └── example_applicants.csv  five invented applicants for testing (not Kaggle data)
├── data/
│   └── download_data.py      downloads the Kaggle data with your own account (data files are not tracked)
└── reports/                  figures and result tables used in the paper and slides
```

## How to reproduce

### 1. Set up the environment
Python 3.11 or later.

**Windows (PowerShell)**
```
git clone https://github.com/ZiyanChen1/predicting-loan-payback.git
cd predicting-loan-payback
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
If PowerShell says "running scripts is disabled on this system", run this once, then activate again:
```
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

**Mac (Terminal)**
```
git clone https://github.com/ZiyanChen1/predicting-loan-payback.git
cd predicting-loan-payback
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the data (not included)
The Kaggle competition rules do not allow sharing the competition data, so **no data files are stored in this repository**. Each user downloads them with their own Kaggle account:

1. Sign in to Kaggle and accept the competition rules: https://www.kaggle.com/competitions/playground-series-s5e11/rules
2. Log in to the Kaggle API (one time; opens a browser):
   ```
   kaggle auth login
   ```
3. Download the files into `data/`:
   ```
   python data/download_data.py
   ```

### 3. Run the project
With the environment active, type the commands below **in this order** in any terminal opened in the project folder (PowerShell, Mac Terminal, or the terminal inside VS Code). Each part uses the files written by the part before it. All results are saved to files, so no browser or VS Code window is needed.

#### 3.1 Data audit script: `src/eda_audit.py`
```
python src/eda_audit.py
```
Expected results:
- A printed audit summary: 593,994 training rows and 254,569 test rows; 0 missing values, duplicates, boundary violations and category issues.
- Outlier and skewness tables, and one highly correlated pair (credit score vs. grade, r = −0.91).
- Two EDA charts in `reports/` (payback by DTI decile and by category), and `train_clean.csv` / `test_clean.csv` in `data/`.

#### 3.2 EDA notebook: `01_eda.ipynb`
```
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb
```
This runs every cell and saves the results inside the notebook.

Expected results (the notebook should finish without errors):
- Data shape, column types, and the target cast from float to integer
- Missing-value, duplicate, boundary and category checks
- Outliers (Tukey 1.5 × IQR) with box plots
- Skewness before and after the log transform
- Correlation heatmap
- Charts: target split, payback by employment status, by debt-to-income decile and by grade
- Business data dictionary (`reports/data_dictionary.csv`) and a check of the cleaned columns

#### 3.3 Modeling notebook: `02_modeling.ipynb`
It takes several minutes.
```
jupyter nbconvert --to notebook --execute --inplace 02_modeling.ipynb
```
Expected results:

| Output | File | What to check |
|---|---|---|
| Model results (5-fold CV and test set) | `reports/test_results.csv`, `reports/roc_curve_test.png` | LightGBM test ROC AUC ≈ 0.921 vs. 0.658 (lender's grade) |
| Risk drivers | `reports/lgbm_feature_importance.png` | Top three: employment status, debt-to-income, credit score |
| Confusion matrix and calibration | `reports/confusion_matrix_test.png`, `reports/calibration_test.png` | Recall (not repaid) 60.8%; probabilities within 1 point of actual payback |
| Fairness by gender | `reports/fairness_gender.csv` | Female vs. male ROC AUC 0.922 vs. 0.920 |
| Business impact | `reports/business_impact.png` | ≈ $97.2M (LightGBM) vs. $242.0M (lender's grade) at 80% kept |
| Same losses, more lending | `reports/same_loss_more_lending.csv` | LightGBM approves 93.4% vs. 80.0% at $242.0M defaulted principal |
| Risk tiers | `reports/risk_tiers.csv`, `reports/tier_sensitivity.csv` | Default rates 3.7% / 15.3% / 66.8% on the test set; all nine scenarios positive |
| Kaggle submission | `data/submission.csv` (not tracked) | One payback probability per test `id` |

#### 3.4 Saved model: `src/train.py` and `tests/test_model.py`
```
python src/train.py
python tests/test_model.py
```
`train.py` trains the final LightGBM model on the same 60% training set as the notebook and saves it to `artifacts/model.joblib` (about 10 seconds). `test_model.py` reloads that file in a fresh run and scores the five invented applicants in `examples/example_applicants.csv`.

Expected results:
- `train.py` prints `Test ROC AUC: 0.9207`, the same as the notebook.
- `test_model.py` prints one line per applicant (A and E approve, B review, C and D decline) and ends with `All checks passed.`

**Use the saved model in your own code:**
```python
import joblib, pandas as pd
model = joblib.load("artifacts/model.joblib")
applicants = pd.read_csv("examples/example_applicants.csv")
model.predict_proba(applicants)[:, 1]   # probability of payback
```
Input: the 10 columns listed in `artifacts/model_metadata.json` (raw values, e.g. `grade_subgrade = "C3"`). Output: probability that the loan is paid back. Tiers: decline below 0.77, review 0.77 – 0.89, approve at or above 0.89.

### 4. View or rerun the notebooks (optional)
Step 3 already saved every table and chart inside the notebooks. Use this step only to look at them, or to rerun cells one at a time.

**Option A: VS Code**
1. In the terminal (environment active, in the project folder), open VS Code:
   ```
   code .
   ```
   The `.` means "this folder". Open the project folder itself, not `src/`, so the paths `data/` and `reports/` work.
   If `code` is not recognized:
   - **Windows:** in VS Code, use **File → Open Folder** and pick the project folder.
   - **Mac:** in VS Code, press `Cmd+Shift+P` and run **Shell Command: Install 'code' command in PATH**, then open a new terminal.
2. Install the **Python** and **Jupyter** extensions by Microsoft (Extensions panel: `Ctrl+Shift+X`, Mac `Cmd+Shift+X`). One time only.
3. Open a notebook → click **Select Kernel** in the upper-right corner → **Python Environments** → `venv`.
4. Read the saved outputs, or click **Run All** to rerun.
5. To run a `.py` file: press `Ctrl+Shift+P` (Mac `Cmd+Shift+P`) → **Python: Select Interpreter** → `venv`, open the file, and click **▶ Run Python File** in the upper-right corner.

**Option B: GitHub (nothing to install)**
Open `01_eda.ipynb` or `02_modeling.ipynb` on the repository page. GitHub shows the saved tables, charts and interpretations. Viewing only; it cannot rerun cells.
