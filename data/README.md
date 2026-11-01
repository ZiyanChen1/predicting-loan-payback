# Data folder

The competition data is **not** stored in this repository, because the Kaggle competition rules
do not allow sharing it with people outside the competition.

Download it yourself:

1. Sign in to Kaggle and accept the rules of
   [Predicting Loan Payback (Playground Series S5E11)](https://www.kaggle.com/competitions/playground-series-s5e11/data).
2. Download `train.csv`, `test.csv`, and `sample_submission.csv` into this folder.

After running `python src/eda_audit.py`, this folder will also contain `train_clean.csv` and `test_clean.csv`;
`02_modeling.ipynb` writes `submission.csv` here. All `.csv` files in this folder are ignored by git.
