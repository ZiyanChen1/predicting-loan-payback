"""
Download the Kaggle competition data into the data/ folder.

The competition rules do not allow sharing the data, so each user downloads it
with their own Kaggle account.

Before running (one time):
  1. Sign in to Kaggle and accept the rules on the competition page:
     https://www.kaggle.com/competitions/playground-series-s5e11/rules
  2. Log in to the Kaggle API from the terminal (opens a browser):
         kaggle auth login

Usage (from the project folder):
    python data/download_data.py
"""
from pathlib import Path
import zipfile

COMPETITION = "playground-series-s5e11"
DATA_DIR = Path(__file__).resolve().parent
FILES = ["train.csv", "test.csv", "sample_submission.csv"]


def main() -> None:
    if all((DATA_DIR / f).exists() for f in FILES):
        print(f"Data already in {DATA_DIR}; nothing to download.")
        return

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise SystemExit("The Kaggle package is missing. Run: pip install -r requirements.txt")

    api = KaggleApi()
    api.authenticate()  # if not logged in, prints how to log in and stops

    print(f"Downloading {COMPETITION} to {DATA_DIR} ...")
    try:
        api.competition_download_files(COMPETITION, path=str(DATA_DIR), quiet=False)
    except Exception as error:
        raise SystemExit(f"Download failed ({error}). Have you accepted the competition rules? See step 1.")

    zip_path = DATA_DIR / f"{COMPETITION}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(DATA_DIR)
        zip_path.unlink()

    missing = [f for f in FILES if not (DATA_DIR / f).exists()]
    if missing:
        raise SystemExit(f"Download finished, but these files are missing: {missing}")
    print("Done:", ", ".join(FILES))


if __name__ == "__main__":
    main()
