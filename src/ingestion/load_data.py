"""
load_data.py

Phase 1 - Data Engineering
This script does the simplest possible ETL job:
  1. EXTRACT: read the raw CSV file
  2. TRANSFORM: do a little light cleaning
  3. LOAD: write it into a real SQL database (SQLite, stored as a file)

Run it with:  uv run python src/ingestion/load_data.py
"""

from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine

# ── 1. Define paths (relative to the project root) ─────────────────
# Path(__file__) = this file's location
# .resolve()      = turn it into a full absolute path
# .parents[2]     = go up 2 folders (src/ingestion -> src -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"
DB_PATH = PROJECT_ROOT / "data" / "processed" / "insighthub.db"


def extract(csv_path: Path) -> pd.DataFrame:
    """Read the raw CSV into a pandas DataFrame."""
    print(f"Reading raw data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Light cleaning before it goes into the database."""
    # Standardize column names: lowercase, no spaces
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # TotalCharges sometimes has blank strings instead of numbers in this dataset
    # coerce = turn anything that isn't a valid number into NaN instead of crashing
    if "totalcharges" in df.columns:
        df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")

    # Drop exact duplicate rows if any exist
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} duplicate rows.")

    return df


def load(df: pd.DataFrame, db_path: Path, table_name: str = "customers") -> None:
    """Write the cleaned DataFrame into a SQLite database file."""
    # Make sure the processed/ folder exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # SQLite connection string: sqlite:///path/to/file.db
    engine = create_engine(f"sqlite:///{db_path}")

    df.to_sql(table_name, con=engine, if_exists="replace", index=False)
    print(f"Wrote {len(df)} rows into table '{table_name}' at: {db_path}")


def main():
    df = extract(RAW_CSV_PATH)
    df = transform(df)
    load(df, DB_PATH)
    print("Done. Phase 1 ingestion complete.")


if __name__ == "__main__":
    main()
