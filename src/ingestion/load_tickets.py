"""
load_tickets.py

Phase 3 - Data Engineering for the support ticket dataset
Same extract -> transform -> load pattern as load_data.py in Phase 1,
just for a different dataset.

Run it with:  uv run python src/ingestion/load_tickets.py
"""

from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "support_tickets.csv"
DB_PATH = PROJECT_ROOT / "data" / "processed" / "insighthub.db"


def extract(csv_path: Path) -> pd.DataFrame:
    print(f"Reading raw data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
    print(f"Columns: {list(df.columns)}")
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize column names: lowercase, spaces -> underscores
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Drop rows with no description or no ticket type — can't train on those
    before = len(df)
    df = df.dropna(subset=["ticket_description", "ticket_type"])
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} rows with missing description/type.")

    # Drop exact duplicates
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} duplicate rows.")

    return df


def load(df: pd.DataFrame, db_path: Path, table_name: str = "support_tickets") -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    df.to_sql(table_name, con=engine, if_exists="replace", index=False)
    print(f"Wrote {len(df)} rows into table '{table_name}' at: {db_path}")


def main():
    df = extract(RAW_CSV_PATH)
    df = transform(df)

    print("\nTicket type distribution:")
    print(df["ticket_type"].value_counts())

    load(df, DB_PATH)
    print("Done. Support ticket ingestion complete.")


if __name__ == "__main__":
    main()
