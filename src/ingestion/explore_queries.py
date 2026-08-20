"""
explore_queries.py

Phase 1 - SQL practice
Runs a handful of real business questions as SQL queries against
the customers table we created in load_data.py.

Run it with:  uv run python src/ingestion/explore_queries.py
"""

from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "processed" / "insighthub.db"

engine = create_engine(f"sqlite:///{DB_PATH}")


def run_query(label: str, sql: str):
    """Run a SQL query and pretty-print the result."""
    print(f"\n--- {label} ---")
    with engine.connect() as conn:
        result = pd.read_sql(text(sql), conn)
    print(result.to_string(index=False))


def main():
    # 1. Overall churn rate
    run_query(
        "Overall churn rate",
        """
        SELECT
            churn,
            COUNT(*) AS customer_count,
            ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM customers), 2) AS pct
        FROM customers
        GROUP BY churn
        """,
    )

    # 2. Churn rate by contract type
    run_query(
        "Churn rate by contract type",
        """
        SELECT
            contract,
            COUNT(*) AS total_customers,
            SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
            ROUND(100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_pct
        FROM customers
        GROUP BY contract
        ORDER BY churn_pct DESC
        """,
    )

    # 3. Average monthly charges: churned vs not churned
    run_query(
        "Average monthly charges by churn status",
        """
        SELECT
            churn,
            ROUND(AVG(monthlycharges), 2) AS avg_monthly_charges,
            ROUND(AVG(tenure), 1) AS avg_tenure_months
        FROM customers
        GROUP BY churn
        """,
    )

    # 4. Top 5 most common internet service types among churned customers
    run_query(
        "Internet service type among churned customers",
        """
        SELECT
            internetservice,
            COUNT(*) AS churned_customers
        FROM customers
        WHERE churn = 'Yes'
        GROUP BY internetservice
        ORDER BY churned_customers DESC
        LIMIT 5
        """,
    )

        # 4. Top 5 most common internet service types among churned customers
    run_query(
        "Internet service type among churned customers",
        """
        SELECT
            internetservice,
            COUNT(*) AS churned_customers
        FROM customers
        WHERE churn = 'Yes'
        GROUP BY internetservice
        ORDER BY churned_customers DESC
        LIMIT 5
        """,
    )

    # 5. Churn rate by senior citizen status
    run_query(
        "Churn rate by senior citizen status",
        """
        SELECT
            seniorcitizen,
            COUNT(*) AS total_customers,
            SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
            ROUND(
                100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*),
                2
            ) AS churn_rate
        FROM customers
        GROUP BY seniorcitizen
        ORDER BY seniorcitizen
        """,
    )

if __name__ == "__main__":
    main()
