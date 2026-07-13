"""
sql/run_queries_sqlite.py

Wires up sql/churn_queries.sql to a real, running database instead of
leaving it as reference SQL. Loads telecom_customers.csv into a local
SQLite database, then executes each query in churn_queries.sql against
it and prints the results — the same CTEs / window-function queries
run unmodified (SQLite supports both).

Why SQLite: zero setup (stdlib `sqlite3`, no server), which keeps this
runnable for anyone cloning the repo. The same connection code swaps to
MySQL with `mysql-connector-python` (see `get_mysql_connection` below,
disabled by default) with no changes to the queries themselves.

Usage:
    python sql/run_queries_sqlite.py
"""
import re
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
CSV_PATH = ROOT / "data" / "telecom_customers.csv"
SQL_PATH = ROOT / "sql" / "churn_queries.sql"
DB_PATH = ROOT / "sql" / "churn.db"


def load_csv_into_sqlite(db_path: Path) -> sqlite3.Connection:
    df = pd.read_csv(CSV_PATH)
    conn = sqlite3.connect(db_path)
    df.to_sql("customers", conn, if_exists="replace", index=False)
    return conn


def split_statements(sql_text: str) -> list[str]:
    """Split the .sql file into individual statements, dropping comments."""
    # Strip full-line comments, keep statements separated by ';'
    lines = [ln for ln in sql_text.splitlines() if not ln.strip().startswith("--")]
    cleaned = "\n".join(lines)
    statements = [s.strip() for s in cleaned.split(";") if s.strip()]
    return statements


def label_for(statement: str, index: int) -> str:
    m = re.search(r"contract_type", statement)
    labels = {
        0: "1. Churn rate by contract type",
        1: "2. Churn rate by tenure bucket",
        2: "3. High-risk customer flag (top 5 rows)",
        3: "4. Revenue at risk from churned accounts",
        4: "5. Add-on adoption vs. churn risk rank",
    }
    return labels.get(index, f"Query {index + 1}")


def run_all(conn: sqlite3.Connection):
    sql_text = SQL_PATH.read_text()
    statements = split_statements(sql_text)

    for i, stmt in enumerate(statements):
        print("=" * 70)
        print(label_for(stmt, i))
        print("=" * 70)
        df = pd.read_sql_query(stmt, conn)
        # Query 3 (high-risk flag) returns one row per customer — show a sample only
        if i == 2:
            df = df.head(5)
        print(df.to_string(index=False))
        print()


def get_mysql_connection():
    """
    Not used by default (keeps the script runnable with zero external
    services), but included to show the same query layer works unchanged
    against a real MySQL instance per the JD's "good to have" MySQL note.

        import mysql.connector
        return mysql.connector.connect(
            host="localhost", user="root", password="...", database="churn_db"
        )
    """
    raise NotImplementedError("Configure credentials and uncomment the mysql-connector import to use this.")


def main():
    conn = load_csv_into_sqlite(DB_PATH)
    print(f"Loaded {CSV_PATH.name} into SQLite DB -> {DB_PATH}\n")
    run_all(conn)
    conn.close()


if __name__ == "__main__":
    main()
