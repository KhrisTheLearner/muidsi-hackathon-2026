"""Load USDA Food Atlas datasets into agriflow.db SQLite database.

Usage:
    python -m src.data_pipeline.load_atlas

Loads:
    food_env_atlas_cleaned_twice 1.csv  -> food_environment table
    food_research_atlas_cleaned 1.csv  -> food_access table

Expects data files in data/raw/.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = os.getenv("DB_PATH", "data/agriflow.db")

ENV_ATLAS_FILE = Path("data/raw/food_env_atlas_cleaned_twice 1.csv")
ACCESS_ATLAS_FILE = Path("data/raw/food_research_atlas_cleaned 1.csv")

# Mapping from full state names to 2-letter codes (for food_access table)
STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}


def load_food_environment(conn: sqlite3.Connection) -> int:
    """Load Food Environment Atlas CSV into food_environment table."""
    print(f"Loading {ENV_ATLAS_FILE} ...")

    df = pd.read_csv(ENV_ATLAS_FILE, low_memory=False)
    print(f"  Read {len(df):,} rows x {len(df.columns)} columns")

    # Ensure FIPS is stored as integer-compatible string with leading zeros
    df["FIPS"] = df["FIPS"].fillna(0).astype(int)

    # Replace -9999 sentinel values with NULL
    df = df.replace(-9999, None).replace(-9999.0, None)

    # Write to SQLite - replace table if exists
    df.to_sql("food_environment", conn, if_exists="replace", index=False)

    # Create indexes for fast queries
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fe_state ON food_environment(State)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fe_fips ON food_environment(FIPS)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fe_county ON food_environment(County)")
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM food_environment").fetchone()[0]
    print(f"  Loaded {count:,} rows into food_environment table")
    return count


def load_food_access(conn: sqlite3.Connection) -> int:
    """Load Food Access Research Atlas CSV into food_access table."""
    print(f"Loading {ACCESS_ATLAS_FILE} ...")

    df = pd.read_csv(ACCESS_ATLAS_FILE, low_memory=False)
    print(f"  Read {len(df):,} rows x {len(df.columns)} columns")

    # Convert full state names to 2-letter codes so tools can query by "MO"
    df["State"] = df["State"].map(STATE_ABBR).fillna(df["State"])

    # Strip " County" suffix from County column to match food_environment style
    df["County"] = df["County"].str.replace(r"\s+County$", "", regex=True).str.strip()

    # Replace -9999 sentinel values with NULL
    df = df.replace(-9999, None).replace(-9999.0, None)

    # Write to SQLite
    df.to_sql("food_access", conn, if_exists="replace", index=False)

    # Create indexes for fast queries
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fa_state ON food_access(State)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fa_county ON food_access(County)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fa_tract ON food_access(CensusTract)")
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM food_access").fetchone()[0]
    print(f"  Loaded {count:,} rows into food_access table")
    return count


def verify(conn: sqlite3.Connection) -> bool:
    """Run basic verification queries to confirm data loaded correctly."""
    print("\nVerifying data...")
    all_pass = True

    checks = [
        ("food_environment row count", "SELECT COUNT(*) FROM food_environment", lambda n: n > 3000),
        ("food_access row count", "SELECT COUNT(*) FROM food_access", lambda n: n > 70000),
        ("MO counties in food_environment", "SELECT COUNT(*) FROM food_environment WHERE State='MO'", lambda n: n == 115),
        ("MO tracts in food_access", "SELECT COUNT(*) FROM food_access WHERE State='MO'", lambda n: n > 1000),
        ("Food insecurity column exists", "SELECT FOODINSEC_21_23 FROM food_environment WHERE State='MO' LIMIT 1", lambda r: True),
        ("LILA column exists", "SELECT LILATracts_1And10 FROM food_access WHERE State='MO' LIMIT 1", lambda r: True),
        ("No -9999 sentinels", "SELECT COUNT(*) FROM food_environment WHERE FOODINSEC_21_23 = -9999", lambda n: n == 0),
        ("State abbr in food_access", "SELECT COUNT(DISTINCT State) FROM food_access WHERE length(State) = 2", lambda n: n >= 50),
    ]

    for label, query, check_fn in checks:
        try:
            result = conn.execute(query).fetchone()[0]
            passed = check_fn(result)
            status = "[PASS]" if passed else "[FAIL]"
            print(f"  {status} {label}: {result}")
            if not passed:
                all_pass = False
        except Exception as e:
            print(f"  [FAIL] {label}: ERROR - {e}")
            all_pass = False

    return all_pass


def print_summary(conn: sqlite3.Connection) -> None:
    """Print a summary of the loaded data."""
    print("\nData Summary:")

    fe_count = conn.execute("SELECT COUNT(*) FROM food_environment").fetchone()[0]
    fa_count = conn.execute("SELECT COUNT(*) FROM food_access").fetchone()[0]
    mo_fe = conn.execute("SELECT COUNT(*) FROM food_environment WHERE State='MO'").fetchone()[0]
    mo_fa = conn.execute("SELECT COUNT(*) FROM food_access WHERE State='MO'").fetchone()[0]
    fe_cols = conn.execute("PRAGMA table_info(food_environment)").fetchall()
    fa_cols = conn.execute("PRAGMA table_info(food_access)").fetchall()

    print(f"  food_environment: {fe_count:,} rows, {len(fe_cols)} columns")
    print(f"    Missouri: {mo_fe} counties")
    print(f"  food_access:      {fa_count:,} rows, {len(fa_cols)} columns")
    print(f"    Missouri: {mo_fa:,} census tracts")

    # Sample MO food insecurity data
    sample = conn.execute(
        "SELECT County, FOODINSEC_21_23, POVRATE21, PCT_LACCESS_POP15 "
        "FROM food_environment WHERE State='MO' AND FOODINSEC_21_23 IS NOT NULL "
        "ORDER BY FOODINSEC_21_23 DESC LIMIT 5"
    ).fetchall()

    if sample:
        print("\n  Top 5 Missouri counties by food insecurity (2021-23):")
        for row in sample:
            county, rate, pov, laccess = row
            print(f"    {county:<20} {rate:.1f}% food insecure | {pov:.1f}% poverty | {laccess:.1f}% low access")


def main() -> None:
    print("=" * 60)
    print("AgriFlow Database Loader")
    print("=" * 60)

    # Verify source files exist
    for f in [ENV_ATLAS_FILE, ACCESS_ATLAS_FILE]:
        if not f.exists():
            raise FileNotFoundError(f"Source file not found: {f}")

    # Create parent directory if needed
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)

    try:
        fe_rows = load_food_environment(conn)
        fa_rows = load_food_access(conn)

        all_pass = verify(conn)
        print_summary(conn)

        print("\n" + "=" * 60)
        if all_pass:
            print("Database loaded successfully!")
            print(f"  Location: {Path(DB_PATH).resolve()}")
            print(f"  food_environment: {fe_rows:,} rows")
            print(f"  food_access:      {fa_rows:,} rows")
            print("\nThe agent can now answer questions about food insecurity.")
            print("Try: python run_agent.py \"Which Missouri counties have highest food insecurity?\"")
        else:
            print("Database loaded with some verification failures - check output above.")
        print("=" * 60)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
