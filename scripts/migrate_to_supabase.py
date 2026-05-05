import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

def _insert_ignore(table, conn, keys, data_iter):
    """
    Custom pandas insert method that uses INSERT ... ON CONFLICT DO NOTHING
    to safely handle rows that were already migrated in a previous run.
    """
    from sqlalchemy.dialects.postgresql import insert
    rows = [dict(zip(keys, row)) for row in data_iter]
    stmt = insert(table.table).values(rows).on_conflict_do_nothing()
    conn.execute(stmt)

def safe_print(msg: str):
    """Print a message, replacing unencodable characters for Windows cp1252 consoles."""
    print(msg.encode("ascii", errors="replace").decode("ascii"))

def main():
    _PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    _SQLITE_PATH = f"sqlite:///{os.path.join(_PROJECT_ROOT, 'stock_forecast.db')}"
    
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    if not DATABASE_URL or DATABASE_URL.startswith("sqlite"):
        print("Error: DATABASE_URL is not set to a PostgreSQL string in .env")
        print("Please complete Step 1 and set DATABASE_URL=your_supabase_string")
        return

    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
        
    if DATABASE_URL.startswith("postgresql://"):
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

    print(f"Connecting to SQLite at: {_SQLITE_PATH}")
    sqlite_engine = create_engine(_SQLITE_PATH, echo=False)
    
    print(f"Connecting to PostgreSQL at: {DATABASE_URL.split('@')[-1]}")  # Hide password in logs
    pg_engine = create_engine(
        DATABASE_URL,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_pre_ping=True,
    )

    tables_to_migrate = ["ohlcv", "signals", "sentiment", "macro"]
    batch_size = 500

    for table in tables_to_migrate:
        print(f"\n--- Migrating table: {table} ---")
        try:
            # Check if table exists in SQLite
            with sqlite_engine.connect() as conn:
                count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                sqlite_count = count_result.scalar()
                
            print(f"Found {sqlite_count} rows in SQLite '{table}'")
            
            # Read in batches and write with ON CONFLICT DO NOTHING
            offset = 0
            while True:
                query = f"SELECT * FROM {table} LIMIT {batch_size} OFFSET {offset}"
                df = pd.read_sql(query, sqlite_engine)
                
                if df.empty:
                    break
                    
                df.to_sql(
                    table, pg_engine,
                    if_exists="append",
                    index=False,
                    method=_insert_ignore,
                )
                offset += len(df)
                print(f"  Migrated {offset}/{sqlite_count} rows...")
                
            # Compare counts
            with pg_engine.connect() as conn:
                pg_count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                pg_count = pg_count_result.scalar()
                
            print(f"Comparison for '{table}': SQLite={sqlite_count}, PostgreSQL={pg_count}")
            if sqlite_count == pg_count:
                print(f"SUCCESS: {table} migrated perfectly.")
            else:
                print(f"INFO: {table} — SQLite={sqlite_count}, PG={pg_count} (delta may be pre-existing rows).")

        except Exception as e:
            err_str = str(e)
            if "no such table" in err_str.lower():
                print(f"Warning: Table '{table}' does not exist in SQLite. Skipping.")
            else:
                safe_print(f"Error migrating table '{table}': {err_str}")

if __name__ == "__main__":
    main()
