import sqlite3
from pathlib import Path

DB_PATH = Path("db/portfolio.db")
SCHEMA_PATH = Path("db/schema.sql")

def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("✅ Database initialized")
