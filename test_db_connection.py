"""Utility script to verify PostgreSQL connectivity."""
from sqlalchemy import text

from app.config import get_settings
from app.database import engine


def main():
    settings = get_settings()
    print(f"Using DB URL: {settings.database_url}")

    with engine.connect() as conn:
        result = conn.execute(text("SELECT current_database(), current_schema()"))
        db_name, schema = result.one()
        print(f"Connected to database: {db_name}, schema: {schema}")

        ping = conn.execute(text("SELECT 1"))
        print("Ping result:", ping.scalar())


if __name__ == "__main__":
    main()
