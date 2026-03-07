import gzip
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from src.utils.config import Config

logger = logging.getLogger(__name__)


def get_engine():
    """Factory to create the SQLAlchemy engine based on Config."""
    return create_engine(Config.get_db_url())


def query_to_df(query_str: str) -> pd.DataFrame:
    """
    Executes a SQL query and returns the results as a pandas DataFrame.
    """
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text(query_str), conn)
    return df


def run_ddl_script(engine, file_path: Path):
    """Execute a SQL file containing DDL statements (CREATE TABLE, DROP SCHEMA, etc.)."""
    if not file_path.exists():
        logger.error("Script not found: %s", file_path)
        return

    logger.info("Running DDL: %s", file_path.name)
    with open(file_path, "r", encoding="utf-8") as f:
        sql_content = f.read()

    with engine.connect() as conn:
        conn.execute(text(sql_content))
        conn.commit()


def load_table_from_csv(
    engine, table_name: str, file_path: Path, compressed: bool = False
):
    """Load CSV data into a table using the Postgres COPY protocol.

    Streams data from Python directly to the DB, bypassing file
    permission issues on the server.
    """
    if not file_path.exists():
        logger.warning("Skipping %s — file not found: %s", table_name, file_path)
        return

    logger.info("Loading table %s from %s", table_name, file_path.name)

    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cursor:
            sql = f"COPY {table_name} FROM STDIN WITH CSV HEADER NULL ''"

            if compressed:
                with gzip.open(file_path, "rt", encoding="utf-8") as f:
                    cursor.copy_expert(sql, f)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    cursor.copy_expert(sql, f)

        raw_conn.commit()
        logger.info("Loaded %s", table_name)
    except Exception as e:
        raw_conn.rollback()
        logger.error("Error loading %s: %s", table_name, e)
    finally:
        raw_conn.close()
