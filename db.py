import sqlite3
import logging
import os
from config import AppConfig # Assuming AppConfig is in config.py

logger = logging.getLogger(__name__)

def get_db_connection():
    """Establishes a connection to the database."""
    conn = sqlite3.connect(AppConfig.DATABASE_URL)
    conn.row_factory = sqlite3.Row # This allows accessing columns by name
    return conn

def init_db():
    """Initializes the database using the schema.sql file."""
    db_path = AppConfig.DATABASE_URL
    schema_path = 'static/schema.sql' # Assuming schema.sql is in the static folder, adjust if not

    try:
        logger.info(f"Initializing database at {db_path} with schema {schema_path}")
        conn = get_db_connection()
        with open(schema_path, 'r') as f:
            conn.executescript(f.read())
        conn.commit()
        logger.info("Database initialized successfully.")
    except sqlite3.Error as e:
        logger.error(f"Error initializing database: {e}", exc_info=True)
    except FileNotFoundError:
        logger.error(f"Schema file not found at {schema_path}. Database not initialized.", exc_info=True)
    finally:
        if 'conn' in locals() and conn:
            conn.close() 