import os
os.environ['NO_COLOR'] = '1' # Add this line to prevent click/colorama console issues on Windows

import logging
import atexit
from dotenv import load_dotenv

# Load environment variables from .env file at the very beginning
load_dotenv()

from flask import Flask
from config3 import AppConfig
from db import init_db
from models3 import initialize_systems, cleanup_local_llm, ModelManager, systems_status
from routes2 import app # Changed from routes to routes2. Import the app object from routes2.py

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()]) # Add FileHandler for production
logger = logging.getLogger(__name__)

def cleanup_resources():
    """Cleans up resources on application exit."""
    logger.info("Application shutting down. Cleaning up resources...")
    if AppConfig.CLOUD_DEPLOYMENT:
        if systems_status.get("local_llm_loaded", False): # Check if model was loaded
            try:
                ModelManager.get_instance().cleanup()
                logger.info("ModelManager resources cleaned up.")
            except Exception as e:
                logger.error(f"Error during ModelManager cleanup: {e}", exc_info=True)
    else:
        cleanup_local_llm() # For non-cloud, direct cleanup
    logger.info("Resource cleanup finished.")

if __name__ == '__main__':
    logger.info("Starting application initialization...")
    
    # Initialize database
    # Check if the DB file exists, if not, init_db will create it.
    # No need to explicitly check for AppConfig.DATABASE_URL existence here,
    # init_db handles logging if the schema file is not found or if there's a DB error.
    init_db()
    
    # Initialize AI models and other systems
    initialize_systems()
    
    # Register cleanup function to be called on exit
    atexit.register(cleanup_resources)
    
    logger.info(f"Starting Flask development server on {AppConfig.FLASK_HOST}:{AppConfig.FLASK_PORT}")
    logger.info(f"Flask Debug Mode: {AppConfig.FLASK_DEBUG}")
    logger.info(f"Cloud Deployment Mode: {AppConfig.CLOUD_DEPLOYMENT}")
    logger.info(f"System Status after init: {systems_status}")

    # Run the Flask app
    # The 'app' object is imported from routes2.py where it's initialized
    app.run(host=AppConfig.FLASK_HOST, port=AppConfig.FLASK_PORT, debug=AppConfig.FLASK_DEBUG, use_reloader=AppConfig.FLASK_DEBUG)
    # use_reloader is often set to False when debug is False to prevent issues,
    # or more robustly, only enable reloader in debug mode.
    # Flask's default is use_reloader=True if debug=True. 