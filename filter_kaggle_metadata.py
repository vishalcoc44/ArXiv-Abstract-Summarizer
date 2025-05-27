import json
import os
import logging
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define directories and paths
input_dir = "C:/Users/Vishal/Downloads/minor1"
input_file = os.path.join(input_dir, "arxiv-metadata-oai-snapshot.json")
output_dir = os.path.join(input_dir, "arxiv_abstracts")
# Define both output file paths
output_json_file = os.path.join(output_dir, "abstracts.json")
output_pickle_file = os.path.join(output_dir, "abstract_metadata.pkl")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Target categories
target_categories = ["cs", "physics", "math"]
filtered_data = []
count = 0

# Load and filter JSON in a memory-efficient way (line by line)
logger.info(f"Processing metadata from {input_file}...")

try:
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())  # Read JSON object from each line
                
                categories = entry.get("categories", "").split()
                if any(cat.startswith(tuple(target_categories)) for cat in categories):
                    count += 1
                    extracted_item_data = {
                        "title": entry.get("title", "N/A").replace("\n", " "),
                        "abstract": entry.get("abstract", "N/A").replace("\n", " "),
                        "arxiv_id": entry.get("id", "N/A"),
                        "categories": categories,
                        "authors": entry.get("authors", "N/A")
                    }
                    filtered_data.append(extracted_item_data)
                    
                    if count % 1000 == 0:
                        logger.info(f"Filtered {count} items...")

            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON line: {e}")
except FileNotFoundError as e:
    logger.error(f"Input file not found: {e}")
    raise

# Save filtered data as JSON
try:
    with open(output_json_file, "w", encoding="utf-8") as f_json:
        json.dump(filtered_data, f_json, indent=2)
    logger.info(f"Filtered {len(filtered_data)} items and saved to {output_json_file} (JSON format)")
except Exception as e:
    logger.error(f"Error saving filtered data as JSON: {e}")
    # We might still want to try saving as pickle, so not raising immediately
    # Consider if one failure should prevent the other save.

# Save filtered data as a pickle file
try:
    with open(output_pickle_file, "wb") as f_pickle: # Use "wb" for pickle
        pickle.dump(filtered_data, f_pickle)
    logger.info(f"Filtered {len(filtered_data)} items and saved to {output_pickle_file} (pickle format)")
except Exception as e:
    logger.error(f"Error saving filtered data as pickle: {e}")
    raise # Raise here if pickle saving fails