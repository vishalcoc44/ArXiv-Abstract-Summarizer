import json
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define directories and paths
input_dir = "C:/Users/Vishal/Downloads/minor1"  
input_file = os.path.join(input_dir, "arxiv-metadata-oai-snapshot.json")  
output_dir = os.path.join(input_dir, "arxiv_abstracts")
output_file = os.path.join(output_dir, "abstracts.json")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Target categories
target_categories = ["cs", "physics", "math"]
filtered_abstracts = []
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
                    abstract_data = {
                        "title": entry.get("title", "N/A").replace("\n", " "),
                        "abstract": entry.get("abstract", "N/A").replace("\n", " "),
                        "arxiv_id": entry.get("id", "N/A"),
                        "categories": categories
                    }
                    filtered_abstracts.append(abstract_data)
                    
                    if count % 1000 == 0:
                        logger.info(f"Filtered {count} abstracts...")

            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON line: {e}")
except FileNotFoundError as e:
    logger.error(f"Input file not found: {e}")
    raise

# Save filtered data
try:
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered_abstracts, f, indent=2)
    logger.info(f"Filtered {len(filtered_abstracts)} abstracts and saved to {output_file}")
except Exception as e:
    logger.error(f"Error saving filtered data: {e}")
    raise