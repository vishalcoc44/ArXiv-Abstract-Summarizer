import arxiv
import os
import time
import json
import requests
import socket

# Define query
query = "cat:cs"
print(f"Search query: {query}")

# Create directory
abstract_dir = "./arxiv_abstracts"
if not os.path.exists(abstract_dir):
    os.makedirs(abstract_dir)

# Test network connectivity
print("Testing network connectivity to ArXiv...")
try:
    response = requests.get("https://api.arxiv.org/", timeout=5)
    print(f"Network test successful. Status code: {response.status_code}")
except Exception as e:
    print(f"Network test failed: {e}")
    # Fallback: Resolve IP manually
    try:
        ip = socket.gethostbyname("api.arxiv.org")
        print(f"Manual DNS resolution succeeded: api.arxiv.org -> {ip}")
    except socket.gaierror as dns_err:
        print(f"DNS resolution failed: {dns_err}")
        exit(1)

# Initialize client
client = arxiv.Client(
    page_size=50,
    delay_seconds=5.0,
    num_retries=3
)
print("Client initialized")

# Define search
search = arxiv.Search(
    query=query,
    max_results=100,
    sort_by=arxiv.SortCriterion.SubmittedDate,
    sort_order=arxiv.SortOrder.Descending
)
print(f"Search defined with max_results={search.max_results}")

# Fetch abstracts
print("Starting to collect abstracts...")
abstracts = []
count = 0
try:
    for paper in client.results(search):
        count += 1
        abstract_data = {
            "title": paper.title,
            "abstract": paper.summary,
            "arxiv_id": paper.entry_id.split('/')[-1],
            "categories": paper.categories
        }
        abstracts.append(abstract_data)
        print(f"[{count}/{search.max_results}] Collected: {paper.title}")
        time.sleep(5)
except Exception as e:
    print(f"Error fetching results: {e}")

# Save abstracts
output_path = os.path.join(abstract_dir, "abstracts.json")
with open(output_path, "w") as f:
    json.dump(abstracts, f, indent=2)

print(f"Collected {count} abstracts and saved to {output_path}")
if count == 0:
    print("Warning: No abstracts collected. Check DNS settings or network.")