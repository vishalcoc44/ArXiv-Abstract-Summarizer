import requests
import json
import logging

logger = logging.getLogger(__name__)

def fetch_wikipedia_summary(query: str) -> str:
    """
    Fetches a summary from Wikipedia for a given query.
    Returns the summary as a string, or an error/not found message.
    """
    SESSION_TIMEOUT = 10 # seconds for requests
    WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
    # It's good practice to set a User-Agent. Replace with your app's info if deploying.
    headers = {
        "User-Agent": "ChatApp/1.0 (Flask_App; https://example.com/contact or mailto:user@example.com)"
    }

    # Step 1: Search for the query to get a page title
    search_params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srlimit": 1, # Get only the top result
        "srprop": ""  # We don't need snippets from the search results themselves
    }
    search_data = {} # Initialize in case of early exit

    try:
        logger.info(f"Searching Wikipedia for: '{query[:100]}...'")
        search_response = requests.get(WIKIPEDIA_API_URL, params=search_params, headers=headers, timeout=SESSION_TIMEOUT)
        search_response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
        search_data = search_response.json()

        if not search_data.get("query", {}).get("search"):
            logger.info(f"No Wikipedia search results found for: '{query[:100]}...'")
            return f"No Wikipedia article found for '{query}'."
        
        page_title = search_data["query"]["search"][0]["title"]
        logger.info(f"Found Wikipedia page title: '{page_title}' for query: '{query[:100]}...'")

    except requests.exceptions.Timeout:
        logger.error(f"Wikipedia API search request timed out for query '{query[:100]}...'", exc_info=True)
        return f"Error: The request to Wikipedia timed out while searching for '{query}'."
    except requests.exceptions.RequestException as e:
        logger.error(f"Wikipedia API search request failed for query '{query[:100]}...': {e}", exc_info=True)
        return f"Error: Could not connect to Wikipedia to search for '{query}'."
    except (KeyError, IndexError) as e:
        logger.error(f"Error parsing Wikipedia search results for query '{query[:100]}...': {e} - Data: {search_data}", exc_info=True)
        return f"Error: Could not process search results from Wikipedia for '{query}'."
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from Wikipedia search for '{query[:100]}...': {e}", exc_info=True)
        return f"Error: Invalid response format from Wikipedia search for '{query}'."


    # Step 2: Get the extract (summary) of that page title
    extract_params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "titles": page_title,
        "exintro": True,      # Get only the introductory section (summary)
        "explaintext": True,  # Get plain text, not HTML
        "exlimit": 1          # Max 1 extract for the given title
    }
    extract_data = {} # Initialize in case of early exit

    try:
        logger.info(f"Fetching Wikipedia extract for page title: '{page_title}'")
        extract_response = requests.get(WIKIPEDIA_API_URL, params=extract_params, headers=headers, timeout=SESSION_TIMEOUT)
        extract_response.raise_for_status()
        extract_data = extract_response.json()

        pages = extract_data.get("query", {}).get("pages")
        if not pages:
            logger.warning(f"No 'pages' data in Wikipedia extract response for title '{page_title}'. Data: {extract_data}")
            # This can happen if the title, though found in search, is invalid for extracts (e.g. special pages)
            return f"Error: Could not retrieve summary content from Wikipedia for the article '{page_title}'."

        # The page ID is dynamic (e.g., "736"), so we get the first (and only) page ID from the 'pages' object
        page_id = next(iter(pages)) # Gets the first key from the pages dictionary
        summary = pages[page_id].get("extract", "").strip()

        if not summary:
            # This could happen for disambiguation pages, very short articles, or protected pages.
            logger.info(f"Empty summary returned from Wikipedia for page title: '{page_title}'")
            return f"No summary found on Wikipedia for the article '{page_title}'. It might be a disambiguation page, very short, or require specific permissions to view."
        
        logger.info(f"Successfully fetched Wikipedia summary for '{page_title}' (length: {len(summary)} chars).")
        return summary

    except requests.exceptions.Timeout:
        logger.error(f"Wikipedia API extract request timed out for title '{page_title}'", exc_info=True)
        return f"Error: The request to Wikipedia timed out while fetching the summary for '{page_title}'."
    except requests.exceptions.RequestException as e:
        logger.error(f"Wikipedia API extract request failed for title '{page_title}': {e}", exc_info=True)
        return f"Error: Could not connect to Wikipedia to get summary for '{page_title}'."
    except (KeyError, StopIteration) as e: # StopIteration for next(iter(pages)) if pages is unexpectedly empty or malformed
        logger.error(f"Error parsing Wikipedia extract results for title '{page_title}': {e} - Data: {extract_data}", exc_info=True)
        return f"Error: Could not process summary from Wikipedia for '{page_title}'."
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from Wikipedia extract for '{page_title}': {e}", exc_info=True)
        return f"Error: Invalid response format from Wikipedia extract for '{page_title}'."
    except Exception as e_gen: # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred in fetch_wikipedia_summary for title '{page_title}': {e_gen}", exc_info=True)
        return f"An unexpected error occurred while fetching the Wikipedia summary for '{page_title}'." 