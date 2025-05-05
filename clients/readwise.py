# Client for Readwise Reader API at https://readwise.io/reader_api

import os
import glob
import json
import requests

from datetime import datetime, timedelta

from dotenv import load_dotenv

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document

from langchain.vectorstores import FAISS

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

def get_all_readwise_articles():
    """Gets all archived articles from Readwise Reader."""

    # Check for recent cache file first
    cache_file = _get_latest_cache_file()
    indexname = "./cache/faiss_index"

    articles = []

    if cache_file:
        with open(cache_file, "r", encoding="utf-8") as f:
            articles = json.load(f)

        faiss_index = FAISS.load_local(indexname, embeddings, allow_dangerous_deserialization=True)

    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./cache/readwise_archived_articles_{timestamp}.json"

        articles = _fetch_all_archived_articles()

        if articles:
            articles = _format_articles(articles)

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(articles, f, ensure_ascii=False, indent=2)

            faiss_index = _create_faiss_index(articles)
            faiss_index.save_local(indexname)

    return [articles, faiss_index]

def _get_latest_cache_file():
    """Returns the path to the most recent cache file if it exists and is less than 1 day old."""
    cache_files = glob.glob("./cache/readwise_archived_articles_*.json")

    if not cache_files:
        return None

    # Get the most recent file
    latest_file = max(cache_files, key=os.path.getctime)

    # Check if the file is less than 1 day old
    file_timestamp = datetime.fromtimestamp(os.path.getctime(latest_file))
    if datetime.now() - file_timestamp < timedelta(days=1):
        return latest_file

    return None

def _fetch_all_archived_articles():
    """Fetches all archived articles from Readwise Reader API."""
    articles = []
    next_page_cursor = None

    while True:
        # Prepare params for pagination
        params = {"limit": 100, "location": "archive"}
        if next_page_cursor:
            params["pageCursor"] = next_page_cursor

        # Make the API request
        response = requests.get(
            f"{os.getenv("READWISE_ENDPOINT")}/list/",
            headers={"Authorization": f"Token {os.getenv('READWISE_API_TOKEN')}"},
            params=params
        )

        # Check if request was successful
        if response.status_code != 200:
            print(f"Error fetching articles: {response.status_code}")
            print(response.text)
            break

        data = response.json()
        articles.extend(data.get("results", []))

        # Check if there are more pages
        next_page_cursor = data.get("nextPageCursor")
        if not next_page_cursor:
            break

    return articles

def _format_articles(articles):
    """Formats article data to extract link and summary."""
    formatted_articles = []

    for article in articles:
        title = article.get("title", "No title available")
        author = article.get("author", "No author available")
        site_name = article.get("site_name", "No source available")
        source_url = article.get("source_url", "No URL available")
        summary = article.get("summary", "No summary available")

        tags = article.get("tags", [])
        tag_names = [tag for tag in tags] if tags else []

        formatted_articles.append({
            "title": title,
            "author": author,
            "site_name": site_name,
            "link": source_url,
            "summary": summary,
            "tags": tag_names
        })

    return formatted_articles

def _create_faiss_index(articles):
    """Creates a FAISS index from the articles using sentence transformers."""

    docs = []
    for article in articles:

        # combine title and summary in a json string
        content = json.dumps({
            "title": article.get("title", "") or "",
            "author": article.get("author", "") or "",
            "site_name": article.get("site_name", "") or "",
            "summary": article.get("summary", "") or ""
        })

        # extract years from the article to make it searchable through the metadata
        years = extract_years(article)

        doc = Document(
            page_content = content,
            metadata = {
                "title": article.get("title", ""),
                "author": article.get("author", ""),
                "site_name": article.get("site_name", ""),
                "link": article.get("link", ""),
                "tags": article.get("tags", []),
                "years": years or []
            }
        )

        docs.append(doc)

    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore

import re

def extract_years(document, min_year=1900, max_year=2030):

    # Handle different input types
    text = json.dumps(document)
    
    # Pattern to match years (standalone 4-digit numbers between min_year and max_year)
    year_pattern = r'\b(19\d{2}|2\d{3})\b'
    
    # Find all matches
    years_found = [int(y) for y in re.findall(year_pattern, text)]
    
    # Filter years within valid range
    valid_years = [y for y in years_found if min_year <= y <= max_year]
    
    return valid_years

from langchain.tools import tool

@tool("search_readwise")
def search_readwise_articles(keywords: str, author: str = None, site: str = None, tag: str = None, year: int = None, num_of_results: int = 100) -> list:
    """
    Query the FAISS index for articles matching the given keywords.

    Args:
        - keywords (str): The keywords to search for.
        - author (str): A specific AUTHOR to filter the results (optional).
        - site (str): A specific site, newsletter, source, etc. to filter the results (optional).
        - tag (str): A specific TAG to filter the results (optional).
        - year (int): A specific YEAR to filter the results (optional).
        - num_of_results (int): The number of results to return (default is 100).

    Returns:
        list: A list of articles matching the criteria.
    """

    from clients.readwise import get_all_readwise_articles

    [_, faiss_index] = get_all_readwise_articles()

    keywords = keywords or ""

    if author: 
        keywords += f" author:{author}"
    
    if site:
        keywords += f" site:{site}"

    filter=lambda metadata: (
        (tag is None or tag in metadata.get('tags', [])) and
        (year is None or year in metadata.get('years', []))
    )

    try:
        # Perform the search
        results = faiss_index.similarity_search(query=keywords, k=num_of_results, filter=filter)

        # Format the results
        formatted_results = []
        for result in results:
            content = json.loads(result.page_content)
            formatted_results.append({
                "title": content.get("title", "No title"),
                "author": content.get("author", "No author"),
                "site_name": content.get("site_name", "No source"),
                "link": result.metadata.get("link", "No URL"),
                "tags": result.metadata.get("tags", []),	
                "summary": content.get("summary", "No summary")
            })

        return formatted_results

    except Exception as e:
        print(f"Error querying FAISS index: {e}")
        return []