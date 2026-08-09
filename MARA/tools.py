from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from dotenv import load_dotenv
from rich import print
load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information."""

    print("\n[Tool] Tavily search started...")

    results = tavily.search(
        query=query,
        max_results=5
    )

    print("[Tool] Tavily search finished...")

    out = []

    for i, r in enumerate(results["results"], 1):
        out.append(
            f"Source {i}\n"
            f"Title: {r['title']}\n"
            f"URL: {r['url']}\n"
            f"Content: {r['content'][:600]}\n"
        )

    return "\n".join(out)
@tool
def scraper_url(url: str) -> str:
    """Scrape and return clean text content for a given url for deeper reading. """
    try:
        response = requests.get(url, timeout = 10, headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"})
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style","nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator = "\n", strip=True)[:6000]
    except Exception as e:
        return f"Could not scrap the URL: {str(e)}"