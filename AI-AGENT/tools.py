# Import the DuckDuckGo search tool from LangChain community package
from langchain_community.tools import DuckDuckGoSearchRun

# Import the generic Tool wrapper from LangChain Core
from langchain_core.tools import Tool

# Standard libraries for scraping and saving
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import re

# Save-to-text tool: saves the output to a text file
def save_to_txt(data: str, filename: str = "leads_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Leads Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    # Open the file in append mode so it keeps growing over time
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"

# Scrape raw text from a website
def scrape_website(url: str) -> str:
    try:
        # Send GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise error for bad HTTP codes

        # Parse and clean up the raw HTML content
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace

        # Limit to 5000 characters for performance and API limits
        return text[:5000]
    except Exception as e:
        return f"Error scraping website: {e}"

# Generate search queries to look for IT services related to a company
def generate_search_queries(company_name: str) -> list[str]:
    keywords = ["IT Services", "managed IT", "technology solutions"]
    return [f"{company_name} {keyword}" for keyword in keywords]

# Combined search and scrape operation for a company
def search_and_scrape(company_name: str) -> str:
    queries = generate_search_queries(company_name)
    results = []

    for query in queries:
        # Run web search
        search_results = search.run(query)

        # Extract URLs from the search output
        urls = re.findall(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            search_results
        )

        # Scrape the first valid URL found
        if urls:
            results.append(scrape_website(urls[0]))

    # Combine all the text results into one big chunk
    return " ".join(results)

# DuckDuckGo search tool
search = DuckDuckGoSearchRun()

# Defining all the tools that we created above

search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information",
)

scrape_tool = Tool(
    name="scrape_website",
    func=search_and_scrape,
    description="Scrape the content of a website and search for related information.",
)

save_tool = Tool(
    name="save",
    func=save_to_txt,
    description="Saves structured data to a text file.",
)
