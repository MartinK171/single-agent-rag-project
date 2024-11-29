from typing import List, Dict
import logging
import time
import random
from urllib.parse import quote
import requests
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup

class WebSearchTool:
    """Enhanced web search tool with multiple search services and robust error handling."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 2.0):
        self.logger = logging.getLogger(__name__)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def _handle_rate_limit(self, attempt: int) -> float:
        """Handle rate limiting with exponential backoff."""
        delay = self.base_delay * (2 ** attempt) * (1 + 0.1 * random.random())
        self.logger.info(f"Rate limit delay: {delay:.2f} seconds")
        time.sleep(delay)
        return delay

    def _ddg_search(self, query: str, max_results: int = 3) -> List[Dict]:
        """Primary search using DuckDuckGo."""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query,
                    region='wt-wt',
                    safesearch='moderate',
                    timelimit='m',  # Last month
                    max_results=max_results
                ))
                
                if results:
                    return [{
                        'title': r.get('title', ''),
                        'link': r.get('link', ''),
                        'snippet': r.get('body', ''),
                        'source': 'duckduckgo'
                    } for r in results]
                return []
        except Exception as e:
            self.logger.error(f"DuckDuckGo search failed: {str(e)}")
            return []

    def _google_news_search(self, query: str, max_results: int = 3) -> List[Dict]:
        """Fallback search using Google News RSS."""
        try:
            # Add 'AI' if not in query to ensure relevance
            if 'ai' not in query.lower():
                query = f"AI {query}"
                
            encoded_query = quote(query)
            url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
            
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'xml')
                items = soup.find_all('item')[:max_results]
                
                results = []
                for item in items:
                    results.append({
                        'title': item.title.text if item.title else '',
                        'link': item.link.text if item.link else '',
                        'snippet': item.description.text if item.description else '',
                        'source': 'google_news',
                        'date': item.pubDate.text if item.pubDate else ''
                    })
                return results
            return []
        except Exception as e:
            self.logger.error(f"Google News search failed: {str(e)}")
            return []

    def _validate_results(self, results: List[Dict]) -> List[Dict]:
        """Validate and filter search results."""
        valid_results = []
        for result in results:
            # Check for required fields
            if not all(k in result for k in ['title', 'link', 'snippet']):
                continue
                
            # Check for minimum content length
            if len(result['snippet']) < 50:  # Arbitrary minimum length
                continue
                
            # Check if content is in English (basic check)
            if not all(ord(c) < 128 for c in result['title'][:100]):
                continue
                
            valid_results.append(result)
            
        return valid_results

    def search(self, query: str, max_results: int = 3) -> List[Dict]:
        """Perform web search with fallback mechanisms."""
        all_results = []
        
        # Try DuckDuckGo first
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"DuckDuckGo search attempt {attempt + 1}/{self.max_retries}")
                results = self._ddg_search(query, max_results)
                if results:
                    all_results.extend(results)
                    break
                    
                self._handle_rate_limit(attempt)
                
            except Exception as e:
                self.logger.error(f"Search attempt {attempt + 1} failed: {str(e)}")
                if 'rate' in str(e).lower():
                    self._handle_rate_limit(attempt)
                continue

        # If DuckDuckGo failed or returned no results, try Google News
        if not all_results:
            self.logger.info("Trying Google News fallback")
            news_results = self._google_news_search(query, max_results)
            if news_results:
                all_results.extend(news_results)

        # Validate and deduplicate results
        valid_results = self._validate_results(all_results)
        
        # Log search outcome
        if valid_results:
            self.logger.info(f"Found {len(valid_results)} valid results")
        else:
            self.logger.warning("No valid results found")
            
        return valid_results[:max_results]  # Return only requested number of results

    def format_results(self, results: List[Dict]) -> str:
        """Format search results into a readable string."""
        if not results:
            return ("I apologize, but I couldn't find any recent information about that topic. "
                   "This might be due to search limitations.")
            
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.extend([
                f"[Source {i}]",
                f"Title: {result['title']}",
                f"Summary: {result['snippet']}",
                f"URL: {result['link']}",
                f"Source: {result.get('source', 'unknown')}",
                "---"
            ])
        
        return "\n".join(formatted)