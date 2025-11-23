"""
Utility functions for the SHL recommendation system
"""

import requests
from bs4 import BeautifulSoup
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_jd_from_url(url: str) -> Optional[str]:
    """
    Fetch job description text from a URL
    
    Args:
        url: URL containing the job description
        
    Returns:
        Extracted text content or None if failed
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        logger.info(f"Fetched {len(text)} characters from {url}")
        return text
        
    except Exception as e:
        logger.error(f"Error fetching URL {url}: {e}")
        return None


def validate_query(query: str) -> bool:
    """
    Validate that a query is not empty and is reasonable
    
    Args:
        query: Query string
        
    Returns:
        True if valid, False otherwise
    """
    if not query or not query.strip():
        return False
    
    if len(query.strip()) < 3:
        return False
    
    return True


def format_recommendations(recommendations: list) -> list:
    """
    Format recommendations for API response with detailed fields
    
    Args:
        recommendations: List of recommendation dicts
        
    Returns:
        Formatted list of recommendations
    """
    formatted = []
    
    for rec in recommendations:
        # Get test type full name
        test_type_full = rec.get('test_type_full', rec.get('category', 'Other'))
        
        formatted.append({
            'url': rec.get('url', ''),
            'name': rec.get('assessment_name', ''),
            'adaptive_support': rec.get('adaptive_support', 'No'),
            'description': rec.get('description', ''),
            'duration': rec.get('duration', 15),
            'remote_support': rec.get('remote_support', 'Yes'),
            'test_type': [test_type_full]  # Return as list
        })
    
    return formatted


def is_url(text: str) -> bool:
    """
    Check if text is a URL
    
    Args:
        text: Text to check
        
    Returns:
        True if text appears to be a URL
    """
    text = text.strip().lower()
    return text.startswith('http://') or text.startswith('https://')
