import re

def clean_text(text: str) -> str:
    """Remove HTML tags, headers, footers, and extra spaces."""
    text = re.sub(r'<[^>]+>', '', text)   # Remove HTML tags
    text = re.sub(r'\s+', ' ', text)      # Collapse whitespace
    return text.strip()
