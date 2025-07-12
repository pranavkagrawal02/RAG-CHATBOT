import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def chunk_text(text: str, min_words=100, max_words=300) -> list:
    """Split text into sentence-aware chunks of 100–300 words."""
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], []

    for sentence in sentences:
        current_chunk.append(sentence)
        word_count = len(" ".join(current_chunk).split())

        if word_count >= max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
