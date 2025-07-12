from sentence_transformers import SentenceTransformer

def generate_embeddings(chunks: list, model_name='all-MiniLM-L6-v2'):
    """Generate embeddings for each chunk using a transformer model."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings
