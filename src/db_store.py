import faiss
import numpy as np
import pickle
import os

def store_embeddings(embeddings: list, chunks: list, vector_path='vectordb'):
    """Save FAISS index and corresponding chunk metadata."""
    os.makedirs(vector_path, exist_ok=True)

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    faiss.write_index(index, os.path.join(vector_path, 'faiss_index.index'))
    with open(os.path.join(vector_path, 'chunk_metadata.pkl'), 'wb') as f:
        pickle.dump(chunks, f)
