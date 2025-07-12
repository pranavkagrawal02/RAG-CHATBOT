from src.cleaner import clean_text
from src.chunker import chunk_text
from src.embedder import generate_embeddings
from src.db_store import store_embeddings

# Step 1: Load document
with open("data/document.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Step 2: Clean text
cleaned = clean_text(raw_text)

# Step 3: Chunk text
chunks = chunk_text(cleaned)

# Step 4: Generate embeddings
embeddings = generate_embeddings(chunks)

# Step 5: Store in vector DB
store_embeddings(embeddings, chunks)
