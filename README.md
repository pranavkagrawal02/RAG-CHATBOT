# RAG-CHATBOT
RAG-CHATBOT

# Vedio Link - https://drive.google.com/drive/folders/1ImqrR-XpnKWvsTd2ctiLlBHWODlllBsM?usp=sharing


This project implements a Retrieval-Augmented Generation (RAG) pipeline to build an intelligent chatbot capable of answering natural language queries based on a custom PDF document. The system combines semantic search over document chunks with language generation using a lightweight transformer model.

## 📐 Project Architecture & Flow

1. **Document Loader**: Loads PDF using `PyPDFLoader`.
2. **Text Preprocessing**: Cleans and chunks text into 100–300 word segments using `RecursiveCharacterTextSplitter`.
3. **Embedding Generation**: Generates dense embeddings using `all-MiniLM-L6-v2`.
4. **Vector Store**: Embeddings are stored and indexed using FAISS for semantic retrieval.
5. **Retriever**: At runtime, user queries are matched against document chunks to find relevant context.
6. **Prompt Injection**: Retrieved chunks + user question are combined into a prompt.
7. **Generator (LLM)**: Uses `distilgpt2` via HuggingFace pipeline to generate a response.
8. **Streaming UI**: Simulated token-by-token response rendered in a Streamlit chatbot interface.
9. **Source Display**: Shows which chunks were used to generate the answer.

---

## ⚙️ Setup & Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/ai-doc-chatbot.git
cd ai-doc-chatbot
```

### 2. Install required packages

```bash
pip install -r requirements.txt
```

### 3. Place your document

Put your document in the `data/` folder. Example:

```
data/AI Training Document.pdf
```

### 4. Run the chatbot

```bash
streamlit run app.py
```

To expose it publicly:
```bash
lt --port 8501      # using LocalTunnel
```

---

## 🤖 Model & Embedding Choices

- **Embedding Model**: `all-MiniLM-L6-v2` (fast, accurate, and lightweight)
- **Language Model**: `distilgpt2` for local inference (can be upgraded to `mistralai/Mistral-7B-Instruct-v0.1`)
- **Vector DB**: FAISS, a fast similarity search index

---

## 📥 Sample Queries

| Query                                 | Response Summary                            |
|--------------------------------------|---------------------------------------------|
| What is the objective of this training? | ❌ Hallucinated due to irrelevant chunk     |
| What is Federated Learning?           | ✅ Correct and contextual response           |
| Who is the Prime Minister of India?   | ✅ "I don't know" (good fallback)            |

---
📺 **Demo video** : https://drive.google.com/drive/folders/1ImqrR-XpnKWvsTd2ctiLlBHWODlllBsM
---

## 📂 Folder Structure

```
├── app.py                 # Streamlit interface
├── data/                  # Source PDF
├── chunks/                # Chunked text (optional)
├── vectordb/              # FAISS index + metadata
├── src/                   # Pipeline scripts
├── requirements.txt
└── README.md
```

---

## ✅ Future Improvements

- Switch to streaming-capable models like Mistral
- Improve chunk retrieval with re-ranking
- Add user-upload interface
- Add citation tracking per token
