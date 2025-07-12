import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.llms import HuggingFacePipeline
import time



st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("Amlgo Labs Case Study Assignment Chatbot")

@st.cache_resource
def setup_rag_pipeline():
    document_path = "data/AI Training Document.pdf"
    #st.warning(f"Looking for file at: {document_path}")
    #st.warning(f"File exists? {os.path.exists(document_path)}")
    if not os.path.exists(document_path):
        st.error("Document not found.")
        return None, None, None

    loader = PyPDFLoader(document_path)
    pages = loader.load()
    cleaned = [p.page_content.replace('\n', ' ').strip() for p in pages]

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.create_documents(cleaned)

    embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embedder)
    retriever = vectordb.as_retriever()

    model_name = "distilbert/distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,
                    max_new_tokens=512, do_sample=True,
                    temperature=0.7, top_k=50, top_p=0.95)

    llm = HuggingFacePipeline(pipeline=pipe)

    template = """Use the following context to answer the question.
If unknown, say "I don't know".

Context:
{context}

Question:
{question}

Answer:"""

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever, chunks

with st.spinner("Initializing RAG pipeline..."):
    rag_chain, retriever, all_chunks = setup_rag_pipeline()
    #if rag_chain:
        #st.success("RAG system ready!")

# Sidebar info
st.sidebar.markdown("### ℹ️ Info")
st.sidebar.markdown("**Model:** `distilgpt2`")
st.sidebar.markdown(f"**Chunks Indexed:** {len(all_chunks) if all_chunks else 'N/A'}")
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Streaming chatbot input/output
if prompt := st.chat_input("Ask a question", disabled=rag_chain is None):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        retrieved_docs = retriever.get_relevant_documents(prompt)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        final_prompt = f"""Use the following context to answer the question.
If unknown, say "I don't know".

Context:
{context}

Question:
{prompt}

Answer:"""

        # Simulated streaming (word-by-word)
        response = rag_chain.invoke(prompt)
        full_response = ""
        with st.chat_message("assistant"):
            response_box = st.empty()
            for word in response.split():
                full_response += word + " "
                response_box.markdown(full_response + "▌")
                #st.sleep(0.03)
                time.sleep(0.03)
            response_box.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Show source context
        with st.expander("📚 Source Chunks"):
            for i, doc in enumerate(retrieved_docs):
                st.markdown(f"**Chunk {i+1}:**\n> {doc.page_content[:300]}...")

    except Exception as e:
        st.error(f"Error: {e}")
        st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
