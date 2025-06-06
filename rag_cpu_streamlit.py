import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import requests
import os

# Load the embedding model (CPU-optimized)
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_embedder()

# Load documents and FAISS index
@st.cache_resource
def load_index_and_docs():
    if not os.path.exists("doc_index.faiss") or not os.path.exists("docs.npy"):
        st.error("Index or document file not found. Please run the index builder first.")
        st.stop()
    index = faiss.read_index("doc_index.faiss")
    docs = np.load("docs.npy", allow_pickle=True)
    return index, docs

index, docs = load_index_and_docs()

# Function to get response from llamafile server
def generate_response(prompt):
    url = "http://localhost:8080/completion"
    payload = {
        "prompt": prompt,
        "temperature": 0.7,
        "n_predict": 300
    }
    try:
        response = requests.post(url, json=payload)
        return response.json().get("content", "No response from LLM.")
    except Exception as e:
        return f"Error: {e}"

# RAG Pipeline
def rag_pipeline(query, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    relevant_docs = [docs[i] for i in indices[0]]

    prompt = f"User Question: {query}\n\nRelevant Information:\n"
    for doc in relevant_docs:
        prompt += f"- {doc}\n"
    prompt += "\nAnswer:"

    response = generate_response(prompt)
    return response, relevant_docs

# Streamlit UI
st.title("CPU-Based RAG Chatbot")
query = st.text_input("Ask your question about any topic:")
top_k = st.slider("Number of documents to retrieve", 1, 5, 3)

if st.button("Generate Answer") and query:
    with st.spinner("Thinking..."):
        answer, top_docs = rag_pipeline(query, top_k)
        st.subheader("Retrieved Documents:")
        for doc in top_docs:
            st.write("- ", doc)
        st.subheader("Answer:")
        st.write(answer)
