import os
import faiss
import pickle
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document

# === CONFIGURATION ===
FAISS_INDEX_PATH = "./data/embeddings/faiss_index"
METADATA_PATH = "./data/embeddings/metadata.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"
GOOGLE_API_KEY = "AIzaSyBfvJTPipAYktCN5TXKdVRZoNPwo_oJZOA"  # <-- Replace this or load from env

# === SETUP ===
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")  # Full ID

embedder = SentenceTransformer(MODEL_NAME)

# === LOAD VECTOR DB ===
index = faiss.read_index(FAISS_INDEX_PATH)
with open(METADATA_PATH, 'rb') as f:
    metadata = pickle.load(f)

# === FAISS RETRIEVAL ===
def query_faiss(query, top_k=5):
    query_vec = embedder.encode([query])
    distances, indices = index.search(query_vec, top_k)

    docs = []
    for idx in indices[0]:
        chunk = metadata[idx].get("chunk", "[Missing chunk]")
        docs.append(Document(page_content=chunk, metadata=metadata[idx]))
    return docs

# === GEMINI ANSWERING ===
def answer_with_gemini(context, question):
    prompt = f"""
You are a helpful computer science tutor. Use the context below to answer the student's question.

Context:
{context}

Question: {question}

Answer:"""
    response = model.generate_content(prompt)
    return response.text.strip()

# === MAIN RAG FUNCTION ===
def answer_question(question):
    docs = query_faiss(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    return answer_with_gemini(context, question)

# === CLI CHAT LOOP ===
if __name__ == "__main__":
    print("🤖 Gemini RAG Chatbot for CS is ready. Ask away!\n")
    while True:
        user_input = input("💬 Your question (type 'exit' to quit): ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = answer_question(user_input)
        print(f"\n🎓 Gemini says:\n{response}\n")
