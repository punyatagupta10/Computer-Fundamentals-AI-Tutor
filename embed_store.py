import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from langchain.vectorstores import FAISS as LangFAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Paths
CHUNKS_PATH = "./data/processed_chunks/"
FAISS_INDEX_PATH = "./data/embeddings/faiss_index"
METADATA_PATH = "./data/embeddings/metadata.pkl"

# Model
MODEL_NAME = "all-MiniLM-L6-v2"

def load_chunks():
    chunks = []
    metadata = []

    for file in os.listdir(CHUNKS_PATH):
        if file.endswith(".txt"):
            with open(os.path.join(CHUNKS_PATH, file), 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.read().split('\n\n') if line.strip()]
                for chunk in lines:
                    chunks.append(chunk)
                    metadata.append({'source': file, 'chunk': chunk})

    return chunks, metadata

def embed_chunks(chunks):
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings

def store_in_faiss(embeddings, metadata, chunks):
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

    docs = [Document(page_content=chunk, metadata=m) for chunk, m in zip(chunks, metadata)]
    embedding_func = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    vectorstore = LangFAISS.from_documents(docs, embedding_func)
    vectorstore.save_local(FAISS_INDEX_PATH, index_name="faiss_store")

if __name__ == "__main__":
    print("🔹 Loading text chunks...")
    chunks, metadata = load_chunks()

    print("🔹 Generating embeddings...")
    embeddings = embed_chunks(chunks)

    print("🔹 Saving to FAISS index...")
    store_in_faiss(embeddings, metadata, chunks)


    print("✅ All done! Embeddings stored.")
