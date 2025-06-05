import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import textwrap

# Paths
FAISS_INDEX_PATH = "./data/embeddings/faiss_index"
METADATA_PATH = "./data/embeddings/metadata.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

# Load FAISS + metadata
index = faiss.read_index(FAISS_INDEX_PATH)
with open(METADATA_PATH, 'rb') as f:
    metadata = pickle.load(f)

model = SentenceTransformer(MODEL_NAME)

def search(query, top_k=5):
    query_vec = model.encode([query])
    distances, indices = index.search(query_vec, top_k)

    print(f"\n🔍 Top {top_k} results for: \"{query}\"")
    for i, idx in enumerate(indices[0]):
        print(f"\n--- Result {i+1} (Score: {distances[0][i]:.2f}) ---")
        print(f"Source: {metadata[idx]['source']}")
        print(f"Content:\n{metadata[idx].get('chunk', '[chunk text not stored]')}")
        print("------------------------------")
        
        


# Optional: store chunk content in metadata if not already
def enrich_metadata(chunks):
    for i, chunk in enumerate(chunks):
        if 'chunk' not in metadata[i]:
            metadata[i]['chunk'] = chunk
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)

if __name__ == "__main__":
    query = input("Enter your question: ")
    search(query)

