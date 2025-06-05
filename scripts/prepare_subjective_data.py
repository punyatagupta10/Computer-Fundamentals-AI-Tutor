import os
import json
import re
import pickle
from sentence_transformers import SentenceTransformer

# === CONFIGURATION ===
MODEL_NAME = "all-MiniLM-L6-v2"
INPUT_JSON = "data/questions/subjective_qna_dataset.json"
OUTPUT_DIR = "./data/subjective_embeddings/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def main():
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    model = SentenceTransformer(MODEL_NAME)

    for idx, item in enumerate(data):
        question_raw = item.get("Question", "")
        answer_raw = item.get("Answer", "")

        if not question_raw or not answer_raw:
            continue

        question = clean_text(question_raw)
        reference = clean_text(answer_raw)

        q_emb = model.encode([question])[0]
        ref_emb = model.encode([reference])[0]

        result = {
            "question": question,
            "reference_answer": reference,
            "question_embedding": q_emb.tolist(),
            "reference_embedding": ref_emb.tolist(),
            "subject": item.get("Subject", ""),
            "topic": item.get("Topic", "")
        }

        out_path = os.path.join(OUTPUT_DIR, f"qa_{idx}.pkl")
        with open(out_path, 'wb') as f:
            pickle.dump(result, f)

    print(f"✅ Processed {len(data)} QA pairs and saved embeddings to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
