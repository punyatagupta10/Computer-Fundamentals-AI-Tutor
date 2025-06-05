import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_DIR = "./data/subjective_embeddings/"

def load_question_entry(idx):
    path = os.path.join(EMBEDDINGS_DIR, f"qa_{idx}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No embedding found for question index {idx}")
    with open(path, 'rb') as f:
        return pickle.load(f)

def grade_student_answer(question_idx, student_answer_text):
    entry = load_question_entry(question_idx)
    model = SentenceTransformer(MODEL_NAME)

    student_emb = model.encode([student_answer_text])[0].reshape(1, -1)
    ref_emb = np.array(entry["reference_embedding"]).reshape(1, -1)

    sim = cosine_similarity(student_emb, ref_emb)[0][0]  # value between 0 and 1
    raw_score = sim * 10.0  # scale similarity to score

    score = round(min(max(raw_score, 0), 10), 1)

    # Basic feedback logic
    if score < 4:
        feedback = "❌ Too low — key concepts are missing."
    elif score < 7:
        feedback = "⚠️ Partial answer — try including more detail or examples."
    elif score < 9:
        feedback = "✅ Good answer — could improve with more depth."
    else:
        feedback = "🌟 Excellent — well-covered and accurate."

    return {
        "score": score,
        "similarity": round(sim, 3),
        "feedback": feedback,
        "reference": entry["reference_answer"]
    }

if __name__ == "__main__":
    idx = int(input("🔢 Enter question index (0-n): "))
    print("\n✍️ Paste your answer below:")
    stu_ans = input("> ")
    result = grade_student_answer(idx, stu_ans)

    print(f"\n🧠 Similarity: {result['similarity']}")
    print(f"📊 Score: {result['score']} / 10")
    print(f"💬 Feedback: {result['feedback']}")
    print(f"\n📚 Reference Answer:\n{result['reference']}")
