import os
import pickle
import numpy as np
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configs
EMBEDDINGS_DIR = "./data/subjective_embeddings/"
MODEL_NAME = "all-MiniLM-L6-v2"

# Load all questions
def load_all_questions():
    questions = []
    for fname in sorted(os.listdir(EMBEDDINGS_DIR)):
        if fname.endswith(".pkl"):
            with open(os.path.join(EMBEDDINGS_DIR, fname), 'rb') as f:
                entry = pickle.load(f)
                questions.append((fname, entry))
    return questions

# Grading function
def grade_answer(reference_embedding, student_answer, model):
    student_emb = model.encode([student_answer])[0].reshape(1, -1)
    ref_emb = np.array(reference_embedding).reshape(1, -1)
    sim = cosine_similarity(student_emb, ref_emb)[0][0]
    raw_score = sim * 10.0
    score = round(min(max(raw_score, 0), 10), 1)

    if score < 4:
        feedback = "❌ Too low — key concepts are missing."
    elif score < 7:
        feedback = "⚠️ Partial answer — try including more detail or examples."
    elif score < 9:
        feedback = "✅ Good answer — could improve with more depth."
    else:
        feedback = "🌟 Excellent — well-covered and accurate."

    return score, sim, feedback

def main():
    model = SentenceTransformer(MODEL_NAME)
    all_questions = load_all_questions()

    print("🎓 Welcome to the CS Subjective Test Bot!")
    print("Type 'exit' at any time to quit.\n")

    while True:
        # Select a random question
        fname, q = random.choice(all_questions)
        print(f"📝 Question: {q['question']}")
        student_ans = input("\n✍️ Your Answer:\n> ").strip()
        if student_ans.lower() == "exit":
            break

        score, sim, feedback = grade_answer(q["reference_embedding"], student_ans, model)

        print(f"\n📊 Score: {score} / 10")
        print(f"🧠 Similarity: {round(sim, 3)}")
        print(f"💬 Feedback: {feedback}")
        print(f"\n📚 Reference Answer:\n{q['reference_answer']}\n")
        print("="*80)

if __name__ == "__main__":
    main()
