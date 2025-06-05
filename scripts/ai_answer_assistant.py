import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os
import datetime

# === CONFIG ===
GOOGLE_API_KEY = "AIzaSyBfvJTPipAYktCN5TXKdVRZoNPwo_oJZOA"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
log_file = "./answer_log.json"

# === RAG Search ===
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import pickle

def search_chunks(query, k=3):
    # Load FAISS index and embeddings manually (no deprecated args)
    with open("faiss_index/faiss_store.pkl", "rb") as f:
        db = pickle.load(f)
    
    docs = db.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in docs])


# === MODEL ANSWER GENERATION ===
def get_model_answer(question):
    context = search_chunks(question, k=3)
    prompt = f"""
You are a Computer Science professor. Use the following textbook content to write a clear, exam-oriented answer.

📘 Textbook Context:
\"\"\"
{context}
\"\"\"

✍️ Question: {question}

✅ Instructions:
- Keep it between 100–150 words
- Include definitions, examples, and key concepts
- Use clear academic language, short paragraphs or bullet points
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"❌ Error generating model answer: {e}")
        return None

# === FEEDBACK GENERATION ===
def get_feedback(question, student_answer, model_answer):
    context = search_chunks(question, k=3)
    prompt = f"""
You are a computer science tutor evaluating a student's written exam answer. Use the textbook content below to assess it.

📘 Textbook Context:
\"\"\"
{context}
\"\"\"

✍️ Question: {question}

✅ Model Answer:
{model_answer}

🧑 Student's Answer:
{student_answer}

💬 Feedback Guidelines:
- Speak directly to the student using "you"/"your"
- Briefly highlight strengths and missing points
- Give one actionable tip
- Then provide a score out of 10
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"❌ Error generating feedback: {e}")
        return "Could not generate feedback."

# === SCORING ===
def score_answer(student_answer, model_answer):
    student_vec = embedding_model.encode([student_answer])[0].reshape(1, -1)
    model_vec = embedding_model.encode([model_answer])[0].reshape(1, -1)
    sim = cosine_similarity(student_vec, model_vec)[0][0]
    score = round(min(sim * 10, 10.0), 1)
    return score, round(sim, 3)

# === LOGGING ===
def log_result(question, student_answer, model_answer, sim_score, score, feedback):
    result = {
        "timestamp": str(datetime.datetime.now()),
        "question": question,
        "student_answer": student_answer,
        "model_answer": model_answer,
        "similarity_score": float(sim_score),
        "score": float(score),
        "feedback": str(feedback)
    }
    try:
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                history = json.load(f)
        else:
            history = []
    except json.JSONDecodeError:
        history = []

    history.append(result)
    with open(log_file, "w") as f:
        json.dump(history, f, indent=2)

# === MAIN CLI INTERFACE ===
def main():
    print("📚 AI-Powered Answer Writing Assistant (RAG + Gemini)")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("📝 Enter your question:\n> ").strip()
        if question.lower() == "exit":
            break

        student_answer = input("\n✍️ Enter your answer:\n> ").strip()
        if student_answer.lower() == "exit":
            break

        print("\n🔄 Generating model answer from textbook content...")
        model_answer = get_model_answer(question)
        if not model_answer:
            continue

        score, sim = score_answer(student_answer, model_answer)
        feedback = get_feedback(question, student_answer, model_answer)

        print("\n✅ Model Answer:\n", model_answer)
        print(f"\n📊 Similarity Score: {sim}")
        print(f"📈 Estimated Score: {score}/10")
        print(f"💬 Gemini Feedback:\n{feedback}")

        log_result(question, student_answer, model_answer, sim, score, feedback)
        print("\n✅ Result saved to log.\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
