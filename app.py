import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import re
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# === API Key ===
GOOGLE_API_KEY = "AIzaSyAw_Y5TAEjeAcdX29gFvXN6JFE5KONAnY0"
genai.configure(api_key=GOOGLE_API_KEY)

# === Models ===
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_vector_store():
    return FAISS.load_local(
        "data/embeddings/faiss_index",
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        index_name="faiss_store"
    )

def search_chunks(query, k=3):
    try:
        db = load_vector_store()
        docs = db.similarity_search(query, k=k)
        return "\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return f"❌ Error loading vector store: {e}"

def generate_summary(topic):
    context = search_chunks(topic)
    prompt = f"""
📘 Topic Context:
\"\"\"{context}\"\"\"

✍️ Task:
Generate a concise summary of the topic "{topic}" suitable for quick revision.
Use 5–8 bullet points. Cover key ideas, processes, or terms.
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini error: {e}"

def get_model_answer(question):
    context = search_chunks(question)
    prompt = f"""
📘 Context:
\"\"\"{context}\"\"\"

❓ Question:
{question}

📝 Instructions:
You are an expert CS tutor. Provide an **exam-style answer**:
- Structure it like a 10-mark university response.
- Include definitions, reasoning, and examples wherever needed.
- Use a formal and academic tone.
- Ensure the answer is complete and well explained.
- Mention any useful external references if applicable.

✍️ Please begin your answer now.
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip(), context
    except Exception as e:
        return f"Gemini error: {e}", ""

def get_feedback(question, student_answer, model_answer):
    prompt = f"""
You are a CS teacher evaluating a student's answer.

❓ Question:
{question}

✅ Model Answer:
{model_answer}

🧑‍🎓 Student Answer:
{student_answer}

💡 Instructions:
- Speak directly to the student ("you", "your")
- Give a short, constructive feedback (1–2 sentences)
- Suggest one specific improvement
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini error: {e}"

def score_answer(student_answer, model_answer):
    student_vec = embedding_model.encode([student_answer])[0].reshape(1, -1)
    model_vec = embedding_model.encode([model_answer])[0].reshape(1, -1)
    sim = cosine_similarity(student_vec, model_vec)[0][0]
    score = round(min(sim * 10, 10.0), 1)
    return score, round(sim, 3)

def rag_chat_response(question, answer_type):
    context = search_chunks(question)
    prompt = f"""
📘 Context:
\"\"\"{context}\"\"\"

❓ Question:
{question}

📝 Instructions:
You are an expert CS tutor. Provide the answer in the format: "{answer_type}".

- If "Exam-style Answer":
    - Structure like a 10-mark university answer.
    - Include definitions, reasoning, and examples.
    - Keep a formal and academic tone.

- If "Detailed Explanation":
    - Give a step-by-step, intuitive explanation.
    - Use analogies, examples, and clear breakdowns.

✍️ Please begin your answer now.
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip(), context
    except Exception as e:
        return f"Gemini error: {e}", ""

def generate_mcqs(subject, topic, num_questions=5):
    context = search_chunks(topic)
    prompt = f"""
📘 Textbook Context:
\"\"\"{context}\"\"\"

✍️ Task:
Generate {num_questions} multiple-choice questions from the topic "{topic}" in "{subject}".
- Each question should have 4 options (A–D)
- Mark the correct option
- Provide a 1-line explanation

Format each as:
Q: ...
A. ...
B. ...
C. ...
D. ...
Answer: B
Explanation: ...

Now generate the quiz:
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini error: {e}"

def parse_mcqs(text):
    pattern = re.compile(r"Q: (.*?)\nA\\. (.*?)\nB\\. (.*?)\nC\\. (.*?)\nD\\. (.*?)\nAnswer: (.)\nExplanation: (.*?)\n", re.DOTALL)
    matches = pattern.findall(text)
    questions = []
    for q, a, b, c, d, ans, exp in matches:
        questions.append({
            "question": q.strip(),
            "options": {"A": a.strip(), "B": b.strip(), "C": c.strip(), "D": d.strip()},
            "correct": ans.strip(),
            "explanation": exp.strip()
        })
    return questions

# === UI ===
st.set_page_config(page_title="Computer Fundamentals Tutor", layout="centered")
st.title("🧠 Computer Fundamentals Tutor")

if "quiz_scores" not in st.session_state:
    st.session_state.quiz_scores = []

tabs = st.tabs(["✍️ Answer Practice", "💬 Ask Doubts", "🧪 Quiz", "📝 Summary", "📈 My Stats"])

# === Tab 1 ===
with tabs[0]:
    st.subheader("✍️ Write and Evaluate Your Answer")
    sample_q = st.selectbox("🎯 Try a sample question or write your own:",
                        ["What is a deadlock in operating systems?",
                            "Explain the sliding window protocol.",
                            "Describe DNS resolution process.",
                            "What is the difference between TCP and UDP?",
                            "Write your own question..."])
    question = st.text_area("Enter your question", height=100) if sample_q == "Write your own question..." else sample_q
    student_answer = st.text_area("Write your answer here", height=200)

    if st.button("🧠 Evaluate My Answer"):
        if not question or not student_answer:
            st.warning("Please enter both question and answer.")
        else:
            with st.spinner():
                model_answer, _ = get_model_answer(question)
                score, sim = score_answer(student_answer, model_answer)
                feedback = get_feedback(question, student_answer, model_answer)
            st.markdown("### ✅ Suggested Answer")
            st.markdown(model_answer)
            st.markdown(f"**📊 Similarity:** `{sim}`")
            st.markdown(f"**📈 Estimated Score:** `{score}/10`")
            st.markdown("**💬 Feedback:**")
            st.markdown(feedback)

# === Tab 2 ===
with tabs[1]:
    st.subheader("💬 Ask Any OS/CN Question")
    query = st.text_input("Ask your question here")
    answer_type = st.radio("Choose Answer Format", ["Exam-style Answer", "Detailed Explanation"])
    if st.button("🤖 Get Answer"):
        if not query:
            st.warning("Please enter a question.")
        else:
            with st.spinner():
                answer, _ = rag_chat_response(query, answer_type)
            st.markdown(answer)

# === Tab 3 ===
with tabs[2]:
    st.subheader("🧪 Topic-wise Quiz")
    if "quiz_started" not in st.session_state:
        st.session_state.quiz_started = False
    if "quiz_questions" not in st.session_state:
        st.session_state.quiz_questions = []
    if "user_answers" not in st.session_state:
        st.session_state.user_answers = {}

    if not st.session_state.quiz_started:
        subject = st.selectbox("📚 Subject", ["Operating System", "Computer Networks"])
        topic = st.text_input("📌 Topic")
        num = st.slider("🔢 Number of Questions", 1, 10, 5)
        if st.button("🚀 Generate Quiz"):
            if not topic:
                st.warning("Enter a topic.")
            else:
                with st.spinner("Creating quiz..."):
                    quiz_text = generate_mcqs(subject, topic, num)
                if quiz_text.startswith("Gemini error"):
                    st.error(quiz_text)
                else:
                    st.session_state.quiz_questions = parse_mcqs(quiz_text)
                    st.session_state.quiz_topic = topic
                    st.session_state.quiz_started = True
                    st.experimental_rerun()
    else:
        score = 0
        for i, q in enumerate(st.session_state.quiz_questions):
            st.markdown(f"**Q{i+1}. {q['question']}**")
            selected = st.radio(
                f"Answer for Q{i+1}",
                list(q["options"].keys()),
                format_func=lambda k: f"{k}. {q['options'][k]}",
                key=f"quiz_q_{i}"
            )
            st.session_state.user_answers[i] = selected

        if st.button("✅ Submit Quiz"):
            st.markdown("### 📊 Results")
            for i, q in enumerate(st.session_state.quiz_questions):
                correct = q["correct"]
                chosen = st.session_state.user_answers.get(i)
                is_right = correct == chosen
                st.markdown(f"**Q{i+1}: {q['question']}**")
                st.markdown(f"- ✅ Correct Answer: `{correct}`")
                st.markdown(f"- 🧠 Explanation: {q['explanation']}")
                st.markdown(f"- {'🎉 Correct!' if is_right else f'❌ Your Answer: {chosen}'}")
                st.markdown("---")
                if is_right:
                    score += 1
            total = len(st.session_state.quiz_questions)
            st.success(f"🏆 Final Score: {score} / {total}")
            st.session_state.quiz_scores.append({"topic": st.session_state.quiz_topic, "score": score, "total": total})
            st.session_state.quiz_started = False

# === Tab 4 ===
with tabs[3]:
    st.subheader("📝 Topic Summary")
    topic = st.text_input("📚 Enter topic to summarize:")
    if st.button("🧠 Generate Summary"):
        with st.spinner("Generating summary..."):
            summary = generate_summary(topic)
        st.markdown("### ✨ Summary")
        st.markdown(summary)

# === Tab 5 ===
with tabs[4]:
    st.subheader("📈 My Quiz Stats")
    scores = st.session_state.quiz_scores
    if not scores:
        st.info("No quiz data yet.")
    else:
        total_quizzes = len(scores)
        avg_score = round(sum(s["score"] for s in scores) / total_quizzes, 2)
        st.markdown(f"- **Total Quizzes:** {total_quizzes}")
        st.markdown(f"- **Average Score:** {avg_score}")
        for i, s in enumerate(scores[::-1], 1):
            st.markdown(f"**#{i} - {s['topic']}**: `{s['score']} / {s['total']}`")
