import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import os

# === Load API Key ===
GOOGLE_API_KEY = "AIzaSyAw_Y5TAEjeAcdX29gFvXN6JFE5KONAnY0"
genai.configure(api_key=GOOGLE_API_KEY)

# === Models ===
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# === Cached Vector Store Loader ===
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

@st.cache_resource
def load_vector_store():
    return FAISS.load_local(
        "data/embeddings/faiss_index", 
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        index_name="faiss_store"
    )

# === RAG Chunk Retriever ===
def search_chunks(query, k=3):
    try:
        db = load_vector_store()
        docs = db.similarity_search(query, k=k)
        return "\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return f"❌ Error loading vector store: {e}"

# === Model Answer Generator ===
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

# === Feedback Generator ===
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

# === Similarity Score Calculator ===
def score_answer(student_answer, model_answer):
    student_vec = embedding_model.encode([student_answer])[0].reshape(1, -1)
    model_vec = embedding_model.encode([model_answer])[0].reshape(1, -1)
    sim = cosine_similarity(student_vec, model_vec)[0][0]
    score = round(min(sim * 10, 10.0), 1)
    return score, round(sim, 3)

# === Chatbot RAG Mode ===
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

# === Streamlit UI ===
st.set_page_config(page_title="Computer Fundamentals Tutor", layout="centered")
st.title("🧠 Computer Fundamentals Tutor")

tabs = st.tabs(["✍️ Answer Practice", "💬 Ask Doubts"])

# === TAB 1: Answer Writing ===
with tabs[0]:
    st.subheader("✍️ Write and Evaluate Your Answer")

    sample_q = st.selectbox("🎯 Try a sample question or write your own:",
        [
            "What is a deadlock in operating systems?",
            "Explain the sliding window protocol.",
            "Describe DNS resolution process.",
            "What is the difference between TCP and UDP?",
            "Write your own question..."
        ])

    if sample_q == "Write your own question...":
        question = st.text_area("Enter your question", height=100)
    else:
        question = sample_q

    student_answer = st.text_area("Write your answer here", height=200)

    if st.button("🧠 Evaluate My Answer"):
        if not question or not student_answer:
            st.warning("Please enter both question and answer.")
        else:
            with st.spinner():
                model_answer, context = get_model_answer(question)

            
            st.markdown("Here is what all you could have written:")
            st.markdown(model_answer)

            score, sim = score_answer(student_answer, model_answer)

            with st.spinner():
                feedback = get_feedback(question, student_answer, model_answer)

            st.markdown(f"**📊 Similarity Score:** `{sim}`")
            st.markdown(f"**📈 Estimated Score:** `{score}/10`")
            st.markdown("**💬 Feedback:**")
            st.markdown(feedback)

# === TAB 2: Chat Mode ===
with tabs[1]:
    st.subheader("💬 Ask Any OS/CN Question")

    query = st.text_input("Ask your question here")
    answer_type = st.radio("Choose Answer Format", ["Exam-style Answer", "Detailed Explanation"])

    if st.button("🤖 Get Answer"):
        if not query:
            st.warning("Please enter a question.")
        else:
            with st.spinner():
                answer, context = rag_chat_response(query, answer_type)

            if answer.startswith("Gemini error:"):
                st.error(answer)
            else:
        
                st.markdown(f"Answer for {query} in **{answer_type}** format:")
                st.markdown(answer)

        
