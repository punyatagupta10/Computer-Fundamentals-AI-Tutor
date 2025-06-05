import os
import re
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter

RAW_DOCS_DIR = "./data/raw_data/"
PROCESSED_DIR = "./data/processed_chunks/"


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9.,;:?!()\[\]\s]', '', text)
    return text.strip()

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

def process_folder():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    for subject in os.listdir(RAW_DOCS_DIR):
        subject_path = os.path.join(RAW_DOCS_DIR, subject)

        if os.path.isdir(subject_path):
            for filename in os.listdir(subject_path):
                if filename.endswith(".pdf"):
                    print(f"Processing: {subject}/{filename}")
                    raw_text = extract_text_from_pdf(os.path.join(subject_path, filename))
                    cleaned = clean_text(raw_text)
                    chunks = split_text(cleaned)

                    out_file = os.path.join(PROCESSED_DIR, f"{subject}_{filename.replace('.pdf', '.txt')}")
                    with open(out_file, 'w', encoding='utf-8') as f:
                        for chunk in chunks:
                            f.write(chunk + "\n\n")

if __name__ == "__main__":
    process_folder()
