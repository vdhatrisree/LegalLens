from flask import Flask, render_template, request
import pdfplumber
import os
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from transformers import AutoModelForSeq2SeqLM
from flask import Flask, render_template, request, session, redirect

app = Flask(__name__)

app.secret_key = "legallens-secret-key"

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

clause_classifier = joblib.load("clause_classifier_final.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer_final.pkl")
risk_model = joblib.load("risk_model.pkl")
risk_vectorizer = joblib.load("risk_vectorizer.pkl")
bert_tokenizer = AutoTokenizer.from_pretrained("bert_risk_model")
bert_model = AutoModelForSequenceClassification.from_pretrained("bert_risk_model")
bert_model.eval()
embedder = SentenceTransformer("all-MiniLM-L6-v2")
faiss_index = faiss.read_index("clause_index.faiss")

with open("clause_pool.pkl", "rb") as f:
    clause_pool = pickle.load(f)
    t5_tokenizer = AutoTokenizer.from_pretrained("flan_t5_rag")
    t5_model = AutoModelForSeq2SeqLM.from_pretrained("flan_t5_rag")

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

import re

def split_into_clauses(text):
    # Join wrapped lines back into continuous paragraphs
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    # Protect common abbreviations from being treated as sentence endings
    text = re.sub(r'\bRs\.\s*', 'Rs__DOT__ ', text)
    text = re.sub(r'\bNo\.\s*', 'No__DOT__ ', text)
    text = re.sub(r'\bMr\.\s*', 'Mr__DOT__ ', text)
    text = re.sub(r'\bMs\.\s*', 'Ms__DOT__ ', text)
    text = re.sub(r'\bDr\.\s*', 'Dr__DOT__ ', text)
    text = re.sub(r'\bU\.S\.\s*', 'U_S__DOT__ ', text)

    # Split into sentences using punctuation, keeping numbered clauses intact
    raw_clauses = re.split(r'(?<=[.;])\s+(?=[A-Z0-9])', text)

    # Restore protected abbreviations
    raw_clauses = [c.replace('__DOT__', '.') for c in raw_clauses]

    clauses = [c.strip() for c in raw_clauses if len(c.strip()) > 15]
    return clauses

def detect_risky_clauses(clauses):
    risky = []
    for clause in clauses:
        for keyword in RISKY_KEYWORDS:
            if keyword.lower() in clause.lower():
                risky.append(clause)
                break
    return risky

def detect_gaps_and_vagueness(clauses):
    amount_triggers = ["fee", "charge", "penalty", "rate", "deposit", "interest", "price", "increase", "payment", "cost"]
    amount_pattern = re.compile(r'(\d+(\.\d+)?\s?%|\$|₹|USD|INR|Rs\.)')

    time_triggers = ["notice", "renewal", "expire", "warranty", "cooling-off", "grace period", "termination"]
    time_pattern = re.compile(r'\d+\s?(day|days|week|weeks|month|months|year|years)')
    # These phrases make timing explicit/definitive, so they should NOT be flagged as vague
    definitive_time_phrases = ["immediately", "without notice", "at once", "forthwith"]

    vague_phrases = ["reasonable", "as needed", "at its discretion", "from time to time", "may vary", "sole discretion", "as applicable", "if necessary"]

    gaps = []
    for clause in clauses:
        lower = clause.lower()

        if any(trigger in lower for trigger in amount_triggers) and not amount_pattern.search(clause):
            gaps.append({"clause": clause, "issue": "Mentions a financial term but no specific amount or percentage is stated."})

        elif (any(trigger in lower for trigger in time_triggers)
              and not time_pattern.search(lower)
              and not any(phrase in lower for phrase in definitive_time_phrases)):
            gaps.append({"clause": clause, "issue": "Mentions a time-sensitive term but no specific duration is stated."})

        elif any(phrase in lower for phrase in vague_phrases):
            gaps.append({"clause": clause, "issue": "Uses vague or subjective language instead of a clear, fixed rule."})

    return gaps

def calculate_risk_score(clauses, risky_clauses):
    if len(clauses) == 0:
        return 0
    score = (len(risky_clauses) / len(clauses)) * 100
    return round(score, 2)

def calculate_gap_score(clauses, gaps):
    if len(clauses) == 0:
        return 0
    score = (len(gaps) / len(clauses)) * 100
    return round(score, 2)

def calculate_gap_score(clauses, gaps):
    if len(clauses) == 0:
        return 0
    score = (len(gaps) / len(clauses)) * 100
    return round(score, 2)

def classify_clauses(clauses):
    if len(clauses) == 0:
        return []
    vectors = tfidf_vectorizer.transform(clauses)
    predictions = clause_classifier.predict(vectors)
    return list(predictions)

def detect_risky_clauses_ml(clauses):
    if len(clauses) == 0:
        return []
    vectors = risk_vectorizer.transform(clauses)
    predictions = risk_model.predict(vectors)
    risky = [clause for clause, label in zip(clauses, predictions) if label == "Risky"]
    return risky

def detect_risky_clauses_bert(clauses):
    if len(clauses) == 0:
        return []

    risky = []
    for clause in clauses:
        inputs = bert_tokenizer(clause, truncation=True, padding=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            outputs = bert_model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
        if prediction == 1:
            risky.append(clause)
    return risky

def search_similar_clauses(query, k=3):
    query_embedding = embedder.encode([query])
    distances, indices = faiss_index.search(np.array(query_embedding), k)
    results = [clause_pool[i] for i in indices[0]]
    return results

def rag_answer(question, k=3, distance_threshold=1.1):
    query_embedding = embedder.encode([question])
    distances, indices = faiss_index.search(np.array(query_embedding), k)

    best_distance = distances[0][0]

    if best_distance > distance_threshold:
        return "This is not mentioned in the document.", []

    retrieved_clauses = [clause_pool[i] for i in indices[0]]

    context = "\n".join(retrieved_clauses)
    prompt = f"Answer the question using only the context below. If the answer is not in the context, say 'This is not mentioned in the document.'\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    inputs = t5_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = t5_model.generate(**inputs, max_length=100, min_length=5, repetition_penalty=1.8, no_repeat_ngram_size=3)
    answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer, retrieved_clauses

def build_document_index(clauses):
    if len(clauses) == 0:
        return None, None
    doc_embeddings = embedder.encode(clauses)
    doc_index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    doc_index.add(np.array(doc_embeddings))
    return doc_index, clauses

def rag_answer_from_document(question, doc_index, doc_clauses, k=3, distance_threshold=1.1):
    query_embedding = embedder.encode([question])
    distances, indices = doc_index.search(np.array(query_embedding), k)

    best_distance = distances[0][0]

    if best_distance > distance_threshold:
        return "This is not mentioned in the document.", []

    retrieved_clauses = [doc_clauses[i] for i in indices[0]]

    # Only check the SINGLE BEST-MATCHING clause (top-1) against gap detector, not all k
    gaps_in_doc = detect_gaps_and_vagueness(doc_clauses)
    gap_clause_texts = {g["clause"]: g["issue"] for g in gaps_in_doc}

    top_clause = retrieved_clauses[0]
    matched_gap_issue = gap_clause_texts.get(top_clause)

    context = "\n".join(retrieved_clauses)

    if matched_gap_issue:
        prompt = f"Answer the question using only the context below. If the answer is not in the context, say 'This is not mentioned in the document.'\n\nContext:\n{context}\n\nNote: {matched_gap_issue}\n\nQuestion: {question}\nAnswer:"
    else:
        prompt = f"Answer the question using only the context below. If the answer is not in the context, say 'This is not mentioned in the document.'\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    inputs = t5_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = t5_model.generate(**inputs, max_length=100, min_length=5, repetition_penalty=1.8, no_repeat_ngram_size=3)
    answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

    if matched_gap_issue and matched_gap_issue.lower() not in answer.lower():
        answer = f"{answer} (Note: {matched_gap_issue})"

    return answer, retrieved_clauses

def generate_report(clauses, risky_clauses, risk_score):
    report = "LegalLens - Contract Risk Report\n"
    report += "=" * 40 + "\n\n"
    report += f"Total Clauses: {len(clauses)}\n"
    report += f"Risky Clauses: {len(risky_clauses)}\n"
    report += f"Risk Score: {risk_score}%\n\n"
    report += "Risky Clauses Found:\n"
    for clause in risky_clauses:
        report += f"- {clause}\n"
    return report


@app.route("/")
def home():
    return render_template("index.html")

RISKY_KEYWORDS = [
    "penalty",
    "termination",
    "liability",
    "indemnify",
    "non-compete",
    "auto-renew",
    "without notice",
    "sole discretion"
]

@app.route("/upload", methods=["POST"])
def upload():
    if "pdf_file" not in request.files or request.files["pdf_file"].filename == "":
        return "No file selected. Go back and choose a PDF.", 400

    pdf_file = request.files["pdf_file"]
    pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], pdf_file.filename)
    pdf_file.save(pdf_path)

    text = extract_text_from_pdf(pdf_path)
    clauses = split_into_clauses(text)
    risky_clauses = detect_risky_clauses_bert(clauses)
    risk_score = calculate_risk_score(clauses, risky_clauses)
    categories = classify_clauses(clauses)    
    report = generate_report(clauses, risky_clauses, risk_score)

    report_filename = pdf_file.filename.replace(".pdf", "_report.txt")
    report_path = os.path.join("reports", report_filename)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    clause_data = list(zip(clauses, categories))

    session["current_clauses"] = clauses

    return render_template(
        "result.html",
        clause_data=clause_data,
        risky_clauses=risky_clauses,
        risk_score=risk_score,
        report_filename=report_filename
    )

@app.route("/search", methods=["GET", "POST"])
def search():
    results = []
    query = ""
    answer = None
    if request.method == "POST":
        query = request.form["query"]
        answer, results = rag_answer(query)
    return render_template("search.html", results=results, query=query, answer=answer)

@app.route("/chat", methods=["GET", "POST"])
def chat():
    if "current_clauses" not in session:
        return "Please upload a document first.", 400

    if "chat_history" not in session:
        session["chat_history"] = []

    clauses = session["current_clauses"]
    doc_index, doc_clauses = build_document_index(clauses)

    if request.method == "POST":
        question = request.form["question"]
        answer, sources = rag_answer_from_document(question, doc_index, doc_clauses)

        chat_history = session["chat_history"]
        chat_history.append({"question": question, "answer": answer, "sources": sources})
        session["chat_history"] = chat_history

    return render_template("chat.html", chat_history=session.get("chat_history", []))

@app.route("/clear_chat")
def clear_chat():
    session["chat_history"] = []
    return redirect("/chat")


from flask import send_from_directory

@app.route("/download/<filename>")
def download(filename):
    return send_from_directory("reports", filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)