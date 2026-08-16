from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates
import os
from app import detect_gaps_and_vagueness
from app import calculate_gap_score

from app import (
    extract_text_from_pdf, split_into_clauses, detect_risky_clauses_bert,
    calculate_risk_score, classify_clauses, generate_report,
    search_similar_clauses, rag_answer, build_document_index,
    rag_answer_from_document
)

app = FastAPI()
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "uploads"
current_document = {"clauses": None, "chat_history": []}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, pdf_file: UploadFile = File(...)):
    if pdf_file.filename == "":
        return HTMLResponse("No file selected. Go back and choose a PDF.", status_code=400)

    pdf_path = os.path.join(UPLOAD_FOLDER, pdf_file.filename)
    with open(pdf_path, "wb") as f:
        f.write(await pdf_file.read())

    text = extract_text_from_pdf(pdf_path)
    clauses = split_into_clauses(text)
    risky_clauses = detect_risky_clauses_bert(clauses)
    risk_score = calculate_risk_score(clauses, risky_clauses)
    categories = classify_clauses(clauses)
    gaps = detect_gaps_and_vagueness(clauses)
    gap_score = calculate_gap_score(clauses, gaps)
    report = generate_report(clauses, risky_clauses, risk_score)

    report_filename = pdf_file.filename.replace(".pdf", "_report.txt")
    report_path = os.path.join("reports", report_filename)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    clause_data = list(zip(clauses, categories))
    current_document["clauses"] = clauses
    current_document["chat_history"] = []

    return templates.TemplateResponse(request, "result.html", {
        "clause_data": clause_data,
        "risky_clauses": risky_clauses,
        "risk_score": risk_score,
        "gap_score": gap_score,
        "report_filename": report_filename,
        "gaps": gaps
    })


@app.get("/search", response_class=HTMLResponse)
def search_get(request: Request):
    return templates.TemplateResponse(request, "search.html", {"results": [], "query": "", "answer": None})


@app.post("/search", response_class=HTMLResponse)
def search_post(request: Request, query: str = Form(...)):
    answer, results = rag_answer(query)
    return templates.TemplateResponse(request, "search.html", {"results": results, "query": query, "answer": answer})


@app.get("/chat", response_class=HTMLResponse)
def chat_get(request: Request):
    if current_document["clauses"] is None:
        return HTMLResponse("Please upload a document first.", status_code=400)
    return templates.TemplateResponse(request, "chat.html", {"chat_history": current_document["chat_history"]})


@app.post("/chat", response_class=HTMLResponse)
def chat_post(request: Request, question: str = Form(...)):
    if current_document["clauses"] is None:
        return HTMLResponse("Please upload a document first.", status_code=400)

    clauses = current_document["clauses"]
    doc_index, doc_clauses = build_document_index(clauses)
    answer, sources = rag_answer_from_document(question, doc_index, doc_clauses)

    current_document["chat_history"].append({"question": question, "answer": answer, "sources": sources})

    return templates.TemplateResponse(request, "chat.html", {"chat_history": current_document["chat_history"]})


@app.get("/clear_chat")
def clear_chat(request: Request):
    current_document["chat_history"] = []
    return RedirectResponse("/chat", status_code=302)


@app.get("/download/{filename}")
def download(filename: str):
    return FileResponse(f"reports/{filename}", filename=filename)