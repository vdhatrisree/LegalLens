<img width="1560" height="932" alt="image" src="https://github.com/user-attachments/assets/8fdd7b03-1d95-43cb-9ec1-e978e9e21497" /># ⚖️ LegalLens

An AI-powered contract intelligence platform that reads legal documents, flags risky clauses, detects missing or ambiguous terms, and answers plain-English questions about what you signed — grounded entirely in the uploaded document. LegalLens combines classical ML, deep learning, semantic search, and retrieval-augmented generation into a single end-to-end pipeline.

## 🚀 Overview

LegalLens is built to make contract review accessible without requiring a legal background. Rather than a single black-box model, it's architected as a pipeline of specialized components — clause extraction, classification, risk scoring, gap detection, and a document-grounded chatbot — each independently trained, evaluated, and benchmarked.

The system is designed around a core principle: every answer must be traceable back to the source text. Out-of-scope questions are explicitly refused rather than hallucinated, and every risk or gap flag is backed by a specific clause the user can inspect.

## ✨ Features

- 📄 PDF upload with automatic text extraction and clause segmentation
- 🏷️ Clause classification across 6 categories (Payment, Termination, Confidentiality, Liability, IP, General/Administrative)
- ⚠️ Deep learning–based risk detection (fine-tuned DistilBERT)
- 🔍 Gap & ambiguity detection — flags clauses that reference amounts, deadlines, or conditions without specifying them
- 💬 Document-grounded chatbot (RAG) with conversational memory
- 🧭 Semantic clause search via sentence embeddings + FAISS
- 🛡️ Hallucination-resistant answering with a similarity-threshold refusal gate
- 📊 Automated evaluation harness with reproducible accuracy benchmarks
- 🐳 Fully containerized with Docker
- 📈 Experiment tracking via MLflow
- 🎨 Custom-designed interface

## 🛠️ Tech Stack

**Backend**
- FastAPI
- Python

**Machine Learning / NLP**
- scikit-learn (TF-IDF + Logistic Regression clause classifier)
- Hugging Face Transformers (DistilBERT risk classifier, FLAN-T5 generator)
- Sentence-Transformers (semantic embeddings)
- FAISS (vector similarity search)
- PyTorch

**Data**
- CUAD (Contract Understanding Atticus Dataset)
- LEDGAR (SEC contract provisions, via LexGLUE)

**Infrastructure**
- Docker
- MLflow
- Uvicorn

**Frontend**
- HTML5 / CSS3 (custom design system)
- Jinja2 templating

## 📦 Model Weights

Two of the trained models — the DistilBERT risk classifier and the FLAN-T5 RAG generator — exceed GitHub's file size limits and are hosted separately on Kaggle:

🔗 **[Download model weights](https://www.kaggle.com/datasets/dhatrivunnava/legal-lens)**

The dataset contains 9 files, each prefixed to indicate which model folder it belongs to. After downloading, sort them into two folders in the project root, removing the prefix:

## Folder Structure
```
LegalLens
│
├── main.py                       # FastAPI application entry point
├── app.py                        # Core ML pipeline (extraction, classification, RAG)
├── evaluate.py                   # Automated evaluation harness
├── legallens_eval_set.json       # Ground-truth benchmark question set
├── eval_results.json             # Latest evaluation run output
│
├── templates/
│   ├── index.html                 # Upload page
│   ├── result.html                # Analysis results
│   ├── search.html                # Semantic clause search
│   └── chat.html                  # Document chatbot
│
├── static/
│   └── style.css                  # Design system
│
├── models/
│   ├── clause_classifier_final.pkl
│   ├── tfidf_vectorizer_final.pkl
│   ├── risk_model.pkl
│   ├── risk_vectorizer.pkl
│   ├── clause_pool.pkl
│   ├── clause_embeddings.npy
│   ├── clause_index.faiss
│   │
│   ├── bert_risk_model/            # Downloaded from Kaggle (see Model Weights)
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   │
│   └── flan_t5_rag/                # Downloaded from Kaggle (see Model Weights)
│       ├── config.json
│       ├── generation_config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── tokenizer_config.json
│
├── uploads/                        # Uploaded documents
├── reports/                        # Generated risk reports
├── Dockerfile
├── .dockerignore
└── requirements.txt
```
## ⚙️ Getting Started

Clone the repository

```bash
git clone https://github.com/vdhatrisree/LegalLens.git
cd LegalLens
```

Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Download the model weights

The DistilBERT risk classifier and FLAN-T5 generation model are too large for GitHub and are hosted on Kaggle:

🔗 [https://www.kaggle.com/datasets/dhatrivunnava/legal-lens](https://www.kaggle.com/datasets/dhatrivunnava/legal-lens)

Download all 9 files from that dataset, then create two folders in the project root and sort the files in, removing the `__` prefix from each filename:

```
LegalLens/
├── bert_risk_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
│
├── flan_t5_rag/
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
```

For example, `bert_risk_model__config.json` → `bert_risk_model/config.json`, and `flan_t5_rag__model.safetensors` → `flan_t5_rag/model.safetensors`.

Create the runtime folders

```bash
mkdir uploads
mkdir reports
```

Run the application

```bash
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000` in your browser.

Run with Docker instead

```bash
docker build -t legallens .
docker run -p 8000:8000 legallens
```

## 🏗️ Architecture

LegalLens follows a modular pipeline rather than a single monolithic model:

- **Extraction Layer** — parses PDFs and segments text into clean, sentence-level clauses
- **Classification Layer** — a TF-IDF + Logistic Regression model (trained on a combined CUAD + LEDGAR corpus) labels each clause by legal category
- **Risk Layer** — a fine-tuned DistilBERT model flags clauses carrying elevated risk
- **Gap Detection Layer** — rule-based pattern matching identifies clauses referencing amounts, deadlines, or conditions that are left unspecified
- **Retrieval Layer** — Sentence-Transformer embeddings + FAISS enable semantic search across clauses
- **Generation Layer** — a FLAN-T5 model generates natural-language answers, constrained to retrieved context and gated by a similarity threshold to prevent hallucination on out-of-scope questions

## How the website looks
<img width="1705" height="957" alt="image" src="https://github.com/user-attachments/assets/6dcf19f9-aee5-4099-982b-d284482d00c7" />

<img width="1693" height="965" alt="image" src="https://github.com/user-attachments/assets/35409461-9f84-446e-a6d0-31677bdbdb01" />

<img width="1562" height="951" alt="image" src="https://github.com/user-attachments/assets/dd690d0d-2e94-4af5-ab40-a1f6ebc760ad" />

<img width="1542" height="933" alt="image" src="https://github.com/user-attachments/assets/e8b40d67-22ba-48d0-80f7-8f5e66c79ba1" />

<img width="1560" height="932" alt="image" src="https://github.com/user-attachments/assets/b4ff0c68-d4fa-400b-89c4-eaaab9100f5d" />

## 📊 Evaluation

LegalLens includes a reproducible evaluation harness (`evaluate.py`) that benchmarks the system against a hand-verified ground-truth question set spanning factual lookup, gap detection, and out-of-scope refusal:

| Metric                                  | Score |
| Clause classification accuracy          | 94.6% |
| Risk detection accuracy (DistilBERT)    | 99.7% |
| Factual QA accuracy                     | 100% |
| Hallucination resistance                | 100% |
| Gap/ambiguity detection coverage        | 100% |

## 🔮 Future Improvements

- Fine-tuned generation model specialized per contract type (rental, employment, vendor)
- Larger, professionally-annotated evaluation set
- Contradiction detection between clauses (entailment-based)
- Automatic document summarization
- Multi-document comparison
- Cloud deployment with autoscaling
- CI/CD pipeline with automated regression testing on every model update

## 📚 Highlights

Building LegalLens involved:
- Designing and training multiple specialized ML models rather than relying on a single general-purpose model
- Building a hybrid rule-based + neural architecture, where deterministic logic (gap detection, refusal gating) compensates for known weaknesses in small language models
- Constructing a reproducible evaluation methodology to measure and iterate on model quality with evidence, not intuition
- Migrating from a monolithic Flask prototype to a FastAPI + Docker production architecture

