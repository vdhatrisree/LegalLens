"""
LegalLens Evaluation Script
Runs the ground-truth question set (legallens_eval_set.json) through the
RAG chatbot pipeline and computes accuracy metrics across 3 question types:
factual, gap, not_mentioned.

Usage: python evaluate.py
"""

import json
import os
from app import (
    extract_text_from_pdf, split_into_clauses, detect_gaps_and_vagueness,
    build_document_index, rag_answer_from_document
)

# Map short names used in the eval set to actual PDF filenames
DOC_MAP = {
    "rental": "sample_rental_agreement.pdf",
    "car": "sample_car_purchase_agreement.pdf",
    "employment": "sample_employment_contract.pdf"
}

UPLOAD_FOLDER = "uploads"


def load_eval_set(path="legallens_eval_set.json"):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def prepare_documents():
    """Extract clauses and build a FAISS index for each sample document once."""
    prepared = {}
    for short_name, filename in DOC_MAP.items():
        path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(path):
            print(f"WARNING: {path} not found. Upload it via the website first, or place it in /uploads.")
            continue
        text = extract_text_from_pdf(path)
        clauses = split_into_clauses(text)
        gaps = detect_gaps_and_vagueness(clauses)
        doc_index, doc_clauses = build_document_index(clauses)
        prepared[short_name] = {
            "clauses": clauses,
            "gaps": gaps,
            "doc_index": doc_index,
            "doc_clauses": doc_clauses
        }
    return prepared


def contains_expected(answer, expected):
    """Loose match: check if key parts of the expected answer appear in the model's answer."""
    if expected is None:
        return None
    answer_low = answer.lower()
    expected_low = expected.lower()
    # Check if the core expected phrase (or its key number/word) appears
    return expected_low in answer_low or any(
        token in answer_low for token in expected_low.split() if len(token) > 3
    )


def is_not_mentioned(answer):
    return "not mentioned" in answer.lower()


def evaluate():
    eval_set = load_eval_set()
    documents = prepare_documents()

    results = []
    correct_factual = 0
    total_factual = 0
    correct_not_mentioned = 0
    total_not_mentioned = 0
    gap_flagged_by_detector = 0
    total_gap = 0

    for case in eval_set["cases"]:
        doc_key = case["document"]
        if doc_key not in documents:
            continue

        doc_data = documents[doc_key]
        question = case["question"]
        case_type = case["type"]

        answer, sources = rag_answer_from_document(
            question, doc_data["doc_index"], doc_data["doc_clauses"]
        )

        row = {
            "id": case["id"],
            "document": doc_key,
            "question": question,
            "type": case_type,
            "model_answer": answer,
        }

        if case_type == "factual":
            total_factual += 1
            is_correct = contains_expected(answer, case["expected_answer"])
            row["expected"] = case["expected_answer"]
            row["correct"] = is_correct
            if is_correct:
                correct_factual += 1

        elif case_type == "not_mentioned":
            total_not_mentioned += 1
            is_correct = is_not_mentioned(answer)
            row["expected"] = "This is not mentioned in the document."
            row["correct"] = is_correct
            if is_correct:
                correct_not_mentioned += 1

        elif case_type == "gap":
            total_gap += 1
            gap_snippet = case.get("gap_snippet", "")
            gap_texts_list = [g["clause"].lower() for g in doc_data["gaps"]]
            flagged = any(gap_snippet in text for text in gap_texts_list) if gap_snippet else False
            row["gap_flagged_by_detector"] = flagged
            row["gap_correctly_surfaced"] = "(Note:" in answer
            if flagged:
                gap_flagged_by_detector += 1

        results.append(row)

    # Print detailed results
    print("=" * 70)
    print("LEGALLENS EVALUATION RESULTS")
    print("=" * 70)
    for r in results:
        print(f"\n[{r['id']}] ({r['type']}) {r['document']}: {r['question']}")
        print(f"  Model answer: {r['model_answer'][:120]}")
        if r["type"] in ("factual", "not_mentioned"):
            status = "PASS" if r["correct"] else "FAIL"
            print(f"  Expected: {r.get('expected')}")
            print(f"  Result: {status}")
        elif r["type"] == "gap":
            print(f"  Gap detector flagged related clause: {r['gap_flagged_by_detector']}")
            print(f"  Gap correctly surfaced to user: {r['gap_correctly_surfaced']}")

    # Print summary metrics
    print("\n" + "=" * 70)
    print("SUMMARY METRICS")
    print("=" * 70)
    if total_factual:
        print(f"Factual accuracy: {correct_factual}/{total_factual} = {100*correct_factual/total_factual:.1f}%")
    if total_not_mentioned:
        print(f"Hallucination resistance (correctly said 'not mentioned'): {correct_not_mentioned}/{total_not_mentioned} = {100*correct_not_mentioned/total_not_mentioned:.1f}%")
    if total_gap:
        print(f"Gap detector coverage: {gap_flagged_by_detector}/{total_gap} = {100*gap_flagged_by_detector/total_gap:.1f}%")

    # Save results to a JSON file for later comparison (e.g. before/after model upgrade)
    with open("eval_results.json", "w", encoding="utf-8-sig") as f:
        json.dump(results, f, indent=2)
    print("\nDetailed results saved to eval_results.json")


if __name__ == "__main__":
    evaluate()
