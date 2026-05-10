"""
Ground Truth Generator — Upgrade 1
Generates reference answers for all 100 benchmark questions using Gemini,
grounded in actual PubMed abstracts retrieved via Entrez.

Why this matters:
- BERTScore needs a reference answer to compare against
- LLM-as-a-Judge needs a reference to grade PASS/FAIL
- Without this file, the entire evaluation suite cannot run

Run: python scripts/generate_ground_truth.py

Output: data/ground_truth.json
Estimated time: 20-30 minutes (API rate limits)
"""

import os
import json
import time
import re
from typing import Optional
from Bio import Entrez
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
Entrez.email = os.getenv("ENTREZ_EMAIL", "raktimchandra26@gmail.com")

QUESTIONS_PATH = "data/questions.json"
OUTPUT_PATH    = "data/ground_truth.json"
PROGRESS_PATH  = "data/ground_truth_progress.json"   # resumable checkpoint

MODEL = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=(
        "You are an expert biomedical scientist. Answer questions with precise, "
        "factual information grounded in peer-reviewed literature. Be specific: "
        "name exact genes, proteins, drugs, pathways, and mechanisms. "
        "Keep answers 2-4 sentences. Prioritize specificity over completeness."
    )
)

RETRIEVAL_PROMPT = """
You have been given the following PubMed abstracts as context:

{abstracts}

---

Using ONLY information from the abstracts above (plus your expert knowledge to fill gaps),
provide a precise reference answer to this question:

Question: {question}

Requirements:
- Name specific entities (gene names, drug names, pathway names)
- Include key mechanistic details
- 2-4 sentences maximum
- No hedging phrases like "may" or "could" unless scientifically required

Reference Answer:"""


# ── PubMed retrieval ─────────────────────────────────────────────────────────

def get_pubmed_abstracts(question: str, n: int = 5) -> str:
    """Retrieve top-n PubMed abstracts relevant to the question."""
    try:
        # Build a focused search query from the question
        # Extract key terms (capitalized words = likely gene/drug names)
        key_terms = re.findall(r'\b[A-Z][A-Z0-9]{1,6}\b|\b[A-Z][a-z]+(?:mab|nib|cin|ine|lib)\b', question)
        search_q = " AND ".join(f'"{t}"' for t in key_terms[:3]) if key_terms else question[:80]
        search_q += " AND (cancer OR tumor OR oncology)"

        handle = Entrez.esearch(db="pubmed", term=search_q, retmax=n, sort="relevance")
        record = Entrez.read(handle)
        pmids  = record["IdList"]

        if not pmids:
            return ""

        handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="abstract", retmode="text")
        abstracts = handle.read()
        return abstracts[:3000]   # cap at 3000 chars to stay within token budget

    except Exception as e:
        print(f"    PubMed fetch failed: {e}")
        return ""


# ── Answer generation ─────────────────────────────────────────────────────────

def generate_answer(question: str, abstracts: str) -> str:
    """Generate a reference answer grounded in PubMed abstracts."""
    prompt = RETRIEVAL_PROMPT.format(
        abstracts=abstracts if abstracts else "(No abstracts retrieved — use expert knowledge)",
        question=question
    )
    try:
        response = MODEL.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=256,
            )
        )
        return response.text.strip()
    except Exception as e:
        print(f"    Gemini error: {e}")
        time.sleep(5)
        return ""


# ── Resumable checkpoint ───────────────────────────────────────────────────────

def load_progress() -> dict:
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return {}


def save_progress(progress: dict):
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Ground Truth Generator")
    print("=" * 60)

    with open(QUESTIONS_PATH) as f:
        data = json.load(f)
    questions = data["questions"]
    print(f"Loaded {len(questions)} questions")

    # Load any saved progress (resumable)
    progress = load_progress()
    answers  = progress.get("answers", {})
    skipped  = 0

    for i, question in enumerate(questions):
        q_id = f"q{i+1:03d}"

        if q_id in answers:
            print(f"  [{i+1:3d}/{len(questions)}] SKIP (cached): {question[:55]}...")
            skipped += 1
            continue

        print(f"\n  [{i+1:3d}/{len(questions)}] {question[:65]}...")

        # Step 1: retrieve PubMed abstracts
        print("    → Fetching PubMed abstracts...")
        abstracts = get_pubmed_abstracts(question)
        time.sleep(0.4)   # NCBI rate limit: max 3 req/sec without API key

        # Step 2: generate reference answer
        print("    → Generating reference answer...")
        answer = generate_answer(question, abstracts)

        if not answer:
            # Fallback: no abstract context
            answer = generate_answer(question, "")

        if answer:
            answers[q_id] = {
                "id":       q_id,
                "question": question,
                "answer":   answer,
                "grounded_in_pubmed": bool(abstracts),
            }
            print(f"    ✓ {answer[:100]}...")
        else:
            print(f"    ✗ Failed to generate answer for q{i+1}")

        # Save checkpoint after every answer
        progress["answers"] = answers
        save_progress(progress)

        # Rate limit: ~2 req/sec to Gemini free tier
        time.sleep(1.2)

    # Build final output
    ground_truth = {
        "metadata": {
            "total":    len(answers),
            "skipped":  skipped,
            "model":    "gemini-1.5-flash",
            "grounded": sum(1 for a in answers.values() if a["grounded_in_pubmed"]),
        },
        "answers": [answers[f"q{i+1:03d}"] for i in range(len(questions))
                    if f"q{i+1:03d}" in answers]
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done! {len(answers)}/{len(questions)} answers generated")
    print(f"Grounded in PubMed: {ground_truth['metadata']['grounded']}")
    print(f"Saved to: {OUTPUT_PATH}")
    print("\nNext: python src/evaluation/benchmark_runner.py")


if __name__ == "__main__":
    main()
