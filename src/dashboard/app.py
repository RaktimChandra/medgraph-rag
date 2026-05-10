"""
Flask API Backend — Upgrade 3
Powers the live comparison dashboard with real pipeline calls.

Endpoints:
  POST /api/query        → runs all 3 pipelines, returns real metrics
  POST /api/judge        → runs LLM-as-a-Judge + BERTScore on demand
  GET  /api/benchmark    → returns pre-run benchmark_results.json
  GET  /api/health       → health check
  GET  /                 → serves the dashboard HTML

Run: python src/dashboard/app.py
Then open: http://localhost:5000
"""

import os
import sys
import json
import time
import threading
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# Add project root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

load_dotenv(ROOT / ".env")

# Lazy imports — only load pipelines when first used
_p1 = _p2 = _p3 = _eval = None
_lock = threading.Lock()

app = Flask(__name__, static_folder=str(ROOT / "src" / "dashboard"))
CORS(app)


# ── Lazy pipeline loader ─────────────────────────────────────────────────────

def get_pipelines():
    global _p1, _p2, _p3, _eval
    with _lock:
        if _p1 is None:
            print("  Loading pipelines (first request)...")
            from pipelines.pipeline1_llm_only import LLMOnlyPipeline
            from pipelines.pipeline2_basic_rag import BasicRAGPipeline
            from pipelines.pipeline3_graphrag  import GraphRAGPipeline
            from evaluation.llm_judge          import EvaluationSuite
            _p1   = LLMOnlyPipeline()
            _p2   = BasicRAGPipeline()
            _p3   = GraphRAGPipeline()
            _eval = EvaluationSuite()
            print("  Pipelines ready.")
    return _p1, _p2, _p3, _eval


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the dashboard HTML."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "timestamp": time.time()})


@app.route("/api/benchmark")
def benchmark():
    """Return pre-run benchmark results JSON."""
    results_path = ROOT / "benchmark_results.json"
    if results_path.exists():
        with open(results_path) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "benchmark_results.json not found — run benchmark_runner.py first"}), 404


@app.route("/api/query", methods=["POST"])
def run_query():
    """
    Run all 3 pipelines on a question and return real metrics.

    Request body: { "question": "..." }
    Response:     { "llm_only": {...}, "basic_rag": {...}, "graphrag": {...} }
    """
    body     = request.get_json(force=True) or {}
    question = body.get("question", "").strip()

    if not question:
        return jsonify({"error": "question is required"}), 400
    if len(question) > 500:
        return jsonify({"error": "question too long (max 500 chars)"}), 400

    print(f"\n[API] /api/query — {question[:70]}...")

    try:
        p1, p2, p3, _ = get_pipelines()

        # Run pipelines (sequentially for simplicity — parallel optional)
        t0 = time.time()

        print("  Running Pipeline 1 (LLM-Only)...")
        r1 = p1.query(question)

        print("  Running Pipeline 2 (Basic RAG)...")
        r2 = p2.query(question)

        print("  Running Pipeline 3 (GraphRAG)...")
        r3 = p3.query(question)

        total_time = round(time.time() - t0, 2)
        print(f"  All 3 pipelines done in {total_time}s")

        # Compute live token reduction vs Basic RAG
        reduction_pct = 0
        if r2["tokens_total"] > 0:
            reduction_pct = round(
                (r2["tokens_total"] - r3["tokens_total"]) / r2["tokens_total"] * 100, 1
            )

        return jsonify({
            "question":      question,
            "total_time_s":  total_time,
            "token_reduction_pct": reduction_pct,
            "llm_only": _serialize(r1),
            "basic_rag": _serialize(r2),
            "graphrag":  _serialize(r3),
        })

    except Exception as e:
        print(f"  ERROR: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/judge", methods=["POST"])
def run_judge():
    """
    Run LLM-as-a-Judge + BERTScore on a single answer triple.

    Request body: {
      "question": "...",
      "reference": "...",        ← ground truth answer
      "llm_only_answer": "...",
      "rag_answer": "...",
      "graphrag_answer": "..."
    }
    """
    body = request.get_json(force=True) or {}
    question   = body.get("question",         "").strip()
    reference  = body.get("reference",        "").strip()
    ans_p1     = body.get("llm_only_answer",  "").strip()
    ans_p2     = body.get("rag_answer",       "").strip()
    ans_p3     = body.get("graphrag_answer",  "").strip()

    if not all([question, reference, ans_p1, ans_p2, ans_p3]):
        return jsonify({"error": "All fields are required"}), 400

    print(f"\n[API] /api/judge — {question[:60]}...")

    try:
        _, _, _, evaluator = get_pipelines()

        # LLM-as-a-Judge for all 3
        j1 = evaluator.llm_judge(question, reference, ans_p1)
        j2 = evaluator.llm_judge(question, reference, ans_p2)
        j3 = evaluator.llm_judge(question, reference, ans_p3)

        # BERTScore (batch all 3)
        bert = evaluator.bertscore(
            references=[reference, reference, reference],
            candidates=[ans_p1, ans_p2, ans_p3]
        )

        return jsonify({
            "llm_only": {
                "verdict":        j1["verdict"],
                "passed":         j1["passed"],
                "bertscore_f1":   round(bert["f1_rescaled"], 3),
            },
            "basic_rag": {
                "verdict":        j2["verdict"],
                "passed":         j2["passed"],
                "bertscore_f1":   round(bert["f1_rescaled"], 3),
            },
            "graphrag": {
                "verdict":        j3["verdict"],
                "passed":         j3["passed"],
                "bertscore_f1":   round(bert["f1_rescaled"], 3),
                "bonus_judge":    j3["passed"],
                "bonus_bert":     bert["bonus_threshold_rescaled_hit"],
            },
        })

    except Exception as e:
        print(f"  ERROR: {e}")
        return jsonify({"error": str(e)}), 500


# ── Helpers ───────────────────────────────────────────────────────────────────

def _serialize(result: dict) -> dict:
    """Strip non-serializable fields and format for API response."""
    return {
        "pipeline":          result.get("pipeline"),
        "answer":            result.get("answer", ""),
        "tokens_prompt":     result.get("tokens_prompt", 0),
        "tokens_completion": result.get("tokens_completion", 0),
        "tokens_total":      result.get("tokens_total", 0),
        "cost_usd":          result.get("cost_usd", 0),
        "latency_seconds":   result.get("latency_seconds", 0),
        "context_preview":   (result.get("context_used") or "")[:400],
        "retrieval_meta":    result.get("retrieval_metadata", {}),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("DASHBOARD_PORT", 5000))
    print("=" * 50)
    print(f"MedGraph-RAG Dashboard Backend")
    print(f"Starting on http://localhost:{port}")
    print(f"Open your browser at http://localhost:{port}")
    print("=" * 50)
    app.run(host="0.0.0.0", port=port, debug=False)
