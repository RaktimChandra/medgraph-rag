"""
Benchmark Runner — Orchestrates all 3 pipelines + evaluation.
Run this to generate the full benchmark_results.json.

Usage: python src/evaluation/benchmark_runner.py
"""

import json
import time
import os
from datetime import datetime
from pipelines.pipeline1_llm_only import LLMOnlyPipeline
from pipelines.pipeline2_basic_rag import BasicRAGPipeline
from pipelines.pipeline3_graphrag import GraphRAGPipeline
from evaluation.llm_judge import EvaluationSuite, compute_improvements

QUESTIONS_PATH = "data/questions.json"
GROUND_TRUTH_PATH = "data/ground_truth.json"
RESULTS_PATH = "benchmark_results.json"


def load_benchmark_data():
    with open(QUESTIONS_PATH) as f:
        questions_data = json.load(f)
    with open(GROUND_TRUTH_PATH) as f:
        ground_truth = json.load(f)
    return questions_data["questions"], ground_truth["answers"]


def run_full_benchmark():
    """
    Full benchmark: 3 pipelines × N questions + evaluation.
    Generates benchmark_results.json for the submission.
    """
    print("=" * 60)
    print("MedGraph-RAG: Full Benchmark Run")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    questions, ground_truth = load_benchmark_data()
    print(f"\nLoaded {len(questions)} benchmark questions")

    evaluator = EvaluationSuite()
    all_results = {}

    # ── Pipeline 1: LLM Only ────────────────────────────────────────
    print(f"\n[1/3] Running LLM-Only Pipeline...")
    p1 = LLMOnlyPipeline()
    p1_results = p1.batch_query(questions)
    p1_eval = evaluator.evaluate_pipeline(p1_results, ground_truth)
    all_results["llm_only"] = {"queries": p1_results, "evaluation": p1_eval}
    print(f"  → Avg tokens: {p1_eval['efficiency']['avg_tokens_total']}")
    print(f"  → LLM-Judge pass rate: {p1_eval['accuracy']['llm_judge_pass_rate']*100:.1f}%")

    # ── Pipeline 2: Basic RAG ───────────────────────────────────────
    print(f"\n[2/3] Running Basic RAG Pipeline...")
    p2 = BasicRAGPipeline()
    p2_results = p2.batch_query(questions)
    p2_eval = evaluator.evaluate_pipeline(p2_results, ground_truth)
    all_results["basic_rag"] = {"queries": p2_results, "evaluation": p2_eval}
    print(f"  → Avg tokens: {p2_eval['efficiency']['avg_tokens_total']}")
    print(f"  → LLM-Judge pass rate: {p2_eval['accuracy']['llm_judge_pass_rate']*100:.1f}%")

    # ── Pipeline 3: GraphRAG ────────────────────────────────────────
    print(f"\n[3/3] Running GraphRAG Pipeline...")
    p3 = GraphRAGPipeline()
    p3_results = p3.batch_query(questions)
    p3_eval = evaluator.evaluate_pipeline(p3_results, ground_truth)
    all_results["graphrag"] = {"queries": p3_results, "evaluation": p3_eval}
    print(f"  → Avg tokens: {p3_eval['efficiency']['avg_tokens_total']}")
    print(f"  → LLM-Judge pass rate: {p3_eval['accuracy']['llm_judge_pass_rate']*100:.1f}%")

    # ── Compute improvements ────────────────────────────────────────
    improvements_vs_rag = compute_improvements(p2_eval, p3_eval)
    improvements_vs_llm = compute_improvements(p1_eval, p3_eval)

    # ── Final report ────────────────────────────────────────────────
    benchmark_report = {
        "metadata": {
            "run_date": datetime.now().isoformat(),
            "dataset": "PubMed Open Access — Cancer Research 2018-2023",
            "dataset_tokens": 2100000,
            "n_questions": len(questions),
            "model": "gemini-1.5-flash",
        },
        "results": {
            "llm_only": p1_eval,
            "basic_rag": p2_eval,
            "graphrag": p3_eval,
        },
        "improvements": {
            "graphrag_vs_basic_rag": improvements_vs_rag,
            "graphrag_vs_llm_only": improvements_vs_llm,
        },
        "bonus_thresholds": {
            "llm_judge_90pct": p3_eval["accuracy"]["bonus_llm_judge_hit"],
            "bertscore_0_55": p3_eval["accuracy"]["bonus_bertscore_hit"],
            "both_hit": p3_eval["accuracy"]["both_bonuses_hit"],
        },
    }

    # Save results
    with open(RESULTS_PATH, "w") as f:
        json.dump(benchmark_report, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Pipeline':<15} {'Avg Tokens':>12} {'Cost/Query':>12} {'Latency':>10} {'Pass Rate':>10} {'BERTScore':>10}")
    print("-" * 75)
    for name, eval_data in [("LLM-Only", p1_eval), ("Basic RAG", p2_eval), ("GraphRAG ⭐", p3_eval)]:
        e = eval_data["efficiency"]
        a = eval_data["accuracy"]
        print(f"{name:<15} {e['avg_tokens_total']:>12.0f} ${e['avg_cost_usd']:>10.5f} {e['avg_latency_s']:>9.2f}s {a['llm_judge_pass_rate']*100:>9.1f}% {a['bertscore']['f1_rescaled']:>10.3f}")

    print(f"\nGraphRAG vs Basic RAG Improvements:")
    i = improvements_vs_rag
    print(f"  Token reduction:    {i['token_reduction_pct']}%")
    print(f"  Cost reduction:     {i['cost_reduction_pct']}%")
    print(f"  Latency reduction:  {i['latency_reduction_pct']}%")
    print(f"  Accuracy gain:      +{i['accuracy_improvement_pct']}%")

    print(f"\nBonus Thresholds:")
    print(f"  LLM-Judge ≥90%:     {'✅ HIT' if benchmark_report['bonus_thresholds']['llm_judge_90pct'] else '❌ MISS'}")
    print(f"  BERTScore ≥0.55:    {'✅ HIT' if benchmark_report['bonus_thresholds']['bertscore_0_55'] else '❌ MISS'}")
    print(f"\nResults saved to: {RESULTS_PATH}")

    return benchmark_report


if __name__ == "__main__":
    run_full_benchmark()
