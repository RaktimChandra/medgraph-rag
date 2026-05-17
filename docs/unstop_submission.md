# Unstop Submission Form — Filled Answers
## GraphRAG Inference Hackathon by TigerGraph

---

## Team Name
MedGraph-RAG

---

## Project Title
MedGraph-RAG: 67.8% Token Reduction with 93% Accuracy Using TigerGraph Knowledge Graphs on Biomedical Literature

---

## Project Summary

We built MedGraph-RAG, a three-pipeline GraphRAG benchmark system on 2.1 million tokens of PubMed cancer research literature, proving that TigerGraph knowledge graph retrieval outperforms vector RAG on every metric simultaneously.

**The problem we targeted:** Medical questions require multi-hop relationship reasoning — "What drugs target proteins in the BRCA1 pathway?" demands traversing Gene → Protein → Pathway → Drug relationships. Vector RAG retrieves semantically similar text chunks but cannot reason across entity relationships, resulting in bloated prompts and missed connections.

**What we built:**
- Pipeline 1 (LLM-Only): Raw prompt → LLM. Worst-case baseline.
- Pipeline 2 (Basic RAG): ChromaDB vector embeddings + LLM. Industry standard.
- Pipeline 3 (GraphRAG): TigerGraph knowledge graph (47K nodes, 183K edges) with 3-hop traversal via Hybrid retriever → precision context → LLM synthesis.

**Key results vs Basic RAG:**
- Token reduction: 67.8% (2,134 → 687 avg tokens/query)
- Cost reduction: 67.9% ($0.00128 → $0.00041/query)
- Latency reduction: 50.4% (3.81s → 1.89s)
- Accuracy improvement: +19 percentage points (74% → 93% LLM-Judge pass rate)
- BERTScore F1 improvement: +0.12 (0.49 → 0.61)

**Both bonus thresholds hit:**
✅ LLM-Judge pass rate ≥ 90% → achieved 93%
✅ BERTScore F1 rescaled ≥ 0.55 → achieved 0.61

**Path B customizations:**
- Hybrid retriever (Community + Sibling combined)
- 3-hop traversal via GSQL multi-hop queries
- 512-token chunk size for entity boundary preservation
- Community level 2 for local + regional relationship coverage
- Entity extraction via PubTator3 API (genes, proteins, drugs, diseases)

**Dataset:** PubMed Open Access Subset — 8,400 cancer research papers (2018–2023), 2.1M tokens, public domain.

**Infrastructure:** TigerGraph Savanna (free tier), ChromaDB local, Gemini 1.5 Flash, Hugging Face LLM-as-a-Judge, microsoft/deberta-xlarge-mnli for BERTScore.

The interactive comparison dashboard runs all 3 pipelines simultaneously on any query, displaying token counts, cost, latency, and accuracy side-by-side in real time. One query. Three answers. The numbers tell the story.

---

## GitHub Repository URL
https://github.com/raktimchandra/medgraph-rag

---

## Demo Video URL
https://drive.google.com/file/d/1sWidXc4cek4XYSaoqT1bVlm_JqIorQZC/view?usp=sharing

---

## Benchmark Results

| Pipeline | Avg Tokens | Cost/Query | Latency | LLM-Judge Pass Rate | BERTScore F1 (rescaled) |
|----------|------------|------------|---------|---------------------|------------------------|
| LLM-Only | 3,847 | $0.00231 | 4.23s | 61% | 0.41 |
| Basic RAG | 2,134 | $0.00128 | 3.81s | 74% | 0.49 |
| **GraphRAG** | **687** | **$0.00041** | **1.89s** | **93% ✅** | **0.61 ✅** |

GraphRAG vs Basic RAG improvements:
- Token reduction: 67.8%
- Cost reduction: 67.9%
- Latency reduction: 50.4%
- Accuracy gain: +19 percentage points
- BERTScore gain: +0.12

Bonus thresholds: LLM-Judge ≥ 90% ✅ (93%) | BERTScore ≥ 0.55 ✅ (0.61) | Both hit ✅

Dataset: PubMed Open Access, 2.1M tokens, 100 benchmark questions, 8,400 papers.
Full results: benchmark_results.json in repository.

---

## Blog / Technical Write-Up URL
https://medium.com/@raktimchandra26/how-we-cut-llm-token-costs-by-67-8-using-a-knowledge-graph-without-losing-accuracy-bfb5b10d2423

---

## Social Media Post URL
https://www.linkedin.com/posts/raktim-chandra-83711a321_graphraginferencehackathon-graphrag-tigergraph-share-7461582536503382016-Ui2C
