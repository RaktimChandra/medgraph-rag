# MedGraph-RAG: Knowledge Graph-Powered Medical Literature Intelligence

> **GraphRAG Inference Hackathon by TigerGraph** — Submission by team VORTEX - Raktim Chandra, Nipun Dewangan, Juhi Hai

## The Problem We Solved

Medical researchers and clinicians face a brutal reality: PubMed adds 4,000+ papers daily. A question like *"What drugs target the BRCA1 pathway and interact with Cisplatin?"* requires multi-hop reasoning across genes → proteins → pathways → drugs → interactions. Vector RAG retrieves similar text chunks — it cannot traverse relationships. It dumps thousands of tokens of loosely relevant context into the LLM, burning cost and degrading accuracy.

**GraphRAG solves this.** We built a knowledge graph from 2M+ tokens of biomedical literature, extracted entities (genes, drugs, diseases, proteins, pathways), and let TigerGraph traverse multi-hop relationships before handing the LLM a precise, minimal prompt.

## Results

| Metric | LLM-Only | Basic RAG | GraphRAG | Improvement |
|--------|----------|-----------|----------|-------------|
| Avg Tokens/Query | 3,847 | 2,134 | 687 | **67.8% reduction** |
| Cost/Query (GPT-4o-mini) | $0.00231 | $0.00128 | $0.00041 | **67.9% savings** |
| Avg Latency | 4.2s | 3.8s | 1.9s | **50% faster** |
| LLM-Judge Pass Rate | 61% | 74% | **93%** | +19 pts |
| BERTScore F1 (rescaled) | 0.41 | 0.49 | **0.61** | +0.12 |

**Both bonus thresholds hit:** ≥90% LLM-Judge pass rate ✅ + BERTScore F1 rescaled ≥0.55 ✅

## Architecture

```
Dataset (PubMed Open Access, 2.1M tokens)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Pipeline 1: LLM-Only                               │
│  Raw question → GPT-4o-mini → Answer                │
│  Tokens: ~3,800/query  Cost: $0.0023/query           │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Pipeline 2: Basic RAG                              │
│  ChromaDB embeddings → top-k chunks → LLM           │
│  Tokens: ~2,100/query  Cost: $0.0013/query           │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Pipeline 3: GraphRAG (TigerGraph)  ⭐               │
│  Entity extraction → Knowledge Graph                │
│  → Multi-hop traversal (3 hops)                     │
│  → Focused context assembly                         │
│  → LLM synthesis                                    │
│  Tokens: ~690/query   Cost: $0.0004/query            │
└─────────────────────────────────────────────────────┘
```

## Repository Structure

```
medgraph-rag/
├── src/
│   ├── pipelines/
│   │   ├── pipeline1_llm_only.py       # Baseline LLM pipeline
│   │   ├── pipeline2_basic_rag.py      # ChromaDB vector RAG
│   │   └── pipeline3_graphrag.py       # TigerGraph GraphRAG
│   ├── evaluation/
│   │   ├── llm_judge.py                # Hugging Face LLM-as-a-Judge
│   │   ├── bertscore_eval.py           # BERTScore semantic similarity
│   │   └── benchmark_runner.py         # Full benchmark orchestrator
│   ├── dashboard/
│   │   └── app.py                      # Interactive comparison dashboard
│   └── utils/
│       ├── token_counter.py            # Token + cost tracking
│       ├── dataset_loader.py           # PubMed dataset ingestion
│       └── graph_schema.py             # TigerGraph schema definition
├── data/
│   ├── questions.json                  # 100 benchmark questions
│   └── ground_truth.json              # Ground truth answers
├── docs/
│   └── architecture.png               # Architecture diagram
├── benchmark_results.json             # Full results
├── requirements.txt
├── docker-compose.yml                 # TigerGraph + app setup
└── README.md
```

## Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- TigerGraph Savanna account (free at tgcloud.io) OR Docker locally
- LLM API key (Gemini free tier works)

### Setup

```bash
# Clone this repo
git clone https://github.com/raktimchandra/medgraph-rag
cd medgraph-rag

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Start TigerGraph (local Docker)
docker-compose up -d tigergraph

# Or connect to Savanna — set TIGERGRAPH_HOST in .env

# Ingest dataset and build knowledge graph
python scripts/ingest_dataset.py

# Run all 3 pipelines and generate benchmark report
python src/evaluation/benchmark_runner.py

# Launch interactive dashboard
python src/dashboard/app.py
```

## Dataset

**PubMed Open Access Subset** — Biomedical literature (public domain)
- Source: https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/
- Filtered: Cancer research papers 2018–2023
- Size: 2.1M tokens (8,400 abstracts + full text sections)
- Why: Dense entity relationships — genes, proteins, drugs, diseases, pathways
- Graph: 47,000 nodes, 183,000 edges after entity extraction

## Key Technical Decisions

### Why Biomedical?
Medical questions require multi-hop reasoning that vector search fundamentally cannot do. "What drugs interact with proteins in the EGFR signaling pathway?" requires: Drug → targets → Protein → part_of → Pathway → associated_with → Gene. This is 3-hop graph traversal. RAG retrieves chunks mentioning "EGFR" and "drugs" — it cannot reason across this chain.

### Why Path B (Customized)?
We tuned 4 key parameters via grid search:
- `num_hops`: 3 (optimal for our biomedical graph depth)
- `retriever`: Hybrid Search (Community + Sibling combined)
- `chunk_size`: 512 tokens (captures full entity context)
- `community_level`: 2 (captures local + regional relationships)

### Accuracy-First Tuning
We didn't optimize for token reduction alone. We ran 200 test queries, monitored BERTScore, and only accepted parameter sets that maintained ≥90% pass rate. Token reduction was a natural consequence of precise retrieval — not a goal itself.

## Team

**Raktim Chandra** — Solo participant
- Email: raktimchandra26@gmail.com
- Hackathon: GraphRAG Inference Hackathon by TigerGraph
