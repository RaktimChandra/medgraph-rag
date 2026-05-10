# How We Cut LLM Token Costs by 67.8% Using a Knowledge Graph — Without Losing Accuracy

*A deep technical walkthrough of MedGraph-RAG: our GraphRAG Inference Hackathon submission that hit both bonus accuracy thresholds while slashing inference costs.*

---

## The Problem Nobody Talks About Loudly Enough

Every team deploying LLMs at scale hits the same wall eventually.

You start with a prototype. It works beautifully. Then you push it to production, usage grows, and suddenly your monthly bill looks like a phone number. You dig into the numbers and find the culprit: **token bloat**.

The culprit isn't the LLM itself. It's what you're feeding it.

Standard RAG (Retrieval-Augmented Generation) solves the hallucination problem by injecting retrieved context before every query. But it introduces a new problem: vector search retrieves *similar text chunks*, not *precise facts*. You end up stuffing 2,000+ tokens of loosely relevant paragraphs into every prompt — most of it noise — just to make sure the one relevant sentence makes it through.

We built **MedGraph-RAG** to prove there's a better way.

---

## Why Biomedical Literature?

We didn't pick biomedical data arbitrarily. Medical questions are structurally perfect for demonstrating GraphRAG's advantage.

Consider this query:

> *"What approved drugs target proteins in the BRCA1 DNA damage response pathway?"*

To answer this correctly, you need to traverse a chain of relationships:

```
BRCA1 [GENE]
  → interacts_with → RAD51 [PROTEIN]
  → participates_in → Homologous_Recombination [PATHWAY]
  → disrupted_by → BRCA1_loss [CONDITION]
  → exploited_by → Olaparib [DRUG, FDA approved]
```

That's three hops. Vector similarity search cannot do this. It retrieves chunks that *mention* BRCA1 and *mention* drugs — but it cannot reason across the chain connecting them.

This is exactly where TigerGraph's multi-hop traversal shines.

We used the **PubMed Open Access Subset** — 8,400 cancer research papers from 2018–2023, totaling 2.1 million tokens. Dense entity relationships between genes, proteins, pathways, drugs, and diseases. The perfect stress test.

---

## The Three Pipelines

### Pipeline 1: LLM-Only (Baseline)

```python
prompt = f"Question: {question}\n\nAnswer:"
response = model.generate_content(prompt)
```

No retrieval. Pure parametric memory. This is our worst-case baseline — the naive approach that most teams start with before they realize how expensive it gets.

**Result:** 3,847 avg tokens/query · $0.00231/query · 61% pass rate

The LLM generates verbose responses because it has no concrete context to anchor to. It hedges, it elaborates, it repeats. Every extra word costs money.

---

### Pipeline 2: Basic RAG (Industry Standard)

```python
results = chroma_collection.query(
    query_texts=[question],
    n_results=5,  # top-5 similar chunks
    include=["documents"]
)
context = "\n\n".join(results["documents"][0])
prompt = f"Context:\n{context}\n\nQuestion: {question}"
```

ChromaDB + `all-MiniLM-L6-v2` embeddings. Top-5 most similar chunks, each averaging ~400 tokens. The industry standard approach.

**Result:** 2,134 avg tokens/query · $0.00128/query · 74% pass rate

Better than LLM-only, but still bloated. The top-5 chunks contain the right answer *somewhere*, but the LLM has to process ~2,000 tokens of context to find it. And vector similarity retrieves chunks that are *topically related*, not *relationally connected* — so for multi-hop questions, you still miss key facts.

---

### Pipeline 3: GraphRAG with TigerGraph (Our Solution)

This is where it gets interesting.

#### Step 1: Entity Extraction

```python
# Extract biomedical entities from the question
entities = extract_entities(question)
# Returns: ["BRCA1", "DNA repair", "drugs"]
```

We anchor the graph traversal to specific named entities. Instead of embedding the whole question and finding similar text, we identify *what things* the question is asking about.

#### Step 2: Multi-Hop Graph Traversal (The Magic)

```gsql
// Hop 1: Direct relationships from seed entities
hop1 = SELECT t FROM start:s -(Entity_Relationship:e)- Entity:t
       WHERE s.name IN ["BRCA1"]
       ACCUM @@context_facts += s.name + " " + e.relation_type + " " + t.name;

// Hop 2: Second-order relationships
hop2 = SELECT t FROM hop1:s -(Entity_Relationship:e)- Entity:t
       WHERE t NOT IN @@visited;

// Hop 3: Drug-pathway-gene connections
hop3 = SELECT t FROM hop2:s -(Entity_Relationship:e)- Entity:t
       WHERE e.confidence_score > 0.72
       LIMIT 15;
```

TigerGraph traverses 47,000 nodes and 183,000 edges to find the exact relational path answering the question. It surfaces connections that are three relationship-hops away — things vector search would never find.

#### Step 3: Precision Context Assembly

```python
# What GraphRAG hands to the LLM:
context = """
Graph relationships:
• BRCA1 → interacts_with → RAD51 [confidence: 0.97]
• RAD51 → participates_in → Homologous_Recombination
• Homologous_Recombination → disrupted_by → BRCA1_deficiency
• BRCA1_deficiency → exploited_by → Olaparib [FDA approved 2014]
• Olaparib → approved_for → BRCA1_mutant_ovarian_cancer
"""
# 623 tokens total vs 2,134 for Basic RAG
```

Instead of 2,000 tokens of paragraphs, the LLM receives a compact, structured set of facts. Every token is load-bearing. The LLM's job is synthesis, not retrieval.

**Result:** 687 avg tokens/query · $0.00041/query · **93% pass rate** · BERTScore F1: **0.61**

---

## The Tuning Process (Path B)

We didn't just use the TigerGraph GraphRAG repo out of the box. We ran a systematic grid search across four key parameters:

| Parameter | Values Tested | Winner |
|-----------|--------------|--------|
| `retriever` | Community, Sibling, Hybrid | **Hybrid** |
| `num_hops` | 1, 2, 3, 4 | **3** |
| `community_level` | 1, 2, 3 | **2** |
| `chunk_size` | 256, 512, 1024 | **512** |

**Why 3 hops?** Biomedical entity chains are typically 2–4 relationships deep. 3 hops captures gene→protein→pathway→drug chains without over-traversing into noise.

**Why Hybrid retriever?** Community retrieval finds thematically clustered nodes. Sibling retrieval finds nodes sharing the same parent relationship. Combined (Hybrid), they cover both topical clusters and relational chains — exactly what biomedical Q&A needs.

**Why 512-token chunks?** 256 tokens cut off entity context mid-sentence. 1024 tokens included surrounding noise. 512 preserved full entity descriptions without padding.

---

## Results That Speak for Themselves

| Metric | LLM-Only | Basic RAG | GraphRAG | vs Basic RAG |
|--------|----------|-----------|----------|-------------|
| Avg Tokens | 3,847 | 2,134 | **687** | **↓ 67.8%** |
| Cost/Query | $0.00231 | $0.00128 | **$0.00041** | **↓ 67.9%** |
| Latency | 4.2s | 3.8s | **1.9s** | **↓ 50.0%** |
| LLM-Judge Pass | 61% | 74% | **93%** | **↑ +19 pts** |
| BERTScore F1 | 0.41 | 0.49 | **0.61** | **↑ +0.12** |

Both hackathon bonus thresholds hit:
- ✅ LLM-Judge pass rate ≥ 90% → **93%**
- ✅ BERTScore F1 rescaled ≥ 0.55 → **0.61**

The key insight: **token reduction and accuracy improvement are not in tension when you use a graph.** They're correlated. Precise context means fewer tokens *and* better answers. The graph finds exactly what's needed. Nothing more. Nothing less.

---

## Why This Matters at Scale

At 1,000 queries/day:

| Pipeline | Daily Cost | Monthly Cost |
|----------|------------|--------------|
| LLM-Only | $2.31 | **$69.30** |
| Basic RAG | $1.28 | **$38.40** |
| GraphRAG | $0.41 | **$12.30** |

GraphRAG saves $26/month at 1,000 queries/day. At 100,000 queries/day — a modest production workload — that's **$2,600/month saved** while *improving* answer quality. The graph pays for itself immediately.

---

## What We'd Do Differently

**More entity types.** We modeled genes, proteins, drugs, diseases, and pathways. Adding clinical trials, dosage information, and contraindications would deepen the graph and improve multi-hop accuracy further.

**Streaming responses.** The dashboard currently waits for all three pipelines before displaying. Streaming each as it completes would cut perceived latency significantly.

**Graph updates.** PubMed adds 4,000 papers daily. An incremental graph update pipeline — extract entities from new papers and merge into the existing graph — would keep the knowledge graph current without full re-ingestion.

---

## Getting Started

Everything is open source:

```bash
git clone https://github.com/raktimchandra/medgraph-rag
cd medgraph-rag
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
docker-compose up -d
python scripts/ingest_dataset.py
python src/evaluation/benchmark_runner.py
```

The TigerGraph GraphRAG repo that powers Pipeline 3: [github.com/tigergraph/graphrag](https://github.com/tigergraph/graphrag)

Free TigerGraph cloud environment: [tgcloud.io](https://tgcloud.io)

---

## Final Thought

Vector RAG was the right answer for 2022. Knowledge graphs are the right answer for 2026.

The difference isn't just cost. It's the nature of what retrieval *means*. Vector search finds what looks similar. Graph traversal finds what *is connected*. For any domain where entities relate to each other — medicine, law, finance, engineering, science — the graph will win on every metric that matters.

We proved it with numbers. Now it's your turn to build on top of it.

---

*Built for the GraphRAG Inference Hackathon by TigerGraph, May 2026.*
*Raktim Chandra · [github.com/raktimchandra/medgraph-rag](https://github.com/raktimchandra/medgraph-rag)*

*Tags: #GraphRAG #TigerGraph #LLM #RAG #KnowledgeGraph #AI #MachineLearning #GraphRAGInferenceHackathon*
