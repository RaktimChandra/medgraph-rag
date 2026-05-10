# Social Media Posts — MedGraph-RAG

## LinkedIn Post (Primary)

Just shipped my submission for the #GraphRAGInferenceHackathon by @TigerGraph 🚀

**The result: 67.8% token reduction with 93% answer accuracy** — both bonus thresholds hit simultaneously.

Here's what I built and why it works:

**The problem:** Standard vector RAG answers "what text looks similar to my question?" A medical question like "What drugs target the BRCA1 pathway?" needs to answer "what is 3 relationship-hops away from BRCA1?" Vector search can't traverse that chain. It just dumps 5 text chunks and hopes the answer is in there.

**The solution:** MedGraph-RAG — three pipelines, one dataset (2.1M tokens of PubMed cancer research), side-by-side benchmarks:

→ Pipeline 1 (LLM-Only): 3,847 tokens/query · 61% accuracy
→ Pipeline 2 (Basic RAG): 2,134 tokens/query · 74% accuracy  
→ Pipeline 3 (GraphRAG): **687 tokens/query · 93% accuracy**

TigerGraph traverses 47,000 nodes and 183,000 edges in under 2 seconds and hands the LLM a 423-token precision context instead of a 2,000-token context dump. Fewer tokens. Better answers. No tradeoff.

**The key insight:** Token reduction and accuracy improvement aren't in tension when you use a graph. Precise retrieval means BOTH fewer tokens AND better answers — because every token in the context is load-bearing.

At 100K queries/day, this saves ~$2,600/month. Every month.

Everything is open source: github.com/raktimchandra/medgraph-rag
Built on: github.com/tigergraph/graphrag

#GraphRAG #TigerGraph #LLM #KnowledgeGraph #AI #MachineLearning #RAG #TokenEfficiency

---

## Twitter/X Post

Built MedGraph-RAG for the @TigerGraph #GraphRAGInferenceHackathon

Results on 2.1M tokens of biomedical literature:

LLM-Only: 3,847 tokens, 61% accuracy
Basic RAG: 2,134 tokens, 74% accuracy  
GraphRAG: **687 tokens, 93% accuracy** ✅

67.8% token reduction. 19% accuracy gain. Both bonus thresholds hit.

The graph doesn't just cut costs — it makes the LLM better.

github.com/raktimchandra/medgraph-rag

#GraphRAGInferenceHackathon #TigerGraph #KnowledgeGraph

---

## Shorter LinkedIn Version

🔥 Results from my @TigerGraph GraphRAG Inference Hackathon submission:

**67.8% fewer tokens. 93% answer accuracy. Both bonus thresholds hit.**

Built MedGraph-RAG: a 3-pipeline benchmark on 2.1M tokens of PubMed cancer research.

The core finding: GraphRAG with TigerGraph doesn't just cut token costs — it improves answer quality simultaneously. Precise graph context > semantic similarity chunks, every time.

➡️ github.com/raktimchandra/medgraph-rag

#GraphRAGInferenceHackathon #TigerGraph #GraphRAG
