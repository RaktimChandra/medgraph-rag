"""
Pipeline 2: Basic RAG (Vector Embeddings + LLM)
ChromaDB + sentence-transformers + LLM.
The industry standard today — our primary comparison target.
"""

import os
import time
import json
from typing import Optional
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from utils.token_counter import TokenCounter

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "gemini-1.5-flash"
COST_PER_1K_INPUT_TOKENS = 0.000075
COST_PER_1K_OUTPUT_TOKENS = 0.0003
TOP_K_CHUNKS = 5        # Retrieve top-5 most similar chunks
CHROMA_COLLECTION = "pubmed_chunks"
CHROMA_PATH = "./data/chroma_db"

SYSTEM_PROMPT = """You are a biomedical research assistant. Answer questions using 
ONLY the provided context documents. If the context doesn't contain the answer, 
say so explicitly. Be precise and cite specific details from the context."""

RAG_PROMPT_TEMPLATE = """Context Documents:
{context}

---

Question: {question}

Answer based on the context above:"""


class BasicRAGPipeline:
    """
    Pipeline 2: Standard vector-based RAG.
    
    Demonstrates the vector RAG limitation:
    - Retrieves semantically SIMILAR chunks, not relationally connected ones
    - Top-k chunks dump large context windows into the LLM
    - Cannot traverse entity relationships (gene → protein → drug → interaction)
    - Moderate token reduction vs LLM-only, but large context still expensive
    """

    def __init__(self, model_name: str = MODEL_NAME, top_k: int = TOP_K_CHUNKS):
        # Vector store
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"  # Fast, good quality, free
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        self.top_k = top_k

        # LLM
        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_PROMPT
        )
        self.token_counter = TokenCounter(
            model=model_name,
            cost_per_1k_input=COST_PER_1K_INPUT_TOKENS,
            cost_per_1k_output=COST_PER_1K_OUTPUT_TOKENS
        )
        self.pipeline_name = "Basic RAG"

    def retrieve(self, question: str) -> tuple[str, list[dict]]:
        """
        Retrieve top-k most similar chunks from ChromaDB.
        Returns concatenated context string and metadata.
        """
        results = self.collection.query(
            query_texts=[question],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"]
        )

        chunks = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        # Assemble context
        context_parts = []
        retrieved_chunks = []
        for i, (chunk, meta, dist) in enumerate(zip(chunks, metadatas, distances)):
            context_parts.append(f"[Document {i+1}] {meta.get('title', 'Unknown')}\n{chunk}")
            retrieved_chunks.append({
                "rank": i + 1,
                "text": chunk[:200] + "...",
                "source": meta.get("pmid", "unknown"),
                "similarity": round(1 - dist, 3)
            })

        context = "\n\n".join(context_parts)
        return context, retrieved_chunks

    def query(self, question: str) -> dict:
        """
        Retrieve relevant chunks and generate answer.
        """
        start_time = time.time()

        # Step 1: Retrieve
        context, retrieved_chunks = self.retrieve(question)

        # Step 2: Build prompt with context
        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )

        # Step 3: Generate
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=512,
            )
        )

        latency = time.time() - start_time
        answer = response.text

        tokens_prompt = response.usage_metadata.prompt_token_count
        tokens_completion = response.usage_metadata.candidates_token_count
        tokens_total = tokens_prompt + tokens_completion
        cost = self.token_counter.calculate_cost(tokens_prompt, tokens_completion)

        return {
            "pipeline": self.pipeline_name,
            "question": question,
            "answer": answer,
            "context_used": context[:500] + "...",
            "retrieved_chunks": retrieved_chunks,
            "tokens_prompt": tokens_prompt,
            "tokens_completion": tokens_completion,
            "tokens_total": tokens_total,
            "cost_usd": cost,
            "latency_seconds": round(latency, 3),
            "model": MODEL_NAME
        }

    def batch_query(self, questions: list[str]) -> list[dict]:
        results = []
        for i, question in enumerate(questions):
            print(f"  [Basic RAG] Query {i+1}/{len(questions)}: {question[:60]}...")
            result = self.query(question)
            results.append(result)
            print(f"    → {result['tokens_total']} tokens, ${result['cost_usd']:.5f}, {result['latency_seconds']}s")
        return results

    def ingest_documents(self, documents: list[dict], batch_size: int = 100):
        """
        Ingest documents into ChromaDB.
        Each document: {id, text, title, pmid, ...}
        """
        print(f"Ingesting {len(documents)} chunks into ChromaDB...")
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self.collection.add(
                ids=[d["id"] for d in batch],
                documents=[d["text"] for d in batch],
                metadatas=[{k: v for k, v in d.items() if k not in ["id", "text"]}
                           for d in batch]
            )
            print(f"  Ingested {min(i + batch_size, len(documents))}/{len(documents)} chunks")
        print("ChromaDB ingestion complete.")


if __name__ == "__main__":
    pipeline = BasicRAGPipeline()
    result = pipeline.query(
        "What proteins are involved in the BRCA1 DNA damage response pathway, "
        "and which approved drugs target these proteins?"
    )
    print(json.dumps(result, indent=2))
