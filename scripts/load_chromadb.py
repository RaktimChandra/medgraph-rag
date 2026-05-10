"""
ChromaDB Loader — loads chunks.json into ChromaDB for Pipeline 2.
Run AFTER ingest_dataset.py completes.

Run: python scripts/load_chromadb.py
"""

import os
import json
import sys
from pathlib import Path
from chromadb.utils import embedding_functions
import chromadb
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

CHUNKS_PATH  = "data/chunks.json"
CHROMA_PATH  = "data/chroma_db"
COLLECTION   = "pubmed_chunks"
BATCH_SIZE   = 100


def main():
    print("=" * 55)
    print("ChromaDB Loader — Pipeline 2 vector store")
    print("=" * 55)

    if not os.path.exists(CHUNKS_PATH):
        print(f"ERROR: {CHUNKS_PATH} not found.")
        print("Run scripts/ingest_dataset.py first.")
        sys.exit(1)

    with open(CHUNKS_PATH) as f:
        data = json.load(f)
    chunks = data["chunks"]
    print(f"Loaded {len(chunks):,} chunks from {CHUNKS_PATH}")

    # Init ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Drop + recreate collection for clean load
    try:
        client.delete_collection(COLLECTION)
        print(f"Dropped existing collection '{COLLECTION}'")
    except Exception:
        pass

    col = client.create_collection(
        name=COLLECTION,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"Created collection '{COLLECTION}'")

    # Batch upsert
    ok = 0
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Ingesting"):
        batch = chunks[i:i + BATCH_SIZE]
        try:
            col.add(
                ids       =[c["id"]   for c in batch],
                documents =[c["text"] for c in batch],
                metadatas =[{k: v for k, v in c.items() if k not in ("id","text")}
                            for c in batch],
            )
            ok += len(batch)
        except Exception as e:
            print(f"\n  Batch {i//BATCH_SIZE} error: {e}")

    print(f"\n✓ Ingested {ok:,} / {len(chunks):,} chunks")
    print(f"✓ ChromaDB stored at: {CHROMA_PATH}")

    # Quick verify
    count = col.count()
    print(f"✓ Collection count: {count:,}")
    if count > 0:
        sample = col.query(
            query_texts=["BRCA1 DNA repair protein interaction"],
            n_results=2,
            include=["documents", "distances"]
        )
        print(f"\nSample query (BRCA1 DNA repair):")
        for doc, dist in zip(sample["documents"][0], sample["distances"][0]):
            print(f"  [{round(1-dist,3)}] {doc[:100]}...")
    print("\nNext: python scripts/load_tigergraph.py")


if __name__ == "__main__":
    main()
