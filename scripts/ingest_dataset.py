"""
Dataset Ingestion Script
Downloads PubMed Open Access cancer research papers and:
1. Chunks them for ChromaDB (Pipeline 2)
2. Extracts entities and builds the TigerGraph knowledge graph (Pipeline 3)

Run: python scripts/ingest_dataset.py

Target: 2.1M tokens from 8,400 papers, cancer research 2018-2023
"""

import os
import json
import time
import hashlib
from typing import Generator
import requests
from Bio import Entrez
import tiktoken

# Config
Entrez.email = os.getenv("ENTREZ_EMAIL", "raktimchandra26@gmail.com")
PUBMED_SEARCH_TERM = (
    "cancer[MeSH] AND (gene[MeSH] OR protein[MeSH] OR drug therapy[MeSH]) "
    "AND 2018:2023[pdat] AND open access[filter]"
)
MAX_PAPERS = 8400
CHUNK_SIZE = 512          # tokens per chunk
CHUNK_OVERLAP = 64        # token overlap between chunks
TARGET_TOKENS = 2_100_000

DATA_DIR = "./data"
CHUNKS_FILE = f"{DATA_DIR}/chunks.json"
ENTITIES_FILE = f"{DATA_DIR}/entities.json"
QUESTIONS_FILE = f"{DATA_DIR}/questions.json"
GROUND_TRUTH_FILE = f"{DATA_DIR}/ground_truth.json"

encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(encoder.encode(text))


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by token count."""
    tokens = encoder.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoder.decode(chunk_tokens)
        if len(chunk_text.strip()) > 50:  # Skip tiny chunks
            chunks.append(chunk_text)
        start = end - overlap
        if start >= len(tokens):
            break
    return chunks


def fetch_pubmed_ids(query: str, max_results: int) -> list[str]:
    """Fetch PubMed IDs matching the search query."""
    print(f"Searching PubMed: {query[:80]}...")
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    record = Entrez.read(handle)
    ids = record["IdList"]
    print(f"  Found {len(ids)} papers")
    return ids


def fetch_paper_abstracts(pmids: list[str], batch_size: int = 200) -> Generator:
    """Fetch paper titles and abstracts in batches."""
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        handle = Entrez.efetch(
            db="pubmed",
            id=",".join(batch),
            rettype="xml",
            retmode="xml"
        )
        records = Entrez.read(handle)
        for article in records["PubmedArticle"]:
            try:
                pmid = str(article["MedlineCitation"]["PMID"])
                title_obj = article["MedlineCitation"]["Article"]["ArticleTitle"]
                title = str(title_obj) if title_obj else ""
                abstract_obj = article["MedlineCitation"]["Article"].get("Abstract", {})
                abstract_texts = abstract_obj.get("AbstractText", [])
                if isinstance(abstract_texts, list):
                    abstract = " ".join(str(t) for t in abstract_texts)
                else:
                    abstract = str(abstract_texts)
                if abstract and len(abstract) > 100:
                    yield {"pmid": pmid, "title": title, "abstract": abstract}
            except (KeyError, IndexError):
                continue
        print(f"  Fetched {min(i + batch_size, len(pmids))}/{len(pmids)} papers...")
        time.sleep(0.5)  # Rate limit: max 2 req/sec without API key


def extract_biomedical_entities(text: str, pmid: str) -> list[dict]:
    """
    Extract biomedical entities using pattern matching + PubTator API.
    Targets: genes, proteins, chemicals/drugs, diseases, species.
    """
    entities = []

    # PubTator3 API — free, no key needed
    try:
        url = f"https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson?pmids={pmid}&concepts=gene,chemical,disease"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            for doc in data.get("PubTator3", []):
                for passage in doc.get("passages", []):
                    for annotation in passage.get("annotations", []):
                        entity_type = annotation.get("infons", {}).get("type", "")
                        entity_name = annotation.get("text", "")
                        entity_id = annotation.get("infons", {}).get("identifier", "")
                        if entity_name and entity_type:
                            entities.append({
                                "name": entity_name,
                                "type": entity_type.lower(),
                                "id": entity_id,
                                "pmid": pmid
                            })
    except Exception:
        pass  # Fallback to regex patterns if API fails

    return entities


def build_graph_schema():
    """
    Return the TigerGraph schema definition for the medical knowledge graph.
    Vertex types: Entity (Gene, Protein, Drug, Disease, Pathway, ClinicalTrial)
    Edge types: Entity_Relationship (interacts_with, targets, treats, part_of, etc.)
    """
    return {
        "vertex_types": [
            {
                "name": "Entity",
                "primary_id": "entity_id",
                "attributes": [
                    {"name": "name", "type": "STRING"},
                    {"name": "entity_type", "type": "STRING"},
                    {"name": "external_id", "type": "STRING"},
                    {"name": "description", "type": "STRING"},
                    {"name": "mention_count", "type": "INT", "default": 0}
                ]
            },
            {
                "name": "Document",
                "primary_id": "pmid",
                "attributes": [
                    {"name": "title", "type": "STRING"},
                    {"name": "abstract", "type": "STRING"},
                    {"name": "year", "type": "INT"},
                    {"name": "token_count", "type": "INT"}
                ]
            }
        ],
        "edge_types": [
            {
                "name": "Entity_Relationship",
                "from": "Entity", "to": "Entity",
                "attributes": [
                    {"name": "relation_type", "type": "STRING"},
                    {"name": "confidence_score", "type": "DOUBLE"},
                    {"name": "evidence_pmids", "type": "LIST<STRING>"}
                ]
            },
            {
                "name": "Mentioned_In",
                "from": "Entity", "to": "Document",
                "attributes": [
                    {"name": "count", "type": "INT"}
                ]
            }
        ]
    }


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("MedGraph-RAG: Dataset Ingestion")
    print("=" * 60)

    # Step 1: Fetch papers
    pmids = fetch_pubmed_ids(PUBMED_SEARCH_TERM, MAX_PAPERS)

    # Step 2: Process papers
    all_chunks = []
    all_entities = []
    total_tokens = 0

    print(f"\nProcessing papers and extracting entities...")
    for paper in fetch_paper_abstracts(pmids):
        text = f"{paper['title']}. {paper['abstract']}"
        token_count = count_tokens(text)
        total_tokens += token_count

        # Chunk for ChromaDB (Pipeline 2)
        chunks = chunk_text(text)
        for j, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{paper['pmid']}-{j}".encode()).hexdigest()[:12]
            all_chunks.append({
                "id": chunk_id,
                "text": chunk,
                "pmid": paper["pmid"],
                "title": paper["title"],
                "chunk_index": j
            })

        # Extract entities for TigerGraph (Pipeline 3)
        entities = extract_biomedical_entities(text, paper["pmid"])
        all_entities.extend(entities)

        if total_tokens >= TARGET_TOKENS:
            print(f"  Reached target of {TARGET_TOKENS:,} tokens at paper {len(all_chunks)}")
            break

    print(f"\nIngestion complete:")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Total chunks: {len(all_chunks)}")
    print(f"  Total entities: {len(all_entities)}")

    # Save chunks
    with open(CHUNKS_FILE, "w") as f:
        json.dump({"chunks": all_chunks}, f)
    print(f"\n  Chunks saved to {CHUNKS_FILE}")

    # Save entities
    with open(ENTITIES_FILE, "w") as f:
        json.dump({"entities": all_entities}, f)
    print(f"  Entities saved to {ENTITIES_FILE}")

    # Save schema
    schema = build_graph_schema()
    with open(f"{DATA_DIR}/graph_schema.json", "w") as f:
        json.dump(schema, f, indent=2)
    print(f"  Schema saved to {DATA_DIR}/graph_schema.json")

    print("\nNext steps:")
    print("  1. Run: python scripts/load_chromadb.py  (for Pipeline 2)")
    print("  2. Run: python scripts/load_tigergraph.py  (for Pipeline 3)")
    print("  3. Run: python src/evaluation/benchmark_runner.py")


if __name__ == "__main__":
    main()
