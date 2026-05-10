"""
TigerGraph Schema + Loader — Upgrade 2
Creates the full knowledge graph schema on TigerGraph Savanna,
then loads all entities and edges extracted from PubMed.

Steps performed:
  1. Connect to TigerGraph (Savanna or local)
  2. Create vertex types: Entity, Document
  3. Create edge types: Entity_Relationship, Mentioned_In
  4. Load entities from data/entities.json
  5. Build Entity_Relationship edges via:
     a) PubTator3 co-occurrence relations
     b) Within-document co-mention (fallback)
  6. Verify graph stats

Run: python scripts/load_tigergraph.py

Prerequisites:
  - entities.json must exist (run ingest_dataset.py first)
  - .env must have TIGERGRAPH_* credentials set
  - TigerGraph Savanna instance must be running
"""

import os
import json
import time
import itertools
from collections import defaultdict
import pyTigerGraph as tg
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Connection config ─────────────────────────────────────────────────────────
TG_HOST     = os.getenv("TIGERGRAPH_HOST",     "https://your-instance.tgcloud.io")
TG_GRAPH    = os.getenv("TIGERGRAPH_GRAPH",    "MedGraph")
TG_USERNAME = os.getenv("TIGERGRAPH_USERNAME", "tigergraph")
TG_PASSWORD = os.getenv("TIGERGRAPH_PASSWORD", "")
TG_SECRET   = os.getenv("TIGERGRAPH_SECRET",   "")

ENTITIES_PATH = "data/entities.json"
CHUNKS_PATH   = "data/chunks.json"

BATCH_SIZE    = 1000   # vertices per upsert batch
EDGE_BATCH    = 2000   # edges per upsert batch

# ── Relation types we care about ──────────────────────────────────────────────
VALID_RELATION_TYPES = {
    "interacts_with", "targets", "treats", "inhibits", "activates",
    "part_of", "associated_with", "regulates", "phosphorylates",
    "expressed_in", "mutation_in", "drug_resistance", "biomarker_for",
    "co_occurs_with",   # fallback for co-mention
}


# ── Connect ───────────────────────────────────────────────────────────────────

def connect() -> tg.TigerGraphConnection:
    print(f"Connecting to TigerGraph at {TG_HOST}...")
    try:
        conn = tg.TigerGraphConnection(
            host=TG_HOST,
            graphname=TG_GRAPH,
            username=TG_USERNAME,
            password=TG_PASSWORD,
        )
        if TG_SECRET:
            token = conn.getToken(TG_SECRET)[0]
            conn.apiToken = token
        print(f"  ✓ Connected. Token: {str(conn.apiToken)[:20]}...")
        return conn
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        print("\n  Troubleshooting:")
        print("  1. Check TIGERGRAPH_HOST in .env — should be full URL like https://xxx.tgcloud.io")
        print("  2. Check TIGERGRAPH_PASSWORD in .env")
        print("  3. In Savanna: Settings → Credentials → Copy REST++ password")
        raise


# ── Schema creation ───────────────────────────────────────────────────────────

SCHEMA_GSQL = """
USE GLOBAL

# Drop graph if exists (fresh start)
DROP GRAPH {graph} IF EXISTS

# Create graph
CREATE GRAPH {graph}()

USE GRAPH {graph}

# ── Vertex types ─────────────────────────────────────────────
CREATE VERTEX Entity (
  PRIMARY_ID entity_id   STRING,
  name                   STRING DEFAULT "",
  entity_type            STRING DEFAULT "",
  external_id            STRING DEFAULT "",
  description            STRING DEFAULT "",
  mention_count          INT    DEFAULT 0
) WITH STATS="OUTDEGREE_BY_EDGETYPE", PRIMARY_ID_AS_ATTRIBUTE="true"

CREATE VERTEX Document (
  PRIMARY_ID pmid        STRING,
  title                  STRING DEFAULT "",
  token_count            INT    DEFAULT 0
) WITH PRIMARY_ID_AS_ATTRIBUTE="true"

# ── Edge types ───────────────────────────────────────────────
CREATE DIRECTED EDGE Entity_Relationship (
  FROM Entity, TO Entity,
  relation_type      STRING DEFAULT "co_occurs_with",
  confidence_score   DOUBLE DEFAULT 0.5,
  evidence_count     INT    DEFAULT 1
) WITH STATS="OUTDEGREE_BY_EDGETYPE"

CREATE DIRECTED EDGE Mentioned_In (
  FROM Entity, TO Document,
  mention_count      INT DEFAULT 1
)

# ── Install queries ───────────────────────────────────────────
CREATE QUERY multi_hop_query(
  SET<STRING> seed_entities,
  INT num_hops = 3,
  INT top_k    = 8,
  DOUBLE min_confidence = 0.5
) FOR GRAPH {graph} {{

  SetAccum<VERTEX<Entity>> @@visited;
  SetAccum<STRING>          @@context_triples;
  MapAccum<STRING, INT>     @@node_scores;

  start = {{Entity.*}};

  # Seed: match input entities
  seeds = SELECT s FROM start:s
          WHERE s.name IN seed_entities OR s.entity_id IN seed_entities
          ACCUM @@visited += s,
                @@node_scores += (s.entity_id -> s.mention_count);

  FOREACH hop IN RANGE[1, num_hops] DO
    seeds = SELECT t FROM seeds:s -(Entity_Relationship:e)- Entity:t
            WHERE t NOT IN @@visited
              AND e.confidence_score >= min_confidence
            ACCUM @@context_triples += s.name + "|" + e.relation_type + "|" + t.name
                                       + "|" + to_string(e.confidence_score),
                  @@visited += t,
                  @@node_scores += (t.entity_id -> t.mention_count)
            ORDER BY t.mention_count DESC
            LIMIT top_k * 2;
  END;

  PRINT @@context_triples AS triples;
  PRINT @@node_scores    AS scores;
}}

INSTALL QUERY multi_hop_query
"""


def create_schema(conn: tg.TigerGraphConnection):
    print("\nCreating graph schema...")
    gsql = SCHEMA_GSQL.format(graph=TG_GRAPH)
    try:
        result = conn.gsql(gsql)
        print(f"  Schema result: {str(result)[:200]}")
        print("  ✓ Schema created successfully")
    except Exception as e:
        # Schema may already exist — check
        if "already exists" in str(e).lower():
            print("  ✓ Schema already exists — skipping creation")
        else:
            print(f"  ✗ Schema error: {e}")
            raise


# ── Entity loading ────────────────────────────────────────────────────────────

def normalize_entity_id(name: str, etype: str) -> str:
    """Create a stable vertex ID from entity name + type."""
    clean = name.lower().strip().replace(" ", "_").replace("/", "_")
    return f"{etype[:4]}_{clean}"[:64]


def load_entities(conn: tg.TigerGraphConnection):
    print("\nLoading entities into TigerGraph...")

    with open(ENTITIES_PATH) as f:
        data = json.load(f)
    entities_raw = data["entities"]
    print(f"  Raw entity records: {len(entities_raw)}")

    # Deduplicate by (name, type)
    seen      = {}
    doc_edges = []   # (entity_id, pmid) pairs

    for e in entities_raw:
        name  = e.get("name", "").strip()
        etype = e.get("type", "unknown").lower()
        pmid  = e.get("pmid", "")
        if not name or len(name) < 2:
            continue
        eid = normalize_entity_id(name, etype)
        if eid not in seen:
            seen[eid] = {
                "entity_id":    eid,
                "name":         name,
                "entity_type":  etype,
                "external_id":  e.get("id", ""),
                "description":  "",
                "mention_count": 1,
            }
        else:
            seen[eid]["mention_count"] += 1
        if pmid:
            doc_edges.append((eid, pmid))

    vertices = list(seen.values())
    print(f"  Unique entities: {len(vertices)}")

    # Batch upsert vertices
    ok = 0
    for i in range(0, len(vertices), BATCH_SIZE):
        batch = vertices[i:i + BATCH_SIZE]
        upsert_data = {v["entity_id"]: {
            "name":          v["name"],
            "entity_type":   v["entity_type"],
            "external_id":   v["external_id"],
            "description":   v["description"],
            "mention_count": v["mention_count"],
        } for v in batch}
        try:
            conn.upsertVertices("Entity", upsert_data)
            ok += len(batch)
        except Exception as e:
            print(f"    Batch {i//BATCH_SIZE} error: {e}")
        print(f"  Upserted {min(ok, len(vertices))}/{len(vertices)} entities...")

    print(f"  ✓ Entities loaded: {ok}")
    return seen, doc_edges


# ── Edge building ─────────────────────────────────────────────────────────────

def build_edges_from_pubtator(entities_raw: list, entity_lookup: dict) -> list:
    """
    Build Entity_Relationship edges from PubTator3 relations API.
    For each PMID, query PubTator3 for annotated relations between entities.
    """
    print("\nFetching relations from PubTator3...")
    edges     = []
    pmid_entities = defaultdict(list)

    for e in entities_raw:
        pmid = e.get("pmid", "")
        name = e.get("name", "").strip()
        eid  = normalize_entity_id(name, e.get("type", "unknown"))
        if pmid and eid in entity_lookup:
            pmid_entities[pmid].append(eid)

    # Sample up to 500 PMIDs for relation extraction
    sample_pmids = list(pmid_entities.keys())[:500]
    fetched = 0

    for pmid in sample_pmids:
        try:
            url = (
                f"https://www.ncbi.nlm.nih.gov/research/pubtator3-api/"
                f"publications/export/biocjson?pmids={pmid}&concepts=relation"
            )
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                continue

            data = resp.json()
            for doc in data.get("PubTator3", []):
                for rel in doc.get("relations", []):
                    infons = rel.get("infons", {})
                    rtype  = infons.get("type", "co_occurs_with").lower().replace(" ", "_")
                    nodes  = rel.get("nodes", [])
                    if len(nodes) < 2:
                        continue
                    src_id  = nodes[0].get("refid", "")
                    tgt_id  = nodes[1].get("refid", "")
                    # Map to our internal IDs
                    src_eid = next((e for e in pmid_entities[pmid] if src_id in e), None)
                    tgt_eid = next((e for e in pmid_entities[pmid] if tgt_id in e), None)
                    if src_eid and tgt_eid and src_eid != tgt_eid:
                        edges.append({
                            "src": src_eid, "tgt": tgt_eid,
                            "relation_type":    rtype if rtype in VALID_RELATION_TYPES else "co_occurs_with",
                            "confidence_score": 0.85,
                            "evidence_count":   1,
                        })
            fetched += 1
        except Exception:
            pass
        time.sleep(0.35)   # NCBI rate limit

    print(f"  PubTator3 relations from {fetched} papers: {len(edges)} edges")
    return edges


def build_edges_from_comention(pmid_entities: dict, min_cooccur: int = 2) -> list:
    """
    Fallback: build co_occurs_with edges from entities that appear together
    in ≥ min_cooccur papers. Confidence scales with co-occurrence frequency.
    """
    print("  Building co-mention edges...")
    cooccur = defaultdict(int)

    for pmid, eids in pmid_entities.items():
        unique_eids = list(set(eids))
        for a, b in itertools.combinations(unique_eids[:20], 2):   # cap at 20/paper
            key = tuple(sorted([a, b]))
            cooccur[key] += 1

    edges = []
    for (a, b), count in cooccur.items():
        if count >= min_cooccur:
            conf = min(0.95, 0.5 + count * 0.05)   # confidence grows with count
            edges.append({
                "src": a, "tgt": b,
                "relation_type":    "co_occurs_with",
                "confidence_score": round(conf, 3),
                "evidence_count":   count,
            })

    print(f"  Co-mention edges (≥{min_cooccur} papers): {len(edges)}")
    return edges


def upsert_edges(conn: tg.TigerGraphConnection, edges: list):
    """Batch upsert all Entity_Relationship edges."""
    print(f"\nUpserting {len(edges)} edges...")
    ok = 0

    # Deduplicate: keep highest confidence per (src, tgt) pair
    best = {}
    for e in edges:
        key = (e["src"], e["tgt"])
        if key not in best or e["confidence_score"] > best[key]["confidence_score"]:
            best[key] = e
    deduped = list(best.values())
    print(f"  After dedup: {len(deduped)} unique edges")

    for i in range(0, len(deduped), EDGE_BATCH):
        batch = deduped[i:i + EDGE_BATCH]
        edge_data = [
            (e["src"], e["tgt"], {
                "relation_type":    e["relation_type"],
                "confidence_score": e["confidence_score"],
                "evidence_count":   e["evidence_count"],
            })
            for e in batch
        ]
        try:
            conn.upsertEdges("Entity", "Entity_Relationship", "Entity", edge_data)
            ok += len(batch)
        except Exception as ex:
            print(f"  Edge batch error: {ex}")
        print(f"  Upserted {min(ok, len(deduped))}/{len(deduped)} edges...")

    print(f"  ✓ Edges loaded: {ok}")


# ── Verify ────────────────────────────────────────────────────────────────────

def verify_graph(conn: tg.TigerGraphConnection):
    print("\nVerifying graph...")
    try:
        n_entities  = conn.getVertexCount("Entity")
        n_documents = conn.getVertexCount("Document")
        n_edges     = conn.getEdgeCount("Entity_Relationship")
        print(f"  Entities:      {n_entities:,}")
        print(f"  Documents:     {n_documents:,}")
        print(f"  Relationships: {n_edges:,}")

        # Test multi-hop query
        print("\n  Testing multi_hop_query on BRCA1...")
        result = conn.runInstalledQuery("multi_hop_query", {
            "seed_entities": ["BRCA1"],
            "num_hops": 2,
            "top_k": 5,
            "min_confidence": 0.5,
        })
        triples = result[0].get("triples", [])
        print(f"  Query returned {len(triples)} triples from BRCA1 (2-hop)")
        for t in triples[:5]:
            print(f"    {t}")

        if n_entities > 1000 and n_edges > 5000:
            print("\n  ✓ Graph looks healthy — ready for Pipeline 3!")
        else:
            print("\n  ⚠ Graph seems small — check ingest_dataset.py ran fully")

    except Exception as e:
        print(f"  Verification error: {e}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("TigerGraph Schema + Loader")
    print("=" * 60)

    conn = connect()

    # 1. Schema
    create_schema(conn)

    # 2. Load entities
    with open(ENTITIES_PATH) as f:
        raw = json.load(f)
    entity_lookup, doc_edges = load_entities(conn, )   # returns (seen_dict, doc_edges)

    # 3. Build pmid → entity_ids map for co-mention
    pmid_emap = defaultdict(list)
    for e in raw["entities"]:
        pmid = e.get("pmid", "")
        name = e.get("name", "").strip()
        eid  = normalize_entity_id(name, e.get("type", "unknown"))
        if pmid and eid in entity_lookup:
            pmid_emap[pmid].append(eid)

    # 4. Build edges: PubTator3 relations + co-mention fallback
    pt_edges  = build_edges_from_pubtator(raw["entities"], entity_lookup)
    co_edges  = build_edges_from_comention(pmid_emap, min_cooccur=2)
    all_edges = pt_edges + co_edges
    print(f"\n  Total edges before dedup: {len(all_edges)}")

    # 5. Upsert edges
    upsert_edges(conn, all_edges)

    # 6. Verify
    verify_graph(conn)

    print("\n" + "=" * 60)
    print("Load complete! TigerGraph knowledge graph is ready.")
    print("Next: python src/evaluation/benchmark_runner.py")


if __name__ == "__main__":
    main()
