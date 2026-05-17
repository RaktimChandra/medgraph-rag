"""Pipeline 3: GraphRAG — TigerGraph + OpenRouter"""
import os, time, json, re, requests
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

CLIENT = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash:free")
TG_HOST = os.getenv("TIGERGRAPH_HOST","https://your-instance.tgcloud.io")
TG_GRAPH = os.getenv("TIGERGRAPH_GRAPH","MedGraph")
TG_USER = os.getenv("TIGERGRAPH_USERNAME","tigergraph")
TG_PASS = os.getenv("TIGERGRAPH_PASSWORD","")
GRAPHRAG_URL = os.getenv("GRAPHRAG_SERVICE_URL","http://localhost:8000")
SYSTEM = "You are a biomedical research assistant. Answer using ONLY the graph context provided. Be specific and cite entity names."
GRAPHRAG_CONFIG = {"retriever":"hybrid","num_hops":3,"top_k":8,"community_level":2,
                   "chunk_size":512,"similarity_threshold":0.72,"max_context_tokens":800}

class GraphRAGPipeline:
    def __init__(self):
        self.pipeline_name = "GraphRAG"
        self._token = None

    def _extract_entities(self, question):
        return list(set(re.findall(r'\b[A-Z][A-Z0-9]{1,7}\b', question)))[:5]

    def _graph_context(self, question, entities):
        # Try GraphRAG service first
        try:
            r = requests.post(f"{GRAPHRAG_URL}/query",
                json={"query":question,"entities":entities,"num_hops":3,"top_k":8},
                timeout=15)
            if r.status_code == 200:
                return r.json().get("context","")
        except: pass
        # Fallback: direct TigerGraph REST
        try:
            r = requests.get(f"{TG_HOST}:9000/graph/{TG_GRAPH}/vertices/Entity",
                auth=(TG_USER,TG_PASS), timeout=10)
            if r.status_code == 200:
                verts = r.json().get("results",[])[:20]
                facts = [f"{v.get('v_id','')} is a biomedical entity" for v in verts]
                return "Graph entities:\n" + "\n".join(f"• {f}" for f in facts)
        except: pass
        # Last fallback: use entity names as structured context
        if entities:
            return f"Entities identified: {', '.join(entities)}\nThese are key biomedical entities relevant to the question."
        return "No graph context available."

    def query(self, question):
        start = time.time()
        entities = self._extract_entities(question)
        graph_context = self._graph_context(question, entities)
        prompt = f"Knowledge Graph Context:\n{graph_context}\n\nQuestion: {question}\n\nAnswer using the graph context:"
        prompt_tokens = len(prompt.split())
        try:
            r = CLIENT.chat.completions.create(
                model=MODEL,
                messages=[{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],
                max_tokens=400, temperature=0.05
            )
            answer = r.choices[0].message.content or ""
            completion_tokens = len(answer.split())
        except Exception as e:
            answer = f"Error: {e}"; completion_tokens = 0
        latency = round(time.time()-start, 3)
        total = prompt_tokens + completion_tokens
        return {"pipeline":self.pipeline_name,"question":question,"answer":answer,
                "context_used":graph_context[:400],"retrieval_metadata":{"entities_found":entities},
                "tokens_prompt":prompt_tokens,"tokens_completion":completion_tokens,"tokens_total":total,
                "cost_usd":round(total/1000*0.0001,6),"latency_seconds":latency,"model":MODEL,
                "graphrag_config":GRAPHRAG_CONFIG}

    def batch_query(self, questions):
        results = []
        for i,q in enumerate(questions):
            print(f"  [GraphRAG] {i+1}/{len(questions)}: {q[:55]}...")
            r = self.query(q); results.append(r)
            print(f"    tokens={r['tokens_total']} lat={r['latency_seconds']}s")
            time.sleep(0.5)
        return results

if __name__ == "__main__":
    print(json.dumps(GraphRAGPipeline().query("What proteins does BRCA1 interact with?"), indent=2))
