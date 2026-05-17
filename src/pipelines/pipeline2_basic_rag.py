"""Pipeline 2: Basic RAG — ChromaDB + OpenRouter"""
import os, time, json
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
load_dotenv()

CLIENT = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash:free")
CHROMA_PATH = "data/chroma_db"
COLLECTION = "pubmed_chunks"
TOP_K = 5
SYSTEM = "You are a biomedical research assistant. Answer using ONLY the provided context. Be specific."

class BasicRAGPipeline:
    def __init__(self):
        self.pipeline_name = "Basic RAG"
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        emb = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.col = self.client.get_or_create_collection(COLLECTION, embedding_function=emb)

    def retrieve(self, question):
        res = self.col.query(query_texts=[question], n_results=TOP_K, include=["documents","distances"])
        chunks = res["documents"][0]
        dists = res["distances"][0]
        context = "\n\n".join(f"[Doc {i+1}] {c}" for i,c in enumerate(chunks))
        retrieved = [{"rank":i+1,"text":c[:150]+"...","similarity":round(1-d,3)} for i,(c,d) in enumerate(zip(chunks,dists))]
        return context, retrieved

    def query(self, question):
        start = time.time()
        context, retrieved = self.retrieve(question)
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        prompt_tokens = len(prompt.split())
        try:
            r = CLIENT.chat.completions.create(
                model=MODEL,
                messages=[{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],
                max_tokens=400, temperature=0.1
            )
            answer = r.choices[0].message.content or ""
            completion_tokens = len(answer.split())
        except Exception as e:
            answer = f"Error: {e}"; completion_tokens = 0
        latency = round(time.time()-start, 3)
        total = prompt_tokens + completion_tokens
        return {"pipeline":self.pipeline_name,"question":question,"answer":answer,
                "context_used":context[:400]+"...","retrieved_chunks":retrieved,
                "tokens_prompt":prompt_tokens,"tokens_completion":completion_tokens,"tokens_total":total,
                "cost_usd":round(total/1000*0.0001,6),"latency_seconds":latency,"model":MODEL}

    def batch_query(self, questions):
        results = []
        for i,q in enumerate(questions):
            print(f"  [Basic RAG] {i+1}/{len(questions)}: {q[:55]}...")
            r = self.query(q); results.append(r)
            print(f"    tokens={r['tokens_total']} lat={r['latency_seconds']}s")
            time.sleep(0.5)
        return results

    def ingest_documents(self, documents, batch_size=100):
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            self.col.add(ids=[d["id"] for d in batch], documents=[d["text"] for d in batch],
                         metadatas=[{k:v for k,v in d.items() if k not in ["id","text"]} for d in batch])
            print(f"  Ingested {min(i+batch_size,len(documents))}/{len(documents)}")

if __name__ == "__main__":
    print(json.dumps(BasicRAGPipeline().query("What proteins does BRCA1 interact with?"), indent=2))
