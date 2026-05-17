"""Pipeline 1: LLM-Only Baseline — OpenRouter"""
import os, time, json
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

CLIENT = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash:free")
SYSTEM = "You are a biomedical research assistant. Answer questions accurately and concisely."

class LLMOnlyPipeline:
    def __init__(self):
        self.pipeline_name = "LLM-Only"

    def query(self, question):
        start = time.time()
        prompt_tokens = len(question.split()) + 20
        try:
            r = CLIENT.chat.completions.create(
                model=MODEL,
                messages=[{"role":"system","content":SYSTEM},{"role":"user","content":question}],
                max_tokens=400, temperature=0.1
            )
            answer = r.choices[0].message.content or ""
            completion_tokens = len(answer.split())
        except Exception as e:
            answer = f"Error: {e}"; completion_tokens = 0
        latency = round(time.time()-start, 3)
        total = prompt_tokens + completion_tokens
        return {"pipeline":self.pipeline_name,"question":question,"answer":answer,"context_used":None,
                "tokens_prompt":prompt_tokens,"tokens_completion":completion_tokens,"tokens_total":total,
                "cost_usd":round(total/1000*0.0001,6),"latency_seconds":latency,"model":MODEL}

    def batch_query(self, questions):
        results = []
        for i,q in enumerate(questions):
            print(f"  [LLM-Only] {i+1}/{len(questions)}: {q[:55]}...")
            r = self.query(q); results.append(r)
            print(f"    tokens={r['tokens_total']} lat={r['latency_seconds']}s")
            time.sleep(0.5)
        return results

if __name__ == "__main__":
    print(json.dumps(LLMOnlyPipeline().query("What proteins does BRCA1 interact with?"), indent=2))
