"""Generate ground truth answers using OpenRouter"""
import os, json, time, re
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

CLIENT = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash:free")
QUESTIONS_PATH = "data/questions.json"
OUTPUT_PATH = "data/ground_truth.json"
PROGRESS_PATH = "data/ground_truth_progress.json"

def generate_answer(question):
    try:
        r = CLIENT.chat.completions.create(
            model=MODEL,
            messages=[
                {"role":"system","content":"You are an expert biomedical scientist. Answer with specific facts: name exact genes, proteins, drugs, pathways. Be precise, 2-4 sentences max."},
                {"role":"user","content":f"Question: {question}\n\nProvide a precise reference answer:"}
            ],
            max_tokens=256, temperature=0.1
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        print(f"    Error: {e}"); time.sleep(5); return ""

def main():
    print("="*55); print("Ground Truth Generator"); print("="*55)
    with open(QUESTIONS_PATH) as f:
        questions = json.load(f)["questions"]
    print(f"Loaded {len(questions)} questions")
    progress = json.load(open(PROGRESS_PATH)) if os.path.exists(PROGRESS_PATH) else {"answers":{}}
    answers = progress.get("answers", {})
    for i, question in enumerate(questions):
        qid = f"q{i+1:03d}"
        if qid in answers:
            print(f"  [{i+1:3d}/{len(questions)}] SKIP (cached)"); continue
        print(f"\n  [{i+1:3d}/{len(questions)}] {question[:65]}...")
        answer = generate_answer(question)
        if answer:
            answers[qid] = {"id":qid,"question":question,"answer":answer,"grounded_in_pubmed":False}
            print(f"    ✓ {answer[:100]}...")
        progress["answers"] = answers
        json.dump(progress, open(PROGRESS_PATH,"w"), indent=2)
        time.sleep(1.5)
    ground_truth = {"metadata":{"total":len(answers),"model":MODEL},
                    "answers":[answers[f"q{i+1:03d}"] for i in range(len(questions)) if f"q{i+1:03d}" in answers]}
    json.dump(ground_truth, open(OUTPUT_PATH,"w"), indent=2)
    print(f"\nDone! {len(answers)}/{len(questions)} answers → {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
