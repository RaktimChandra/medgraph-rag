"""
Pipeline 1: LLM-Only Baseline
No retrieval. Raw question → LLM → Answer.
This is our worst-case baseline showing the token cost of zero context management.
"""

import os
import time
import json
from typing import Optional
import google.generativeai as genai
from utils.token_counter import TokenCounter

# Configure LLM
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "gemini-1.5-flash"  # Free tier works
COST_PER_1K_INPUT_TOKENS = 0.000075   # Gemini Flash pricing
COST_PER_1K_OUTPUT_TOKENS = 0.0003

SYSTEM_PROMPT = """You are a biomedical research assistant with deep knowledge of 
molecular biology, pharmacology, and clinical medicine. Answer questions accurately 
and concisely based on your training knowledge."""


class LLMOnlyPipeline:
    """
    Pipeline 1: Pure LLM inference with no retrieval augmentation.
    
    This baseline demonstrates the core problem:
    - No relevant context → LLM must hallucinate or use training data
    - High token count because LLM generates verbose responses to compensate
    - Lower accuracy on domain-specific, relationship-heavy questions
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_PROMPT
        )
        self.token_counter = TokenCounter(
            model=model_name,
            cost_per_1k_input=COST_PER_1K_INPUT_TOKENS,
            cost_per_1k_output=COST_PER_1K_OUTPUT_TOKENS
        )
        self.pipeline_name = "LLM-Only"

    def query(self, question: str) -> dict:
        """
        Run the question through the LLM with no retrieval.
        
        Returns:
            dict with keys: answer, tokens_prompt, tokens_completion, 
                           tokens_total, cost, latency_seconds, pipeline
        """
        start_time = time.time()

        # Direct query — no context, no retrieval
        prompt = f"Question: {question}\n\nAnswer:"

        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=512,
            )
        )

        latency = time.time() - start_time
        answer = response.text

        # Token counting
        tokens_prompt = response.usage_metadata.prompt_token_count
        tokens_completion = response.usage_metadata.candidates_token_count
        tokens_total = tokens_prompt + tokens_completion
        cost = self.token_counter.calculate_cost(tokens_prompt, tokens_completion)

        return {
            "pipeline": self.pipeline_name,
            "question": question,
            "answer": answer,
            "context_used": None,
            "tokens_prompt": tokens_prompt,
            "tokens_completion": tokens_completion,
            "tokens_total": tokens_total,
            "cost_usd": cost,
            "latency_seconds": round(latency, 3),
            "model": MODEL_NAME
        }

    def batch_query(self, questions: list[str]) -> list[dict]:
        """Run multiple questions and return results."""
        results = []
        for i, question in enumerate(questions):
            print(f"  [LLM-Only] Query {i+1}/{len(questions)}: {question[:60]}...")
            result = self.query(question)
            results.append(result)
            print(f"    → {result['tokens_total']} tokens, ${result['cost_usd']:.5f}, {result['latency_seconds']}s")
        return results


if __name__ == "__main__":
    # Quick test
    pipeline = LLMOnlyPipeline()
    result = pipeline.query(
        "What proteins are involved in the BRCA1 DNA damage response pathway, "
        "and which approved drugs target these proteins?"
    )
    print(json.dumps(result, indent=2))
