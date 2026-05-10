"""
Token Counter + Cost Calculator
Tracks tokens and compute costs per query across all 3 pipelines.
"""

class TokenCounter:
    def __init__(self, model: str, cost_per_1k_input: float, cost_per_1k_output: float):
        self.model = model
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.query_count = 0

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        cost = (input_tokens / 1000 * self.cost_per_1k_input) + \
               (output_tokens / 1000 * self.cost_per_1k_output)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.query_count += 1
        return round(cost, 6)

    def summary(self) -> dict:
        total_cost = self.calculate_cost(0, 0)  # won't double count
        return {
            "model": self.model,
            "total_queries": self.query_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "avg_tokens_per_query": round(
                (self.total_input_tokens + self.total_output_tokens) / max(1, self.query_count)
            ),
        }
