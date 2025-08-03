import tiktoken

# ===================
# Token Counting Logic
# ===================

_MODEL_TOKEN_RULES = {
    # model: (tokens_per_message, tokens_per_name, tokens_per_reply)
    "gpt-3.5-turbo":    (4, -1,  2),
    "gpt-3.5-turbo-0301": (4, -1,  2),
    "gpt-4":            (3,  1,  3),
    "gpt-4-0314":       (3,  1,  3),
    "gpt-4o":           (3,  1,  3),  # Assume gpt-4o follows gpt-4 rules
}

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in a raw string using tiktoken."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def count_message_tokens(messages: list[dict], model: str = "gpt-4o") -> int:
    """
    Count tokens in a chat-style message list following OpenAI’s
    “num_tokens_from_messages” logic.
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    tpm, tpn, tpr = _MODEL_TOKEN_RULES.get(model, _MODEL_TOKEN_RULES["gpt-3.5-turbo"])
    total = 0
    for msg in messages:
        total += tpm
        for key, val in msg.items():
            total += len(enc.encode(str(val)))
            if key == "name":
                total += tpn
    total += tpr
    return total

# ===========================
# Dynamic Token Usage Tracker
# ===========================

class TokenUsageTracker:
    # Default OpenAI pricing as of 2024–2025 (USD)
    MODEL_PRICING = {
        "gpt-4o": {"input_per_m": 2.50, "output_per_m": 10.00},
        "o4-mini": {"input_per_m": 1.10, "output_per_m": 4.40},
        "o4-mini-flex": {"input_per_m": 0.55, "output_per_m": 2.20},
        "gpt-4.1":         {"input_per_m": 5.00, "output_per_m": 15.00},
        # Add others if needed
    }
    def __init__(self, model="gpt-4o"):
        self.model = model
        self.calls = []  # Each call: {"call", "prompt_tokens", "completion_tokens", "total_tokens", "cost_usd"}

    def calc_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        p = self.MODEL_PRICING.get(self.model, self.MODEL_PRICING["gpt-4o"])
        input_cost = (prompt_tokens / 1_000_000) * p["input_per_m"]
        output_cost = (completion_tokens / 1_000_000) * p["output_per_m"]
        return round(input_cost + output_cost, 8)

    def add_call(self, call_name: str, prompt_tokens: int, completion_tokens: int):
        """Record one model call's token usage and cost."""
        cost = self.calc_cost(prompt_tokens, completion_tokens)
        self.calls.append({
            "call": call_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": cost
        })

    def get_totals(self):
        prompt = sum(c["prompt_tokens"] for c in self.calls)
        completion = sum(c["completion_tokens"] for c in self.calls)
        total = prompt + completion
        cost = sum(c["cost_usd"] for c in self.calls)
        return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total, "cost_usd": round(cost, 6)}

    def as_dataframe_dict(self, prefix=""):
        """
        Output all calls + row totals in a flat dict for DataFrame integration.
        """
        output = {}
        for c in self.calls:
            k = f"{prefix}{c['call']}_"
            output[k+"prompt_tokens"] = c["prompt_tokens"]
            output[k+"completion_tokens"] = c["completion_tokens"]
            output[k+"total_tokens"] = c["total_tokens"]
            output[k+"cost_usd"] = c["cost_usd"]
        totals = self.get_totals()
        for k, v in totals.items():
            output[f"{prefix}total_{k}"] = v
        return output

    def print_report(self):
        print("="*38)
        for i, c in enumerate(self.calls, 1):
            print(
                f"Call {i}: {c['call']:<30} "
                f"Prompt: {c['prompt_tokens']:>4} | Completion: {c['completion_tokens']:>4} | "
                f"Total: {c['total_tokens']:>4} | Cost: ${c['cost_usd']:.6f}"
            )
        t = self.get_totals()
        print("-"*38)
        print(
            f"Row totals: Prompt: {t['prompt_tokens']} | "
            f"Completion: {t['completion_tokens']} | "
            f"Total: {t['total_tokens']} | Cost: ${t['cost_usd']:.6f}"
        )
        print("="*38)

# =================
# Quick Demo
# =================

if __name__ == "__main__":
    # Demo: count tokens
    system = {"role": "system", "content": "Only call relevant tools for the scan group content."}
    user   = {"role": "user",   "content": "Scan groups: ['manifested', 'collection failed - not ready']"}
    print("→ Tokens (raw string):", count_tokens(user["content"], model="gpt-4o"))
    print("→ Tokens (full message):", count_message_tokens([system, user], model="gpt-4o"))

    # Demo: track LLM calls and total cost
    tracker = TokenUsageTracker(model="gpt-4o")
    tracker.add_call("progress", prompt_tokens=54, completion_tokens=23)
    tracker.add_call("router", prompt_tokens=33, completion_tokens=17)
    tracker.add_call("get_delivery_failure_outcome", prompt_tokens=28, completion_tokens=11)
    tracker.print_report()
    print(tracker.as_dataframe_dict())