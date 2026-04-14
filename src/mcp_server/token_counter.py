"""
OpenAI token usage tracker for the MCP server.

Persists cumulative counts to data/token_stats.json so stats survive
across MCP server restarts and can be read by the CLI token-stats command.
"""

import json
import threading
from pathlib import Path

from pydantic import BaseModel

# OpenAI pricing as of 2026 (USD per 1M tokens)
PRICE_EMBEDDING_PER_M = 0.020       # text-embedding-3-small
PRICE_LLM_INPUT_PER_M = 0.150       # gpt-4o-mini input
PRICE_LLM_OUTPUT_PER_M = 0.600      # gpt-4o-mini output

STATS_FILE = Path("data/token_stats.json")


class TokenReport(BaseModel):
    pages_processed: int
    embedding_tokens: int
    llm_input_tokens: int
    llm_output_tokens: int
    estimated_cost_usd: float
    projected_cost_usd: float | None = None


class TokenCounter:
    """Thread-safe singleton token counter with file persistence."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        """Load existing counts from disk (survives MCP server restarts)."""
        if STATS_FILE.exists():
            try:
                data = json.loads(STATS_FILE.read_text())
                self.pages_processed = data.get("pages_processed", 0)
                self.embedding_tokens = data.get("embedding_tokens", 0)
                self.llm_input_tokens = data.get("llm_input_tokens", 0)
                self.llm_output_tokens = data.get("llm_output_tokens", 0)
                return
            except Exception:
                pass
        self.pages_processed = 0
        self.embedding_tokens = 0
        self.llm_input_tokens = 0
        self.llm_output_tokens = 0

    def _save(self) -> None:
        """Persist current counts to disk (called under lock)."""
        STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATS_FILE.write_text(json.dumps({
            "pages_processed": self.pages_processed,
            "embedding_tokens": self.embedding_tokens,
            "llm_input_tokens": self.llm_input_tokens,
            "llm_output_tokens": self.llm_output_tokens,
            "estimated_cost_usd": self._cost(),
        }, indent=2))

    def _cost(self) -> float:
        return (
            self.embedding_tokens / 1_000_000 * PRICE_EMBEDDING_PER_M
            + self.llm_input_tokens / 1_000_000 * PRICE_LLM_INPUT_PER_M
            + self.llm_output_tokens / 1_000_000 * PRICE_LLM_OUTPUT_PER_M
        )

    def add_embedding(self, tokens: int) -> None:
        with self._lock:
            self.embedding_tokens += tokens
            self.pages_processed += 1
            self._save()

    def add_llm(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.llm_input_tokens += input_tokens
            self.llm_output_tokens += output_tokens
            self._save()

    def report(self, total_pages: int | None = None) -> TokenReport:
        with self._lock:
            cost = self._cost()
            projected = None
            if total_pages and self.pages_processed > 0:
                projected = round(cost * total_pages / self.pages_processed, 4)
            return TokenReport(
                pages_processed=self.pages_processed,
                embedding_tokens=self.embedding_tokens,
                llm_input_tokens=self.llm_input_tokens,
                llm_output_tokens=self.llm_output_tokens,
                estimated_cost_usd=round(cost, 6),
                projected_cost_usd=projected,
            )

    def reset(self) -> None:
        with self._lock:
            self.pages_processed = 0
            self.embedding_tokens = 0
            self.llm_input_tokens = 0
            self.llm_output_tokens = 0
            self._save()


# Module-level singleton — shared across all MCP tool calls in one process
token_counter = TokenCounter()
