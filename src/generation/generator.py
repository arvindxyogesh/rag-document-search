"""
Generation module.
Supports two modes:
  1. OpenAI GPT — if OPENAI_API_KEY is set in the environment
  2. Template fallback — for offline demos / CI environments
"""
import logging
import os

logger = logging.getLogger(__name__)


class RAGGenerator:
    """Generates answers conditioned on retrieved context."""

    SYSTEM_PROMPT = (
        "You are a helpful assistant. Answer the user's question using ONLY the "
        "provided context. If the context is insufficient, say so explicitly."
    )

    def __init__(self, config: dict):
        self.config = config["generation"]
        self.mode = "openai" if os.getenv("OPENAI_API_KEY") else "local"
        logger.info(f"Generator mode: {self.mode}")

    def generate(self, query: str, context: str) -> str:
        """Generate an answer given a query and retrieved context string."""
        if self.mode == "openai":
            return self._generate_openai(query, context)
        return self._generate_local(query, context)

    def _generate_openai(self, query: str, context: str) -> str:
        from openai import OpenAI
        client = OpenAI()
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]
        response = client.chat.completions.create(
            model=self.config.get("model", "gpt-3.5-turbo"),
            messages=messages,
            temperature=self.config.get("temperature", 0.0),
            max_tokens=self.config.get("max_tokens", 512),
        )
        return response.choices[0].message.content.strip()

    def _generate_local(self, query: str, context: str) -> str:
        """Template-based fallback — surfaces retrieved context directly."""
        if not context.strip():
            return "No relevant context found. Please try a different query."
        return (
            f"Based on the retrieved context:\n\n"
            f"{context}\n\n"
            f"[Set OPENAI_API_KEY to enable LLM-generated answers for: '{query}']"
        )
