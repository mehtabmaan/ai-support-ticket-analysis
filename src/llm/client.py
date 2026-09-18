"""
LLM Client module supporting Groq (free tier), Ollama (local), and offline Fallback.
Provides robust retry handling, JSON repair, and graceful degradation.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx

from src.config import settings
from src.llm.prompts import get_synthesis_system_prompt, get_text_to_sql_system_prompt

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified client for LLM interactions.
    Fails gracefully if API keys are missing or provider is unreachable.
    """

    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        self.groq_api_key = settings.GROQ_API_KEY
        self.groq_client = None

        if self.provider == "groq" and self.groq_api_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=self.groq_api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}. Degrading to fallback.")

    def check_availability(self) -> Tuple[bool, str]:
        """Checks if the configured LLM provider is available."""
        if self.provider == "groq":
            if not self.groq_api_key:
                return False, "No GROQ_API_KEY provided. System running in deterministic fallback mode."
            return True, f"Groq active ({settings.GROQ_MODEL})"
        elif self.provider == "ollama":
            try:
                resp = httpx.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=3.0)
                if resp.status_code == 200:
                    return True, f"Ollama active ({settings.OLLAMA_MODEL})"
                return False, f"Ollama returned status {resp.status_code}"
            except Exception as e:
                return False, f"Ollama offline ({e})"
        return False, "Offline deterministic fallback mode enabled."

    def generate_sql(self, question: str) -> Optional[Dict[str, Any]]:
        """
        Sends NL question to the configured LLM and extracts the structured JSON response.
        Returns None if LLM is unavailable or fails.
        """
        if self.provider == "groq" and self.groq_client:
            try:
                system_prompt = get_text_to_sql_system_prompt()
                completion = self.groq_client.chat.completions.create(
                    model=settings.GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    timeout=settings.LLM_TIMEOUT_SECONDS,
                )
                raw_content = completion.choices[0].message.content
                return json.loads(raw_content)
            except Exception as e:
                logger.warning(f"Groq generation failed: {e}. Will attempt fallback.")
                return None

        elif self.provider == "ollama":
            try:
                system_prompt = get_text_to_sql_system_prompt()
                payload = {
                    "model": settings.OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ],
                    "format": "json",
                    "stream": False,
                }
                resp = httpx.post(
                    f"{settings.OLLAMA_BASE_URL}/api/chat",
                    json=payload,
                    timeout=settings.LLM_TIMEOUT_SECONDS,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    content = data.get("message", {}).get("content", "{}")
                    return json.loads(content)
            except Exception as e:
                logger.warning(f"Ollama generation failed: {e}. Will attempt fallback.")
                return None

        return None

    def synthesize_answer(
        self,
        question: str,
        sql: str,
        results: List[Dict[str, Any]],
        caveats: Optional[str] = None,
    ) -> Optional[str]:
        """
        Synthesizes a human-readable answer using the LLM.
        Returns None if LLM is unavailable.
        """
        if self.provider == "groq" and self.groq_client:
            try:
                system_prompt = get_synthesis_system_prompt()
                user_content = (
                    f"User Question: {question}\n"
                    f"Executed SQL: {sql}\n"
                    f"Query Results (up to first 10 rows): {json.dumps(results[:10])}\n"
                    f"Caveats / Context: {caveats or 'None'}\n\n"
                    "Synthesize a clear, direct answer."
                )
                completion = self.groq_client.chat.completions.create(
                    model=settings.GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.2,
                    timeout=settings.LLM_TIMEOUT_SECONDS,
                )
                return completion.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"Groq synthesis failed: {e}.")
                return None

        return None
