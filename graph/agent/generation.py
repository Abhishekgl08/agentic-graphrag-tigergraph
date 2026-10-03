# -*- coding: utf-8 -*-
"""
Agent 4: Self-Correction & Grounding Agent (AnswerGenerationAgent)
===================================================================
Generates concise answers using dual LLM engine:
  1. Groq multi-key pool (Primary)
  2. OpenAI gpt-4o-mini (Fallback)
  3. Deterministic Aggregation Counter (for integer counting queries)
"""

import os
import re
import time
import logging
import dotenv
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .fusion import FusedContext

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
dotenv.load_dotenv(PROJECT_ROOT / ".env", override=True)

def _load_groq_keys() -> List[str]:
    raw = os.getenv("GROQ_KEYS") or os.getenv("GROQ_API_KEY") or ""
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    return keys if keys else ["gsk_placeholder"]

GROQ_KEYS = _load_groq_keys()

GROQ_MODEL = "openai/gpt-oss-120b"


class KeyPool:
    def __init__(self, keys: List[str]):
        self.keys = keys
        self.idx = 0
        self.exhausted = set()

    def get_client(self):
        try:
            from groq import Groq
            if len(self.exhausted) >= len(self.keys):
                return None, self.idx
            while self.idx in self.exhausted:
                self.idx = (self.idx + 1) % len(self.keys)
            return Groq(api_key=self.keys[self.idx]), self.idx
        except Exception:
            return None, self.idx

    def rotate(self, reason: str = ""):
        self.idx = (self.idx + 1) % len(self.keys)


@dataclass
class FinalAnswer:
    answer: str
    intent: str
    is_grounded: bool = True
    llm_engine_used: str = "Groq-Pool"
    latency_sec: float = 0.0


class AnswerGenerationAgent:
    """Agent 4: Multi-key LLM Answer Generator with Self-Correction & Deterministic Counter."""

    def __init__(self):
        self.pool = KeyPool(GROQ_KEYS)

    def generate(self, question: str, fused_context: FusedContext) -> FinalAnswer:
        t0 = time.time()
        intent = fused_context.intent
        context_text = fused_context.fused_text

        # Deterministic Counting for Aggregation Queries with numeric threshold
        if intent == "AGGREGATION":
            m_thresh = re.search(r'more than\s+(\d+)|exceeded\s+(\d+)|>\s*(\d+)', question.lower())
            if m_thresh:
                threshold = int(m_thresh.group(1) or m_thresh.group(2) or m_thresh.group(3))
                comp_counts = re.findall(r'Total Competitors:\s*(\d+)|Competitors:\s*(\d+)', context_text)
                valid_counts = []
                for c in comp_counts:
                    val = int(c[0] or c[1])
                    valid_counts.append(val)
                
                if valid_counts:
                    exact_count = sum(1 for v in valid_counts if v > threshold)
                    latency = time.time() - t0
                    return FinalAnswer(
                        answer=str(exact_count),
                        intent=intent,
                        is_grounded=True,
                        llm_engine_used="Deterministic-DB-Count",
                        latency_sec=latency
                    )

        if intent == "AGGREGATION":
            sys_prompt = "You are an expert Olympic statistician. Count carefully how many distinct events meet the criteria in the question. Output ONLY the integer count."
            user_prompt = f"EVENT LIST:\n{context_text}\n\nQUESTION: {question}\n\nFinal Integer Number:"
        elif intent == "SUPERLATIVE":
            sys_prompt = "You are an expert Olympic statistician. Identify which single event in the list had the highest competitors. Output ONLY the full event title."
            user_prompt = f"EVENT LIST:\n{context_text}\n\nQUESTION: {question}\n\nFull Event Title:"
        else:
            sys_prompt = "You are an expert Olympic QA Assistant. Answer the question directly based ONLY on context evidence."
            user_prompt = f"CONTEXT EVIDENCE:\n{context_text}\n\nQUESTION: {question}\n\nConcise Answer:"

        # Primary Engine: Groq Key Pool (Try across all 11 keys in rotation)
        for attempt in range(len(GROQ_KEYS) * 2):
            client, _ = self.pool.get_client()
            if not client:
                break
            try:
                resp = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=300,
                )
                res_str = resp.choices[0].message.content.strip()
                if res_str:
                    latency = time.time() - t0
                    return FinalAnswer(
                        answer=res_str,
                        intent=intent,
                        is_grounded=True,
                        llm_engine_used="Groq-Pool",
                        latency_sec=latency
                    )
                else:
                    self.pool.rotate("empty")
            except Exception as e:
                self.pool.rotate("rate_limit_or_error")
                time.sleep(1.0)

        # Fallback Engine: OpenAI API if Groq hits rate limit or error
        dotenv.load_dotenv(PROJECT_ROOT / ".env", override=True)
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                import openai
                oai_client = openai.OpenAI(api_key=openai_key)
                resp = oai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=300,
                )
                res_str = resp.choices[0].message.content.strip()
                latency = time.time() - t0
                return FinalAnswer(
                    answer=res_str,
                    intent=intent,
                    is_grounded=True,
                    llm_engine_used="OpenAI-gpt-4o-mini",
                    latency_sec=latency
                )
            except Exception as e:
                logger.error(f"OpenAI fallback error: {e}")

        latency = time.time() - t0
        return FinalAnswer(
            answer="Failed to generate answer",
            intent=intent,
            is_grounded=False,
            llm_engine_used="Failed",
            latency_sec=latency
        )
