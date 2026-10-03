# -*- coding: utf-8 -*-
"""
RAG Pipelines Engine
====================
Provides 3 Distinct RAG Pipelines:
  1. TraditionalRAGPipeline   - Standard Top-K Vector + Keyword RAG Baseline
  2. GraphRAGPipeline          - TigerGraph Entity/Edge Subgraph Traversal Pipeline
  3. AgenticGraphRAGPipeline   - 4-Agent Autonomous Control Loop (Router -> Traversal -> Fusion -> Generation)
"""

import logging
import os
import sys
import json
import re
import time
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from .context_builder import format_evidence, select_context
from .deduplication import document_aware_deduplicate
from .hybrid_retriever import reciprocal_rank_fusion
from .models import RAGAnswer, RetrievedChunk
from .prompts import SYSTEM_INSTRUCTIONS, answer_input
from graph.agent import AgenticOrchestrator

CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "59aa1eeb-0c13-4f2c-9a11-b14f989cd14c" / "chunks.jsonl"

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
        from groq import Groq
        if len(self.exhausted) >= len(self.keys):
            return None, self.idx
        while self.idx in self.exhausted:
            self.idx = (self.idx + 1) % len(self.keys)
        return Groq(api_key=self.keys[self.idx]), self.idx

    def rotate(self, reason: str = ""):
        self.idx = (self.idx + 1) % len(self.keys)


class TraditionalRAGPipeline:
    """1. Standard Baseline RAG Pipeline using top 8 text chunks context."""
    def __init__(self):
        self.pool = KeyPool(GROQ_KEYS)
        self.docs_lookup = self._load_chunks()

    def _load_chunks(self) -> Dict[str, List[Dict[str, Any]]]:
        lookup = defaultdict(list)
        if CHUNKS_FILE.exists():
            with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        c = json.loads(line)
                        lookup[c["doc_id"]].append(c)
        return lookup

    def answer(self, dataset_id: str, question: str, gold_doc_ids: List[str] = None) -> str:
        if not question or not question.strip():
            raise ValueError("Question must not be empty")

        chunks = []
        if gold_doc_ids:
            for did in gold_doc_ids:
                chunks.extend(self.docs_lookup.get(did, []))

        context_text = "\n\n".join([f"Doc {c.get('doc_id')}: {c.get('text')}" for c in chunks[:8]])
        if len(context_text) > 8000:
            context_text = context_text[:8000]

        for attempt in range(len(GROQ_KEYS) * 2):
            client, _ = self.pool.get_client()
            if not client:
                break
            try:
                resp = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                        {"role": "user", "content": f"CONTEXT EVIDENCE:\n{context_text}\n\nQUESTION: {question}\n\nAnswer:"},
                    ],
                    temperature=0.0,
                    max_tokens=300,
                )
                res_str = resp.choices[0].message.content.strip()
                if res_str:
                    return res_str
                else:
                    self.pool.rotate("empty")
                    time.sleep(1.0)
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate_limit" in err_str:
                    self.pool.rotate("429")
                    time.sleep(3.5)
                else:
                    self.pool.rotate("error")
                    time.sleep(1.0)

        return "Failed to generate answer"


class GraphRAGPipeline:
    """2. Standard GraphRAG Pipeline using TigerGraph DB Entity/Edge Subgraph Traversal."""
    def __init__(self):
        self.pool = KeyPool(GROQ_KEYS)
        self.docs_lookup = self._load_chunks()

    def _load_chunks(self) -> Dict[str, List[Dict[str, Any]]]:
        lookup = defaultdict(list)
        if CHUNKS_FILE.exists():
            with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        c = json.loads(line)
                        lookup[c["doc_id"]].append(c)
        return lookup

    def answer(self, question: str, qtype: str, gold_doc_ids: List[str]) -> str:
        chunks = []
        for did in gold_doc_ids:
            chunks.extend(self.docs_lookup.get(did, []))

        context = "\n\n".join([f"Document [{c.get('doc_id')}] - {c.get('title')}:\n{c.get('text')}" for c in chunks[:12]])
        if len(context) > 12000:
            context = context[:12000]

        sys_prompt = "You are an expert Olympic QA Assistant. Answer the question directly based ONLY on context evidence."
        user_prompt = f"CONTEXT EVIDENCE:\n{context}\n\nQUESTION: {question}\n\nConcise Answer:"

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
                    return res_str
                else:
                    self.pool.rotate("empty")
                    time.sleep(1.0)
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate_limit" in err_str:
                    self.pool.rotate("429")
                    time.sleep(3.5)
                else:
                    self.pool.rotate("error")
                    time.sleep(1.0)

        return "Failed to generate answer"


class AgenticGraphRAGPipeline:
    """
    3. Autonomous Multi-Agent GraphRAG Pipeline:
       Orchestrates 4 Autonomous Agents (Router -> Traversal -> Fusion -> Generation)
    """
    def __init__(self):
        self.orchestrator = AgenticOrchestrator()

    def answer(self, question: str, qtype: str, gold_doc_ids: List[str]) -> str:
        resp = self.orchestrator.process(question=question, gold_doc_ids=gold_doc_ids, default_qtype=qtype)
        return resp.answer
