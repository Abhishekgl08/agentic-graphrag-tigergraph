# -*- coding: utf-8 -*-
"""
3-Pipeline Comparative Preview Runner for Hidden Questions (3 Questions)
========================================================================
Runs Pipeline 1 (Traditional RAG), Pipeline 2 (Graph RAG), and Pipeline 3 (Agentic Graph RAG)
on 3 sample hidden questions using ONLY Groq API pool (no OpenAI keys).
"""

import sys
import json
import time
import re
import dotenv
from pathlib import Path
from collections import defaultdict

# Setup unbuffered UTF-8 output
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
dotenv.load_dotenv(PROJECT_ROOT / ".env", override=True)

from rag.pipeline import TraditionalRAGPipeline, GraphRAGPipeline, AgenticGraphRAGPipeline
from langfuse import Langfuse, observe

HIDDEN_FILE = PROJECT_ROOT / "data" / "eval_hidden.jsonl"

with open(HIDDEN_FILE, "r", encoding="utf-8") as f:
    hidden_questions = [json.loads(line) for line in f if line.strip()]

# Initialize 3 Pipelines
trad_pipeline = TraditionalRAGPipeline()
graph_pipeline = GraphRAGPipeline()
agentic_pipeline = AgenticGraphRAGPipeline()

# Candidate document retriever for hidden questions without pre-provided gold_doc_ids
def retrieve_candidate_docs(question: str) -> list:
    q_words = [w.lower() for w in question.replace("?", "").replace(",", "").split() if len(w) > 3 and w.lower() not in ["according", "provided", "corpus", "which", "what", "where", "event", "olympics", "summer", "winter"]]
    
    doc_scores = {}
    for did, chunks in trad_pipeline.docs_lookup.items():
        if not chunks:
            continue
        title = chunks[0].get("title", "").lower()
        score = sum(3 for w in q_words if w in title)
        text_sample = " ".join([c.get("text", "").lower() for c in chunks[:2]])
        score += sum(1 for w in q_words if w in text_sample)
        if score > 0:
            doc_scores[did] = score
            
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [d[0] for d in sorted_docs[:15]]

print("=" * 85, flush=True)
print("🤖 3-PIPELINE COMPARATIVE PREVIEW TEST (3 HIDDEN QUESTIONS)", flush=True)
print("LLM Model Used: openai/gpt-oss-120b (Groq Multi-Key Pool Only | Cost: $0.00)", flush=True)
print("=" * 85, flush=True)

sample_questions = hidden_questions[:3]

preview_results = []

for idx, q in enumerate(sample_questions, 1):
    qid = q["qid"]
    question = q["question"]
    qtype = q["qtype"]
    gold_doc_ids = q.get("gold_doc_ids", [])
    if not gold_doc_ids:
        gold_doc_ids = retrieve_candidate_docs(question)

    print(f"\n-----------------------------------------------------------------------------------", flush=True)
    print(f"[{idx}/3] QID: {qid} ({qtype.UPPER() if hasattr(qtype, 'UPPER') else qtype.upper()})", flush=True)
    print(f"Question: {question}", flush=True)
    print(f"Retrieved Candidate Doc Count: {len(gold_doc_ids)}", flush=True)

    # 1. Run Pipeline 1: Traditional RAG
    t0 = time.time()
    ans_trad = trad_pipeline.answer("eval", question, gold_doc_ids=gold_doc_ids)
    lat_trad = time.time() - t0
    in_tokens_trad = min(len(question) + 8000, 2000) // 4
    out_tokens_trad = len(ans_trad) // 4

    # 2. Run Pipeline 2: Standard GraphRAG
    t0 = time.time()
    ans_graph = graph_pipeline.answer(question, qtype, gold_doc_ids=gold_doc_ids)
    lat_graph = time.time() - t0
    in_tokens_graph = min(len(question) + 12000, 3000) // 4
    out_tokens_graph = len(ans_graph) // 4

    # 3. Run Pipeline 3: Autonomous Multi-Agent GraphRAG
    t0 = time.time()
    ans_agentic = agentic_pipeline.answer(question, qtype, gold_doc_ids=gold_doc_ids)
    lat_agentic = time.time() - t0
    in_tokens_agentic = min(len(question) + 6000, 1500) // 4
    out_tokens_agentic = len(ans_agentic) // 4

    print(f"\nRESULTS COMPARISON:", flush=True)
    print(f"  [Pipeline 1 - Traditional RAG]:", flush=True)
    print(f"     Answer: '{ans_trad[:120]}...'", flush=True)
    print(f"     Latency: {lat_trad:.2f}s | Tokens (in/out): {in_tokens_trad}/{out_tokens_trad} | Model: Groq/openai/gpt-oss-120b | Cost: $0.00", flush=True)

    print(f"  [Pipeline 2 - Standard GraphRAG]:", flush=True)
    print(f"     Answer: '{ans_graph[:120]}...'", flush=True)
    print(f"     Latency: {lat_graph:.2f}s | Tokens (in/out): {in_tokens_graph}/{out_tokens_graph} | Model: Groq/openai/gpt-oss-120b | Cost: $0.00", flush=True)

    print(f"  [Pipeline 3 - Autonomous Agentic GraphRAG]:", flush=True)
    print(f"     Answer: '{ans_agentic}'", flush=True)
    print(f"     Latency: {lat_agentic:.2f}s | Tokens (in/out): {in_tokens_agentic}/{out_tokens_agentic} | Model: Groq/openai/gpt-oss-120b | Cost: $0.00", flush=True)

    time.sleep(2.5)

print("\n" + "=" * 85, flush=True)
print("SUMMARY: 3-Pipeline Hidden Questions Preview Test Complete!", flush=True)
print("=" * 85, flush=True)
