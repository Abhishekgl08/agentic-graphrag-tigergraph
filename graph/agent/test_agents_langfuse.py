# -*- coding: utf-8 -*-
"""
Langfuse Integrated Multi-Agent Preview Runner (3 Questions)
============================================================
Tests the 4-Agent Control Loop with Langfuse Observability Tracing on 3 sample questions.
"""

import os
import sys
import json
import time
import dotenv
from pathlib import Path

# Setup unbuffered UTF-8 output
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
dotenv.load_dotenv(PROJECT_ROOT / ".env", override=True)

from langfuse import Langfuse
from graph.agent import AgenticOrchestrator
from rag.benchmark_comparison import evaluate_match_enhanced

print("=" * 80, flush=True)
print("🤖 TESTING 4-AGENT AUTONOMOUS SYSTEM WITH LANGFUSE OBSERVABILITY (3 QUESTIONS)", flush=True)
print("=" * 80, flush=True)

# Initialize Langfuse client
secret_key = os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-9cc129b6-87b1-4e06-b60d-b58b1b223c74")
public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-747945a4-5649-43f4-929f-cadf786c056d")
host = os.getenv("LANGFUSE_HOST", "https://hipaa.cloud.langfuse.com")

langfuse_client = Langfuse(
    secret_key=secret_key,
    public_key=public_key,
    host=host
)

sample_qids = ["pub-001", "pub-002", "pub-004"]

questions_map = {}
with open(PROJECT_ROOT / "data" / "evaluation_questions.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            q = json.loads(line)
            questions_map[q["qid"]] = q

orchestrator = AgenticOrchestrator()

preview_results = []

for idx, qid in enumerate(sample_qids, 1):
    q = questions_map[qid]
    question = q["question"]
    qtype = q["qtype"]
    ground_truth = q["answer"]
    gold_doc_ids = q.get("gold_doc_ids", [])

    print(f"\n----------------------------------------------------------------------", flush=True)
    print(f"[{idx}/3] TESTING QID: {qid} ({qtype.upper()})", flush=True)
    print(f"Question: {question}", flush=True)
    print(f"Ground Truth: {ground_truth}", flush=True)

    # Process via Langfuse-instrumented 4-Agent Orchestrator
    response = orchestrator.process(question=question, gold_doc_ids=gold_doc_ids, default_qtype=qtype)

    # Evaluate match
    is_correct, reason = evaluate_match_enhanced(response.answer, ground_truth, qtype)

    print(f"\n🤖 AGENT CONTROL LOOP METADATA:", flush=True)
    print(f"  - Agent 1 Router Decision: Intent='{response.intent}'", flush=True)
    print(f"  - Agent 2 Graph Traversal: Summary Count={response.traversal_result.summary_count}", flush=True)
    print(f"  - Agent 3 Hybrid Fusion: Fused {response.fused_context.char_length} chars", flush=True)
    print(f"  - Agent 4 LLM Generation: Engine={response.llm_engine} | Latency={response.latency_sec:.2f}s", flush=True)
    print(f"  - System Output Answer: '{response.answer}'", flush=True)
    print(f"  - Match Evaluation: is_correct={is_correct} | reason='{reason}'", flush=True)

    preview_results.append({
        "qid": qid,
        "qtype": qtype,
        "is_correct": is_correct,
        "reason": reason,
        "answer": response.answer,
        "llm_engine": response.llm_engine
    })

    time.sleep(1.0)

# Flush traces to Langfuse Dashboard
langfuse_client.flush()
print("\n📡 Langfuse Traces successfully flushed to https://hipaa.cloud.langfuse.com dashboard!", flush=True)

passed = sum(1 for r in preview_results if r["is_correct"])
print("=" * 80, flush=True)
print(f"SUMMARY OF 3-QUESTION LANGFUSE PREVIEW TEST: Passed {passed} / 3 ({passed / 3.0 * 100:.1f}%)", flush=True)
print("=" * 80, flush=True)
