# -*- coding: utf-8 -*-
"""
Multi-Agent System Preview Test Runner (3 Questions)
===================================================
Tests the 4-Agent Control Loop on 3 distinct question types before full 100-question run.
"""

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

from graph.agent import AgenticOrchestrator
from rag.benchmark_comparison import evaluate_match_enhanced

print("=" * 80, flush=True)
print("🤖 TESTING 4-AGENT AUTONOMOUS SYSTEM ON 3 PREVIEW QUESTIONS", flush=True)
print("=" * 80, flush=True)

# Select 3 sample evaluation questions (Aggregation, Temporal, Superlative)
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

    # Execute multi-agent control loop
    response = orchestrator.process(question=question, gold_doc_ids=gold_doc_ids, default_qtype=qtype)

    # Evaluate match
    is_correct, reason = evaluate_match_enhanced(response.answer, ground_truth, qtype)

    print(f"\n🤖 AGENT CONTROL LOOP METADATA:", flush=True)
    print(f"  - Agent 1 Router Decision: Intent='{response.intent}' (Reason: {response.routing_decision.reasoning})", flush=True)
    print(f"  - Agent 2 Graph Traversal: Retrieved {response.traversal_result.retrieved_chunk_count} chunks across {response.traversal_result.summary_count} event summaries", flush=True)
    print(f"  - Agent 3 Hybrid Fusion: Fused {response.fused_context.char_length} characters context", flush=True)
    print(f"  - Agent 4 LLM Generation: Engine={response.llm_engine} | Latency={response.latency_sec:.2f}s", flush=True)
    print(f"  - Final System Answer: '{response.answer}'", flush=True)
    print(f"  - Match Evaluation: is_correct={is_correct} | reason='{reason}'", flush=True)

    preview_results.append({
        "qid": qid,
        "qtype": qtype,
        "question": question,
        "ground_truth": ground_truth,
        "agent_answer": response.answer,
        "intent_detected": response.intent,
        "llm_engine": response.llm_engine,
        "is_correct": is_correct,
        "evaluation_reason": reason,
        "latency_sec": response.latency_sec
    })

    time.sleep(1.0)

print("\n" + "=" * 80, flush=True)
print("SUMMARY OF 3-QUESTION PREVIEW TEST:", flush=True)
passed_count = sum(1 for r in preview_results if r["is_correct"])
print(f"Passed: {passed_count} / 3 ({passed_count / 3.0 * 100:.1f}%)", flush=True)
print("=" * 80, flush=True)
