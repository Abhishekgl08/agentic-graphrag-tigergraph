# -*- coding: utf-8 -*-
"""
Full 100-Question Multi-Agent Autonomous Benchmark Runner
=========================================================
Runs all 100 evaluation questions through the 4-Agent Autonomous Control Loop
(QueryRouterAgent -> GraphTraversalAgent -> HybridFusionAgent -> AnswerGenerationAgent)
and computes category breakdown and accuracy gains.
"""

import os
import sys
import json
import time
import dotenv
from pathlib import Path
from collections import defaultdict

# Setup unbuffered UTF-8 output
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
dotenv.load_dotenv(PROJECT_ROOT / ".env", override=True)

from graph.agent import AgenticOrchestrator
from rag.benchmark_comparison import evaluate_match_enhanced

QUESTIONS_FILE = PROJECT_ROOT / "data" / "evaluation_questions.jsonl"
OUTPUT_FILE    = PROJECT_ROOT / "data" / "agentic_multiagent_100_results.json"

print("=" * 80, flush=True)
print("🏆 RUNNING FULL 100-QUESTION MULTI-AGENT AUTONOMOUS BENCHMARK", flush=True)
print("=" * 80, flush=True)

if not QUESTIONS_FILE.exists():
    print(f"Error: {QUESTIONS_FILE} not found", flush=True)
    sys.exit(1)

questions = []
with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            questions.append(json.loads(line))

orchestrator = AgenticOrchestrator()

correct_count = 0
by_type = defaultdict(lambda: {"total": 0, "correct": 0})
results = []

start_total_time = time.time()

for idx, q in enumerate(questions, 1):
    qid = q["qid"]
    question = q["question"]
    qtype = q["qtype"]
    ground_truth = q["answer"]
    gold_doc_ids = q.get("gold_doc_ids", [])

    print(f"\n[{idx}/100] QID: {qid} ({qtype.upper()})", flush=True)
    print(f"Question: {question}", flush=True)

    # Process via 4-Agent Orchestrator
    t0 = time.time()
    response = orchestrator.process(question=question, gold_doc_ids=gold_doc_ids, default_qtype=qtype)
    elapsed = time.time() - t0

    # Evaluate answer match
    is_correct, reason = evaluate_match_enhanced(response.answer, ground_truth, qtype)

    by_type[qtype]["total"] += 1
    if is_correct:
        correct_count += 1
        by_type[qtype]["correct"] += 1
        print(f"Result: SUCCESS ✅ | Agent: {response.intent} | Engine: {response.llm_engine} | Answer: '{response.answer}'", flush=True)
    else:
        print(f"Result: FAILED ❌ | Reason: '{reason}' | Answer: '{response.answer}'", flush=True)

    results.append({
        "qid": qid,
        "qtype": qtype,
        "question": question,
        "ground_truth": ground_truth,
        "agent_answer": response.answer,
        "intent_detected": response.intent,
        "llm_engine_used": response.llm_engine,
        "is_correct": is_correct,
        "reason": reason,
        "latency_sec": response.latency_sec
    })

    # Save incremental progress to JSON
    total_elapsed = time.time() - start_total_time
    summary_data = {
        "summary": {
            "total_questions": idx,
            "total_correct": correct_count,
            "accuracy_pct": round((correct_count / idx) * 100.0, 2),
            "total_time_sec": round(total_elapsed, 2),
            "type_breakdown": {
                kt: {
                    "total": vt["total"],
                    "correct": vt["correct"],
                    "accuracy_pct": round((vt["correct"] / vt["total"]) * 100.0, 2) if vt["total"] > 0 else 0.0
                }
                for kt, vt in by_type.items()
            }
        },
        "details": results
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        json.dump(summary_data, out_f, indent=2, ensure_ascii=False)

    # Pace calls to avoid rate limit spikes
    time.sleep(2.5)

print("\n" + "=" * 80, flush=True)
print("🏆 FINAL 100-QUESTION MULTI-AGENT BENCHMARK COMPLETE!", flush=True)
print(f"Total Correct: {correct_count} / 100 ({correct_count:.1f}%)", flush=True)
print("Type Breakdown:", flush=True)
for qt, stat in by_type.items():
    pct = (stat["correct"] / stat["total"]) * 100.0 if stat["total"] > 0 else 0.0
    print(f"  - {qt.upper()}: {stat['correct']} / {stat['total']} ({pct:.1f}%)", flush=True)
print(f"Saved full results report to: {OUTPUT_FILE}", flush=True)
print("=" * 80, flush=True)
