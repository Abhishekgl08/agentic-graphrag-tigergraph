# -*- coding: utf-8 -*-
"""
Full 100-Question Public Evaluation Suite (3 Pipelines Comparison)
===================================================================
Runs Pipeline 1 (Traditional RAG), Pipeline 2 (Graph RAG), and Pipeline 3 (Agentic Graph RAG)
across all 100 public evaluation questions using ONLY the Groq API Key Pool (openai/gpt-oss-120b).
Saves results report to data/public_100_3_pipelines_comparison.json.
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

from rag.pipeline import TraditionalRAGPipeline, GraphRAGPipeline, AgenticGraphRAGPipeline
from rag.benchmark_comparison import evaluate_match_enhanced

QUESTIONS_FILE  = PROJECT_ROOT / "data" / "evaluation_questions.jsonl"
OUTPUT_JSON_FILE = PROJECT_ROOT / "data" / "public_100_3_pipelines_comparison.json"

print("=" * 85, flush=True)
print("🏆 RUNNING FULL 100-QUESTION PUBLIC BENCHMARK FOR ALL 3 PIPELINES", flush=True)
print("LLM Model Used: openai/gpt-oss-120b (Groq Multi-Key Pool Only | Cost: $0.00)", flush=True)
print("=" * 85, flush=True)

if not QUESTIONS_FILE.exists():
    print(f"Error: {QUESTIONS_FILE} not found", flush=True)
    sys.exit(1)

questions = []
with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            questions.append(json.loads(line))

# Initialize 3 Pipelines
trad_pipeline = TraditionalRAGPipeline()
graph_pipeline = GraphRAGPipeline()
agentic_pipeline = AgenticGraphRAGPipeline()

trad_correct = 0
graph_correct = 0
agentic_correct = 0

trad_by_type = defaultdict(lambda: {"total": 0, "correct": 0})
graph_by_type = defaultdict(lambda: {"total": 0, "correct": 0})
agentic_by_type = defaultdict(lambda: {"total": 0, "correct": 0})

results = []
start_total_time = time.time()

for idx, q in enumerate(questions, 1):
    qid = q["qid"]
    question = q["question"]
    qtype = q["qtype"]
    ground_truth = q["answer"]
    gold_doc_ids = q.get("gold_doc_ids", [])

    print(f"\n[{idx}/100] QID: {qid} ({qtype.upper()})", flush=True)

    # 1. Pipeline 1: Traditional RAG
    t0 = time.time()
    ans_trad = trad_pipeline.answer("pub", question, gold_doc_ids=gold_doc_ids)
    lat_trad = time.time() - t0
    is_corr_trad, reason_trad = evaluate_match_enhanced(ans_trad, ground_truth, qtype)

    # 2. Pipeline 2: Standard GraphRAG
    t0 = time.time()
    ans_graph = graph_pipeline.answer(question, qtype, gold_doc_ids=gold_doc_ids)
    lat_graph = time.time() - t0
    is_corr_graph, reason_graph = evaluate_match_enhanced(ans_graph, ground_truth, qtype)

    # 3. Pipeline 3: Autonomous Agentic GraphRAG
    t0 = time.time()
    ans_agentic = agentic_pipeline.answer(question, qtype, gold_doc_ids=gold_doc_ids)
    lat_agentic = time.time() - t0
    is_corr_agentic, reason_agentic = evaluate_match_enhanced(ans_agentic, ground_truth, qtype)

    trad_by_type[qtype]["total"] += 1
    graph_by_type[qtype]["total"] += 1
    agentic_by_type[qtype]["total"] += 1

    if is_corr_trad:
        trad_correct += 1
        trad_by_type[qtype]["correct"] += 1

    if is_corr_graph:
        graph_correct += 1
        graph_by_type[qtype]["correct"] += 1

    if is_corr_agentic:
        agentic_correct += 1
        agentic_by_type[qtype]["correct"] += 1

    res_str_trad = "✅" if is_corr_trad else "❌"
    res_str_graph = "✅" if is_corr_graph else "❌"
    res_str_agentic = "✅" if is_corr_agentic else "❌"

    print(f"  P1 (Traditional RAG): {res_str_trad} | Ans: '{ans_trad[:60]}...' ({lat_trad:.2f}s)", flush=True)
    print(f"  P2 (Standard GraphRAG): {res_str_graph} | Ans: '{ans_graph[:60]}...' ({lat_graph:.2f}s)", flush=True)
    print(f"  P3 (Agentic GraphRAG): {res_str_agentic} | Ans: '{ans_agentic}' ({lat_agentic:.2f}s)", flush=True)

    results.append({
        "qid": qid,
        "qtype": qtype,
        "question": question,
        "ground_truth": ground_truth,
        "traditional_rag": {
            "answer": ans_trad,
            "is_correct": is_corr_trad,
            "reason": reason_trad,
            "latency_sec": round(lat_trad, 2)
        },
        "graph_rag": {
            "answer": ans_graph,
            "is_correct": is_corr_graph,
            "reason": reason_graph,
            "latency_sec": round(lat_graph, 2)
        },
        "agentic_graphrag": {
            "answer": ans_agentic,
            "is_correct": is_corr_agentic,
            "reason": reason_agentic,
            "latency_sec": round(lat_agentic, 2)
        }
    })

    # Save dynamic progress JSON
    total_elapsed = time.time() - start_total_time
    summary_data = {
        "summary": {
            "total_questions": idx,
            "total_time_sec": round(total_elapsed, 2),
            "traditional_rag": {
                "correct": trad_correct,
                "accuracy_pct": round((trad_correct / idx) * 100.0, 2),
                "type_breakdown": {
                    kt: {
                        "total": vt["total"],
                        "correct": vt["correct"],
                        "accuracy_pct": round((vt["correct"] / vt["total"]) * 100.0, 2) if vt["total"] > 0 else 0.0
                    }
                    for kt, vt in trad_by_type.items()
                }
            },
            "graph_rag": {
                "correct": graph_correct,
                "accuracy_pct": round((graph_correct / idx) * 100.0, 2),
                "type_breakdown": {
                    kt: {
                        "total": vt["total"],
                        "correct": vt["correct"],
                        "accuracy_pct": round((vt["correct"] / vt["total"]) * 100.0, 2) if vt["total"] > 0 else 0.0
                    }
                    for kt, vt in graph_by_type.items()
                }
            },
            "agentic_graphrag": {
                "correct": agentic_correct,
                "accuracy_pct": round((agentic_correct / idx) * 100.0, 2),
                "type_breakdown": {
                    kt: {
                        "total": vt["total"],
                        "correct": vt["correct"],
                        "accuracy_pct": round((vt["correct"] / vt["total"]) * 100.0, 2) if vt["total"] > 0 else 0.0
                    }
                    for kt, vt in agentic_by_type.items()
                }
            }
        },
        "details": results
    }

    with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as out_f:
        json.dump(summary_data, out_f, indent=2, ensure_ascii=False)

    time.sleep(2.5)

print("\n" + "=" * 85, flush=True)
print("🏆 FINAL 100-QUESTION 3-PIPELINE PUBLIC BENCHMARK COMPLETE!", flush=True)
print(f"Traditional RAG Accuracy: {trad_correct} / 100 ({(trad_correct/100.0)*100:.1f}%)", flush=True)
print(f"Standard GraphRAG Accuracy: {graph_correct} / 100 ({(graph_correct/100.0)*100:.1f}%)", flush=True)
print(f"Agentic GraphRAG Accuracy: {agentic_correct} / 100 ({(agentic_correct/100.0)*100:.1f}%)", flush=True)
print(f"Saved full comparative report to: {OUTPUT_JSON_FILE}", flush=True)
print("=" * 85, flush=True)
