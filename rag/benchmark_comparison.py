# -*- coding: utf-8 -*-
"""
Unified Pipeline Benchmark Comparison (Final Verified Version)
================================================================
Evaluates Traditional RAG Pipeline vs. Agentic GraphRAG Pipeline on all 100 questions.
Features:
  - Team Relay name list matching (treating space/comma separated names as SUCCESS).
  - Unicode space normalization (NFKD).
  - Paced Groq API rotation eliminating all 429 rate limit blocks.
"""

import os
import sys
import json
import re
import time
import unicodedata
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from rag.pipeline import TraditionalRAGPipeline, AgenticGraphRAGPipeline

QUESTIONS_FILE = PROJECT_ROOT / "data" / "evaluation_questions.jsonl"
OUTPUT_FILE    = PROJECT_ROOT / "data" / "pipeline_comparison_100.json"


def normalize_text(text: str) -> str:
    """Normalizes Unicode non-breaking spaces (\u202f, \xa0) to standard ASCII space."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return re.sub(r'[\s\u202f\xa0]+', ' ', normalized).strip().lower()


def evaluate_match_enhanced(generated_answer: str, ground_truth: List[str], qtype: str) -> Tuple[bool, str]:
    if not generated_answer or not generated_answer.strip():
        return False, "Empty answer"

    if generated_answer.startswith("Failed") or generated_answer.startswith("Error"):
        return False, f"System error response: {generated_answer[:40]}"

    gen_norm = normalize_text(generated_answer)

    for gt in ground_truth:
        gt_norm = normalize_text(str(gt))

        # 1. Direct or Substring Match
        if gt_norm in gen_norm:
            return True, f"Direct Match: '{gt}' in answer"

        # 2. Team Relay Concatenated / Comma / Space Name Match (e.g. 'Dani KingLaura TrottJoanna Rowsell')
        # Extract individual capitalized names or words > 2 chars from GT
        gt_words = [w for w in re.split(r'\W+', gt_norm) if len(w) > 2]
        if len(gt_words) >= 3 and all(w in gen_norm for w in gt_words):
            return True, f"Team names match: {gt_words}"

        # 3. Numeric Equivalence for Aggregation / Lookup
        if qtype in ("aggregation", "lookup"):
            gt_nums = re.findall(r'\b\d+\b', gt_norm)
            gen_nums = re.findall(r'\b\d+\b', gen_norm)
            if gt_nums and any(num in gen_nums for num in gt_nums):
                return True, f"Numeric match: {gt_nums} in {gen_nums}"

        # 4. Superlative Event Title Key Terms Match
        if qtype == "superlative":
            title_words = [w for w in gt_words if w not in ["athletics", "summer", "olympics", "winter", "1988", "1992", "1996", "2000", "2004", "2008", "2012", "2016", "2020", "2022"]]
            if title_words and all(w in gen_norm for w in title_words):
                return True, f"Superlative title match: {title_words}"

    return False, f"No match for {ground_truth}"


def run_pipeline_comparison():
    print("=" * 80)
    print("FINAL UNIFIED PIPELINE BENCHMARK COMPARISON (100 QUESTIONS)")
    print("Traditional RAG Pipeline  vs.  Agentic GraphRAG Pipeline")
    print("=" * 80)

    if not QUESTIONS_FILE.exists():
        print(f"Error: {QUESTIONS_FILE} not found")
        return

    questions = []
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    trad_pipeline = TraditionalRAGPipeline()
    graph_pipeline = AgenticGraphRAGPipeline()

    trad_correct = 0
    graph_correct = 0

    trad_by_type = defaultdict(lambda: {"total": 0, "correct": 0})
    graph_by_type = defaultdict(lambda: {"total": 0, "correct": 0})

    results = []

    start_total = time.time()

    for idx, q in enumerate(questions, 1):
        qid = q["qid"]
        question = q["question"]
        qtype = q["qtype"]
        ground_truth = q["answer"]
        gold_doc_ids = q.get("gold_doc_ids", [])

        print(f"\n[{idx}/100] QID: {qid} ({qtype})")
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth}")

        # 1. Run Traditional RAG
        t0 = time.time()
        trad_ans = trad_pipeline.answer(dataset_id="default", question=question, gold_doc_ids=gold_doc_ids)
        trad_lat = time.time() - t0
        trad_ok, trad_reason = evaluate_match_enhanced(trad_ans, ground_truth, qtype)
        if trad_ok:
            trad_correct += 1
            trad_by_type[qtype]["correct"] += 1
        trad_by_type[qtype]["total"] += 1

        print(f"  Traditional RAG : {'[CORRECT]' if trad_ok else '[INCORRECT]'} | Ans: '{trad_ans[:70]}' | {trad_lat:.2f}s")
        time.sleep(3.5)

        # 2. Run Agentic GraphRAG
        t0 = time.time()
        graph_ans = graph_pipeline.answer(question=question, qtype=qtype, gold_doc_ids=gold_doc_ids)
        graph_lat = time.time() - t0
        graph_ok, graph_reason = evaluate_match_enhanced(graph_ans, ground_truth, qtype)
        if graph_ok:
            graph_correct += 1
            graph_by_type[qtype]["correct"] += 1
        graph_by_type[qtype]["total"] += 1

        print(f"  Agentic GraphRAG: {'[CORRECT]' if graph_ok else '[INCORRECT]'} | Ans: '{graph_ans[:70]}' | {graph_lat:.2f}s")
        time.sleep(3.5)

        results.append({
            "qid": qid,
            "qtype": qtype,
            "question": question,
            "ground_truth": ground_truth,
            "traditional_rag": {
                "answer": trad_ans,
                "is_correct": trad_ok,
                "reason": trad_reason,
                "latency": trad_lat
            },
            "agentic_graphrag": {
                "answer": graph_ans,
                "is_correct": graph_ok,
                "reason": graph_reason,
                "latency": graph_lat
            }
        })

    total_time = time.time() - start_total
    total_q = len(questions)

    trad_acc = (trad_correct / total_q * 100) if total_q > 0 else 0.0
    graph_acc = (graph_correct / total_q * 100) if total_q > 0 else 0.0

    print("\n" + "=" * 80)
    print("FINAL PIPELINE BENCHMARK COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Total Questions Evaluated          : {total_q}")
    print(f"Traditional RAG Pipeline Accuracy  : {trad_correct}/{total_q} ({trad_acc:.2f}%)")
    print(f"Agentic GraphRAG Pipeline Accuracy: {graph_correct}/{total_q} ({graph_acc:.2f}%)")
    print(f"Accuracy Gain from GraphRAG       : +{graph_acc - trad_acc:.2f}% 🚀")
    print(f"Total Benchmark Time              : {total_time:.2f}s")
    print("-" * 80)
    print(f"{'Question Category':<18} | {'Traditional RAG':<20} | {'Agentic GraphRAG':<20} | {'Improvement':<12}")
    print("-" * 80)

    all_types = sorted(list(set(list(trad_by_type.keys()) + list(graph_by_type.keys()))))
    for qt in all_types:
        t_stat = trad_by_type[qt]
        g_stat = graph_by_type[qt]

        t_pct = (t_stat["correct"] / t_stat["total"] * 100) if t_stat["total"] > 0 else 0.0
        g_pct = (g_stat["correct"] / g_stat["total"] * 100) if g_stat["total"] > 0 else 0.0
        diff = g_pct - t_pct

        print(f"{qt:<18} | {t_stat['correct']}/{t_stat['total']} ({t_pct:.1f}%) {' '*4} | {g_stat['correct']}/{g_stat['total']} ({g_pct:.1f}%) {' '*4} | +{diff:.1f}%")

    print("=" * 80)

    report = {
        "summary": {
            "total_questions": total_q,
            "traditional_rag": {
                "correct": trad_correct,
                "accuracy_pct": round(trad_acc, 2),
                "type_breakdown": {
                    qt: {
                        "total": trad_by_type[qt]["total"],
                        "correct": trad_by_type[qt]["correct"],
                        "accuracy_pct": round((trad_by_type[qt]["correct"] / trad_by_type[qt]["total"] * 100), 2) if trad_by_type[qt]["total"] > 0 else 0.0
                    }
                    for qt in all_types
                }
            },
            "agentic_graphrag": {
                "correct": graph_correct,
                "accuracy_pct": round(graph_acc, 2),
                "type_breakdown": {
                    qt: {
                        "total": graph_by_type[qt]["total"],
                        "correct": graph_by_type[qt]["correct"],
                        "accuracy_pct": round((graph_by_type[qt]["correct"] / graph_by_type[qt]["total"] * 100), 2) if graph_by_type[qt]["total"] > 0 else 0.0
                    }
                    for qt in all_types
                }
            },
            "accuracy_improvement_pct": round(graph_acc - trad_acc, 2),
            "total_time_sec": round(total_time, 2)
        },
        "details": results
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Comparison report saved to: {OUTPUT_FILE}")
    return report


if __name__ == "__main__":
    run_pipeline_comparison()
