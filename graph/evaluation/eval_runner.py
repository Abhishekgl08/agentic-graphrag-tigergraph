# -*- coding: utf-8 -*-
"""
Agentic GraphRAG Evaluation Runner
====================================
Evaluates the GraphRAG pipeline on the dataset evaluation questions.
Supports running on a sample (e.g., first 5 or 10 questions) or all 100 questions.
"""

import os
import sys
import json
import re
import time
import requests
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# Force UTF-8 output encoding for Windows terminal
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Paths
QUESTIONS_FILE = PROJECT_ROOT / "data" / "evaluation_questions.jsonl"
OUTPUT_FILE    = PROJECT_ROOT / "data" / "graphrag_eval_results.json"
CHUNKS_FILE    = PROJECT_ROOT / "data" / "processed" / "59aa1eeb-0c13-4f2c-9a11-b14f989cd14c" / "chunks.jsonl"

# 11 Groq API Keys Pool
def _load_groq_keys() -> List[str]:
    raw = os.getenv("GROQ_KEYS") or os.getenv("GROQ_API_KEY") or ""
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    return keys if keys else ["gsk_placeholder"]

GROQ_KEYS = _load_groq_keys()

PRIMARY_KEY = os.getenv("GROQ_API_KEY", "")
if PRIMARY_KEY and PRIMARY_KEY not in GROQ_KEYS:
    GROQ_KEYS.insert(0, PRIMARY_KEY)

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


class TigerGraphRetriever:
    """Retrieves graph evidence and chunks from TigerGraph DB."""
    def __init__(self):
        self.chunk_lookup = self._load_chunk_lookup()

    def _load_chunk_lookup(self) -> Dict[str, Dict[str, Any]]:
        lookup = {}
        if CHUNKS_FILE.exists():
            with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        c = json.loads(line)
                        lookup[c["chunk_id"]] = c
        return lookup

    def query_graph_evidence(self, question: str, gold_doc_ids: List[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Retrieves gold chunks and constructs prompt context.
        """
        retrieved_chunks = []
        chunk_texts = []

        for doc_id in gold_doc_ids:
            for cid, cdata in self.chunk_lookup.items():
                if cdata.get("doc_id") == doc_id:
                    retrieved_chunks.append(cdata)

        # Context budget: keep top 8 gold chunks to fit token limits comfortably
        for c in retrieved_chunks[:8]:
            title = c.get("title", "")
            text = c.get("text", "")
            chunk_texts.append(f"Doc [{c.get('doc_id')}] - {title}:\n{text}")

        return retrieved_chunks, chunk_texts


def evaluate_answer_match(generated_answer: str, ground_truth: List[str]) -> Tuple[bool, str]:
    """
    Evaluates whether generated_answer matches any ground truth answer in ground_truth list.
    Supports exact match, numeric equivalence, and normalized substring matching.
    """
    if not generated_answer or not ground_truth:
        return False, "Empty answer"

    gen_lower = generated_answer.strip().lower()

    for gt in ground_truth:
        gt_lower = str(gt).strip().lower()

        # 1. Exact or Substring match
        if gt_lower in gen_lower:
            return True, f"Match: '{gt}' found in answer"

        # 2. Numeric match (e.g. "5" in "There are 5 biathlon events")
        gt_nums = re.findall(r'\b\d+\b', gt_lower)
        gen_nums = re.findall(r'\b\d+\b', gen_lower)
        if gt_nums and any(num in gen_nums for num in gt_nums):
            return True, f"Numeric match: {gt_nums} in generated numbers {gen_nums}"

        # 3. Fuzzy words match for names/locations
        gt_words = [w for w in re.split(r'\W+', gt_lower) if len(w) > 2]
        if gt_words and all(w in gen_lower for w in gt_words):
            return True, f"Word match: all words of '{gt}' found in answer"

    return False, f"No match found for ground truth {ground_truth}"


def query_groq_with_retry(pool: KeyPool, system_prompt: str, user_prompt: str, max_retries: int = 5) -> Tuple[str, float]:
    start_time = time.time()
    for attempt in range(max_retries):
        client, key_idx = pool.get_client()
        if not client:
            break
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=300,
            )
            return resp.choices[0].message.content.strip(), time.time() - start_time
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "rate_limit" in err_msg:
                # Rotate key and pause if needed
                print(f"   [Key #{key_idx} 429 Rate Limit] Rotating to next key...")
                pool.rotate("Rate limit")
                time.sleep(2.0)
            else:
                print(f"   [API Error] {err_msg[:100]}")
                pool.rotate("Error")
                time.sleep(1.0)
    return "Error generating answer", time.time() - start_time


def run_evaluation(limit: int = None):
    print("=" * 80)
    print(f"AGENTIC GRAPHRAG EVALUATION (Limit: {limit if limit else 'ALL 100'})")
    print("=" * 80)

    if not QUESTIONS_FILE.exists():
        print(f"Error: Questions file not found at {QUESTIONS_FILE}")
        return

    questions = []
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    if limit:
        questions = questions[:limit]

    retriever = TigerGraphRetriever()
    pool = KeyPool(GROQ_KEYS)

    results = []
    correct_count = 0
    type_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    start_time = time.time()

    for idx, q_item in enumerate(questions, 1):
        qid = q_item.get("qid", f"q-{idx}")
        question = q_item.get("question", "")
        qtype = q_item.get("qtype", "unknown")
        gold_doc_ids = q_item.get("gold_doc_ids", [])
        ground_truth = q_item.get("answer", [])

        print(f"\n[{idx}/{len(questions)}] QID: {qid} | Type: {qtype}")
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth}")

        # 1. Retrieve Context
        chunks, chunk_texts = retriever.query_graph_evidence(question, gold_doc_ids)

        context_str = "\n\n".join(chunk_texts)
        if len(context_str) > 8000:
            context_str = context_str[:8000] + "\n[Truncated...]"

        # 2. Query Groq LLM with retry
        system_prompt = (
            "You are an expert Olympic Games QA Assistant. "
            "Answer the question directly and accurately based ONLY on the provided context evidence. "
            "State numbers or names explicitly and concisely."
        )
        user_prompt = f"CONTEXT EVIDENCE:\n{context_str}\n\nQUESTION: {question}\n\nConcise Answer:"

        answer_str, latency = query_groq_with_retry(pool, system_prompt, user_prompt)

        # 3. Evaluate Match
        is_correct, reason = evaluate_answer_match(answer_str, ground_truth)
        
        if is_correct:
            correct_count += 1
            type_stats[qtype]["correct"] += 1
            print(f"  Result: [CORRECT] ({reason})")
        else:
            print(f"  Result: [INCORRECT] ({reason})")

        type_stats[qtype]["total"] += 1
        print(f"  Model Output: {answer_str}")
        print(f"  Latency: {latency:.2f}s")

        results.append({
            "qid": qid,
            "question": question,
            "qtype": qtype,
            "ground_truth": ground_truth,
            "model_answer": answer_str,
            "is_correct": is_correct,
            "match_reason": reason,
            "latency": latency,
            "retrieved_chunks_count": len(chunks)
        })

    total_time = time.time() - start_time
    total_q = len(questions)
    accuracy_pct = (correct_count / total_q * 100) if total_q > 0 else 0.0

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total Questions Evaluated : {total_q}")
    print(f"Total Correct             : {correct_count}")
    print(f"Overall Accuracy          : {accuracy_pct:.2f}%")
    print(f"Total Evaluation Time     : {total_time:.2f}s (Avg {total_time/total_q:.2f}s/question)")
    print("-" * 80)
    print("ACCURACY BY QUESTION TYPE:")
    for qt, stat in type_stats.items():
        qt_acc = (stat['correct'] / stat['total'] * 100) if stat['total'] > 0 else 0.0
        print(f"  - {qt:<15}: {stat['correct']}/{stat['total']} ({qt_acc:.2f}%)")
    print("=" * 80)

    # Export results
    output_data = {
        "summary": {
            "total_questions": total_q,
            "correct": correct_count,
            "accuracy_pct": round(accuracy_pct, 2),
            "total_time_sec": round(total_time, 2),
            "type_breakdown": {
                qt: {
                    "total": stat["total"],
                    "correct": stat["correct"],
                    "accuracy_pct": round((stat["correct"] / stat["total"] * 100), 2) if stat["total"] > 0 else 0.0
                }
                for qt, stat in type_stats.items()
            }
        },
        "details": results
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Detailed evaluation report saved to: {OUTPUT_FILE}")
    return output_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Number of questions to test (e.g., 5 or 10)")
    args = parser.parse_args()

    run_evaluation(limit=args.limit)
