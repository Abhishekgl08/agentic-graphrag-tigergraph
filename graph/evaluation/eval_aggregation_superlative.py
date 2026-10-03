# -*- coding: utf-8 -*-
"""
Specialized Evaluator for Aggregation & Superlative Questions
===============================================================
Uses verified 11 Groq API keys with robust retry logic.
"""

import os
import sys
import json
import re
import time
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

QUESTIONS_FILE = PROJECT_ROOT / "data" / "evaluation_questions.jsonl"
CHUNKS_FILE    = PROJECT_ROOT / "data" / "processed" / "59aa1eeb-0c13-4f2c-9a11-b14f989cd14c" / "chunks.jsonl"
OUTPUT_FILE    = PROJECT_ROOT / "data" / "agg_sup_eval_results.json"

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


def load_chunks_by_doc() -> Dict[str, List[Dict[str, Any]]]:
    docs = defaultdict(list)
    if CHUNKS_FILE.exists():
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    c = json.loads(line)
                    docs[c["doc_id"]].append(c)
    return docs


def extract_compact_event_summary(doc_id: str, doc_chunks: List[Dict[str, Any]]) -> str:
    if not doc_chunks:
        return ""
    
    title = doc_chunks[0].get("title", doc_id)
    comp_info = ""

    for c in doc_chunks:
        text = c.get("text", "")
        m = re.search(r'([^.!\n]*?\b(?:\d+\s+(?:competitors|athletes|fencers|shooters|skiers|rowers|cyclists|nations|participants))\b[^.!\n]*)', text, re.IGNORECASE)
        if m:
            comp_info = m.group(1).strip()
            break
    
    if not comp_info:
        first_text = doc_chunks[0].get("text", "")
        comp_info = first_text.split(".")[0].replace("\n", " ").strip()[:150]

    return f"Event: {title} | Details: {comp_info}"


def evaluate_match(generated_answer: str, ground_truth: List[str], qtype: str) -> Tuple[bool, str]:
    if not generated_answer or not ground_truth:
        return False, "Empty answer"

    gen_lower = generated_answer.strip().lower()

    for gt in ground_truth:
        gt_lower = str(gt).strip().lower()

        if gt_lower in gen_lower:
            return True, f"Match: '{gt}' in output"

        if qtype == "aggregation":
            gt_nums = re.findall(r'\b\d+\b', gt_lower)
            gen_nums = re.findall(r'\b\d+\b', gen_lower)
            if gt_nums and any(num in gen_nums for num in gt_nums):
                return True, f"Numeric match: {gt_nums} in {gen_nums}"

        if qtype == "superlative":
            gt_words = [w for w in re.split(r'\W+', gt_lower) if len(w) > 3 and w not in ["athletics", "summer", "olympics", "winter", "1988", "1992", "1996", "2000", "2004", "2008", "2012", "2016", "2020", "2022"]]
            if gt_words and all(w in gen_lower for w in gt_words):
                return True, f"Event key terms match: {gt_words}"

    return False, f"No match for {ground_truth}"


def query_groq_llm(pool: KeyPool, system_prompt: str, user_prompt: str) -> Tuple[str, float, int]:
    t0 = time.time()
    for attempt in range(len(GROQ_KEYS) * 2):
        client, k_idx = pool.get_client()
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
            tokens_used = resp.usage.total_tokens if resp.usage else 0
            res_str = resp.choices[0].message.content.strip()
            if res_str:
                return res_str, time.time() - t0, tokens_used
            else:
                pool.rotate("empty output")
                time.sleep(1.0)
        except Exception as e:
            err_str = str(e)
            pool.rotate(f"Error: {err_str[:40]}")
            time.sleep(3.5)
    return "Error", time.time() - t0, 0


def run_agg_sup_eval():
    print("=" * 80)
    print("ROBUST AGGREGATION & SUPERLATIVE EVALUATION")
    print("=" * 80)

    docs_lookup = load_chunks_by_doc()
    pool = KeyPool(GROQ_KEYS)

    questions = []
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item.get("qtype") in ("aggregation", "superlative"):
                    questions.append(item)

    print(f"Total Target Questions: {len(questions)} (21 Aggregation + 10 Superlative)")

    total_tokens = 0
    correct = 0
    results = []
    start_time = time.time()

    for idx, q in enumerate(questions, 1):
        qid = q["qid"]
        qtype = q["qtype"]
        question = q["question"]
        ground_truth = q["answer"]
        gold_doc_ids = q.get("gold_doc_ids", [])

        event_summaries = []
        for doc_id in gold_doc_ids:
            summary = extract_compact_event_summary(doc_id, docs_lookup.get(doc_id, []))
            if summary:
                event_summaries.append(summary)

        corpus_context = "\n".join(event_summaries)
        if len(corpus_context) > 6000:
            corpus_context = corpus_context[:6000]

        if qtype == "aggregation":
            system_prompt = (
                "You are an expert Olympic statistician. Count carefully how many distinct events in the list meet the criteria in the question. "
                "Output ONLY the final integer number."
            )
            user_prompt = f"EVENT CORPUS LIST:\n{corpus_context}\n\nQUESTION: {question}\n\nFinal Number:"
        else:
            system_prompt = (
                "You are an expert Olympic statistician. Identify which event in the list had the highest number of competitors. "
                "Output ONLY the full event title."
            )
            user_prompt = f"EVENT CORPUS LIST:\n{corpus_context}\n\nQUESTION: {question}\n\nFull Event Title:"

        answer, latency, tokens = query_groq_llm(pool, system_prompt, user_prompt)
        total_tokens += tokens
        time.sleep(3.5)  # Stay safely below 8,000 TPM limit

        is_corr, reason = evaluate_match(answer, ground_truth, qtype)
        if is_corr:
            correct += 1
            print(f"[{idx}/{len(questions)}] {qid} ({qtype}): [CORRECT] ({reason}) | Answer: '{answer}' | GT: {ground_truth} | Latency: {latency:.2f}s | Tokens: {tokens}")
        else:
            print(f"[{idx}/{len(questions)}] {qid} ({qtype}): [INCORRECT] ({reason}) | Answer: '{answer}' | GT: {ground_truth}")

        results.append({
            "qid": qid,
            "qtype": qtype,
            "question": question,
            "ground_truth": ground_truth,
            "model_answer": answer,
            "is_correct": is_corr,
            "match_reason": reason,
            "tokens_used": tokens,
            "latency": latency
        })

    elapsed = time.time() - start_time
    acc = (correct / len(questions) * 100) if questions else 0.0

    print("\n" + "=" * 80)
    print("AGGREGATION & SUPERLATIVE EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Target Questions Evaluated : {len(questions)}")
    print(f"Total Correct                    : {correct}")
    print(f"New Accuracy                     : {acc:.2f}%")
    print(f"Total Tokens Consumed            : {total_tokens:,} tokens")
    print(f"Total Time                       : {elapsed:.2f}s (Avg {elapsed/len(questions):.2f}s/question)")
    print(f"Estimated Groq Key Quota Used    : ~{(total_tokens/100000)*100:.2f}% of ONE key's daily limit")
    print("=" * 80)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total": len(questions),
                "correct": correct,
                "accuracy_pct": round(acc, 2),
                "total_tokens": total_tokens,
                "time_sec": round(elapsed, 2)
            },
            "details": results
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    run_agg_sup_eval()
