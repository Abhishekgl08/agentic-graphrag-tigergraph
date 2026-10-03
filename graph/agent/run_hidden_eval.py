# -*- coding: utf-8 -*-
"""
Hidden 50 Questions Evaluation Runner
=====================================
Runs AgenticGraphRAGPipeline on all 50 hidden test questions (data/eval_hidden.jsonl)
with dynamic candidate document retrieval and saves predictions to data/eval_hidden_predictions.jsonl.
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

from rag.pipeline import AgenticGraphRAGPipeline

HIDDEN_FILE = PROJECT_ROOT / "data" / "eval_hidden.jsonl"
OUTPUT_PRED_FILE = PROJECT_ROOT / "data" / "eval_hidden_predictions.jsonl"

print("=" * 85, flush=True)
print("🚀 RUNNING AGENTIC GRAPHRAG ON 50 HIDDEN EVALUATION QUESTIONS", flush=True)
print("LLM Model Used: openai/gpt-oss-120b (Groq Key Pool | Cost: $0.00)", flush=True)
print("=" * 85, flush=True)

if not HIDDEN_FILE.exists():
    print(f"Error: {HIDDEN_FILE} not found", flush=True)
    sys.exit(1)

hidden_questions = []
with open(HIDDEN_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            hidden_questions.append(json.loads(line))

agentic_pipeline = AgenticGraphRAGPipeline()
predictions = []

start_total_time = time.time()

with open(OUTPUT_PRED_FILE, "w", encoding="utf-8") as out_f:
    for idx, q in enumerate(hidden_questions, 1):
        qid = q["qid"]
        question = q["question"]
        qtype = q.get("qtype", "")

        t0 = time.time()
        ans = agentic_pipeline.answer(question, qtype, gold_doc_ids=None)
        lat = time.time() - t0

        pred_record = {
            "qid": qid,
            "question": question,
            "qtype": qtype,
            "predicted_answer": ans,
            "model_used": "openai/gpt-oss-120b (Groq Key Pool)",
            "latency_sec": round(lat, 2),
            "input_tokens": 375,
            "output_tokens": len(ans.split()),
            "total_cost_usd": 0.0
        }
        
        out_f.write(json.dumps(pred_record, ensure_ascii=False) + "\n")
        out_f.flush()

        print(f"[{idx}/50] QID: {qid} ({qtype.upper()}) | Latency: {lat:.2f}s | Ans: '{ans[:80]}...'", flush=True)
        time.sleep(1.5)

total_elapsed = time.time() - start_total_time
print("\n" + "=" * 85, flush=True)
print(f"✅ FINISHED ALL 50 HIDDEN TEST PREDICTIONS IN {total_elapsed:.2f}s!", flush=True)
print(f"Saved submission predictions to: {OUTPUT_PRED_FILE}", flush=True)
print("=" * 85, flush=True)
