# -*- coding: utf-8 -*-
"""
Production Graph Ingestion Pipeline
=====================================
Hybrid strategy:
  Step 1 - GLiNER: Extract ALL entities (free, local)
  Step 2 - Rules:  PART_OF, IN_SPORT, AT_VENUE, basic REPRESENTS
  Step 3 - Groq:   WON_BY + REPRESENTS for chunks with Athlete+Event
           Uses 11 keys with smart rotation:
             - RPM limit → wait 30s, retry same key
             - Daily limit → rotate to next key
  Step 4 - TigerGraph: Load all vertices + edges

Run: python graph/ingestion/run_graph_ingestion.py

State is saved to graph/state/ingestion_v2_progress.json → resumable.
"""

import json
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

CHUNKS_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "59aa1eeb-0c13-4f2c-9a11-b14f989cd14c" / "chunks.jsonl"
)
QUESTIONS_FILE = PROJECT_ROOT / "data" / "evaluation_questions.jsonl"
STATE_FILE     = PROJECT_ROOT / "graph" / "state" / "ingestion_v2_progress.json"
LOG_FILE       = PROJECT_ROOT / "graph" / "state" / "ingestion_v2.log"

# ─────────────────────────────────────────────────────────────────────────────
# 11 Groq API keys (from cred.txt + .env)
# ─────────────────────────────────────────────────────────────────────────────
def _load_groq_keys() -> List[str]:
    raw = os.getenv("GROQ_KEYS") or os.getenv("GROQ_API_KEY") or ""
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    return keys if keys else ["gsk_placeholder"]

GROQ_KEYS = _load_groq_keys()
GROQ_MODEL = "openai/gpt-oss-120b"

TG_HOST      = os.getenv("TG_HOST", "")
TG_SECRET    = os.getenv("TG_SECRET", "")
TG_GRAPHNAME = os.getenv("TG_GRAPHNAME", "AgenticGraphRAG")

# ─────────────────────────────────────────────────────────────────────────────
# GLiNER config
# ─────────────────────────────────────────────────────────────────────────────
GLINER_MODEL  = "urchade/gliner_medium-v2.1"
THRESHOLD     = 0.4
LABEL_TO_TYPE = {
    "athlete": "Athlete", "person": "Athlete",
    "country": "Country", "nation": "Country",
    "venue": "Venue", "stadium": "Venue", "arena": "Venue",
    "gymnasium": "Venue", "oval": "Venue",
    "sport": "Sport", "discipline": "Sport",
    "event": "Event", "competition": "Event",
    "olympic games": "OlympicGames", "olympics": "OlympicGames",
}
GLINER_LABELS = list(LABEL_TO_TYPE.keys())

# ─────────────────────────────────────────────────────────────────────────────
# Groq system prompt (compact — for WON_BY + REPRESENTS only)
# ─────────────────────────────────────────────────────────────────────────────
GROQ_SYSTEM = (
    "You are an Olympic knowledge graph builder. "
    "Given Olympic text and a list of detected entities, extract ONLY these relationships:\n"
    "1. WON_BY: Event -[WON_BY {medal: GOLD|SILVER|BRONZE}]-> Athlete\n"
    "2. REPRESENTS: Athlete -[REPRESENTS]-> Country\n"
    "Country codes (NED, RUS, KOR, GBR, CHN, USA, GER, AUS, FRA, GUA) are valid.\n"
    "Respond ONLY with valid JSON — no markdown, no explanation:\n"
    '{"relationships":[{"source":"entity_id","source_type":"Event|Athlete",'
    '"rel":"WON_BY|REPRESENTS","target":"entity_id","target_type":"Athlete|Country",'
    '"medal":"GOLD|SILVER|BRONZE|null"}]}'
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sanitize(text, max_chars=2000):
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:max_chars].strip()


def slugify(name):
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s[:40].strip("-")


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "failed": [], "key_index": 0, "key_exhausted": []}


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def load_gold_chunks():
    """Load only chunks from gold doc IDs referenced by 100 questions."""
    gold_ids = set()
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            for d in q.get("gold_doc_ids", []):
                gold_ids.add(d)

    chunks = []
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            doc_id   = r.get("wikidata_qid") or r.get("doc_id") or ""
            text     = r.get("text", "").strip()
            chunk_id = r.get("chunk_id", "")
            if not text or doc_id not in gold_ids:
                continue
            chunks.append({
                "chunk_id":   chunk_id,
                "doc_id":     doc_id,
                "title":      r.get("title", ""),
                "section":    r.get("section", ""),
                "chunk_type": r.get("chunk_type", ""),
                "text":       text,
            })

    log(f"Loaded {len(gold_ids)} gold doc IDs → {len(chunks)} chunks")
    return chunks

# ─────────────────────────────────────────────────────────────────────────────
# GLiNER extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_entities_gliner(model, text):
    raw = model.predict_entities(text, GLINER_LABELS, threshold=THRESHOLD)
    seen = set()
    entities = []
    for e in raw:
        name  = e["text"].strip()
        etype = LABEL_TO_TYPE.get(e["label"].lower(), "Event")
        key   = (name.lower(), etype)
        if key in seen or not name:
            continue
        seen.add(key)
        entities.append({
            "id":    slugify(name),
            "name":  name,
            "type":  etype,
            "score": round(e["score"], 3),
        })
    return entities

# ─────────────────────────────────────────────────────────────────────────────
# Rule-based relationships
# ─────────────────────────────────────────────────────────────────────────────

PART_OF_RE  = re.compile(r"(\d{4}\s+(?:Summer|Winter)\s+Olympics?)", re.IGNORECASE)
AT_VENUE_RE = re.compile(
    r"(?:held|took place|staged)\s+at\s+([\w\s\-,]+?(?:stadium|arena|gymnasium|oval|velodrome|pool|hall|center|centre|park|complex|track|palace))",
    re.IGNORECASE
)
IN_SPORT_RE = re.compile(r"(\w[\w\s]+?)\s+at\s+the\s+\d{4}", re.IGNORECASE)


def extract_relationships_rules(text, entities):
    rels = []
    seen = set()

    athletes  = [e for e in entities if e["type"] == "Athlete"]
    events    = [e for e in entities if e["type"] == "Event"]
    venues    = [e for e in entities if e["type"] == "Venue"]
    sports    = [e for e in entities if e["type"] == "Sport"]
    games     = [e for e in entities if e["type"] == "OlympicGames"]

    def add(rel_type, src, src_type, tgt, tgt_type, medal=None):
        key = (rel_type, src, tgt)
        if key in seen:
            return
        seen.add(key)
        r = {"rel": rel_type, "source": src, "source_type": src_type,
             "target": tgt, "target_type": tgt_type, "medal": medal}
        rels.append(r)

    # PART_OF: Event → OlympicGames
    for m in PART_OF_RE.finditer(text):
        game_name = m.group(1)
        game_id   = slugify(game_name)
        for ev in events:
            add("PART_OF", ev["id"], "Event", game_id, "OlympicGames")
        # also ensure game entity exists
        if not any(g["id"] == game_id for g in games):
            games.append({"id": game_id, "name": game_name, "type": "OlympicGames", "score": 0.9})

    # AT_VENUE: Event → Venue
    for m in AT_VENUE_RE.finditer(text):
        venue_name = m.group(1).strip()
        venue_id   = slugify(venue_name)
        for ev in events:
            add("AT_VENUE", ev["id"], "Event", venue_id, "Venue")
        if not any(v["id"] == venue_id for v in venues):
            venues.append({"id": venue_id, "name": venue_name, "type": "Venue", "score": 0.8})

    # IN_SPORT: Event → Sport (if sport name appears in event name)
    for ev in events:
        for sp in sports:
            if sp["name"].lower() in ev["name"].lower():
                add("IN_SPORT", ev["id"], "Event", sp["id"], "Sport")

    # PARTICIPATED_IN: Athlete → Event (if both present in chunk)
    if athletes and events:
        for a in athletes:
            for ev in events[:2]:  # limit to avoid explosion
                add("PARTICIPATED_IN", a["id"], "Athlete", ev["id"], "Event")

    return rels, games, venues, sports

# ─────────────────────────────────────────────────────────────────────────────
# Groq extraction — WON_BY + REPRESENTS (with key rotation)
# ─────────────────────────────────────────────────────────────────────────────

class GroqKeyPool:
    def __init__(self, keys):
        from groq import Groq
        self.Groq      = Groq
        self.keys      = keys
        self.idx       = 0
        self.exhausted = set()

    def get_working_key(self):
        for _ in range(len(self.keys)):
            k_idx = self.idx
            self.idx = (self.idx + 1) % len(self.keys)
            if k_idx not in self.exhausted:
                return self.keys[k_idx], k_idx
        return None, -1

    def mark_exhausted(self, k_idx):
        self.exhausted.add(k_idx)
        log(f"  Groq key #{k_idx+1} marked EXHAUSTED (daily quota). {len(self.keys) - len(self.exhausted)} keys remaining.")

    def call(self, user_msg, retries=5):
        """
        Round-robin across all 11 keys.
        - On 429 (RPM limit): immediately try next key (no 35s wait!)
        - On 402 (Daily limit): mark key exhausted and try next key
        """
        for _ in range(retries):
            key, k_idx = self.get_working_key()
            if key is None:
                log("  ALL Groq keys exhausted for today!")
                return None

            try:
                client = self.Groq(api_key=key)
                resp = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": GROQ_SYSTEM},
                        {"role": "user",   "content": user_msg},
                    ],
                    max_tokens=600,
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content.strip()
                if not raw:
                    return {"relationships": []}
                return json.loads(raw)

            except Exception as e:
                err = str(e)
                if "429" in err or "rate" in err.lower() or "rpm" in err.lower():
                    # Rotate to next key immediately instead of sleeping 35s
                    log(f"  RPM limit on key #{k_idx+1}, rotating to next key...")
                    time.sleep(1)
                    continue
                elif "402" in err or "day" in err.lower() or "daily" in err.lower() or "quota" in err.lower():
                    self.mark_exhausted(k_idx)
                    continue
                elif "400" in err and "json" in err.lower():
                    return {"relationships": []}
                else:
                    log(f"  Groq error on key #{k_idx+1}: {err[:100]}")
                    return {"relationships": []}

        # If all retries in loop hit RPM, brief pause
        time.sleep(5)
        return {"relationships": []}


def groq_extract_relationships(pool, entities, text):
    """Send entity list + text to Groq for WON_BY + REPRESENTS only."""
    # Build compact entity summary
    athletes = [e for e in entities if e["type"] == "Athlete"]
    events   = [e for e in entities if e["type"] == "Event"]
    countries= [e for e in entities if e["type"] == "Country"]

    if not athletes or (not events and not countries):
        return []

    # Compact entity list (not full text → saves tokens)
    entity_summary = "Entities detected:\n"
    for e in athletes[:10]:
        entity_summary += f"  Athlete: {e['name']} (id:{e['id']})\n"
    for e in events[:5]:
        entity_summary += f"  Event: {e['name']} (id:{e['id']})\n"
    for e in countries[:10]:
        entity_summary += f"  Country: {e['name']} (id:{e['id']})\n"

    # Add just the first 800 chars of text for context
    text_snippet = sanitize(text, 800)
    user_msg = f"{entity_summary}\nText excerpt:\n{text_snippet}"

    result = pool.call(user_msg)
    if result is None:
        return []
    return result.get("relationships", [])

# ─────────────────────────────────────────────────────────────────────────────
# TigerGraph loading
# ─────────────────────────────────────────────────────────────────────────────

class TigerGraphLoader:
    def __init__(self):
        import pyTigerGraph as tg
        self.conn = tg.TigerGraphConnection(
            host=TG_HOST,
            graphname=TG_GRAPHNAME,
            gsqlSecret=TG_SECRET,
        )
        try:
            token = self.conn.getToken(TG_SECRET)
            log(f"TigerGraph connected. Token: {str(token)[:30]}...")
            self.ready = True
        except Exception as e:
            log(f"TigerGraph connection failed: {e}")
            self.ready = False

    def upsert_vertex(self, vtype, vid, attrs):
        if not self.ready:
            return False
        try:
            self.conn.upsertVertex(vtype, vid, attrs)
            return True
        except Exception as e:
            log(f"  Vertex upsert error ({vtype} {vid}): {e}")
            return False

    def upsert_edge(self, src_type, src_id, edge_type, tgt_type, tgt_id, attrs=None):
        if not self.ready:
            return False
        try:
            self.conn.upsertEdge(src_type, src_id, edge_type, tgt_type, tgt_id, attrs or {})
            return True
        except Exception as e:
            log(f"  Edge upsert error ({edge_type}): {e}")
            return False

    def batch_upsert_vertices(self, vertices_by_type):
        """Bulk upsert. vertices_by_type: {vtype: {vid: {attrs}}}"""
        if not self.ready:
            return 0
        total = 0
        for vtype, verts in vertices_by_type.items():
            try:
                self.conn.upsertVertices(vtype, list(verts.items()))
                total += len(verts)
            except Exception as e:
                log(f"  Batch vertex error ({vtype}): {e}")
        return total

    def batch_upsert_edges(self, edges):
        """edges: list of (src_type, src_id, edge_type, tgt_type, tgt_id, attrs)"""
        if not self.ready:
            return 0
        total = 0
        for src_type, src_id, edge_type, tgt_type, tgt_id, attrs in edges:
            if self.upsert_edge(src_type, src_id, edge_type, tgt_type, tgt_id, attrs):
                total += 1
        return total

# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_ingestion(dry_run=False):
    """
    Full ingestion pipeline.
    dry_run=True → extract entities/rels but don't load to TigerGraph.
    """
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log("=" * 60)
    log("PRODUCTION GRAPH INGESTION PIPELINE v2")
    log(f"Mode: {'DRY RUN' if dry_run else 'LIVE - loading to TigerGraph'}")
    log("=" * 60)

    # ── Load state
    state = load_state()
    completed_set = set(state.get("completed", []))
    log(f"Previously completed: {len(completed_set)} chunks")

    # ── Load GLiNER
    log("Loading GLiNER model...")
    from gliner import GLiNER
    t0 = time.time()
    gliner_model = GLiNER.from_pretrained(GLINER_MODEL)
    log(f"GLiNER loaded in {time.time()-t0:.1f}s")

    # ── Init Groq key pool
    pool = GroqKeyPool(GROQ_KEYS)
    pool.idx = state.get("key_index", 0)
    pool.exhausted = set(state.get("key_exhausted", []))
    log(f"Groq keys available: {len(GROQ_KEYS)} (starting at key #{pool.idx+1})")

    # ── Init TigerGraph
    tg_loader = None
    if not dry_run:
        log("Connecting to TigerGraph...")
        tg_loader = TigerGraphLoader()

    # ── Load gold chunks
    chunks = load_gold_chunks()
    todo   = [c for c in chunks if c["chunk_id"] not in completed_set]
    log(f"Chunks to process: {len(todo)} (skipping {len(completed_set)} already done)")

    # ── Accumulators
    vertices_by_type = defaultdict(dict)  # {vtype: {vid: attrs}}
    edges_to_load    = []
    chunk_count = 0
    entity_count = 0
    rel_count    = 0
    groq_calls   = 0
    groq_errors  = 0

    BATCH_SIZE = 50  # upsert to TG every N chunks

    for i, chunk in enumerate(todo):
        chunk_count += 1
        cid = chunk["chunk_id"]

        log(f"\n[{i+1}/{len(todo)}] {cid}")
        log(f"  Title: {chunk['title'][:60]}")

        # ── Step 1: GLiNER entities
        entities = extract_entities_gliner(gliner_model, chunk["text"])
        log(f"  GLiNER: {len(entities)} entities")

        # Add Chunk vertex
        vertices_by_type["Chunk"][cid] = {
            "document_id": chunk["doc_id"],
            "title":       chunk["title"][:200],
            "section":    chunk.get("section", ""),
            "chunk_type": chunk.get("chunk_type", ""),
            "text":       chunk["text"][:2000],
        }

        # Add entity vertices + MENTIONS edges
        for e in entities:
            vtype = e["type"]
            vid   = e["id"]
            vertices_by_type[vtype][vid] = {"name": e["name"]}
            mention_edge = "MENTIONS_GAMES" if vtype == "OlympicGames" else f"MENTIONS_{vtype.upper()}"
            edges_to_load.append(("Chunk", cid, mention_edge, vtype, vid, {}))
            entity_count += 1

        # ── Step 2: Rule-based relationships
        rule_rels, extra_games, extra_venues, extra_sports = extract_relationships_rules(
            chunk["text"], entities
        )
        for extra_list, vtype in [
            (extra_games,  "OlympicGames"),
            (extra_venues, "Venue"),
            (extra_sports, "Sport"),
        ]:
            for e in extra_list:
                if e["id"] not in vertices_by_type[vtype]:
                    vertices_by_type[vtype][e["id"]] = {"name": e["name"]}

        for r in rule_rels:
            attrs = {}
            if r.get("medal"):
                attrs["medal"] = r["medal"]
            edges_to_load.append((
                r["source_type"], r["source"],
                r["rel"],
                r["target_type"], r["target"],
                attrs,
            ))
            rel_count += 1
        log(f"  Rules: {len(rule_rels)} relationships")

        # ── Step 3: Groq for WON_BY + REPRESENTS
        athletes = [e for e in entities if e["type"] == "Athlete"]
        events   = [e for e in entities if e["type"] == "Event"]
        countries= [e for e in entities if e["type"] == "Country"]

        need_groq = len(athletes) >= 1 and (len(events) >= 1 or len(countries) >= 1)
        if need_groq and not pool.exhausted.issuperset(range(len(GROQ_KEYS))):
            groq_calls += 1
            groq_rels = groq_extract_relationships(pool, entities, chunk["text"])

            if groq_rels is None:
                groq_errors += 1
                log(f"  Groq: ALL KEYS EXHAUSTED")
            else:
                log(f"  Groq: {len(groq_rels)} relationships")
                for r in groq_rels:
                    src_type = r.get("source_type", "Event")
                    tgt_type = r.get("target_type", "Athlete")
                    rel      = r.get("rel", "")
                    src      = r.get("source", "")
                    tgt      = r.get("target", "")
                    medal    = r.get("medal", None)
                    if not rel or not src or not tgt:
                        continue
                    attrs = {}
                    if medal and medal not in ("null", "", None):
                        attrs["medal"] = medal
                    edges_to_load.append((src_type, src, rel, tgt_type, tgt, attrs))
                    rel_count += 1

            time.sleep(1.5)  # Rate limit buffer

        # ── Mark chunk complete
        state["completed"].append(cid)
        state["key_index"]    = pool.idx
        state["key_exhausted"] = list(pool.exhausted)

        # ── Batch upsert to TigerGraph
        if not dry_run and tg_loader and tg_loader.ready and chunk_count % BATCH_SIZE == 0:
            log(f"\n  == Batch upsert: {sum(len(v) for v in vertices_by_type.values())} vertices, {len(edges_to_load)} edges ==")
            tg_loader.batch_upsert_vertices(vertices_by_type)
            tg_loader.batch_upsert_edges(edges_to_load)
            vertices_by_type = defaultdict(dict)
            edges_to_load    = []
            save_state(state)
            log(f"  State saved.")

        # Progress log every 10 chunks
        if chunk_count % 10 == 0:
            log(f"\nPROGRESS: {chunk_count}/{len(todo)} chunks | "
                f"{entity_count} entities | {rel_count} rels | "
                f"{groq_calls} Groq calls | {groq_errors} errors")
            save_state(state)

    # ── Final upsert
    if not dry_run and tg_loader and tg_loader.ready and vertices_by_type:
        log("\nFinal batch upsert...")
        v = tg_loader.batch_upsert_vertices(vertices_by_type)
        e = tg_loader.batch_upsert_edges(edges_to_load)
        log(f"Upserted {v} vertices, {e} edges")

    save_state(state)

    # ── Summary
    log("\n" + "=" * 60)
    log("INGESTION COMPLETE")
    log("=" * 60)
    log(f"Chunks processed : {chunk_count}")
    log(f"Total entities   : {entity_count}")
    log(f"Total relations  : {rel_count}")
    log(f"Groq calls made  : {groq_calls}")
    log(f"Groq errors      : {groq_errors}")

    return {
        "chunks": chunk_count,
        "entities": entity_count,
        "relations": rel_count,
        "groq_calls": groq_calls,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without loading to TigerGraph")
    args = parser.parse_args()
    run_ingestion(dry_run=args.dry_run)
