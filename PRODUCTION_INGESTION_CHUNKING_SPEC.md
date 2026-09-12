# Agentic GraphRAG — Production Dataset Ingestion & Chunking Specification

## Purpose

This is the implementation specification for converting the current experimental dataset-processing code into a **production-grade FastAPI dataset ingestion and chunking service**.

Treat this document as the source of truth for the ingestion/chunking stage.

**Scope ends at validated retrieval chunks.**

Do NOT implement embeddings, vector indexing, graph extraction, graph loading, agents, or answer generation in this task.

---

## 1. Current Repository

```text
agentic-graphrag-tigergraph/
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── tigergraph.py
├── data/
│   ├── raw/
│   │   └── corpus.jsonl
│   └── processed/
│       ├── chunks.jsonl
│       ├── dataset_profile.json
│       └── retrieval_chunks.jsonl
├── scripts/
│   ├── audit_corpus.py
│   ├── audit_olympic.py
│   ├── audit_parser_content_loss.py
│   ├── audit_parser_loss.py
│   ├── audit_results_content.py
│   ├── diagnose_content_difference.py
│   ├── final_content_integrity.py
│   ├── inspect_dataset.py
│   ├── inspect_doc.py
│   ├── inspect_olympic_candidates.py
│   ├── inspect_parser_document.py
│   ├── investigate_content_mismatches.py
│   ├── process_dataset.py
│   ├── profile_dataset.py
│   ├── smallest_docs.py
│   ├── validate_chunks.py
│   ├── validate_content_integrity.py
│   ├── validate_lossless_content.py
│   └── validate_retrieval_chunks.py
└── tests/
    └── integration/
        ├── test_tigergraph.py
        ├── test_tigergraph_graphs.py
        └── test_tigergraph_schema.py
```

### Authoritative files

The current raw dataset is:

```text
data/raw/corpus.jsonl
```

The authoritative reference implementation for chunking is:

```text
scripts/process_dataset.py
```

The authoritative final integrity validator is:

```text
scripts/final_content_integrity.py
```

Do not redesign the validated chunking algorithm. Refactor it into production modules while preserving its behavior.

---

## 2. Current Dataset Baseline

Current corpus:

```text
Documents: approximately 2,951
Source tokens: approximately 5.47M
```

Current validated retrieval output:

```text
Input documents:       2,951
Output chunks:        18,080

Infobox chunks:        2,934
Text chunks:          11,951
Table chunks:          3,195

Approx chunk tokens: 5,104,367
```

Final integrity test:

```text
Raw documents:    2,951
Processed chunks: 18,080

Documents missing chunks: 0
Unknown chunk documents:  0

Original tokens:      3,932,649
Reconstructed tokens: 3,932,649
Token ratio:          1.000000

Token mismatches:     0

STATUS: PASS — NO TRUE TOKEN LOSS DETECTED
```

These values are the regression baseline.

---

## 3. Raw Dataset Rules

The raw corpus is the source of truth.

The application MUST NOT:

- modify `data/raw/corpus.jsonl`
- crawl URLs contained in records
- download Wikipedia pages
- supplement facts from external sources
- replace corpus facts with current internet facts
- silently discard table content
- silently discard infobox content

The `url` field is metadata only.

Each JSONL record contains fields such as:

```json
{
  "doc_id": "...",
  "title": "...",
  "url": "...",
  "wikidata_qid": "...",
  "wikipedia_pageid": "...",
  "approx_tokens": 1234,
  "text": "..."
}
```

Required:

```text
doc_id
title
text
```

Optional:

```text
url
wikidata_qid
wikipedia_pageid
approx_tokens
```

Malformed JSONL must produce a controlled validation error, not silent skipping.

---

# 4. Target Production Structure

Build toward:

```text
agentic-graphrag-tigergraph/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── models.py
│   ├── exceptions.py
│   ├── logging_config.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   └── datasets.py
│   │   └── schemas/
│   │       ├── __init__.py
│   │       └── datasets.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── dataset_service.py
│   │   ├── chunking_service.py
│   │   └── validation_service.py
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   └── local_storage.py
│   │
│   └── chunking/
│       ├── __init__.py
│       ├── constants.py
│       ├── parser.py
│       ├── splitter.py
│       └── chunker.py
│
├── data/
│   ├── raw/
│   │   └── corpus.jsonl
│   └── processed/
│       └── <dataset_id>/
│           ├── chunks.jsonl
│           ├── metadata.json
│           └── validation.json
│
├── scripts/
│   ├── profile_dataset.py
│   ├── validate_retrieval_chunks.py
│   ├── final_content_integrity.py
│   └── archive/
│
├── tests/
│   ├── unit/
│   │   ├── test_chunking.py
│   │   ├── test_parser.py
│   │   ├── test_validation.py
│   │   ├── test_storage.py
│   │   └── test_dataset_service.py
│   ├── integration/
│   │   ├── test_dataset_api.py
│   │   └── test_chunking_pipeline.py
│   └── existing TigerGraph tests
│
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
├── pytest.ini
└── README.md
```

Do not create unnecessary abstractions just to increase the number of files.

---

# 5. Existing Scripts

These are development/diagnostic tools, not production application code.

Useful operational scripts:

```text
scripts/profile_dataset.py
scripts/validate_retrieval_chunks.py
scripts/final_content_integrity.py
```

Historical/debug scripts may be moved to:

```text
scripts/archive/
```

Do not delete them without approval.

---

# 6. Production Architecture

Do NOT put chunking/business logic inside FastAPI routes.

Required conceptual flow:

```text
HTTP request
    ↓
FastAPI route
    ↓
Dataset service
    ↓
Storage
    ↓
Streaming JSONL reader
    ↓
Chunking service
    ↓
Validation service
    ↓
Atomic processed dataset
```

The API layer should orchestrate the request only.

Business logic must be independently testable.

---

# 7. Chunking Algorithm — MUST PRESERVE

The production implementation must preserve the exact behavior of:

```text
scripts/process_dataset.py
```

Do not invent a new chunking strategy during this refactor.

## Configuration

```text
TARGET_MIN_TOKENS = 400
TARGET_MAX_TOKENS = 800
SMALL_CHUNK_THRESHOLD = 50
```

Current approximate-token calculation:

```python
words = re.findall(r"\S+", text)
approx_tokens = max(1, int(len(words) * 1.3))
```

Do not silently replace this during the refactor.

---

# 8. Infobox Processing

Detect an initial:

```text
[Infobox ...]
```

block.

Store it as a separate chunk:

```text
chunk_type = "infobox"
section = "infobox"
section_path = ["infobox"]
```

Do not merge infobox chunks with prose or tables.

Preserve infobox content.

---

# 9. Section Processing

Recognized standalone section headings become metadata.

Current section vocabulary:

```text
early life
early life and career
early life and education
career
personal life
education
history
background
legacy
reception
critical reception
critical response
accolades
awards
filmography
discography
bibliography
references
see also
external links
plot
cast
production
development
writing
new writers
casting
filming
post-production
soundtrack
release
box office
home media
television premiere
marketing
sequel
sequels
music
impact
franchise
theatrical
re-releases
effects
design
summary
qualification
qualifications
competition format
schedule
records
results
results table
heats
semifinals
quarterfinals
final
finals
qualifying
qualifying round
repechage
reserves
political career
political positions
tenure
committee assignments
operations
products and services
management
financial information
corporate affairs
controversies
```

Headings are represented through:

```text
section
section_path
```

Do not duplicate standalone headings into normal chunk text.

---

# 10. Prose Processing

Normal prose:

```text
section
 ↓
paragraph splitting
 ↓
large-paragraph splitting
 ↓
sentence-aware splitting
 ↓
chunk aggregation
```

Rules:

- Blank lines define paragraphs.
- Paragraphs <= approximately 800 tokens remain intact.
- Oversized paragraphs are split by sentences where possible.
- If sentence splitting cannot produce multiple pieces, split by words.
- Accumulate sentences until adding another would exceed approximately 800 tokens.
- Never discard content.

Use whitespace-preserving joins:

```python
" ".join(...)
"

".join(...)
```

Avoid unsafe concatenation that joins words together.

---

# 11. Table Processing

The corpus contains table-like text using constructs such as:

```text
|
!!
```

Table-like sections must remain:

```text
chunk_type = "table"
```

Do not convert them to prose.

The current strategy is:

```text
table section
 ↓
individual lines/rows
 ↓
accumulate until approximately 800 tokens
```

Join rows using:

```python
"
".join(...)
```

Never discard sparse or unusual table rows.

This is especially important for Olympic result data.

---

# 12. Small Chunk Merging

Chunks below:

```text
50 approximate tokens
```

are candidates for merging.

Rules:

### Never

- merge infoboxes
- merge across documents
- merge different chunk types
- merge a table with prose
- alter genuinely tiny source documents
- exceed approximately 800 tokens

### Prefer

1. Merge forward.
2. If impossible, merge backward.
3. Only same-document/same-type merges.
4. Re-index chunks afterward.

---

# 13. Chunk Schema

Every chunk must contain:

```json
{
  "chunk_id": "...",
  "doc_id": "...",
  "title": "...",
  "wikidata_qid": "...",
  "wikipedia_pageid": "...",
  "chunk_index": 0,
  "chunk_type": "text",
  "section": "introduction",
  "section_path": ["introduction"],
  "approx_tokens": 500,
  "text": "..."
}
```

Allowed:

```text
text
table
infobox
```

Required:

```text
chunk_id
doc_id
title
chunk_index
chunk_type
section
section_path
approx_tokens
text
```

---

# 14. Dataset IDs

Every ingestion request must create a unique server-generated:

```text
dataset_id
```

Use UUID4 or another collision-safe identifier.

Never use an uploaded filename as the dataset ID.

Multiple datasets must be able to coexist.

---

# 15. Storage

Initial implementation may use local filesystem storage.

Use:

```text
data/
├── raw/
└── processed/
    └── <dataset_id>/
        ├── chunks.jsonl
        ├── metadata.json
        └── validation.json
```

Do not hard-code Windows-specific paths.

All paths must come from configuration.

The storage layer should be replaceable later with:

```text
S3
Azure Blob
GCS
```

without rewriting the dataset service.

---

# 16. Streaming

The corpus is millions of tokens.

Do NOT unnecessarily load the complete dataset into memory.

Preferred:

```text
uploaded file
 ↓
stream/read JSONL incrementally
 ↓
validate one record
 ↓
chunk one record
 ↓
write chunks
 ↓
next record
```

Use generators/iterators where appropriate.

The production path must support thousands of documents and tens of thousands of chunks without requiring all records in RAM.

---

# 17. Upload API

Create:

```http
POST /v1/datasets
```

Accept:

```text
multipart/form-data
file=<dataset.jsonl>
```

Validate:

- file exists
- filename safely handled
- extension
- content type when available
- maximum size
- valid JSONL
- required fields
- duplicate document IDs
- non-empty text

Generate `dataset_id` server-side.

Never trust an uploaded path or filename.

---

# 18. Dataset APIs

Implement:

```http
POST /v1/datasets
```

Creates/uploads and processes a dataset according to the selected processing design.

```http
GET /v1/datasets/{dataset_id}
```

Returns metadata/status.

```http
GET /v1/datasets/{dataset_id}/chunks
```

Returns paginated chunk metadata.

```http
GET /v1/datasets/{dataset_id}/chunks/{chunk_id}
```

Returns one chunk.

Also:

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

The API design should allow background/asynchronous processing later. Do not make the business layer depend on FastAPI background tasks.

---

# 19. Dataset Status

Support statuses such as:

```text
uploaded
processing
validating
completed
failed
```

Example completed metadata:

```json
{
  "dataset_id": "...",
  "source_filename": "corpus.jsonl",
  "status": "completed",
  "document_count": 2951,
  "chunk_count": 18080,
  "infobox_chunks": 2934,
  "text_chunks": 11951,
  "table_chunks": 3195,
  "approx_tokens": 5104367
}
```

Errors must be safe and must not expose credentials or secrets.

---

# 20. Atomic Processing

Do not mark a dataset completed while its final output is incomplete.

Preferred approach:

```text
data/processing/<dataset_id>/
    chunks.tmp.jsonl
```

Process and validate.

Only after success:

```text
data/processed/<dataset_id>/
    chunks.jsonl
    metadata.json
    validation.json
```

A failed processing run must not be presented as a completed dataset.

---

# 21. Validation

Dataset-level validation:

- valid JSONL
- required fields
- duplicate `doc_id`
- empty documents
- malformed records

Chunk-level validation:

- unique `chunk_id`
- known `doc_id`
- non-empty text
- valid chunk type
- valid token count
- valid chunk index
- document coverage

Content integrity:

```text
missing true tokens = 0
extra true tokens = 0
missing documents = 0
unknown documents = 0
```

Section headings are metadata and therefore do not need to be duplicated in chunk text.

The production validator must be robust against whitespace/line-boundary differences and must not falsely classify heading removal as content loss.

---

# 22. Regression Testing

Run the new production implementation against the current corpus.

Expected baseline:

```text
2,951 documents
approximately 18,080 chunks

2,934 infobox chunks
11,951 text chunks
3,195 table chunks
```

The exact counts should be compared with the current baseline.

If they differ materially, investigate.

Most importantly:

```text
0 true missing tokens
0 true extra tokens
```

The refactor must not silently change chunk semantics.

---

# 23. Logging

Implement application logging.

At minimum record:

```text
dataset_id
filename
status
document count
chunk count
processing duration
validation result
error information when failed
```

Never log:

```text
TG_SECRET
OPENAI_API_KEY
API tokens
credentials
full environment variables
```

Do not dump entire documents into logs.

---

# 24. Configuration

Centralize configuration using environment variables / `.env`.

At minimum:

```text
APP_NAME
APP_VERSION
ENVIRONMENT

DATA_DIR
RAW_DATA_DIR
PROCESSED_DATA_DIR

MAX_UPLOAD_SIZE_MB

CHUNK_TARGET_MIN_TOKENS
CHUNK_TARGET_MAX_TOKENS
SMALL_CHUNK_THRESHOLD
```

Do not hard-code machine-specific paths.

---

# 25. Chunking Version

Store:

```text
chunking_version
```

in metadata.

Initial version:

```text
1.0
```

If the algorithm changes later, increment the version.

This is required for reproducibility of future embeddings and benchmark results.

---

# 26. Security

Treat uploads as untrusted.

At minimum:

- enforce upload size
- generate server-side dataset IDs
- sanitize filenames
- prevent path traversal
- validate JSONL
- never execute uploaded content
- never follow corpus URLs
- don't expose internal filesystem paths unnecessarily
- don't expose environment variables
- don't expose secrets

---

# 27. Error Handling

Use application/domain exceptions.

Recommended HTTP mappings:

```text
400 — invalid request
404 — dataset not found
409 — dataset conflict
413 — file too large
422 — invalid dataset structure
500 — unexpected internal error
```

Business services should not be tightly coupled to HTTP exceptions.

---

# 28. FastAPI Requirements

`app/main.py` must:

- create the FastAPI application
- register routers
- configure logging
- expose health
- expose application metadata
- register exception handlers

Do not put chunking logic in `main.py`.

FastAPI OpenAPI documentation must work:

```text
/docs
/redoc
/openapi.json
```

Use Pydantic request/response models.

---

# 29. Testing Requirements

Create unit tests for:

```text
token approximation
section detection
infobox splitting
table detection
paragraph splitting
sentence splitting
large paragraph splitting
table chunking
small chunk merging
chunk creation
validation
storage
dataset service
```

Create integration tests for:

```text
dataset upload
successful processing
dataset status
chunk listing
single chunk retrieval
invalid JSONL
missing required fields
duplicate IDs
oversized upload
content integrity
```

Chunking tests must not require TigerGraph or OpenAI.

---

# 30. Existing TigerGraph Code

Existing:

```text
app/tigergraph.py
```

must remain separate.

Do NOT mix dataset ingestion/chunking logic into TigerGraph code.

TigerGraph integration is a later stage.

Do not modify the existing `AgenticGraphRAG` TigerGraph graph during this task.

---

# 31. Dependencies

Keep dependencies minimal for this stage.

Expected:

```text
fastapi
uvicorn
pydantic
pydantic-settings
python-multipart
pytest
httpx
```

Do NOT add ML dependencies yet.

Do NOT implement:

```text
OpenAI embeddings
Gemini embeddings
HuggingFace embeddings
PyTorch
Transformers
Sentence Transformers
```

unless absolutely necessary for tests. Embeddings are explicitly out of scope.

---

# 32. Do NOT Implement Yet

This task ends at validated chunks.

Do NOT implement:

```text
embeddings
embedding APIs
reranking
TigerGraph Vector DB loading
entity extraction
relationship extraction
graph schema
graph loading
MCP
RAG
GraphRAG
Agentic GraphRAG
LLM answer generation
benchmark evaluation
```

Future pipeline:

```text
Dataset API
    ↓
Validated chunks
    ↓
Embedding service
    ↓
TigerGraph Vector DB
    ↓
Entity/relationship extraction
    ↓
TigerGraph graph
    ↓
RAG / GraphRAG
    ↓
Agentic GraphRAG
```

---

# 33. Production Quality Requirements

The coding agent must NOT simply copy the script into a giant FastAPI file.

Instead:

1. Read `scripts/process_dataset.py`.
2. Preserve its behavior.
3. Separate responsibilities into:
   - parser
   - splitter
   - chunker
   - storage
   - validation
   - dataset service
   - API routes
4. Add unit tests.
5. Add API integration tests.
6. Run the complete corpus through the new implementation.
7. Compare output against the validated baseline.
8. Run final content-integrity validation.
9. Only declare the work complete after all acceptance criteria pass.

The implementation must be deterministic.

Given identical input and chunking configuration, chunk contents must be identical.

---

# 34. Acceptance Criteria

The work is complete only when:

## API

```text
FastAPI starts
GET /health works
/docs works
POST /v1/datasets works
GET /v1/datasets/{dataset_id} works
GET /v1/datasets/{dataset_id}/chunks works
GET /v1/datasets/{dataset_id}/chunks/{chunk_id} works
```

## Processing

```text
JSONL validated
documents processed incrementally
infoboxes preserved
tables preserved
prose preserved
section metadata preserved
chunking strategy preserved
small chunk merging preserved
```

## Integrity

```text
0 true missing tokens
0 true extra tokens
0 missing documents
0 unknown documents
```

## Regression

Baseline approximately:

```text
2,951 documents
18,080 chunks
2,934 infobox
11,951 text
3,195 table
```

## Engineering

```text
no business logic in routes
centralized configuration
safe storage
atomic output
structured logging
controlled errors
stream-friendly processing
unit tests
integration tests
```

---

# 35. Final Instruction

This is a **one-time production refactor**.

Do not make unnecessary architectural changes after implementation starts.

Do not change the validated chunking strategy for stylistic reasons.

Do not add embeddings or TigerGraph retrieval prematurely.

First deliver this stable boundary:

```text
                    DATASET
                       |
                       v
                FastAPI ingestion
                       |
                       v
                  validation
                       |
                       v
             validated chunking
                       |
                       v
              integrity validation
                       |
                       v
             persistent chunks
                       |
                       v
                  STABLE API
```

Only after this boundary is stable and tested should the project proceed to embedding-model evaluation.
