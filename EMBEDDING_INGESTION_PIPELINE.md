# Embedding + TigerGraph Ingestion Pipeline

## Production-ready implementation plan for Agentic GraphRAG

**Project:** `agentic-graphrag-tigergraph`\
**Stage:** Chunk → Embedding → TigerGraph ingestion\
**Embedding model:** `text-embedding-3-small`\
**Embedding dimension:** `1536`\
**Vector metric:** `COSINE`\
**Vector index:** `HNSW`\
**Graph database:** TigerGraph 4.2.x / current workspace 4.2.5

------------------------------------------------------------------------

# 1. Purpose of this stage

The previous stage established that:

1.  The corpus has already been processed into chunks.
2.  Each chunk has useful metadata.
3.  OpenAI embeddings work correctly.
4.  TigerGraph can store the 1536-dimensional embedding as a native
    VECTOR attribute.
5.  TigerGraph's native `vectorSearch()` works against those stored
    vectors.

The goal of this stage is to turn those experiments into a reusable
ingestion pipeline:

``` text
Processed chunks
      ↓
Validate chunk
      ↓
Create embedding
      ↓
Build graph record
      ↓
Upsert into TigerGraph
      ↓
Track ingestion status
      ↓
Ready for GraphRAG retrieval
```

The implementation must be designed so that the later GraphRAG and
Agentic RAG layers can be added without replacing the ingestion
architecture.

------------------------------------------------------------------------

# 2. Existing files and experiments that must be reused

Do not duplicate the experiments. The production implementation should
build on the existing files.

## 2.1 TigerGraph connection

### File

``` text
app/tigergraph.py
```

This file contains the existing `create_connection()` function.

It loads `.env` from the project root and creates:

``` python
tg.TigerGraphConnection(
    host=host,
    graphname=graphname,
    gsqlSecret=secret,
)
```

All application code that needs TigerGraph should reuse this connection
module.

Do **not** create separate TigerGraph connection logic inside every
endpoint or service.

------------------------------------------------------------------------

## 2.2 TigerGraph schema creation

### File

``` text
scripts/setup_tigergraph_schema.py
```

This file was used to create the initial `Chunk` vertex and then add the
vector attribute.

The current tested vector configuration is:

``` text
Chunk.embedding
    dimension = 1536
    metric = COSINE
    index type = HNSW
    data type = FLOAT
```

The schema was successfully created in:

``` text
AgenticGraphRAG
```

The final production schema should preserve this vector configuration.

------------------------------------------------------------------------

## 2.3 OpenAI API test

### File

``` text
app/test_openai.py
```

This established that the OpenAI API key is being loaded correctly and
that an OpenAI request succeeds.

The API key must continue to come from:

``` text
.env
```

and must never be hard-coded.

------------------------------------------------------------------------

## 2.4 Embedding experiment

The embedding experiment successfully used:

``` text
text-embedding-3-small
```

and verified:

``` text
1536 dimensions
```

This is the exact embedding configuration that should be used
consistently throughout the project.

Do not mix embedding models inside the same vector index.

If the model is changed in the future, treat that as a new embedding
version and plan for re-embedding.

------------------------------------------------------------------------

## 2.5 First five chunk experiment

### File

``` text
scripts/add_first_5_chunks.py
```

This is the most important prototype for this stage.

It already demonstrated:

``` text
chunks.jsonl
      ↓
read first 5 chunks
      ↓
batch OpenAI embedding call
      ↓
1536-dimensional vectors
      ↓
TigerGraph upsertVertex()
      ↓
Chunk vertices stored successfully
```

The production ingestion implementation should extract the reusable
logic from this experiment rather than copy/paste the entire script.

------------------------------------------------------------------------

## 2.6 Vector search experiment

### File

``` text
scripts/setup_vector_search.py
```

This created and installed the TigerGraph query:

``` text
vector_search
```

The query uses:

``` gsql
vectorSearch(
    {Chunk.embedding},
    query_vector,
    5,
    {distance_map: @@distances}
)
```

This proves that the stored vectors can later be retrieved semantically.

------------------------------------------------------------------------

## 2.7 Vector search test

### File

``` text
scripts/test_vector_search.py
```

This successfully performed:

``` text
User query
    ↓
OpenAI embedding
    ↓
1536-dimensional query vector
    ↓
TigerGraph vectorSearch()
    ↓
similar Chunk vertices
```

This file should remain a verification/regression test.

The production ingestion endpoint should not directly depend on this
test script.

------------------------------------------------------------------------

# 3. Current chunk source

The current processed corpus is available at:

``` text
data/processed/59aa1eeb-0c13-4f2c-9a11-b14f989cd14c/chunks.jsonl
```

The chunks contain fields such as:

``` json
{
  "chunk_id": "Q303623_infobox",
  "doc_id": "Q303623",
  "title": "Canoeing at the 2012 Summer Olympics – Men's K-2 1000 metres",
  "wikidata_qid": "Q303623",
  "wikipedia_pageid": 35771859,
  "chunk_index": 0,
  "chunk_type": "infobox",
  "section": "infobox",
  "section_path": ["infobox"],
  "approx_tokens": 63,
  "text": "[Infobox Olympic event]..."
}
```

Important:

**Do not throw away these metadata fields.**

They will become useful for:

-   GraphRAG traversal
-   source attribution
-   citations
-   filtering
-   document-level retrieval
-   section-aware retrieval
-   agent reasoning
-   debugging
-   evaluation
-   re-indexing
-   provenance

------------------------------------------------------------------------

# 4. Important schema decision before full ingestion

The initial experiment used a simplified TigerGraph vertex:

``` text
Chunk
 ├── chunk_id
 ├── text
 ├── document_id
 ├── page_number
 └── embedding
```

That was sufficient for proving vector search.

It is **not sufficient as the final GraphRAG schema** because the actual
corpus contains richer metadata.

Before ingesting the entire corpus, the schema should be expanded so the
information is not lost.

A better long-term model is:

``` text
Document
   │
   │ HAS_CHUNK
   ▼
Chunk
   │
   ├── embedding
   ├── text
   ├── section
   ├── chunk_index
   ├── chunk_type
   ├── metadata
   │
   └── MENTIONS
          │
          ▼
        Entity
```

Later, additional relationships can be introduced without changing the
embedding pipeline.

------------------------------------------------------------------------

# 5. Recommended production graph model

The long-term graph should conceptually contain:

``` text
Document
Chunk
Entity
```

with relationships such as:

``` text
Document ──HAS_CHUNK──> Chunk

Chunk ──MENTIONS──> Entity

Chunk ──NEXT──> Chunk
```

Potential future relationships:

``` text
Entity ──RELATED_TO──> Entity
Document ──RELATED_TO──> Document
Chunk ──DERIVED_FROM──> Chunk
```

Do not create every possible relationship immediately.

The important point is that the ingestion pipeline must be able to add
these relationships later.

------------------------------------------------------------------------

# 6. Recommended Chunk vertex attributes

The final `Chunk` vertex should preserve the source information.

Recommended conceptual structure:

``` text
Chunk
├── PRIMARY_ID chunk_id STRING
├── text STRING
├── document_id STRING
├── title STRING
├── wikidata_qid STRING
├── wikipedia_pageid INT
├── chunk_index INT
├── chunk_type STRING
├── section STRING
├── section_path STRING
├── approx_tokens INT
├── embedding VECTOR(1536)
├── embedding_model STRING
├── embedding_version STRING
└── content_hash STRING
```

The exact TigerGraph syntax should be validated against the current
TigerGraph version before applying the final schema change.

### Why keep `embedding_model`?

Because later you may change:

``` text
text-embedding-3-small
```

to another embedding model.

You must know which vectors were generated using which model.

### Why keep `embedding_version`?

It allows controlled re-indexing.

For example:

``` text
embedding_version = v1
```

Later:

``` text
embedding_version = v2
```

This is much safer than silently replacing vectors without knowing which
version is stored.

### Why keep `content_hash`?

It allows idempotency.

If the same chunk is processed twice:

``` text
chunk text
    ↓
SHA-256
    ↓
content_hash
```

the ingestion service can recognize that the content has not changed.

------------------------------------------------------------------------

# 7. Important design: raw text vs normalized text

The current experiment sends the chunk's `text` to the embedding model
as-is.

For example:

``` text
[Infobox Olympic event]
event: Men's canoe sprint K-2 1,000 metres
games: 2012 Summer
venue: Eton Dorney
...
```

The existing experiment established that this works.

Therefore:

**Do not introduce aggressive text cleaning at this stage.**

In particular, do not blindly replace:

``` text
\n
```

with:

``` text
|
```

The original text should remain available for source fidelity.

If normalization is introduced later, use separate concepts:

``` text
raw_text
normalized_text
embedding_text
```

This prevents the source data from being destroyed merely because an
embedding preprocessing experiment changed.

For the first production version:

``` text
embedding_text = text
```

------------------------------------------------------------------------

# 8. Embedding service architecture

Do not put the OpenAI embedding call directly inside the FastAPI route.

Bad architecture:

``` text
POST /datasets
      ↓
FastAPI route
      ↓
OpenAI
      ↓
TigerGraph
      ↓
response
```

This will become problematic for a large corpus.

Instead:

``` text
API
 │
 ▼
Ingestion service
 │
 ▼
Embedding service
 │
 ▼
TigerGraph repository
```

Recommended modules:

``` text
app/
├── api/
│   └── datasets.py
│
├── services/
│   ├── ingestion_service.py
│   └── embedding_service.py
│
├── repositories/
│   └── tigergraph_repository.py
│
├── models/
│   └── chunk.py
│
└── tigergraph.py
```

The exact folders can be adjusted to the existing project structure, but
the responsibilities should remain separated.

------------------------------------------------------------------------

# 9. Embedding service responsibility

Create a reusable embedding service.

Conceptually:

``` python
class EmbeddingService:

    def embed_texts(self, texts):
        ...
```

It should:

1.  Validate input.
2.  Preserve input order.
3.  Send multiple texts in one embedding request where practical.
4.  Return one vector per text.
5.  Verify vector dimensions.
6.  Handle API errors.
7.  Support retry/backoff.
8.  Avoid logging API keys.
9.  Be independent of TigerGraph.

The service should know about:

``` text
OpenAI
embedding model
embedding dimension
```

but it should **not** know about HTTP endpoints.

------------------------------------------------------------------------

# 10. Why batching is mandatory

Do not do this:

``` text
chunk 1 → OpenAI
chunk 2 → OpenAI
chunk 3 → OpenAI
chunk 4 → OpenAI
...
```

for thousands of chunks.

Instead:

``` text
100 chunks
     ↓
one embedding request
     ↓
100 vectors
```

Then:

``` text
next 100 chunks
     ↓
one embedding request
```

The exact batch size should be configurable and tested against
token/request limits.

For example:

``` text
EMBEDDING_BATCH_SIZE=100
```

should be configuration, not hard-coded business logic.

------------------------------------------------------------------------

# 11. Preserve ordering

Suppose the batch contains:

``` text
chunk A
chunk B
chunk C
```

The returned vectors must be mapped back to:

``` text
chunk A → vector A
chunk B → vector B
chunk C → vector C
```

Never assume an unordered mapping.

Use the input index or another deterministic mapping mechanism.

------------------------------------------------------------------------

# 12. TigerGraph repository

Do not let the embedding service call:

``` python
tg.upsertVertex(...)
```

directly.

Instead create a repository layer.

Conceptually:

``` python
class TigerGraphRepository:

    def upsert_chunk(self, chunk, embedding):
        ...

    def upsert_chunks(self, chunks):
        ...

    def get_chunk(self, chunk_id):
        ...

    def list_chunks(self, dataset_id, offset, limit):
        ...
```

This gives the project a clean separation:

``` text
EmbeddingService
      │
      ▼
IngestionService
      │
      ▼
TigerGraphRepository
      │
      ▼
TigerGraph
```

If the database implementation changes later, the embedding logic does
not need to be rewritten.

------------------------------------------------------------------------

# 13. Do not store only the vector

The TigerGraph record should contain both:

``` text
metadata
+
text
+
embedding
```

The vector is used for retrieval.

The text and metadata are required to understand the retrieved result.

For example:

``` text
Chunk
├── text
├── title
├── section
├── chunk_index
├── document_id
└── embedding
```

When vector search returns a chunk, the GraphRAG system needs the actual
chunk text and provenance.

------------------------------------------------------------------------

# 14. Idempotent ingestion

This is one of the most important production requirements.

Suppose the pipeline processes:

``` text
10,000 chunks
```

and crashes after:

``` text
7,200 chunks
```

The next run should **not create duplicates**.

The primary key:

``` text
chunk_id
```

should be deterministic.

TigerGraph upsert semantics should be used so that:

``` text
same chunk_id
```

updates the existing record rather than creating another vertex.

The pipeline should therefore be safe to run multiple times.

------------------------------------------------------------------------

# 15. Content hash

Calculate a deterministic hash from the source text.

Conceptually:

``` python
content_hash = sha256(text.encode("utf-8")).hexdigest()
```

Then the system can distinguish:

``` text
same chunk + same content
```

from:

``` text
same chunk_id + changed content
```

Example:

``` text
chunk_id = Q303623_chunk_0001
content_hash = abc123...
```

If the content has not changed:

``` text
skip unnecessary re-embedding
```

If the content has changed:

``` text
generate new embedding
update TigerGraph
```

This becomes especially useful for incremental ingestion later.

------------------------------------------------------------------------

# 16. Dataset-level isolation

The API has:

``` text
dataset_id
```

and the chunk data has:

``` text
doc_id
```

Do not confuse these.

Conceptually:

``` text
dataset_id
    ↓
corpus/import job

doc_id
    ↓
source document

chunk_id
    ↓
individual chunk
```

The ingestion pipeline should retain all three concepts.

For example:

``` text
dataset_id = 59aa...
doc_id = Q303623
chunk_id = Q303623_chunk_0001
```

This allows future queries such as:

``` text
search only within dataset X
```

or:

``` text
show all chunks belonging to document Y
```

------------------------------------------------------------------------

# 17. Recommended API

The existing:

``` text
POST /v1/datasets
```

should remain the user-facing dataset upload endpoint.

It should eventually trigger the ingestion workflow.

For the chunk/embedding stage, introduce a clear internal/service-level
operation.

Recommended conceptual endpoint:

``` text
POST /v1/datasets/{dataset_id}/ingest
```

Purpose:

``` text
Start processing the prepared chunks for this dataset.
```

Request:

``` json
{
  "source": "data/processed/.../chunks.jsonl"
}
```

The exact request body can instead use a server-side dataset location;
do not expose arbitrary local filesystem paths in a production API.

The endpoint should return quickly:

``` json
{
  "dataset_id": "59aa...",
  "job_id": "job-123",
  "status": "queued"
}
```

------------------------------------------------------------------------

# 18. Why asynchronous ingestion is better

A corpus may contain:

``` text
100 PDFs
10,000 chunks
100,000 chunks
```

Embedding all of them may take significant time.

Therefore:

``` text
POST /v1/datasets/{dataset_id}/ingest
```

should ideally do:

``` text
request
  ↓
create job
  ↓
return job_id
```

rather than keeping the HTTP connection open.

Then:

``` text
worker
  ↓
read chunks
  ↓
batch embeddings
  ↓
TigerGraph upserts
  ↓
update progress
```

The existing dataset endpoint:

``` text
GET /v1/datasets/{dataset_id}
```

can expose the status.

------------------------------------------------------------------------

# 19. Recommended ingestion states

Use explicit states such as:

``` text
created
queued
processing
completed
failed
partial
```

Example:

``` json
{
  "dataset_id": "59aa...",
  "status": "processing",
  "total_chunks": 10000,
  "processed_chunks": 6200,
  "embedded_chunks": 6200,
  "stored_chunks": 6200
}
```

This is much better than a simple boolean:

``` text
done = true
```

------------------------------------------------------------------------

# 20. Failure handling

The pipeline must assume external services can fail.

Possible failures:

``` text
OpenAI timeout
OpenAI rate limit
OpenAI server error
TigerGraph timeout
TigerGraph authentication failure
malformed chunk
missing text
invalid embedding dimension
network failure
process crash
```

Use retries for transient failures.

Do not retry forever.

Conceptually:

``` text
attempt 1
   ↓ fail
wait
   ↓
attempt 2
   ↓ fail
wait longer
   ↓
attempt 3
   ↓
mark failed
```

Use exponential backoff with a maximum retry count.

------------------------------------------------------------------------

# 21. Partial failure strategy

Suppose a batch has:

``` text
100 chunks
```

and TigerGraph fails.

Do not mark the entire dataset as permanently failed without recording
progress.

Maintain progress such as:

``` text
total = 10000
processed = 7200
failed = 100
```

The ingestion job should be restartable.

The goal is:

``` text
restart
   ↓
skip already successfully ingested chunks
   ↓
retry remaining chunks
```

This is another reason deterministic IDs and content hashes are
important.

------------------------------------------------------------------------

# 22. Logging

Log operational information such as:

``` text
dataset_id
job_id
batch number
batch size
processed count
failed count
embedding model
elapsed time
```

Do **not** log:

``` text
OPENAI_API_KEY
TG_SECRET
full authorization headers
```

Avoid dumping full embeddings into normal logs.

A 1536-number vector does not belong in application logs.

------------------------------------------------------------------------

# 23. Configuration

Use environment variables for deployment-sensitive configuration.

Example:

``` text
OPENAI_API_KEY=...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
EMBEDDING_BATCH_SIZE=100

TG_HOST=...
TG_GRAPHNAME=AgenticGraphRAG
TG_SECRET=...
```

Do not hard-code these values inside the endpoint.

The default model can be:

``` text
text-embedding-3-small
```

but the model should still be configuration-driven.

------------------------------------------------------------------------

# 24. Embedding dimension validation

Every generated vector must be checked.

Expected:

``` text
1536
```

If the service returns:

``` text
1536
```

continue.

If it returns:

``` text
1024
```

or another dimension:

``` text
fail the batch
```

Do not attempt to insert a vector of the wrong dimension into
TigerGraph.

This prevents silent corruption and makes model changes explicit.

------------------------------------------------------------------------

# 25. Embedding model versioning

Store:

``` text
embedding_model
embedding_version
```

Example:

``` text
embedding_model = text-embedding-3-small
embedding_version = v1
```

Later, if the embedding strategy changes:

``` text
embedding_model = another-model
embedding_version = v2
```

The retrieval layer can then know which vector representation it is
using.

------------------------------------------------------------------------

# 26. Vector search compatibility

The ingestion pipeline must preserve compatibility with the
already-tested query:

``` text
vector_search
```

which uses:

``` gsql
vectorSearch(
    {Chunk.embedding},
    query_vector,
    5,
    {distance_map: @@distances}
)
```

The query vector must be generated using the **same embedding model and
dimension** as the stored chunk vectors.

Therefore:

``` text
stored chunks:
text-embedding-3-small / 1536

query:
text-embedding-3-small / 1536
```

Do not mix embedding spaces.

------------------------------------------------------------------------

# 27. Future GraphRAG retrieval

The ingestion pipeline should produce data suitable for two retrieval
mechanisms.

## Vector retrieval

``` text
query
 ↓
embedding
 ↓
TigerGraph vectorSearch
 ↓
top-k chunks
```

## Graph retrieval

``` text
query
 ↓
identify entities/documents
 ↓
traverse graph
 ↓
related chunks/entities/documents
```

Then the eventual GraphRAG system can combine them:

``` text
                 Query
                   │
          ┌────────┴────────┐
          ▼                 ▼
    Vector Search      Graph Search
          │                 │
          └────────┬────────┘
                   ▼
             Ranked context
                   │
                   ▼
                  LLM
```

The current ingestion design must therefore preserve metadata and
relationships rather than treating the vector database as the entire
system.

------------------------------------------------------------------------

# 28. Future Agentic RAG

Later, agents may perform multiple retrieval actions:

``` text
User question
      ↓
Planner Agent
      ↓
┌───────────────┬────────────────┐
│               │                │
▼               ▼                ▼
Vector search   Graph traversal  Metadata filter
│               │                │
└───────────────┴────────────────┘
                ↓
        Evidence aggregation
                ↓
        Reranking / validation
                ↓
          Answer generation
```

The ingestion layer should remain independent of those agents.

That means:

``` text
ingestion ≠ retrieval ≠ reasoning
```

Keep these responsibilities separate.

------------------------------------------------------------------------

# 29. Recommended project structure

A production-oriented structure can evolve toward:

``` text
agentic-graphrag-tigergraph/
│
├── app/
│   ├── main.py
│   ├── tigergraph.py
│   │
│   ├── api/
│   │   ├── datasets.py
│   │   └── health.py
│   │
│   ├── services/
│   │   ├── ingestion_service.py
│   │   ├── embedding_service.py
│   │   └── dataset_service.py
│   │
│   ├── repositories/
│   │   └── tigergraph_repository.py
│   │
│   ├── models/
│   │   ├── chunk.py
│   │   ├── dataset.py
│   │   └── job.py
│   │
│   └── config.py
│
├── agents/
│
├── graph/
│
├── rag/
│
├── llm/
│
├── api/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── scripts/
│   ├── setup_tigergraph_schema.py
│   ├── setup_vector_search.py
│   ├── test_vector_search.py
│   └── add_first_5_chunks.py
│
├── tests/
│
├── requirements.txt
├── .env
├── .env.example
└── README.md
```

The existing scripts remain useful as setup/diagnostic tools.

Production code should move reusable logic into `services` and
`repositories`.

------------------------------------------------------------------------

# 30. Endpoint responsibility

A clean API boundary is:

## Upload

``` text
POST /v1/datasets
```

Responsibility:

``` text
accept corpus
create dataset
start/queue processing
```

## Dataset status

``` text
GET /v1/datasets/{dataset_id}
```

Responsibility:

``` text
return dataset metadata and ingestion status
```

## List chunks

``` text
GET /v1/datasets/{dataset_id}/chunks
```

Responsibility:

``` text
retrieve chunks for inspection/debugging/API clients
```

## Get one chunk

``` text
GET /v1/datasets/{dataset_id}/chunks/{chunk_id}
```

Responsibility:

``` text
retrieve a specific chunk
```

## Start prepared-chunk ingestion

Recommended:

``` text
POST /v1/datasets/{dataset_id}/ingest
```

Responsibility:

``` text
start chunk → embedding → TigerGraph pipeline
```

The endpoint should not contain the embedding implementation itself.

------------------------------------------------------------------------

# 31. Important: endpoint vs service

This distinction should remain clear.

The API endpoint:

``` text
POST /v1/datasets/{dataset_id}/ingest
```

is only the entry point.

The actual workflow should be:

``` text
API endpoint
     ↓
Dataset/Ingestion service
     ↓
Chunk reader
     ↓
Embedding service
     ↓
TigerGraph repository
```

This means later a scheduled job, CLI command, or event can also trigger
ingestion without calling the HTTP endpoint.

For example:

``` text
API
 ───────────────┐
                │
CLI ────────────┼──> IngestionService
                │
Scheduler ──────┘
```

This is much more reusable.

------------------------------------------------------------------------

# 32. First implementation milestone

Do not implement the entire Agentic RAG system at once.

The immediate milestone should be:

``` text
chunks.jsonl
      ↓
POST /v1/datasets/{dataset_id}/ingest
      ↓
read chunks
      ↓
batch embedding
      ↓
validate 1536 dimensions
      ↓
upsert Chunk into TigerGraph
      ↓
return job/status
```

For the first working version, synchronous execution can be used for a
tiny test dataset if necessary.

Then move the same service into a background worker/job architecture
before large-scale ingestion.

------------------------------------------------------------------------

# 33. First test dataset

Before processing the complete corpus, use:

``` text
first 5 chunks
```

from:

``` text
data/processed/59aa1eeb-0c13-4f2c-9a11-b14f989cd14c/chunks.jsonl
```

This is important because those chunks were already successfully
embedded and inserted manually.

The new pipeline should reproduce the same result automatically.

Expected flow:

``` text
5 chunks
 ↓
1 batched embedding request
 ↓
5 × 1536 vectors
 ↓
5 TigerGraph Chunk vertices
```

Then verify:

``` text
TigerGraph vertex count = 5
```

and run:

``` text
scripts/test_vector_search.py
```

to confirm semantic retrieval still works.

------------------------------------------------------------------------

# 34. Production test cases

Create tests for at least:

### Valid chunk

``` text
text present
chunk_id present
doc_id present
```

Expected:

``` text
embedded + stored
```

### Missing text

Expected:

``` text
validation failure
```

### Empty text

Expected:

``` text
validation failure
```

### Duplicate chunk

Expected:

``` text
idempotent upsert
```

not a duplicate vertex.

### Same chunk, changed text

Expected:

``` text
new content hash
new embedding
updated vertex
```

### Wrong embedding dimension

Expected:

``` text
batch rejected
```

### OpenAI transient failure

Expected:

``` text
retry with backoff
```

### TigerGraph transient failure

Expected:

``` text
retry/recover
```

### Pipeline restart

Expected:

``` text
already completed chunks are skipped
remaining chunks continue
```

------------------------------------------------------------------------

# 35. Verification checklist

After implementing the pipeline, verify:

``` text
[ ] API starts successfully
[ ] dataset_id is handled correctly
[ ] chunks are read correctly
[ ] chunk metadata is preserved
[ ] embedding model is configurable
[ ] embedding dimension is validated
[ ] batches are used
[ ] OpenAI errors are handled
[ ] TigerGraph connection is reused
[ ] Chunk upserts are idempotent
[ ] content_hash is generated
[ ] embedding metadata is stored
[ ] ingestion status is tracked
[ ] vector index remains HNSW
[ ] vector metric remains COSINE
[ ] vector_search query still works
[ ] retrieved chunks contain text
[ ] no secrets appear in logs
[ ] tests cover failure cases
```

------------------------------------------------------------------------

# 36. What NOT to do

Avoid these designs:

### Do not embed inside every HTTP request for a huge corpus

``` text
POST
 ↓
embed 100,000 chunks
 ↓
wait
```

Use jobs/workers.

### Do not create a new TigerGraph connection for every chunk

Reuse the connection/repository.

### Do not make one OpenAI request per chunk

Batch embeddings.

### Do not discard metadata

It will be required by GraphRAG.

### Do not store only vectors

The text and provenance are necessary.

### Do not hard-code API keys

Use `.env` locally and secret management in deployment.

### Do not mix embedding models

Keep one consistent vector space per index/version.

### Do not make ingestion logic depend on agents

Agents belong above retrieval.

------------------------------------------------------------------------

# 37. Final target architecture

The architecture after this stage should look like:

``` text
                         CLIENT
                           │
                           ▼
                 POST /v1/datasets
                           │
                           ▼
                    Dataset Service
                           │
                           ▼
                     Ingestion Job
                           │
                           ▼
                     Chunk Reader
                           │
                           ▼
                  Chunk Validation
                           │
                           ▼
                 Content Hash Check
                           │
                           ▼
                   Embedding Service
                           │
                    OpenAI API
                           │
                    1536-d vector
                           │
                           ▼
                TigerGraph Repository
                           │
                           ▼
                    TigerGraph
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
        Chunk vertices              Graph edges
             │
             ▼
      HNSW vector index
             │
             ▼
       Future Retrieval
             │
       ┌─────┴─────┐
       ▼           ▼
 Vector Search   Graph Search
       │           │
       └─────┬─────┘
             ▼
        GraphRAG Layer
             │
             ▼
        Agentic RAG
             │
             ▼
            LLM
```

------------------------------------------------------------------------

# 38. Implementation order

Implement in this exact order.

## Step 1 --- Finalize TigerGraph schema

Before full ingestion, make sure the final `Chunk` vertex preserves the
useful metadata from `chunks.jsonl`.

Do not ingest the full corpus until this is settled.

## Step 2 --- Create data models

Create a validated internal representation of a chunk.

Example conceptual object:

``` text
ChunkRecord
├── chunk_id
├── doc_id
├── title
├── wikidata_qid
├── wikipedia_pageid
├── chunk_index
├── chunk_type
├── section
├── section_path
├── approx_tokens
└── text
```

## Step 3 --- Create `EmbeddingService`

Move the proven OpenAI embedding logic out of:

``` text
scripts/add_first_5_chunks.py
```

into a reusable service.

## Step 4 --- Create `TigerGraphRepository`

Move the proven:

``` python
upsertVertex(...)
```

logic into the repository.

## Step 5 --- Create `IngestionService`

Connect:

``` text
ChunkReader
   ↓
EmbeddingService
   ↓
TigerGraphRepository
```

## Step 6 --- Add idempotency

Use:

``` text
chunk_id
content_hash
embedding_model
embedding_version
```

## Step 7 --- Add API endpoint

Implement:

``` text
POST /v1/datasets/{dataset_id}/ingest
```

## Step 8 --- Test with 5 chunks

Use the existing:

``` text
chunks.jsonl
```

and confirm the new pipeline reproduces the previous experiment.

## Step 9 --- Verify vector search

Run:

``` text
scripts/test_vector_search.py
```

and confirm that the newly ingested vectors can be retrieved.

## Step 10 --- Scale to the full corpus

Only after the small test passes should the complete corpus be ingested.

------------------------------------------------------------------------

# 39. Definition of done for this stage

This stage is complete when the following works end-to-end:

``` text
POST /v1/datasets/{dataset_id}/ingest
                    │
                    ▼
             read chunks.jsonl
                    │
                    ▼
             validate chunks
                    │
                    ▼
            calculate hashes
                    │
                    ▼
          batch OpenAI embeddings
                    │
                    ▼
             verify 1536 dims
                    │
                    ▼
        upsert into TigerGraph
                    │
                    ▼
           update job status
                    │
                    ▼
            ingestion complete
```

And then:

``` text
query
 ↓
OpenAI embedding
 ↓
TigerGraph vectorSearch()
 ↓
stored Chunk
 ↓
text + metadata
```

must continue to work.

------------------------------------------------------------------------

# 40. Key architectural principle

The most important decision for avoiding future rework is:

``` text
INGESTION
   ≠
EMBEDDING
   ≠
STORAGE
   ≠
RETRIEVAL
   ≠
GRAPH REASONING
   ≠
AGENTS
```

They should communicate through clean interfaces.

That gives the project this evolution path:

``` text
Stage 1
Corpus → chunks

Stage 2  ← CURRENT
Chunks → embeddings → TigerGraph

Stage 3
TigerGraph → vector + graph retrieval

Stage 4
GraphRAG → hybrid retrieval

Stage 5
Agentic RAG → planning + tools + iterative retrieval

Stage 6
Evaluation → quality, latency, grounding, observability
```

The work already completed in:

``` text
app/tigergraph.py
scripts/setup_tigergraph_schema.py
scripts/add_first_5_chunks.py
scripts/setup_vector_search.py
scripts/test_vector_search.py
app/test_openai.py
```

should therefore be treated as the **validated foundation** for Stage 2
rather than discarded and rebuilt.
