# LexRAG — Project Roadmap

A legal RAG (Retrieval-Augmented Generation) system built over the ILDC (Indian Legal Documents Corpus, via IL-TUR) dataset. This doc lays out the full 10-phase build plan — what's done, what's next, and why each phase exists — so the whole arc is visible in one place instead of scattered across per-phase docs.

**Workflow convention:** one git branch per phase (`feat/phase-name`), merged into `main` only after that phase's tests pass. Commit messages follow Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`).

---

## Table of Contents
1. [Phase 1 — Data Acquisition & EDA](#phase-1--data-acquisition--eda-done)
2. [Phase 2 — Data Cleaning](#phase-2--data-cleaning-next)
3. [Phase 3 — Chunking](#phase-3--chunking)
4. [Phase 4 — Embeddings](#phase-4--embeddings)
5. [Phase 5 — Vector Store & Ingestion](#phase-5--vector-store--ingestion-pgvector)
6. [Phase 6 — Retrieval + Generation (Core RAG Loop)](#phase-6--retrieval--generation-core-rag-loop)
7. [Phase 7 — API Layer & Tests](#phase-7--api-layer--tests)
8. [Phase 8 — Evaluation Harness](#phase-8--evaluation-harness)
9. [Phase 9 — Hybrid Search + Reranking](#phase-9--hybrid-search--reranking)
10. [Phase 10 — Deployment](#phase-10--deployment)

---

## Phase 1 — Data Acquisition & EDA ✅ (done)

**Goal:** Get the raw ILDC_single data and understand its shape before writing any pipeline logic.

- Got Hugging Face access to `Exploration-Lab/IL-TUR`, downloaded `ILDC_single` train/dev/test CSVs (5082/2511/1517 rows) into `data/raw/`.
- Only using the `text` column — `label`/`expert_*` columns belong to a judgment-prediction task, not RAG.
- EDA findings: heavy right-skew in word count (mean 4012, median 2982, max 87,530), zero nulls, 5 duplicate texts in `train`, and a systematic OCR artifact ("not"/"no." → "number").
- These findings directly shape Phase 2 (dedup + OCR fix) and Phase 3 (chunking can't use one fixed size).
- *(Detailed writeup: `docs/phase01.md`)*

---

## Phase 2 — Data Cleaning (next)

**Goal:** Turn the raw, noisy corpus into something safe to chunk and embed.

- Drop the 5 duplicate rows found in `train`.
- Fix the "not/no." → "number" OCR substitution — likely a targeted string-replace/regex pass since it's a consistent, known pattern rather than random noise.
- General text normalization pass: whitespace cleanup, encoding issues, any other artifacts surfaced during cleaning.
- Output: a cleaned corpus written to `data/processed/`, ready for chunking.

---

## Phase 3 — Chunking

**Goal:** Split full judgments into retrieval-sized chunks without losing legal meaning.

- Can't use a single fixed chunk size — the extreme length skew (2,982-word median vs 87,530-word max) means one size either wastes effort on short docs or produces wildly uneven chunk counts per document.
- Needs a length-aware or recursive chunking strategy (e.g., recursive character/token splitting with overlap, possibly section-aware splitting if judgments have identifiable structure like headings or paragraph numbering).
- Decide on and record: target chunk size, overlap size, and how chunk-to-source-document mapping is preserved (needed later for citation/traceability in generation).

---

## Phase 4 — Embeddings

**Goal:** Turn each text chunk into a vector representation for semantic search.

- Choose an embedding model — likely a choice between a general-purpose model and a legal-domain-tuned one; worth benchmarking both given the domain-specific vocabulary in Indian case law.
- Decide on embedding dimensionality (affects pgvector index size/performance in Phase 5).
- Batch-embed all chunks; store alongside chunk metadata (source document id, chunk index, etc.) for later ingestion.

---

## Phase 5 — Vector Store & Ingestion (pgvector)

**Goal:** Get embedded chunks into a queryable vector database.

- Vector store choice already locked in: **pgvector** (Postgres extension), reusing Postgres experience from the url-shortener project.
- Design the schema: chunks table with embedding column, foreign key back to source document, metadata columns.
- Write the ingestion pipeline: read processed + embedded chunks, batch-insert into Postgres, build the vector index (e.g., IVFFlat or HNSW depending on pgvector version/support).
- Sanity-check ingestion: row counts match expectations, spot-check a few embeddings retrieve sensible neighbors.

---

## Phase 6 — Retrieval + Generation (Core RAG Loop)

**Goal:** Wire up the actual RAG loop — the heart of the project.

- Retriever: given a query, embed it and pull top-k nearest chunks from pgvector.
- LLM integration: pass retrieved chunks + query into a prompt template, get a generated answer.
- This is where `app/core/retriever.py`, `app/core/llm.py`, and `app/core/rag_pipeline.py` (already scaffolded) get their actual logic.
- First end-to-end "ask a legal question, get an answer grounded in retrieved case text" milestone.

---

## Phase 7 — API Layer & Tests

**Goal:** Expose the RAG pipeline as a usable service.

- Build out the FastAPI routes already scaffolded (`app/api/v1/routes/query`, `documents`, `health`).
- `query` endpoint: accept a question, run it through the Phase 6 pipeline, return the answer + source chunks.
- `documents` endpoint: likely for inspecting/searching the underlying corpus directly.
- `health` endpoint: basic liveness/readiness check.
- Write tests (`app/tests/`) covering routes and core pipeline logic — this is also where CI (`.github/workflows/ci.yml`, already scaffolded) starts actually running something meaningful.

---

## Phase 8 — Evaluation Harness

**Goal:** Measure whether the RAG system is actually good, not just functional.

- Define metrics: retrieval quality (e.g., recall@k, MRR) and generation quality (faithfulness to retrieved context, relevance to the query).
- Build a small evaluation set — possibly hand-labeled queries with known relevant judgments, or leveraging existing IL-TUR benchmark structure if applicable.
- Worth a quick check here for cross-split duplicate/leakage issues (train/test contamination) before trusting any eval numbers.

---

## Phase 9 — Hybrid Search + Reranking

**Goal:** Improve retrieval quality beyond pure vector similarity.

- Hybrid search: combine vector similarity with keyword-based search (e.g., BM25/full-text search) — useful in legal text where exact terminology (case citations, statute names) matters and pure semantic similarity can miss it.
- Reranking: run initial retrieved candidates through a cross-encoder or reranking model to reorder by finer-grained relevance before passing to generation.
- Compare against Phase 8's evaluation harness to confirm this actually improves metrics, not just intuition.

---

## Phase 10 — Deployment

**Goal:** Get LexRAG running somewhere real, not just on localhost.

- Docker setup already scaffolded (`docker/Dockerfile`, `compose.yml`, `compose.override.yml`) — finalize these for production use.
- Likely candidates given prior project experience: a platform like Render (used for the url-shortener project), or similar.
- Environment/config management for production Postgres + pgvector, embedding model access, and LLM API keys.
- Final smoke test: the deployed API answering real legal queries end-to-end.

---

## Notes

- Phase docs (`docs/phaseNN.md`) are written *after* each phase's code is actually built, not in advance — this roadmap is the forward-looking companion to those retrospective writeups.
- Shelved for later: adding Google's "OKF" (Open Knowledge Format) — revisit once the core 10 phases are done.
