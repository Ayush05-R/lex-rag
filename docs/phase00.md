# Phase 0 — Project Scaffold
### LexRAG Build Log

> This document explains what Phase 0 actually built — the repo, the folder
> structure, the tooling — and why each decision was made. No business logic
> exists yet. That starts in Phase 1.

---

## Table of Contents

1. [What Phase 0 Actually Built](#1-what-phase-0-actually-built)
2. [Repo Setup — uv and Public Visibility](#2-repo-setup)
3. [Project Structure — Why This Layout](#3-project-structure)
4. [Why pgvector Over a Dedicated Vector DB](#4-why-pgvector)
5. [Dependencies Installed — What Each One Is For](#5-dependencies-installed)
6. [Docker + CI Scaffold — From the Mandatory Templates](#6-docker-ci-scaffold)
7. [Git Workflow — Conventional Commits](#7-git-workflow)
8. [What Phase 0 Does NOT Have Yet](#8-what-phase-0-does-not-have-yet)
9. [Phase 0 File Reference](#9-phase-0-file-reference)

---

## 1. What Phase 0 Actually Built

Zero business logic. No dataset loaded, no chunking, no embeddings, no API
logic. Phase 0 is the foundation every later phase writes real code into.

What exists after Phase 0:

- A public GitHub repo, initialized with `uv`
- The full folder skeleton — every phase's eventual home, currently empty placeholders
- Every core dependency installed and locked
- Docker + CI + pre-commit scaffold, copied from your two mandatory template repos
- A Conventional Commits workflow, committed to `main`

## 2. Repo Setup

Repo: `github.com/Ayush05-R/lex-rag`, public.

**Why public, with data gitignored:** the ILDC dataset is under an academic-use
license from a gated Hugging Face repo. You can publish the *pipeline code*
freely, but not the raw dataset. This is why `data/raw/` and `data/processed/`
are gitignored — only the code that produces and consumes that data is public.

**Why `uv` for this project too:** same reasoning as url-shortener — a single
tool for dependency management, virtual environments, and locking, replacing
`pip` + `venv` + `pip-tools`. `uv.lock` guarantees identical dependency versions
between your machine and anywhere this gets deployed later (Phase 10).

## 3. Project Structure

```
lex-rag/
├── app/
│   ├── main.py                      # not yet a real FastAPI app — placeholder
│   ├── api/
│   │   └── v1/
│   │       └── routes/
│   │           ├── query.py         # empty — Phase 7
│   │           ├── documents.py     # empty — Phase 7
│   │           └── health.py        # empty — Phase 7
│   ├── core/
│   │   ├── config.py                # empty — settings management, Phase 1+
│   │   ├── db.py                    # empty — Postgres connection, Phase 5
│   │   ├── logging.py               # empty
│   │   ├── embeddings.py            # empty — Phase 4
│   │   ├── vector_store.py          # empty — Phase 5
│   │   ├── retriever.py             # empty — Phase 6
│   │   ├── llm.py                   # empty — Phase 6
│   │   └── rag_pipeline.py          # empty — Phase 6
│   └── tests/                       # empty — tests added per-phase
├── data/
│   ├── raw/                         # gitignored — raw ILDC CSVs land here, Phase 1
│   ├── processed/                   # gitignored — cleaned text, Phase 2
│   └── scripts/                     # download + ingestion scripts, Phase 1+
├── docker/
│   └── Dockerfile                   # scaffolded, not yet built/tested — Phase 10
├── notebooks/
│   └── explore_ildc.ipynb           # empty — EDA notebook, Phase 1
├── .github/
│   └── workflows/
│       └── ci.yml                   # scaffolded, not yet verified running
├── compose.yml                      # base services — Postgres, app
├── compose.override.yml             # local dev overrides
├── .pre-commit-config.yaml          # ruff hooks
├── .editorconfig
├── .flake8
└── pyproject.toml
```

### Why `app/api/v1/`, Not Just `app/api/`

Versioning the API path from day one (`/api/v1/...`) means that if the request/
response schema ever needs a breaking change later, you add `/api/v2/` instead
of breaking every existing client. Retrofitting versioning after a v1 client
already exists is far more painful than starting with it. This came directly
from the mandatory `fastapi/full-stack-fastapi-template` reference.

### Why `app/core/` Splits Into Named Files Instead of One `rag.py`

Each file in `core/` maps to exactly one stage of the pipeline described in
`why_this_project.md` Section 8: `embeddings.py` only knows how to turn text
into vectors, `vector_store.py` only knows how to talk to Postgres, `retriever.py`
only knows how to run a similarity search, `llm.py` only knows how to call the
model. `rag_pipeline.py` is the only file that knows about all of them —
it orchestrates, the others don't know about each other.

This mirrors the layered separation from url-shortener (API / domain /
infrastructure) applied to a RAG-specific pipeline instead of a CRUD app.

## 4. Why pgvector

Three real reasons for choosing pgvector over a dedicated vector database
(Pinecone, Weaviate, Qdrant):

**Direct experience reuse.** Postgres + async SQLAlchemy is exactly what you
already built and shipped in url-shortener. The connection patterns, migration
tooling, and query experience transfer directly — no new infrastructure to learn
from zero.

**One database, not two.** Chunk metadata (doc_id, chunk_index, char span) and
the vectors themselves live in the same table, in the same transaction. A
dedicated vector DB usually means keeping metadata in Postgres and vectors in
a separate system, and keeping the two in sync yourself.

**It's a real production pattern, not a shortcut.** Companies run pgvector at
real scale precisely because "one database instead of two" is operationally
simpler. It's a legitimate architecture choice to defend in an interview, not
a beginner's workaround.

## 5. Dependencies Installed

Via `uv add` / `uv add --dev`:

```
# main dependencies
fastapi[standard]      # API framework, matches url-shortener
pydantic-settings      # env-based config, same pattern as url-shortener's Settings
sqlalchemy             # ORM / query layer
psycopg2-binary        # Postgres driver
pgvector               # the vector column type + similarity operators for Postgres

# dev dependencies
ruff                   # linter/formatter
pytest                 # test runner
pre-commit             # runs ruff automatically before every commit
```

Note: `psycopg2-binary` here is the sync driver, not `asyncpg` (which url-shortener
used for full async). This is worth revisiting explicitly in Phase 5 when
`db.py` gets written — whether LexRAG's ingestion/query pattern actually needs
async Postgres access, or whether sync is fine for a system that isn't serving
high-concurrency traffic the way a URL redirect endpoint does.

## 6. Docker + CI Scaffold

Copied structurally from the two mandatory template repos
(`docs/backend-templates.md`):

- `docker/Dockerfile` + `compose.yml` + `compose.override.yml` — from
  `fastapi/full-stack-fastapi-template`
- `.pre-commit-config.yaml`, `.editorconfig`, `.flake8` — from
  `mirzadelic/fastapi-starter-project`
- `.github/workflows/ci.yml` — scaffolded but not yet verified to actually
  run (no tests exist yet for it to run against)

None of these are functional yet — they're structurally in place so that when
Phase 1 onward adds real code and real tests, CI and Docker builds don't need
to be retrofitted later.

## 7. Git Workflow

Same Conventional Commits convention as url-shortener:

```
feat: ...       new feature or phase
fix: ...        bug fix
chore: ...      non-functional changes (deps, config, scaffolding)
docs: ...       documentation
refactor: ...   restructuring, no behavior change
test: ...       adding or changing tests
```

Phase 0 was committed as a single commit to `main`:
```
chore: scaffold project structure with app, config, and CI placeholders
```

Starting Phase 1, the branch-per-phase workflow from url-shortener applies:
branch off `main` as `feat/phase-name`, commit as you go, merge back only once
that phase's tests are green.

## 8. What Phase 0 Does NOT Have Yet

- **No dataset** — nothing in `data/raw/` yet; that's the first thing Phase 1 does
- **No config values** — `config.py` is an empty file, not a working `Settings` class
- **No FastAPI app** — `main.py` still holds the original placeholder script, not
  wired up as an actual FastAPI app with a lifespan hook
- **No database connection** — `db.py` is empty; nothing has ever connected to Postgres
- **No tests** — the `tests/` folder exists but is empty; the CI workflow has
  nothing to run yet
- **No chunking, embedding, retrieval, or generation logic** — every file under
  `app/core/` is an empty placeholder

## 9. Phase 0 File Reference

Every file below is currently either empty or unchanged from its scaffolded
placeholder state. Nothing here is real logic yet — this section exists so
future-you can check, at a glance, exactly what's real and what isn't.

| File | State |
|---|---|
| `app/main.py` | Original hello-world script moved here as-is, not rewritten |
| `app/core/config.py` | Empty |
| `app/core/db.py` | Empty |
| `app/core/embeddings.py` | Empty |
| `app/core/vector_store.py` | Empty |
| `app/core/retriever.py` | Empty |
| `app/core/llm.py` | Empty |
| `app/core/rag_pipeline.py` | Empty |
| `app/api/v1/routes/query.py` | Empty |
| `app/api/v1/routes/documents.py` | Empty |
| `app/api/v1/routes/health.py` | Empty |
| `notebooks/explore_ildc.ipynb` | Empty |
| `docker/Dockerfile` | Scaffolded, unbuilt |
| `.github/workflows/ci.yml` | Scaffolded, unverified |

---

## What Comes Next — Phase 1 Preview

Phase 1 is the first phase that touches real data: downloading ILDC_single
from Hugging Face, loading it, and running a sanity check — row counts, null
checks, and the text-length distribution that Phase 3's chunking strategy
depends on.

Nothing in Phase 1 is chunking, embedding, or retrieval yet. It's purely
"get the real data in, and know its actual shape" — you cannot design a
chunker for documents whose length distribution you haven't measured.

---

*Phase 0 complete. Structure exists. Nothing runs yet. Every phase after this
fills in exactly one empty file with real, tested logic.*
