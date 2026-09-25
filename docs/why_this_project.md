# Why LexRAG — Concept Primer & Project Rationale
### Read This Before Phase 1. Every Concept You Need, Explained From First Principles.

> This is not a build log — nothing here depends on code that doesn't exist yet.
> This is the theory layer: what RAG actually is, why legal text is a hard case for it,
> and what every later phase is building toward. Once Phase 1 is done, `phase01.md`
> will document the real, built thing. This document explains the *why* underneath it.

---

## Table of Contents

1. [What LexRAG Actually Is](#1-what-lexrag-actually-is)
2. [Why a Legal RAG System — The Portfolio Case](#2-why-a-legal-rag-system)
3. [RAG From First Principles](#3-rag-from-first-principles)
4. [Embeddings — Turning Text Into Geometry](#4-embeddings)
5. [Similarity Search — What "Closest" Means](#5-similarity-search)
6. [Why Legal Text Breaks Naive RAG](#6-why-legal-text-breaks-naive-rag)
7. [The ILDC Dataset — What You're Actually Working With](#7-the-ildc-dataset)
8. [The Full Pipeline — How Every Phase Connects](#8-the-full-pipeline)
9. [Vector Indexes — HNSW vs IVFFlat, and Why It Matters](#9-vector-indexes)
10. [Grounding and Citation — The Difference Between RAG and a Chatbot](#10-grounding-and-citation)
11. [Evaluation — How You'll Know If Any of This Actually Works](#11-evaluation)
12. [Glossary — Every Term Used Across All Phases](#12-glossary)

---

## 1. What LexRAG Actually Is

LexRAG is a **Retrieval-Augmented Generation system for Indian case law**. In plain terms:
you ask a legal question, the system finds the actual judgment text most relevant to
that question, and hands both the question and that retrieved text to an LLM, which
answers *grounded in the real documents* — not from what the LLM memorized during training.

The core promise of RAG is this: **the LLM's opinion doesn't matter. The retrieved
document does.** If the system can't find a relevant judgment, it should say so —
not hallucinate a plausible-sounding one.

## 2. Why a Legal RAG System

Three reasons this is a strong portfolio piece, not just "another RAG project":

**Domain difficulty.** Legal text is long, dense, and citation-heavy. A RAG system
that works on Wikipedia paragraphs often breaks on a 15,000-word judgment. Solving
that (Section 6) is a real engineering problem, not a tutorial exercise.

**Grounding matters more here than anywhere else.** A wrong answer about a recipe is
annoying. A wrong answer about a legal precedent is the exact failure mode every
serious RAG deployment has to solve for. Building citation-forced generation
(Section 10) is the single most interview-relevant thing in this project.

**It maps directly to your stated goal** — AI/DL research engineering. Retrieval,
embeddings, and grounding are the same primitives underneath every serious LLM
application being built at frontier labs right now. This project is a small,
correct version of a hard, real problem — not a toy.

## 3. RAG From First Principles

### The Problem It Solves

An LLM's knowledge is frozen at training time and general-purpose. Two failure
modes follow directly from that:

1. **It doesn't know your specific documents** — your legal case corpus was never
   in its training data.
2. **It hallucinates** when asked about something specific and unfamiliar — it
   produces fluent, confident, wrong text, because that's what the training
   objective rewards when it's unsure.

RAG fixes both by never asking the LLM to recall facts from memory. Instead:

```
User question
     │
     ▼
Retrieve top-k relevant chunks from YOUR document store
     │
     ▼
Build a prompt: "Here is the question. Here is the retrieved text. Answer using
                 ONLY this text, and cite which chunk supports each claim."
     │
     ▼
LLM generates an answer constrained to the retrieved evidence
```

### The Analogy

> Think of the LLM as a brilliant law graduate who has read every legal textbook
> ever written but has never seen *your* client's case file.
>
> You wouldn't ask them to argue the case from memory — they'd invent plausible
> but wrong details. Instead, you hand them the actual case file (retrieval) and
> say: "Argue using only what's in this folder, and point to the page for every
> claim you make" (grounding). The graduate's legal reasoning ability is still
> doing the work — but it's now anchored to real evidence instead of guesswork.

RAG doesn't make the LLM smarter. It makes the LLM's answers *checkable*.

## 4. Embeddings

An embedding is a function that turns a piece of text into a vector — a list of
numbers, typically 384 to 1536 dimensions — such that texts with similar *meaning*
end up as vectors that are close together in that space.

```python
embed("The court dismissed the appeal") → [0.02, -0.31, 0.88, ..., 0.14]  # 768 numbers
embed("The appellate court rejected the case") → [0.03, -0.29, 0.85, ..., 0.11]  # close to above
embed("The recipe calls for two eggs") → [0.71, 0.42, -0.05, ..., -0.63]  # far from both
```

The key property: this is **semantic** closeness, not keyword overlap. The first
two sentences share almost no exact words, but an embedding model trained on
enough text learns that "dismissed the appeal" and "rejected the case" mean
close to the same thing, and places them near each other geometrically.

### Why This Matters for Retrieval

Keyword search (like a `Ctrl+F` or SQL `LIKE`) fails when the user's wording
doesn't match the document's wording. Embedding-based search doesn't care about
exact wording — it searches by meaning. This is why RAG systems embed both the
document chunks and the incoming query, then search for the chunks whose vectors
are closest to the query's vector.

## 5. Similarity Search

"Closest" needs a precise definition. The standard measure is **cosine similarity**
— the cosine of the angle between two vectors, ranging from -1 (opposite meaning)
to 1 (identical meaning).

```
cosine_similarity(A, B) = (A · B) / (‖A‖ × ‖B‖)
```

Why cosine and not plain distance (Euclidean)? Because embedding vectors' *direction*
carries the meaning — magnitude is mostly an artifact of text length. Two chunks
about the same topic, one long and one short, might have very different magnitudes
but point in almost the same direction. Cosine similarity ignores magnitude and
compares direction only — exactly what you want when comparing meaning, not length.

pgvector (your chosen vector store) supports cosine distance natively via the
`<=>` operator, computed directly inside Postgres.

## 6. Why Legal Text Breaks Naive RAG

This is the part that makes LexRAG a real engineering project instead of a
copy-paste tutorial. Three specific failure modes:

**Length.** ILDC judgments can run past 10,000 words. Embedding models have a
token limit (usually 512 tokens per chunk for common models) — you cannot embed
a whole judgment as one vector without losing most of its content to truncation.
This is why chunking (Phase 3) exists at all.

**Structural density.** Judgments have numbered paragraphs, nested citations to
other cases, and quoted statute text. A naive fixed-size chunker (split every
N characters) will cut a sentence in half mid-citation, destroying the exact
information a legal question is most likely to ask about.

**Precision requirements.** In a chatbot, an approximately-right retrieved chunk
is fine. In a legal context, retrieving the *wrong* paragraph of a judgment
(e.g., the dissenting opinion instead of the majority holding) produces an
answer that is confidently, dangerously wrong. This is why grounding + citation
(Section 10) isn't optional polish — it's the actual point.

## 7. The ILDC Dataset

ILDC (Indian Legal Documents Corpus) is part of the IL-TUR benchmark
(`Exploration-Lab/IL-TUR` on Hugging Face) — real Indian court judgment text,
originally built for a judgment-prediction task (predicting case outcome from
judgment text). You're repurposing only the `text` column — the raw judgment
text — as your document corpus. The `label`/`split` columns (built for outcome
prediction) are irrelevant to a RAG system and are ignored, as already decided.

This matters because it's **real, messy, production-shaped data** — not a
clean toy dataset built for tutorials. OCR artifacts, inconsistent formatting,
and genuine length variance are exactly the conditions a production RAG system
has to survive.

## 8. The Full Pipeline

```
Raw judgment text (ILDC)
     │
     ▼  Phase 2: clean boilerplate, normalize whitespace
Cleaned text
     │
     ▼  Phase 3: paragraph/section-aware chunking
Chunks + metadata (doc_id, chunk_index, char span)
     │
     ▼  Phase 4: embedding model
Chunk vectors
     │
     ▼  Phase 5: pgvector ingestion + indexing
Searchable vector store
     │
     ▼  Phase 6: query → embed → retrieve top-k → build grounded prompt → LLM
Cited, grounded answer
     │
     ▼  Phase 7: FastAPI endpoints wrap the pipeline
     ▼  Phase 8: evaluation harness scores retrieval + faithfulness
     ▼  Phase 9: hybrid search + reranking improve on the baseline
     ▼  Phase 10: Dockerized, deployed
```

Every phase's `phaseNN.md` will document the real, built version of one arrow
in this diagram — with real code, real numbers, once it exists.

## 9. Vector Indexes

Once you have thousands of chunk vectors, brute-force comparing a query vector
against every single one is `O(n)` and gets slow at scale. A vector index trades
a small amount of accuracy for massive speed.

**IVFFlat** — partitions vectors into clusters ("lists") at index-build time. A
query only searches the nearest few clusters instead of everything. Simpler,
needs `lists` tuned to your row count, degrades if data changes a lot after
building.

**HNSW** (Hierarchical Navigable Small World) — builds a multi-layer graph where
each vector links to its nearest neighbors. Search "hops" through the graph
toward the query. Slower to build, faster and more accurate to query, handles
new inserts better than IVFFlat.

For LexRAG's scale (thousands, not billions, of chunks), either works — Phase 5
will pick one and explain the actual tradeoff once row counts are known.

## 10. Grounding and Citation

This is the difference between a RAG system and "a chatbot with search enabled."

**Grounding** means the prompt explicitly instructs the LLM to answer *only*
from retrieved text, and structurally makes it easy to check — e.g. requiring
the LLM to tag each claim with the chunk ID it came from. This is why chunk
metadata (Phase 3) exists: without a stable `chunk_id`, there's nothing to cite.

**Why this is the hard part:** LLMs are trained to be helpful and fluent by
default — including when they don't have enough grounding to answer honestly.
A well-designed RAG prompt has to explicitly permit "I don't have enough
information to answer this" as a valid, expected output. A system that never
says "I don't know" is a system that's silently hallucinating on hard questions.

## 11. Evaluation

Two separate things need measuring, and conflating them is a common mistake:

**Retrieval quality** — did the system find the right chunks at all? Measured
with metrics like `precision@k` (of the top-k retrieved chunks, what fraction
are actually relevant) against a small hand-labeled set of question→relevant-chunk
pairs you'll build in Phase 8.

**Generation faithfulness** — given the retrieved chunks (even if they're
correct), did the LLM's answer actually stick to them, or did it add
unsupported claims? This is checked separately, because a system can retrieve
perfectly and still hallucinate on top of good evidence.

Phase 8 builds baseline numbers for both. Phase 9's hybrid search + reranking
changes are judged by whether they move these numbers, not by vibes.

## 12. Glossary

| Term | Meaning |
|---|---|
| RAG | Retrieval-Augmented Generation — retrieve relevant text, then generate an answer grounded in it |
| Embedding | A vector representation of text where semantic similarity = geometric closeness |
| Chunk | A sub-document-sized piece of text, small enough to embed meaningfully |
| Cosine similarity | Angle-based similarity between two vectors; ignores magnitude |
| pgvector | Postgres extension adding a vector column type + similarity search operators |
| HNSW / IVFFlat | Approximate nearest-neighbor index types for fast vector search at scale |
| Grounding | Constraining an LLM's output to only use retrieved evidence |
| Hallucination | An LLM generating fluent but factually unsupported text |
| precision@k | Of the top-k retrieved results, the fraction that are actually relevant |
| Reranking | A second, more expensive scoring pass over an initial retrieval set to improve ordering |
| Hybrid search | Combining keyword search (BM25) with vector search to catch both exact-term and semantic matches |

---

*Everything above is general knowledge about RAG systems — none of it is LexRAG's
actual code. `phase00.md` documents what's actually been built so far. Every
phase after Phase 1 gets its own `phaseNN.md`, written only once that phase's
real code exists and runs.*
