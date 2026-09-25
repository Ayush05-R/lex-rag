# Phase 01 — Data Acquisition & Exploratory Data Analysis

### LexRAG Build Log

> This document explains what Phase 1 actually built — the download script,
> the EDA notebook, and the real numbers we now know about the corpus. No
> cleaning, chunking, or embedding logic exists yet. That starts in Phase 2.

---

## Table of Contents

1. [What Phase 1 Actually Built](#1-what-phase-1-actually-built)
2. [Dataset Overview](#2-dataset-overview)
3. [Download Script — What It Does and Why](#3-download-script)
4. [EDA Notebook — Cell by Cell](#4-eda-notebook)
5. [Findings — The Real Numbers](#5-findings)
6. [Decisions Carried Into Phase 2 and Phase 3](#6-decisions-carried-forward)
7. [What Phase 1 Does NOT Have Yet](#7-what-phase-1-does-not-have-yet)
8. [Phase 1 File Reference](#8-phase-1-file-reference)

---

## 1. What Phase 1 Actually Built

Real data landed on disk for the first time. No cleaning, no chunking, no
embeddings — this phase answers exactly one question: **what does the
ILDC_single corpus actually look like?**

What exists after Phase 1:

- `data/raw/` populated with the real ILDC_single CSVs (train/dev/test)
- `data/scripts/download_data.py` — a repeatable script to re-fetch the data
- `notebooks/explore_ildc.ipynb` — length distribution, null/duplicate checks,
  manual sample reads
- Concrete numbers that Phase 2 (cleaning) and Phase 3 (chunking) now depend on,
  instead of guesses

## 2. Dataset Overview

Source: `Exploration-Lab/IL-TUR` on Hugging Face, config `cjpe`.

Downloaded splits (`data/raw/`, gitignored per the Phase 0 decision):

| File | Rows |
| --- | --- |
| `cjpe_dataset_single_train.csv` | 5,082 |
| `cjpe_dataset_single_dev.csv` | 2,511 |
| `cjpe_dataset_test.csv` | 1,517 |

All three share the identical schema: `id`, `text`, `label`, `expert_1`
through `expert_5`. `cjpe_dataset_test.csv` has no `single`-specific name in
its filename, but its columns match train/dev exactly — confirmed it's
single's test split, not a shared/ambiguous file across task variants.

Only `text` is used going forward. `label` and `expert_*` exist because
ILDC's original purpose is judgment-outcome prediction, not retrieval — as
already decided in `why_this_project.md` Section 7, these columns are ignored
by LexRAG.

## 3. Download Script

`data/scripts/download_data.py`:

```python
from datasets import load_dataset
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

def main():
    dataset = load_dataset("Exploration-Lab/IL-TUR", "cjpe")
    for split_name, split_data in dataset.items():
        out_path = RAW_DIR / f"ildc_single_{split_name}.csv"
        split_data.to_csv(out_path)
        print(f"Saved {split_name}: {len(split_data)} rows -> {out_path}")

if __name__ == "__main__":
    main()
```

**Why this is a script, not a one-off manual download:** anyone (including
future-you on a new machine) can re-run this and get the same data, as long
as they have their own Hugging Face account with ILDC access approved.

**Why no credentials live in this file:** `load_dataset()` relies on a token
cached locally by `huggingface-cli login`, stored outside the repo. Cloning
`lex-rag` does not grant dataset access — each user authenticates
independently. This keeps the repo genuinely public without leaking gated
access, consistent with the Phase 0 decision to gitignore the data but not
the code.

**Why `dataset.items()` works this way:** a Hugging Face `DatasetDict` behaves
like a plain Python dict keyed by split name (`train`, `dev`, `test`), so one
loop handles every split without three copy-pasted blocks.

## 4. EDA Notebook

`notebooks/explore_ildc.ipynb`, in the order it was actually run:

**Load all splits into one dict:**

```python
splits = {
    "train": pd.read_csv("../data/raw/cjpe_dataset_single_train.csv"),
    "dev": pd.read_csv("../data/raw/cjpe_dataset_single_dev.csv"),
    "test": pd.read_csv("../data/raw/cjpe_dataset_test.csv"),
}
```

A dict mapping split name → dataframe lets every check below run once per
split via a loop, instead of tripling every line of analysis code.

**Word count per document:**

```python
df["word_count"] = df["text"].str.split().str.len()
```

`.str.split()` breaks each row into a list of words on whitespace;
`.str.len()` counts that list's length. This runs vectorized across the whole
column — no manual row-by-row Python loop.

**Null and empty check:**

```python
n_null = df["text"].isna().sum()
n_empty = (df["text"].str.strip() == "").sum()
```

`.isna()` only catches genuinely missing cells (`NaN`). A row containing only
whitespace is not `NaN` — it's a real, present, useless value — so both
checks are needed to catch missingness in either form.

**Duplicate check:**

```python
df["text"].duplicated().sum()
```

Flags every repeat occurrence of a text value as `True` (the first occurrence
stays `False`), so the sum is a count of redundant rows.

**Manual sample read:**

```python
sample = splits["train"].sample(5, random_state=42)
for i, row in sample.iterrows():
    print(row["text"][:2000])
```

`random_state=42` pins the random sample so the same 5 rows come back on
re-runs — useful for referencing a specific row's issue later without the
sample shifting underneath you.

## 5. Findings

**Word count distribution — heavily right-skewed in every split:**

| Split | Mean | Median | Min | Max |
| --- | --- | --- | --- | --- |
| train | 4,012 | 2,982 | 147 | 87,530 |
| dev | 3,812 | 2,909 | 73 | 42,198 |
| test | 3,850 | 2,940 | 73 | 40,487 |

Mean sitting well above median in every split, plus a max roughly 30x the
median, confirms what `why_this_project.md` Section 6 predicted in theory:
judgment length varies enormously, and a small number of very long documents
pull the average up. This isn't a hypothetical concern anymore — it's a
measured property of the actual corpus.

**Nulls/empty strings:** zero, across all three splits. No missing-value
handling needed in Phase 2.

**Duplicates:** 5 duplicate texts in train, 0 in dev, 0 in test. Small, but
non-zero — needs a drop step in Phase 2.

**Manual read (5 random train rows):** confirmed these are real Indian
Supreme Court judgments — case/petition numbers and counsel names first, then
the judgment body delivered by a named judge. Also surfaced a **systematic
OCR/text-extraction artifact**: "not"/"no." appears corrupted to "number" in
multiple places (e.g. "it is number necessary" instead of "it is not
necessary"; "respondent number 1" instead of "respondent no. 1"). Appearing
in more than one of only 5 sampled rows suggests this is corpus-wide, not an
isolated glitch.

## 6. Decisions Carried Forward

- **Phase 2 cleaning must include:** dropping the 5 duplicate train rows, and
  a targeted fix for the "number" ↔ "not"/"no." substitution — likely a
  regex pass with manual spot-checking rather than a blind find-replace,
  since "number" also has legitimate uses (e.g. "Civil Appeal Number 704").
- **Phase 3 chunking cannot use a single fixed chunk size.** The length skew
  measured here rules out a uniform character-count or word-count split — a
  147-word document and an 87,530-word document need fundamentally different
  handling. The actual strategy is decided in Phase 3, once cleaning is done
  and the length distribution is re-measured on cleaned text.
- **`label` and `expert_*` columns confirmed unused for RAG** — the corpus is
  `text` only, consistent with the plan in `why_this_project.md` Section 7.

## 7. What Phase 1 Does NOT Have Yet

- **No cleaning** — duplicates and the OCR artifact are identified, not fixed
- **No chunking strategy** — the length distribution is measured, but how to
  split documents is a Phase 3 decision, not made here
- **No processed data** — `data/processed/` remains empty; only `data/raw/`
  has real files
- **No config, database, or pipeline logic** — every file under `app/core/`
  is still an empty placeholder, unchanged from Phase 0

## 8. Phase 1 File Reference

| File | State |
| --- | --- |
| `data/scripts/download_data.py` | Real, working — fetches ILDC_single from HF |
| `data/raw/cjpe_dataset_single_train.csv` | Real data, 5,082 rows |
| `data/raw/cjpe_dataset_single_dev.csv` | Real data, 2,511 rows |
| `data/raw/cjpe_dataset_test.csv` | Real data, 1,517 rows |
| `notebooks/explore_ildc.ipynb` | Real EDA — shape, length stats, null/dup checks, sample reads |
| `data/processed/` | Still empty — Phase 2 |
| `app/core/*` | Still empty placeholders — unchanged from Phase 0 |

---

## What Comes Next — Phase 2 Preview

Phase 2 takes the two concrete problems this phase surfaced — 5 duplicate
rows and a systematic OCR substitution error — and actually fixes them,
producing cleaned text in `data/processed/`. Nothing in Phase 2 touches
chunking or embeddings yet; it's purely "make the raw text trustworthy"
before anything downstream reads it.

---

*Phase 1 complete. Real data is on disk, and its shape is measured, not
assumed. Phase 2 cleans it based on exactly what was found here — nothing
more, nothing speculative.*
