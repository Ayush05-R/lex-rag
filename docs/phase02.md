# Phase 02 — Data Cleaning

**Branch:** `feat/phase-cleaning`
**Script:** `data/scripts/preprocess.py`
**Input:** `data/raw/` (never modified)
**Output:** `data/processed/train_clean.csv`, `dev_clean.csv`, `test_clean.csv`

---

## Table of Contents

1. [What this phase does](#1-what-this-phase-does)
2. [What Phase 1 handed over](#2-what-phase-1-handed-over)
3. [The decision: why the OCR fix was removed](#3-the-decision-why-the-ocr-fix-was-removed)
4. [The final code](#4-the-final-code)
5. [Code walkthrough, line by line](#5-code-walkthrough-line-by-line)
6. [Running the script](#6-running-the-script)
7. [Verifying the output](#7-verifying-the-output)
8. [Linting lessons: F401 and I001](#8-linting-lessons-f401-and-i001)
9. [Git workflow and the license rule](#9-git-workflow-and-the-license-rule)
10. [Theory Q&A](#10-theory-qa)
11. [Known limitations and open questions](#11-known-limitations-and-open-questions)
12. [What comes next](#12-what-comes-next)

---

## 1. What this phase does

Think of a warehouse with three areas:

- **Incoming dock** (`data/raw/`): goods exactly as they arrived. Nobody edits anything here.
- **Inspection station** (`preprocess.py`): picks up each item, removes the defective ones, and logs what it removed.
- **Outgoing shelf** (`data/processed/`): the inspected goods, ready for the next stage.

Because the raw files are never touched, the script can be rerun any number of times and always starts from the original data. This property is called **idempotence**: running the same operation twice gives the same result as running it once.

The honest summary of this phase: the script does **one** real cleaning operation, removing 5 duplicate rows from the train split. That is the correct amount of cleaning for this dataset, because that is all Phase 1's EDA proved necessary.

---

## 2. What Phase 1 handed over

The EDA in `notebooks/explore_ildc.ipynb` produced these findings. Each one needed a decision in this phase.

| Finding from Phase 1 | Decision in Phase 2 |
|---|---|
| 5 duplicate texts in train, 0 in dev and test | Drop them from train |
| Zero nulls or empty strings in all splits | No null handling needed |
| Systematic OCR artifact: "not" / "no." corrupted into "number" | Investigated, **deliberately not fixed** (see Section 3) |
| Word counts heavily right-skewed (mean 4012, median 2982, max 87530) | Not a cleaning problem. Carried to Phase 3 (chunking) |

---

## 3. The decision: why the OCR fix was removed

The first version of the script contained this function:

```python
def fix_ocr_artifacts(text: str) -> str:
    text = re.sub(r"\bnumber\b", "not", text)
    return text
```

### How it works

`re.sub(pattern, replacement, text)` finds every match of `pattern` in `text` and replaces it. Here the pattern `\bnumber\b` matches the whole word "number". The `\b` is a **word boundary**, so "numbers" and "renumbered" are not matched.

### Why it was wrong

It replaces **every** standalone "number", not only the corrupted ones. Legal judgments use the word legitimately all the time:

- "the number of witnesses" would become "the not of witnesses"
- "Criminal Appeal number 123 of 1998" would become "Criminal Appeal not 123 of 1998"
- "registration number", "section number", "serial number" would all be damaged

The analogy: fixing a few smudged words in a book by running find-and-replace on every copy of that word. A handful of errors get fixed, and a much larger number of correct passages get broken.

Two further problems:

- **Case sensitivity.** The pattern never matches "Number".
- **Ambiguity.** The corruption can come from "not" or from "no.", and a replacement to "not" cannot tell which one the original was.

### Why leaving the artifact alone is acceptable

Each judgment averages around 4000 words, and the artifact affects a small number of tokens inside them. An embedding model compresses a whole chunk into one vector, so a few wrong tokens barely move it. Silently corrupting legitimate text is a worse outcome than leaving a few known typos.

### What this phase did not do

No measurement of how many "number" occurrences are corrupted versus legitimate is recorded in this phase. If retrieval quality later looks poor on negation-heavy queries (for example, judgments that hinge on "not guilty" versus "guilty"), revisit this. The right approach then is a **narrow** fix targeting specific corrupted phrases found by counting the words around "number", never a blanket replace.

---

## 4. The final code

```python
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "train": "cjpe_dataset_single_train.csv",
    "dev": "cjpe_dataset_single_dev.csv",
    "test": "cjpe_dataset_test.csv",
}


def load_split(filename: str) -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / filename)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset="text", keep="first")
    after = len(df)
    print(f"Dropped {before - after} duplicate rows.")
    return df


def clean_split(name: str, filename: str) -> None:
    print(f"--- Cleaning {name} ---")
    df = load_split(filename)

    if name == "train":
        df = drop_duplicates(df)

    out_path = PROCESSED_DIR / f"{name}_clean.csv"
    df.to_csv(out_path, index=False)
    print(f"--- {name} cleaned and saved to {out_path} ---")


if __name__ == "__main__":
    for name, filename in SPLITS.items():
        clean_split(name, filename)
```

---

## 5. Code walkthrough, line by line

### Imports

```python
from pathlib import Path

import pandas as pd
```

- `Path` (from `pathlib`, part of Python's standard library) is a safer way to handle file paths than plain strings. It joins paths with `/` and works on both Windows and Linux.
- `pandas` is the library for working with tables. A table in pandas is a **DataFrame**, which you can picture as an Excel sheet held in memory.
- The order matters: standard library first, blank line, then third-party packages. See Section 8.

### Constants

```python
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
```

- Capital-letter names are a Python convention for **constants**: values defined once and never changed.
- `mkdir` creates the output folder. `parents=True` also creates any missing parent folders. `exist_ok=True` means "do not crash if it already exists", which is what makes reruns safe.
- These paths are **relative**, meaning they are resolved from the folder you launch the command from. That is why the script must be run from the repo root.

### The SPLITS dictionary

```python
SPLITS = {
    "train": "cjpe_dataset_single_train.csv",
    "dev": "cjpe_dataset_single_dev.csv",
    "test": "cjpe_dataset_test.csv",
}
```

A dictionary maps a short name to a filename. It lets one loop handle all three splits instead of copy-pasting the same code three times.

### load_split

```python
def load_split(filename: str) -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / filename)
```

- `RAW_DIR / filename` joins folder and filename into one path.
- `pd.read_csv` reads the file into a DataFrame.
- `-> pd.DataFrame` is a **type hint**. It labels what the function returns. Python ignores it at runtime, but readers and tools use it.

### drop_duplicates

```python
def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset="text", keep="first")
    after = len(df)
    print(f"Dropped {before - after} duplicate rows.")
    return df
```

- `len(df)` is the number of rows. Recording it before and after lets the script report how many rows were removed.
- `subset="text"` compares only the `text` column. Two rows could have different `id` values but hold the same judgment, so comparing whole rows would miss them.
- `keep="first"` keeps the first copy and drops the later ones.
- `drop_duplicates` does **not** modify the table in place. It returns a new DataFrame, so the result must be caught with `df = ...`. Forgetting the assignment is a classic pandas bug where the code runs and nothing changes.

### clean_split

```python
def clean_split(name: str, filename: str) -> None:
    print(f"--- Cleaning {name} ---")
    df = load_split(filename)

    if name == "train":
        df = drop_duplicates(df)

    out_path = PROCESSED_DIR / f"{name}_clean.csv"
    df.to_csv(out_path, index=False)
    print(f"--- {name} cleaned and saved to {out_path} ---")
```

This is the **orchestrator**: the function that calls the others in order (load, dedupe, save).

- `if name == "train"`: dedupe runs only on train, because Phase 1 found duplicates only there.
- `f"{name}_clean.csv"` is an **f-string**: the value of `name` is inserted into the text, giving `train_clean.csv`.
- `index=False` stops pandas from writing its internal row numbers as an extra column, which would pollute the CSV.
- `-> None` says the function returns nothing. It exists for its side effect (writing a file).

### The entry point

```python
if __name__ == "__main__":
    for name, filename in SPLITS.items():
        clean_split(name, filename)
```

- `__name__ == "__main__"` is true only when the file is run directly, not when another file imports it. This makes the functions reusable later without the script running itself.
- `SPLITS.items()` yields each (name, filename) pair for the loop.

---

## 6. Running the script

Run from the **repo root** (the folder containing `pyproject.toml`):

```bash
uv run python data/scripts/preprocess.py
```

`uv run` executes the command inside the project's virtual environment, so pandas is found.

Expected output:

```
--- Cleaning train ---
Dropped 5 duplicate rows.
--- train cleaned and saved to data/processed/train_clean.csv ---
--- Cleaning dev ---
--- dev cleaned and saved to data/processed/dev_clean.csv ---
--- Cleaning test ---
--- test cleaned and saved to data/processed/test_clean.csv ---
```

The "Dropped 5" line matches the Phase 1 EDA. No "Dropped" line appears for dev and test because the dedupe step does not run for them.

---

## 7. Verifying the output

A script that did not crash is not a script that worked. Read the written files back from disk:

```bash
uv run python -c "
import pandas as pd
for name in ['train', 'dev', 'test']:
    df = pd.read_csv(f'data/processed/{name}_clean.csv')
    print(name, len(df), 'rows | nulls:', df['text'].isna().sum(), '| dupes:', df['text'].duplicated().sum())
"
```

Expected results:

| Split | Rows | Nulls | Duplicates |
|---|---|---|---|
| train | 5077 (5082 minus 5) | 0 | 0 |
| dev | 2511 | 0 | 0 |
| test | 1517 | 0 | 0 |

Reading back from disk proves the files exist, are valid CSVs, and contain what you think they contain.

---

## 8. Linting lessons: F401 and I001

A **linter** reads code without running it and flags mistakes and sloppiness. This project uses **ruff**.

### F401: imported but unused

After deleting `fix_ocr_artifacts`, the line `import re` was left behind. Python does not complain about unused imports, so the script ran fine, but ruff flags it as `F401`. It also misleads readers, who assume regex is used somewhere in the file.

**Lesson:** when you delete code, ask what else existed only to support it: imports, helper functions, constants, comments.

### I001: import block un-sorted

After the fix, ruff reported `I001`. Imports are grouped into three blocks separated by blank lines:

1. **Standard library**: ships with Python (`pathlib`, `os`, `json`)
2. **Third-party**: installed with uv (`pandas`, `fastapi`)
3. **Local**: your own modules (`from app.core import config`)

Within each block, imports are sorted alphabetically. Think of a toolbox with labeled drawers: built-in tools, bought tools, tools you made.

### The command

```bash
uv run ruff check data/scripts/preprocess.py
```

Expected: `All checks passed!`. Run this yourself before every commit. `ruff check --fix` can auto-correct some rules, but always read `git diff` afterwards, because an auto-fixer edits files without asking.

---

## 9. Git workflow and the license rule

Phase 1 was committed straight to `main` by mistake. From Phase 2 onward the rule is **one branch per phase**:

```bash
git checkout -b feat/phase-cleaning
```

### The license rule

The ILDC dataset is licensed for academic use, and LexRAG is open-source and non-commercial. Only **pipeline code** is published, never dataset files. Before committing, verify the processed CSVs are ignored:

```bash
git check-ignore -v data/processed/train_clean.csv
```

If this prints nothing, the file is **not** ignored. Add `data/processed/` to `.gitignore` and commit that change first.

### Commit sequence

Stage files by name, never with `git add .`, so nothing unintended enters the commit:

```bash
git add data/scripts/preprocess.py
git diff --staged
git commit -m "feat: add preprocessing script to deduplicate train split"
git add docs/phase02.md
git commit -m "docs: add phase 02 cleaning documentation"
git push -u origin feat/phase-cleaning
```

One commit for code, one for docs. Then open a pull request into `main`.

---

## 10. Theory Q&A

**Q1. Why dedupe at all?**
Duplicate documents in a retrieval corpus waste top-k slots: one query can return the same judgment twice, pushing out a different relevant one. They also waste embedding compute in Phase 4.

**Q2. Why dedupe before chunking instead of after?**
Deduping whole documents is cheap (one comparison per document). After chunking, the same judgment would already be split into many pieces, and finding duplicate chunks is slower and messier.

**Q3. Why dedupe only on the `text` column?**
Two rows can carry different ids but the same judgment. The text is what gets embedded, so the text is what must be unique.

**Q4. Why never modify `data/raw/`?**
It is the only copy of the original. If cleaning has a bug, you rerun from pristine input. If raw were edited in place, a bug would be permanent.

**Q5. Why is a blanket regex replace dangerous?**
A regex has no understanding of meaning. It matches characters, not intent. `\bnumber\b` cannot tell a corrupted "not" from a correct "case number".

**Q6. What does `\b` do in a regex?**
It matches a word boundary, the position between a word character and a non-word character. `\bnumber\b` matches "number" but not "numbers".

**Q7. Why does `drop_duplicates` need `df = ...`?**
pandas operations generally return a new DataFrame instead of changing the original. Without catching the return value, the original stays unchanged.

**Q8. Why `index=False` in `to_csv`?**
Otherwise pandas writes its row numbers as an unnamed first column. Reading the file back would then show a stray extra column.

**Q9. Why `if __name__ == "__main__":`?**
So the file can be both run as a script and imported as a module without the cleaning running on import.

**Q10. Why is minimal cleaning acceptable for embeddings?**
An embedding compresses a whole chunk into one vector. A few typos inside hundreds of words barely shift it. Aggressive cleaning carries a real risk of destroying meaning, and the risk is larger than the benefit for rare noise.

---

## 11. Known limitations and open questions

- **Exact-match dedupe only.** Two judgments that differ by a single space or punctuation mark are not caught. Near-duplicate detection (hashing normalized text, or similarity-based methods) was not attempted.
- **Hardcoded train-only dedupe.** This reflects the Phase 1 finding. A different dataset version would need a fresh check.
- **Cross-split overlap not checked.** The script removes duplicates within train. It does not check whether the same judgment appears in both train and dev or test. Whether this matters depends on which splits end up as the RAG corpus and which are used for evaluation in Phase 8.
- **OCR artifact unfixed.** See Section 3.

---

## 12. What comes next

**Phase 3: Chunking.** This is the first phase with real design decisions. Judgments range from a few hundred to 87,530 words, so a single fixed chunk size will not work well. The phase has to decide how to split documents, how large chunks should be, and how much they should overlap, while keeping legal reasoning intact across chunk boundaries.
