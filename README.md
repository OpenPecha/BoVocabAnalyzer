<h1 align="center">
  <br>
  <a href="https://openpecha.org"><img src="https://avatars.githubusercontent.com/u/82142807?s=400&u=19e108a15566f3a1449bafb03b8dd706a72aebcd&v=4" alt="OpenPecha" width="150"></a>
  <br>
</h1>

## BoVocabAnalyzer

A Python toolkit for extracting vocabulary from Tibetan KenLM language models (`.arpa` files) and validating each word with [botok](https://github.com/OpenPecha/Botok), the rule-based Tibetan tokenizer.

## Owner(s)

- [@kaldan007](https://github.com/kaldan007)

This repository is created under the scope of [The BDRC E-Text Corpus Project](https://www.bdrc.io/blog/2026/02/28/bdrc-launches-major-initiative-to-build-open-buddhist-datasets-for-ai/).

## Table of contents

<p align="center">
  <a href="#project-description">Project description</a> •
  <a href="#project-dependencies">Project dependencies</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#api-reference">API reference</a> •
  <a href="#project-structure">Project structure</a> •
  <a href="#contributing-guidelines">Contributing guidelines</a> •
  <a href="#how-to-get-help">How to get help</a> •
  <a href="#terms-of-use">Terms of use</a>
</p>
<hr>

## Project description

BoVocabAnalyzer takes a KenLM `.arpa` language model — either from a local file or a Hugging Face repository — and runs every unigram through the botok tokenizer to determine whether it is a valid Tibetan word. It produces:

- **Vocab TSV** — a flat list of all vocabulary words sorted by frequency.
- **Detail TSV** — one row per word with rank, validity, botok category, per-token POS/lemma/chunk-type breakdowns, Sanskrit detection, and any model metadata.
- **Summary report** — an overview with total/valid/invalid counts, a category breakdown, and the first 200 invalid words for manual inspection.

### How it works

1. **Extract** — Parses the `\1-grams:` section of an ARPA file to collect each word and its log-probability, converting to relative frequency via `10^log_prob`.
2. **Tokenize** — Feeds every word to `botok.WordTokenizer` (with affix splitting) and records token-level details (text, POS, lemma, chunk type, senses, Sanskrit flag).
3. **Classify** — Labels each word as valid (`WORD`), invalid (`NON_WORD`, `NO_POS`, `PUNCT`, `MULTI_TOKEN(…)`, etc.), or errored.
4. **Report** — Writes the vocab list, a per-word detail TSV, and a human-readable summary.


## Project dependencies

| Dependency | Purpose |
|---|---|
| Python >= 3.10 | Runtime |
| [botok](https://github.com/OpenPecha/Botok) | Tibetan word tokenization and validation |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) | Download `.arpa` models from Hugging Face |


## Installation

```bash
# Clone the repository
git clone https://github.com/OpenPecha/BoVocabAnalyzer.git
cd BoVocabAnalyzer

# Install in editable mode
pip install -e .

# Or with dev extras (pytest, pre-commit)
pip install -e ".[dev]"
```


## Usage

### Quick start — local ARPA file

```python
from pathlib import Path
from BoVocabAnalyzer import run_analysis

results = run_analysis(
    arpa_source=Path("path/to/model.arpa"),
    detail_path=Path("reports/botok_detail.tsv"),
    summary_path=Path("reports/botok_summary.txt"),
    vocab_path=Path("vocabs/model_vocab.tsv"),
)
```

### From a Hugging Face repository

Pass a repo id string instead of a `Path`. The tool auto-detects the `.arpa` file, downloads it, and runs the analysis:

```python
results = run_analysis(
    arpa_source="openpecha/bo-kenlm-model",
    detail_path=Path("hf_detail.tsv"),
    summary_path=Path("hf_summary.txt"),
)
```

### Batch processing

The included `example.py` scans a directory for all `.arpa` files and runs the pipeline on each one:

```bash
python example.py
```

It expects models in `./data/models/` and writes outputs to `./data/vocabs/` and `./data/reports/<model_name>/`.

### Inspecting results

`run_analysis` returns a `list[WordResult]` you can work with directly:

```python
for r in results[:10]:
    print(r.rank, r.word, r.is_valid, r.category)
```

### Companion metadata

Place a `<model_stem>-meta.txt` file next to the `.arpa` file (e.g. `BoKenlm-sp-meta.txt`) with `Key: Value` lines. The metadata is automatically picked up and included in reports.


## API reference

### Top-level function

| Function | Description |
|---|---|
| `run_analysis(arpa_source, detail_path?, summary_path?, vocab_path?)` | Full pipeline: extract vocab, tokenize with botok, write reports. Accepts a local `Path` or a Hugging Face repo id string. |

### Core

| Function / Class | Description |
|---|---|
| `analyse_word(wt, word, rank)` | Tokenize a single word with botok and return a `WordResult`. |
| `extract_vocab_from_arpa(arpa_path)` | Parse the `\1-grams:` section of an ARPA file into a list of `VocabEntry`, sorted by frequency descending. |

### Models (dataclasses)

| Class | Fields |
|---|---|
| `ModelInfo` | `name`, `arpa_path`, `source`, `metadata` |
| `VocabEntry` | `word`, `log_prob`, `frequency` |
| `TokenInfo` | `text`, `text_cleaned`, `chunk_type`, `pos`, `lemma`, `senses`, `skrt` |
| `WordResult` | `rank`, `word`, `is_valid`, `category`, `num_tokens`, `tokens`, `error` |

### Utilities

| Function | Description |
|---|---|
| `download_arpa_from_hf(repo_id)` | Download the first `.arpa` file from a Hugging Face repo and return a `ModelInfo`. |
| `save_vocab_to_tsv(vocab, output_path)` | Write vocabulary words to a one-word-per-line TSV. |
| `save_detail_tsv(results, output_path, model_info)` | Write per-word detail rows (rank, validity, tokens, POS, etc.) to a TSV. |
| `save_summary_report(results, output_path, model_info)` | Write a human-readable summary with category breakdown and invalid word list. |


## Project structure

```
BoVocabAnalyzer/
├── src/BoVocabAnalyzer/
│   ├── __init__.py            # Public API re-exports
│   ├── core/
│   │   ├── analyzer.py        # botok analysis + run_analysis orchestration
│   │   ├── arpa_parser.py     # ARPA file parsing / vocab extraction
│   │   └── models.py          # Dataclasses (ModelInfo, VocabEntry, TokenInfo, WordResult)
│   └── utils/
│       ├── hf_utils.py        # Hugging Face download helpers
│       └── report.py          # TSV + summary report writers
├── tests/
├── docs/
├── example.py                 # Batch-processing script
├── pyproject.toml
└── LICENSE
```


## Contributing guidelines

If you'd like to help out, check out our [contributing guidelines](/CONTRIBUTING.md).

Install dev dependencies and set up pre-commit hooks:

```bash
pip install -e ".[dev]"
pre-commit install
```

Run tests:

```bash
PYTHONPATH=src pytest
```


## How to get help

- File an issue.
- Email us at openpecha[at]gmail.com.
- Join our [discord](https://discord.com/invite/7GFpPFSTeA).


## Terms of use

BoVocabAnalyzer is licensed under the [MIT License](/LICENSE).

---

Developed as part of [The BDRC E-Text Corpus project](https://www.bdrc.io/blog/2026/02/28/bdrc-launches-major-initiative-to-build-open-buddhist-datasets-for-ai/).
