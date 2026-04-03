"""Example usage of BoVocabAnalyzer.

Shows how to run the full vocabulary extraction + botok validation pipeline
from either a local ARPA file or a Hugging Face repository.
"""

from pathlib import Path

from BoVocabAnalyzer import run_analysis

# ---------------------------------------------------------------------------
# Option 1: Analyse local ARPA files (batch pipeline)
# ---------------------------------------------------------------------------
# Scans a ``models/`` directory for all .arpa files, then runs the full
# extraction + botok validation pipeline for each one.  Per-model outputs
# are written to ``vocabs/`` and ``report/<model_name>/``.

MODELS_DIR = Path("./data/models")
VOCABS_DIR = Path("./data/vocabs")
REPORTS_DIR = Path("./data/reports")

VOCABS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

arpa_files = sorted(MODELS_DIR.glob("*.arpa"))
if not arpa_files:
    raise FileNotFoundError(f"No .arpa files found in {MODELS_DIR}")

for arpa_path in arpa_files:
    model_name = arpa_path.stem
    report_dir = REPORTS_DIR / model_name
    report_dir.mkdir(parents=True, exist_ok=True)

    results = run_analysis(
        arpa_source=arpa_path,
        detail_path=report_dir / "botok_detail.tsv",
        summary_path=report_dir / "botok_summary.txt",
        vocab_path=VOCABS_DIR / f"{model_name}_vocab.tsv",
    )

# ---------------------------------------------------------------------------
# Option 2: Analyse from a Hugging Face repo
# ---------------------------------------------------------------------------
# Pass a repo id string (e.g. "org/repo-name").  The tool will auto-detect
# the .arpa file in the repo, download it, and run the analysis.

# results = run_analysis(
#     arpa_source="openpecha/bo-kenlm-model",
#     detail_path=Path("hf_detail.tsv"),
#     summary_path=Path("hf_summary.txt"),
#     vocab_path=Path("hf_vocab.tsv"),
# )

# ---------------------------------------------------------------------------
# The returned `results` is a list[WordResult] you can inspect further:
#
#   for r in results[:10]:
#       print(r.rank, r.word, r.is_valid, r.category)
# ---------------------------------------------------------------------------
