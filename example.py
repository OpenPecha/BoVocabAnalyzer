"""Example usage of BoVocabAnalyzer.

Shows how to run the full vocabulary extraction + botok validation pipeline
from either a local ARPA file or a Hugging Face repository.
"""

from pathlib import Path

from BoVocabAnalyzer import run_analysis

# ---------------------------------------------------------------------------
# Option 1: Analyse a local ARPA file
# ---------------------------------------------------------------------------
# Provide a pathlib.Path pointing to your .arpa language model.
# The model name in the report will be derived from the file stem.

# results = run_analysis(
#     arpa_source=Path("BoKenlm-botok-syl.arpa"),
#     detail_path=Path("local_detail.tsv"),
#     summary_path=Path("local_summary.txt"),
#     vocab_path=Path("local_vocab.tsv"),
# )

# ---------------------------------------------------------------------------
# Option 2: Analyse from a Hugging Face repo
# ---------------------------------------------------------------------------
# Pass a repo id string (e.g. "org/repo-name").  The tool will auto-detect
# the .arpa file in the repo, download it, and run the analysis.

results = run_analysis(
    arpa_source="openpecha/bo-kenlm-model",
    detail_path=Path("hf_detail.tsv"),
    summary_path=Path("hf_summary.txt"),
    vocab_path=Path("hf_vocab.tsv"),
)

# ---------------------------------------------------------------------------
# The returned `results` is a list[WordResult] you can inspect further:
#
#   for r in results[:10]:
#       print(r.rank, r.word, r.is_valid, r.category)
# ---------------------------------------------------------------------------
