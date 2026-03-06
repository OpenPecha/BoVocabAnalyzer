"""BoVocabAnalyzer — analyse Tibetan KenLM vocabulary with botok."""

__version__ = "0.1.0"

from BoVocabAnalyzer.core.analyzer import analyse_word, run_analysis
from BoVocabAnalyzer.core.arpa_parser import extract_vocab_from_arpa
from BoVocabAnalyzer.core.models import ModelInfo, TokenInfo, VocabEntry, WordResult
from BoVocabAnalyzer.utils.hf_utils import download_arpa_from_hf
from BoVocabAnalyzer.utils.report import (
    save_detail_tsv,
    save_summary_report,
    save_vocab_to_tsv,
)

__all__ = [
    "analyse_word",
    "download_arpa_from_hf",
    "extract_vocab_from_arpa",
    "ModelInfo",
    "run_analysis",
    "save_detail_tsv",
    "save_summary_report",
    "save_vocab_to_tsv",
    "TokenInfo",
    "VocabEntry",
    "WordResult",
]
