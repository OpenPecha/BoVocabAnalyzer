"""Shared data models for BoVocabAnalyzer."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelInfo:
    """Metadata about the language model being analysed.

    Attributes:
        name: Human-readable model name (e.g. repo id or file stem).
        arpa_path: Local path to the ARPA file (after download if remote).
        source: Original source string — a file path or HuggingFace repo id.
    """

    name: str
    arpa_path: Path
    source: str


@dataclass
class VocabEntry:
    """A single vocabulary item with its KenLM log-probability and approximate frequency."""

    word: str
    log_prob: float  # log10 probability from KenLM
    frequency: float  # 10^log_prob (relative frequency)


@dataclass
class TokenInfo:
    """Details extracted from a single botok Token."""

    text: str
    text_cleaned: str
    chunk_type: str
    pos: str
    lemma: str
    senses: str
    skrt: bool


@dataclass
class WordResult:
    """Validation result for a single vocabulary word."""

    rank: int
    word: str
    is_valid: bool
    category: str  # WORD / NO_POS / NON_WORD / PUNCT / MULTI_TOKEN / …
    num_tokens: int
    tokens: list[TokenInfo] = field(default_factory=list)
    error: str = ""
