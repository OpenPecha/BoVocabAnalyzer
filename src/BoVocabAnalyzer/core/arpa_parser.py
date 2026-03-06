"""Parse KenLM ARPA language model files and extract vocabulary."""

import math
from pathlib import Path

from BoVocabAnalyzer.core.models import VocabEntry


def extract_vocab_from_arpa(arpa_path: Path) -> list[VocabEntry]:
    """Extract vocabulary and frequencies from a KenLM ARPA file.

    Parses the \\1-grams section to get each word and its log10 probability,
    then converts to a relative frequency via 10^log_prob.

    Args:
        arpa_path: Path to the .arpa language model file.

    Returns:
        List of VocabEntry sorted by frequency (descending).

    Raises:
        FileNotFoundError: If the ARPA file does not exist.
        ValueError: If no unigrams are found in the file.
    """
    if not arpa_path.exists():
        raise FileNotFoundError(f"ARPA file not found: {arpa_path}")

    vocab: list[VocabEntry] = []
    in_unigrams = False

    with open(arpa_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line == "\\1-grams:":
                in_unigrams = True
                continue
            if line.startswith("\\") and in_unigrams:
                break  # reached \2-grams: or \end\
            if not in_unigrams or not line:
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            log_prob = float(parts[0])
            word = parts[1]
            frequency = math.pow(10, log_prob)

            vocab.append(VocabEntry(word=word, log_prob=log_prob, frequency=frequency))

    if not vocab:
        raise ValueError(f"No unigrams found in {arpa_path}")

    vocab.sort(key=lambda v: v.frequency, reverse=True)
    return vocab
