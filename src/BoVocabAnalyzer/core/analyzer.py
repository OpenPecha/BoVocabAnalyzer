"""Botok-based vocabulary analysis and top-level orchestration."""

from pathlib import Path

from botok import WordTokenizer
from botok.vars import ChunkMarkers, WordMarkers

from BoVocabAnalyzer.core.arpa_parser import extract_vocab_from_arpa
from BoVocabAnalyzer.core.models import (
    ModelInfo,
    TokenInfo,
    VocabEntry,
    WordResult,
)
from BoVocabAnalyzer.utils.hf_utils import download_arpa_from_hf
from BoVocabAnalyzer.utils.report import (
    save_detail_tsv,
    save_summary_report,
    save_vocab_to_tsv,
)

# ---------------------------------------------------------------------------
# Chunk-type helpers
# ---------------------------------------------------------------------------

_CHUNK_LABEL: dict[int, str] = {
    **{m.value: m.name for m in WordMarkers},
    **{m.value: m.name for m in ChunkMarkers},
}


def _label(chunk_type: int | None) -> str:
    """Return a human-readable label for a botok chunk_type int."""
    if chunk_type is None:
        return "NONE"
    return _CHUNK_LABEL.get(chunk_type, str(chunk_type))


def _is_valid_word(chunk_type: int | None) -> bool:
    """Check whether *chunk_type* represents a valid Tibetan word."""
    return chunk_type == WordMarkers.WORD


# ---------------------------------------------------------------------------
# Single-word analysis
# ---------------------------------------------------------------------------


def analyse_word(wt: WordTokenizer, word: str, rank: int) -> WordResult:
    """Tokenize *word* with botok and return a structured result.

    Args:
        wt: An initialised botok WordTokenizer.
        word: The vocabulary entry to analyse.
        rank: Positional rank (from the vocab file).

    Returns:
        A WordResult with validity and per-token details.
    """
    try:
        tokens = wt.tokenize(word, split_affixes=True)
    except Exception as exc:  # noqa: BLE001
        return WordResult(
            rank=rank,
            word=word,
            is_valid=False,
            category="ERROR",
            num_tokens=0,
            error=str(exc),
        )

    token_infos: list[TokenInfo] = []
    for tok in tokens:
        token_infos.append(
            TokenInfo(
                text=tok.text,
                text_cleaned=tok.text_cleaned or "",
                chunk_type=_label(tok.chunk_type),
                pos=tok.pos or "",
                lemma=tok.lemma or "",
                senses=" | ".join(
                    ", ".join(f"{k}:{v}" for k, v in s.items())
                    for s in (tok.senses or [])
                ),
                skrt=bool(tok.skrt),
            )
        )

    # Decide overall validity & category
    if len(tokens) == 0:
        category = "EMPTY"
        is_valid = False
    elif len(tokens) == 1:
        category = _label(tokens[0].chunk_type)
        is_valid = tokens[0].chunk_type != WordMarkers.NON_WORD
    else:
        labels = "+".join(_label(t.chunk_type) for t in tokens)
        category = f"MULTI_TOKEN({labels})"
        is_valid = False

    return WordResult(
        rank=rank,
        word=word,
        is_valid=is_valid,
        category=category,
        num_tokens=len(tokens),
        tokens=token_infos,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _resolve_source(arpa_source: Path | str) -> ModelInfo:
    """Resolve *arpa_source* into a :class:`ModelInfo`.

    Args:
        arpa_source: Either a local :class:`~pathlib.Path` to an ``.arpa``
            file **or** a Hugging Face repo id string (e.g. ``"org/model"``).

    Returns:
        A populated :class:`ModelInfo`.

    Raises:
        FileNotFoundError: If the local file doesn't exist or no ``.arpa``
            file is found on Hugging Face.
    """
    if isinstance(arpa_source, Path):
        if not arpa_source.exists():
            raise FileNotFoundError(f"ARPA file not found: {arpa_source}")
        return ModelInfo(
            name=arpa_source.stem,
            arpa_path=arpa_source,
            source=str(arpa_source),
        )
    # Treat as HuggingFace repo id
    return download_arpa_from_hf(arpa_source)


def run_analysis(
    arpa_source: Path | str,
    detail_path: Path | None = None,
    summary_path: Path | None = None,
    vocab_path: Path | None = None,
) -> list[WordResult]:
    """Run the full vocabulary extraction and botok validation pipeline.

    Args:
        arpa_source: A local :class:`~pathlib.Path` to an ``.arpa`` file
            **or** a Hugging Face repo id string (e.g. ``"org/model"``).
        detail_path: Optional path for the per-word detail TSV.  Defaults to
            ``<model_name>_detail.tsv``.
        summary_path: Optional path for the human-readable summary report.
            Defaults to ``<model_name>_summary.txt``.
        vocab_path: Optional path to save the extracted vocab list.  If
            *None*, the vocab TSV is not written.

    Returns:
        The list of :class:`WordResult` objects produced by the analysis.
    """
    # 1. Resolve source → ModelInfo
    model_info = _resolve_source(arpa_source)
    print(f"Model : {model_info.name}")
    print(f"Source: {model_info.source}\n")

    # 2. Extract vocabulary from ARPA
    print("Extracting vocabulary from ARPA file …")
    entries = extract_vocab_from_arpa(model_info.arpa_path)
    print(f"  {len(entries)} vocab entries extracted.\n")

    if vocab_path is not None:
        save_vocab_to_tsv(entries, vocab_path)

    # 3. Analyse each word with botok
    print("Initialising botok WordTokenizer …")
    wt = WordTokenizer()
    print("  Ready.\n")

    words = [e.word for e in entries]
    total = len(words)

    print("Analysing words …")
    results: list[WordResult] = []
    for idx, word in enumerate(words, 1):
        results.append(analyse_word(wt, word, rank=idx))
        if idx % 5000 == 0 or idx == total:
            valid_so_far = sum(1 for r in results if r.is_valid)
            print(
                f"  [{idx:>{len(str(total))}}/{total}]  "
                f"valid so far: {valid_so_far}"
            )

    # 4. Write reports
    print()
    _detail = detail_path or Path(f"{model_info.name}_detail.tsv")
    _summary = summary_path or Path(f"{model_info.name}_summary.txt")

    save_detail_tsv(results, _detail, model_info)
    save_summary_report(results, _summary, model_info)

    valid = sum(1 for r in results if r.is_valid)
    print(
        f"\nDone. {valid}/{total} words are valid "
        f"({100 * valid / total:.2f}%)."
    )

    return results
