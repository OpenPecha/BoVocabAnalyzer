"""Report generation for BoVocabAnalyzer results."""

import csv
from datetime import datetime
from pathlib import Path

from BoVocabAnalyzer.core.models import ModelInfo, VocabEntry, WordResult


def save_vocab_to_tsv(vocab: list[VocabEntry], output_path: Path) -> None:
    """Save extracted vocabulary words to a one-word-per-line TSV file.

    Args:
        vocab: List of VocabEntry items.
        output_path: Path for the output TSV file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in vocab:
            f.write(f"{entry.word}\n")

    print(f"Saved {len(vocab)} entries to {output_path}")


def save_detail_tsv(
    results: list[WordResult],
    output_path: Path,
    model_info: ModelInfo,
) -> None:
    """Write per-word detail rows to a TSV file.

    Args:
        results: List of WordResult objects.
        output_path: Destination TSV path.
        model_info: Model metadata (included as a column).
    """
    meta_fields = sorted(model_info.metadata.keys())
    fieldnames = [
        "model",
        *meta_fields,
        "rank",
        "word",
        "is_valid",
        "category",
        "num_tokens",
        "token_texts",
        "token_pos",
        "token_lemmas",
        "token_chunk_types",
        "has_sanskrit",
        "error",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        meta_row = {k: model_info.metadata[k] for k in meta_fields}
        for r in results:
            writer.writerow(
                {
                    "model": model_info.name,
                    **meta_row,
                    "rank": r.rank,
                    "word": r.word,
                    "is_valid": r.is_valid,
                    "category": r.category,
                    "num_tokens": r.num_tokens,
                    "token_texts": " | ".join(t.text for t in r.tokens),
                    "token_pos": " | ".join(t.pos for t in r.tokens),
                    "token_lemmas": " | ".join(t.lemma for t in r.tokens),
                    "token_chunk_types": " | ".join(
                        t.chunk_type for t in r.tokens
                    ),
                    "has_sanskrit": any(t.skrt for t in r.tokens),
                    "error": r.error,
                }
            )
    print(f"  Detail TSV saved → {output_path}")


def save_summary_report(
    results: list[WordResult],
    output_path: Path,
    model_info: ModelInfo,
) -> None:
    """Write a human-readable summary report.

    Args:
        results: List of WordResult objects.
        output_path: Destination text file path.
        model_info: Model metadata (printed in the report header).
    """
    total = len(results)
    valid = sum(1 for r in results if r.is_valid)
    invalid = total - valid

    # Count by category
    cat_counts: dict[str, int] = {}
    for r in results:
        cat_counts[r.category] = cat_counts.get(r.category, 0) + 1

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("=" * 70 + "\n")
        fh.write("  BOTOK VOCABULARY VALIDATION REPORT\n")
        fh.write(f"  Model : {model_info.name}\n")
        fh.write(f"  Source: {model_info.source}\n")
        for key, value in model_info.metadata.items():
            fh.write(f"  {key}: {value}\n")
        fh.write(f"  Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        fh.write("=" * 70 + "\n\n")

        fh.write("SUMMARY\n")
        fh.write("-" * 70 + "\n")
        fh.write(f"  Total words analysed : {total}\n")
        fh.write(
            f"  Valid words          : {valid}  ({100 * valid / total:.2f}%)\n"
        )
        fh.write(
            f"  Invalid words        : {invalid}  "
            f"({100 * invalid / total:.2f}%)\n\n"
        )

        fh.write("BREAKDOWN BY CATEGORY\n")
        fh.write("-" * 70 + "\n")
        for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
            fh.write(
                f"  {cat:<30} {cnt:>8}  ({100 * cnt / total:.2f}%)\n"
            )
        fh.write("\n")

        # List invalid words (first 200)
        invalids = [r for r in results if not r.is_valid]
        fh.write(
            f"INVALID WORDS (showing first 200 of {len(invalids)})\n"
        )
        fh.write("-" * 70 + "\n")
        for r in invalids[:200]:
            tok_detail = ", ".join(
                f"'{t.text}'({t.chunk_type})" for t in r.tokens
            )
            fh.write(
                f"  [{r.rank:>6}] {r.word:<25} → {r.category:<20} "
                f"tokens: {tok_detail}\n"
            )
        if len(invalids) > 200:
            fh.write(f"  ... and {len(invalids) - 200} more\n")

    print(f"  Summary report saved → {output_path}")
