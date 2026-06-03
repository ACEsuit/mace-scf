import argparse
import json
from pathlib import Path

from tests.run_train_pipeline.harness import (
    ACTUAL_DIR,
    EXPECTED_DIR,
    INITIAL_MODEL_CASES,
    SKIPPED_INITIAL_MODEL_CASES,
    actual_summary_path,
    expected_summary_path,
    summary_differences,
)


def _load(path: Path):
    return json.loads(path.read_text())


def _format_value(value):
    text = repr(value)
    if len(text) > 180:
        return text[:177] + "..."
    return text


def compare_model(model: str, *, max_diffs: int):
    expected_path = expected_summary_path(model)
    actual_path = actual_summary_path(model)
    if not expected_path.exists():
        return [(model, None, None, f"missing expected summary: {expected_path}")]
    if not actual_path.exists():
        return [(model, None, None, f"missing actual summary: {actual_path}")]

    diffs = summary_differences(_load(actual_path), _load(expected_path))
    if max_diffs > 0:
        diffs = diffs[:max_diffs]
    return diffs


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare tests/run_train_pipeline/actual summaries against expected "
            "summaries and print concise path-level differences."
        )
    )
    parser.add_argument("--models", nargs="+", default=list(INITIAL_MODEL_CASES))
    parser.add_argument(
        "--max-diffs",
        type=int,
        default=50,
        help="Maximum differences to print per model. Use 0 for no limit.",
    )
    args = parser.parse_args()

    any_diffs = False
    print(f"expected: {EXPECTED_DIR}")
    print(f"actual:   {ACTUAL_DIR}")
    for model in args.models:
        if model in SKIPPED_INITIAL_MODEL_CASES:
            print(f"\n{model}: skipped ({SKIPPED_INITIAL_MODEL_CASES[model]})")
            continue
        diffs = compare_model(model, max_diffs=args.max_diffs)
        if not diffs:
            print(f"\n{model}: no differences")
            continue

        any_diffs = True
        print(f"\n{model}: {len(diffs)} difference(s)")
        for path, actual, expected, reason in diffs:
            print(f"  {path}: {reason}")
            print(f"    expected: {_format_value(expected)}")
            print(f"    actual:   {_format_value(actual)}")

    raise SystemExit(1 if any_diffs else 0)


if __name__ == "__main__":
    main()
