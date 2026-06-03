import argparse
import shutil

from tests.run_train_pipeline.harness import (
    INITIAL_MODEL_CASES,
    actual_log_path,
    actual_summary_path,
    expected_log_path,
    expected_summary_path,
)


def update_model(model: str):
    actual_summary = actual_summary_path(model)
    actual_log = actual_log_path(model)
    if not actual_summary.exists():
        raise FileNotFoundError(f"Missing actual summary for {model}: {actual_summary}")
    if not actual_log.exists():
        raise FileNotFoundError(f"Missing actual log for {model}: {actual_log}")

    expected_summary = expected_summary_path(model)
    expected_log = expected_log_path(model)
    expected_summary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(actual_summary, expected_summary)
    shutil.copyfile(actual_log, expected_log)
    print(f"updated {model}")
    print(f"  {actual_summary} -> {expected_summary}")
    print(f"  {actual_log} -> {expected_log}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Promote tests/run_train_pipeline/actual summaries/logs to expected "
            "references after reviewing differences."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--models", nargs="+")
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()

    models = list(INITIAL_MODEL_CASES) if args.all else args.models
    for model in models:
        update_model(model)


if __name__ == "__main__":
    main()
