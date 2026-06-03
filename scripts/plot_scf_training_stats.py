#!/usr/bin/env python

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPOCH_RE = re.compile(r"Epoch\s+(\d+):\s+loss=")
STEP_RE = re.compile(r"step\s+(\d+), .*abs_change[s]?=([-+0-9.eE]+)")
CONVERGED_RE = re.compile(r"(?:SCF\s+)?converged at step\s+(\d+).*abs_change[s]?=([-+0-9.eE]+)", re.IGNORECASE)
DIVERGED_RE = re.compile(r"SCF diverged at step\s+(\d+).*abs_change[s]?=([-+0-9.eE]+)", re.IGNORECASE)
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} ")


def infer_fit_name(log_path: Path) -> str:
    stem = log_path.stem
    if stem.endswith("_debug"):
        stem = stem[: -len("_debug")]
    run_index = stem.rfind("_run-")
    if run_index != -1:
        stem = stem[:run_index]
    return stem


def default_output_path(log_path: Path) -> Path:
    return Path(f"{infer_fit_name(log_path)}_scf_stats.png")


def read_logical_lines(log_path: Path):
    logical_lines = []
    current_line = None
    for raw_line in log_path.read_text().splitlines():
        if TIMESTAMP_RE.match(raw_line):
            if current_line is not None:
                logical_lines.append(current_line)
            current_line = raw_line
        else:
            if current_line is None:
                current_line = raw_line
            else:
                current_line = current_line + " " + raw_line.strip()
    if current_line is not None:
        logical_lines.append(current_line)
    return logical_lines


def _finalize_record(records, current_record):
    if current_record is None:
        return None
    if current_record["final_step"] is None and current_record["last_step"] is not None:
        current_record["final_step"] = current_record["last_step"]
    if (
        current_record["final_abs_change"] is None
        and current_record["last_abs_change"] is not None
    ):
        current_record["final_abs_change"] = current_record["last_abs_change"]
    records.append(current_record)
    return None


def parse_scf_log(log_path: Path):
    current_epoch = 0
    current_record = None
    records = []
    solve_index = 0
    epoch_boundaries = []

    for line in read_logical_lines(log_path):
        epoch_match = EPOCH_RE.search(line)
        if epoch_match is not None:
            finished_epoch = int(epoch_match.group(1))
            epoch_boundaries.append((finished_epoch, solve_index))
            current_epoch = finished_epoch + 1
            continue

        converged_match = CONVERGED_RE.search(line)
        if converged_match is not None:
            if current_record is None:
                current_record = {
                    "solve_index": solve_index,
                    "epoch": current_epoch,
                    "status": "converged",
                    "last_step": None,
                    "last_abs_change": None,
                    "final_step": None,
                    "final_abs_change": None,
                }
                solve_index += 1
            current_record["status"] = "converged"
            current_record["final_step"] = int(converged_match.group(1))
            current_record["final_abs_change"] = float(converged_match.group(2))
            current_record = _finalize_record(records, current_record)
            continue

        diverged_match = DIVERGED_RE.search(line)
        if diverged_match is not None:
            if current_record is None:
                current_record = {
                    "solve_index": solve_index,
                    "epoch": current_epoch,
                    "status": "diverged",
                    "last_step": None,
                    "last_abs_change": None,
                    "final_step": None,
                    "final_abs_change": None,
                }
                solve_index += 1
            current_record["status"] = "diverged"
            current_record["final_step"] = int(diverged_match.group(1))
            current_record["final_abs_change"] = float(diverged_match.group(2))
            current_record = _finalize_record(records, current_record)
            continue

        step_match = STEP_RE.search(line)
        if step_match is None:
            continue

        step_index = int(step_match.group(1))
        abs_change = float(step_match.group(2))
        if step_index == 0:
            current_record = _finalize_record(records, current_record)
            current_record = {
                "solve_index": solve_index,
                "epoch": current_epoch,
                "status": "unknown",
                "last_step": step_index,
                "last_abs_change": abs_change,
                "final_step": None,
                "final_abs_change": None,
            }
            solve_index += 1
        elif current_record is None:
            current_record = {
                "solve_index": solve_index,
                "epoch": current_epoch,
                "status": "unknown",
                "last_step": step_index,
                "last_abs_change": abs_change,
                "final_step": None,
                "final_abs_change": None,
            }
            solve_index += 1
        else:
            current_record["last_step"] = step_index
            current_record["last_abs_change"] = abs_change

    _finalize_record(records, current_record)
    return records, epoch_boundaries


def filter_records(records, min_epoch=None, max_epoch=None):
    return [
        record
        for record in records
        if record["final_step"] is not None
        and record["final_abs_change"] is not None
        and (min_epoch is None or record["epoch"] >= min_epoch)
        and (max_epoch is None or record["epoch"] <= max_epoch)
    ]


def _add_epoch_markers(ax, x_values, epoch_boundaries):
    min_solve = x_values[0]
    max_solve = x_values[-1]
    for epoch, boundary_solve in epoch_boundaries:
        if boundary_solve <= min_solve or boundary_solve > max_solve:
            continue
        ax.axvline(boundary_solve - 0.5, color="0.8", linestyle="--", linewidth=1.0)
        ax.text(
            boundary_solve - 0.5,
            1.0,
            f"epoch {epoch}",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=8,
            color="0.4",
        )


def plot_scf_stats(records, epoch_boundaries, output_path: Path, title: str):
    if not records:
        raise ValueError("No SCF step records found in the requested epoch range.")

    statuses = {"converged": "tab:blue", "diverged": "tab:red", "unknown": "tab:gray"}
    x_values = [record["solve_index"] for record in records]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for status, color in statuses.items():
        subset = [record for record in records if record["status"] == status]
        if not subset:
            continue
        subset_x = [record["solve_index"] for record in subset]
        subset_abs_change = [record["final_abs_change"] for record in subset]
        subset_steps = [record["final_step"] for record in subset]
        axes[0].scatter(subset_x, subset_abs_change, label=status, color=color, s=18)
        axes[1].scatter(subset_x, subset_steps, label=status, color=color, s=18)

    for ax in axes:
        _add_epoch_markers(ax, x_values, epoch_boundaries)

    axes[0].set_ylabel("Final abs_change")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].set_ylabel("Final SCF step")
    axes[1].set_xlabel("SCF solve index")
    axes[1].grid(True, alpha=0.25)

    fig.suptitle(title)
    if x_values:
        axes[1].set_xlim(min(x_values) - 0.5, max(x_values) + 0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Plot final-step SCF convergence stats from a MACE_SCF debug log."
    )
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--min_epoch", type=int, default=None)
    parser.add_argument("--max_epoch", type=int, default=None)
    parser.add_argument(
        "--name",
        type=Path,
        default=None,
        help="Output image path. Defaults to <fit_name>_scf_stats.png inferred from the log filename.",
    )
    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    records, epoch_boundaries = parse_scf_log(args.log_path)
    records = filter_records(records, min_epoch=args.min_epoch, max_epoch=args.max_epoch)
    output_path = args.name if args.name is not None else default_output_path(args.log_path)
    plot_scf_stats(
        records,
        epoch_boundaries,
        output_path,
        f"{infer_fit_name(args.log_path)} SCF training stats",
    )
    print(output_path)


if __name__ == "__main__":
    main()
