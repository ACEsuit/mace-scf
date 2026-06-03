#!/usr/bin/env python

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOSS_BREAKDOWN_RE = re.compile(r"loss breakdown:\s*(.*)")
EPOCH_RE = re.compile(r"Epoch\s+(\d+):\s+loss=")
LOSS_ITEM_RE = re.compile(r"([A-Za-z0-9_]+):\s*([-+0-9.eE]+)")
DEFAULT_EMA_DECAY = 0.99


def infer_fit_name(log_path: Path) -> str:
    stem = log_path.stem
    if stem.endswith("_debug"):
        stem = stem[: -len("_debug")]
    run_index = stem.rfind("_run-")
    if run_index != -1:
        stem = stem[:run_index]
    return stem


def default_output_path(log_path: Path) -> Path:
    return Path(f"{infer_fit_name(log_path)}.png")


def parse_loss_log(log_path: Path):
    current_epoch = 0
    batch_index = 0
    records = []
    epoch_boundaries = []

    for line in log_path.read_text().splitlines():
        epoch_match = EPOCH_RE.search(line)
        if epoch_match is not None:
            finished_epoch = int(epoch_match.group(1))
            epoch_boundaries.append((finished_epoch, batch_index))
            current_epoch = finished_epoch + 1
            continue

        loss_match = LOSS_BREAKDOWN_RE.search(line)
        if loss_match is None:
            continue

        components = {
            name: float(value)
            for name, value in LOSS_ITEM_RE.findall(loss_match.group(1))
        }
        if not components:
            continue

        records.append(
            {
                "batch_index": batch_index,
                "epoch": current_epoch,
                "components": components,
            }
        )
        batch_index += 1

    return records, epoch_boundaries


def filter_records(records, min_epoch=None, max_epoch=None):
    return [
        record
        for record in records
        if (min_epoch is None or record["epoch"] >= min_epoch)
        and (max_epoch is None or record["epoch"] <= max_epoch)
    ]


def compute_ema(values, decay=DEFAULT_EMA_DECAY):
    ema_values = []
    ema = None
    for value in values:
        if value != value:
            ema_values.append(float("nan"))
            continue
        if ema is None:
            ema = value
        else:
            ema = decay * ema + (1.0 - decay) * value
        ema_values.append(ema)
    return ema_values


def _add_epoch_markers(ax, x_values, epoch_boundaries):
    min_batch = x_values[0]
    max_batch = x_values[-1]
    for epoch, boundary_batch in epoch_boundaries:
        if boundary_batch <= min_batch or boundary_batch > max_batch:
            continue
        ax.axvline(boundary_batch - 0.5, color="0.8", linestyle="--", linewidth=1.0)
        ax.text(
            boundary_batch - 0.5,
            1.0,
            f"epoch {epoch}",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=8,
            color="0.4",
        )


def plot_losses(records, epoch_boundaries, output_path: Path, title: str, decay=DEFAULT_EMA_DECAY):
    if not records:
        raise ValueError("No batch loss breakdown lines found in the requested epoch range.")

    component_names = sorted(
        {
            component_name
            for record in records
            for component_name in record["components"]
        }
    )
    x_values = [record["batch_index"] for record in records]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for component_name in component_names:
        y_values = [
            record["components"].get(component_name, float("nan")) for record in records
        ]
        axes[0].plot(x_values, y_values, label=component_name, linewidth=1.2, alpha=0.8)
        axes[1].plot(
            x_values,
            compute_ema(y_values, decay=decay),
            label=component_name,
            linewidth=1.5,
        )

    for ax in axes:
        _add_epoch_markers(ax, x_values, epoch_boundaries)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Loss component value")
    axes[0].set_title(title)
    axes[0].legend(loc="best", ncol=2)

    axes[1].set_xlabel("Batch step")
    axes[1].set_ylabel(f"EMA loss ({decay:.2f})")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Plot per-batch loss breakdown terms from a MACE_SCF debug log."
    )
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--min_epoch", type=int, default=None)
    parser.add_argument("--max_epoch", type=int, default=None)
    parser.add_argument(
        "--name",
        type=Path,
        default=None,
        help="Output image path. Defaults to <fit_name>.png inferred from the log filename.",
    )
    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    records, epoch_boundaries = parse_loss_log(args.log_path)
    records = filter_records(
        records, min_epoch=args.min_epoch, max_epoch=args.max_epoch
    )
    output_path = args.name if args.name is not None else default_output_path(args.log_path)
    plot_losses(
        records,
        epoch_boundaries,
        output_path=output_path,
        title=f"{infer_fit_name(args.log_path)} batch losses",
    )
    print(output_path)


if __name__ == "__main__":
    main()
