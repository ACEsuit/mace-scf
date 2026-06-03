from pathlib import Path
import importlib.util


def _load_plot_batch_losses_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "plot_batch_losses.py"
    )
    spec = importlib.util.spec_from_file_location("plot_batch_losses", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


plot_batch_losses = _load_plot_batch_losses_module()
compute_ema = plot_batch_losses.compute_ema
default_output_path = plot_batch_losses.default_output_path
filter_records = plot_batch_losses.filter_records
infer_fit_name = plot_batch_losses.infer_fit_name
parse_loss_log = plot_batch_losses.parse_loss_log
plot_losses = plot_batch_losses.plot_losses


def _write_log(path: Path):
    path.write_text(
        "\n".join(
            [
                "2026-02-25 11:59:19.214 DEBUG: loss breakdown: energy_per_atom: 0.3, forces: 0.1, ",
                "2026-02-25 11:59:19.314 DEBUG: loss breakdown: energy_per_atom: 0.2, forces: 0.05, dipole_per_atom: 0.4, ",
                "2026-02-25 11:59:20.000 INFO: Epoch 0: loss=1.2345, RMSE_E_per_atom=1.0 meV, RMSE_F=2.0 meV / A",
                "2026-02-25 12:00:19.214 DEBUG: loss breakdown: energy_per_atom: 0.1, forces: 0.02, ",
                "2026-02-25 12:00:20.000 INFO: Epoch 1: loss=0.9876, RMSE_E_per_atom=0.9 meV, RMSE_F=1.8 meV / A",
            ]
        )
        + "\n"
    )
    return path


def test_infer_fit_name_and_default_output_path():
    log_path = Path("logs/demo_linearize_solve_run-123_debug.log")

    assert infer_fit_name(log_path) == "demo_linearize_solve"
    assert default_output_path(log_path) == Path("demo_linearize_solve.png")


def test_parse_loss_log_assigns_epochs_from_epoch_lines(tmp_path):
    log_path = _write_log(tmp_path / "demo_run-123_debug.log")

    records, epoch_boundaries = parse_loss_log(log_path)

    assert [record["epoch"] for record in records] == [0, 0, 1]
    assert [record["batch_index"] for record in records] == [0, 1, 2]
    assert epoch_boundaries == [(0, 2), (1, 3)]
    assert records[1]["components"]["dipole_per_atom"] == 0.4


def test_filter_records_by_epoch():
    records = [
        {"batch_index": 0, "epoch": 0, "components": {"forces": 0.1}},
        {"batch_index": 1, "epoch": 1, "components": {"forces": 0.2}},
        {"batch_index": 2, "epoch": 2, "components": {"forces": 0.3}},
    ]

    filtered = filter_records(records, min_epoch=1, max_epoch=1)

    assert filtered == [records[1]]


def test_compute_ema():
    ema = compute_ema([1.0, 2.0, 3.0], decay=0.5)

    assert ema == [1.0, 1.5, 2.25]


def test_plot_losses_writes_output(tmp_path):
    log_path = _write_log(tmp_path / "demo_run-123_debug.log")
    records, epoch_boundaries = parse_loss_log(log_path)
    output_path = tmp_path / "plot.png"

    plot_losses(records, epoch_boundaries, output_path, "demo")

    assert output_path.exists()
    assert output_path.stat().st_size > 0
