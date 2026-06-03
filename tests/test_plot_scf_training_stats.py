from pathlib import Path
import importlib.util


def _load_plot_scf_training_stats_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "plot_scf_training_stats.py"
    )
    spec = importlib.util.spec_from_file_location(
        "plot_scf_training_stats", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


plot_scf_training_stats = _load_plot_scf_training_stats_module()
default_output_path = plot_scf_training_stats.default_output_path
filter_records = plot_scf_training_stats.filter_records
infer_fit_name = plot_scf_training_stats.infer_fit_name
parse_scf_log = plot_scf_training_stats.parse_scf_log
plot_scf_stats = plot_scf_training_stats.plot_scf_stats


def _write_constant_fermi_log(path: Path):
    path.write_text(
        "\n".join(
            [
                "2026-04-17 16:51:03.543 DEBUG: step 0, total_charges=tensor([0.5]), abs_changes=0.0041",
                "2026-04-17 16:51:03.551 DEBUG: step 1, total_charges=tensor([0.4]), abs_changes=0.0027",
                "2026-04-17 16:51:03.801 DEBUG: SCF converged at step 38, total_charges=tensor([0.3]), abs_changes=8.393224404933699e-07",
                "2026-04-17 16:51:04.100 INFO: Epoch 4: loss=1.0, RMSE_E_per_atom=1.0 meV, RMSE_F=2.0 meV / A",
                "2026-04-17 16:51:44.464 DEBUG: step 0, total_charges=tensor([0.1]), abs_changes=0.0045",
                "2026-04-17 16:51:44.570 DEBUG: step 29, total_charges=tensor([0.1]), abs_changes=6.0e-06",
                "2026-04-17 16:51:44.871 DEBUG: SCF diverged at step 29, total_charges=tensor([12.0]), abs_changes=6.0e-06",
            ]
        )
        + "\n"
    )
    return path


def _write_constant_charge_log(path: Path):
    path.write_text(
        "\n".join(
            [
                "2026-04-15 18:51:38.173 DEBUG: step 1, fermi_levels=tensor([-4.1]), gradient=tensor([-0.1]), total_qs=tensor([0.06]), abs_change=0.0026",
                "2026-04-15 18:51:38.279 DEBUG: step 20, fermi_levels=tensor([-2.0]), gradient=tensor([-0.1]), total_qs=tensor([0.001]), abs_change=4.4e-05",
                "2026-04-15 18:51:38.428 DEBUG: converged at step 39, fermi_levels=tensor([-2.0]), total_charges=tensor([8e-06]), abs_changes=8.337233642964015e-07",
            ]
        )
        + "\n"
    )
    return path


def test_infer_fit_name_and_default_output_path():
    log_path = Path("logs/demo_linearize_solve_run-123_debug.log")

    assert infer_fit_name(log_path) == "demo_linearize_solve"
    assert default_output_path(log_path) == Path("demo_linearize_solve_scf_stats.png")


def test_parse_scf_log_supports_constant_fermi_and_divergence(tmp_path):
    log_path = _write_constant_fermi_log(tmp_path / "demo_run-123_debug.log")

    records, epoch_boundaries = parse_scf_log(log_path)

    assert len(records) == 2
    assert records[0]["status"] == "converged"
    assert records[0]["epoch"] == 0
    assert records[0]["final_step"] == 38
    assert records[1]["status"] == "diverged"
    assert records[1]["epoch"] == 5
    assert records[1]["final_abs_change"] == 6.0e-06
    assert epoch_boundaries == [(4, 1)]


def test_parse_scf_log_supports_constant_charge_format(tmp_path):
    log_path = _write_constant_charge_log(tmp_path / "quick_test_run-123_debug.log")

    records, epoch_boundaries = parse_scf_log(log_path)

    assert len(records) == 1
    assert records[0]["status"] == "converged"
    assert records[0]["final_step"] == 39
    assert records[0]["final_abs_change"] == 8.337233642964015e-07
    assert epoch_boundaries == []


def test_filter_records_by_epoch():
    records = [
        {"solve_index": 0, "epoch": 0, "status": "converged", "final_step": 10, "final_abs_change": 1e-3},
        {"solve_index": 1, "epoch": 1, "status": "diverged", "final_step": 20, "final_abs_change": 1e-2},
    ]

    filtered = filter_records(records, min_epoch=1, max_epoch=1)

    assert filtered == [records[1]]


def test_plot_scf_stats_writes_output(tmp_path):
    log_path = _write_constant_fermi_log(tmp_path / "demo_run-123_debug.log")
    records, epoch_boundaries = parse_scf_log(log_path)
    output_path = tmp_path / "scf_stats.png"

    plot_scf_stats(records, epoch_boundaries, output_path, "demo")

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_parse_records_keep_logged_terminal_step_index(tmp_path):
    log_path = _write_constant_fermi_log(tmp_path / "demo_run-123_debug.log")

    records, _ = parse_scf_log(log_path)

    assert records[0]["final_step"] == 38
    assert records[1]["final_step"] == 29
