import logging

import pytest

from mace_scf.utils.logging import setup_logger


@pytest.fixture(autouse=True)
def _reset_root_logger():
    root = logging.getLogger()
    handlers, filters, level = list(root.handlers), list(root.filters), root.level
    yield
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    for filt in root.filters[:]:
        root.removeFilter(filt)
    for handler in handlers:
        root.addHandler(handler)
    for filt in filters:
        root.addFilter(filt)
    root.setLevel(level)


def test_log_all_ranks_writes_per_rank_files(tmp_path):
    setup_logger(
        level="INFO", tag="run", directory=tmp_path, rank=1, log_all_ranks=True
    )
    logging.info("hello from rank 1")
    logging.debug("debug from rank 1")

    main_log = (tmp_path / "run_rank1.log").read_text()
    debug_log = (tmp_path / "run_rank1_debug.log").read_text()

    assert "hello from rank 1" in main_log
    assert "debug from rank 1" not in main_log
    assert "hello from rank 1" in debug_log
    assert "debug from rank 1" in debug_log


def test_log_all_ranks_keeps_non_zero_rank_off_console(tmp_path, capsys):
    setup_logger(
        level="INFO", tag="run", directory=tmp_path, rank=1, log_all_ranks=True
    )
    logging.info("should not reach stdout")

    assert "should not reach stdout" not in capsys.readouterr().out


def test_log_all_ranks_false_falls_back_to_rank_zero_only_filter(tmp_path):
    setup_logger(
        level="INFO", tag="run", directory=tmp_path, rank=1, log_all_ranks=False
    )
    logging.info("dropped by mace.tools.setup_logger's rank filter")

    # mace.tools.setup_logger's rank==0 filter drops the record before any
    # handler emits it; the file still gets created (FileHandler opens on
    # construction) but stays empty, and no per-rank file is created at all.
    assert (tmp_path / "run.log").read_text() == ""
    assert not (tmp_path / "run_rank1.log").exists()
