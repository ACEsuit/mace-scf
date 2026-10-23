import logging
import os
import sys
from typing import Optional, Union

import mace.tools


def setup_logger(
    level: Union[int, str] = logging.INFO,
    tag: Optional[str] = None,
    directory: Optional[str] = None,
    rank: int = 0,
    log_all_ranks: bool = False,
):
    """Like mace.tools.setup_logger, but with an opt-in mode that writes each rank's log records to its own file instead of silencing non-zero ranks."""
    if not log_all_ranks:
        return mace.tools.setup_logger(level=level, tag=tag, directory=directory, rank=rank)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create console handler
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    if rank != 0:
        # so only rank 0 writes to stdout
        ch.addFilter(lambda _: False)
    logger.addHandler(ch)

    if directory is not None and tag is not None:
        os.makedirs(name=directory, exist_ok=True)
        rank_tag = f"{tag}_rank{rank}"

        # Create file handler for non-debug logs
        main_log_path = os.path.join(directory, f"{rank_tag}.log")
        fh_main = logging.FileHandler(main_log_path)
        fh_main.setLevel(level)
        fh_main.setFormatter(formatter)
        logger.addHandler(fh_main)

        # Create file handler for debug logs
        debug_log_path = os.path.join(directory, f"{rank_tag}_debug.log")
        fh_debug = logging.FileHandler(debug_log_path)
        fh_debug.setLevel(logging.DEBUG)
        fh_debug.setFormatter(formatter)
        logger.addHandler(fh_debug)
