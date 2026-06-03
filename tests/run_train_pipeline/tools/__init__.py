"""Utilities for maintaining run_train metric references.

Typical workflow:

1. Run the regression tests. They always write latest summaries/logs to
   tests/run_train_pipeline/actual/.

   python -m pytest -q tests/run_train_pipeline --basetemp tests/run_train_pipeline/test_outputs -s

2. Inspect concise differences.

   python -m tests.run_train_pipeline.tools.compare_expected

3. Accept deliberate changes.

   python -m tests.run_train_pipeline.tools.update_expected --models MACE FixedPoint

Use --all with update_expected to promote every actual summary/log to expected.

To run only selected model cases while iterating:

   export RUN_TRAIN_MODELS="FixedChargeBaselinedMACE" <pytest command above>

To skip slow cases:

   RUN_TRAIN_SKIP_MODELS="FixedPoint LocalSplitCharges" <pytest command above>
"""
