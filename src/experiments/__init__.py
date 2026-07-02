"""
One module per experiment (E0..E9).

The master notebook cells are thin wrappers that call
`src.experiments.eX_*.main()`. All logic lives here so that a bug fix only
needs `git pull` in the E0 setup cell - the notebook itself never changes.

Every main():
  1. checks its Drive DONE marker + required artifacts and skips instantly
     if the experiment already completed (so "Run all" is safe and free),
  2. re-stages any local data it needs after a runtime restart,
  3. ends with resultlog.log_run(...) and expstate.mark_done(...).
"""
