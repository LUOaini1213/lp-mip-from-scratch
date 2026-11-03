# Changelog

## 2026-11-03 — review fixes

- `bench/bench_vsp.py --hunt 6000` now also re-verifies which of the 6 fractional-vertex
  seeds have a positive bound gap (5 of them; seed 670 is fractional with a 0.00 % gap).
  The 5 seeds are frozen in `vsp.FRACTIONAL_SEEDS_WITH_GAP` and written to
  `results/vsp.json` (`hunt.gap_seeds`). The README now says "fractional LP vertex
  returned by this simplex" where it used to say "fractional relaxation".
- `tests/test_readme_numbers.py`: every table in the README except the speed table
  (which `bench_all.py --check` already renders and compares) and every number in the
  English summary is compared with `results/bench.json` / `results/vsp.json`.
- `requirements.txt` is ASCII-only, so `pip < 23.3` on a GBK-locale Windows can read it.
- CI also runs on `windows-latest`, and an `unpinned` job runs the tests, both `--check`
  commands and both random searches on Python 3.12 / 3.13 with the latest numpy and scipy.
- Docstrings no longer point at files that do not exist (`bench/bench_pivot.py`,
  `bench/bench_formulations.py`, `problems/cycling.py`); "two orders of magnitude" is
  now "one order (1 node vs 21-31)", which is what `results/bench.json` shows.
- Removed a dead link (NCSU column-generation PDF, HTTP 404).

## 2026-10-12 — finite iteration budgets are not infeasibility certificates

Moved here from the top of the README; it replaces `docs/budget-status-checks.json`.

- Simplex phase I that runs out of its iteration budget returns `iteration_limit`; it no
  longer reads an unconverged artificial objective as proof of infeasibility. The same
  holds after the automatic restart with Bland's rule.
- Branch-and-bound that meets an LP `iteration_limit` (or the legacy `cycling` status)
  returns `lp_limit`: the incumbent is kept, global optimality is not claimed and the
  gap is unknown (`NaN`), not 0. Callers must check `status` before using a solution.
- An unbounded LP relaxation returns `lp_unbounded`. It proves neither that the integer
  model is infeasible nor that it has an integer unbounded ray.
- A node-budget stop and every other result without a proven optimum also report no
  finite gap.
- Regression tests: `tests/test_iteration_limits.py` (12 tests). The fix did not change
  `results/bench.json` or `results/vsp.json` (same blobs before and after: `45f11c4`,
  `64270d4`).
