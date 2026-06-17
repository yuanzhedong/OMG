# Known Test Issues (stale tests in the release)

While setting up reproduction we found three tests in `tests/` that fail against
the shipped `src/` because they were written against older interfaces/formats.
None are data problems; all are test-side staleness. Fixes are included in this
working tree. This file is a ready-to-file summary for upstream.

Environment: clean `make install`, Python 3.10, torch 2.6.0+cu124.
`python -m pytest -q` before fixes: **4 failed, 121 passed, 3 skipped**
(3 skips are the optional `finedance` data tests — expected).

## 1. `tests/generation/test_materialized_sampler.py` (2 failures)

**Symptom:** `TypeError: DistributedMaterializedShardSampler.__init__() got an
unexpected keyword argument 'block_size'`.

**Cause:** The test constructs the sampler with `block_size=` / `interleave_window=`
and asserts an *interleaving* order. The shipped sampler
(`src/omg/data/datamodule.py:186`) has signature `(dataset, *, seed=0)` and is
intentionally **shard-local** (docstring: "keeping consecutive samples
shard-local so a training batch can reuse the loaded shard"). The test reflects a
removed/older design.

**Fix:** Rewrote the test to the current contract — full coverage, one contiguous
run per shard span, determinism per seed, and disjoint whole-span partitioning
across ranks. (Did **not** change the sampler, to preserve training data order.)

## 2. `tests/generation/test_materialized_dataset.py` (1 failure)

**Symptom:** `FileNotFoundError: Materialized summary not found: .../train/summary.json`.

**Cause:** The test materializes a synthetic shard and writes a legacy
`index.jsonl`, but `MaterializedG1MotionDataset` (`src/omg/data/materialized.py:29-31`)
now requires a `summary.json` (`{samples, shards, shard_size}`).

**Fix:** Write `summary.json` alongside the synthetic shard.

## 3. `tests/robots/test_g1_kinematics_fast.py::test_prev_state_features_accept_pos_only_fk`

**Symptom (with a 125D stats file present):** `ValueError: Stats dim mismatch:
expected 123, got mean=125, std=125`.

**Cause:** The test builds `G1MotionRepresentation(num_prev_states=2)`, which uses
the constructor defaults `rotation_representation="quat"`, `feat_dim=123`, but the
default `stats_path` is the rot6d **125D** file `assets/stats/g1_125d_stats.json`.
quat→`3+4+29+87=123`, rot6d→`3+6+29+87=125`. This test would fail against the real
released 125D stats file too, independent of any synthetic data.

**Fix:** Construct the representation consistently with the released file:
`G1MotionRepresentation(num_prev_states=2, feat_dim=125, rotation_representation="rot6d")`.
The test only compares baseline-vs-fast FK paths, so the representation dims are
incidental — they just need to match the on-disk stats file.

> Note: this test (and others) requires a generated `assets/stats/g1_125d_stats.json`.
> The README states this file is not distributed and must be produced by
> `omg.cli.generation.compute_stats` after downloading OMG-Data.

## After fixes

`python -m pytest -q`: **126 passed, 3 skipped**.
