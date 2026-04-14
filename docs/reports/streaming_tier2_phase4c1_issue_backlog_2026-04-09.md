# Streaming Tier 2 Phase 4C.1 Issue Backlog

Use this file as the direct issue-opening source for `Phase 4C.1`.

This backlog exists to stop further planning churn. The issue order below is the execution order.

## 1. Benchmark: Add Canonical Chunked Open-World Asset Ladder

Priority: `P0`

Depends on: none

Unblocks:

- lane mapping
- large-world benchmark proof
- honest `real_chunked` classification

Problem:

The benchmark surface is honest now, but it still does not contain a benchmark-side chunked asset family that can prove large-world streaming. The streaming-shaped lanes still resolve to `test_splats.ply`, so they remain smoke-only evidence.

Scope:

- add three canonical benchmark-side chunked assets:
  - `open_world_corridor_20m`
  - `open_world_boundary_50m`
  - `open_world_city_100m`
- ensure the assets are genuinely chunked:
  - large total splat count
  - bounded visible working set
  - no monolithic single-file shortcut that bypasses residency behavior
- wire those assets through the canonical manifest path used by the benchmark runners
- keep the existing smoke/support assets unchanged

Primary files:

- `tests/fixtures/benchmark_asset_manifest.json`
- `tests/examples/godot/test_project_deck/tests/fixtures/benchmark_asset_manifest.json`
- `tests/runtime/prepare_synthetic_assets.py`
- any new asset-generation or asset-staging helpers

Non-goals:

- changing the canonical runtime streaming gate
- rewriting benchmark lane logic
- publishing large-world claims before the assets are proven

Validation:

```bash
python3 tests/runtime/prepare_synthetic_assets.py --check
python3 tests/runtime/check_benchmark_asset_paths.py
```

Done when:

- at least one benchmark lane can resolve to a real chunked asset through the canonical manifest path
- no existing smoke lane silently changes classification

## 2. Benchmark: Map Open-World Proof Ladder To Benchmark Lanes

Priority: `P0`

Depends on:

- `Benchmark: add canonical chunked open-world asset ladder`

Unblocks:

- metrics contract
- CI surface split
- real benchmark proof runs

Problem:

We have the right scenario shapes in the repo, but they are not yet turned into one explicit proof ladder for large-world streaming.

Scope:

- map the proof ladder to concrete lanes:
  - corridor return
  - budget-churn corridor
  - biome boundary crossing
  - city-block roam + soak
- reuse existing benchmark/runtime lane shapes before adding new scenes
- write one-sentence failure intent for each lane
- keep mixed-residency and multi-asset hub cases non-blocking for now

Primary files:

- `tests/examples/godot/test_project/scenes/benchmark_suite/benchmark_suite_lane.gd`
- `tests/runtime/test_world_streaming_gate.gd`
- `tests/runtime/test_gpu_streaming_eviction_churn_probe.gd`
- `tests/runtime/test_gpu_streaming_stress.gd`
- `docs/testing/benchmark-suite.md`

Non-goals:

- making hub or mixed-residency scenarios part of the blocking path
- changing Phase 4A / 4B regression coverage

Validation:

- lane roster and notes read cleanly from docs
- lane purpose matches the asset/evidence role emitted by the runners

Done when:

- every streaming-shaped lane has a declared proof role
- the lane ladder can be explained without branch history

## 3. Benchmark: Lock Large-World Streaming Proof Metrics And Thresholds

Priority: `P0`

Depends on:

- `Benchmark: map open-world proof ladder to benchmark lanes`

Unblocks:

- real run and reclassification
- defensible claim boundary

Problem:

The benchmark proof surface must be based on streaming behavior, not just average FPS. Right now the repo already exposes the right metrics, but the lane-level threshold contract is not yet locked.

Scope:

- define and document lane-specific thresholds for:
  - corridor return
  - boundary crossing
  - city roam + soak
- keep the proof metrics centered on:
  - `first_visible_ms`
  - `residency_ratio`
  - `frame_p95_ms`
  - `frame_p95_to_avg_ratio`
  - `queue_pressure_frames`
  - `no_progress_frames`
  - `scan_starved_frames`
  - `vram_cap_hit_frames`
  - `chunk_loads_per_frame`
  - `chunk_evictions_per_frame`
- make correctness failures distinguishable from transient machine noise
- extend runner output only if current JSON/report fields are insufficient

Primary files:

- `tests/runtime/run_benchmark.py`
- `tests/runtime/run_benchmark_suite.py`
- `tests/examples/godot/test_project/scenes/benchmark_suite/benchmark_suite_lane.gd`
- `tests/runtime/test_gpu_streaming_stress.gd`
- `docs/testing/benchmark-suite.md`

Non-goals:

- adding new generic perf dashboards
- turning one noisy machine run into a blocker

Validation:

- benchmark summaries clearly separate correctness failures from perf noise
- thresholds line up with lane purpose

Done when:

- the benchmark contract is auditable from docs and runner output
- reviewers can tell why a lane failed without reading branch history

## 4. CI: Split Large-World Streaming Proof Into Blocking And Evidence-Only Surfaces

Priority: `P1`

Depends on:

- `Benchmark: lock large-world streaming proof metrics and thresholds`

Unblocks:

- repeatable dev proof runs
- weekly or scheduled large-world evidence

Problem:

We need large-world proof surfaces, but we do not want to turn per-PR CI into a hundred-million soak queue. The runtime gate policy must stay simple.

Scope:

- keep `streaming-gpu-ci` as the only canonical blocking GPU-backed streaming gate
- add non-blocking proof surfaces:
  - `openworld-proof-dev`
  - `openworld-proof-weekly`
- suggested scope:
  - `openworld-proof-dev` = `20M corridor` + `50M boundary`
  - `openworld-proof-weekly` = `100M city roam + soak`
- align docs and workflow naming so there is no second competing “streaming gate”

Primary files:

- `tests/runtime/runtime_scenarios.json`
- `.github/workflows/gaussian_production_gates.yml`
- `.github/workflows/README.md`
- `tests/runtime/README.md`
- `docs/testing/benchmark-suite.md`

Non-goals:

- making `100M` soak part of per-PR CI
- replacing `streaming-gpu-ci`

Validation:

```bash
python3 tests/runtime/run_runtime_validation.py --list-profiles
```

Done when:

- there is one blocking streaming gate and two clearly non-blocking proof surfaces
- reviewers can tell regression gates from evidence collection runs

## 5. Benchmark: Run Open-World Proof Lanes And Reclassify Validated Chunked Evidence

Priority: `P1`

Depends on:

- `CI: split large-world streaming proof into blocking and evidence-only surfaces`

Unblocks:

- public large-world claim updates
- performance docs update
- Tier 2 Phase 4C.2

Problem:

We cannot honestly upgrade any lane to `real_chunked` until we have actual results on a Windows-capable runner with the canonical benchmark path.

Scope:

- run the proof lanes on a Windows-capable runner
- capture results for:
  - `20M corridor`
  - `50M boundary`
  - `100M city roam + soak`
- reclassify only lanes that have:
  - real chunked asset backing
  - stable results
  - threshold contract defined in docs
- leave every `test_splats.ply` lane classified as smoke/support only
- record hardware, VRAM budget, and resident chunk budget next to the result

Primary files:

- `tests/runtime/run_benchmark.py`
- `tests/output/benchmark_suite/<timestamp>/...`
- `docs/testing/benchmark-suite.md`
- `docs/performance/index.md`
- `docs/reports/streaming_tier2_execution_roadmap_2026-04-09.md`

Canonical commands:

```bash
python3 tests/runtime/run_runtime_validation.py --profile streaming-gpu-ci --skip-cpp --fail-on-skip
python3 tests/runtime/run_benchmark.py --profile performance --fail-fast
```

Non-goals:

- backfilling fake results from smoke-only lanes
- claiming hundred-million support from a single unstable run

Done when:

- at least one large-world lane is promoted to honest `real_chunked` evidence
- the public docs stop implying that smoke-only lanes are representative chunked proof

## 6. Docs: Lock Claim Boundary For Large-World Streaming

Priority: `P1`

Depends on:

- `Benchmark: run open-world proof lanes and reclassify validated chunked evidence`

Problem:

The project needs one explicit statement of what can be claimed after `20M`, `50M`, and `100M` proof milestones, and all wording must stay resident/visible-budget based.

Scope:

- after `20M corridor` passes, allow only:
  - `large chunked worlds with tens of millions total splats validated`
- after `50M boundary` passes, allow only:
  - `large-world boundary-crossing streaming validated under representative motion`
- after `100M city roam + soak` passes, allow:
  - `hundred-million-total open-world streaming validated on reference hardware`
- update wording so nothing says or implies:
  - `renders hundred millions`
  - `all scene splats resident`
  - `all scene splats visible`

Primary files:

- `docs/testing/benchmark-suite.md`
- `tests/runtime/README.md`
- `.github/workflows/README.md`
- `docs/performance/index.md`
- `docs/reports/streaming_tier2_execution_roadmap_2026-04-09.md`

Non-goals:

- changing the technical proof itself
- widening the claim beyond the validated hardware/profile envelope

Done when:

- docs, workflow notes, and benchmark notes all use the same claim boundary
- large-world claims are tied to proof milestones, not optimism

## Opening Order

1. `Benchmark: add canonical chunked open-world asset ladder`
2. `Benchmark: map open-world proof ladder to benchmark lanes`
3. `Benchmark: lock large-world streaming proof metrics and thresholds`
4. `CI: split large-world streaming proof into blocking and evidence-only surfaces`
5. `Benchmark: run open-world proof lanes and reclassify validated chunked evidence`
6. `Docs: lock claim boundary for large-world streaming`
