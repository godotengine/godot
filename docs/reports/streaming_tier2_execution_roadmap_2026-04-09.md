# Streaming Tier 2 Execution Roadmap

**Date:** 2026-04-09  
**Scope:** `feat/streaming-tier2-refactor` after the validated `Phase 4B` batching cluster  
**Purpose:** Record what Tier 2 streaming work is already complete, what remains, and what must be true before Tier 2 can be called done.

## Current Status

- `Phase 1` Stabilization foundation: complete
- `Phase 2` Hot-path simplification and runtime coverage: complete
- `Phase 3` Contracts and backend planning: complete
- `Phase 4A` Upload and sort ownership simplification: complete and Windows-validated
- `Phase 4B` Current batching/coalescing cluster: complete and Windows-validated
- Separate performance tracking remains out of the main execution path. The earlier `tier_1m` stress miss reproduced as a transient machine-load issue and passed on rerun.
- Known pre-existing editor-side crashes remain separate from Tier 2 streaming work.

## Completed Work

### Phase 1: Stabilization Foundation

Delivered:

- explicit residency state model and scheduler-consistent admission cleanup
- primary-asset residency handling fixes
- residency finalize-state cleanup
- eviction/accounting and atlas publication hardening
- authoritative runtime profile behavior over environment overrides
- explicit residency failure-state surfacing

Key checkpoints:

- `adc39747c3` `streaming: stabilize residency requests and atlas publication`
- `c6cfe994b8` `tests: add dedicated streaming GPU runtime profile`
- `00d25806f2` `tests: make runtime profiles authoritative over env`
- `f36b7286c4` `streaming: surface residency failure states`

### Phase 2: Hot-Path Simplification and Runtime Coverage

Delivered:

- long-walk and residency runtime coverage
- revision-driven asset membership and instance sync caching
- instance upload/rebuild gating on cached state
- route-policy cache and bootstrap retry cleanup
- recovery regression fixes

Key checkpoints:

- `78c6c10063` `tests: expand tier2 streaming runtime coverage`
- `669eb37303` `streaming: cache tier2 instance sync and upload paths`
- `1caefacd1a` `streaming: fix phase2 recovery regressions`

### Phase 3: Contracts and Backend Planning

Delivered:

- shared runtime fidelity policy
- typed world submission contracts
- restore-state preservation for same-owner and cross-world submission flows
- explicit `FrameBackendPlan` routing
- planner/executor stabilization
- zero-splat submission cleanup for routing and shared-content heuristics

Key checkpoints:

- `623571c811` `renderer: centralize runtime fidelity policy`
- `18a76e3ec5` `renderer: introduce world submission contracts`
- `e5e8564cbb` `scene_director: preserve world submission restore state`
- `f9da313524` `renderer: plan frame backend routing`
- `17596a5059` `renderer: stabilize phase3 backend and submission seams`
- `9ec2691332` `scene_director: ignore empty world submissions in shared-content queries`

### Phase 4A: Upload and Sort Ownership Simplification

Delivered:

- semaphore-agnostic pack-queue ownership
- explicit worker-vs-sync ownership boundary coverage
- Windows test-plumbing fixes required to validate the path
- registered Windows-reachable ownership tests
- centralized instance-count readback ownership for the sorter path

Key checkpoints:

- `bdd06116a6` `streaming: make pack queue ownership semaphore-agnostic`
- `441f81335c` `tests: cover pack queue ownership boundaries`
- `a7a1cba43d` `fix: resolve five MSVC build failures blocking Windows test validation`
- `ac1f085601` `fix: world submission respects route_policy instead of forcing resident`
- `9404f50388` `tests: register pack ownership boundary coverage`
- `1b1125c235` `sort: centralize instance-count readback ownership`

Validation status:

- Windows `tests=yes` build: passed
- targeted ownership doctests: passed
- `streaming-gpu-ci`: passed

### Phase 4B: Current Batching / Coalescing Cluster

Delivered:

- `TESTS_ENABLED` plumbing for doctest targets
- chunk-meta upload planning:
  - compact dirty spans stay incremental
  - fragmented or sufficiently large churn escalates to one full upload
- pack snapshot scratch-buffer reuse
- contiguous upload-slice coalescing for full-slot front-of-queue jobs within budget
- registered tests for incremental/full chunk-meta planning and upload coalescing behavior

Key checkpoints:

- `9f24e19e6b` `tests: define TESTS_ENABLED for doctest targets`
- `443b8d31d6` `streaming: coalesce fragmented chunk meta uploads`
- `e820ad3b6b` `streaming: reuse pack snapshot scratch buffers`
- `13ee319f46` `streaming: coalesce contiguous upload slices`

Validation status:

- Windows `tests=yes` build: passed
- Windows targeted batching/ownership doctests: passed
- targeted doctests:
  - `Sync pack rescue does not steal worker-owned pack jobs`
  - `Clearing instance pipeline inputs resets readback ownership state`
  - `Chunk meta upload planner keeps compact dirty spans incremental`
  - `Chunk meta upload planner escalates fragmented churn to a full upload`
  - `Upload coalescing planner batches contiguous full-slot uploads`
  - `Upload coalescing planner stops at partial or noncontiguous uploads`
- module lane: passed apart from the unchanged pre-existing editor crash
- `streaming-gpu-ci`: passed, including `GPU Streaming Stress`

## Remaining Roadmap

### Phase 4C: Benchmark and Operations Closeout

Status: next active work

Goal:

- close the remaining Tier 2 production-readiness gaps that are about operability and confidence, not architecture

Rules for this phase:

- do not reopen the architectural seams already stabilized in `Phase 1` through `Phase 4B`
- keep `streaming-gpu-ci` as the single canonical blocking GPU-backed streaming gate
- treat performance anomalies as separate investigations unless a `Phase 4C` change clearly causes them
- land this phase in narrow closeout checkpoints, not one large “ops cleanup” patch

#### Phase 4C.1: Benchmark and Gate Closeout

Goal:

- prove godotGS can stream hundred-million-total-splat worlds with a bounded resident and visible working set

This phase is not trying to prove “render hundreds of millions at once.”
The proof target is:

- very large total scene size
- bounded resident VRAM state
- bounded visible / dispatched working set
- stable forward progress during roam, revisit, and churn
- honest blocking-vs-evidence CI surfaces

Why this comes first:

- the benchmark and gate surfaces are the public and CI-facing proof that the refactor is production-ready
- if those surfaces remain ambiguous, the rest of the Tier 2 closeout work is hard to trust

Primary files:

- `tests/fixtures/benchmark_asset_manifest.json`
- `tests/runtime/prepare_synthetic_assets.py`
- `tests/runtime/run_benchmark.py`
- `tests/runtime/run_benchmark_suite.py`
- `tests/examples/godot/test_project/scenes/benchmark_suite/*`
- `tests/runtime/test_world_streaming_gate.gd`
- `docs/testing/benchmark-suite.md`
- `tests/runtime/README.md`
- `.github/workflows/gaussian_production_gates.yml`
- `.github/workflows/README.md`

Current architectural proof target:

- chunked streaming uses `65,536` splats per chunk
- scene totals are tracked separately from resident state
- resident capacity is budgeted independently of total scene size
- visible / rendered budget is determined by runtime dispatch, not total serialized world size

The missing gap after the earlier `4C.1` honesty slice is benchmark evidence, not architecture direction.
The current streaming-shaped benchmark lanes still resolve to `test_splats.ply` and therefore remain smoke-only evidence.

Required work:

##### Phase 4C.1A: Canonical Real Chunked Benchmark Asset Family

Goal:

- add one canonical benchmark asset ladder that is genuinely chunked and large enough to prove open-world streaming behavior

Required work:

- add a chunked benchmark asset family with at least:
  - `open_world_corridor_20m`
  - `open_world_boundary_50m`
  - `open_world_city_100m`
- require these assets to be genuinely chunked:
  - large total splat count
  - bounded visible working set
  - no monolithic single-asset shortcut
- keep the manifest honest:
  - smoke/support lanes stay `lightweight_smoke`
  - only lanes backed by the new chunked family are promoted to `real_chunked`
- keep runner policy aligned between:
  - `tests/runtime/run_benchmark.py`
  - `tests/runtime/run_benchmark_suite.py`

Exit:

- at least one benchmark lane is backed by a real chunked asset and classified `real_chunked`

##### Phase 4C.1B: Open-World Scenario Ladder

Goal:

- turn the current scenario shapes into actual proof lanes for large streamed worlds

Base surfaces:

- `tests/examples/godot/test_project/scenes/benchmark_suite/benchmark_suite_lane.gd`
- `tests/runtime/test_world_streaming_gate.gd`

Scenario ladder:

- corridor return:
  - preload, forward traversal, return-path reload, no dead zone after backtracking
- budget-churn corridor:
  - tight VRAM, eviction/reload stability, forward progress under pressure
- biome boundary crossing:
  - large visibility-set change with forward and reverse crossing
- city-block roam + soak:
  - sustained roaming, revisits, and long-run stability

Rules:

- do not make hub or mixed-residency scenarios blocking proof lanes yet
- keep those surfaces evidence-only until the single-world large-streaming path is solid

##### Phase 4C.1C: Large-World Proof Metrics

Goal:

- use metrics that prove streaming behavior, not just average FPS

Track and gate:

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
- fallback / route changes when present

Success means:

- visible content appears quickly
- revisits recover correctly
- queue starvation is not sustained
- VRAM-cap pinning is not sustained
- churn is bounded and not pathological

##### Phase 4C.1D: Blocking vs Evidence Surfaces

Goal:

- keep the runtime gate policy strict without overloading per-PR CI with hundred-million soak runs

Rules:

- keep `streaming-gpu-ci` as the single canonical blocking GPU-backed streaming gate
- add non-blocking proof surfaces for large-world evidence:
  - `openworld-proof-dev`
  - `openworld-proof-weekly`
- suggested scope:
  - `openworld-proof-dev`: `20M corridor` + `50M boundary`
  - `openworld-proof-weekly`: `100M city roam + soak`
- only promote a large-world lane to blocking after repeated stable runs on the same runner class
- do not put `100M` soak on per-PR CI

##### Phase 4C.1E: Architecture Ceilings To Watch While Proving

Goal:

- validate the real scaling ceilings while the large-world proof lanes come online

Likely bottlenecks:

- all-chunk scan cost in `streaming_visibility_controller.cpp`
- visible / dispatch scaling in `render_streaming_orchestrator.cpp`
- sort-capacity and visible-cap clamps in `render_streaming_orchestrator.cpp`

Reference scale:

- `100M` total splats at `65,536` per chunk is about `1,526` chunks
- `500M` total splats at `65,536` per chunk is about `7,629` chunks

That is why the proof target is open-world streaming behavior with a bounded working set, not raw rendered splat count.

Expected output:

- one canonical real chunked benchmark asset family
- at least one benchmark lane promoted from smoke/support evidence to genuine `real_chunked` evidence
- an open-world scenario ladder covering return, churn, boundary crossing, and soak behavior
- benchmark docs that distinguish:
  - real chunked evidence
  - deterministic synthetic support
  - lightweight smoke/support lanes
- one canonical description of:
  - the blocking streaming runtime gate
  - the non-blocking large-world proof surfaces

Validation:

```bash
python3 tests/runtime/run_runtime_validation.py --list-profiles
python3 tests/runtime/run_runtime_validation.py --profile streaming-gpu-ci --skip-cpp --fail-on-skip
python3 tests/runtime/run_benchmark.py --profile performance --fail-fast
```

Exit criteria:

- every streaming-relevant benchmark lane has an explicit asset/evidence classification
- there is no hidden manifest fallback that can make a streaming lane look more representative than it is
- at least one large-world benchmark lane is backed by a genuine chunked asset and classified `real_chunked`
- smoke-only streaming lanes remain documented as smoke-only unless and until they are truly upgraded
- `streaming-gpu-ci` remains the canonical blocking streaming gate in both docs and workflow surfaces
- large-world proof surfaces stay non-blocking until they are stable enough to promote deliberately

##### Phase 4C.1 Implementation Checklist

Use this checklist as the execution order for proving hundred-million-total-splat open-world streaming with a bounded resident and visible working set.

###### 4C.1 Checklist A: Asset Ladder

- [ ] Open issue: `Benchmark: add canonical chunked open-world asset ladder`
- [ ] Add three benchmark-side chunked assets:
  - [ ] `open_world_corridor_20m`
  - [ ] `open_world_boundary_50m`
  - [ ] `open_world_city_100m`
- [ ] Ensure each asset is genuinely chunked:
  - [ ] large total splat count
  - [ ] bounded visible working set
  - [ ] no monolithic single-file shortcut that bypasses residency behavior
- [ ] Add manifest entries in the main test project and Deck project where needed.
- [ ] Keep smoke/support assets unchanged so the old lanes remain comparable.

Primary files:

- `tests/fixtures/benchmark_asset_manifest.json`
- `tests/examples/godot/test_project_deck/tests/fixtures/benchmark_asset_manifest.json`
- `tests/runtime/prepare_synthetic_assets.py`
- any new asset-generation or staging helpers introduced for the chunked ladder

Done when:

- one real chunked benchmark asset resolves through the canonical manifest path
- no existing smoke lane changes classification by accident

###### 4C.1 Checklist B: Lane Mapping

- [ ] Open issue: `Benchmark: map open-world proof ladder to benchmark lanes`
- [ ] Map the scenario ladder to concrete lane ids:
  - [ ] corridor return
  - [ ] budget-churn corridor
  - [ ] biome boundary crossing
  - [ ] city-block roam + soak
- [ ] Reuse the current benchmark and runtime shapes before inventing new scenes.
- [ ] Keep mixed-residency and multi-asset hub cases non-blocking for now.
- [ ] Write one-sentence failure intent for each lane:
  - missing preload
  - failed revisit recovery
  - boundary dead zone
  - sustained churn / soak instability

Primary files:

- `tests/examples/godot/test_project/scenes/benchmark_suite/benchmark_suite_lane.gd`
- `tests/runtime/test_world_streaming_gate.gd`
- `tests/runtime/test_gpu_streaming_eviction_churn_probe.gd`
- `tests/runtime/test_gpu_streaming_stress.gd`
- `docs/testing/benchmark-suite.md`

Done when:

- each streaming-shaped lane has a declared proof role
- the lane roster can be explained without branch history or tribal knowledge

###### 4C.1 Checklist C: Metrics Contract

- [ ] Open issue: `Benchmark: lock large-world streaming proof metrics and thresholds`
- [ ] Keep the proof metrics centered on streaming behavior:
  - [ ] `first_visible_ms`
  - [ ] `residency_ratio`
  - [ ] `frame_p95_ms`
  - [ ] `frame_p95_to_avg_ratio`
  - [ ] `queue_pressure_frames`
  - [ ] `no_progress_frames`
  - [ ] `scan_starved_frames`
  - [ ] `vram_cap_hit_frames`
  - [ ] `chunk_loads_per_frame`
  - [ ] `chunk_evictions_per_frame`
- [ ] Define lane-specific pass/fail thresholds for:
  - [ ] corridor return
  - [ ] boundary crossing
  - [ ] city roam + soak
- [ ] Keep correctness and performance noise separate in docs and result summaries.
- [ ] Extend runner output only if current JSON/report fields are insufficient.

Primary files:

- `tests/runtime/run_benchmark.py`
- `tests/runtime/run_benchmark_suite.py`
- `tests/examples/godot/test_project/scenes/benchmark_suite/benchmark_suite_lane.gd`
- `tests/runtime/test_gpu_streaming_stress.gd`
- `docs/testing/benchmark-suite.md`

Done when:

- benchmark summaries make correctness failures distinguishable from transient machine noise
- threshold policy matches the purpose of each lane

###### 4C.1 Checklist D: CI Surface Split

- [ ] Open issue: `CI: split large-world streaming proof into blocking and evidence-only surfaces`
- [ ] Keep `streaming-gpu-ci` as the only canonical blocking GPU-backed streaming gate.
- [ ] Add non-blocking proof surfaces:
  - [ ] `openworld-proof-dev`
  - [ ] `openworld-proof-weekly`
- [ ] Keep suggested scope:
  - [ ] `openworld-proof-dev` = `20M corridor` + `50M boundary`
  - [ ] `openworld-proof-weekly` = `100M city roam + soak`
- [ ] Do not add `100M` soak to per-PR blocking CI.
- [ ] Align docs and workflow names so there is no second competing “streaming gate.”

Primary files:

- `tests/runtime/runtime_scenarios.json`
- `.github/workflows/gaussian_production_gates.yml`
- `.github/workflows/README.md`
- `tests/runtime/README.md`
- `docs/testing/benchmark-suite.md`

Done when:

- there is one blocking streaming gate and two clearly non-blocking proof surfaces
- reviewers can tell which runs are regression gates vs evidence collection

###### 4C.1 Checklist E: Real Run and Reclassification

- [ ] Open issue: `Benchmark: run open-world proof lanes and reclassify validated chunked evidence`
- [ ] Run the proof lanes on a Windows-capable runner with the canonical benchmark entrypoint.
- [ ] Capture results for:
  - [ ] `20M corridor`
  - [ ] `50M boundary`
  - [ ] `100M city roam + soak`
- [ ] Reclassify only the lanes that have:
  - [ ] real chunked asset backing
  - [ ] stable results
  - [ ] threshold contract defined in docs
- [ ] Leave every `test_splats.ply` lane classified as smoke/support only.
- [ ] Record hardware, VRAM budget, and resident chunk budget next to the result.

Primary files:

- `tests/runtime/run_benchmark.py`
- `tests/output/benchmark_suite/<timestamp>/...`
- `docs/testing/benchmark-suite.md`
- `docs/performance/index.md`
- `docs/reports/streaming_tier2_execution_roadmap_2026-04-09.md`

Canonical run commands:

```bash
python3 tests/runtime/run_runtime_validation.py --profile streaming-gpu-ci --skip-cpp --fail-on-skip
python3 tests/runtime/run_benchmark.py --profile performance --fail-fast
```

Done when:

- at least one large-world lane is promoted to honest `real_chunked` evidence
- the public docs stop implying that smoke-only lanes are representative chunked proof

###### 4C.1 Checklist F: Claim Boundary

- [ ] After `20M corridor` passes, allow only:
  - `large chunked worlds with tens of millions total splats validated`
- [ ] After `50M boundary` passes, allow only:
  - `large-world boundary-crossing streaming validated under representative motion`
- [ ] After `100M city roam + soak` passes, allow:
  - `hundred-million-total open-world streaming validated on reference hardware`
- [ ] Keep all wording resident/visible-budget based, never “renders hundred millions.”

###### Suggested Commit Order

1. add canonical chunked asset ladder and manifest wiring
2. map proof scenarios to lanes
3. lock metrics and thresholds
4. wire blocking vs evidence-only CI surfaces
5. run proof lanes on Windows-capable hardware
6. reclassify lanes and publish claim-boundary docs

#### Phase 4C.2: Monitor and Telemetry Closeout

Goal:

- make the streaming monitor surface strong enough to explain sustained pressure, queue health, and multi-renderer behavior without reading code

Prerequisite:

- `Phase 4C.1` has already produced real chunked benchmark evidence, not just smoke-only benchmark classifications

Why this is separate:

- the current monitor surface is broad, but it is still biased toward instantaneous values and “most recently active renderer” behavior
- Tier 2 needs an operable monitor contract, not just a large monitor list

Primary files:

- `modules/gaussian_splatting/core/performance_monitors.cpp`
- `modules/gaussian_splatting/core/performance_monitors.h`
- renderer-side streaming snapshot producers consumed by `get_monitor_streaming_snapshot()`
- relevant test files under `modules/gaussian_splatting/tests/`

Required work:

- add sustained-frame counters for the conditions that matter operationally:
  - queue pressure active for N consecutive frames
  - upload frame cap hit for N consecutive frames
  - upload bandwidth cap hit for N consecutive frames
  - chunk load cap hit for N consecutive frames
  - VRAM chunk cap hit for N consecutive frames
  - runtime-not-ready or invalid-buffer conditions that persist beyond one transient frame
- define and document the intended multi-renderer behavior for streaming monitors:
  - active renderer only
  - aggregate across registered renderers
  - aggregate for some monitors and active-only for others
- add focused test coverage for:
  - sustained-counter rollover and reset
  - multi-renderer aggregation or selection semantics
  - monitor readiness behavior when streaming data is absent vs present
- make sure the monitor names and semantics line up with the playbook and CI expectations

Expected output:

- new sustained counters or equivalent analytics surfaces for streaming pressure conditions
- clarified multi-renderer monitor semantics
- tests that lock those semantics down

Validation:

```bash
python3 tests/ci/run_module_tests.py --guard-only
python3 tests/runtime/run_runtime_validation.py --profile streaming-gpu-ci --skip-cpp --fail-on-skip
```

Exit criteria:

- an operator can tell from monitors whether a failure mode is transient or sustained
- multi-renderer behavior is deliberate and documented, not incidental
- the monitor surface can explain queue pressure, cap hits, and readiness failures without log spelunking

#### Phase 4C.3: Streaming Production Playbook and Final Closeout

Goal:

- make Tier 2 operable by someone who did not implement the refactor

Primary files:

- `docs/operations/streaming-production-playbook.md`
- `tests/runtime/README.md`
- `docs/testing/benchmark-suite.md`
- `.github/workflows/README.md`
- `docs/reports/streaming_tier2_execution_roadmap_2026-04-09.md`

Required work:

- add a production playbook that answers:
  - what to run locally before asking for review
  - what the blocking CI surfaces are
  - how to interpret the key streaming monitors
  - how to distinguish correctness regressions from transient perf noise
  - what to check first when `World Streaming Gate`, `Streaming Residency API`, or `GPU Streaming Stress` fails
- include the canonical validation commands for:
  - Windows `tests=yes` build
  - targeted ownership/batching doctests
  - module lane
  - `streaming-gpu-ci`
  - benchmark collection
- document the benchmark evidence policy:
  - which lanes are representative
  - which lanes are synthetic support scenes
  - what is acceptable evidence for a production-readiness claim
- finish the Tier 2 closeout note and remaining issue list

Expected output:

- `docs/operations/streaming-production-playbook.md`
- aligned docs across runtime, workflow, benchmark, and roadmap surfaces
- a final Tier 2 closeout checklist that a reviewer can execute without tribal knowledge

Validation:

- docs link cleanly from the reports/testing/operations surfaces
- the playbook commands match the actual workflow and runtime entrypoints
- the existing validated `Phase 4A` / `Phase 4B` regression set still passes

Exit criteria:

- a contributor can run the Tier 2 streaming validation path from docs alone
- the playbook explains the monitor surface and failure-triage flow in practical terms
- Tier 2 no longer depends on private context from the refactor branch history

#### Phase 4C Non-Goals

The following remain out of scope for this closeout phase:

- reopening sorter internals beyond the stabilized ownership/readback seams
- changing the canonical runtime gate from `streaming-gpu-ci` to a new profile
- broad renderer architecture refactors
- folding pre-existing editor crashes into the Tier 2 streaming execution path
- treating one-off perf noise as a reason to reopen already-validated functional seams

### Tier 2 Closeout

Tier 2 should only be called complete when all of the following are true:

- the validated `Phase 4A` and `Phase 4B` ownership/batching regression set stays green as `Phase 4C` lands
- `streaming-gpu-ci` stays green as the canonical blocking streaming gate
- the benchmark and runtime docs clearly distinguish chunked evidence from synthetic support lanes
- the benchmark manifest and lane policy no longer overstate streaming coverage
- sustained monitor counters and multi-renderer semantics are implemented and tested
- `docs/operations/streaming-production-playbook.md` is merged and linked from the docs surfaces that need it
- the remaining benchmark/ops closeout work is merged without reopening Tier 1-4B architecture
- pre-existing non-streaming crashes are either unchanged and tracked separately, or fixed in separate work

## Separate From The Main Tier 2 Path

These should stay out of the execution-critical refactor path unless they become reproducible regressions caused by Tier 2 work:

- pre-existing `[Editor]` importer inspector teardown crash
- unrelated editor / SceneTree crash behavior outside the streaming path
- transient machine-load performance anomalies unless they reproduce as real regressions

## Recommended Next Order

1. Land `Phase 4C.1` benchmark and gate closeout first.
2. Keep the current `Phase 4A` / `Phase 4B` validation set as the regression gate while `Phase 4C.1` lands.
3. Land `Phase 4C.2` monitor and telemetry closeout second.
4. Land `Phase 4C.3` production playbook and final Tier 2 closeout last.
5. Write the Tier 2 completion note and remaining non-Tier-2 issue list once all `Phase 4C` checkpoints are merged.
