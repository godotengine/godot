# Issue Prioritization Roadmap

**Date**: 2026-02-03  
**Open Issues**: 64 (GitHub shows 64 open issues as of 2026-02-03; includes #808 which was not in the provided 63 list)  
**Analysis By**: Codex

## Executive Summary
The most impactful path is to fix correctness bugs and QA blockers first, then eliminate per-frame performance costs that affect every scene. In parallel, land a minimal metrics and testing foundation so larger refactors can proceed safely. Once stability and performance are predictable, depth integration and Mip-Splatting are the highest-visibility quality upgrades. Long-term artistic/editor features should remain in the backlog until the core pipeline is reliable and measurable.

## Impact/Effort Matrix

### Quick Wins (High Impact, Low Effort)
| Issue | Title | Est. Effort | Why Now |
|-------|-------|-------------|---------|
| #625 | VIS-3: Tile Cutoff Artifacts at Screen Edges | 1-2d | Highly visible artifact in every render. |
| #638 | BUF-9: Unconditional uniform buffer updates every frame | 1-2d | Removes constant CPU/GPU churn. |
| #643 | BUF-16: CPU position buffer rebuild each frame | 1-2d | Large per-frame CPU savings. |
| #644 | BUF-17: CPU-zeroed visibility bitmask each cull | 1d | Removes avoidable CPU work on every cull. |
| #649 | SORT-5: Handle sort_indirect failure without hard abort | 0.5-1d | Prevents hard failures in critical path. |
| #661 | MO-5: Redundant affine_inverse() in sort cache orchestrator | 0.5-1d | Free performance win; low risk. |
| #662 | MO-6: Redundant affine_inverse() in GPU sorting pipeline | 0.5-1d | Same as #661 in another hot path. |
| #734 | Replace ProjectSettings polling with callback system | 1d | Eliminates per-frame polling overhead. |
| #740 | Move per-chunk quantization to initialize() only | 1d | Avoids repeated work each frame. |
| #746 | Cache visible chunks set to avoid rebuild in LRU eviction | 1-2d | Reduces allocations and iteration churn. |
| #748 | Redundant sorter/streaming init, VRAM budget inconsistency | 1-2d | Stabilizes memory budget behavior. |
| #787 | SPZ colors incorrect high contrast (SH_C0 division) | 1-2d | Obvious correctness defect in assets. |
| #788 | PLY color drift (SH coefficient mismatch) | 1-2d | Obvious correctness defect in assets. |
| #795 | Import density_multiplier creates holes | 1-2d | Correctness of imported data. |
| #799 | Streaming performance monitors return 0 in QA tests | 0.5-1d | Unblocks 3 QA tests immediately. |

### Strategic Investments (High Impact, High Effort)
| Issue | Title | Est. Effort | Why Important |
|-------|-------|-------------|---------------|
| #629 | Consolidate streaming with shared culler/sorter flows | 1-3w | Unifies pipeline; unlocks multiple perf features. |
| #636 | BUF-7: Cluster cull atomic contention pattern | 1-2w | GPU bottleneck that scales poorly. |
| #650 | SORT-6: Selection and override tests | 3-5d | Enables safe sorting changes. |
| #653 | TEST-1: Metrics collection framework | 1-2w | Required to prove perf wins. |
| #654 | TEST-2: Synthetic scene generators | 1-2w | Reproducible perf/quality testing. |
| #655 | TEST-3: Stress test harness | 1-2w | Confidence under extreme loads. |
| #656 | TEST-4: Regression detection and CI integration | 2-3w | Prevents silent regressions. |
| #724 | Evaluate AOS->SOA data layout for GPU bandwidth | 2-4w | Potential major GPU bandwidth improvement. |
| #735 | Replace pack thread queue with lock-free implementation | 1-2w | Reduces contention in pack stage. |
| #736 | Defer synchronous chunk loads | 1-2w | Eliminates frame hitches. |
| #737 | SIMD for pack_gaussian() | 3-6d | Speeds CPU-heavy path. |
| #738 | Batch upload slices for GPU throughput | 1-2w | Improves upload efficiency. |
| #739 | Multi-horizon predictive prefetch | 2-3w | Streaming smoothness in large scenes. |
| #742 | Adaptive chunk size based on scene | 2-3w | Better memory/perf tradeoffs. |
| #747 | Consolidate GaussianSplatAsset and GaussianData | 2-4w | Removes conversion complexity and bugs. |
| #797 | Compute rasterizer tile artifacts at distance | 1-2w | High-visibility quality regression. |
| #798 | Decouple max_splat_count from streaming chunk budget | 1-2w | Restores correct scaling and tuning. |
| #806 | LOD compression in StreamingLODManager | 2-4w | Big memory win for large scenes. |
| #810 | Integrate splat depth buffer with Godot scene depth | 2-4w | Enables mesh+splat compositing. |
| #811 | Mip-Splatting for anti-aliasing and zoom stability | 2-4w | Major quality improvement. |
| #812 | Vector Quantization compression | 3-6w | Significant memory reduction. |
| #815 | Painterly pipeline stability review | 1-2w | Avoid regressions in a visible mode. |
| #818 | Post-process FX pipeline | 2-4w | Polishes visuals and sets future hooks. |
| #15 | Production memory management and asset lifecycle | 3-6w | Needed for large content pipelines. |
| #7 | Artistic lighting and color quantization | 3-6w | Long-term differentiator once core is stable. |

### Nice to Have (Low Impact, Low Effort)
| Issue | Title | Notes |
|-------|-------|-------|
| #641 | BUF-14: Interactive state dead buffer allocation | Cleanup of unused allocations. |
| #648 | SORT-4: Runtime override switch for sorting | Helpful for debugging, not user-facing. |
| #651 | SORT-7: Tile sort failure test hook | Small testing hook; not critical alone. |
| #658 | TRACE-2: RenderDoc capture automation | Dev convenience tooling. |
| #673 | DOC-8: Algorithm overview comments | Documentation clarity. |
| #707 | Runtime debug overlay showing render state | Helpful for developers. |
| #708 | Enable INFO-level logging in dev builds | Debug quality-of-life. |
| #744 | Improve or remove _optimize_data_layout() | Refactor cleanup. |
| #745 | Remove redundant frame data ring buffer | Refactor cleanup. |
| #802 | Remove obsolete ProjectSettings for legacy merge path | Cleanup. |
| #805 | Remove vestigial instance pipeline enabled field | Cleanup. |
| #807 | Migration note for async_compute setting | Documentation only. |
| #808 | Phase 10: Render Features Roadmap | Meta planning; convert to doc. |
| #816 | Platform compatibility matrix and perf guide | Good onboarding doc. |
| #817 | Improve Inspector organization | UX polish. |
| #819 | Editor tools scope discussion | Clarify direction; not code. |
| #820 | Code Quality Master Tracking Issue | Meta tracking; move to project board. |

### Reconsider (Low Impact, High Effort)
| Issue | Title | Recommendation |
|-------|-------|----------------|
| #8 | Editor Plugin for Painterly Splat Editing | Defer until core render pipeline is stable. |
| #10 | Modular falloff function system | Redesign into smaller, testable milestones. |
| #11 | Modular lighting model system | Redesign into incremental shader hooks. |
| #16 | Advanced asset features and dependency management | Split into concrete sub-features. |
| #109 | True async compute when Godot supports it | Keep blocked; revisit when engine support lands. |
| #813 | Procedural wind animation | Defer until depth/quality baseline is done. |
| #814 | Runtime color animation and tinting | Defer or combine with animation system roadmap. |

## Dependency Graph

```
[Correctness + QA Blockers]
(#625 #787 #788 #795 #797 #799 #748 #649)
        |
        v
[Testing + Metrics Foundation]
(#650 #651 #653 #654 #655 #656)
        |
        v
[Perf + Architecture Refactors]
(#638 #643 #644 #661 #662 #734 #740 #746 #629 #724 #735 #736 #737 #738 #739 #742 #747 #798)
        |
        v
[Streaming + Compression]
(#806 #812)
        |
        v
[Render Quality Features]
(#810 #811 #818 #7 #15 #815)
```

Key dependencies:
- #653 and #654 should precede #724, #735-#738, #742 so perf changes can be measured.
- #629 and #798 should land before deeper streaming work (#736, #739, #742, #806) to avoid rework.
- #747 simplifies data flow and should precede compression work (#806, #812).
- #810 enables mesh+splat compositing and is a prerequisite for many quality features and post-process work (#818).
- #799 must be fixed to restore QA coverage before large refactors.

## Recommended Sprint Plan

### Sprint 1: Stability Foundation (Week 1-2)
**Theme**: Fix critical bugs, unblock QA, stabilize core pipeline

| Priority | Issue | Title | Est. |
|----------|-------|-------|------|
| P0 | #799 | Streaming performance monitors return 0 | 1d |
| P0 | #787 | SPZ colors incorrect high contrast | 1-2d |
| P0 | #788 | PLY color drift (SH layout mismatch) | 1-2d |
| P0 | #625 | Tile cutoff artifacts at screen edges | 1-2d |
| P0 | #649 | Handle sort_indirect failure without abort | 1d |
| P0 | #748 | Redundant sorter/streaming init, VRAM budget inconsistency | 1-2d |
| P1 | #795 | Import density_multiplier holes | 1-2d |
| P1 | #797 | Compute rasterizer tile artifacts at distance | 1-2w |
| P1 | #650 | Selection and override tests | 3-5d |
| P1 | #651 | Tile sort failure test hook | 1-2d |

**Success Criteria**: QA tests 11/11 passing, no visible color drift, no screen-edge tile cutoff artifacts, sorting failure is non-fatal, basic sorting tests in place.

### Sprint 2: Performance Wins (Week 3-4)
**Theme**: Capture low-hanging perf wins and add metrics

| Priority | Issue | Title | Est. |
|----------|-------|-------|------|
| P0 | #638 | Unconditional uniform buffer updates | 1-2d |
| P0 | #643 | CPU position buffer rebuild each frame | 1-2d |
| P0 | #644 | CPU-zeroed visibility bitmask each cull | 1d |
| P0 | #661 | Redundant affine_inverse in sort cache | 1d |
| P0 | #662 | Redundant affine_inverse in GPU sorting | 1d |
| P0 | #734 | Replace ProjectSettings polling with callbacks | 1d |
| P0 | #740 | Move per-chunk quantization to initialize() only | 1d |
| P0 | #746 | Cache visible chunks set for LRU eviction | 1-2d |
| P1 | #653 | Metrics collection framework | 1-2w |
| P1 | #654 | Synthetic scene generators | 1-2w |

**Success Criteria**: Measured CPU time reduction in frame time for 1M splats, metrics logging available for perf regressions, no behavior change in visuals.

### Sprint 3: Render Quality (Week 5-6)
**Theme**: Visual quality and integration milestones

| Priority | Issue | Title | Est. |
|----------|-------|-------|------|
| P0 | #810 | Integrate splat depth buffer with Godot scene depth | 2-4w |
| P0 | #811 | Mip-Splatting for anti-aliasing and zoom stability | 2-4w |
| P1 | #815 | Painterly pipeline stability review | 1-2w |
| P1 | #806 | LOD compression in StreamingLODManager | 2-4w |
| P1 | #812 | Vector Quantization compression (research spike + plan) | 1-2w |

**Success Criteria**: Mesh+splat compositing works with depth, zoomed views are stable, painterly mode has defined test coverage, LOD memory reduction quantified.

### Sprint 4: Polish & Integration (Week 7-8)
**Theme**: UX, docs, tooling, cleanup

| Priority | Issue | Title | Est. |
|----------|-------|-------|------|
| P0 | #817 | Improve Inspector organization | 1-3d |
| P0 | #707 | Runtime debug overlay | 3-5d |
| P0 | #708 | INFO-level logging in dev builds | 1-2d |
| P0 | #816 | Platform compatibility matrix and perf guide | 2-4d |
| P1 | #673 | Algorithm overview comments | 2-3d |
| P1 | #658 | RenderDoc capture automation | 3-5d |
| P1 | #807 | Migration note for async_compute setting | 1d |
| P1 | #802 | Remove obsolete ProjectSettings | 1-2d |
| P1 | #805 | Remove vestigial instance pipeline enabled field | 1-2d |
| P1 | #744 | Improve/remove _optimize_data_layout() | 1-3d |
| P1 | #745 | Remove redundant frame data ring buffer | 1-2d |
| P1 | #641 | Dead buffer allocation cleanup | 1-2d |
| P1 | #648 | Runtime override switch for sorting | 1-2d |
| P1 | #808 | Convert roadmap issue to docs | 1d |
| P1 | #820 | Code Quality Master tracking into project board | 1d |
| P1 | #819 | Editor tools scope discussion note | 1d |

**Success Criteria**: Cleaner Inspector UX, improved debug signal, docs updated, and refactor debt reduced.

## Issues to Close or Redesign

| Issue | Title | Recommendation | Reason |
|-------|-------|----------------|--------|
| #808 | Phase 10: Render Features Roadmap | Convert to docs and close | Roadmap belongs in docs, not an issue. |
| #819 | Editor tools scope discussion | Convert to design note | Not actionable; needs decision. |
| #820 | Code Quality Master Tracking Issue | Move to project board | Better tracked as project meta. |
| #8 | Editor Plugin for Painterly Splat Editing | Redesign into milestones | Too large and unbounded. |
| #10 | Modular falloff function system | Redesign into shader hooks | Reduce scope and enable incremental delivery. |
| #11 | Modular lighting model system | Redesign into shader hooks | Reduce scope and enable incremental delivery. |
| #16 | Advanced asset features and dependency management | Break into concrete tasks | Too broad; clarify MVP. |
| #109 | True async compute | Keep blocked/label and defer | Depends on upstream engine support. |

## Risk Assessment

### High-Risk Issues (Handle Carefully)
- #724: Data layout migration risks shader and ABI mismatches; mitigate with staged rollout and A/B toggle.
- #629: Streaming consolidation can regress streaming behavior; mitigate with #650/#651 tests first.
- #735: Lock-free queue can introduce races; mitigate with proven algorithm and fallback path.
- #736: Deferred chunk loads can cause visible pops; mitigate with guard thresholds and metrics.
- #739: Predictive prefetch can spike memory; enforce budgets via #798.
- #747: Asset/data consolidation risks import compatibility; add migration tests and versioning.
- #806: Compression can introduce corruption or quality loss; add encode/decode tests and visual baselines.
- #810: Depth integration impacts ordering and performance; ship behind toggle and compare baselines.
- #811: Mip-Splatting may increase cost; tune using #653 metrics and scenes.
- #812: VQ compression affects quality and pipeline; treat as research spike first.

### Issues Requiring Research First
- #811, #812: Validate quality/perf tradeoffs and implement offline pipeline plan.
- #724: Prototype with one buffer set before full migration.
- #735: Evaluate lock-free queue options and memory ordering.
- #739: Collect streaming stats to tune prediction.
- #109: Track Godot RD support for async compute before re-opening.

## Recommended Immediate Actions

1. **This Week**: Tackle #799, #787, #788 (QA unblocks + obvious correctness).
2. **Next Week**: Land #625, #649, #748, #795 to stabilize core pipeline.
3. **This Month**: Finish Sprint 1 and start Sprint 2 perf wins (#638, #643, #644, #661, #662, #734, #740, #746).

## Long-Term Vision

Deliver a render engine that is visually correct by default, measurably fast, and predictable under stress. The near-term focus is on correctness and metrics, followed by data-driven performance refactors. Once the pipeline is stable, depth integration and Mip-Splatting become the quality milestones that enable real game usage. Long-term, compression and memory management allow massive scenes, while artistic and editor tooling features are added once the core engine is reliable.

*Generated: 2026-02-03*
