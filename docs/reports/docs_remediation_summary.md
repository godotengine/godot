# Documentation Remediation Summary

**Date:** 2026-03-20
**Baseline assessment:** docs/reports/docs_assessment_2026-03-19.md (score: 2.41/5.00)
**Scope:** All findings F1-F16 from the baseline assessment

---

!!! info "Historical snapshot"
    This summary captures the remediation pass as of 2026-03-20. Some items referenced here were later reworked as the docs IA and landing pages changed.

    See the [baseline assessment](docs_assessment_2026-03-19.md).

    See the [latest assessment](latest-assessment.md).

## Finding Resolution Status

| ID | Priority | Title | Status | Files Changed | Notes |
|----|----------|-------|--------|---------------|-------|
| F1 | P0 | API docs cover 1 of 29 classes | **Resolved** | `docs/api/gaussian_data.md`, `gaussian_splat_asset.md`, `gaussian_splat_world3d.md`, `gaussian_splat_container.md`, `gaussian_splat_dynamic_instance3d.md`, `gaussian_splat_manager.md`, `gaussian_splat_renderer.md`, `internal-classes.md`, `docs/api/README.md` | 8 new API pages created (7 class references + 1 internal-classes index). Total: 9 maintained API pages plus 2 generated references. |
| F2 | P0 | `index.md` non-navigable | **Resolved** | `docs/index.md` | All 40 entries converted to clickable `[title](path)` markdown links. Missing release file references removed. Streaming, animation, migration, and new API pages added to index. |
| F3 | P0 | Compatibility matrix empty | **Resolved** | `docs/reference/compatibility_sources.yaml`, `docs/reference/compatibility-matrix.md` | Windows/Linux/macOS marked "supported" with evidence from `config.py:SUPPORTED_PLATFORMS`. Android/iOS marked "unsupported". Troubleshooting table added. |
| F4 | P0 | Stale line refs in `installation.md` | **Resolved** | `docs/getting-started/installation.md` | Line references updated to current source: `GaussianSplatNode3D` at `:77`, `GaussianSplatDynamicInstance3D` at `:80`, `GaussianSplatWorld3D` at `:81`, `GaussianSplatManager` at `:92-100`. |
| F5 | P0 | `doc_classes` XML stubs empty | **Resolved** | 14 XML files in `modules/gaussian_splatting/doc_classes/` | Expanded from 3 stub-only files to 14 fully populated XML files containing 423 method entries, 164 member entries, and 3 signal entries across all files. |
| F6 | P1 | Shader reference 95% placeholder | **Partially resolved** | `docs/api/shader_reference.md` | Regenerated to omit undocumented entries by default (file reduced from 2,832 to 1,111 lines). 49 functions and 68 uniform fields are documented. 146 functions and 123 fields remain undocumented in GLSL source. Full resolution requires adding `///` comments to GLSL shader files. |
| F7 | P1 | Duplicate quickstarts | **Resolved** | `docs/user/quickstart.md` | Converted to a redirect/landing page pointing to `getting-started/quick-start.md` with artist-specific resource links. No duplicate step-by-step content remains. |
| F8 | P1 | Troubleshooting redirect stubs | **Resolved** | Deleted `docs/troubleshooting/quick-reference.md`, `docs/troubleshooting/build-troubleshooting.md` | Both stub files removed. Index updated to point directly to `recurring-issues.md`. |
| F9 | P1 | Performance presets skeletal | **Resolved** | `docs/user/manual/performance-presets.md` | Expanded from 21 lines to 130 lines with three detailed preset value tables (rendering/LOD, LOD distance ratios, GPU memory/streaming), GPU memory guidance table, decision flowchart, and Custom preset documentation. |
| F10 | P1 | Streaming/animation zero docs | **Resolved** | `docs/features/streaming.md` (114 lines), `docs/features/animation.md` (173 lines) | Two new feature guides created with purpose, usage, configuration, examples, and troubleshooting sections. |
| F11 | P1 | CHANGELOG "not maintained" | **Resolved** | `CHANGELOG.md` | CHANGELOG now follows Keep a Changelog format with active entries under `[Unreleased]` covering Added, Changed, and Fixed categories. The "not maintained" statement has been removed. |
| F12 | P2 | Empty backtick placeholders | **Resolved** | `docs/troubleshooting/recurring-issues.md`, `docs/user/manual/faq.md` | Empty inline-code placeholders filled with substantive content. Verified: no empty backtick pairs remain in either file. |
| F13 | P2 | No screenshots or annotated visuals | **Deferred** | -- | Requires running the Godot editor with the module to capture screenshots of inspector panels, import dialogs, debug HUD, and visual results. Cannot be completed in a text-only remediation pass. |
| F14 | P2 | 22 internal classes lack documentation | **Partially resolved** | `docs/api/internal-classes.md`, 14 XML doc_classes | Internal-classes index page created cataloging all infrastructure classes with registration sites and purpose summaries. `get_doc_classes()` expanded from 3 to 14 classes. 15 classes remain without dedicated XML files (abstract bases, sort implementations, internal utilities). |
| F15 | P3 | No migration/upgrade guide | **Resolved** | `docs/migration/README.md` | Migration guide template created with versioned delta structure, compatibility checklist, and per-release sections. Linked from `docs/index.md`. |
| F16 | P3 | GDScript reference documents mostly test scripts | **Partially resolved** | `docs/api/gdscript_reference.md`, `docs/api/README.md` | README updated to note that the default `--scope public` excludes test/internal scripts. Full resolution requires regenerating the reference with the updated scope filter. |

---

## Metrics Comparison

| Metric | Before (2026-03-19) | After (2026-03-20) |
|--------|---------------------|---------------------|
| API pages in `docs/api/` (excluding README, generated refs) | 1 (`gaussian_splat_node3d.md`) | 9 (7 class references + `internal-classes.md` + existing Node3D page) |
| `doc_classes/` XML files | 3 (stubs, no methods) | 14 (with 423 methods, 164 members, 3 signals total) |
| Classes in `get_doc_classes()` | 3 | 14 |
| Clickable links in `docs/index.md` | 0 | 40 |
| Compatibility matrix entries | 0 validated (all "unknown") | 5 (3 supported, 2 unsupported with evidence) |
| Feature guide pages in `docs/features/` | 5 | 7 (added `streaming.md`, `animation.md`) |
| Performance presets page lines | 21 | 130 |
| Troubleshooting redirect stubs | 2 | 0 |
| Duplicate quickstart pages | 2 (identical flows) | 1 canonical + 1 redirect |
| CHANGELOG status | "not maintained" | Active with categorized entries |
| Banned word instances (public docs) | Not audited | 0 (hits only in `docs/reports/` internal files, all are C++ `clear()` method names) |
| Shader reference visible placeholders | 127 "Undocumented" + 116 "(undocumented)" | 0 visible (undocumented entries omitted; 146+123 remain in GLSL source) |
| Migration guide | None | Template created at `docs/migration/README.md` |
| Empty backtick placeholders | 2 (in `recurring-issues.md`, `faq.md`) | 0 |
| Link checker result | PASS | PASS |

---

## Remaining Gaps

### Deferred items requiring non-text tooling

1. **F6 -- Shader source documentation (146 undocumented functions, 123 undocumented fields)**
   Requires adding `///` documentation comments directly to GLSL shader source files in `modules/gaussian_splatting/shaders/`. The generated reference now omits undocumented entries by default so the visible output is clean, but the underlying coverage gap persists. This is rendering-engineer work, not a docs-only task.

2. **F13 -- Screenshots and annotated visuals**
   Requires launching the Godot editor with the gaussian_splatting module built and capturing screenshots of: inspector panels, import dialog, debug HUD, brush tools, benchmark results. A screenshot capture specification exists at `docs/development/screenshot-capture-spec.md` but execution requires a running editor session.

3. **F14 -- 15 remaining classes without XML doc_classes**
   Classes like `VRAMBudgetRegulator`, `GPUBufferManager`, `BitonicSort`, `RadixSort`, `OneSweepSort`, `ClusterCuller`, `GaussianSplatSceneDirector`, `GaussianSceneSerializer`, `GaussianIncrementalSaver`, `AssetDependencyManager`, `IGaussianLoader`, `IGPUSorter`, `ColorGradingResource`, `GaussianAnimationStateMachine`, and `GaussianSplatAsset` do not have XML doc_classes files. Many are internal infrastructure; the `internal-classes.md` index page provides brief coverage. Priority classes for future XML creation: `ColorGradingResource`, `GaussianAnimationStateMachine`, `GaussianSplatAsset`.

4. **F16 -- GDScript reference scope filtering**
   The generator supports `--scope public` to exclude test scripts, and the README documents this. A clean regeneration pass would improve the reference but was not executed during this campaign.

### Items not in the original assessment but discovered during remediation

5. **Per-GPU/driver validation data**
   The compatibility matrix now correctly reflects platform support status from `config.py`, but the GPU and Driver columns remain empty ("-"). Populating these requires hardware-specific test runs and is tracked as a future QA task.

6. **Versioned release notes**
   The CHANGELOG is now active, but no per-version release notes exist yet. The `mike` versioned docs pipeline is defined but has not been used to publish tagged release documentation.

---

## Validation Summary

- **Link checker:** PASS -- `python scripts/docs/check_links.py docs README.md BUILDING.md CONTRIBUTING.md` reports all links valid
- **File count -- `docs/api/`:** 12 `.md` files (meets >= 10 requirement)
- **File count -- `doc_classes/`:** 14 `.xml` files (meets >= 14 requirement)
- **`get_doc_classes()` entries:** 14 classes listed in `config.py:85-101`
- **No broken references in `docs/index.md`:** Verified, all 40 links use markdown format
- **No empty backtick placeholders:** Verified in `recurring-issues.md` and `faq.md`
- **No banned words in public docs:** Verified (0 instances outside `docs/reports/`)
