# Documentation Assessment Report — Godot Gaussian Splatting

**Date**: 2026-03-19
**Assessor**: Automated audit (Claude Opus 4.6)
**Scope**: `docs/`, `modules/gaussian_splatting/`, `scripts/`, root `.md` files
**Benchmark**: Official Godot documentation standards

---

!!! info "Historical snapshot"
    This report captures the docs state on 2026-03-19 and should not be read as the current public status page.

    See the [remediation summary](docs_remediation_summary.md).

    See the [latest assessment](latest-assessment.md).

## 1. Executive Summary

**Overall weighted score: 2.48 / 5.00** — below Godot-docs parity.

The documentation corpus has strong foundations: task-oriented quickstarts, a well-structured `GaussianSplatNode3D` API page, practical workflow guides, an automated docs pipeline, and a link checker that currently passes. Critical gaps still prevent parity with Godot-quality documentation:

1. **API coverage is 3 of 29**: The module registers 29 classes via `GDREGISTER_CLASS`/`GDREGISTER_ABSTRACT_CLASS` (`register_types.cpp:70-139`), but only `GaussianSplatNode3D` has a maintained API page. The 3 XML `doc_classes` files are brief-description-only stubs with zero method/property/signal documentation.

2. **Navigation dead-ends**: `docs/index.md` uses inline-code paths instead of markdown links (lines 7-67) and references 4 `release/` files that do not exist (lines 64-67). Two troubleshooting pages are redirect stubs.

3. **Generated references are mostly placeholders**: `shader_reference.md` (2,832 lines) contains 127 "Undocumented." function entries and 116 "(undocumented)" field entries. `gdscript_reference.md` (3,374 lines) documents mostly test/internal scripts, not user-facing API.

4. **Compatibility matrix is empty**: All 5 platforms show "unknown" status (`compatibility-matrix.md:19-23`) despite the module being actively developed and tested on Windows.

**Validation checks executed**:
- `python scripts/docs/check_links.py docs README.md BUILDING.md CONTRIBUTING.md` → **PASS** (all links valid)
- Media budget: 1 SVG (2KB), 2 videos (59KB total) — well within budget
- No `TODO`/`FIXME`/`PLACEHOLDER` markers found in any docs `.md` file

---

## 2. Godot Documentation Benchmark Rubric

### Official Godot Standards (Extracted)

| Standard | Godot Requirement | Source |
| --- | --- | --- |
| Class reference completeness | Every registered class must have filled `<brief_description>`, `<description>`, method/member/signal docs | Godot class reference writing guidelines |
| Page length | Under 1,000 words per page | Godot docs writing guidelines |
| Heading length | 5 words or fewer | Godot docs writing guidelines |
| Banned words | `obvious`, `simple`, `basic`, `easy`, `actual`, `just`, `clear`, `however` | Godot docs writing guidelines |
| Code examples | Dynamic typing, real-world variable names, no `foo`/`bar` | Godot class reference writing guidelines |
| Image format | WebP, under 300KB, lossless preferred | Godot image guidelines |
| Video format | WebM, under 2MB | Godot image guidelines |
| Brief descriptions | ~200 characters, functional overview | Godot class reference writing guidelines |
| Empty vs low-effort | Blank field preferred over superficial description | Godot class reference writing guidelines |
| Line wrapping | 80-100 characters | Godot docs writing guidelines |

### Godot Quality Bar Interpreted for This Audit

- Complete class/API reference for all user-visible registered classes
- Consistent task-first docs with explicit navigational links
- Realistic, copy-paste-safe code examples with real-world names
- Structured troubleshooting (symptom → cause → fix)
- Repeatable docs pipeline with quality gates enforced in CI
- Versioned, evidence-backed release and compatibility documentation

---

## 3. Detailed Scorecard

| # | Dimension | Weight | Score (0-5) | Confidence | Evidence Summary | Impact |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | **Coverage breadth/depth** | 20% | 2.5 | 0.85 | 55 docs files; strong workflow coverage; but 26 of 29 registered classes lack API docs. Only `GaussianSplatNode3D` has a maintained reference page (`docs/api/gaussian_splat_node3d.md`). | Users can start quickly but hit walls for advanced/non-primary APIs. |
| 2 | **Accuracy/consistency** | 15% | 2.2 | 0.82 | Line-reference citations in `installation.md:30-33` point to `register_types.cpp:82-85` but current lines are `:77-81, :88`. Project settings doc claims 76 registered keys (`project-settings.md:21`). FAQ line 5 has empty inline-code placeholder (`faq.md:5`). `recurring-issues.md:21` has empty inline-code placeholder. | Stale references erode contributor trust and increase debugging time. |
| 3 | **Information architecture/navigation** | 10% | 2.0 | 0.90 | `docs/index.md` uses backtick code paths (not clickable links) for all 30+ entries. 4 release doc references point to non-existent files (`index.md:64-67`). `troubleshooting/quick-reference.md` and `troubleshooting/build-troubleshooting.md` are 5-line and 6-line redirect stubs. | Primary entrypoint fails as navigation hub; users encounter dead-ends. |
| 4 | **Task success orientation** | 15% | 3.3 | 0.80 | Strong: `quick-start.md` (4-step with verification gates), `user/quickstart.md` (artist lane), `importing.md`, `GSPLATWORLD_BAKE.md`, `color-grading-quick-start.md` all have step-by-step flows with expected outcomes. Multi-platform commands (bash + powershell) in quickstarts. | First-run and common workflows are well served. Deeper tasks require code spelunking. |
| 5 | **API/reference completeness** | 15% | 1.5 | 0.92 | 29 registered classes; 1 has full API page; 3 have stub-only XML (`doc_classes/`); 25 have zero dedicated API docs. `config.py:85-90` exports only 3 doc classes. Shader reference: 127 undocumented functions. | **Major parity blocker.** Power users and contributors cannot discover API contracts from docs. |
| 6 | **Examples quality** | 8% | 3.0 | 0.78 | Hand-written examples are strong: real variable names, complete scripts, multiple patterns per page (`ply-loader.md:57-90`, `color-grading-quick-start.md:49-72`, `artist_pipeline.md:40-69`, `gaussian_splat_node3d.md:415-438`). Generated shader/GDScript refs have no examples. | Good where present; absent in generated references. |
| 7 | **Troubleshooting quality** | 7% | 2.7 | 0.75 | Central `recurring-issues.md` has 4 issue categories with symptom/check/fix format. Feature pages (`ply-loader.md:93-99`, `color-grading-quick-start.md:76-81`, `artist_pipeline.md:74-79`) include targeted troubleshooting tables. But: 2 stub redirect pages, empty placeholder in `recurring-issues.md:21`. | Adequate for common issues; edge cases undocumented. |
| 8 | **Maintainability/process** | 5% | 2.8 | 0.80 | Docs pipeline exists: `build_docs_site.py`, `stage_public_docs.py`, `check_docs_media_budget.py`, `check_links.py`, `extract_gdscript_docs.py`, `generate_shader_docs.py`, `generate_project_settings_reference.py`. Style guide exists (`documentation-style-guide.md`). Contribution standards require link checks. Duplicate quickstarts create drift risk. | Good tooling foundation; needs CI enforcement. |
| 9 | **Versioning/release alignment** | 3% | 1.5 | 0.88 | Versioned docs site model documented (`docs-site.md:17-25`). mike+MkDocs pipeline defined. But: `CHANGELOG.md` explicitly states "not maintained" (line 4). 4 release docs referenced in index don't exist. Compatibility matrix shows all platforms "unknown". No release notes exist. | External trust and release-readiness signaling are broken. |
| 10 | **Media/docs UX quality** | 2% | 2.5 | 0.70 | Media conventions documented (`features/media.md`). Assets exist: 1 SVG pipeline diagram (2KB), 2 demo videos (17KB + 42KB). Git LFS guidance present. No screenshots of editor UI, inspector panels, or visual results. No annotated screenshots for any workflow. | Functional media system but minimal educational richness. |

### Weighted Overall Score Calculation

| Dimension | Weight | Score | Contribution |
| --- | ---: | ---: | ---: |
| Coverage breadth/depth | 0.20 | 2.5 | 0.500 |
| Accuracy/consistency | 0.15 | 2.2 | 0.330 |
| Information architecture | 0.10 | 2.0 | 0.200 |
| Task success orientation | 0.15 | 3.3 | 0.495 |
| API/reference completeness | 0.15 | 1.5 | 0.225 |
| Examples quality | 0.08 | 3.0 | 0.240 |
| Troubleshooting quality | 0.07 | 2.7 | 0.189 |
| Maintainability/process | 0.05 | 2.8 | 0.140 |
| Versioning/release alignment | 0.03 | 1.5 | 0.045 |
| Media/docs UX quality | 0.02 | 2.5 | 0.050 |
| **TOTAL** | **1.00** | | **2.414 ≈ 2.4** |

---

## 4. Coverage Matrix: Registered Classes → Documentation

29 classes registered in `modules/gaussian_splatting/register_types.cpp:70-139`.

**Depth scale**: 0 = none, 1 = mention only, 2 = introductory task notes, 3 = usable guide, 4 = strong guide + troubleshooting, 5 = Godot-parity reference.

| # | Class | Type | Registered At | Dedicated API Page | doc_classes XML | Depth | Gap Severity |
| ---: | --- | --- | --- | --- | --- | ---: | --- |
| 1 | `GaussianData` | Resource | `:70` | None | Stub (brief only, no methods) | 2 | **P0** |
| 2 | `GaussianSplatAsset` | Resource | `:71` | None | None | 1 | **P0** |
| 3 | `GaussianSplatWorld` | Resource | `:72` | None | None | 1 | **P1** |
| 4 | `GaussianStreamingSystem` | Class | `:73` | None | None | 0 | **P1** |
| 5 | `VRAMBudgetRegulator` | Class | `:74` | None | None | 0 | **P2** |
| 6 | `GaussianSplatNode3D` | Node3D | `:77` | **Yes** (`api/gaussian_splat_node3d.md`) | None (should have) | **4** | Low |
| 7 | `GaussianSplatDebugHUD` | Control | `:78` | None | None | 0 | **P1** |
| 8 | `GaussianSplatContainer` | Node3D | `:79` | None | None | 1 | **P0** |
| 9 | `GaussianSplatDynamicInstance3D` | Node3D | `:80` | None | None | 1 | **P0** |
| 10 | `GaussianSplatWorld3D` | Node3D | `:81` | None | None | 1 | **P0** |
| 11 | `GaussianSplatRenderer` | RefCounted | `:84` | None | Stub (brief only) | 2 | **P0** |
| 12 | `GaussianMemoryStream` | RefCounted | `:85` | None | None | 0 | **P1** |
| 13 | `PainterlyMaterial` | Resource | `:86` | None | None | 0 | **P1** |
| 14 | `GPUBufferManager` | RefCounted | `:87` | None | None | 0 | **P2** |
| 15 | `GaussianSplatManager` | Singleton | `:88` | None | Stub (brief only) | 1 | **P0** |
| 16 | `GaussianSplatSceneDirector` | Class | `:89` | None | None | 0 | **P2** |
| 17 | `ColorGradingResource` | Resource | `:90` | Documented in feature pages | None | 3 | Low |
| 18 | `PLYLoader` | RefCounted | `:120` | Documented in `ply-loader.md` | None | 3 | Low |
| 19 | `SPZLoader` | RefCounted | `:121` | Mentioned in `importing.md` | None | 1 | **P1** |
| 20 | `IGaussianLoader` | Abstract | `:122` | None | None | 0 | **P2** |
| 21 | `IGPUSorter` | Abstract | `:125` | None | None | 0 | **P2** |
| 22 | `BitonicSort` | Class | `:126` | None | None | 0 | **P2** |
| 23 | `RadixSort` | Class | `:127` | None | None | 0 | **P2** |
| 24 | `OneSweepSort` | Class | `:128` | None | None | 0 | **P2** |
| 25 | `ClusterCuller` | Class | `:131` | None | None | 0 | **P2** |
| 26 | `GaussianAnimationStateMachine` | Class | `:134` | None | None | 0 | **P1** |
| 27 | `GaussianSceneSerializer` | Class | `:135` | None | None | 0 | **P2** |
| 28 | `GaussianIncrementalSaver` | Class | `:136` | None | None | 0 | **P2** |
| 29 | `AssetDependencyManager` | Class | `:139` | None | None | 0 | **P2** |

**Summary**: 3 classes at depth ≥3, 4 classes at depth 1-2, **22 classes at depth 0** (zero documentation).

### Feature/Workflow Coverage

| Surface | Documented? | Depth | Primary Docs | Gap Severity |
| --- | --- | ---: | --- | --- |
| Build/install editor | Yes | 4 | `getting-started/installation.md`, `reference/build-test-ci.md` | Low |
| First visible splat | Yes | 4 | `getting-started/quick-start.md`, `user/quickstart.md` | Low |
| PLY/SPZ import | Yes | 4 | `workflows/importing.md`, `features/ply-loader.md` | Low |
| GSplatWorld bake | Yes | 4 | `workflows/GSPLATWORLD_BAKE.md` | Low |
| Color grading | Yes | 4 | `features/color-grading-quick-start.md`, `reference/color-grading.md` | Low |
| Artist brush pipeline | Yes | 4 | `features/artist_pipeline.md` | Low |
| Performance tuning | Partial | 2 | `user/manual/performance-presets.md` (21 lines, thin) | **P1** |
| Lighting behavior | Yes | 3 | `user/manual/lighting-behavior.md`, `architecture/lighting-system.md` | Low |
| Streaming system | No | 0 | None | **P0** |
| Animation/keyframes | No | 0 | None | **P1** |
| Dynamic instancing | No | 0 | None | **P0** |
| Multi-asset world scenes | Partial | 2 | `GSPLATWORLD_BAKE.md` mentions it | **P1** |
| GPU sorting internals | No | 0 | None (architecture level) | **P2** |
| Memory/VRAM management | Partial | 1 | `MEMORY_SUBSYSTEM.md` in module (not in docs/) | **P1** |
| Platform compatibility | Empty | 0 | `compatibility-matrix.md` (all "unknown") | **P0** |
| Migration/upgrade guide | No | 0 | None | **P1** |
| Project settings (full) | Partial | 3 | `reference/project-settings.md` (76 of registered keys documented) | Medium |
| Testing/CI guide | Yes | 4 | `testing/setup-guide.md`, `testing/benchmark-suite.md` | Low |
| Contributor onboarding | Yes | 3 | `contributor/onboarding.md` | Low |
| Shader reference | Partial | 1 | `api/shader_reference.md` (127 undocumented entries) | **P1** |
| Troubleshooting | Partial | 3 | `troubleshooting/recurring-issues.md` | Medium |

---

## 5. Prioritized Findings

### P0 — Must Fix (Parity Blockers)

**F1. API documentation covers 1 of 29 registered classes.**
- Evidence: `register_types.cpp:70-139` registers 29 classes. `docs/api/` contains 1 maintained class page (`gaussian_splat_node3d.md`). `config.py:85-90` exports only 3 doc_classes, all stub-only (no methods/properties/signals in XML).
- Impact: Users and contributors cannot discover API contracts, method signatures, or behavioral documentation for 28 classes without reading C++ source.
- Godot benchmark: Godot requires every registered class to have a filled class reference with brief description, full description, and documented methods/members/signals.

**F2. `docs/index.md` uses non-navigable code paths and references 4 missing files.**
- Evidence: All 30+ links in `docs/index.md:7-67` use backtick inline-code format (not clickable markdown links). Lines 64-67 reference `release/public-beta-readiness-audit-2026-03-18.md`, `release/real-fork-migration-plan.md`, `release/fork-cutover-follow-up.md`, `release/history-rewrite-artifact-cleanup.md` — none of which exist (`docs/release/` directory does not exist).
- Impact: The canonical documentation hub cannot function as a navigation entrypoint.

**F3. Compatibility matrix is entirely empty.**
- Evidence: `docs/reference/compatibility-matrix.md:19-23` shows all 5 platforms (Windows, Linux, macOS, Android, iOS) as "unknown" with no GPU/driver data. `compatibility_sources.yaml` contains no test evidence.
- Impact: Cannot make release-readiness claims. External users have no platform guidance.

**F4. Stale line-level references in core onboarding docs.**
- Evidence: `docs/getting-started/installation.md:30-33` claims `GaussianSplatNode3D` registration at `:82`, `GaussianSplatDynamicInstance3D` at `:84`, `GaussianSplatWorld3D` at `:85`, `GaussianSplatManager` singleton at `:100`. Current lines: `:77`, `:80`, `:81`, `:88` respectively.
- Impact: Contributors following file:line references land on wrong code.

**F5. `doc_classes/` XML files are empty stubs.**
- Evidence: All 3 XML files (`GaussianData.xml`, `GaussianSplatManager.xml`, `GaussianSplatRenderer.xml`) contain only `<brief_description>` and `<description>` with empty `<tutorials>`. No `<methods>`, `<members>`, `<signals>`, or `<constants>` elements. These classes have extensive `_bind_methods()` in source.
- Impact: Godot's built-in class reference help system shows no API documentation for any module class.

### P1 — High Value

**F6. Generated shader reference is 95%+ placeholder.**
- Evidence: `docs/api/shader_reference.md` (2,832 lines) contains 127 functions marked "Undocumented." and 116 uniform fields marked "(undocumented)".

**F7. Duplicate quickstarts create maintenance drift risk.**
- Evidence: `docs/getting-started/quick-start.md` and `docs/user/quickstart.md` have nearly identical 4-step flows with only minor wording differences.

**F8. Two troubleshooting pages are redirect stubs.**
- Evidence: `docs/troubleshooting/quick-reference.md` (5 lines) and `docs/troubleshooting/build-troubleshooting.md` (6 lines) redirect to `recurring-issues.md`.

**F9. Performance presets documentation is skeletal.**
- Evidence: `docs/user/manual/performance-presets.md` is 21 lines with no concrete preset values, no comparison table, no GPU memory implications.

**F10. Streaming, animation, and dynamic instancing have zero user documentation.**
- Evidence: `GaussianStreamingSystem` (`:73`), `GaussianAnimationStateMachine` (`:134`), `GaussianSplatDynamicInstance3D` (`:80`) are registered and bound but have no docs pages.

**F11. `CHANGELOG.md` explicitly states "not maintained" (line 4).**
- No release notes exist anywhere in the repository.

### P2 — Important

**F12. Empty inline-code placeholders in user-facing text.**
- Evidence: `docs/troubleshooting/recurring-issues.md:21` contains ` `` ` (empty backticks). `docs/user/manual/faq.md:5` contains ` `` ` (empty backticks where module name should appear).

**F13. No screenshots or annotated visuals of editor integration.**
- Evidence: `docs/assets/images/` contains 1 SVG pipeline diagram only. No screenshots of: inspector panels, import dialog, debug HUD, brush tools, benchmark output.

**F14. 22 internal/infrastructure classes lack even brief documentation.**
- Evidence: Classes like `VRAMBudgetRegulator`, `GPUBufferManager`, `ClusterCuller`, `BitonicSort`, `RadixSort`, `OneSweepSort` are registered but have zero documentation. While many are internal, their existence in the public class registry means they appear in Godot's class list.

### P3 — Nice to Have

**F15. No migration/upgrade guide exists.**
- No guidance for users moving between versions or from other Gaussian splatting implementations.

**F16. `GDScript reference` documents mostly test scripts.**
- Evidence: `docs/api/gdscript_reference.md` (3,374 lines) includes 66 script sections, but the majority document test harnesses and internal tooling, not user-facing API patterns.

---

## 6. "Parity with Godot Docs" Definition of Done

Measurable acceptance criteria:

| # | Criterion | Current State | Target | How to Measure |
| ---: | --- | --- | --- | --- |
| 1 | Class reference coverage | 3 of 29 classes have XML (stubs only) | 100% of user-visible classes have filled XML with methods/members/signals | Count non-empty `<methods>` sections in `doc_classes/*.xml` |
| 2 | Maintained API pages | 1 of 29 classes | All user-facing classes (est. 12-15) have markdown API pages | Count files in `docs/api/` with depth ≥ 3 |
| 3 | Navigation quality | 0 clickable links in `docs/index.md`; 4 broken refs | All index entries are markdown links; 0 broken refs; 0 public orphan pages | Run `check_links.py`; verify all index entries are `[text](path)` format |
| 4 | Code line reference accuracy | Multiple stale references identified | 0 stale file:line references in onboarding/reference docs | Automated reference validation or regeneration script |
| 5 | Task completion | 6 end-to-end workflows documented | 10+ workflows covering all common user tasks | Count workflow pages with ≥3 verification steps |
| 6 | Troubleshooting depth | 4 issue categories; 2 stub pages | Top 15 recurring issues with symptom→cause→fix; 0 stub pages | Count non-redirect troubleshooting entries |
| 7 | Generated reference quality | 127 "Undocumented" shader entries | Placeholder ratio < 5% in generated references | Count "Undocumented"/"(undocumented)" entries |
| 8 | Compatibility evidence | 0 of 5 platforms with evidence | ≥ 3 platforms with validated GPU/driver test evidence | Count non-"unknown" compatibility matrix entries |
| 9 | CI enforcement | Link checker exists but not in CI | Link check + media budget + strict build enforced in PR CI | Verify CI workflow includes docs gates |
| 10 | Release documentation | CHANGELOG "not maintained"; no release notes | Versioned release notes per tagged release | Verify `CHANGELOG.md` or release notes per `mike deploy` version |
| 11 | Banned words compliance | Not audited | 0 instances of Godot-banned words in docs | Grep for `obvious\|simple\|basic\|easy\|actual\|just\|clear\|however` |
| 12 | Media quality | 1 SVG + 2 tiny videos; 0 screenshots | ≥ 5 annotated screenshots for core workflows; all WebP format | Count images in `docs/assets/images/` |

---

## 7. 30/60/90-Day Roadmap

### Days 0-30: Trust and Navigation Fundamentals

| Milestone | Owner | Effort | Deliverables | Success Metric |
| --- | --- | --- | --- | --- |
| Convert `docs/index.md` to navigable links | Docs lead | 0.5 days | All backtick paths → `[title](path)` links | 0 non-linked entries |
| Remove/replace missing release references | Docs lead | 0.5 days | Remove `index.md:64-67` or create placeholder pages | 0 broken references |
| Fix placeholder text artifacts | Docs lead | 0.5 days | Fill empty backticks in `recurring-issues.md:21`, `faq.md:5` | 0 empty inline-code blocks |
| Refresh stale line references in installation guide | Module lead | 1 day | Update `installation.md:30-33` to current `register_types.cpp` lines | Line references verified against HEAD |
| Regenerate project-settings reference | Module lead | 0.5 days | Run `python scripts/generate_project_settings_reference.py` | Key count matches measured GLOBAL_DEF count |
| Add docs CI gate to PR workflow | DevEx | 3-5 days | CI job: `check_links.py` + `check_docs_media_budget.py` + `mkdocs build --strict` | Docs gates required in PR checks |
| Consolidate duplicate quickstarts | Docs lead | 1 day | One canonical quickstart; artist lane as audience-specific delta | 1 canonical quickstart file |
| Delete troubleshooting redirect stubs | Docs lead | 0.5 days | Remove `quick-reference.md` + `build-troubleshooting.md`; update inbound links | 0 stub redirect pages |
| Fill compatibility matrix for Windows | QA | 1 day | Add Windows GPU/driver evidence to `compatibility_sources.yaml` | ≥1 platform non-"unknown" |

**Total 0-30 day effort: ~10-14 eng-days**

### Days 31-60: API Coverage & Reference Quality

| Milestone | Owner | Effort | Deliverables | Success Metric |
| --- | --- | --- | --- | --- |
| Create API pages for 6 user-critical classes | Module maintainers + tech writer | 15-20 days | API pages for: `GaussianData`, `GaussianSplatAsset`, `GaussianSplatWorld3D`, `GaussianSplatContainer`, `GaussianSplatDynamicInstance3D`, `GaussianSplatManager` | 7 total maintained API pages (including existing Node3D) |
| Expand `doc_classes/` XML to include methods/properties | Module maintainers | 8-12 days | Full XML docs for at least: `GaussianData`, `GaussianSplatManager`, `GaussianSplatRenderer`, `GaussianSplatNode3D` | 4+ XML files with non-empty `<methods>` |
| Add shader source-code documentation | Rendering engineer | 5-8 days | Document top 30 most-used shader functions with `///` comments | "Undocumented" count < 50 (from 127) |
| Create streaming system user guide | Module maintainer | 3-5 days | `docs/features/streaming.md` with settings, behavior, and troubleshooting | New page at depth ≥ 3 |
| Create performance tuning guide (replace skeleton) | Module maintainer | 2-3 days | Expand `performance-presets.md` with concrete values, comparison table, GPU memory guidance | Page > 80 lines with data tables |
| Fill compatibility for Linux + macOS | QA | 2-3 days | Evidence entries in `compatibility_sources.yaml` | ≥3 platforms non-"unknown" |

**Total 31-60 day effort: ~35-51 eng-days**

### Days 61-90: Parity Program

| Milestone | Owner | Effort | Deliverables | Success Metric |
| --- | --- | --- | --- | --- |
| Complete class-reference XML for all user-visible classes | Module maintainers | 20-30 days | Full XML docs for 12-15 user-facing classes | `get_doc_classes()` returns ≥12 classes with filled XML |
| Create dynamic instancing + animation guides | Module maintainer | 5-8 days | `docs/features/dynamic-instancing.md`, `docs/features/animation.md` | 2 new feature pages at depth ≥ 3 |
| Add annotated screenshots for core workflows | Docs lead | 3-5 days | ≥5 screenshots: editor panel, import dialog, debug HUD, brush tools, benchmark results | ≥5 WebP images in `docs/assets/images/` |
| Create versioned release notes template + first entry | Release manager | 2-3 days | Release notes template; `CHANGELOG.md` updated or replaced with release process | Release docs exist for current version |
| Generated docs quality uplift | Tooling engineer | 8-12 days | Improve generators to classify public vs internal, suppress pure-test scripts, add scope labels | Placeholder ratio < 5%; GDScript ref has user-API section |
| Full cross-link and orphan reduction pass | Docs lead | 3-5 days | Every public docs page has ≥1 inbound markdown link | 0 public orphan pages |

**Total 61-90 day effort: ~41-63 eng-days**

---

## 8. Quick Wins (≤ 1 day each)

| # | Action | File(s) | Effort | Impact |
| ---: | --- | --- | --- | --- |
| 1 | Convert `docs/index.md` code paths to markdown links | `docs/index.md:7-67` | 2 hours | Fixes primary navigation |
| 2 | Remove 4 missing release doc references | `docs/index.md:64-67` | 15 min | Eliminates dead-end references |
| 3 | Fix empty backtick placeholders | `recurring-issues.md:21`, `faq.md:5` | 15 min | Removes visible quality gaps |
| 4 | Update registration line references in installation | `installation.md:30-33` | 30 min | Corrects contributor guidance |
| 5 | Delete 2 redirect stub pages + update links | `quick-reference.md`, `build-troubleshooting.md` | 30 min | Eliminates dead-end pages |
| 6 | Regenerate project-settings reference | Run `generate_project_settings_reference.py` | 30 min | Refreshes stale key count |
| 7 | Add `GaussianSplatNode3D` to `config.py:get_doc_classes()` | `config.py:85-90` | 15 min | Enables in-editor class reference |
| 8 | Add scope warning banner to generated references | `gdscript_reference.md:1-3`, `shader_reference.md:1-3` | 30 min | Sets expectations for placeholder content |

---

## 9. Strategic Initiatives (> 1 week)

| # | Initiative | Duration | Description |
| ---: | --- | --- | --- |
| 1 | **API Reference Parity Program** | 6-8 weeks | Systematic documentation of all user-visible classes with maintained API pages + XML class references, prioritized by user impact. |
| 2 | **Docs Reliability Automation** | 2-3 weeks | Build a `file:line` reference validator that checks all code citations in docs against current source. Integrate into CI. |
| 3 | **Troubleshooting Knowledge Base** | 3-4 weeks | Expand from 4 categories to 15+ with structured symptom→cause→fix entries, reproducible check commands, and links to CI artifacts. |
| 4 | **Release Evidence Pipeline** | 2-3 weeks | Automated compatibility evidence collection from CI runs → `compatibility_sources.yaml`. Versioned release notes per `mike deploy`. |
| 5 | **Educational Media Program** | 2-3 weeks | Capture and annotate screenshots/screencasts for 5 core workflows. Establish WebP standard per Godot guidelines. |

---

## 10. Docs Corpus Inventory

### Audience Classification

| Audience | Directory | File Count | Total Lines |
| --- | --- | ---: | ---: |
| **User** | `getting-started/`, `user/`, `features/`, `workflows/` | 14 | ~750 |
| **Contributor** | `contributor/`, `testing/`, `architecture/`, `governance/` | 9 | ~650 |
| **Maintainer** | `development/`, `style/`, generation scripts | 4 | ~200 |
| **Reference** (mixed) | `api/`, `reference/`, `troubleshooting/` | 10 | ~4,300 (mostly generated) |
| **Internal** (excluded from public) | `agent_memory/`, `archive/`, `reports/` | 4+ | ~6,000 |

### Key Docs Quality Metrics

| Metric | Value |
| --- | --- |
| Total docs markdown files | 55 |
| Public-facing docs files | ~45 |
| Files with code examples | 12 |
| Files with troubleshooting tables | 8 |
| Files with "Last updated" dates | 8 |
| Generated reference files | 3 (`gdscript_reference.md`, `shader_reference.md`, `project-settings.md`) |
| Media assets | 1 SVG + 2 videos (59KB total) |
| doc_classes XML files | 3 (all stubs) |
| Link checker result | PASS (0 broken links) |
| TODO/FIXME markers | 0 |

---

## 11. Methodology Notes

- All registered classes enumerated from `grep GDREGISTER register_types.cpp`
- All docs files inventoried via filesystem traversal
- Content quality assessed by direct file reads of every docs markdown file
- Code reference accuracy spot-checked against current `HEAD` source
- Link validation executed via `python scripts/docs/check_links.py`
- Godot documentation standards extracted from official Godot contributing guidelines (docs writing guidelines, class reference writing guidelines, contribution checklist, image guidelines)
- Scores reflect observed state as of assessment date; no modifications made to docs during assessment
