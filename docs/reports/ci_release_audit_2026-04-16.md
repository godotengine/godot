# CI & Windows Release Audit — 2026-04-16

**Branch:** `investigate/ci-and-windows-release` (worktree at `C:/projects/godotgs-ci-audit`)
**Base:** master @ `b93ba6ef92` (current master tip @ `63d990eb10` — commit `b93ba6ef92` is one revert behind tip)
**Repo under audit:** `klausi3D/godotGS` (the fork; not upstream `godotengine/godot`)

---

## TL;DR

1. **A Windows zip IS being published.** The `nightly-20260416` release (today, 03:53 UTC) contains `godotgs-windows-x86_64-nightly-20260416.zip`. This is the **first** nightly to include a Windows artifact. Yesterday's `nightly-20260415` and earlier were Linux-only. The `release_builds.yml` Windows lane is now wired through end-to-end.
2. **README is stale.** `README.md` still says "public Releases are still Linux-only until the first Windows publish lands" — that publish landed today. Same stale claim in `.github/workflows/README.md:15`.
3. **3 of 4 master CI workflows are red on the latest push** (`63d990eb10`, PR #245). Two failures point at real code-level issues (one Linux-only ODR error from `#define private public`, one module test failure on Windows). One failure is a simple broken docs link — fixed in this audit branch.
4. **No README → download path exists for end users.** README links to docs but never names "Releases" or links the GitHub Releases page. A first-time user has no obvious path to the Windows zip.

---

## 1. Workflow Inventory

5 active workflows in `.github/workflows/` plus 5 archived in `.github/archived-workflows/`.

| Workflow | File | Triggers | Purpose | Latest master status |
| --- | --- | --- | --- | --- |
| Release Builds | `release_builds.yml` | push (master/main/develop on src paths), PR (same), tag `v*`, schedule `30 2 * * *`, dispatch | Build Linux (ubuntu-latest) + Windows (self-hosted) editor; package to `.tar.xz` / `.zip`; nightly publish to GitHub Releases; prune old nightlies | ✅ success (38m18s, run 24506709811) — produced `godotgs-linux-ci-249` and `godotgs-windows-ci-249` artifacts |
| Baseline QA Automation | `baseline_qa.yml` | push (master/develop/feature/phase on `modules/gaussian_splatting/**`/`tests/**`), PR (master/develop), merge_group, schedule `30 3 * * *`, dispatch | Linux CPU job (ubuntu-latest, full build + headless QA categories `ply,pipeline,runtime,module`) + Windows GPU job (self-hosted, `sorting` category) | ❌ failure (run 24506709782) — Linux build failed, Windows passed |
| Gaussian Production Gates | `gaussian_production_gates.yml` | PR (master), push (master/develop), merge_group, schedule `30 3 * * 1` (weekly), dispatch | Guards (Windows self-hosted) → GPU evidence requirement (Linux compute) → Module Build + Runtime Harness (Windows self-hosted self-hosted GPU) → optional openworld-proof evidence | ❌ failure (run 24506709800) — `Run module tests` failed |
| Docs Pages (Versioned) | `docs_pages.yml` | push (master/main on `docs/**`/scripts/mkdocs/module), tag `v*`, dispatch | mkdocs build (`--strict`) + mike deploy to `gh-pages` (`latest` from master, versioned from `v*` tags) | ❌ failure (run 24506709793) — mkdocs strict mode aborted on broken intra-docs link |
| Gaussian Shader Validation | `gaussian_shader_validation.yml` | PR + push (master/develop) on shader/renderer paths, merge_group, dispatch | Shader compile matrix + contracts on self-hosted Windows runner | (no recent run — gated path didn't trigger; not in this push's check set) |

Archived (in `.github/archived-workflows/`, suffix `.disabled` so GitHub ignores them): `benchmark.yml`, `build-engine.yml`, `gaussian_pipeline_validation.yml`, `test_gaussian_splatting.yml`, `test_phase4.yml`. Each tracked in `.github/workflows/README.md:54-58`.

The `.github/workflows/README.md` file is well-maintained and authoritative — it lists triggers, dispatch inputs, schedules, and dependencies. Match against actual yml files: ✅ accurate as of `b93ba6ef92`.

---

## 2. Master Failure Diagnosis

All three failures are on commit `63d990eb10` (PR #245 "fix: per-node color grading independence"), pushed to master at 11:03 UTC. The `Release Builds` workflow on the same commit succeeded.

### 2.1 Docs Pages — broken link in `docs/reports/index.md` ⚠️ FIXED IN THIS AUDIT

**Run:** [`24506709793`](https://github.com/klausi3D/godotGS/actions/runs/24506709793) — failed in `Validate MkDocs build (strict)` after 2m05s.

**Root cause:** `docs/reports/index.md:7` links to `streaming_remaining_architecture_issue_backlog_2026-04-10.md`, which doesn't exist anywhere in the tree (verified by Grep). `mkdocs build --strict` (workflow line 75) treats unresolved intra-docs links as fatal:

```
WARNING - Doc file 'reports/index.md' contains a link 'streaming_remaining_architecture_issue_backlog_2026-04-10.md',
          but the target 'reports/streaming_remaining_architecture_issue_backlog_2026-04-10.md' is not found
          among documentation files.
Aborted with 1 warnings in strict mode!
##[error]Process completed with exit code 1.
```

The two sibling roadmap docs (`streaming_tier2_execution_roadmap_2026-04-09.md`, `streaming_tier2_phase4c1_issue_backlog_2026-04-09.md`) exist; only the `2026-04-10` backlog file is missing. `git log --all` shows no record of that file ever being committed — the index entry was added speculatively.

**Classification:** Misconfigured docs (stale link), not a code regression.

**Fix:** Removed the dead link in `docs/reports/index.md`. Committed in this audit branch (see §5).

### 2.2 Baseline QA Automation (Linux CPU lane) — `#define private public` GCC ODR error ❌ NEEDS USER REVIEW

**Run:** [`24506709782`](https://github.com/klausi3D/godotGS/actions/runs/24506709782) — Linux job failed in `Build module-enabled Godot editor` after 6m18s. Windows GPU lane (sorting category) passed.

**Root cause:** Compilation of `modules/gaussian_splatting/tests/test_gaussian_streaming_lifecycle.cpp` fails with:

```
./core/templates/list.h:223:9: error: 'struct List<T, A>::_Data' redeclared with different access
./core/templates/rb_map.h:196:9: error: 'struct RBMap<K, V, C, A>::_Data' redeclared with different access
./core/templates/rb_set.h:167:9: error: 'struct RBSet<T, C, A>::_Data' redeclared with different access
./core/io/resource_loader.h:176:9: error: 'struct ResourceLoader::ThreadLoadTask' redeclared with different access
scons: *** [bin/obj/modules/gaussian_splatting/tests/test_gaussian_streaming_lifecycle.linuxbsd.editor.dev.x86_64.o] Error 1
```

The test file at `modules/gaussian_splatting/tests/test_gaussian_streaming_lifecycle.cpp:1-5` does:

```cpp
#define _ALLOW_KEYWORD_MACROS
#define private public
#include "../core/gaussian_streaming.h"
#undef private
```

`gaussian_streaming.h` transitively pulls Godot core templates (`list.h`, `rb_map.h`, `rb_set.h`, `resource_loader.h`). With `private` redefined to `public` during that include, the inner `_Data` / `ThreadLoadTask` structs end up declared with `public` access. Other TUs see them with `private`. GCC under strict flags treats this as ODR violation; MSVC silently merges them, which is why the Windows lane passes.

**Classification:** Real code regression in this fork's test code, Linux-only. Same family as the documented `MSVC PMF forward-decl ODR trap` (project memory) — different mechanism, same root cause (test code reaching into private state via header trickery).

**Fix proposal:** Replace the `#define private public` hack with a proper test-only `friend` declaration in `gaussian_streaming.h`, or move the test to use only public API. **Do not band-aid.** This is the kind of issue the team has already paid for once with the PMF/ODR incident — worth fixing properly. Out of scope for this audit (touches module test surface and core header guarantees).

### 2.3 Gaussian Production Gates — module test failure ❌ NEEDS USER REVIEW

**Run:** [`24506709800`](https://github.com/klausi3D/godotGS/actions/runs/24506709800) — `Module Build + Runtime Harness (Windows Self-Hosted)` failed in `Run module tests` after 3m09s. Build succeeded, smoke tests succeeded.

**Root cause (from log):**

```
[module-tests] 'GaussianSplatting [Editor]' failed.
##[error]Process completed with exit code 1.
```

The doctest `GaussianSplatting [Editor]` suite fails. Per project memory (`project_test_baseline.md`), there are 4 pre-existing GaussianSplatting test failures proven on `d78d180c70`. **This may be a known-pre-existing failure.** Cannot confirm from CI log alone — it just reports the suite name failed without listing which test cases.

**Fix proposal:** Verify against the documented test baseline. If this is one of the 4 known pre-existing failures, the gate should either be marked non-blocking or those tests should be quarantined with an explicit skip + tracking issue. If it's a *new* failure introduced by PR #245 (color grading per-node), it needs a code fix. The PR #245 area (color grading) is plausibly related to the `[Editor]` suite scope. Out of scope for this audit — the test runner output needs to be expanded to surface which doctest cases failed.

### 2.4 Why was master red yesterday too?

Looking back at the runs list: every master push since at least `2026-04-15 13:58Z` shows the same pattern — `Release Builds` ✅, `Baseline QA Automation` ❌, `Gaussian Production Gates` ❌, `Docs Pages` ❌. The failure mode is consistent with what's documented above (build error from `#define private public`, doctest failure, broken docs link). **None of these are flakes.**

---

## 3. Release Pipeline Audit

### Where Windows builds are produced

`release_builds.yml`, job `build_windows` (lines 272-407). Triggers on push (master/main/develop), tag `v*`, nightly schedule (`30 2 * * *`), and dispatch. **Does not** run on PRs (line 278: `if: github.event_name != 'pull_request'`) — this is a deliberate guard against running self-hosted Windows on untrusted PR code, which is correct.

Build command (lines 313-327):
```powershell
python -m SCons platform=windows target=editor gs_native_arch=no \
    cache_path=C:/godotgs-scons-cache cache_limit=50 -j12
# + dev_build=yes tests=yes for non-stable channels
```

### Artifact format and location

- **Format:** `godotgs-windows-x86_64-<tag>.zip` containing the editor binary + `BUILD-INFO.txt`. SHA256 sidecar `.sha256`.
- **CI artifact (per run):** uploaded as `godotgs-windows-<channel>-<run_number>` to the workflow artifacts dropdown (14-day retention, lines 400-407).
- **Public release attachment:** when `publish=true` (nightly schedule or `v*` tag), `softprops/action-gh-release@v2` (lines 437-453) attaches the Windows zip + sha256 to a GitHub Release on `klausi3D/godotGS`.

### Existing GitHub Releases

20 releases on `klausi3D/godotGS`, all `nightly-YYYYMMDD` prereleases (no `v*` stable tags ever published). Sample audit:

| Release | Linux tar.xz | Windows zip | Notes |
| --- | --- | --- | --- |
| nightly-20260416 (today) | ✅ | ✅ **first Windows publish** | Includes BUILD-INFO.txt, both sha256 sidecars |
| nightly-20260415 | ✅ | ❌ Linux-only | |
| nightly-20260406 | ✅ | ❌ Linux-only | |
| (older nightlies) | ✅ | ❌ Linux-only | |

The Windows publish landed today (2026-04-16 03:53 UTC) — likely the first night after PR #239 ("tier1 adoption path — public evaluator scene, Windows release packaging, docs", in nightly-20260416 changelog) reached master.

### End-user reachability

Walk the path a first-time visitor takes:

1. Visit `https://github.com/klausi3D/godotGS` (1 click).
2. Read README. The README mentions "Linux nightly editor" as the fastest evaluation path, never names "Releases" page, never links it. Lines 11/17/34 explicitly say Windows is *not* in public releases (now stale).
3. To find the Windows zip, the user has to know to click the "Releases" sidebar link on the GitHub repo page, then expand the latest prerelease → scroll to assets → click `godotgs-windows-x86_64-nightly-20260416.zip`.

**Click count:** 4-5 clicks, none of them signposted. The README actively misdirects users away from looking for a Windows download.

---

## 4. Gap Punch List (ordered by effort)

### Easy (≤30 min, individually)

1. **Update README to tell users where the Windows zip is.** Update lines 11, 17, 34 to reflect that Windows nightly zips now ship. Add a "Download" section at the top with a direct link to `https://github.com/klausi3D/godotGS/releases/latest`. ⚠️ User decision: phrasing/positioning of the alpha disclaimer.
2. **Update `.github/workflows/README.md:15`** to drop "still Linux-only" caveat now that Windows is publishing.
3. **Already done in this audit:** removed broken docs link in `docs/reports/index.md` (see §5).
4. **Add Node.js 24 opt-in to workflows** to silence the deprecation warnings on every run. Add `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: "true"` at workflow `env:` level. Already done in `gaussian_shader_validation.yml:45` — the other four workflows haven't adopted it. Cosmetic but reduces noise; mandatory before September 2026 anyway.

### Medium (1-3h)

5. **Fix the `Run module tests` failure in `gaussian_production_gates.yml`.** Surface *which* doctest cases failed in the workflow log (currently the runner just prints `'GaussianSplatting [Editor]' failed`). If the failures match the 4 documented pre-existing ones, mark the suite continue-on-error or quarantine those specific cases with a tracking issue. If new, fix in code.
6. **Make the test runner less opaque.** `tests/ci/run_module_tests.py` should print per-doctest pass/fail with a summary at the end, so failures surface in the GitHub Actions step summary, not just exit 1.
7. **Wire a `Latest Windows Build` README badge** that points to `https://github.com/klausi3D/godotGS/releases/latest`. Trivial Markdown — but worth confirming with user that public links are wanted.

### Larger (half-day or more)

8. **Fix the Linux build error in `test_gaussian_streaming_lifecycle.cpp`.** Replace `#define private public` with proper friend declarations or restructure the test to use public APIs. Verify against full Linux QA suite. This is the same anti-pattern family as the previously documented `MSVC PMF forward-decl ODR trap` — worth doing right.
9. **First stable release.** No `v*` tag has ever been pushed. The `release_builds.yml` workflow already supports stable tags (line 105-110: tag `v*` push → `channel=stable`, full build without `dev_build=yes`, attached to a non-prerelease release). Cutting `v0.1.0-alpha` would give end users a non-prerelease download to land on. Decision needed on naming + readiness.
10. **Decide what "ready to ship" means** for the Windows binary. The current zip is a `dev_build=yes` editor (not optimized, not stripped, with debug assertions). For end-user evaluation that's fine; for a stable tag, the workflow already switches to the optimized `target=editor` build automatically (line 208). User decision: communicate to users that the nightly zip is dev-flavored.

---

## 5. Quick-Fix Commit Manifest

One commit on `investigate/ci-and-windows-release`:

| Commit | File(s) | Reason | Risk |
| --- | --- | --- | --- |
| (this audit's commit) | `docs/reports/index.md` (-1 line), `docs/reports/ci_release_audit_2026-04-16.md` (new) | Removes the broken link to a never-existed file. Unblocks `Docs Pages (Versioned)` workflow without touching content semantics. | Low — the deleted line pointed at a 404. No content lost (file never existed). |

**Not auto-committed (deferred for user review):**

- README changes about Windows availability — touches user-facing claims about project state, user should land that themselves with chosen messaging.
- Workflows README "still Linux-only" line — same reasoning.
- Node.js 24 opt-in — should be done as a coordinated single PR across all four workflows.

---

## 6. Decisions Needed From User

1. **Where to link the Windows zip from the README** — direct link to releases page? Releases badge? A new "Downloads" section at the top of the README?
2. **Whether to cut a stable `v0.1.0-alpha` (or similar) tag** to give end users a non-prerelease landing page. Workflow is ready; just needs tag + go-ahead.
3. **How to handle the `Run module tests` failure** — quarantine known pre-existing tests vs. fix forward. Need to know if those 4 baseline failures are still authoritative.
4. **Whether the `#define private public` test pattern should be eradicated** repo-wide or only in the file currently breaking. There may be other test files using the same trick that haven't yet broken on Linux.
5. **Should the dev-flavored nightly Windows zip be marked clearly** as "developer build, not optimized" in the release body? `BUILD-INFO.txt` is in the zip but no end user looks inside before downloading.

---

## Appendix: Tools used

- `gh run list --repo klausi3D/godotGS --branch master --limit 15`
- `gh run view <id> --repo klausi3D/godotGS --log-failed`
- `gh release view nightly-2026041{5,6} --repo klausi3D/godotGS`
- `gh release list --repo klausi3D/godotGS --limit 20`
- File reads on `.github/workflows/*.yml`, `README.md`, `docs/reports/index.md`, `modules/gaussian_splatting/tests/test_gaussian_streaming_lifecycle.cpp`

**Important:** the local `gh` config defaults to `godotengine/godot` (upstream) in this worktree because the `audit` remote points at upstream. All `gh` calls in this audit explicitly used `--repo klausi3D/godotGS` to target the fork. Future investigators should do the same or set `gh repo set-default klausi3D/godotGS` in the worktree.
