# Documentation Assessment Report - Godot Gaussian Splatting

**Date**: 2026-03-29
**Assessor**: Automated audit (Codex)
**Scope**: `docs/`, `scripts/`, root docs entry files, public docs surface
**Benchmark**: Internal docs quality and Godot-style docs expectations

---

!!! info "Historical snapshot"
    This report captures the post-PR180 docs state on 2026-03-29.

    See the [latest assessment](latest-assessment.md) for the current public status pages and the report bridge.

## Executive Summary

The docs surface is substantially better than the 2026-03-19 baseline.

The site now has audience-shaped navigation, real landing pages, a calmer visual system, explicit compatibility evidence levels, and a release-channel page that matches the current nightly-first public reality. The docs release-acceptance gate also exists and passes on the current public docs surface.

The remaining gaps are now mostly evidence and proof gaps rather than IA gaps: real screenshots, richer benchmark results, per-GPU/driver compatibility rows, and a named release series still need follow-up.

## Current Assessment

| Area | Status | Notes |
| --- | --- | --- |
| Information architecture | Improved | Four-lane top-level navigation, section landing pages, and clearer task routing are in place. |
| Homepage / front door | Improved | The homepage now states maturity, preferred evaluation path, and the current public binary reality. |
| Compatibility evidence | Partial | The matrix now uses explicit evidence levels plus a concrete Windows RTX 3090 row and an Ubuntu 24.04 Linux QA row, but full Windows driver/OS identifiers and hardware-backed Linux validation are still incomplete. |
| Release guidance | Improved | Release docs now describe nightly-first reality and the latent stable-tag path honestly. |
| Visual proof | Partial | Brand assets and diagrams are in place, but authoritative editor screenshots are still missing. |
| Benchmark publication | Partial | The benchmark runner and dashboard exist, but the public results surface is still thin. |
| Repo hygiene / reporting | Improved | The release-acceptance check is now formalized and the report bridge points to the current state. |

## Remaining Gaps

1. Real editor screenshots are still missing from the highest-value workflow pages.
2. Published benchmark evidence is still thin and needs curated, date-stamped scenarios.
3. Compatibility rows still need full driver identifiers and stronger hardware-backed Linux/macOS evidence.
4. The first named release remains deferred, so public guidance is still nightly-first.
5. Deeper API/reference and generated-reference gaps still exist outside the current front-door work.

## Validation Summary

- `python scripts/docs/check_links.py docs README.md BUILDING.md CONTRIBUTING.md` - PASS
- `python scripts/docs/release_acceptance.py` - PASS
- `python scripts/build_docs_site.py --skip-generate` - PASS with the repo's existing MkDocs warning baseline about out-of-scope links to root-engine files

## Comparison Notes

Compared with the 2026-03-19 assessment, the biggest improvements are:

- the homepage is now a real front door instead of a link dump
- top-level navigation is audience-shaped instead of repo-shaped
- compatibility status is evidence-based instead of empty
- release-channel language now matches the actual public distribution model
- the docs site has a coherent visual system instead of stock Material defaults

The main unresolved work is evidence gathering, not structural cleanup.
