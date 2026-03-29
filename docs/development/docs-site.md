# Versioned Docs Site

## Purpose

Define the GitHub Pages publication pipeline for versioned Gaussian Splatting documentation.

## Stack

| Component | Tool |
| --- | --- |
| Static site generator | MkDocs + Material |
| Version publishing | mike |
| Source docs root | `docs/` |
| Public staging dir | `.site/public-docs/` |
| Pages target | `gh-pages` branch |
| Navigation order | `docs/.pages` via `mkdocs-awesome-nav` |

Canonical site URL defaults to `https://klausi3d.github.io/godotGS/` and can be overridden for forks or custom domains with `DOCS_SITE_URL`.

## Versioning Model

| Source ref | Published version path |
| --- | --- |
| `master` / `main` push | `/latest/` |
| `v*` tag push | `/<tag>/` (for example `/v1.2.0/`) |

Default root points to `latest` via `mike set-default latest`.

Repository setting requirement:

- GitHub Pages source must be configured to deploy from the `gh-pages` branch.

## Navigation Rules

- Keep `docs/.pages` as the source of truth for top-level section order.
- Use `index.md` for section landing pages. Do not use `README.md` for published section indexes.
- Root-level `docs/README.md` is excluded from the published site so `docs/index.md` remains the single docs homepage.
- Published URLs stay anchored to the folder path (`section/`), so renaming a section landing page from `README.md` to `index.md` does not require a redirect by itself.
- Add `redirect_maps` entries only when a page's published URL changes.

## Local Commands

```bash
python3 -m pip install -r docs/requirements.txt -r docs/requirements-site.txt
ENABLE_GIT_DATES=false python3 scripts/build_docs_site.py --strict
python3 scripts/docs/release_acceptance.py
```

Equivalent Make targets:

```bash
make docs-site
```

## CI Pipeline

1. Generate docs artifacts:
   - `python3 scripts/build_documentation.py --all`
2. Stage public docs:
   - `python3 scripts/stage_public_docs.py --source docs --output .site/public-docs`
3. Enforce media budgets:
   - `python3 scripts/check_docs_media_budget.py`
4. Validate site:
   - `ENABLE_GIT_DATES=false mkdocs build --strict`
   - `python3 scripts/docs/release_acceptance.py`
5. Publish version:
   - `mike deploy`

## Release Acceptance

Run these checks before every publish candidate:

1. Stage and build the public docs tree:
   - `ENABLE_GIT_DATES=false python3 scripts/build_docs_site.py`
2. Run the repository markdown link check:
   - `python3 scripts/docs/check_links.py docs README.md BUILDING.md CONTRIBUTING.md`
3. Run the release gate:
   - `python3 scripts/docs/release_acceptance.py`

The release gate currently verifies:

- broken internal links
- public orphan pages in `.site/public-docs/`
- redirect target correctness from `mkdocs.yml`
- media budget across images and video
- missing alt text in staged public docs

## Manual QA Checklist

Usability paths:

- From the homepage, a new user can find `Installation`.
- From the homepage, a new user can find `First Run`.
- From the homepage, a contributor can find `Build / Test / CI Command Reference`.
- From the homepage, a reviewer can see the project-status box and reach `Compatibility Matrix`.

Visual QA:

- Desktop light mode: homepage and section cards look intentional and readable.
- Desktop dark mode: homepage still has correct contrast and hierarchy.
- Mobile: the drawer opens, the top-level nav is usable, and pages do not introduce page-level horizontal overflow.
- Tables: review `Compatibility Matrix` and `Build from Source` at mobile width for readable overflow handling.
- Cards: review homepage and section landing pages for wrapping at narrow widths.

Search QA:

- `install` should surface `Installation` first.
- `first run` should surface `First Run` first.
- `build test` should surface `Build / Test / CI Command Reference` first.
- `compatibility` should surface `Compatibility Matrix` first.

Release spot-check pages:

- Homepage
- Start Here landing page
- `Installation`
- `First Run`
- `Build from Source`
- `Compatibility Matrix`
- Reference landing page

## Engine patch report automation

- Generator: `python3 scripts/generate_engine_patch_report.py`
- Baseline config: `docs/reference/engine_patch_sources.yaml`
- Outputs (committed):
  - `docs/reference/engine-patch.md`
  - `docs/reference/engine-patch.json`

Operational policy:

- Keep `upstream_ref` pinned to a stable commit/tag in `engine_patch_sources.yaml`.
- Update that pin only as an explicit maintenance change.
- Generation is non-blocking in docs CI (`|| true` in workflow step).
- Use `--strict` locally/CI only when intentionally gating on report freshness.

## Public Scope

The staged docs copy excludes internal docs directories:

- `docs/agent_memory/`
- `docs/archive/`

Out-of-scope relative links are rewritten to GitHub `blob`/`tree` URLs so public pages remain navigable.

The published MkDocs config enables instant navigation, top tabs, sticky tabs, section indexes, navigation path breadcrumbs, and footer next/previous links on the staged docs tree.

## Doxygen Inclusion

`docs/Doxyfile` writes C++ API HTML to `docs/api/cpp`. These generated files are copied into the staged docs and shipped with each docs version.
