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

## Versioning Model

| Source ref | Published version path |
| --- | --- |
| `master` / `main` push | `/latest/` |
| `v*` tag push | `/<tag>/` (for example `/v1.2.0/`) |

Default root points to `latest` via `mike set-default latest`.

Repository setting requirement:

- GitHub Pages source must be configured to deploy from the `gh-pages` branch.

## Local Commands

```bash
python3 -m pip install -r docs/requirements.txt -r docs/requirements-site.txt
python3 scripts/build_docs_site.py --strict
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
   - `mkdocs build --strict`
5. Publish version:
   - `mike deploy`

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

## Doxygen Inclusion

`docs/Doxyfile` writes C++ API HTML to `docs/api/cpp`. These generated files are copied into the staged docs and shipped with each docs version.
