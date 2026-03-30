# Migration Guide

## Purpose

Provide a release-to-release upgrade path for Gaussian Splatting users and contributors, with explicit compatibility checks and remediation steps.

## Current scope

This guide is intentionally process-first because a complete versioned delta requires release history curation (tags/changelog/PR-level breakage records).

Use this page as the canonical migration template for each future release.

## Migration data contract (required per release)

For each release, fill these sections:

1. Version transition (`from -> to`) and release date.
2. Breaking API changes (class/method/property/signal changes).
3. Project setting changes (`added`, `renamed`, `removed`, `default changed`).
4. Asset/workflow changes (import, bake, runtime behavior).
5. Validation command updates (build/test/benchmark docs).
6. Rollback and mitigation guidance.

## Upgrade checklist

- [ ] Build commands validated against [Build / Test / CI Command Reference](../reference/build-test-ci.md).
- [ ] Project settings diff reviewed against [Project settings reference](../reference/project-settings.md).
- [ ] Import and bake workflows revalidated:
  - [Import workflow](../workflows/importing.md)
  - [Gaussian Splat World Bake Workflow](../workflows/GSPLATWORLD_BAKE.md)
- [ ] Runtime smoke run completed from [First Run](../getting-started/quick-start.md).
- [ ] Troubleshooting updates added to [Recurring issues](../troubleshooting/recurring-issues.md) for newly observed regressions.
- [ ] Compatibility evidence updated in:
  - `compatibility_sources.yaml`
  - [Compatibility Matrix](../reference/compatibility-matrix.md)

## Release entry template

Copy this block for each release transition:

```md
## vX.Y.Z -> vA.B.C

### Breaking changes
- ...

### Project settings migration
- old_key -> new_key
- removed_key (replacement: ...)

### Workflow changes
- import: ...
- bake: ...
- runtime: ...

### Validation updates
- command changes: ...

### Known risks and mitigations
- risk: ...
- mitigation: ...
```

## Ownership

- Primary owner: release manager / module maintainer.
- Supporting owners: docs maintainer, QA owner.
