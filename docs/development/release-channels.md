# Release Channels

## Purpose

Define how executable builds are distributed from this repository and what guarantees each channel provides.

## Channels

| Channel | Source | Intended audience | Retention | Stability expectation |
| --- | --- | --- | --- | --- |
| CI artifact | `.github/workflows/release_builds.yml` on PR/push | Contributors validating branch changes | 14 days (GitHub Actions artifact retention) | No compatibility guarantee; debugging/testing only |
| Nightly prerelease | `.github/workflows/release_builds.yml` schedule/manual with `publish_channel=nightly` | Early testers validating latest integration changes | Keep latest 14 nightly prereleases | May break at any time; not for production |
| Stable release | Tag push `v*` or manual with `publish_channel=stable` | End users and downstream integrators | All stable releases retained | Production channel; bugfix patches published as new tags |

## Trigger Model

| Trigger | Behavior |
| --- | --- |
| Pull request touching engine/module paths | Build Linux editor and upload contributor artifact |
| Push to `master`/`main`/`develop` | Build Linux editor and upload contributor artifact |
| Nightly schedule (`30 2 * * *` UTC) | Build and publish nightly prerelease (`nightly-YYYYMMDD`) then prune old nightlies |
| Tag push (`v*`) | Build and publish stable release from tag |
| Manual dispatch | Optional publish mode: `none`, `nightly`, or `stable` |

## Package Contents

Each publish includes:

- Linux editor tarball (`godotgs-linux-x86_64-<tag-or-sha>.tar.xz`)
- SHA-256 checksum file
- `BUILD-INFO.txt` metadata (channel, commit, binary name, generation timestamp)

## Current Limits and Caveats

| Area | Constraint |
| --- | --- |
| GitHub Release assets | Individual file size must remain under 2 GiB |
| Actions artifacts | Retention window is bounded by GitHub settings and this workflow's retention setting |
| Public preview channel | Nightly releases are prereleases and intentionally not production-stable |
| Platform coverage (current) | Linux editor binary only; expand matrix when release demand requires it |

## Manual Examples

```bash
# Manual nightly publish from the Actions UI:
# workflow_dispatch -> publish_channel=nightly

# Manual stable publish from the Actions UI:
# workflow_dispatch -> publish_channel=stable, release_tag=vX.Y.Z
```
