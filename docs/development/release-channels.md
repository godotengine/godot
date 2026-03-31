# Release Channels

## Purpose

Define how executable builds are distributed from this repository and what guarantees each channel provides.

## Current Public Reality

- Public GitHub releases are currently nightlies first.
- The stable `v*` publish path exists in automation, but it is not the default public install track.
- Public binaries currently cover the Linux editor only.
- Preferred evaluation path: use the latest Linux nightly if it fits your environment, otherwise build from source.

## Channels

| Channel | Source | Current public reality | Intended audience | Stability expectation |
| --- | --- | --- | --- | --- |
| CI artifact | `.github/workflows/release_builds.yml` on PR/push | Exists for contributors, not as a public install surface | Contributors validating branch changes | No compatibility guarantee; debugging/testing only |
| Nightly prerelease | `.github/workflows/release_builds.yml` schedule/manual with `publish_channel=nightly` | Primary visible public binary channel at present | Early testers validating latest integration changes | May break at any time; not for production |
| Stable tag path | Tag push `v*` or manual with `publish_channel=stable` | Automation path exists, but users should only treat it as active when a visible `v*` release is actually published | Later versioned drops and downstream integrators | Intended stable path when used, but not the default public install track |

## Trigger Model

| Trigger | Behavior |
| --- | --- |
| Pull request touching engine/module paths | Build Linux editor and upload contributor artifact |
| Push to `master`/`main`/`develop` | Build Linux editor and upload contributor artifact |
| Nightly schedule (`30 2 * * *` UTC) | Build and publish nightly prerelease (`nightly-YYYYMMDD`) then prune old nightlies |
| Tag push (`v*`) | Build and publish a versioned stable-tag release when that path is intentionally used |
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
| Public install guidance | Treat nightlies as the default public install path until a visible `v*` series exists on Releases/Tags |
| Stable wording | Do not describe the repo as having a stable public channel unless a versioned release is actually visible |
| Platform coverage (current) | Linux editor binary only; Windows and macOS currently require source builds |

## Manual Examples

```bash
# Manual nightly publish from the Actions UI:
# workflow_dispatch -> publish_channel=nightly

# Manual stable publish from the Actions UI:
# workflow_dispatch -> publish_channel=stable, release_tag=vX.Y.Z
```
