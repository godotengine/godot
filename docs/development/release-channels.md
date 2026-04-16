# Release Channels

## Purpose

Define how executable builds are distributed from this repository and what guarantees each channel provides.

## Current Public Reality

- Public GitHub releases are currently nightlies first.
- The stable `v*` publish path exists in automation, but it is not the default public install track.
- Visible public binaries cover the Linux editor (`tar.xz`) and the Windows editor (`zip`).
- Preferred evaluation path: use the latest Linux or Windows nightly. macOS still requires building from source.

## Channels

| Channel | Source | Current public reality | Intended audience | Stability expectation |
| --- | --- | --- | --- | --- |
| CI artifact | `.github/workflows/release_builds.yml` on PR/push | Builds Linux and Windows contributor artifacts, not a public install surface | Contributors validating branch changes | No compatibility guarantee; debugging/testing only |
| Nightly prerelease | `.github/workflows/release_builds.yml` schedule/manual with `publish_channel=nightly` | Primary visible public binary channel at present; publishes Linux and Windows editor assets every night | Early testers validating latest integration changes | May break at any time; not for production |
| Stable tag path | Tag push `v*` or manual with `publish_channel=stable` | Automation path exists, but users should only treat it as active when a visible `v*` release is actually published | Later versioned drops and downstream integrators | Intended stable path when used, but not the default public install track |

## Trigger Model

| Trigger | Behavior |
| --- | --- |
| Pull request touching engine/module paths | Build the Linux editor on a GitHub-hosted runner and upload its contributor artifact. The Windows editor lane is intentionally skipped for pull requests because it runs on a self-hosted runner and must not execute untrusted PR code. |
| Push to `master`/`main`/`develop` | Build Linux and Windows editors and upload contributor artifacts |
| Nightly schedule (`30 2 * * *` UTC) | Build and publish nightly prerelease (`nightly-YYYYMMDD`) with Linux and Windows editor assets, then prune old nightlies |
| Tag push (`v*`) | Build and publish a versioned stable-tag release with Linux and Windows assets when that path is intentionally used |
| Manual dispatch | Optional publish mode: `none`, `nightly`, or `stable` |

## Package Contents

Each publish includes:

- Linux editor tarball (`godotgs-linux-x86_64-<tag-or-sha>.tar.xz`)
- Windows editor zip (`godotgs-windows-x86_64-<tag-or-sha>.zip`)
- SHA-256 checksum files
- `BUILD-INFO.txt` metadata (channel, commit, binary name, generation timestamp)

## Current Limits and Caveats

| Area | Constraint |
| --- | --- |
| GitHub Release assets | Individual file size must remain under 2 GiB |
| Actions artifacts | Retention window is bounded by GitHub settings and this workflow's retention setting |
| Public install guidance | Treat nightlies as the default public install path until a visible `v*` series exists on Releases/Tags |
| Stable wording | Do not describe the repo as having a stable public channel unless a versioned release is actually visible |
| Platform coverage (current) | Visible public binaries cover Linux and Windows editors; macOS currently requires source builds |

## Manual Examples

```bash
# Manual nightly publish from the Actions UI:
# workflow_dispatch -> publish_channel=nightly

# Manual stable publish from the Actions UI:
# workflow_dispatch -> publish_channel=stable, release_tag=vX.Y.Z
```
