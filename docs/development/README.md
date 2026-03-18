# Development Documentation

_Last updated: 2026-02-20_

## Purpose

Find build, test, architecture, and contributor workflow references.

## Usage

| Need | Path |
| --- | --- |
| Build command and platform prerequisites | `../BUILDING.md` |
| Build failure diagnosis | `../troubleshooting/build-troubleshooting.md` |
| Test environment setup | `../testing/setup-guide.md` |
| Contribution policy | `../../CONTRIBUTING.md` |
| Module architecture overview | `../../modules/gaussian_splatting/README.md` |
| Documentation style policy | `../style/documentation-style-guide.md` |
| CI workflow overview | `../../.github/workflows/README.md` |

## Examples

```bash
scons platform=linuxbsd target=editor dev_build=yes -n
python3 tests/ci/run_module_tests.py --guard-only
python3 scripts/build_documentation.py --help
```
