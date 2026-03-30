# Development Documentation

## Purpose

Find build, test, architecture, and contributor workflow references.

## Usage

| Need | Guide |
| --- | --- |
| Build command and platform prerequisites | [Build from Source](../BUILDING.md) |
| Build failure diagnosis | [Recurring issues](../troubleshooting/recurring-issues.md) |
| Test environment setup | [Testing setup guide](../testing/setup-guide.md) |
| Versioned docs site build and deploy | [Docs site maintenance guide](docs-site.md) |
| Screenshot capture process | [Screenshot capture spec](screenshot-capture-spec.md) |
| Contribution policy | [Contribution guide](../../CONTRIBUTING.md) |
| Module architecture overview | [Module README](../../modules/gaussian_splatting/README.md) |
| Documentation style policy | [Documentation style guide](../style/documentation-style-guide.md) |
| CI workflow overview | [Workflow overview](../../.github/workflows/README.md) |

## Examples

```bash
scons platform=linuxbsd target=editor dev_build=yes -n
python3 tests/ci/run_module_tests.py --guard-only
python3 scripts/build_documentation.py --help
```
