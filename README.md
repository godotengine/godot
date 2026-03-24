# Godot Gaussian Splatting

https://github.com/user-attachments/assets/71542f8d-ccf6-433c-b920-66fdf9ae8c84

GodotGS is a Godot 4.5 fork with an in-tree Gaussian Splatting module.

## Start Here

### Artists and Non-Programmers

- First run and first visible result: [docs/user/quickstart.md](docs/user/quickstart.md)
- Practical usage guides: [docs/user/manual/](docs/user/manual/)
- Recurring fixes: [docs/troubleshooting/recurring-issues.md](docs/troubleshooting/recurring-issues.md)

### Contributors and Engineers

- Setup and first safe PR: [docs/contributor/onboarding.md](docs/contributor/onboarding.md)
- Canonical build/test/CI commands: [docs/reference/build-test-ci.md](docs/reference/build-test-ci.md)
- Contribution rules: [docs/governance/contribution-standards.md](docs/governance/contribution-standards.md)

## Documentation

- Canonical docs index: [docs/index.md](docs/index.md)
- Versioned docs site pipeline: [docs/development/docs-site.md](docs/development/docs-site.md)
- Architecture overview: [docs/architecture/overview.md](docs/architecture/overview.md)
- API index: [docs/api/README.md](docs/api/README.md)
- Archive policy and historical materials: [docs/archive/README.md](docs/archive/README.md)

## Repository Layout

- Root engine directories such as [`core/`](core/), [`editor/`](editor/), [`scene/`](scene/), [`servers/`](servers/), and [`thirdparty/`](thirdparty/): upstream Godot now lives at repository root.
- [modules/gaussian_splatting/](modules/gaussian_splatting/): module implementation.
- [tests/](tests/): CI and runtime validation harnesses.
- [docs/](docs/): user, contributor, architecture, and reference docs.

## License

Repository code and documentation are MIT-licensed unless noted otherwise.
Upstream engine code at repository root follows upstream licensing.
