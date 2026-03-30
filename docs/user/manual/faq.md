# FAQ

## Why is the node visible in one project but not another?

Most often the editor binary was not built with `modules/gaussian_splatting`.

## Why do tests fail with a stock Godot binary?

Module-dependent tests require the module-enabled editor binary. Pass `--godot` or `--godot-binary`.

## Which page is canonical for commands?

[Build / Test / CI Command Reference](../../reference/build-test-ci.md)

## Where do I report recurring issues?

Use repository issues and include:
- command used
- platform/GPU/driver
- relevant logs/artifacts
