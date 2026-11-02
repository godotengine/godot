# WebGPU dependency references

Tint (from Dawn), SPIRV-Tools, SPIRV-Headers and Emdawnwebgpu are pinned by
upstream revision, archive URL, SHA-256 and source selection rules in
[`dependencies.json`](dependencies.json). Their source files and generated tables
are ignored by Git. Local compatibility patches remain in `thirdparty/*/patches`.
Native Dawn already uses the separately pinned SDK described in
[`native/README.md`](native/README.md).

SCons prepares the sources when `webgpu=yes`, before configuring the platform,
and only those the target platform uses (Emdawnwebgpu is web-only). The host Tint
CLI prepares them before compiling. Builds with WebGPU disabled require no
dependency download. Preparation can also run explicitly from this
Godot checkout:

```text
uv run --no-project python drivers/webgpu/dependencies.py
uv run --no-project python drivers/webgpu/dependencies.py --verify
```

Preparation verifies each archive, extracts only the selected files, regenerates
SPIRV-Tools' tables using its upstream Python generators, and applies the sorted
patch stack with `git apply`. The legacy Emscripten JavaScript is derived from
the original release. Selection rules exclude unused backends, tests and build
tools; locked file counts catch accidental changes to the imported subset.
`sources.py` supplies the same compiler input selection to SCons and the Tint CLI.

`thirdparty/.webgpu-cache` caches archives by digest. Set `WEBGPU_DEPS_CACHE` or
pass `--cache <directory>` to share a cache between worktrees or prepopulate a
CI machine. Once the archives have been cached, set `WEBGPU_DEPS_OFFLINE=1` for
SCons, or use:

```text
uv run --no-project python drivers/webgpu/dependencies.py --offline --cache <directory>
```

Prepared checkouts build without network access. Prepared files are replaced
atomically, so concurrent builds of one checkout never read partial sources. `--verify` checks fingerprints
and every prepared file's hash without downloading or changing files. Normal
preparation preserves timestamps for unchanged sources and refuses to replace
locally modified files recorded by an earlier preparation. Save changes as patch
files before changing the pins or switching between different patch stacks.

To update a dependency, change its revision, URL, checksum and selection rules in the
lock file, rebase its patches, prepare, and verify. Tint's SPIRV-Tools and header
revisions should follow Dawn's `DEPS`. Keep the table-generator grammar list in
`dependencies.py` aligned with the pinned SPIRV-Tools `source/CMakeLists.txt`.
Update `thirdparty/README.md` and rerun the WebGPU compiler and renderer checks
when changing dependency behavior.

The SPIR-V headers use the pinned upstream sources with only the extension enums
required by SPIRV-Tools added to the C++11 header. Historical rollbacks of the C
and C++ headers and license trimming are unnecessary.

This uses archive references instead of submodules so the patched, trimmed Tint
sources do not require a separately maintained Dawn fork. Existing Git history
still contains the earlier vendored snapshots; neither WebGPU branch is rewritten.
