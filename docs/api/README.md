# API Reference

## Purpose
Use this folder for Gaussian Splatting API references and regeneration scripts.

## Usage
<table>
  <thead>
    <tr>
      <th>Task</th>
      <th>Document</th>
      <th>Source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Read public script API extracted from <code>.gd</code> files (test, internal tooling, and benchmark harness scripts excluded by default).</td>
      <td><a href="gdscript_reference.md"><code>gdscript_reference.md</code></a></td>
      <td><code>scripts/extract_gdscript_docs.py</code></td>
    </tr>
    <tr>
      <td>Read shader functions and uniform blocks extracted from GLSL files (undocumented entries omitted by default).</td>
      <td><a href="shader_reference.md"><code>shader_reference.md</code></a></td>
      <td><code>scripts/generate_shader_docs.py</code></td>
    </tr>
    <tr>
      <td>Read the maintained node-level API guide.</td>
      <td><a href="gaussian_splat_node3d.md"><code>gaussian_splat_node3d.md</code></a></td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:84</code></td>
    </tr>
    <tr>
      <td>Read the GaussianData API reference.</td>
      <td><a href="gaussian_data.md"><code>gaussian_data.md</code></a></td>
      <td><code>modules/gaussian_splatting/core/gaussian_data.cpp</code></td>
    </tr>
    <tr>
      <td>Read the GaussianSplatAsset API reference.</td>
      <td><a href="gaussian_splat_asset.md"><code>gaussian_splat_asset.md</code></a></td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_asset.cpp</code></td>
    </tr>
    <tr>
      <td>Read the GaussianSplatWorld3D API reference.</td>
      <td><a href="gaussian_splat_world3d.md"><code>gaussian_splat_world3d.md</code></a></td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_world3d.cpp</code></td>
    </tr>
    <tr>
      <td>Read the GaussianSplatContainer API reference.</td>
      <td><a href="gaussian_splat_container.md"><code>gaussian_splat_container.md</code></a></td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_container.cpp</code></td>
    </tr>
    <tr>
      <td>Read the GaussianSplatDynamicInstance3D API reference.</td>
      <td><a href="gaussian_splat_dynamic_instance3d.md"><code>gaussian_splat_dynamic_instance3d.md</code></a></td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_dynamic_instance3d.cpp</code></td>
    </tr>
    <tr>
      <td>Read the GaussianSplatManager API reference.</td>
      <td><a href="gaussian_splat_manager.md"><code>gaussian_splat_manager.md</code></a></td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp</code></td>
    </tr>
    <tr>
      <td>Read the GaussianSplatRenderer API reference.</td>
      <td><a href="gaussian_splat_renderer.md"><code>gaussian_splat_renderer.md</code></a></td>
      <td><code>modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp</code></td>
    </tr>
    <tr>
      <td>Read project setting controls used by the module.</td>
      <td><a href="../reference/project-settings.md"><code>../reference/project-settings.md</code></a></td>
      <td><code>docs/reference/project-settings.md</code></td>
    </tr>
    <tr>
      <td>Browse internal infrastructure classes that have no dedicated API pages.</td>
      <td><a href="internal-classes.md"><code>internal-classes.md</code></a></td>
      <td><code>modules/gaussian_splatting/register_types.cpp</code></td>
    </tr>
  </tbody>
</table>

## API
<table>
  <thead>
    <tr>
      <th>Artifact</th>
      <th>Type</th>
      <th>Regeneration entrypoint</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>gdscript_reference.md</code></td>
      <td>Generated</td>
      <td><code>scripts/extract_gdscript_docs.py</code></td>
    </tr>
    <tr>
      <td><code>shader_reference.md</code></td>
      <td>Generated</td>
      <td><code>scripts/generate_shader_docs.py</code></td>
    </tr>
    <tr>
      <td><code>gaussian_splat_node3d.md</code></td>
      <td>Maintained</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.h:341</code></td>
    </tr>
  </tbody>
</table>

## Examples
```bash
python3 scripts/extract_gdscript_docs.py
python3 scripts/extract_gdscript_docs.py --scope all
python3 scripts/generate_shader_docs.py
python3 scripts/generate_shader_docs.py --include-undocumented
python3 scripts/docs/check_links.py docs/api
```

### Strict mode

Use `--strict` to fail the script when undocumented entries exceed a threshold.
Because many shaders are not yet fully commented, bare `--strict` (which
defaults to zero allowed undocumented entries) will exit non-zero.  Pass
explicit thresholds to use it as a CI gate while documentation is still being
expanded:

```bash
# Allow up to 40 undocumented functions and 10 undocumented fields
python3 scripts/generate_shader_docs.py --strict \
    --max-undocumented-functions 40 \
    --max-undocumented-fields 10
```

## Troubleshooting
<table>
  <thead>
    <tr>
      <th>Problem</th>
      <th>Action</th>
      <th>Source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Internal links fail in this folder.</td>
      <td>Run <code>python3 scripts/docs/check_links.py docs/api</code> and fix reported links.</td>
      <td><code>scripts/docs/check_links.py</code></td>
    </tr>
    <tr>
      <td>Node API docs drift from code.</td>
      <td>Reconcile methods, properties, and signals against <code>_bind_methods()</code>.</td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:84</code></td>
    </tr>
    <tr>
      <td>Generated GDScript reference includes internal/test/tooling scripts.</td>
      <td>Regenerate with default <code>--scope public</code> (or use explicit <code>--scope all</code> only for internal audits).</td>
      <td><code>scripts/extract_gdscript_docs.py</code></td>
    </tr>
    <tr>
      <td>Shader docs generation fails in strict mode.</td>
      <td>Add missing shader comments in GLSL sources, or temporarily adjust strict thresholds for audit-only runs.</td>
      <td><code>scripts/generate_shader_docs.py --strict</code></td>
    </tr>
  </tbody>
</table>
