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
      <td>Read script API extracted from <code>.gd</code> files.</td>
      <td><a href="gdscript_reference.md"><code>gdscript_reference.md</code></a></td>
      <td><code>scripts/extract_gdscript_docs.py:75</code></td>
    </tr>
    <tr>
      <td>Read shader functions and uniform blocks extracted from GLSL files.</td>
      <td><a href="shader_reference.md"><code>shader_reference.md</code></a></td>
      <td><code>scripts/generate_shader_docs.py:223</code></td>
    </tr>
    <tr>
      <td>Read the maintained node-level API guide.</td>
      <td><a href="gaussian_splat_node3d.md"><code>gaussian_splat_node3d.md</code></a></td>
      <td><code>modules/gaussian_splatting/nodes/gaussian_splat_node_3d.cpp:84</code></td>
    </tr>
    <tr>
      <td>Read project setting controls used by the module.</td>
      <td><a href="../reference/project-settings.md"><code>../reference/project-settings.md</code></a></td>
      <td><code>docs/reference/project-settings.md</code></td>
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
      <td><code>scripts/extract_gdscript_docs.py:117</code></td>
    </tr>
    <tr>
      <td><code>shader_reference.md</code></td>
      <td>Generated</td>
      <td><code>scripts/generate_shader_docs.py:274</code></td>
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
python3 scripts/generate_shader_docs.py
python3 scripts/docs/check_links.py docs/api
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
  </tbody>
</table>
