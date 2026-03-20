# GDScript API Reference

Last generated: 2026-03-19

Scope: `public`

Scripts scanned: `9`

## Script

```
scripts/ab_pipeline_features.gd
```

### Class

```
ab_pipeline_features
```

<table>
  <thead>
    <tr>
      <th>Member</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>_apply_config(cfg: Dictionary)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_build_configs()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_find_target_node()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_init()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_make_config(name: String, tighter: bool, packed: bool, amortize: bool)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_parse_args()</code></pre></td>
      <td>Parses --key=value arguments from the command line. @return Dictionary of parsed arguments.</td>
    </tr>
    <tr>
      <td><pre><code>_process(_delta: float)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_record_result()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_start_next_config()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_write_csv()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_write_results()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
  </tbody>
</table>

## Script

```
scripts/bake_gsplatworld.gd
```

### Class

```
bake_gsplatworld
```

<table>
  <thead>
    <tr>
      <th>Member</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>_apply_chunk_size(container: GaussianSplatContainer, args: Dictionary)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_bake_from_inputs(args: Dictionary)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_bake_from_scene(args: Dictionary)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_ensure_container_assets(container: GaussianSplatContainer)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_find_container(root: Node, container_path: String)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_init()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_load_asset(path: String)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_parse_args()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_print_usage()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_split_list(value: String)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
  </tbody>
</table>

## Script

```
scripts/capture_pipeline_baseline.gd
```

### Class

```
capture_pipeline_baseline
```

<table>
  <thead>
    <tr>
      <th>Member</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>_dump_baseline()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_find_target_node()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_init()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_parse_args()</code></pre></td>
      <td>Parses --key=value arguments from the command line. @return Dictionary of parsed arguments.</td>
    </tr>
    <tr>
      <td><pre><code>_process(_delta: float)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_start_capture()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_vector2i_to_array(value)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
  </tbody>
</table>

## Script

```
scripts/core/gaussian_splatting_manager.gd
```

### Class

```
gaussian_splatting_manager
```

<table>
  <thead>
    <tr>
      <th>Member</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>_deferred_gpu_init()</code></pre></td>
      <td>Allocates a local RenderingDevice and prints adapter information.</td>
    </tr>
    <tr>
      <td><pre><code>_gs_allow_log(key: String)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_gs_debug_flag()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_gs_log_info(message: String, key: String)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_initialize_gpu_resources()</code></pre></td>
      <td>Schedules GPU initialization after the rendering server is ready.</td>
    </tr>
    <tr>
      <td><pre><code>_process(_delta: float)</code></pre></td>
      <td>Updates frame counters and emits periodic FPS metrics. @param _delta: Frame delta in seconds.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Initializes GPU resources for the Gaussian Splatting manager.</td>
    </tr>
    <tr>
      <td><pre><code>get_performance_stats()</code></pre></td>
      <td>Returns performance metrics including sort/render times and GPU memory usage.</td>
    </tr>
    <tr>
      <td><pre><code>load_compute_shaders()</code></pre></td>
      <td>Validates availability of embedded radix-sort compute kernels.</td>
    </tr>
    <tr>
      <td><pre><code>sort_keys_gpu(keys: PackedInt32Array, values: PackedInt32Array = PackedInt32Array()</code></pre></td>
      <td>Sorts key/value pairs using the GPU radix sort pipeline (CPU fallback for now). @param keys: Keys to sort. @param values: Optional values to keep in sync with keys. @return Sorted keys array.</td>
    </tr>
  </tbody>
</table>

## Script

```
scripts/culling_investigation_path.gd
```

### Class

```
culling_investigation_path
```

<table>
  <thead>
    <tr>
      <th>Member</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>_capture_frame()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_drive_camera()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_dump_report()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_find_camera()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_find_target_node()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_init()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_parse_args()</code></pre></td>
      <td>Parses --key=value arguments from the command line. @return Dictionary of parsed arguments.</td>
    </tr>
    <tr>
      <td><pre><code>_process(_delta: float)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_start_capture()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_vector3_to_array(value)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
  </tbody>
</table>

## Script

```
scripts/export_gaussian_scene.gd
```

### Class

```
export_gaussian_scene
```

<table>
  <thead>
    <tr>
      <th>Member</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>_init()</code></pre></td>
      <td>Parses command-line arguments and exports Gaussian data to a .gsf file.</td>
    </tr>
    <tr>
      <td><pre><code>_parse_args()</code></pre></td>
      <td>Parses --key=value arguments from the command line. @return Dictionary of parsed arguments.</td>
    </tr>
  </tbody>
</table>

## Script

```
scripts/import_gaussian_scene.gd
```

### Class

```
import_gaussian_scene
```

<table>
  <thead>
    <tr>
      <th>Member</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>_init()</code></pre></td>
      <td>Merges a baseline .gsf with optional incremental edits and writes output.</td>
    </tr>
    <tr>
      <td><pre><code>_parse_args()</code></pre></td>
      <td>Parses --key=value arguments from the command line. @return Dictionary of parsed arguments.</td>
    </tr>
  </tbody>
</table>

## Script

```
scripts/tools/capture_painterly_references.gd
```

### Class

```
capture_painterly_references
```

<table>
  <thead>
    <tr>
      <th>Member</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>_camera_indices_for_scene(scene: Dictionary)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_ensure_directory(path: String)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_init()</code></pre></td>
      <td>Entry point for headless capture: generates references then exits.</td>
    </tr>
    <tr>
      <td><pre><code>_join_string_array(items: Array)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_parse_args()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_parse_image_size(token: String)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_print_usage()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_record_failure(message: String)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_record_warning(message: String)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_save_image(image: Image, filepath: String)</code></pre></td>
      <td>Saves a rendered reference image into the output directory. @param image: Image to save. @param filepath: Target filepath. @return true when save succeeds.</td>
    </tr>
    <tr>
      <td><pre><code>_slugify(value: String)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_write_manifest()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>capture_all()</code></pre></td>
      <td>Renders each painterly scene and saves camera reference images to disk.</td>
    </tr>
  </tbody>
</table>

## Script

```
scripts/tools/run_painterly_regression.gd
```

### Class

```
run_painterly_regression
```

<table>
  <thead>
    <tr>
      <th>Member</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><pre><code>_ensure_directory(path: String)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_init()</code></pre></td>
      <td>Entry point for the regression harness; runs checks and exits with status.</td>
    </tr>
    <tr>
      <td><pre><code>_join_string_array(items: Array)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_parse_args()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_parse_image_size(token: String)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_print_usage()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_record_failure(scope: String, message: String)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_record_warning(scope: String, message: String)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_run_checks()</code></pre></td>
      <td>Executes shader compilation and image-based sanity checks for painterly scenes.</td>
    </tr>
    <tr>
      <td><pre><code>_save_scene_artifacts(scene_name: String, scene: Dictionary, splats: Array, pre_rendered: Dictionary)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_slugify(value: String)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_validate_metric_bounds(scene_name: String, metric_name: String, value: float, minimum: float, maximum: float)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_write_summary()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
  </tbody>
</table>

Generated by:

```
scripts/extract_gdscript_docs.py
```
