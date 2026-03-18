# GDScript API Reference

Last generated: 2026-02-13

## Script

```
modules/gaussian_splatting/tests/painterly_scenes/painterly_demo_scene.gd
```

### Class

```
painterly_demo_scene
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
      <td><pre><code>_display_image(image: Image)</code></pre></td>
      <td>Updates the sprite texture with the rendered image. @param image: Rendered image to display.</td>
    </tr>
    <tr>
      <td><pre><code>_process(delta: float)</code></pre></td>
      <td>Advances the camera animation (if enabled) and re-renders the current frame. @param delta: Frame delta in seconds.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Loads the painterly scene definition, compiles shader permutations, and renders the initial frame.</td>
    </tr>
    <tr>
      <td><pre><code>_render_frame(camera_index: int)</code></pre></td>
      <td>Renders the scene from the specified camera keyframe index. @param camera_index: Index into the scene camera path.</td>
    </tr>
  </tbody>
</table>

## Script

```
modules/gaussian_splatting/tests/painterly_scenes/painterly_scene_util.gd
```

### Class

```
PainterlySceneUtil
```

No documented functions.

## Script

```
modules/gaussian_splatting/tests/test_gpu_sorting_performance.gd
```

### Class

```
test_gpu_sorting_performance
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
      <td><pre><code>_ready()</code></pre></td>
      <td>Initializes the renderer and runs GPU sorting benchmarks for each algorithm.</td>
    </tr>
    <tr>
      <td><pre><code>benchmark_algorithm(algorithm_name: String)</code></pre></td>
      <td>Benchmarks the configured sorting method across all test sizes. @param algorithm_name: Label for the algorithm under test.</td>
    </tr>
    <tr>
      <td><pre><code>create_test_data(size: int)</code></pre></td>
      <td>Generates deterministic Gaussian test data for benchmarking. @param size: Number of splats to generate. @return Dictionary with GaussianData and positions.</td>
    </tr>
    <tr>
      <td><pre><code>generate_performance_report()</code></pre></td>
      <td>Prints a summary table of benchmark results to stdout.</td>
    </tr>
    <tr>
      <td><pre><code>save_results_to_file()</code></pre></td>
      <td>Writes benchmark results to a JSON file in the user directory.</td>
    </tr>
    <tr>
      <td><pre><code>validate_speedups()</code></pre></td>
      <td>Validates speedup expectations for the largest dataset.</td>
    </tr>
  </tbody>
</table>

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

## Script

```
templates/gaussian_splat_template/autoload/gaussian_bootstrap.gd
```

### Class

```
gaussian_bootstrap
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
      <td><pre><code>_log_runtime_configuration()</code></pre></td>
      <td>Prints adapter and sorting configuration information for diagnostics.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Resolves the GaussianSplatManager singleton and logs runtime configuration.</td>
    </tr>
    <tr>
      <td><pre><code>ensure_submission_lock()</code></pre></td>
      <td>Acquires a submission lock from the manager when supported. @return Lock object or null if unavailable.</td>
    </tr>
    <tr>
      <td><pre><code>get_global_stats()</code></pre></td>
      <td>Returns global renderer statistics when available.</td>
    </tr>
  </tbody>
</table>

## Script

```
templates/gaussian_splat_template/scripts/camera/orbit_camera_rig.gd
```

### Class

```
OrbitCameraRig
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
      <td><pre><code>_handle_mouse_button(event: InputEventMouseButton)</code></pre></td>
      <td>Updates orbit/pan state and applies zoom on wheel input. @param event: Mouse button event.</td>
    </tr>
    <tr>
      <td><pre><code>_handle_mouse_motion(event: InputEventMouseMotion)</code></pre></td>
      <td>Applies orbit rotation or panning based on the current input state. @param event: Mouse motion event.</td>
    </tr>
    <tr>
      <td><pre><code>_physics_process(delta: float)</code></pre></td>
      <td>Handles keyboard-driven translation for the orbit rig. @param delta: Frame delta in seconds.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Initializes the camera reference and cached orbit angles.</td>
    </tr>
    <tr>
      <td><pre><code>_unhandled_input(event: InputEvent)</code></pre></td>
      <td>Dispatches mouse events to orbit or pan handlers. @param event: Input event from the scene tree.</td>
    </tr>
    <tr>
      <td><pre><code>_zoom(amount: float)</code></pre></td>
      <td>Moves the camera along its local forward axis. @param amount: Positive or negative zoom distance.</td>
    </tr>
    <tr>
      <td><pre><code>focus(bounds: AABB)</code></pre></td>
      <td>Repositions the rig to frame the provided bounds. @param bounds: Axis-aligned bounds to focus on.</td>
    </tr>
  </tbody>
</table>

## Script

```
templates/gaussian_splat_template/scripts/main_scene.gd
```

### Class

```
GaussianTemplateRoot
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
      <td><pre><code>_configure_gaussian_node()</code></pre></td>
      <td>Applies template defaults to the GaussianSplatNode3D instance.</td>
    </tr>
    <tr>
      <td><pre><code>_focus_camera()</code></pre></td>
      <td>Centers the orbit camera on the current Gaussian bounds.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Configures the template scene by wiring the node, overlay, and camera focus.</td>
    </tr>
    <tr>
      <td><pre><code>_wire_overlay()</code></pre></td>
      <td>Binds the overlay to the gaussian node and camera rig.</td>
    </tr>
  </tbody>
</table>

## Script

```
templates/gaussian_splat_template/scripts/ui/performance_overlay.gd
```

### Class

```
GaussianPerformanceOverlay
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
      <td><pre><code>_colorize_buffer_percent(percent: float, value_str: String)</code></pre></td>
      <td>Returns color-coded string for buffer usage percentage</td>
    </tr>
    <tr>
      <td><pre><code>_colorize_gpu_time(time_ms: float, value_str: String)</code></pre></td>
      <td>Returns color-coded string for GPU timing (ms)</td>
    </tr>
    <tr>
      <td><pre><code>_colorize_lod_reduction(percent: float, value_str: String)</code></pre></td>
      <td>Returns color-coded string for LOD reduction percentage (higher = more aggressive = red)</td>
    </tr>
    <tr>
      <td><pre><code>_colorize_vram_percent(percent: float, value_str: String)</code></pre></td>
      <td>Returns color-coded string for VRAM usage percentage</td>
    </tr>
    <tr>
      <td><pre><code>_format_compute_policy(policy: int)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_format_number(value: int)</code></pre></td>
      <td>Formats large numbers with K/M suffixes for readability</td>
    </tr>
    <tr>
      <td><pre><code>_process(delta: float)</code></pre></td>
      <td>Tracks frame timing and refreshes the overlay at the configured interval. @param delta: Frame delta in seconds.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Enables processing so the overlay refreshes at runtime.</td>
    </tr>
    <tr>
      <td><pre><code>_refresh_overlay()</code></pre></td>
      <td>Rebuilds the overlay text with the latest renderer statistics using Custom Performance Monitors.</td>
    </tr>
    <tr>
      <td><pre><code>_toggle_compute_raster_policy()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_try_resolve_camera()</code></pre></td>
      <td>Resolves the camera from the configured NodePath when missing.</td>
    </tr>
    <tr>
      <td><pre><code>_unhandled_input(event: InputEvent)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>set_camera_node(node: Node3D)</code></pre></td>
      <td>Assigns the camera rig node used for pose reporting. @param node: Camera rig node.</td>
    </tr>
    <tr>
      <td><pre><code>set_gaussian_node(node: GaussianSplatNode3D)</code></pre></td>
      <td>Assigns the Gaussian node used for statistics queries. @param node: GaussianSplatNode3D to monitor.</td>
    </tr>
  </tbody>
</table>

## Script

```
test_data/demo_controller.gd
```

### Class

```
demo_controller
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
      <td><pre><code>_clear_all_splats()</code></pre></td>
      <td>Removes and frees all currently loaded splat nodes.</td>
    </tr>
    <tr>
      <td><pre><code>_cycle_quality()</code></pre></td>
      <td>Cycles through quality presets in order.</td>
    </tr>
    <tr>
      <td><pre><code>_format_number(num: int)</code></pre></td>
      <td>Formats large numbers into human-readable K/M strings. @param num: Value to format. @return Formatted string.</td>
    </tr>
    <tr>
      <td><pre><code>_handle_camera_movement(delta)</code></pre></td>
      <td>Applies keyboard-driven camera translation. @param delta: Frame delta in seconds.</td>
    </tr>
    <tr>
      <td><pre><code>_highlight_quality_button(index: int)</code></pre></td>
      <td>Highlights the active quality button and resets others. @param index: Button index to highlight.</td>
    </tr>
    <tr>
      <td><pre><code>_input(event)</code></pre></td>
      <td>Handles camera mouse look, hotkeys, and quality toggles. @param event: Input event dispatched by the scene tree.</td>
    </tr>
    <tr>
      <td><pre><code>_instantiate_splat_node()</code></pre></td>
      <td>Instantiates a GaussianSplatNode3D and validates the module is available. @return New splat node or null when unavailable.</td>
    </tr>
    <tr>
      <td><pre><code>_load_splat_file(file_path: String, position: Vector3)</code></pre></td>
      <td>Instantiates a splat node for the given file and places it in the scene. @param file_path: PLY file path to load. @param position: World position to place the node.</td>
    </tr>
    <tr>
      <td><pre><code>_notification(what)</code></pre></td>
      <td>Handles cleanup when the window close request is received. @param what: Notification identifier.</td>
    </tr>
    <tr>
      <td><pre><code>_on_load_100k()</code></pre></td>
      <td>UI callback to load the 100K splat dataset.</td>
    </tr>
    <tr>
      <td><pre><code>_on_load_1k()</code></pre></td>
      <td>UI callback to load the 1K splat dataset.</td>
    </tr>
    <tr>
      <td><pre><code>_on_load_1m()</code></pre></td>
      <td>UI callback to load the 1M splat dataset.</td>
    </tr>
    <tr>
      <td><pre><code>_on_load_multiple()</code></pre></td>
      <td>UI callback to load multiple splat instances at preset positions.</td>
    </tr>
    <tr>
      <td><pre><code>_on_quality_high()</code></pre></td>
      <td>UI callback to switch to high quality preset.</td>
    </tr>
    <tr>
      <td><pre><code>_on_quality_low()</code></pre></td>
      <td>UI callback to switch to low quality preset.</td>
    </tr>
    <tr>
      <td><pre><code>_on_quality_medium()</code></pre></td>
      <td>UI callback to switch to medium quality preset.</td>
    </tr>
    <tr>
      <td><pre><code>_on_quality_ultra()</code></pre></td>
      <td>UI callback to switch to ultra quality preset.</td>
    </tr>
    <tr>
      <td><pre><code>_process(delta)</code></pre></td>
      <td>Updates camera movement, auto-rotation, and UI statistics per frame. @param delta: Frame delta in seconds.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Wires UI signals, configures input capture, and loads the default splat file.</td>
    </tr>
    <tr>
      <td><pre><code>_set_quality(quality: String)</code></pre></td>
      <td>Applies the quality preset to new and existing splat nodes. @param quality: Preset name string.</td>
    </tr>
    <tr>
      <td><pre><code>_update_fps(delta)</code></pre></td>
      <td>Accumulates FPS samples once per second for UI display. @param delta: Frame delta in seconds.</td>
    </tr>
    <tr>
      <td><pre><code>_update_stats()</code></pre></td>
      <td>Updates the on-screen stats label with splat and memory information.</td>
    </tr>
  </tbody>
</table>

## Script

```
test_data/test_phase4_integration.gd
```

### Class

```
test_phase4_integration
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
      <td><pre><code>_create_splat_node()</code></pre></td>
      <td>Creates a GaussianSplatNode3D instance and asserts availability. @return Instantiated splat node.</td>
    </tr>
    <tr>
      <td><pre><code>_init()</code></pre></td>
      <td>Entry point for the phase 4C integration suite.</td>
    </tr>
    <tr>
      <td><pre><code>assert(condition: bool, message: String = "Assertion failed")</code></pre></td>
      <td>Records a failed assertion with context for the current test. @param condition: Boolean condition to validate. @param message: Optional failure message.</td>
    </tr>
    <tr>
      <td><pre><code>calculate_average(values: Array)</code></pre></td>
      <td>Returns the arithmetic mean of numeric values, or 0 for an empty array. @param values: Array of numeric values. @return Average value or 0.0 when empty.</td>
    </tr>
    <tr>
      <td><pre><code>calculate_percentile(values: Array, percentile: float)</code></pre></td>
      <td>Returns the percentile value from a numeric array. @param values: Array of numeric values. @param percentile: Percentile in the range 0.0-1.0. @return Percentile value or 0.0 when empty.</td>
    </tr>
    <tr>
      <td><pre><code>print_summary()</code></pre></td>
      <td>Prints a summary of all executed tests and performance metrics.</td>
    </tr>
    <tr>
      <td><pre><code>run_all_tests()</code></pre></td>
      <td>Runs the full battery of integration tests and exits with a status code.</td>
    </tr>
    <tr>
      <td><pre><code>run_test(test_name: String, test_func: Callable)</code></pre></td>
      <td>Runs a named test case and records pass/fail results. @param test_name: Display name for the test case. @param test_func: Callable to execute (may use await).</td>
    </tr>
    <tr>
      <td><pre><code>test_error_handling()</code></pre></td>
      <td>Ensures invalid data and GPU issues are handled gracefully.</td>
    </tr>
    <tr>
      <td><pre><code>test_memory_management()</code></pre></td>
      <td>Checks CPU/GPU memory usage and allocation behavior across lifecycles.</td>
    </tr>
    <tr>
      <td><pre><code>test_multi_instance()</code></pre></td>
      <td>Verifies multiple Gaussian splat instances can coexist and load independently.</td>
    </tr>
    <tr>
      <td><pre><code>test_performance_benchmarks()</code></pre></td>
      <td>Measures frame timing across datasets and records performance metrics.</td>
    </tr>
    <tr>
      <td><pre><code>test_ply_loading()</code></pre></td>
      <td>Validates loading behavior for PLY files of varying sizes and invalid input.</td>
    </tr>
    <tr>
      <td><pre><code>test_rendering_quality()</code></pre></td>
      <td>Exercises quality presets and transparency sorting toggles.</td>
    </tr>
    <tr>
      <td><pre><code>test_streaming_buffer()</code></pre></td>
      <td>Exercises streaming buffer rotation and LOD updates across distances.</td>
    </tr>
    <tr>
      <td><pre><code>test_visual_regression()</code></pre></td>
      <td>Performs basic visual consistency checks across frames.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/ci/test_gpu_sorting_ci.gd
```

### Class

```
test_gpu_sorting_ci
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
      <td><pre><code>_apply_suite_gates()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_env_flag(name: String, default_value: bool)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_is_ci()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_is_gpu_required_ci_mode()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_is_strict_mode()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Entry point for the GPU sorting CI suite; runs tests and exits with status.</td>
    </tr>
    <tr>
      <td><pre><code>_record_test_result(test_name: String, test_detail: Dictionary, start_time: float)</code></pre></td>
      <td>Updates counters and stores a completed test result entry.</td>
    </tr>
    <tr>
      <td><pre><code>_validation_mode()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>create_test_data(count: int)</code></pre></td>
      <td>Builds deterministic GaussianData for sort tests. @param count: Number of splats to generate. @return GaussianData populated with positions/scales/opacities.</td>
    </tr>
    <tr>
      <td><pre><code>generate_test_report()</code></pre></td>
      <td>Prints a summary report of GPU sorting CI results.</td>
    </tr>
    <tr>
      <td><pre><code>get_expected_max_time(count: int)</code></pre></td>
      <td>Returns expected maximum sort time thresholds for CI.</td>
    </tr>
    <tr>
      <td><pre><code>run_all_tests()</code></pre></td>
      <td>Runs all GPU sorting test cases in sequence.</td>
    </tr>
    <tr>
      <td><pre><code>run_test(test_name: String, test_function: Callable)</code></pre></td>
      <td>Runs a synchronous test and records its result. @param test_name: Name of the test case. @param test_function: Callable returning a result dictionary.</td>
    </tr>
    <tr>
      <td><pre><code>run_test_async(test_name: String, test_function: Callable)</code></pre></td>
      <td>Runs an async test and records its result. @param test_name: Name of the test case. @param test_function: Callable returning a result dictionary.</td>
    </tr>
    <tr>
      <td><pre><code>save_test_results_json()</code></pre></td>
      <td>Persists detailed CI results to a JSON file for later inspection.</td>
    </tr>
    <tr>
      <td><pre><code>test_dataset_sorting(count: int, size_label: String)</code></pre></td>
      <td>Executes a sorting test for a dataset of the given size. @param count: Number of splats to sort. @param size_label: Size label for reporting.</td>
    </tr>
    <tr>
      <td><pre><code>test_large_dataset_sorting()</code></pre></td>
      <td>Runs the dataset sorting test with a large dataset.</td>
    </tr>
    <tr>
      <td><pre><code>test_medium_dataset_sorting()</code></pre></td>
      <td>Runs the dataset sorting test with a medium dataset.</td>
    </tr>
    <tr>
      <td><pre><code>test_performance_validation()</code></pre></td>
      <td>Validates collected performance metrics for basic scaling expectations.</td>
    </tr>
    <tr>
      <td><pre><code>test_radix_sorter()</code></pre></td>
      <td>Confirms the RadixSort class is instantiable.</td>
    </tr>
    <tr>
      <td><pre><code>test_renderer_initialization()</code></pre></td>
      <td>Validates renderer construction and access to render stats.</td>
    </tr>
    <tr>
      <td><pre><code>test_small_dataset_sorting()</code></pre></td>
      <td>Runs the dataset sorting test with a small dataset.</td>
    </tr>
    <tr>
      <td><pre><code>test_sorting_method_config()</code></pre></td>
      <td>Ensures sorting method configuration can be set and queried.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/ci/test_ply_loader_ci.gd
```

### Class

```
test_ply_loader_ci
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
      <td><pre><code>_ready()</code></pre></td>
      <td>Entry point for the PLY loader CI suite; runs tests and exits with status.</td>
    </tr>
    <tr>
      <td><pre><code>generate_test_report()</code></pre></td>
      <td>Prints the PLY loader test summary and persists results.</td>
    </tr>
    <tr>
      <td><pre><code>run_all_tests()</code></pre></td>
      <td>Runs the full set of PLY loader tests.</td>
    </tr>
    <tr>
      <td><pre><code>run_test(test_name: String, test_function: Callable)</code></pre></td>
      <td>Executes a test case and records pass/fail details. @param test_name: Name of the test case. @param test_function: Callable returning a result dictionary.</td>
    </tr>
    <tr>
      <td><pre><code>save_test_results_json()</code></pre></td>
      <td>Writes detailed PLY loader CI results to a JSON file.</td>
    </tr>
    <tr>
      <td><pre><code>test_ply_error_handling()</code></pre></td>
      <td>Ensures invalid file paths return error codes as expected.</td>
    </tr>
    <tr>
      <td><pre><code>test_ply_loading_basic()</code></pre></td>
      <td>Verifies loading of a fixture PLY file into GaussianData.</td>
    </tr>
    <tr>
      <td><pre><code>test_ply_roundtrip()</code></pre></td>
      <td>Confirms saved PLY data can be reloaded with consistent values.</td>
    </tr>
    <tr>
      <td><pre><code>test_ply_saving()</code></pre></td>
      <td>Validates saving GaussianData to a PLY file.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/ci/test_ply_pipeline_ci.gd
```

### Class

```
test_ply_pipeline_ci
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
      <td><pre><code>_apply_suite_gates()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_env_flag(name: String, default_value: bool)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_is_ci()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_is_gpu_required_ci_mode()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Entry point for the PLY pipeline CI suite; runs tests and exits with status.</td>
    </tr>
    <tr>
      <td><pre><code>create_test_ply_file()</code></pre></td>
      <td>Builds and saves a temporary PLY file for integration tests. @return True when the file is saved successfully.</td>
    </tr>
    <tr>
      <td><pre><code>generate_test_report()</code></pre></td>
      <td>Prints the PLY pipeline test summary and persists results.</td>
    </tr>
    <tr>
      <td><pre><code>run_all_tests()</code></pre></td>
      <td>Runs the full set of PLY pipeline integration tests.</td>
    </tr>
    <tr>
      <td><pre><code>run_test(test_name: String, test_function: Callable)</code></pre></td>
      <td>Executes a test case and records pass/fail details. @param test_name: Name of the test case. @param test_function: Callable returning a result dictionary.</td>
    </tr>
    <tr>
      <td><pre><code>save_test_results_json()</code></pre></td>
      <td>Writes detailed PLY pipeline CI results to a JSON file.</td>
    </tr>
    <tr>
      <td><pre><code>test_data_consistency()</code></pre></td>
      <td>Checks round-trip data consistency across asset and data loaders.</td>
    </tr>
    <tr>
      <td><pre><code>test_gaussian_splat_asset_loading()</code></pre></td>
      <td>Validates loading a PLY file into GaussianSplatAsset.</td>
    </tr>
    <tr>
      <td><pre><code>test_ply_loader_integration()</code></pre></td>
      <td>Ensures the PLYLoader populates GaussianData correctly.</td>
    </tr>
    <tr>
      <td><pre><code>test_renderer_integration()</code></pre></td>
      <td>Exercises the end-to-end pipeline through the renderer.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/hello_splat_example.gd
```

### Class

```
hello_splat_example
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
      <td><pre><code>_get_metric(stats: Dictionary, key: String, default_value)</code></pre></td>
      <td>Reads a renderer stat, preferring the telemetry snapshot when available.</td>
    </tr>
    <tr>
      <td><pre><code>_input(event)</code></pre></td>
      <td>Handles simple keyboard shortcuts for toggling renderer options. @param event: Input event dispatched by the scene tree.</td>
    </tr>
    <tr>
      <td><pre><code>_notification(what)</code></pre></td>
      <td>Quits the demo when the window close request is received. @param what: Notification identifier.</td>
    </tr>
    <tr>
      <td><pre><code>_print_performance_stats()</code></pre></td>
      <td>Prints the current renderer statistics to the console.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Builds the demo renderer, configures defaults, and starts periodic stats output.</td>
    </tr>
    <tr>
      <td><pre><code>_setup_camera()</code></pre></td>
      <td>Positions the active 3D camera to frame the default splat cube.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/measure_performance.gd
```

### Class

```
measure_performance
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
      <td><pre><code>_process(delta)</code></pre></td>
      <td>Samples per-frame FPS and prints rolling stats once per second. @param delta: Frame delta in seconds.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Initializes the measurement run by locating the GaussianSplatRenderer node.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/optimize_sorting_thresholds.gd
```

### Class

```
optimize_sorting_thresholds
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
      <td><pre><code>_ready()</code></pre></td>
      <td>Runs the GPU sorting threshold optimization workflow end-to-end.</td>
    </tr>
    <tr>
      <td><pre><code>analyze_optimization_results()</code></pre></td>
      <td>Summarizes the best thresholds and prints a performance breakdown.</td>
    </tr>
    <tr>
      <td><pre><code>apply_optimal_thresholds()</code></pre></td>
      <td>Writes the optimal threshold configuration to a JSON file for reuse.</td>
    </tr>
    <tr>
      <td><pre><code>calculate_overall_score(performance_scores: Dictionary)</code></pre></td>
      <td>Aggregates per-size scores into a weighted overall score. @param performance_scores: Per-size performance results. @return Weighted overall score.</td>
    </tr>
    <tr>
      <td><pre><code>calculate_size_score(time_ms: float, size: int)</code></pre></td>
      <td>Calculates a weighted performance score for a dataset size. @param time_ms: Average sort time in milliseconds. @param size: Dataset size. @return Score between 0 and 1.</td>
    </tr>
    <tr>
      <td><pre><code>create_test_data(size: int)</code></pre></td>
      <td>Creates deterministic Gaussian data for threshold optimization runs. @param size: Number of splats to generate. @return GaussianData populated with positions/colors/scales.</td>
    </tr>
    <tr>
      <td><pre><code>determine_algorithm(size: int, bitonic_threshold: int, radix_threshold: int)</code></pre></td>
      <td>Determines which algorithm would be selected for the given thresholds. @param size: Dataset size. @param bitonic_threshold: Max size for bitonic sorting. @param radix_threshold: Max size for radix sorting. @return Algorithm name.</td>
    </tr>
    <tr>
      <td><pre><code>find_current_best()</code></pre></td>
      <td>Finds the best-performing threshold combination collected so far. @return Dictionary describing the best result.</td>
    </tr>
    <tr>
      <td><pre><code>run_threshold_optimization()</code></pre></td>
      <td>Sweeps threshold combinations and records performance scores per dataset size.</td>
    </tr>
    <tr>
      <td><pre><code>test_threshold_combination(bitonic_threshold: int, radix_threshold: int)</code></pre></td>
      <td>Tests a specific threshold combination across all dataset sizes. @param bitonic_threshold: Max size for bitonic sorting. @param radix_threshold: Max size for radix sorting. @return Performance score dictionary keyed by dataset size.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/test_100_splats.gd
```

### Class

```
test_100_splats
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
      <td><pre><code>_input(event)</code></pre></td>
      <td>Handles hotkeys for quitting or switching sorting methods. @param event: Input event dispatched by the scene tree.</td>
    </tr>
    <tr>
      <td><pre><code>_process(delta)</code></pre></td>
      <td>Rotates the camera and prints periodic FPS measurements. @param delta: Frame delta in seconds.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Creates a 100-splat test scene and configures the renderer for inspection.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/test_compute_pipeline.gd
```

### Class

```
test_compute_pipeline
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
      <td><pre><code>_ready()</code></pre></td>
      <td>Builds a minimal renderer, uploads data, and validates compute pipeline setup.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/test_compute_shader.gd
```

### Class

```
test_compute_shader
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
      <td><pre><code>_ready()</code></pre></td>
      <td>Runs a GPU sort smoke test and prints renderer statistics.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/test_full_pipeline.gd
```

### Class

```
test_full_pipeline
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
      <td><pre><code>_create_test_data()</code></pre></td>
      <td>Generates synthetic splats to validate pipeline behavior without external data.</td>
    </tr>
    <tr>
      <td><pre><code>_handle_camera_movement(delta)</code></pre></td>
      <td>Moves the camera in response to keyboard and mouse input. @param delta: Frame delta in seconds.</td>
    </tr>
    <tr>
      <td><pre><code>_input(event)</code></pre></td>
      <td>Handles hotkeys for renderer toggles and diagnostics. @param event: Input event dispatched by the scene tree.</td>
    </tr>
    <tr>
      <td><pre><code>_load_ply_data()</code></pre></td>
      <td>Loads a PLY file specified on the command line or falls back to synthetic data.</td>
    </tr>
    <tr>
      <td><pre><code>_process(delta)</code></pre></td>
      <td>Updates camera movement and prints periodic performance stats. @param delta: Frame delta in seconds.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Initializes the full pipeline test scene and prints controls.</td>
    </tr>
    <tr>
      <td><pre><code>_setup_camera()</code></pre></td>
      <td>Creates the test camera with a default view of the scene.</td>
    </tr>
    <tr>
      <td><pre><code>_setup_renderer()</code></pre></td>
      <td>Instantiates and configures the GaussianSplatRenderer for the test.</td>
    </tr>
    <tr>
      <td><pre><code>_update_performance_stats(delta)</code></pre></td>
      <td>Prints renderer and FPS stats once per second. @param delta: Frame delta in seconds.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/test_gpu_buffers.gd
```

### Class

```
test_gpu_buffers
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
      <td><pre><code>_ready()</code></pre></td>
      <td>Builds a renderer and uploads synthetic data to validate GPU buffer setup.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/test_gpu_pipeline.gd
```

### Class

```
test_gpu_pipeline
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
      <td><pre><code>_ready()</code></pre></td>
      <td>Executes a smoke test across the GPU pipeline components.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/test_project/fps_camera_controller.gd
```

### Class

```
fps_camera_controller
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
      <td><pre><code>_adjust_aspect_clamp(delta: float)</code></pre></td>
      <td>Adjusts the max conic aspect ratio clamp by the given delta. @param delta: Increment applied to the current clamp.</td>
    </tr>
    <tr>
      <td><pre><code>_find_gaussian_renderer()</code></pre></td>
      <td>Locates the first Gaussian renderer in the scene tree.</td>
    </tr>
    <tr>
      <td><pre><code>_find_renderer_recursive(node: Node)</code></pre></td>
      <td>Recursively searches for a node exposing get_renderer.</td>
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
      <td><pre><code>_input(event: InputEvent)</code></pre></td>
      <td>Handles mouse look, exit, and Jacobian diagnostic hotkeys. @param event: Input event dispatched by the scene tree.</td>
    </tr>
    <tr>
      <td><pre><code>_physics_process(delta: float)</code></pre></td>
      <td>Applies WASD-style movement each physics frame. @param delta: Physics frame delta in seconds.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Captures mouse input for FPS camera control. Controls: WASD=move, E/Q=up/down, Shift=fast, ESC=quit Debug: 1/2/3=Jacobian toggles, ,/.=aspect clamp, 0=reset</td>
    </tr>
    <tr>
      <td><pre><code>_set_aspect_clamp(value: float)</code></pre></td>
      <td>Sets the max conic aspect ratio clamp to an explicit value. @param value: Clamp value to apply.</td>
    </tr>
    <tr>
      <td><pre><code>_toggle_jacobian_bypass_clamp()</code></pre></td>
      <td>Toggles the Jacobian column clamp bypass in the renderer.</td>
    </tr>
    <tr>
      <td><pre><code>_toggle_jacobian_bypass_depth()</code></pre></td>
      <td>Toggles the radius depth-floor bypass in the renderer.</td>
    </tr>
    <tr>
      <td><pre><code>_toggle_jacobian_invert_sign()</code></pre></td>
      <td>Toggles the Jacobian sign inversion flag in the renderer.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/test_project/scenes/diagnostic_distance_artifacts.gd
```

### Class

```
diagnostic_distance_artifacts
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
      <td><pre><code>_input(event: InputEvent)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_process(delta: float)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_report_distance_summary(stats: Dictionary)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_report_final_summary()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_set_camera_distance(distance: float)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/test_project/scenes/lighting_test.gd
```

### Class

```
lighting_test
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
      <td><pre><code>_input(event: InputEvent)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_print_overflow_stats()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_toggle_projection_debug()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_toggle_shadow_opacity_debug()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_toggle_unclustered_lights()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_toggle_white_albedo()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_update_label()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/test_project/scenes/painterly_test.gd
```

### Class

```
painterly_test
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
      <td><pre><code>_read_visible_splats(splat_node: Node)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/test_project/scenes/perf_test_50_instances.gd
```

### Class

```
perf_test_50_instances
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
      <td><pre><code>_get_avg_fps()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_input(event: InputEvent)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_process(delta: float)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/test_project/scenes/perf_test_clean.gd
```

### Class

```
perf_test_clean
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
      <td><pre><code>_get_avg_fps()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_input(event: InputEvent)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_process(delta: float)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/test_project/scenes/phase66d_validation.gd
```

### Class

```
phase66d_validation
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
      <td><pre><code>_calc_avg(samples: Array[float])</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_finish_current_test()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_percentile(samples: Array[float], p: float)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_print_final_results()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_process(delta: float)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_start_instance_test()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_start_legacy_test()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/test_project/scenes/qa/qa_performance_budget.gd
```

### Class

```
qa_performance_budget
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
      <td><pre><code>_calculate_avg(samples: Array[float])</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_on_test_complete()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_on_test_frame(delta: float)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_on_test_start()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/test_project/scenes/qa/qa_scale_validation.gd
```

### Class

```
qa_scale_validation
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
      <td><pre><code>_on_test_complete()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_on_test_start()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/test_project/scenes/qa/qa_sh_rotation.gd
```

### Class

```
qa_sh_rotation
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
      <td><pre><code>_apply_angle(angle_deg: float)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_on_test_complete()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_on_test_frame(_delta: float)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_on_test_start()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_set_visibility(world_path_visible: bool, instance_path_visible: bool)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/test_project/scenes/qa/qa_sort_depth_order.gd
```

### Class

```
qa_sort_depth_order
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
      <td><pre><code>_on_test_complete()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_on_test_frame(_delta: float)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_on_test_start()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_sample_center_color(image: Image, center: Vector2i, radius: int = 1)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/test_project/scenes/qa/qa_sort_multi_instance.gd
```

### Class

```
qa_sort_multi_instance
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
      <td><pre><code>_apply_renderer_overrides(renderer: Object, prev_settings: Dictionary)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_on_test_complete()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_on_test_frame(_delta: float)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_on_test_start()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_restore_renderer_overrides(renderer: Object, prev_settings: Dictionary)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_sample_center_color(image: Image, center: Vector2i, radius: int = 1)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_set_renderer_override_values(renderer: Object, settings: Dictionary)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_snapshot_renderer_override_settings(renderer: Object)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/test_project/scenes/qa/qa_sort_tie_breaker.gd
```

### Class

```
qa_sort_tie_breaker
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
      <td><pre><code>_on_test_complete()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_on_test_frame(_delta: float)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_on_test_start()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/test_project/scenes/qa/qa_static_fast_path.gd
```

### Class

```
qa_static_fast_path
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
      <td><pre><code>_on_test_complete()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_on_test_frame(_delta: float)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_on_test_start()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/test_project/scenes/qa/qa_stream_chunk_loading.gd
```

### Class

```
qa_stream_chunk_loading
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
      <td><pre><code>_on_test_complete()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_on_test_frame(_delta: float)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_on_test_start()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/test_project/scenes/qa/qa_stream_eviction_churn.gd
```

### Class

```
qa_stream_eviction_churn
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
      <td><pre><code>_on_test_complete()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_on_test_frame(_delta: float)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_on_test_start()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/test_project/scenes/qa/qa_stream_multi_asset.gd
```

### Class

```
qa_stream_multi_asset
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
      <td><pre><code>_on_test_complete()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_on_test_frame(_delta: float)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_on_test_start()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/test_project/scenes/qa/qa_stream_visual_smoke.gd
```

### Class

```
qa_stream_visual_smoke
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
      <td><pre><code>_color_distance(a: Color, b: Color)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_compute_corner_mean_color(image: Image, half_extent: int)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_compute_global_luma_variance(image: Image, sample_stride: int = 8)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_compute_patch_stats(image: Image, center: Vector2i, half_extent: int)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_evaluate_readiness(visible_splats: int, center_to_corner_delta: float)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_is_headless_runtime()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_on_test_complete()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_on_test_frame(_delta: float)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_on_test_start()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_read_renderer_stats()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_read_visible_splats()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_resolve_focus_point()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/test_project/scenes/qa/qa_visual_diff.gd
```

### Class

```
qa_visual_diff
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
      <td><pre><code>_on_test_complete()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_on_test_frame(_delta: float)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_on_test_start()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_set_visibility(world_path_visible: bool, instance_path_visible: bool)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/test_project/scenes/static_10_spinning.gd
```

### Class

```
static_10_spinning
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
      <td><pre><code>_process(delta: float)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/test_project/scenes/static_50_no_spin.gd
```

### Class

```
static_50_no_spin
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
      <td><pre><code>_ready()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/test_project/scripts/bake_gsplatworld.gd
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
      <td><pre><code>_find_container(scene_root: Node, container_path: String)</code></pre></td>
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
      <td><pre><code>_run_bake()</code></pre></td>
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
tests/examples/godot/test_project/scripts/qa_test_base.gd
```

### Class

```
GSQATest
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
      <td><pre><code>_compute_luma_buffer(img: Image)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_finish_test()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_get_ssim_kernel()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_on_test_complete()</code></pre></td>
      <td>Override in subclass - called when test duration ends Set _test_result and _test_message before returning</td>
    </tr>
    <tr>
      <td><pre><code>_on_test_frame(_delta: float)</code></pre></td>
      <td>Override in subclass - called each frame during test</td>
    </tr>
    <tr>
      <td><pre><code>_on_test_start()</code></pre></td>
      <td>Override in subclass - called when test phase starts</td>
    </tr>
    <tr>
      <td><pre><code>_prepare_ssim_image(img: Image)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_process(delta: float)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>append_renderer_diagnostics(metric_prefix: String, renderer: Object)</code></pre></td>
      <td>Helper: Append a stable subset of renderer diagnostics into result metrics.</td>
    </tr>
    <tr>
      <td><pre><code>calculate_ssim(img_a: Image, img_b: Image)</code></pre></td>
      <td>Helper: Calculate SSIM between two images (real SSIM with gaussian window)</td>
    </tr>
    <tr>
      <td><pre><code>capture_viewport()</code></pre></td>
      <td>Helper: Capture current viewport as Image</td>
    </tr>
    <tr>
      <td><pre><code>check_fps_in_range(samples: Array[float], min_fps: float, max_fps: float)</code></pre></td>
      <td>Helper: Check if FPS is within expected range</td>
    </tr>
    <tr>
      <td><pre><code>get_custom_monitor_value(monitor_id: String)</code></pre></td>
      <td>Helper: Read a custom Performance monitor value (returns null if unavailable)</td>
    </tr>
    <tr>
      <td><pre><code>get_gs_renderer(node_path: NodePath)</code></pre></td>
      <td>Helper: Get renderer from a GaussianSplatNode3D child</td>
    </tr>
    <tr>
      <td><pre><code>get_result_metrics()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>percentile(samples: Array[float], p: float)</code></pre></td>
      <td>Helper: Calculate percentile from sample array</td>
    </tr>
    <tr>
      <td><pre><code>start_test()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/test_project/scripts/qa_test_runner.gd
```

### Class

```
qa_test_runner
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
      <td><pre><code>_cleanup_current_test()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_find_qa_test(node: Node)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_init()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_on_test_completed(passed: bool, message: String)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_print_summary()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_resolve_output_path()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_run()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_run_next_test()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_write_results_json(suite_end_time: float, passed_count: int, failed_count: int)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/test_project/scripts/ui/performance_overlay.gd
```

### Class

```
GaussianPerformanceOverlay
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
      <td><pre><code>_m(name: String)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_n(val: float)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_pct_color(pct: float, good_above: float = 90.0)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_process(delta: float)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_refresh()</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_set_label(name: String, text: String, color: Color = WHITE)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
    <tr>
      <td><pre><code>_time_color(ms: float)</code></pre></td>
      <td>Undocumented.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/test_scenes/hello_splat_test.gd
```

### Class

```
hello_splat_test
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
      <td><pre><code>_input(event)</code></pre></td>
      <td>Handles input shortcuts for stats output and renderer visibility. @param event: Input event dispatched by the scene tree.</td>
    </tr>
    <tr>
      <td><pre><code>_print_stats()</code></pre></td>
      <td>Prints current renderer statistics to the console.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Instantiates the Hello Splat renderer, camera, and lighting for the demo scene.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/examples/godot/test_toggle_painterly.gd
```

### Class

```
test_toggle_painterly
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
      <td>Defers execution until the SceneTree loop is initialized.</td>
    </tr>
    <tr>
      <td><pre><code>_run()</code></pre></td>
      <td>Sets up a splat node, toggles painterly mode, and prints visibility stats.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/runtime/test_engine_capabilities.gd
```

### Class

```
test_engine_capabilities
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
      <td><pre><code>_check_camera_updates()</code></pre></td>
      <td>Verifies renderer stats advance when the camera moves.</td>
    </tr>
    <tr>
      <td><pre><code>_check_data_assignment()</code></pre></td>
      <td>Checks assigning splat data to a node updates statistics.</td>
    </tr>
    <tr>
      <td><pre><code>_check_deletion_safety()</code></pre></td>
      <td>Confirms that deleting nodes updates residency counts safely.</td>
    </tr>
    <tr>
      <td><pre><code>_check_empty_node()</code></pre></td>
      <td>Ensures an empty GaussianSplatNode3D reports zero visible splats.</td>
    </tr>
    <tr>
      <td><pre><code>_check_large_dataset()</code></pre></td>
      <td>Loads a large dataset and ensures visibility and count are reported.</td>
    </tr>
    <tr>
      <td><pre><code>_check_launch()</code></pre></td>
      <td>Confirms engine singletons and global stats availability.</td>
    </tr>
    <tr>
      <td><pre><code>_check_multiple_nodes()</code></pre></td>
      <td>Ensures multiple splat nodes render and retain distinct colors.</td>
    </tr>
    <tr>
      <td><pre><code>_check_node_creation()</code></pre></td>
      <td>Validates that GaussianSplatNode3D can enter the scene tree.</td>
    </tr>
    <tr>
      <td><pre><code>_check_persistence()</code></pre></td>
      <td>Validates baseline/incremental save-load behavior for splat data.</td>
    </tr>
    <tr>
      <td><pre><code>_check_scene_switching()</code></pre></td>
      <td>Ensures global stats update during scene switching.</td>
    </tr>
    <tr>
      <td><pre><code>_check_single_node_render()</code></pre></td>
      <td>Ensures a single node renders visible splats.</td>
    </tr>
    <tr>
      <td><pre><code>_cleanup_user_file(path: String)</code></pre></td>
      <td>Removes a user:// file if present.</td>
    </tr>
    <tr>
      <td><pre><code>_color_distance(a: Color, b: Color)</code></pre></td>
      <td>Computes Euclidean distance between two colors.</td>
    </tr>
    <tr>
      <td><pre><code>_ensure_runtime_camera()</code></pre></td>
      <td>Ensures a deterministic camera setup for runtime visibility checks.</td>
    </tr>
    <tr>
      <td><pre><code>_global_residency_metric(stats: Dictionary)</code></pre></td>
      <td>Returns the best available residency metric from manager stats.</td>
    </tr>
    <tr>
      <td><pre><code>_init()</code></pre></td>
      <td>Defers the capability checks until the SceneTree is initialized.</td>
    </tr>
    <tr>
      <td><pre><code>_is_headless_runtime()</code></pre></td>
      <td>Returns true when running on a headless display server.</td>
    </tr>
    <tr>
      <td><pre><code>_make_cluster_asset(count: int, center: Vector3, color: Color, radius: float)</code></pre></td>
      <td>Generates a clustered GaussianSplatAsset with the requested properties. @param count: Number of splats. @param center: Cluster center. @param color: Base color for splats. @param radius: Cluster radius for random jitter. @return Configured GaussianSplatAsset instance.</td>
    </tr>
    <tr>
      <td><pre><code>_print_summary()</code></pre></td>
      <td>Prints a summary of all capability checks.</td>
    </tr>
    <tr>
      <td><pre><code>_record(name: String, result: Dictionary)</code></pre></td>
      <td>Records a check result and prints status to stdout. @param name: Human-readable check name. @param result: Result dictionary containing success/details.</td>
    </tr>
    <tr>
      <td><pre><code>_run()</code></pre></td>
      <td>Executes all capability checks and exits with status.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/runtime/test_gpu_streaming_stress.gd
```

### Class

```
test_gpu_streaming_stress
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
      <td><pre><code>_build_dataset(size: int)</code></pre></td>
      <td>Creates random GaussianData for streaming stress tests. @param size: Number of splats to generate. @return GaussianData populated with positions/colors/scales.</td>
    </tr>
    <tr>
      <td><pre><code>_exercise_dataset(size: int)</code></pre></td>
      <td>Runs a streaming and sorting pass for the given dataset size. @param size: Number of splats to generate and upload.</td>
    </tr>
    <tr>
      <td><pre><code>_init()</code></pre></td>
      <td>Defers execution until the SceneTree loop is initialized.</td>
    </tr>
    <tr>
      <td><pre><code>_is_headless_runtime()</code></pre></td>
      <td>Returns true when running on a headless display server.</td>
    </tr>
    <tr>
      <td><pre><code>_print_summary()</code></pre></td>
      <td>Prints an explicit failure list for CI log triage.</td>
    </tr>
    <tr>
      <td><pre><code>_read_stat_float(stats: Dictionary, keys: Array[String], default_value: float = 0.0)</code></pre></td>
      <td>Returns the first float-valued entry for the provided stat keys.</td>
    </tr>
    <tr>
      <td><pre><code>_read_stat_int(stats: Dictionary, keys: Array[String], default_value: int = 0)</code></pre></td>
      <td>Returns the first integer-valued entry for the provided stat keys.</td>
    </tr>
    <tr>
      <td><pre><code>_read_stat_int_max(stats: Dictionary, keys: Array[String], default_value: int = 0)</code></pre></td>
      <td>Returns the maximum integer-valued metric from the provided keys.</td>
    </tr>
    <tr>
      <td><pre><code>_record_failure(reason: String, context: Dictionary = {})</code></pre></td>
      <td>Records a failure with context and marks the script as failed.</td>
    </tr>
    <tr>
      <td><pre><code>_run()</code></pre></td>
      <td>Initializes the renderer and exercises streaming datasets.</td>
    </tr>
    <tr>
      <td><pre><code>_validate_residency(size: int, stats: Dictionary)</code></pre></td>
      <td>Validates upload residency and visibility ratios for the dataset. @param size: Dataset size. @param stats: Renderer stats dictionary.</td>
    </tr>
    <tr>
      <td><pre><code>_validate_sort_metrics(method_name: String, size: int, stats: Dictionary, history: Array)</code></pre></td>
      <td>Validates GPU sort metrics and history for the given dataset size. @param method_name: Sorting method label. @param size: Dataset size. @param stats: Renderer stats dictionary. @param history: Sort metrics history array.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/runtime/test_interactive_state.gd
```

### Class

```
test_interactive_state
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
      <td>Defers test execution until the SceneTree loop is initialized.</td>
    </tr>
    <tr>
      <td><pre><code>_is_headless_runtime()</code></pre></td>
      <td>Returns true when running on a headless display server.</td>
    </tr>
    <tr>
      <td><pre><code>_print_summary()</code></pre></td>
      <td>Prints final pass/fail summary.</td>
    </tr>
    <tr>
      <td><pre><code>_record_check(condition: bool, label: String, context: Dictionary = {})</code></pre></td>
      <td>Records a validation outcome with optional context details.</td>
    </tr>
    <tr>
      <td><pre><code>_record_failure(label: String, context: Dictionary = {})</code></pre></td>
      <td>Records a failure and prints a machine-readable marker for the Python runner.</td>
    </tr>
    <tr>
      <td><pre><code>_run()</code></pre></td>
      <td>Creates a renderer and runs state transition, visual, and performance checks.</td>
    </tr>
    <tr>
      <td><pre><code>test_state_performance(renderer)</code></pre></td>
      <td>Benchmarks rapid state transitions to validate performance targets. @param renderer: GaussianSplatRenderer instance under test.</td>
    </tr>
    <tr>
      <td><pre><code>test_state_transitions(renderer)</code></pre></td>
      <td>Validates allowed state transitions for the interactive state machine. @param renderer: GaussianSplatRenderer instance under test.</td>
    </tr>
    <tr>
      <td><pre><code>test_visual_effects(renderer)</code></pre></td>
      <td>Exercises highlight/outline effects and reset behavior. @param renderer: GaussianSplatRenderer instance under test.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/test_gaussian_splat_visibility.gd
```

### Class

```
test_gaussian_splat_visibility
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
      <td><pre><code>_ready()</code></pre></td>
      <td>Defers the regression runner to ensure the scene tree is initialized before asserting visibility; no side effects beyond scheduling the test.</td>
    </tr>
    <tr>
      <td><pre><code>_run()</code></pre></td>
      <td>Verifies a GaussianSplatNode3D reports visible splats once populated, exiting with code 0 on success and 1 when the regression reproduces.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/test_octree_correctness.gd
```

### Class

```
test_octree_correctness
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
      <td><pre><code>_populate_gaussians(data: GaussianData, positions: PackedVector3Array, scales: PackedVector3Array, opacities: PackedFloat32Array)</code></pre></td>
      <td>Populates the GaussianData container with the supplied arrays so individual tests can exercise octree queries without redundant setup; mutates the provided data.</td>
    </tr>
    <tr>
      <td><pre><code>_ready()</code></pre></td>
      <td>Entry point when run from Godot's headless test runner; exits with code 0 on success and 1 on failure for CI integration.</td>
    </tr>
    <tr>
      <td><pre><code>assert_contains(array: Array, value, test_name: String)</code></pre></td>
      <td>Confirms the queried array contains the provided value and records the result, emitting diagnostic text for visibility.</td>
    </tr>
    <tr>
      <td><pre><code>assert_equal(actual, expected, test_name: String)</code></pre></td>
      <td>Verifies that the expected value matches the actual one while tracking pass/fail counters; only prints to stdout as a side effect.</td>
    </tr>
    <tr>
      <td><pre><code>run_tests()</code></pre></td>
      <td>Executes all octree regression scenarios and returns true when every assertion succeeds; prints a summary but performs no additional side effects.</td>
    </tr>
    <tr>
      <td><pre><code>test_boundary_cases()</code></pre></td>
      <td>Exercises queries along octree boundaries to ensure splats on volume edges stay discoverable without duplicating references.</td>
    </tr>
    <tr>
      <td><pre><code>test_empty_tree()</code></pre></td>
      <td>Validates that querying an empty octree yields no results, ensuring default state safety without mutating shared data.</td>
    </tr>
    <tr>
      <td><pre><code>test_large_scale_gaussians()</code></pre></td>
      <td>Verifies that extremely large splats expand query coverage correctly while small neighbors continue operating normally.</td>
    </tr>
    <tr>
      <td><pre><code>test_overlapping_gaussians()</code></pre></td>
      <td>Confirms overlapping Gaussians remain discoverable when occupying shared space, validating node subdivision and accumulation behavior.</td>
    </tr>
    <tr>
      <td><pre><code>test_query_consistency()</code></pre></td>
      <td>Checks repeated queries of a dense grid return consistent counts and omit duplicates, guarding against nondeterminism.</td>
    </tr>
    <tr>
      <td><pre><code>test_single_gaussian()</code></pre></td>
      <td>Ensures a single Gaussian is correctly indexed and retrievable at boundaries; the scenario touches both hit and miss queries for coverage.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/test_octree_fix.gd
```

### Class

```
test_octree_fix
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
      <td><pre><code>_ready()</code></pre></td>
      <td>Validates the octree subdivision fix by mixing extreme Gaussian scales and measuring build/query times; exits with non-zero when any threshold fails.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/test_octree_overflow.gd
```

### Class

```
test_octree_overflow
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
      <td>Stress test ensuring octrees with more than 255 nodes remain addressable after the uint32_t fix; exits with default code 0 when successful.</td>
    </tr>
    <tr>
      <td><pre><code>_test_concept()</code></pre></td>
      <td>Provides a textual walkthrough of the overflow fix when the native class is unavailable; purely informational with console output as the only side effect.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/test_ply_loader.gd
```

### Class

```
test_ply_loader
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
      <td><pre><code>_ready()</code></pre></td>
      <td>Defers the ResourceLoader regression check until the scene is ready.</td>
    </tr>
    <tr>
      <td><pre><code>_run()</code></pre></td>
      <td>Validates ResourceLoader returns a GaussianSplatAsset with expected metadata.</td>
    </tr>
  </tbody>
</table>

## Script

```
tests/test_streaming.gd
```

### Class

```
test_streaming
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
      <td>Entry point when run as standalone script</td>
    </tr>
    <tr>
      <td><pre><code>test_streaming_system()</code></pre></td>
      <td>Validates the GPU streaming system by simulating a frame of updates over a large splat dataset; only prints metrics as a side effect.</td>
    </tr>
  </tbody>
</table>

Generated by:

```
scripts/extract_gdscript_docs.py
```
