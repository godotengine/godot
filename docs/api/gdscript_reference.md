# GDScript API Reference

Last generated: 2026-03-20

Scope: `public`

Scripts scanned: `5`

Undocumented members are omitted by default. Use `--include-undocumented` to include them.

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
      <td><pre><code>sort_keys_gpu(keys: PackedInt32Array, values: PackedInt32Array = PackedInt32Array())</code></pre></td>
      <td>Sorts key/value pairs using the GPU radix sort pipeline (CPU fallback for now). @param keys: Keys to sort. @param values: Optional values to keep in sync with keys. @return Sorted keys array.</td>
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
      <td><pre><code>_try_resolve_camera()</code></pre></td>
      <td>Resolves the camera from the configured NodePath when missing.</td>
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

Generated by:

```
scripts/extract_gdscript_docs.py
```
