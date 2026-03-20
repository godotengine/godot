# GaussianSplatManager API Reference

## Purpose
`GaussianSplatManager` is the engine-wide singleton that coordinates GPU buffer registration, performance monitoring, sorting configuration, and RenderingDevice lifecycle management for all Gaussian Splatting operations. Access it via `Engine.get_singleton("GaussianSplatManager")` (`modules/gaussian_splatting/core/gaussian_splat_manager.h:71`).

## Usage
<table>
  <thead>
    <tr>
      <th>Task</th>
      <th>Primary API</th>
      <th>Implementation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Query global rendering statistics.</td>
      <td><code>get_global_stats()</code></td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:887</code></td>
    </tr>
    <tr>
      <td>Toggle GPU-based sorting on or off.</td>
      <td><code>set_gpu_sorting_enabled(enabled)</code></td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:903</code></td>
    </tr>
    <tr>
      <td>Enable multi-threaded shared submission device.</td>
      <td><code>set_shared_submission_device_enabled(enabled)</code></td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:907</code></td>
    </tr>
    <tr>
      <td>Inspect sorting algorithm thresholds.</td>
      <td><code>get_sorting_config()</code></td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:911</code></td>
    </tr>
  </tbody>
</table>

## API
### Enums
This class does not expose any enums through `BIND_ENUM_CONSTANT`.

### Properties
<table>
  <thead>
    <tr>
      <th>Inspector path</th>
      <th>Type</th>
      <th>Accessors</th>
      <th>Notes</th>
      <th>Source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>gpu_sorting_enabled</code></td>
      <td><code>bool</code></td>
      <td><code>set_gpu_sorting_enabled</code>, <code>is_gpu_sorting_enabled</code></td>
      <td>When disabled, falls back to CPU sorting. Initialized from <code>rendering/gaussian_splatting/gpu_sorting_enabled</code> project setting.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:641</code></td>
    </tr>
    <tr>
      <td><code>shared_submission_device_enabled</code></td>
      <td><code>bool</code></td>
      <td><code>set_shared_submission_device_enabled</code>, <code>is_shared_submission_device_enabled</code></td>
      <td>When enabled, uses a dedicated RenderingDevice for multi-threaded GPU submission. Initialized from <code>rendering/gaussian_splatting/shared_submission_device_enabled</code> project setting.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:642</code></td>
    </tr>
  </tbody>
</table>

### Methods
<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>Behavior</th>
      <th>Source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>get_global_stats() -> Dictionary</code></td>
      <td>Returns a Dictionary with keys: <code>total_gaussians</code> (int), <code>total_memory_mb</code> (float), <code>buffer_count</code> (int), <code>gpu_sorting_enabled</code> (bool), <code>shared_submission_device_enabled</code> (bool), <code>sorting</code> (Dictionary from <code>get_sorting_config()</code>), <code>reported_gaussians</code> (int), and <code>reported_memory_mb</code> (float). Thread-safe via resource maps mutex.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:887</code></td>
    </tr>
    <tr>
      <td><code>set_gpu_sorting_enabled(enabled: bool)</code></td>
      <td>Enables or disables GPU-based sorting for all renderers. When false, renderers fall back to CPU sorting.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:903</code></td>
    </tr>
    <tr>
      <td><code>is_gpu_sorting_enabled() -> bool</code></td>
      <td>Returns the current GPU sorting state.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_manager.h:376</code></td>
    </tr>
    <tr>
      <td><code>set_shared_submission_device_enabled(enabled: bool)</code></td>
      <td>Enables or disables the shared RenderingDevice used for multi-threaded GPU submission. When enabled, GPU work from multiple threads is serialized through a dedicated submission device protected by a mutex.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:907</code></td>
    </tr>
    <tr>
      <td><code>is_shared_submission_device_enabled() -> bool</code></td>
      <td>Returns true if the shared submission device is enabled.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_manager.h:385</code></td>
    </tr>
    <tr>
      <td><code>get_sorting_config() -> Dictionary</code></td>
      <td>Returns sorting algorithm thresholds as a Dictionary with keys: <code>bitonic_max_elements</code>, <code>radix_max_elements</code>, <code>onesweep_max_elements</code>, <code>hybrid_trigger_elements</code>, <code>hybrid_batch_size</code>, <code>history_size</code>, <code>log_interval_frames</code>, <code>target_sort_time_ms</code>, <code>log_metrics</code>, <code>force_algorithm</code> (0=auto, 1=radix, 2=bitonic, 3=onesweep), and <code>force_cpu_sort</code>. Values are read from project settings at startup.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:911</code></td>
    </tr>
  </tbody>
</table>

### Signals
This class does not expose any signals through `ADD_SIGNAL`.

## Examples
```gdscript
extends Node

func _ready() -> void:
    var manager = Engine.get_singleton("GaussianSplatManager")
    var stats: Dictionary = manager.get_global_stats()
    print("Active Gaussians: ", stats["total_gaussians"])
    print("GPU Memory: ", snprintf("%.1f", stats["total_memory_mb"]), " MB")
    print("Buffer Count: ", stats["buffer_count"])
```

```gdscript
extends Node

func _ready() -> void:
    var manager = Engine.get_singleton("GaussianSplatManager")

    # Disable GPU sorting for debugging (falls back to CPU sort)
    manager.set_gpu_sorting_enabled(false)

    # Check current sorting algorithm thresholds
    var sort_cfg: Dictionary = manager.get_sorting_config()
    print("Radix sort max: ", sort_cfg["radix_max_elements"])
    print("Target sort time: ", sort_cfg["target_sort_time_ms"], " ms")
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
      <td><code>get_global_stats()</code> shows zero <code>total_gaussians</code>.</td>
      <td>Confirm that at least one <code>GaussianSplatNode3D</code> has loaded an asset and is inside the scene tree. The manager tracks buffers registered by active renderer instances.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:652</code></td>
    </tr>
    <tr>
      <td>RenderDoc crashes when Gaussian Splatting is active.</td>
      <td>The manager auto-detects RenderDoc and skips creating local RenderingDevice instances. If detection fails, set the project setting <code>rendering/gaussian_splatting/renderdoc_compatibility</code> to <code>true</code>.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:106</code></td>
    </tr>
    <tr>
      <td>Sorting performance is unexpectedly slow.</td>
      <td>Call <code>get_sorting_config()</code> to verify thresholds match your splat counts. The <code>force_algorithm</code> key (0=auto) can be overridden via <code>rendering/gaussian_splatting/sorting/force_algorithm</code> project setting to test specific algorithms.</td>
      <td><code>modules/gaussian_splatting/core/gaussian_splat_manager.cpp:283</code></td>
    </tr>
  </tbody>
</table>
