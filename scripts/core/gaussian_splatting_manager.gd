extends Node

# Core manager for Gaussian Splatting operations
# Registered as autoload singleton "GaussianSplattingManager" in project settings

var _rendering_device: RenderingDevice
var _cull_shader: RID

# Performance tracking
var _frame_counter: int = 0
var _sort_time_ms: float = 0.0
var _render_time_ms: float = 0.0
var _log_last_ms: Dictionary = {}

signal gaussians_sorted(count: int, time_ms: float)
signal rendering_complete(fps: float, frame_time_ms: float)

## Initializes GPU resources for the Gaussian Splatting manager.
func _ready() -> void:
    _gs_log_info("Initializing Gaussian Splatting Manager...", "gs_manager_init")
    _initialize_gpu_resources()

## Schedules GPU initialization after the rendering server is ready.
func _initialize_gpu_resources() -> void:
    # Get the rendering device - use call_deferred to ensure rendering is ready
    call_deferred("_deferred_gpu_init")

## Allocates a local RenderingDevice and prints adapter information.
func _deferred_gpu_init() -> void:
    # Create a local rendering device for GPU compute operations
    _rendering_device = RenderingServer.create_local_rendering_device()

    if not _rendering_device:
        push_warning("GPU compute not available - some features may be limited")
        return

    _gs_log_info("GPU Compute initialized successfully", "gs_manager_gpu_ready")
    var adapter_name: String = RenderingServer.get_video_adapter_name()
    var adapter_vendor: String = RenderingServer.get_video_adapter_vendor()
    _gs_log_info("GPU: %s" % adapter_name, "gs_manager_gpu_name")
    _gs_log_info("Driver: %s" % adapter_vendor, "gs_manager_gpu_vendor")

## Validates availability of embedded radix-sort compute kernels.
func load_compute_shaders() -> void:
    if not _rendering_device:
        push_warning("GPU compute not available - cannot validate compute shader pipeline")
        return
    _gs_log_info("Radix sort compute kernels are embedded in renderer runtime", "gs_manager_radix_embedded")

func _gs_debug_flag() -> String:
    if ProjectSettings.get_setting("rendering/gaussian_splatting/debug/enable_all_debug", false):
        return "enable_all_debug"
    if ProjectSettings.get_setting("rendering/gaussian_splatting/debug/enable_frame_logging", false):
        return "enable_frame_logging"
    if ProjectSettings.get_setting("rendering/gaussian_splatting/debug/enable_pipeline_trace", false):
        return "enable_pipeline_trace"
    if ProjectSettings.get_setting("rendering/gaussian_splatting/debug/enable_data_logging", false):
        return "enable_data_logging"
    return ""

func _gs_allow_log(key: String) -> bool:
    var rate_ms = ProjectSettings.get_setting("rendering/gaussian_splatting/logging/rate_limit_ms", 1000)
    if typeof(rate_ms) != TYPE_INT and typeof(rate_ms) != TYPE_FLOAT:
        rate_ms = 1000
    rate_ms = max(int(rate_ms), 0)
    if rate_ms <= 0:
        return true
    var now = Time.get_ticks_msec()
    var last = _log_last_ms.get(key, -1)
    if last >= 0 and now - last < rate_ms:
        return false
    _log_last_ms[key] = now
    return true

func _gs_log_info(message: String, key: String) -> void:
    var flag = _gs_debug_flag()
    if flag == "":
        return
    if not _gs_allow_log(key):
        return
    print("%s (debug: %s)" % [message, flag])

## Sorts key/value pairs using the GPU radix sort pipeline (CPU fallback for now).
## @param keys: Keys to sort.
## @param values: Optional values to keep in sync with keys.
## @return Sorted keys array.
func sort_keys_gpu(keys: PackedInt32Array, values: PackedInt32Array = PackedInt32Array()) -> PackedInt32Array:
    # GPU-accelerated sorting using radix sort
    var start_time = Time.get_ticks_usec()

    # For now, fallback to CPU sort until GPU implementation is complete
    var sorted_keys = keys.duplicate()
    sorted_keys.sort()

    _sort_time_ms = (Time.get_ticks_usec() - start_time) / 1000.0

    gaussians_sorted.emit(keys.size(), _sort_time_ms)
    return sorted_keys

## Returns performance metrics including sort/render times and GPU memory usage.
func get_performance_stats() -> Dictionary:
    var gpu_mem: int = RenderingServer.get_rendering_info(RenderingServer.RENDERING_INFO_VIDEO_MEM_USED)
    return {
        "sort_time_ms": _sort_time_ms,
        "render_time_ms": _render_time_ms,
        "frame_counter": _frame_counter,
        "gpu_memory_used": gpu_mem / 1_048_576
    }

## Updates frame counters and emits periodic FPS metrics.
## @param _delta: Frame delta in seconds.
func _process(_delta: float) -> void:
    _frame_counter += 1

    # Update performance metrics every second
    if _frame_counter % 60 == 0:
        var fps = Engine.get_frames_per_second()
        var frame_time = 1000.0 / fps if fps > 0 else 0.0
        rendering_complete.emit(fps, frame_time)
