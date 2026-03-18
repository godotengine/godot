extends Node
## Project-wide bootstrap that ensures the Gaussian splatting module is ready
## and exposes convenience helpers for querying global statistics.

var gaussian_manager: Object
var is_ready: bool = false
var _log_last_ms: Dictionary = {}

## Resolves the GaussianSplatManager singleton and logs runtime configuration.
func _ready() -> void:
    if Engine.has_singleton("GaussianSplatManager"):
        gaussian_manager = Engine.get_singleton("GaussianSplatManager")
    else:
        gaussian_manager = null

    if gaussian_manager:
        is_ready = true
        _log_runtime_configuration()
    else:
        push_error("GaussianSplatManager singleton is unavailable. Ensure the Gaussian splatting module is enabled in this build.")

## Prints adapter and sorting configuration information for diagnostics.
func _log_runtime_configuration() -> void:
    var gpu_name := RenderingServer.get_video_adapter_name()
    var supports_vulkan := RenderingServer.get_rendering_device() != null
    _gs_log_info("[GaussianBootstrap] GPU: %s" % gpu_name, "bootstrap_gpu_name")
    _gs_log_info("[GaussianBootstrap] Vulkan available: %s" % supports_vulkan, "bootstrap_vulkan")

    if gaussian_manager.has_method("get_sorting_config"):
        var config: Dictionary = gaussian_manager.get_sorting_config()
        _gs_log_info("[GaussianBootstrap] Sorting configuration:", "bootstrap_sorting_header")
        for key in config.keys():
            _gs_log_info("  - %s: %s" % [key, config[key]], "bootstrap_sorting_entry_%s" % key)

    if gaussian_manager.has_method("is_gpu_sorting_enabled"):
        _gs_log_info("[GaussianBootstrap] GPU sorting enabled: %s" % gaussian_manager.is_gpu_sorting_enabled(), "bootstrap_gpu_sorting")
    if gaussian_manager.has_method("is_shared_submission_device_enabled"):
        _gs_log_info("[GaussianBootstrap] Shared submission device enabled: %s" % gaussian_manager.is_shared_submission_device_enabled(), "bootstrap_shared_submission")

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

## Returns global renderer statistics when available.
func get_global_stats() -> Dictionary:
    if gaussian_manager and gaussian_manager.has_method("get_global_stats"):
        return gaussian_manager.get_global_stats()
    return {}

## Acquires a submission lock from the manager when supported.
## @return Lock object or null if unavailable.
func ensure_submission_lock() -> Object:
    if gaussian_manager and gaussian_manager.has_method("acquire_submission_lock"):
        return gaussian_manager.acquire_submission_lock()
    return null
