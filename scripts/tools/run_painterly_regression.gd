extends SceneTree

const UTIL := preload("res://modules/gaussian_splatting/tests/painterly_scenes/painterly_scene_util.gd")
const DEFAULT_ARTIFACT_DIR := "user://painterly_audit/regression"
const DEFAULT_SUMMARY_FILE := "summary.json"
const METRIC_EPSILON := 0.0001

var _success := true
var _scenes_executed: int = 0
var _rendering_device_available: bool = false
var _failures: Array[String] = []
var _warnings: Array[String] = []
var _scene_reports: Array = []

var _artifact_dir: String = DEFAULT_ARTIFACT_DIR
var _summary_path: String = ""
var _summary_path_explicit: bool = false
var _image_size: Vector2i = Vector2i(128, 128)
var _save_artifacts: bool = true
var _require_rendering_device: bool = false
var _enforce_performance_budget: bool = false
var _global_performance_budget_ms: float = 0.0
var _requested_help: bool = false

## Entry point for the regression harness; runs checks and exits with status.
func _init() -> void:
    if not _parse_args():
        quit(0 if _requested_help else 1)
        return

    _run_checks()
    _write_summary()
    if _success:
        print("PAINTERLY_TEST_PASSED")
        quit(0)
        return
    print("PAINTERLY_TEST_FAILED")
    push_error("Painterly regression failed (%d failure(s))" % _failures.size())
    quit(1)

## Executes shader compilation and image-based sanity checks for painterly scenes.
func _run_checks() -> void:
    var scene_paths: Array = UTIL.list_scene_definition_paths()
    if _summary_path.is_empty():
        _summary_path = "%s/%s" % [_artifact_dir, DEFAULT_SUMMARY_FILE]

    var rd: RenderingDevice = UTIL.ensure_rendering_device()
    _rendering_device_available = rd != null
    if rd == null:
        _record_warning("", "RenderingDevice unavailable - shader compilation checks skipped")
        if _require_rendering_device:
            _record_failure("", "RenderingDevice required for this run but unavailable")

    if _save_artifacts:
        _ensure_directory(_artifact_dir)
    _ensure_directory(_summary_path.get_base_dir())

    for path in scene_paths:
        var definition: Dictionary = UTIL.load_definition(path)
        if definition.is_empty():
            _record_failure(path, "Unable to load painterly definition")
            continue
        _scenes_executed += 1

        var scene: Dictionary = UTIL.build_scene(definition, path)
        var scene_name: String = str(scene.get("name", path.get_file().get_basename()))
        var audit: Dictionary = scene.get("audit", {})
        var splats: Array = UTIL.generate_gaussians(scene)
        var scene_start_usec: int = Time.get_ticks_usec()

        var compile_results: Dictionary = {}
        var compile_failures: Array[String] = []
        var compile_start_usec: int = Time.get_ticks_usec()
        if rd != null:
            compile_results = UTIL.compile_permutations(scene, rd)
            for perm_name in compile_results:
                if not bool(compile_results[perm_name]):
                    compile_failures.append(str(perm_name))
        var compile_ms: float = float(Time.get_ticks_usec() - compile_start_usec) / 1000.0

        if not compile_failures.is_empty():
            _record_failure(scene_name, "Shader permutation compilation failed: %s" % _join_string_array(compile_failures))

        var render0_start_usec: int = Time.get_ticks_usec()
        var frame0: Image = UTIL.render_headless(scene, splats, 0, _image_size)
        var render0_ms: float = float(Time.get_ticks_usec() - render0_start_usec) / 1000.0
        var stats0: Dictionary = UTIL.compute_image_stats(frame0)

        _validate_metric_bounds(scene_name, "coverage", float(stats0.get("coverage", 0.0)), float(audit.get("coverage_min", 0.0)), float(audit.get("coverage_max", -1.0)))
        _validate_metric_bounds(scene_name, "luminance", float(stats0.get("luminance", 0.0)), float(audit.get("luminance_min", 0.0)), float(audit.get("luminance_max", -1.0)))

        var repeat_frame: Image = UTIL.render_headless(scene, splats, 0, _image_size)
        var stability_delta: float = UTIL.compute_image_difference(frame0, repeat_frame)
        var stability_max: float = max(0.0, float(audit.get("stability_max", 0.0)))
        if stability_max > 0.0 and stability_delta > stability_max + METRIC_EPSILON:
            _record_failure(scene_name, "Frame stability drift above threshold (actual=%.5f, max=%.5f)" % [stability_delta, stability_max])

        var camera_path: Array = scene.get("camera_path", [])
        var comparison_index: int = 0
        if camera_path.size() > 1:
            comparison_index = clamp(int(audit.get("comparison_camera_index", max(1, camera_path.size() / 2))), 0, camera_path.size() - 1)
        var delta: float = -1.0
        var compare_render_ms: float = 0.0
        var compare_frame: Image = null
        if camera_path.size() > 1:
            var compare_start_usec: int = Time.get_ticks_usec()
            compare_frame = UTIL.render_headless(scene, splats, comparison_index, _image_size)
            compare_render_ms = float(Time.get_ticks_usec() - compare_start_usec) / 1000.0
            delta = UTIL.compute_image_difference(frame0, compare_frame)
            var delta_min: float = max(0.0, float(audit.get("delta_min", 0.0)))
            if delta_min > 0.0 and delta < delta_min - METRIC_EPSILON:
                _record_failure(scene_name, "Animated camera delta below threshold (actual=%.5f, min=%.5f)" % [delta, delta_min])

        var perf_budget_ms: float = max(_global_performance_budget_ms, float(audit.get("performance_budget_ms", 0.0)))
        if perf_budget_ms > 0.0 and render0_ms > perf_budget_ms + METRIC_EPSILON:
            var perf_message := "Render time above budget (actual=%.3fms, budget=%.3fms)" % [render0_ms, perf_budget_ms]
            if _enforce_performance_budget:
                _record_failure(scene_name, perf_message)
            else:
                _record_warning(scene_name, perf_message)

        var artifact_files: Array = []
        if _save_artifacts:
            var pre_rendered: Dictionary = {0: frame0}
            if compare_frame != null:
                pre_rendered[comparison_index] = compare_frame
            artifact_files = _save_scene_artifacts(scene_name, scene, splats, pre_rendered)

        var total_scene_ms: float = float(Time.get_ticks_usec() - scene_start_usec) / 1000.0
        _scene_reports.append({
            "name": scene_name,
            "definition_path": path,
            "density": scene.get("density", "dense"),
            "splat_count": splats.size(),
            "metrics": {
                "coverage": float(stats0.get("coverage", 0.0)),
                "luminance": float(stats0.get("luminance", 0.0)),
                "luminance_stddev": float(stats0.get("luminance_stddev", 0.0)),
                "alpha_mean": float(stats0.get("alpha_mean", 0.0)),
                "camera_delta": delta,
                "stability_delta": stability_delta
            },
            "timings_ms": {
                "compile": compile_ms,
                "render_camera0": render0_ms,
                "render_comparison": compare_render_ms,
                "total_scene": total_scene_ms
            },
            "thresholds": audit,
            "compile": {
                "checked": rd != null,
                "total": compile_results.size(),
                "failed": compile_failures
            },
            "artifacts": artifact_files
        })

        print("Painterly regression -> %s | coverage=%.3f luminance=%.3f delta=%.4f render=%.2fms" % [
            scene_name,
            float(stats0.get("coverage", 0.0)),
            float(stats0.get("luminance", 0.0)),
            delta if delta >= 0.0 else 0.0,
            render0_ms
        ])

    if _scenes_executed == 0:
        _record_failure("", "Painterly regression loaded no scenes")

func _parse_args() -> bool:
    var args: PackedStringArray = OS.get_cmdline_user_args()
    for arg in args:
        if arg == "--":
            continue
        elif arg == "--help" or arg == "-h":
            _requested_help = true
            _print_usage()
            return false
        elif arg == "--no-artifacts":
            _save_artifacts = false
        elif arg == "--require-rendering-device":
            _require_rendering_device = true
        elif arg == "--enforce-performance-budget":
            _enforce_performance_budget = true
        elif arg.begins_with("--artifact-dir="):
            _artifact_dir = arg.substr("--artifact-dir=".length())
        elif arg.begins_with("--summary-path="):
            _summary_path_explicit = true
            _summary_path = arg.substr("--summary-path=".length())
        elif arg.begins_with("--image-size="):
            var size_token: String = arg.substr("--image-size=".length())
            if not _parse_image_size(size_token):
                push_error("Invalid --image-size value: %s (expected WIDTHxHEIGHT)" % size_token)
                return false
        elif arg.begins_with("--perf-budget-ms="):
            var budget_token: String = arg.substr("--perf-budget-ms=".length())
            if not budget_token.is_valid_float():
                push_error("Invalid --perf-budget-ms value: %s" % budget_token)
                return false
            _global_performance_budget_ms = max(0.0, float(budget_token))
        else:
            push_error("Unknown painterly regression argument: %s" % arg)
            return false

    if not _summary_path_explicit:
        _summary_path = "%s/%s" % [_artifact_dir, DEFAULT_SUMMARY_FILE]
    return true

func _parse_image_size(token: String) -> bool:
    var parts: PackedStringArray = token.to_lower().split("x")
    if parts.size() != 2:
        return false
    if not parts[0].is_valid_int() or not parts[1].is_valid_int():
        return false
    var width: int = int(parts[0])
    var height: int = int(parts[1])
    if width < 16 or height < 16:
        return false
    _image_size = Vector2i(width, height)
    return true

func _print_usage() -> void:
    print("Painterly regression harness usage:")
    print("  godot --headless --script scripts/tools/run_painterly_regression.gd -- [options]")
    print("Options:")
    print("  --artifact-dir=<path>            Artifact output directory (default: %s)" % DEFAULT_ARTIFACT_DIR)
    print("  --summary-path=<path>            Summary JSON output path")
    print("  --image-size=<W>x<H>             Render resolution (default: 128x128)")
    print("  --no-artifacts                   Skip PNG artifact writing")
    print("  --require-rendering-device       Fail if RenderingDevice is unavailable")
    print("  --perf-budget-ms=<ms>            Global render budget hint (per scene frame0)")
    print("  --enforce-performance-budget     Treat budget overruns as failures")

func _validate_metric_bounds(scene_name: String, metric_name: String, value: float, minimum: float, maximum: float) -> void:
    if value < minimum - METRIC_EPSILON:
        _record_failure(scene_name, "%s below minimum (actual=%.5f, min=%.5f)" % [metric_name, value, minimum])
    if maximum >= 0.0 and value > maximum + METRIC_EPSILON:
        _record_failure(scene_name, "%s above maximum (actual=%.5f, max=%.5f)" % [metric_name, value, maximum])

func _save_scene_artifacts(scene_name: String, scene: Dictionary, splats: Array, pre_rendered: Dictionary) -> Array:
    var scene_slug := _slugify(scene_name)
    var scene_dir := "%s/%s" % [_artifact_dir, scene_slug]
    _ensure_directory(scene_dir)

    var audit: Dictionary = scene.get("audit", {})
    var artifact_indices = audit.get("artifact_camera_indices", [0])
    var files: Array = []
    if typeof(artifact_indices) != TYPE_ARRAY:
        artifact_indices = [0]

    for index_variant in artifact_indices:
        var camera_index: int = int(index_variant)
        var image: Image = pre_rendered.get(camera_index)
        if image == null:
            image = UTIL.render_headless(scene, splats, camera_index, _image_size)
        var filename := "%s_camera_%02d.png" % [scene_slug, camera_index]
        var filepath := "%s/%s" % [scene_dir, filename]
        var err: Error = image.save_png(filepath)
        if err != OK:
            _record_warning(scene_name, "Failed to save artifact %s (error=%d)" % [filepath, err])
            continue
        files.append(filepath)
    return files

func _write_summary() -> void:
    if _summary_path.is_empty():
        return

    _ensure_directory(_summary_path.get_base_dir())

    var summary := {
        "success": _success,
        "timestamp_unix": Time.get_unix_time_from_system(),
        "rendering_device_available": _rendering_device_available,
        "scenes_executed": _scenes_executed,
        "image_size": [_image_size.x, _image_size.y],
        "artifact_dir": _artifact_dir,
        "save_artifacts": _save_artifacts,
        "warnings": _warnings,
        "failures": _failures,
        "scenes": _scene_reports
    }

    var file: FileAccess = FileAccess.open(_summary_path, FileAccess.WRITE)
    if file == null:
        push_warning("Failed to write painterly summary: %s" % _summary_path)
        return
    file.store_string(JSON.stringify(summary, "  "))
    file.close()
    print("Painterly summary -> %s" % _summary_path)

func _ensure_directory(path: String) -> void:
    if path.is_empty():
        return
    var absolute_path: String = ProjectSettings.globalize_path(path)
    var err: Error = DirAccess.make_dir_recursive_absolute(absolute_path)
    if err != OK and err != ERR_ALREADY_EXISTS:
        _record_warning("", "Failed to create directory: %s" % path)

func _slugify(value: String) -> String:
    var slug := value.strip_edges()
    if slug.is_empty():
        slug = "scene"
    slug = slug.replace(" ", "_")
    slug = slug.replace("/", "_")
    slug = slug.replace("\\", "_")
    return slug

func _record_failure(scope: String, message: String) -> void:
    var line := message if scope.is_empty() else "[%s] %s" % [scope, message]
    _failures.append(line)
    _success = false
    push_error(line)

func _record_warning(scope: String, message: String) -> void:
    var line := message if scope.is_empty() else "[%s] %s" % [scope, message]
    _warnings.append(line)
    push_warning(line)

func _join_string_array(items: Array) -> String:
    if items.is_empty():
        return ""
    var out: String = ""
    for i in range(items.size()):
        if i > 0:
            out += ", "
        out += str(items[i])
    return out
