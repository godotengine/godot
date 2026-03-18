extends SceneTree

const UTIL := preload("res://modules/gaussian_splatting/tests/painterly_scenes/painterly_scene_util.gd")
const DEFAULT_OUTPUT_DIR := "res://modules/gaussian_splatting/tests/painterly_scenes/references"
const DEFAULT_MANIFEST_FILE := "manifest.json"

var output_dir: String = DEFAULT_OUTPUT_DIR
var manifest_path: String = ""
var image_size: Vector2i = Vector2i(256, 256)
var require_rendering_device: bool = false

var _success: bool = true
var _warnings: Array[String] = []
var _failures: Array[String] = []
var _scene_reports: Array = []
var _requested_help: bool = false

## Entry point for headless capture: generates references then exits.
func _init() -> void:
    if not _parse_args():
        quit(0 if _requested_help else 1)
        return

    capture_all()
    if _success:
        print("PAINTERLY_REFERENCE_CAPTURE_COMPLETE")
        quit(0)
        return
    print("PAINTERLY_REFERENCE_CAPTURE_FAILED")
    quit(1)

## Renders each painterly scene and saves camera reference images to disk.
func capture_all() -> void:
    var scene_paths: Array = UTIL.list_scene_definition_paths()
    if manifest_path.is_empty():
        manifest_path = "%s/%s" % [output_dir, DEFAULT_MANIFEST_FILE]

    _ensure_directory(output_dir)
    _ensure_directory(manifest_path.get_base_dir())

    var rd: RenderingDevice = UTIL.ensure_rendering_device()
    if rd == null:
        _record_warning("RenderingDevice unavailable - capturing references with shader compilation skipped")
        if require_rendering_device:
            _record_failure("RenderingDevice required for capture but unavailable")

    for path in scene_paths:
        var definition: Dictionary = UTIL.load_definition(path)
        if definition.is_empty():
            _record_failure("Skipping painterly reference capture for missing definition: %s" % path)
            continue
        var scene: Dictionary = UTIL.build_scene(definition, path)
        var scene_name: String = str(scene.get("name", path.get_file().get_basename()))
        var scene_slug: String = _slugify(scene_name)
        var scene_dir := "%s/%s" % [output_dir, scene_slug]
        _ensure_directory(scene_dir)

        var splats: Array = UTIL.generate_gaussians(scene)
        var compile_results: Dictionary = {}
        var compile_failures: Array[String] = []

        if rd != null:
            compile_results = UTIL.compile_permutations(scene, rd)
            for perm_name in compile_results:
                if not bool(compile_results[perm_name]):
                    compile_failures.append(str(perm_name))
        if not compile_failures.is_empty():
            _record_failure("[%s] Shader permutations failed: %s" % [scene_name, _join_string_array(compile_failures)])

        var camera_indices: Array = _camera_indices_for_scene(scene)
        var files: Array = []
        var stats_by_camera: Array = []
        for index_variant in camera_indices:
            var camera_index: int = int(index_variant)
            var image: Image = UTIL.render_headless(scene, splats, camera_index, image_size)
            var filename := "%s_camera_%02d.png" % [scene_slug, camera_index]
            var filepath := "%s/%s" % [scene_dir, filename]
            if _save_image(image, filepath):
                files.append(filepath)
                stats_by_camera.append({
                    "camera_index": camera_index,
                    "stats": UTIL.compute_image_stats(image)
                })

        _scene_reports.append({
            "name": scene_name,
            "definition_path": path,
            "image_size": [image_size.x, image_size.y],
            "camera_indices": camera_indices,
            "compile": {
                "checked": rd != null,
                "total": compile_results.size(),
                "failed": compile_failures
            },
            "files": files,
            "camera_stats": stats_by_camera
        })
        print("Painterly references -> %s (%d images)" % [scene_name, files.size()])

    _write_manifest()
    print("Painterly reference capture complete -> %s" % output_dir)

## Saves a rendered reference image into the output directory.
## @param image: Image to save.
## @param filepath: Target filepath.
## @return true when save succeeds.
func _save_image(image: Image, filepath: String) -> bool:
    if image == null:
        _record_failure("Cannot save null painterly image: %s" % filepath)
        return false
    var err: Error = image.save_png(filepath)
    if err != OK:
        _record_failure("Failed to save painterly reference: %s (error=%d)" % [filepath, err])
        return false
    return true

func _camera_indices_for_scene(scene: Dictionary) -> Array:
    var camera_path: Array = scene.get("camera_path", [])
    var camera_count: int = camera_path.size()
    if camera_count <= 0:
        return [0]
    var audit: Dictionary = scene.get("audit", {})
    var indices: Array = []
    var configured = audit.get("artifact_camera_indices", [])
    if typeof(configured) == TYPE_ARRAY:
        for index_variant in configured:
            var camera_index: int = clamp(int(index_variant), 0, camera_count - 1)
            if not indices.has(camera_index):
                indices.append(camera_index)
    if indices.is_empty():
        indices.append(0)
        if camera_count > 1:
            var last_index: int = camera_count - 1
            if not indices.has(last_index):
                indices.append(last_index)
    return indices

func _write_manifest() -> void:
    if manifest_path.is_empty():
        return

    _ensure_directory(manifest_path.get_base_dir())
    var payload := {
        "success": _success,
        "timestamp_unix": Time.get_unix_time_from_system(),
        "output_dir": output_dir,
        "manifest_path": manifest_path,
        "image_size": [image_size.x, image_size.y],
        "warnings": _warnings,
        "failures": _failures,
        "scenes": _scene_reports
    }

    var file: FileAccess = FileAccess.open(manifest_path, FileAccess.WRITE)
    if file == null:
        _record_failure("Unable to write painterly reference manifest: %s" % manifest_path)
        return
    file.store_string(JSON.stringify(payload, "  "))
    file.close()
    print("Painterly reference manifest -> %s" % manifest_path)

func _parse_args() -> bool:
    var args: PackedStringArray = OS.get_cmdline_user_args()
    for arg in args:
        if arg == "--":
            continue
        elif arg == "--help" or arg == "-h":
            _requested_help = true
            _print_usage()
            return false
        elif arg == "--require-rendering-device":
            require_rendering_device = true
        elif arg.begins_with("--output-dir="):
            output_dir = arg.substr("--output-dir=".length())
        elif arg.begins_with("--manifest-path="):
            manifest_path = arg.substr("--manifest-path=".length())
        elif arg.begins_with("--image-size="):
            var size_token: String = arg.substr("--image-size=".length())
            if not _parse_image_size(size_token):
                push_error("Invalid --image-size value: %s (expected WIDTHxHEIGHT)" % size_token)
                return false
        else:
            push_error("Unknown painterly reference argument: %s" % arg)
            return false

    if manifest_path.is_empty():
        manifest_path = "%s/%s" % [output_dir, DEFAULT_MANIFEST_FILE]
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
    image_size = Vector2i(width, height)
    return true

func _print_usage() -> void:
    print("Painterly reference capture usage:")
    print("  godot --headless --script scripts/tools/capture_painterly_references.gd -- [options]")
    print("Options:")
    print("  --output-dir=<path>              Capture output directory (default: %s)" % DEFAULT_OUTPUT_DIR)
    print("  --manifest-path=<path>           Manifest JSON output path")
    print("  --image-size=<W>x<H>             Render resolution (default: 256x256)")
    print("  --require-rendering-device       Fail if RenderingDevice is unavailable")

func _ensure_directory(path: String) -> void:
    if path.is_empty():
        return
    var absolute_path: String = ProjectSettings.globalize_path(path)
    var err: Error = DirAccess.make_dir_recursive_absolute(absolute_path)
    if err != OK and err != ERR_ALREADY_EXISTS:
        _record_failure("Failed to create directory: %s" % path)

func _slugify(value: String) -> String:
    var slug := value.strip_edges()
    if slug.is_empty():
        slug = "scene"
    slug = slug.replace(" ", "_")
    slug = slug.replace("/", "_")
    slug = slug.replace("\\", "_")
    return slug

func _record_warning(message: String) -> void:
    _warnings.append(message)
    push_warning(message)

func _record_failure(message: String) -> void:
    _failures.append(message)
    _success = false
    push_error(message)

func _join_string_array(items: Array) -> String:
    if items.is_empty():
        return ""
    var out: String = ""
    for i in range(items.size()):
        if i > 0:
            out += ", "
        out += str(items[i])
    return out
