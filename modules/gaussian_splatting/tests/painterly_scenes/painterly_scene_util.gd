extends RefCounted
class_name PainterlySceneUtil

const DEFAULT_PALETTE := [
    Color(0.82, 0.72, 0.63, 0.9),
    Color(0.32, 0.36, 0.42, 0.85),
    Color(0.91, 0.52, 0.36, 0.88)
]
const SCENE_DEFINITION_PATHS := [
    "res://modules/gaussian_splatting/tests/painterly_scenes/dense_atelier.json",
    "res://modules/gaussian_splatting/tests/painterly_scenes/sparse_gallery.json",
    "res://modules/gaussian_splatting/tests/painterly_scenes/animated_orbit.json"
]

static func load_definition(path: String) -> Dictionary:
    var file: FileAccess = FileAccess.open(path, FileAccess.READ)
    if file == null:
        push_error("Unable to open painterly scene definition: %s" % path)
        return {}
    var text: String = file.get_as_text()
    file.close()
    var result = JSON.parse_string(text)
    if typeof(result) != TYPE_DICTIONARY:
        push_error("Painterly scene definition must be a dictionary: %s" % path)
        return {}
    return result

static func build_scene(definition: Dictionary, path: String = "") -> Dictionary:
    var scene: Dictionary = {}
    scene["name"] = definition.get("name", path.get_file())
    scene["density"] = definition.get("density", "dense")
    scene["splat_count"] = int(definition.get("splat_count", 1024))
    scene["radius"] = float(definition.get("radius", 0.25))
    scene["fov"] = float(definition.get("fov_degrees", 60.0))
    var palette: Array = []
    for entry in definition.get("palette", []):
        if typeof(entry) == TYPE_COLOR:
            palette.append(entry)
        elif typeof(entry) == TYPE_ARRAY and entry.size() >= 3:
            var alpha := float(entry[3]) if entry.size() > 3 else 0.85
            palette.append(Color(entry[0], entry[1], entry[2], alpha))
    if palette.is_empty():
        palette = DEFAULT_PALETTE.duplicate()
    scene["palette"] = palette

    var camera_path: Array = []
    for camera_entry in definition.get("camera_path", []):
        if typeof(camera_entry) == TYPE_DICTIONARY:
            camera_path.append({
                "time": float(camera_entry.get("time", camera_path.size())),
                "position": _parse_vector3(camera_entry.get("position", Vector3(0, 0, 6))),
                "target": _parse_vector3(camera_entry.get("target", Vector3.ZERO))
            })
    if camera_path.is_empty():
        camera_path.append({
            "time": 0.0,
            "position": Vector3(0, 0, 6),
            "target": Vector3.ZERO
        })
    scene["camera_path"] = camera_path

    var permutations: Array = []
    for perm_entry in definition.get("shader_permutations", []):
        if typeof(perm_entry) == TYPE_DICTIONARY:
            var defines: Array = []
            for value in perm_entry.get("defines", []):
                defines.append(str(value))
            permutations.append({
                "name": perm_entry.get("name", "perm_%d" % permutations.size()),
                "defines": defines
            })
    if permutations.is_empty():
        permutations.append({
            "name": "%s_default" % scene.get("name"),
            "defines": ["PAINTERLY_STYLE_BRUSH"]
        })
    scene["permutations"] = permutations
    scene["audit"] = _normalize_audit_config(definition.get("audit", {}), scene)
    return scene

static func list_scene_definition_paths() -> Array:
    return SCENE_DEFINITION_PATHS.duplicate()

static func generate_gaussians(scene: Dictionary, seed: int = 1337) -> Array:
    var rng: RandomNumberGenerator = RandomNumberGenerator.new()
    rng.seed = seed + int(hash(scene.get("name", "painterly")))
    var splats: Array = []
    var dense: bool = scene.get("density", "dense") == "dense"
    var radius: float = float(scene.get("radius", 0.25))
    var palette: Array = scene.get("palette", DEFAULT_PALETTE)
    if palette.is_empty():
        palette = DEFAULT_PALETTE

    var splat_count: int = int(scene.get("splat_count", 0))
    for i in range(splat_count):
        var base_position: Vector3
        if dense:
            var angle: float = rng.randf_range(0.0, TAU)
            var ring: float = rng.randf_range(0.0, 2.5)
            base_position = Vector3(cos(angle) * ring, rng.randf_range(-1.5, 1.5), sin(angle) * ring)
        else:
            base_position = Vector3(
                rng.randf_range(-3.5, 3.5),
                rng.randf_range(-2.5, 2.5),
                rng.randf_range(-1.5, 1.5)
            )
        base_position.z += rng.randf_range(-0.8, 0.8)

        var scale_variation: float = rng.randf_range(radius * 0.4, radius) if dense else rng.randf_range(radius * 0.6, radius * 1.4)
        var opacity: float = rng.randf_range(0.65, 0.95) if dense else rng.randf_range(0.35, 0.75)
        var palette_color: Color = palette[i % palette.size()]

        splats.append({
            "position": base_position,
            "scale": scale_variation,
            "color": Color(palette_color.r, palette_color.g, palette_color.b, opacity),
            "opacity": opacity
        })
    return splats

static func render_headless(scene: Dictionary, splats: Array, camera_index: int = 0, size: Vector2i = Vector2i(128, 128)) -> Image:
    var width: int = size.x
    var height: int = size.y
    var image: Image = Image.create(width, height, false, Image.FORMAT_RGBA8)
    image.fill(Color(0, 0, 0, 0))

    var camera_keys: Array = scene.get("camera_path", [])
    if camera_keys.is_empty():
        return image
    camera_index = clamp(camera_index, 0, camera_keys.size() - 1)
    var camera_variant = camera_keys[camera_index]
    var camera: Dictionary = camera_variant if typeof(camera_variant) == TYPE_DICTIONARY else {}
    var cam_pos: Vector3 = camera.get("position", Vector3(0, 0, 6))
    var cam_target: Vector3 = camera.get("target", Vector3.ZERO)

    var view: Transform3D = Transform3D(Basis().looking_at((cam_target - cam_pos).normalized(), Vector3.UP), cam_pos).affine_inverse()
    var focal: float = 0.5 * float(width) / tan(deg_to_rad(float(scene.get("fov", 60.0))) * 0.5)

    var order: Array = []
    order.resize(splats.size())
    for i in range(splats.size()):
        order[i] = i
    order.sort_custom(func(a, b):
        var pos_a: Vector3 = splats[a]["position"]
        var pos_b: Vector3 = splats[b]["position"]
        var depth_a := -(view * pos_a).z
        var depth_b := -(view * pos_b).z
        return depth_a > depth_b
    )

    for idx in order:
        var splat_variant = splats[idx]
        if typeof(splat_variant) != TYPE_DICTIONARY:
            continue
        var splat: Dictionary = splat_variant
        var position: Vector3 = splat.get("position", Vector3.ZERO)
        var scale_value: float = float(splat.get("scale", 1.0))
        var color: Color = splat.get("color", Color.WHITE)
        var opacity: float = float(splat.get("opacity", 1.0))

        var camera_space: Vector3 = view * position
        var depth: float = -camera_space.z
        if depth <= 0.01:
            continue

        var ndc_x: float = camera_space.x / depth
        var ndc_y: float = camera_space.y / depth
        var screen_x: float = float(width) * 0.5 + ndc_x * focal
        var screen_y: float = float(height) * 0.5 - ndc_y * focal
        var pixel_radius: float = max(1.0, scale_value * focal / depth)

        var min_x: int = max(0, int(floor(screen_x - pixel_radius * 2.0)))
        var max_x: int = min(width - 1, int(ceil(screen_x + pixel_radius * 2.0)))
        var min_y: int = max(0, int(floor(screen_y - pixel_radius * 2.0)))
        var max_y: int = min(height - 1, int(ceil(screen_y + pixel_radius * 2.0)))

        for y in range(min_y, max_y + 1):
            var dy: float = (float(y) + 0.5 - screen_y) / pixel_radius
            var dy2: float = dy * dy
            for x in range(min_x, max_x + 1):
                var dx: float = (float(x) + 0.5 - screen_x) / pixel_radius
                var dist_sq: float = dx * dx + dy2
                if dist_sq > 4.0:
                    continue
                var weight: float = exp(-dist_sq * 1.5)
                if weight < 0.01:
                    continue
                var alpha: float = opacity * weight
                var src: Color = Color(color.r, color.g, color.b, alpha)
                var dst: Color = image.get_pixel(x, y)
                var out_alpha: float = src.a + dst.a * (1.0 - src.a)
                if out_alpha <= 0.0001:
                    continue
                var out_color: Color = Color(
                    (src.r * src.a + dst.r * dst.a * (1.0 - src.a)) / out_alpha,
                    (src.g * src.a + dst.g * dst.a * (1.0 - src.a)) / out_alpha,
                    (src.b * src.a + dst.b * dst.a * (1.0 - src.a)) / out_alpha,
                    out_alpha
                )
                image.set_pixel(x, y, out_color)
    return image

static func ensure_rendering_device() -> RenderingDevice:
    var rd: RenderingDevice = RenderingServer.get_rendering_device()
    if rd == null:
        rd = RenderingServer.create_local_rendering_device()
    return rd

static func compile_permutations(scene: Dictionary, rd: RenderingDevice) -> Dictionary:
    var results: Dictionary = {}
    if rd == null:
        return results
    var shader_path := "res://modules/gaussian_splatting/shaders/painterly_resolve.glsl"
    var shader_text: String = FileAccess.get_file_as_string(shader_path)
    if shader_text.is_empty():
        push_error("Painterly shader source missing: %s" % shader_path)
        return results
    for perm_variant in scene.get("permutations", []):
        if typeof(perm_variant) != TYPE_DICTIONARY:
            continue
        var perm: Dictionary = perm_variant
        var defines_block: String = ""
        for define_value in perm.get("defines", []):
            var define := str(define_value).strip_edges()
            if define.is_empty():
                continue
            if define.begins_with("#"):
                defines_block += define + "\n"
            else:
                defines_block += "#define %s\n" % define
        var final_source: String = shader_text
        var version_index: int = shader_text.find("#version")
        if version_index >= 0:
            var newline_index: int = shader_text.find("\n", version_index)
            if newline_index >= 0:
                final_source = shader_text.substr(0, newline_index + 1) + defines_block + shader_text.substr(newline_index + 1)
            else:
                final_source = shader_text + "\n" + defines_block
        else:
            final_source = defines_block + shader_text
        var shader_source: RDShaderSource = RDShaderSource.new()
        shader_source.source_compute = final_source
        var spirv: RDShaderSPIRV = rd.shader_compile_spirv_from_source(shader_source)
        var perm_name: String = str(perm.get("name", "perm_%d" % results.size()))
        if spirv == null:
            results[perm_name] = false
            continue
        var compile_error: String = spirv.compile_error_compute
        var bytecode: PackedByteArray = spirv.bytecode_compute
        var compile_ok: bool = compile_error.is_empty() and not bytecode.is_empty()
        if not compile_ok:
            if compile_error.is_empty():
                push_warning("Painterly compute permutation produced no bytecode: %s" % perm_name)
            else:
                push_warning("Painterly compute permutation failed (%s): %s" % [perm_name, compile_error])
        results[perm_name] = compile_ok
    return results

static func compute_image_stats(image: Image) -> Dictionary:
    if image == null:
        return {
            "coverage": 0.0,
            "luminance": 0.0,
            "luminance_stddev": 0.0,
            "alpha_mean": 0.0
        }

    var width: int = image.get_width()
    var height: int = image.get_height()
    var total: int = width * height
    if total <= 0:
        return {
            "coverage": 0.0,
            "luminance": 0.0,
            "luminance_stddev": 0.0,
            "alpha_mean": 0.0
        }

    var coverage: float = 0.0
    var luminance_sum: float = 0.0
    var luminance_sq_sum: float = 0.0
    var alpha_sum: float = 0.0
    for y in range(height):
        for x in range(width):
            var c: Color = image.get_pixel(x, y)
            if c.a > 0.01:
                coverage += 1.0
            var luminance: float = c.get_luminance()
            luminance_sum += luminance
            luminance_sq_sum += luminance * luminance
            alpha_sum += c.a

    var luminance_mean: float = luminance_sum / max(1, total)
    var luminance_variance: float = max(0.0, luminance_sq_sum / max(1, total) - luminance_mean * luminance_mean)
    return {
        "coverage": coverage / max(1, total),
        "luminance": luminance_mean,
        "luminance_stddev": sqrt(luminance_variance),
        "alpha_mean": alpha_sum / max(1, total)
    }

static func compute_image_difference(a: Image, b: Image) -> float:
    if a == null or b == null:
        return 0.0
    if a.get_width() != b.get_width() or a.get_height() != b.get_height():
        return 0.0
    var width: int = a.get_width()
    var height: int = a.get_height()
    var total: int = width * height
    var diff: float = 0.0
    for y in range(height):
        for x in range(width):
            var ca: Color = a.get_pixel(x, y)
            var cb: Color = b.get_pixel(x, y)
            diff += abs(ca.r - cb.r)
            diff += abs(ca.g - cb.g)
            diff += abs(ca.b - cb.b)
    return diff / (max(1, total) * 3.0)

static func _parse_vector3(value, default_value: Vector3 = Vector3.ZERO) -> Vector3:
    if typeof(value) == TYPE_VECTOR3:
        return value
    if typeof(value) == TYPE_ARRAY and value.size() >= 3:
        return Vector3(value[0], value[1], value[2])
    return default_value

static func _normalize_audit_config(audit_source, scene: Dictionary) -> Dictionary:
    var camera_path: Array = scene.get("camera_path", [])
    var camera_count: int = camera_path.size()
    var is_dense: bool = scene.get("density", "dense") == "dense"

    var defaults: Dictionary = {
        "coverage_min": 0.18 if is_dense else 0.05,
        "coverage_max": -1.0,
        "luminance_min": 0.01,
        "luminance_max": -1.0,
        "delta_min": 0.015 if camera_count > 1 else 0.0,
        "stability_max": 0.0005,
        "comparison_camera_index": max(1, camera_count / 2) if camera_count > 1 else 0,
        "artifact_camera_indices": [],
        "performance_budget_ms": 0.0
    }

    var source: Dictionary = {}
    if typeof(audit_source) == TYPE_DICTIONARY:
        source = audit_source

    var comparison_index: int = int(source.get("comparison_camera_index", defaults["comparison_camera_index"]))
    if camera_count > 0:
        comparison_index = clamp(comparison_index, 0, camera_count - 1)
    else:
        comparison_index = 0

    var artifact_indices: Array = []
    var source_indices = source.get("artifact_camera_indices", [])
    if typeof(source_indices) == TYPE_ARRAY:
        for index_variant in source_indices:
            if typeof(index_variant) == TYPE_INT or typeof(index_variant) == TYPE_FLOAT:
                _append_unique_index(artifact_indices, int(index_variant), camera_count)

    if artifact_indices.is_empty():
        _append_unique_index(artifact_indices, 0, camera_count)
        if camera_count > 1:
            _append_unique_index(artifact_indices, comparison_index, camera_count)
            _append_unique_index(artifact_indices, camera_count - 1, camera_count)

    return {
        "coverage_min": float(source.get("coverage_min", defaults["coverage_min"])),
        "coverage_max": float(source.get("coverage_max", defaults["coverage_max"])),
        "luminance_min": float(source.get("luminance_min", defaults["luminance_min"])),
        "luminance_max": float(source.get("luminance_max", defaults["luminance_max"])),
        "delta_min": float(source.get("delta_min", defaults["delta_min"])),
        "stability_max": max(0.0, float(source.get("stability_max", defaults["stability_max"]))),
        "comparison_camera_index": comparison_index,
        "artifact_camera_indices": artifact_indices,
        "performance_budget_ms": max(0.0, float(source.get("performance_budget_ms", defaults["performance_budget_ms"])))
    }

static func _append_unique_index(indices: Array, index: int, max_count: int) -> void:
    if max_count <= 0:
        return
    var safe_index: int = clamp(index, 0, max_count - 1)
    if not indices.has(safe_index):
        indices.append(safe_index)
