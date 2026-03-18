## Drives a camera away and back while sampling culling stats each frame.
## Usage (requires a display or virtual framebuffer):
##   godot --script scripts/culling_investigation_path.gd -- --scene=res://path.tscn \
##     [--node_path=/root/Scene/GaussianSplatNode3D] [--camera_path=/root/Scene/Camera3D] \
##     [--frames=240] [--distance=50.0] [--log_every=1] [--output_path=user://culling_path_report.json]
extends SceneTree

var _args: Dictionary = {}
var _frames_total: int = 0
var _frame_index: int = 0
var _node_path: NodePath = NodePath()
var _camera_path: NodePath = NodePath()
var _output_path: String = "user://culling_path_report.json"
var _target_node: Node = null
var _renderer: Object = null
var _camera: Camera3D = null
var _start_transform: Transform3D = Transform3D()
var _distance: float = 50.0
var _log_every: int = 1
var _capture_started: bool = false
var _samples: Array = []

func _init():
    _args = _parse_args()
    if not _args.has("scene"):
        printerr("Usage: --scene=<tscn> [--node_path=<node>] [--camera_path=<node>] [--frames=<count>] [--distance=<meters>] [--log_every=<n>] [--output_path=<path>]")
        quit(1)
        return

    var main_rd: RenderingDevice = RenderingServer.get_rendering_device()
    if main_rd == null:
        printerr("No RenderingDevice available. Run without --headless or use a virtual display (e.g., xvfb-run).")
        quit(6)
        return

    _frames_total = int(_args.get("frames", "240"))
    if _frames_total < 2:
        _frames_total = 2

    if _args.has("node_path"):
        _node_path = NodePath(_args["node_path"])
    if _args.has("camera_path"):
        _camera_path = NodePath(_args["camera_path"])

    _distance = float(_args.get("distance", "50.0"))
    _log_every = int(_args.get("log_every", "1"))
    if _log_every < 1:
        _log_every = 1

    _output_path = _args.get("output_path", _output_path)

    var packed_scene: PackedScene = load(_args["scene"]) as PackedScene
    if packed_scene == null:
        printerr("Failed to load scene: %s" % _args["scene"])
        quit(2)
        return

    var instance: Node = packed_scene.instantiate()
    if instance == null:
        printerr("Failed to instantiate scene: %s" % _args["scene"])
        quit(3)
        return

    get_root().add_child(instance)
    if instance is Node:
        current_scene = instance

func _process(_delta: float) -> bool:
    if not _capture_started:
        _capture_started = _start_capture()
        return false

    if _frame_index >= _frames_total:
        _dump_report()
        return true

    _drive_camera()
    if (_frame_index % _log_every) == 0:
        _capture_frame()

    _frame_index += 1
    return false

func _start_capture() -> bool:
    _target_node = _find_target_node()
    if _target_node == null:
        printerr("Failed to locate GaussianSplatNode3D; pass --node_path to specify it explicitly.")
        quit(4)
        return false

    _renderer = _target_node.call("get_renderer")
    if _renderer == null:
        printerr("GaussianSplatNode3D returned no renderer instance.")
        quit(5)
        return false

    _camera = _find_camera()
    if _camera == null:
        printerr("Failed to locate Camera3D; pass --camera_path to specify it explicitly.")
        quit(7)
        return false

    _start_transform = _camera.global_transform
    _renderer.call("set_debug_cull_guardrails_enabled", true)
    return true

func _drive_camera() -> void:
    if _frames_total <= 1:
        return

    var t: float = float(_frame_index) / float(_frames_total - 1)
    var phase: float = t * 2.0
    var distance: float = phase * _distance if phase <= 1.0 else (2.0 - phase) * _distance

    var forward: Vector3 = -_start_transform.basis.z.normalized()
    var new_transform: Transform3D = _start_transform
    new_transform.origin = _start_transform.origin + forward * distance
    _camera.global_transform = new_transform

func _capture_frame() -> void:
    var stats: Dictionary = _renderer.call("get_render_stats")
    var entry: Dictionary = {
        "frame": _frame_index,
        "camera_pos": _vector3_to_array(_camera.global_transform.origin),
        "visible_splats": int(_renderer.call("get_visible_splat_count")),
        "stage_cull_visible": int(stats.get("stage_cull_visible_count", 0)),
        "stage_cull_candidates": int(stats.get("stage_cull_candidate_count", 0)),
        "cull_visible_static_chunks": int(stats.get("cull_visible_static_chunks", 0)),
        "cull_static_chunk_total": int(stats.get("cull_static_chunk_total", 0)),
        "cull_gpu_visible": int(stats.get("cull_gpu_visible_count", 0)),
        "cull_cpu_visible": int(stats.get("cull_cpu_visible_count", 0)),
        "cull_total_splats_pre_cull": int(stats.get("cull_total_splats_pre_cull", 0)),
    }
    _samples.append(entry)

func _dump_report() -> void:
    var report: Dictionary = {
        "scene": _args.get("scene", ""),
        "node_path": str(_target_node.get_path()) if _target_node != null else "",
        "camera_path": str(_camera.get_path()) if _camera != null else "",
        "frames": _frames_total,
        "distance": _distance,
        "log_every": _log_every,
        "output_path": _output_path,
        "samples": _samples,
    }

    var file := FileAccess.open(_output_path, FileAccess.WRITE)
    if file == null:
        printerr("Failed to open output: %s" % _output_path)
    else:
        file.store_string(JSON.stringify(report, "  "))
        file.close()

    print("Culling report written to %s (samples=%d)" % [_output_path, _samples.size()])
    quit()

func _find_target_node() -> Node:
    if _node_path != NodePath():
        var by_path := get_root().get_node_or_null(_node_path)
        if by_path != null:
            return by_path

    var queue: Array[Node] = [get_root()]
    while not queue.is_empty():
        var node: Node = queue.pop_front()
        if node.get_class() == "GaussianSplatNode3D":
            return node
        for child in node.get_children():
            if child is Node:
                queue.push_back(child)
    return null

func _find_camera() -> Camera3D:
    if _camera_path != NodePath():
        var by_path := get_root().get_node_or_null(_camera_path)
        if by_path is Camera3D:
            return by_path

    var queue: Array[Node] = [get_root()]
    while not queue.is_empty():
        var node: Node = queue.pop_front()
        if node is Camera3D:
            return node
        for child in node.get_children():
            if child is Node:
                queue.push_back(child)
    return null

func _vector3_to_array(value) -> Array:
    if typeof(value) == TYPE_VECTOR3:
        return [value.x, value.y, value.z]
    return []

## Parses --key=value arguments from the command line.
## @return Dictionary of parsed arguments.
func _parse_args() -> Dictionary:
    var result := {}
    for arg in OS.get_cmdline_user_args():
        if arg.begins_with("--"):
            var tokens := arg.substr(2).split("=", false, 2)
            if tokens.size() == 2:
                result[tokens[0]] = tokens[1]
    return result
