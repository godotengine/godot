## Captures a pipeline trace and baseline stats for a GaussianSplatNode3D scene.
## Usage (requires a display or virtual framebuffer):
##   godot --script scripts/capture_pipeline_baseline.gd -- --scene=res://path.tscn \
##     [--node_path=/root/Scene/GaussianSplatNode3D] [--frames=120] \
##     [--trace_path=user://pipeline_trace_baseline.json] [--baseline_path=user://pipeline_baseline.json]
extends SceneTree

var _args: Dictionary = {}
var _frames_left: int = 0
var _node_path: NodePath = NodePath()
var _trace_path: String = "user://pipeline_trace_baseline.json"
var _baseline_path: String = "user://pipeline_baseline.json"
var _target_node: Node = null
var _renderer: Object = null
var _capture_started: bool = false

func _init():
    _args = _parse_args()
    if not _args.has("scene"):
        printerr("Usage: --scene=<tscn> [--node_path=<node>] [--frames=<count>] [--trace_path=<path>] [--baseline_path=<path>]")
        quit(1)
        return

    var main_rd: RenderingDevice = RenderingServer.get_rendering_device()
    if main_rd == null:
        printerr("No RenderingDevice available. Run without --headless or use a virtual display (e.g., xvfb-run).")
        quit(6)
        return

    _frames_left = int(_args.get("frames", "120"))
    if _frames_left < 1:
        _frames_left = 1

    if _args.has("node_path"):
        _node_path = NodePath(_args["node_path"])

    _trace_path = _args.get("trace_path", _trace_path)
    _baseline_path = _args.get("baseline_path", _baseline_path)

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

    if _frames_left > 0:
        _frames_left -= 1
        return false

    _dump_baseline()
    return true

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

    _renderer.call("set_debug_pipeline_trace_enabled", true)
    return true

func _dump_baseline() -> void:
    var trace_err: Error = _renderer.call("dump_pipeline_trace_to_file", _trace_path)
    var stats: Dictionary = _renderer.call("get_render_stats")
    var sort_metrics: Dictionary = _renderer.call("get_last_sort_metrics")

    var baseline := {
        "scene": _args.get("scene", ""),
        "node_path": str(_target_node.get_path()),
        "frames": int(_args.get("frames", "120")),
        "trace_path": _trace_path,
        "trace_dump_error": trace_err,
        "visible_splats": _renderer.call("get_visible_splat_count"),
        "sort_time_ms": _renderer.call("get_sort_time_ms"),
        "render_time_ms": _renderer.call("get_render_time_ms"),
        "viewport_copy_success": _renderer.call("was_last_viewport_copy_successful"),
        "viewport_copy_source_size": _vector2i_to_array(_renderer.call("get_last_viewport_copy_source_size")),
        "viewport_copy_dest_size": _vector2i_to_array(_renderer.call("get_last_viewport_copy_dest_size")),
        "render_stats": stats,
        "sort_metrics": sort_metrics,
    }

    var file := FileAccess.open(_baseline_path, FileAccess.WRITE)
    if file == null:
        printerr("Failed to open baseline output: %s" % _baseline_path)
    else:
        file.store_string(JSON.stringify(baseline, "  "))
        file.close()

    print("Baseline written to %s (trace: %s, err=%s)" % [_baseline_path, _trace_path, str(trace_err)])
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

func _vector2i_to_array(value) -> Array:
    if typeof(value) == TYPE_VECTOR2I:
        return [value.x, value.y]
    if typeof(value) == TYPE_VECTOR2:
        return [value.x, value.y]
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
