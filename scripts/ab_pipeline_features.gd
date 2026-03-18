## A/B harness for pipeline feature toggles (tighter bounds, packed stage data, SH amortization).
## Usage (requires a display or virtual framebuffer):
##   godot --script scripts/ab_pipeline_features.gd -- --scene=res://path.tscn \
##     [--node_path=/root/Scene/GaussianSplatNode3D] [--frames=120] [--warmup=30] \
##     [--configs=baseline,tighter_bounds,packed,sh_amortization,all] \
##     [--sh_divisor=10] [--output_path=user://pipeline_ab_results.json] \
##     [--csv_path=user://pipeline_ab_results.csv]
extends SceneTree

var _args: Dictionary = {}
var _scene_path: String = ""
var _node_path: NodePath = NodePath()
var _frames_total: int = 120
var _warmup_frames: int = 30
var _cooldown_frames: int = 3
var _output_path: String = "user://pipeline_ab_results.json"
var _csv_path: String = "user://pipeline_ab_results.csv"
var _sh_divisor: int = 10
var _configs: Array = []
var _config_index: int = -1
var _phase: String = "init"
var _frames_left: int = 0
var _warmup_left: int = 0
var _cooldown_left: int = 0
var _capture_start_usec: int = 0
var _capture_frames: int = 0
var _target_node: Node = null
var _renderer: Object = null
var _results: Array = []

func _init():
    _args = _parse_args()
    if not _args.has("scene"):
        printerr("Usage: --scene=<tscn> [--node_path=<node>] [--frames=<count>] [--warmup=<count>] [--configs=...]")
        quit(1)
        return

    var main_rd: RenderingDevice = RenderingServer.get_rendering_device()
    if main_rd == null:
        printerr("No RenderingDevice available. Run without --headless or use a virtual display (e.g., xvfb-run).")
        quit(6)
        return

    _scene_path = str(_args["scene"])
    _frames_total = int(_args.get("frames", "120"))
    _warmup_frames = int(_args.get("warmup", "30"))
    _output_path = str(_args.get("output_path", _output_path))
    _csv_path = str(_args.get("csv_path", _csv_path))
    _sh_divisor = int(_args.get("sh_divisor", "10"))
    if _args.has("node_path"):
        _node_path = NodePath(_args["node_path"])

    if _frames_total < 1:
        _frames_total = 1
    if _warmup_frames < 0:
        _warmup_frames = 0
    if _sh_divisor < 1:
        _sh_divisor = 1

    var packed_scene: PackedScene = load(_scene_path) as PackedScene
    if packed_scene == null:
        printerr("Failed to load scene: %s" % _scene_path)
        quit(2)
        return

    var instance: Node = packed_scene.instantiate()
    if instance == null:
        printerr("Failed to instantiate scene: %s" % _scene_path)
        quit(3)
        return

    get_root().add_child(instance)
    current_scene = instance

    _build_configs()
    if _configs.is_empty():
        printerr("No valid configs specified. Use --configs=baseline,tighter_bounds,packed,sh_amortization,all")
        quit(4)
        return

func _process(_delta: float) -> bool:
    if _renderer == null:
        _target_node = _find_target_node()
        if _target_node == null:
            printerr("Failed to locate GaussianSplatNode3D; pass --node_path to specify it explicitly.")
            quit(5)
            return true
        _renderer = _target_node.call("get_renderer")
        if _renderer == null:
            printerr("GaussianSplatNode3D returned no renderer instance.")
            quit(6)
            return true

        _renderer.call("set_debug_pipeline_trace_enabled", false)
        _renderer.call("set_debug_dump_gpu_counters", true)
        _renderer.call("set_debug_binning_counters_enabled", true)
        _start_next_config()
        return false

    if _phase == "warmup":
        _warmup_left -= 1
        if _warmup_left <= 0:
            _phase = "capture"
            _frames_left = _frames_total
            _capture_frames = 0
            _capture_start_usec = int(OS.get_singleton().get_ticks_usec())
        return false

    if _phase == "capture":
        _frames_left -= 1
        _capture_frames += 1
        if _frames_left <= 0:
            _phase = "cooldown"
            _cooldown_left = _cooldown_frames
        return false

    if _phase == "cooldown":
        _renderer.call("get_binning_debug_counters")
        _cooldown_left -= 1
        if _cooldown_left <= 0:
            _record_result()
            _start_next_config()
        return false

    return false

func _start_next_config() -> void:
    _config_index += 1
    if _config_index >= _configs.size():
        _write_results()
        quit()
        return

    var cfg: Dictionary = _configs[_config_index]
    _apply_config(cfg)
    _phase = "warmup"
    _warmup_left = _warmup_frames

func _apply_config(cfg: Dictionary) -> void:
    ProjectSettings.set_setting("rendering/gaussian_splatting/pipeline/enable_tighter_bounds", cfg.get("enable_tighter_bounds", false))
    ProjectSettings.set_setting("rendering/gaussian_splatting/pipeline/enable_packed_stage_data", cfg.get("enable_packed_stage_data", false))
    ProjectSettings.set_setting("rendering/gaussian_splatting/pipeline/enable_sh_amortization", cfg.get("enable_sh_amortization", false))
    ProjectSettings.set_setting("rendering/gaussian_splatting/pipeline/sh_amortization_divisor", cfg.get("sh_amortization_divisor", 1))
    _renderer.call("reload_pipeline_feature_set")

func _record_result() -> void:
    var elapsed_usec: int = int(OS.get_singleton().get_ticks_usec()) - _capture_start_usec
    var elapsed_sec: float = float(elapsed_usec) / 1_000_000.0
    var fps: float = elapsed_sec > 0.0 ? float(_capture_frames) / elapsed_sec : 0.0

    var stats: Dictionary = _renderer.call("get_render_stats")
    var counters: Dictionary = _renderer.call("get_binning_debug_counters")

    var entry := {
        "name": _configs[_config_index]["name"],
        "features": _configs[_config_index],
        "frames": _capture_frames,
        "elapsed_sec": elapsed_sec,
        "fps_avg": fps,
        "visible_splats": stats.get("visible_splats", 0),
        "overlap_records": stats.get("overlap_records", 0),
        "sh_cache_hit_rate": stats.get("sh_cache_hit_rate", counters.get("sh_cache_hit_rate", 0.0)),
        "render_stats": stats,
        "binning_debug_counters": counters,
    }
    _results.push_back(entry)

func _write_results() -> void:
    var output := {
        "scene": _scene_path,
        "node_path": str(_target_node.get_path()),
        "frames": _frames_total,
        "warmup_frames": _warmup_frames,
        "configs": _results,
    }

    var file := FileAccess.open(_output_path, FileAccess.WRITE)
    if file == null:
        printerr("Failed to open output path: %s" % _output_path)
    else:
        file.store_string(JSON.stringify(output, "  "))
        file.close()

    _write_csv()
    print("A/B results written to %s (csv: %s)" % [_output_path, _csv_path])

func _write_csv() -> void:
    var header := [
        "name",
        "tighter_bounds",
        "packed_stage_data",
        "sh_amortization",
        "sh_divisor",
        "fps_avg",
        "visible_splats",
        "overlap_records",
        "sh_cache_hits",
        "sh_cache_updates",
        "sh_cache_forced_updates",
        "sh_cache_hit_rate"
    ]
    var lines: Array[String] = []
    lines.push_back(",".join(header))

    for entry in _results:
        var features: Dictionary = entry.get("features", {})
        var counters: Dictionary = entry.get("binning_debug_counters", {})
        var row := [
            str(entry.get("name", "")),
            str(features.get("enable_tighter_bounds", false)),
            str(features.get("enable_packed_stage_data", false)),
            str(features.get("enable_sh_amortization", false)),
            str(features.get("sh_amortization_divisor", 1)),
            String.num(entry.get("fps_avg", 0.0), 3),
            str(entry.get("visible_splats", 0)),
            str(entry.get("overlap_records", 0)),
            str(counters.get("sh_cache_hits", 0)),
            str(counters.get("sh_cache_updates", 0)),
            str(counters.get("sh_cache_forced_updates", 0)),
            String.num(entry.get("sh_cache_hit_rate", 0.0), 4),
        ]
        lines.push_back(",".join(row))

    var file := FileAccess.open(_csv_path, FileAccess.WRITE)
    if file == null:
        printerr("Failed to open CSV path: %s" % _csv_path)
        return
    file.store_string("\n".join(lines))
    file.close()

func _build_configs() -> void:
    var list: Array[String] = []
    if _args.has("configs"):
        list = str(_args["configs"]).split(",", false)
    else:
        list = ["baseline", "tighter_bounds", "packed", "sh_amortization", "all"]

    for name in list:
        var trimmed := name.strip_edges()
        if trimmed == "":
            continue
        match trimmed:
            "baseline":
                _configs.push_back(_make_config("baseline", false, false, false))
            "tighter_bounds":
                _configs.push_back(_make_config("tighter_bounds", true, false, false))
            "packed":
                _configs.push_back(_make_config("packed", false, true, false))
            "sh_amortization":
                _configs.push_back(_make_config("sh_amortization", false, false, true))
            "all":
                _configs.push_back(_make_config("all", true, true, true))
            _:
                printerr("Unknown config: %s" % trimmed)

func _make_config(name: String, tighter: bool, packed: bool, amortize: bool) -> Dictionary:
    return {
        "name": name,
        "enable_tighter_bounds": tighter,
        "enable_packed_stage_data": packed,
        "enable_sh_amortization": amortize,
        "sh_amortization_divisor": _sh_divisor,
    }

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
