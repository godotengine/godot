## CLI bake utility for GaussianSplatWorld resources.
## Usage (scene-based, preserves transforms):
##   godot --headless --path <project> --script scripts/bake_gsplatworld.gd -- \
##     --scene=res://scenes/world.tscn --container=/root/World/GaussianSplatContainer \
##     --output=res://worlds/world.gsplatworld --chunk_size=25.0
##
## Usage (input list, identity transforms):
##   godot --headless --path <project> --script scripts/bake_gsplatworld.gd -- \
##     --inputs=res://a.ply,res://b.spz --output=res://worlds/world.gsplatworld --chunk_size=25.0
extends SceneTree

func _init():
    # Defer to allow scene tree to initialize
    call_deferred("_run_bake")

func _run_bake():
    var args := _parse_args()
    if args.has("help") or args.has("h"):
        _print_usage()
        quit(0)
        return

    if not args.has("output"):
        printerr("Missing --output path.")
        _print_usage()
        quit(1)
        return

    var world: GaussianSplatWorld = null
    if args.has("scene"):
        world = _bake_from_scene(args)
    elif args.has("inputs") or args.has("input"):
        world = _bake_from_inputs(args)
    else:
        printerr("Missing --scene or --inputs.")
        _print_usage()
        quit(1)
        return

    if world == null:
        printerr("Bake failed; no world resource produced.")
        quit(2)
        return

    var output_path: String = args["output"]
    if not output_path.ends_with(".gsplatworld"):
        printerr("Output must end with .gsplatworld: %s" % output_path)
        quit(3)
        return

    var save_err := ResourceSaver.save(world, output_path)
    if save_err != OK:
        printerr("Failed to save gsplatworld: %s (err=%d)" % [output_path, save_err])
        quit(4)
        return

    var splat_count := 0
    var gdata = world.get_gaussian_data()
    if gdata != null:
        splat_count = gdata.get_count()
    print("Saved gsplatworld: %s (splats=%d, chunks=%d)" % [
        output_path, splat_count, world.get_chunk_count()
    ])
    quit(0)

func _bake_from_scene(args: Dictionary) -> GaussianSplatWorld:
    var scene_path := String(args["scene"])
    var packed := load(scene_path)
    if packed == null or not (packed is PackedScene):
        printerr("Failed to load scene: %s" % scene_path)
        return null

    var scene_root: Node = packed.instantiate()
    if scene_root == null:
        printerr("Failed to instantiate scene: %s" % scene_path)
        return null

    # Add to tree so transforms work
    root.add_child(scene_root)

    var container_path := ""
    if args.has("container"):
        container_path = String(args["container"])

    var container := _find_container(scene_root, container_path)
    if container == null:
        scene_root.queue_free()
        return null

    _apply_chunk_size(container, args)
    _ensure_container_assets(container)
    container.merge_children()

    var world := container.export_world_resource()
    scene_root.queue_free()
    if world == null:
        printerr("Container produced no world resource.")
    return world

func _bake_from_inputs(args: Dictionary) -> GaussianSplatWorld:
    var inputs_arg := ""
    if args.has("inputs"):
        inputs_arg = String(args["inputs"])
    else:
        inputs_arg = String(args["input"])

    var paths := _split_list(inputs_arg)
    if paths.is_empty():
        printerr("No inputs provided.")
        return null

    # Add container to tree so transforms work
    var container := GaussianSplatContainer.new()
    root.add_child(container)
    _apply_chunk_size(container, args)

    for path in paths:
        print("Loading asset: %s" % path)
        var asset := _load_asset(path)
        if asset == null:
            container.queue_free()
            return null
        print("Loaded %d splats from %s" % [asset.get_splat_count(), path])
        var node := GaussianSplatNode3D.new()
        node.set_splat_asset(asset)
        container.add_child(node)

    print("Merging %d children..." % container.get_child_count())
    container.merge_children()

    var world := container.export_world_resource()
    container.queue_free()
    if world == null:
        printerr("Container produced no world resource.")
    return world

func _find_container(scene_root: Node, container_path: String) -> GaussianSplatContainer:
    if container_path != "":
        var node := scene_root.get_node_or_null(container_path)
        if node == null:
            printerr("Container path not found: %s" % container_path)
            return null
        var container := node as GaussianSplatContainer
        if container == null:
            printerr("Node at %s is not a GaussianSplatContainer." % container_path)
        return container

    var matches := root.find_children("*", "GaussianSplatContainer", true, false)
    if matches.is_empty():
        printerr("No GaussianSplatContainer found in scene.")
        return null
    if matches.size() > 1:
        printerr("Multiple GaussianSplatContainer nodes found; pass --container=<path>.")
        return null
    return matches[0] as GaussianSplatContainer

func _ensure_container_assets(container: GaussianSplatContainer) -> void:
    var count := container.get_child_count()
    for i in range(count):
        var child := container.get_child(i)
        var splat_node := child as GaussianSplatNode3D
        if splat_node == null:
            continue
        if splat_node.get_splat_asset() != null:
            continue
        var ply_path := splat_node.get_ply_file_path()
        if ply_path.is_empty():
            continue
        var asset := _load_asset(ply_path)
        if asset != null:
            splat_node.set_splat_asset(asset)

func _load_asset(path: String) -> GaussianSplatAsset:
    var ext := path.get_extension().to_lower()
    if ext == "tres" or ext == "res":
        var res := ResourceLoader.load(path)
        if res == null or not (res is GaussianSplatAsset):
            printerr("Resource is not a GaussianSplatAsset: %s" % path)
            return null
        return res as GaussianSplatAsset

    if ext != "ply" and ext != "spz":
        printerr("Unsupported input format: %s" % path)
        return null

    var asset := GaussianSplatAsset.new()
    var err := asset.load_from_file(path)
    if err != OK:
        printerr("Failed to load splat asset: %s (err=%d)" % [path, err])
        return null
    asset.set_source_path(path)
    return asset

func _apply_chunk_size(container: GaussianSplatContainer, args: Dictionary) -> void:
    if not args.has("chunk_size"):
        return
    var size := float(args["chunk_size"])
    if size > 0.0:
        container.set_chunk_size(size)

func _split_list(value: String) -> PackedStringArray:
    var result: PackedStringArray = []
    var parts := value.split(",", false)
    for part in parts:
        var trimmed := part.strip_edges()
        if trimmed != "":
            result.push_back(trimmed)
    return result

func _parse_args() -> Dictionary:
    var result := {}
    for arg in OS.get_cmdline_user_args():
        if not arg.begins_with("--"):
            continue
        var tokens := arg.substr(2).split("=", false, 2)
        if tokens.size() == 1:
            result[tokens[0]] = ""
        else:
            result[tokens[0]] = tokens[1]
    return result

func _print_usage() -> void:
    print("GaussianSplatWorld bake utility")
    print("Required: --output=<path.gsplatworld>")
    print("Scene mode: --scene=<scene.tscn> [--container=<nodepath>] [--chunk_size=25]")
    print("Input mode: --inputs=<a.ply,b.ply> [--chunk_size=25]")
