## Batch export utility for Gaussian splat datasets.
## Usage:
##   godot --headless --script scripts/export_gaussian_scene.gd -- --input=input.ply --output=scene.gsf
extends SceneTree

## Parses command-line arguments and exports Gaussian data to a .gsf file.
func _init():
    var args := _parse_args()
    if not args.has("input") or not args.has("output"):
        printerr("Usage: --input=<ply|gsf> --output=<gsf>")
        get_tree().quit(1)
        return

    var serializer := GaussianSceneSerializer.new()
    var gaussian_data := GaussianData.new()
    var animation := GaussianAnimationStateMachine.new()
    var err := OK

    if args["input"].to_lower().ends_with(".ply"):
        var loader := PLYLoader.new()
        err = loader.load_file(args["input"])
        if err != OK:
            printerr("Failed to load PLY file: %s (%d)" % [args["input"], err])
            get_tree().quit(2)
            return
        gaussian_data = loader.get_gaussian_data()
    elif args["input"].to_lower().ends_with(".gsf"):
        err = serializer.load_scene(args["input"], gaussian_data, animation)
        if err != OK:
            printerr("Failed to load GSF file: %s (%d)" % [args["input"], err])
            get_tree().quit(3)
            return
    else:
        printerr("Unsupported input format: %s" % args["input"])
        get_tree().quit(4)
        return

    err = serializer.save_scene(args["output"], gaussian_data, animation)
    if err != OK:
        printerr("Failed to save scene: %s (%d)" % [args["output"], err])
        get_tree().quit(5)
        return

    print("Exported Gaussian scene to %s" % args["output"])
    get_tree().quit()

## Parses --key=value arguments from the command line.
## @return Dictionary of parsed arguments.
func _parse_args() -> Dictionary:
    var result := {}
    for arg in OS.get_cmdline_args():
        if arg.begins_with("--"):
            var tokens := arg.substr(2).split("=", false, 2)
            if tokens.size() == 2:
                result[tokens[0]] = tokens[1]
    return result
