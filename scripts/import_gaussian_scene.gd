## Batch import utility that applies incremental edits to a baseline scene.
## Usage:
##   godot --headless --script scripts/import_gaussian_scene.gd -- --baseline=scene.gsf --incremental=delta.gsif --output=merged.gsf
extends SceneTree

## Merges a baseline .gsf with optional incremental edits and writes output.
func _init():
    var args := _parse_args()
    if not args.has("baseline") or not args.has("output"):
        printerr("Usage: --baseline=<gsf> [--incremental=<gsif>] --output=<gsf>")
        get_tree().quit(1)
        return

    var serializer := GaussianSceneSerializer.new()
    var data := GaussianData.new()
    var animation := GaussianAnimationStateMachine.new()
    var err := serializer.load_scene(args["baseline"], data, animation)
    if err != OK:
        printerr("Failed to load baseline: %s (%d)" % [args["baseline"], err])
        get_tree().quit(2)
        return

    if args.has("incremental"):
        var saver := GaussianIncrementalSaver.new()
        data.set_incremental_saver(saver)
        animation.set_incremental_saver(saver)
        saver.start_tracking(args["baseline"])
        err = saver.load_and_apply_changes(args["incremental"], data, animation)
        if err != OK:
            printerr("Failed to apply incremental file: %s (%d)" % [args["incremental"], err])
            get_tree().quit(3)
            return
        print("Applied incremental file %s" % args["incremental"])

    err = serializer.save_scene(args["output"], data, animation)
    if err != OK:
        printerr("Failed to save merged scene: %s (%d)" % [args["output"], err])
        get_tree().quit(4)
        return

    print("Wrote merged scene to %s" % args["output"])
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
