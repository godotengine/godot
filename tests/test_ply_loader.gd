extends Node

const TEST_PLY_PATH := "res://test_data/fixtures/basic_splat.ply"

## Defers the ResourceLoader regression check until the scene is ready.
func _ready() -> void:
    call_deferred("_run")

## Validates ResourceLoader returns a GaussianSplatAsset with expected metadata.
func _run() -> void:
    print("[GaussianSplatAsset] ResourceLoader regression starting")

    if not ResourceLoader.exists(TEST_PLY_PATH):
        push_error("Test asset does not exist: %s" % TEST_PLY_PATH)
        get_tree().quit(1)
        return

    var resource := ResourceLoader.load(TEST_PLY_PATH, "GaussianSplatAsset")
    if resource == null:
        push_error("ResourceLoader returned null for %s" % TEST_PLY_PATH)
        get_tree().quit(1)
        return

    if not (resource is GaussianSplatAsset):
        push_error("Loaded resource is not a GaussianSplatAsset (got %s)" % resource)
        get_tree().quit(1)
        return

    var asset: GaussianSplatAsset = resource
    var splat_count := asset.get_splat_count()
    if splat_count <= 0:
        push_error("GaussianSplatAsset contains no splats")
        get_tree().quit(1)
        return

    var metadata := asset.get_import_metadata()
    var required_keys := ["source_path", "resource_path", "resource_loader", "loaded_via_resource_loader"]
    for key in required_keys:
        if not metadata.has(key) or metadata[key] == "":
            push_error("Missing metadata key '%s' on GaussianSplatAsset" % key)
            get_tree().quit(1)
            return

    if metadata["source_path"] != TEST_PLY_PATH:
        push_error("Unexpected source_path metadata: %s" % metadata["source_path"])
        get_tree().quit(1)
        return

    var has_positions := asset.get_positions().size() == splat_count * 3
    var has_colors := asset.get_colors().size() == splat_count
    if not (has_positions and has_colors):
        push_error("GaussianSplatAsset buffers are incomplete (positions: %s, colors: %s)" % [has_positions, has_colors])
        get_tree().quit(1)
        return

    print("\u2713 PASSED: ResourceLoader produced GaussianSplatAsset with", splat_count, "splats")
    get_tree().quit(0)
