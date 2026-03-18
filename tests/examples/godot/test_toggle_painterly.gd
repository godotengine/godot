extends SceneTree

# Regression sample verifying painterly toggle keeps splats visible.

## Defers execution until the SceneTree loop is initialized.
func _init() -> void:
    call_deferred("_run")

## Sets up a splat node, toggles painterly mode, and prints visibility stats.
func _run() -> void:
    print("[Painterly Toggle Test] Creating GaussianSplatNode3D sample...")

    var scene_root := Node3D.new()
    scene_root.name = "PainterlyToggleRoot"
    root.add_child(scene_root)

    var splat_node := GaussianSplatNode3D.new()
    scene_root.add_child(splat_node)

    var positions := PackedVector3Array([Vector3(0.0, 0.0, -3.0)])
    var colors := PackedColorArray([Color(1.0, 1.0, 1.0, 0.9)])
    var scales := PackedVector3Array([Vector3(0.5, 0.5, 0.5)])

    # New API: directly provide gaussian data without creating a GaussianSplatAsset.
    splat_node.set_splat_data(positions, colors, scales)
    splat_node.set_enable_painterly(true)
    splat_node.force_update()

    var stats_enabled := splat_node.get_statistics()
    print("Painterly enabled visible splats: ", stats_enabled.get("visible_splats", -1))

    splat_node.set_enable_painterly(false)
    splat_node.force_update()
    var stats_disabled := splat_node.get_statistics()
    print("Painterly disabled visible splats: ", stats_disabled.get("visible_splats", -1))

    splat_node.set_enable_painterly(true)
    splat_node.force_update()
    var stats_reenabled := splat_node.get_statistics()
    print("Painterly re-enabled visible splats: ", stats_reenabled.get("visible_splats", -1))

    print("[Painterly Toggle Test] Done. All stages should report visible splats >= 1.")
    quit()
