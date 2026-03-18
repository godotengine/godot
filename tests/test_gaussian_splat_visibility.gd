extends Node

## Defers the regression runner to ensure the scene tree is initialized before asserting
## visibility; no side effects beyond scheduling the test.
func _ready():
    call_deferred("_run")

## Verifies a GaussianSplatNode3D reports visible splats once populated, exiting with
## code 0 on success and 1 when the regression reproduces.
func _run() -> void:
    print("[GaussianSplatNode3D] Regression check starting")
    var splat_node := GaussianSplatNode3D.new()
    add_child(splat_node)

    await get_tree().process_frame

    var asset := GaussianSplatAsset.new()
    asset.set_positions(PackedFloat32Array([0.0, 0.0, 0.0]))
    asset.set_colors(PackedColorArray([Color(1.0, 0.8, 0.2, 1.0)]))
    asset.set_scales(PackedFloat32Array([0.5, 0.5, 0.5]))
    asset.set_rotations(PackedFloat32Array([1.0, 0.0, 0.0, 0.0]))

    splat_node.set_splat_asset(asset)
    splat_node.force_update()

    await get_tree().process_frame

    var baseline_visible := splat_node.get_visible_splat_count()
    if baseline_visible <= 0:
        push_error("\u2717 FAILED: Painterly-on baseline reported " + str(baseline_visible) + " splats")
        get_tree().quit(1)
        return

    splat_node.set_enable_painterly(false)
    splat_node.force_update()
    await get_tree().process_frame
    var painterly_off_visible := splat_node.get_visible_splat_count()
    if painterly_off_visible <= 0:
        push_error("\u2717 FAILED: Painterly-off fallback reported " + str(painterly_off_visible) + " splats")
        get_tree().quit(1)
        return

    splat_node.set_enable_painterly(true)
    splat_node.force_update()
    await get_tree().process_frame
    var painterly_on_visible := splat_node.get_visible_splat_count()
    if painterly_on_visible <= 0:
        push_error("\u2717 FAILED: Re-enabling painterly reported " + str(painterly_on_visible) + " splats")
        get_tree().quit(1)
        return

    print("\u2713 PASSED: Visible splats -> baseline:", baseline_visible,
        ", painterly off:", painterly_off_visible,
        ", painterly on:", painterly_on_visible)
    get_tree().quit(0)
