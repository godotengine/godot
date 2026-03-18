extends Node

# Minimal painterly smoke test script for the test project.
# Keeps runtime behavior defensive so it can run in reduced environments.

func _ready() -> void:
    var failures: Array[String] = []
    var splat_obj := ClassDB.instantiate("GaussianSplatNode3D")
    if splat_obj == null or not (splat_obj is Node):
        push_error("GaussianSplatNode3D class unavailable; painterly smoke test cannot run.")
        print("PAINTERLY_TEST_FAILED")
        get_tree().quit(1)
        return

    var splat_node: Node = splat_obj
    add_child(splat_node)

    if not splat_node.has_method("set_splat_data"):
        failures.append("set_splat_data method unavailable")
    else:
        var positions := PackedVector3Array([Vector3(0.0, 0.0, -3.0)])
        var colors := PackedColorArray([Color(1.0, 0.95, 0.9, 0.9)])
        var scales := PackedVector3Array([Vector3(0.4, 0.4, 0.4)])
        splat_node.call("set_splat_data", positions, colors, scales)

    if not splat_node.has_method("set_enable_painterly"):
        failures.append("set_enable_painterly method unavailable")
    if not splat_node.has_method("force_update"):
        failures.append("force_update method unavailable")

    if failures.is_empty():
        splat_node.call("set_enable_painterly", true)
        splat_node.call("force_update")
        await get_tree().process_frame
        var visible_after_enable := _read_visible_splats(splat_node)
        if visible_after_enable <= 0:
            failures.append("visible splats with painterly enabled: %d" % visible_after_enable)

        splat_node.call("set_enable_painterly", false)
        splat_node.call("force_update")
        await get_tree().process_frame
        var visible_after_disable := _read_visible_splats(splat_node)
        if visible_after_disable <= 0:
            failures.append("visible splats with painterly disabled: %d" % visible_after_disable)

        splat_node.call("set_enable_painterly", true)
        splat_node.call("force_update")
        await get_tree().process_frame
        var visible_after_reenable := _read_visible_splats(splat_node)
        if visible_after_reenable <= 0:
            failures.append("visible splats after re-enabling painterly: %d" % visible_after_reenable)

        print("[Painterly Test] visible (enabled/disabled/re-enabled): %d / %d / %d" % [
            visible_after_enable,
            visible_after_disable,
            visible_after_reenable
        ])

    if failures.is_empty():
        print("PAINTERLY_TEST_PASSED")
        get_tree().quit(0)
        return

    for failure in failures:
        push_error("[Painterly Test] %s" % failure)
    print("PAINTERLY_TEST_FAILED")
    get_tree().quit(1)

func _read_visible_splats(splat_node: Node) -> int:
    if splat_node.has_method("get_statistics"):
        var stats = splat_node.call("get_statistics")
        if typeof(stats) == TYPE_DICTIONARY:
            return int(stats.get("visible_splats", -1))
    if splat_node.has_method("get_visible_splat_count"):
        return int(splat_node.call("get_visible_splat_count"))
    return -1
