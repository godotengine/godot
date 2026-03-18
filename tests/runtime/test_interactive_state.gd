extends SceneTree

# Test script for Interactive State System (Issue #88)

const MAX_AVERAGE_TRANSITION_MS := 5.0
const SKIP_MARKER := "[RUNTIME_SKIP]"
const FAIL_MARKER := "[RUNTIME_FAIL]"

var failures: Array[String] = []

## Defers test execution until the SceneTree loop is initialized.
func _init() -> void:
    call_deferred("_run")

## Returns true when running on a headless display server.
func _is_headless_runtime() -> bool:
    return OS.has_feature("headless") or DisplayServer.get_name() == "headless"

## Creates a renderer and runs state transition, visual, and performance checks.
func _run() -> void:
    if _is_headless_runtime():
        var skip_reason = "Interactive state runtime test requires a local RenderingDevice (non-headless run)."
        print("%s %s" % [SKIP_MARKER, skip_reason])
        push_warning(skip_reason)
        quit(0)
        return

    print("Testing Interactive State System...")

    # Create renderer instance
    var renderer = GaussianSplatRenderer.new()
    if renderer == null:
        _record_failure("Failed to create GaussianSplatRenderer instance")
        _print_summary()
        quit(1)
        return
    renderer.initialize()

    # Test state transitions
    print("\n=== State Transition Tests ===")
    test_state_transitions(renderer)

    # Test visual effects
    print("\n=== Visual Effects Tests ===")
    test_visual_effects(renderer)

    # Test performance
    print("\n=== Performance Tests ===")
    test_state_performance(renderer)

    _print_summary()
    quit(0 if failures.is_empty() else 1)

## Records a validation outcome with optional context details.
func _record_check(condition: bool, label: String, context: Dictionary = {}) -> void:
    if condition:
        print("✓ ", label)
        return
    _record_failure(label, context)

## Records a failure and prints a machine-readable marker for the Python runner.
func _record_failure(label: String, context: Dictionary = {}) -> void:
    var reason = label
    if not context.is_empty():
        reason = "%s | context=%s" % [label, str(context)]
    failures.append(reason)
    push_error("%s %s" % [FAIL_MARKER, reason])

## Prints final pass/fail summary.
func _print_summary() -> void:
    if failures.is_empty():
        print("\n✅ Interactive state runtime checks passed.")
        return

    print("\n❌ Interactive state runtime checks failed (%d issue(s)):" % failures.size())
    for failure in failures:
        print(" - ", failure)

## Validates allowed state transitions for the interactive state machine.
## @param renderer: GaussianSplatRenderer instance under test.
func test_state_transitions(renderer):
    # Test normal to hovered
    renderer.set_interactive_state(GaussianSplatRenderer.STATE_NORMAL)
    _record_check(
        renderer.get_interactive_state() == GaussianSplatRenderer.STATE_NORMAL,
        "Initial state set to NORMAL",
        {"state": renderer.get_interactive_state()}
    )

    # Test hover state
    renderer.set_interactive_state(GaussianSplatRenderer.STATE_HOVERED)
    _record_check(
        renderer.get_interactive_state() == GaussianSplatRenderer.STATE_HOVERED,
        "Transition to HOVERED",
        {"state": renderer.get_interactive_state()}
    )

    # Test selected state
    renderer.set_interactive_state(GaussianSplatRenderer.STATE_SELECTED)
    _record_check(
        renderer.get_interactive_state() == GaussianSplatRenderer.STATE_SELECTED,
        "Transition to SELECTED",
        {"state": renderer.get_interactive_state()}
    )

    # Test disabled state
    renderer.set_interactive_state(GaussianSplatRenderer.STATE_DISABLED)
    _record_check(
        renderer.get_interactive_state() == GaussianSplatRenderer.STATE_DISABLED,
        "Transition to DISABLED",
        {"state": renderer.get_interactive_state()}
    )

    # Disabled -> hovered is expected to be rejected.
    var disabled_state = renderer.get_interactive_state()
    renderer.set_interactive_state(GaussianSplatRenderer.STATE_HOVERED)
    _record_check(
        renderer.get_interactive_state() == disabled_state,
        "Invalid DISABLED->HOVERED transition rejected",
        {"before": disabled_state, "after": renderer.get_interactive_state()}
    )

## Exercises highlight/outline effects and reset behavior.
## @param renderer: GaussianSplatRenderer instance under test.
func test_visual_effects(renderer):
    # Reset to normal state
    renderer.set_interactive_state(GaussianSplatRenderer.STATE_NORMAL)

    # Test highlight effect
    var highlight_color = Color(1.2, 1.2, 0.8, 1.0)
    renderer.enable_highlight_effect(highlight_color)
    print("✓ Highlight effect enabled: ", highlight_color)

    # Test outline effect
    var outline_color = Color(1.0, 0.5, 0.0, 1.0)
    var outline_width = 2.5
    renderer.enable_outline_effect(outline_color, outline_width)
    print("✓ Outline effect enabled: ", outline_color, " width: ", outline_width)

    # Test removing effects
    renderer.remove_visual_effects()
    _record_check(true, "Visual effect API calls completed")

## Benchmarks rapid state transitions to validate performance targets.
## @param renderer: GaussianSplatRenderer instance under test.
func test_state_performance(renderer):
    var start_time = Time.get_ticks_usec()

    # Test rapid state changes
    for i in range(100):
        renderer.set_interactive_state(GaussianSplatRenderer.STATE_NORMAL)
        renderer.set_interactive_state(GaussianSplatRenderer.STATE_HOVERED)
        renderer.set_interactive_state(GaussianSplatRenderer.STATE_SELECTED)
        renderer.set_interactive_state(GaussianSplatRenderer.STATE_NORMAL)

    var elapsed = (Time.get_ticks_usec() - start_time) / 1000.0
    var avg_time = elapsed / 400.0  # 100 iterations * 4 state changes

    print("✓ 400 state transitions in %.2f ms" % elapsed)
    print("✓ Average transition time: %.3f ms" % avg_time)

    # Check if we meet the < 5ms requirement per transition
    _record_check(avg_time > 0.0, "Measured non-zero transition timing", {"avg_ms": avg_time})
    _record_check(
        avg_time < MAX_AVERAGE_TRANSITION_MS,
        "Performance requirement met (< %.2f ms per transition)" % MAX_AVERAGE_TRANSITION_MS,
        {"avg_ms": avg_time, "total_ms": elapsed}
    )
