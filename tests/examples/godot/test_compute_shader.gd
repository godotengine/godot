extends Node3D

# Test script for compute shader infrastructure
## Runs a GPU sort smoke test and prints renderer statistics.
func _ready():
    print("[Test] Testing Compute Shader Infrastructure...")

    # Create a GaussianSplatRenderer node
    var renderer = GaussianSplatRenderer.new()
    renderer.name = "TestRenderer"
    add_child(renderer)

    print("[Test] GaussianSplatRenderer created and added to scene")

    # Wait a frame for initialization
    await get_tree().process_frame

    # Test GPU sort functionality
    print("[Test] Testing GPU sort...")
    renderer.test_gpu_sort()

    print("[Test] Test complete!")

    # Show some stats
    var stats = renderer.get_render_stats()
    print("[Test] Render stats: ", stats)
    print("[Test] Sort time: ", renderer.get_sort_time_ms(), " ms")

    # Clean up
    await get_tree().create_timer(2.0).timeout
    print("[Test] Cleaning up...")
    queue_free()
