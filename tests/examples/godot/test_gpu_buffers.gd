extends Node

# Test script for GPU buffer creation in Gaussian Splatting module

## Builds a renderer and uploads synthetic data to validate GPU buffer setup.
func _ready():
    print("Testing GPU Buffer Manager...")

    # Create a GaussianSplatRenderer node
    var renderer = GaussianSplatRenderer.new()
    renderer.name = "TestRenderer"
    add_child(renderer)

    # Create test Gaussian data
    var gaussian_data = GaussianData.new()
    gaussian_data.resize(1000)  # Create 1000 test gaussians

    # Set some test positions
    var positions = PackedVector3Array()
    for i in range(1000):
        positions.append(Vector3(
            randf_range(-10, 10),
            randf_range(-10, 10),
            randf_range(-10, 10)
        ))
    gaussian_data.set_positions(positions)

    # Set the data to the renderer (this should trigger GPU buffer creation)
    renderer.gaussian_data = gaussian_data

    # Wait a frame for initialization
    await get_tree().process_frame

    # Test GPU sort (which will also test buffer status)
    renderer.test_gpu_sort()

    # Get render stats
    var stats = renderer.get_render_stats()
    print("Render Stats:", stats)

    print("GPU Buffer test complete!")
