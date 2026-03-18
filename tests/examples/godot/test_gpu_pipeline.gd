extends Node

## Executes a smoke test across the GPU pipeline components.
func _ready():
    print("Testing Gaussian Splatting GPU Pipeline...")

    # Test PLY Loader
    var ply_loader = PLYLoader.new()
    print("✓ PLY Loader created")

    # Test GPU Buffer Manager
    var gpu_buffer = GPUBufferManager.new()
    print("✓ GPU Buffer Manager created")

    # Test Gaussian Data
    var gaussian_data = GaussianData.new()
    gaussian_data.resize(1000)

    # Fill with test data
    for i in range(1000):
        gaussian_data.positions[i] = Vector3(randf() * 10 - 5, randf() * 10 - 5, randf() * 10 - 5)
        gaussian_data.colors[i] = Color(randf(), randf(), randf(), 1.0)
        gaussian_data.scales[i] = Vector3(0.1, 0.1, 0.1)
        gaussian_data.rotations[i] = Quaternion.IDENTITY
        gaussian_data.opacities[i] = 0.9

    gaussian_data.splat_count = 1000
    print("✓ Created test data with 1000 Gaussians")

    # Get rendering device
    var rd = RenderingServer.create_local_rendering_device()
    if rd:
        print("✓ Rendering device obtained")

        # Initialize GPU buffers
        var err = gpu_buffer.initialize(rd, 10000)
        if err == OK:
            print("✓ GPU buffers initialized (%.2f MB)" % gpu_buffer.get_memory_usage_mb())

            # Upload data
            err = gpu_buffer.upload_gaussian_data(gaussian_data)
            if err == OK:
                print("✓ Data uploaded to GPU")
                print("  - Gaussian count: %d" % gpu_buffer.get_gaussian_count())
            else:
                print("✗ Failed to upload data: %d" % err)
        else:
            print("✗ Failed to initialize GPU buffers: %d" % err)
    else:
        print("✗ Could not get rendering device")

    print("\nGPU Pipeline Test Complete!")
    get_tree().quit()
