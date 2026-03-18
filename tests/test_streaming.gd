extends SceneTree

## Validates the GPU streaming system by simulating a frame of updates over a large splat
## dataset; only prints metrics as a side effect.
func test_streaming_system() -> int:
    print("Testing GPU Memory Streaming System...")

    # Create test data
    var gaussian_data = GaussianData.new()
    gaussian_data.resize(1000000)  # 1M splats for testing

    # Initialize streaming system
    var streaming = GaussianStreamingSystem.new()
    streaming.initialize(gaussian_data)

    # Attach GPU memory stream to validate scheduler plumbing even without hardware
    var memory_stream = GaussianMemoryStream.new()
    streaming.attach_memory_stream(memory_stream)

    var task_state = memory_stream.get_task_debug_state()
    if not task_state.has("upload"):
        printerr("FAIL: task_state missing 'upload'")
        return 1
    if not task_state.has("residency"):
        printerr("FAIL: task_state missing 'residency'")
        return 1
    if not task_state.has("eviction"):
        printerr("FAIL: task_state missing 'eviction'")
        return 1

    # Test frame management
    for i in range(6):
        streaming.begin_frame()
        streaming.end_frame()

    # Stress triple-buffering bookkeeping by forcing rapid swaps
    for i in range(9):
        streaming.begin_frame()
        streaming.update_streaming(Transform3D(Basis(), Vector3(0, 0, -10)), Projection())
        streaming.end_frame()

    var analytics = streaming.get_streaming_analytics()
    if not analytics.has("loaded_chunks"):
        printerr("FAIL: analytics missing 'loaded_chunks'")
        return 1
    if not analytics.has("visible_splats"):
        printerr("FAIL: analytics missing 'visible_splats'")
        return 1

    # Check metrics
    print("Visible splats: ", streaming.get_visible_count())
    print("VRAM usage: ", streaming.get_vram_usage() / 1048576.0, " MB")
    print("Loaded chunks: ", streaming.get_loaded_chunks())
    print("Task analytics: ", analytics)
    print("Streaming test PASSED!")
    return 0

## Entry point when run as standalone script
func _init():
    var exit_code = test_streaming_system()
    quit(exit_code)

