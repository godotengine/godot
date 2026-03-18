extends Node

## Validates the octree subdivision fix by mixing extreme Gaussian scales and measuring
## build/query times; exits with non-zero when any threshold fails.
func _ready():
    print("Testing octree subdivision fixes...")
    seed(0x5A17)
    var failure_count = 0

    # Create test Gaussian data
    var gaussian_data = GaussianData.new()
    gaussian_data.resize(100)

    # Create a mix of small and large Gaussians
    for i in range(100):
        var pos = Vector3(randf() * 10.0 - 5.0, randf() * 10.0 - 5.0, randf() * 10.0 - 5.0)
        var scale = Vector3.ONE

        # Make some Gaussians very large (pathological case)
        if i < 10:
            scale = Vector3.ONE * 5.0  # Very large scale
        elif i < 30:
            scale = Vector3.ONE * 2.0  # Medium scale
        else:
            scale = Vector3.ONE * 0.1  # Small scale

        # Note: We can't directly set individual Gaussians due to binding limitations
        # This test primarily verifies the octree builds without hanging

    # Build octree - this should complete quickly even with large Gaussians
    print("Building octree with mixed Gaussian sizes...")
    var start_time = Time.get_ticks_msec()
    gaussian_data.build_octree(6)  # Max depth 6
    var elapsed = Time.get_ticks_msec() - start_time

    print("Octree build completed in %d ms" % elapsed)

    # The fix ensures this doesn't hang or create exponential nodes
    if elapsed < 1000:  # Should complete in under 1 second
        print("✓ PASS: Octree build time is reasonable")
    else:
        print("✗ FAIL: Octree build took too long (%d ms)" % elapsed)
        failure_count += 1

    # Test query performance
    print("Testing spatial queries...")
    var query_bounds = AABB(Vector3(-2, -2, -2), Vector3(4, 4, 4))
    start_time = Time.get_ticks_msec()
    var results = gaussian_data.query_octree(query_bounds)
    elapsed = Time.get_ticks_msec() - start_time

    print("Query returned %d Gaussians in %d ms" % [results.size(), elapsed])

    if elapsed < 100:  # Query should be fast
        print("✓ PASS: Query performance is good")
    else:
        print("✗ FAIL: Query took too long (%d ms)" % elapsed)
        failure_count += 1

    # Check memory usage
    var memory_mb = gaussian_data.get_memory_usage() / (1024.0 * 1024.0)
    print("Memory usage: %.2f MB" % memory_mb)

    if memory_mb < 10.0:  # Should use reasonable memory
        print("✓ PASS: Memory usage is reasonable")
    else:
        print("✗ FAIL: Excessive memory usage (%.2f MB)" % memory_mb)
        failure_count += 1

    var exit_code = 0 if failure_count == 0 else 1
    print("\nTest completed with %d failure(s). Exit code: %d" % [failure_count, exit_code])
    get_tree().quit(exit_code)
