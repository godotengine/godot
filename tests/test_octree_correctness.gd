extends Node

## TestOctree drives GaussianData's octree queries against synthetic datasets to
## validate correctness across typical and edge-case configurations.
class TestOctree:
    var passed: int = 0
    var failed: int = 0

    ## Verifies that the expected value matches the actual one while tracking pass/fail
    ## counters; only prints to stdout as a side effect.
    func assert_equal(actual, expected, test_name: String):
        if actual == expected:
            passed += 1
            print("✓ %s" % test_name)
        else:
            failed += 1
            print("✗ %s: expected %s, got %s" % [test_name, str(expected), str(actual)])

    ## Confirms the queried array contains the provided value and records the result,
    ## emitting diagnostic text for visibility.
    func assert_contains(array: Array, value, test_name: String):
        if value in array:
            passed += 1
            print("✓ %s" % test_name)
        else:
            failed += 1
            print("✗ %s: array does not contain %s" % [test_name, str(value)])

    ## Populates the GaussianData container with the supplied arrays so individual tests
    ## can exercise octree queries without redundant setup; mutates the provided data.
    func _populate_gaussians(data: GaussianData, positions: PackedVector3Array, scales: PackedVector3Array, opacities: PackedFloat32Array):
        var count := positions.size()
        assert(scales.size() == count)
        assert(opacities.size() == count)

        data.resize(count)
        data.set_positions(positions)
        data.set_scales(scales)

        var rotations: Array[Quaternion] = []
        rotations.resize(count)
        for i in range(count):
            rotations[i] = Quaternion()
        data.set_rotations(rotations)

        data.set_opacities(opacities)

    ## Executes all octree regression scenarios and returns true when every assertion
    ## succeeds; prints a summary but performs no additional side effects.
    func run_tests():
        print("\n=== Octree Correctness Tests ===\n")

        test_empty_tree()
        test_single_gaussian()
        test_overlapping_gaussians()
        test_boundary_cases()
        test_large_scale_gaussians()
        test_query_consistency()

        print("\n=== Test Results ===")
        print("Passed: %d" % passed)
        print("Failed: %d" % failed)

        if failed == 0:
            print("All tests passed!")
        else:
            print("Some tests failed!")

        return failed == 0

    ## Validates that querying an empty octree yields no results, ensuring default state
    ## safety without mutating shared data.
    func test_empty_tree():
        print("\n--- Testing Empty Tree ---")
        var data = GaussianData.new()
        data.build_octree(8)

        var results = data.query_octree(AABB(Vector3.ZERO, Vector3(100, 100, 100)))
        assert_equal(results.size(), 0, "Empty tree returns no results")

    ## Ensures a single Gaussian is correctly indexed and retrievable at boundaries;
    ## the scenario touches both hit and miss queries for coverage.
    func test_single_gaussian():
        print("\n--- Testing Single Gaussian ---")
        var data = GaussianData.new()
        var positions = PackedVector3Array([Vector3(5, 5, 5)])
        var scales = PackedVector3Array([Vector3(1, 1, 1)])
        var opacities = PackedFloat32Array([1.0])
        _populate_gaussians(data, positions, scales, opacities)

        data.build_octree(4)

        # Query that contains the Gaussian
        var results = data.query_octree(AABB(Vector3(0, 0, 0), Vector3(10, 10, 10)))
        assert_equal(results.size(), 1, "Query containing Gaussian returns it")

        # Query that doesn't contain the Gaussian
        results = data.query_octree(AABB(Vector3(20, 20, 20), Vector3(5, 5, 5)))
        assert_equal(results.size(), 0, "Query not containing Gaussian returns empty")

        # Query at edge (considering 3-sigma coverage)
        results = data.query_octree(AABB(Vector3(7.5, 5, 5), Vector3(1, 1, 1)))
        assert_equal(results.size(), 1, "Query at edge still finds Gaussian due to scale")

    ## Confirms overlapping Gaussians remain discoverable when occupying shared space,
    ## validating node subdivision and accumulation behavior.
    func test_overlapping_gaussians():
        print("\n--- Testing Overlapping Gaussians ---")
        var data = GaussianData.new()
        # Three overlapping Gaussians - use batch setters
        var positions = PackedVector3Array([
            Vector3(5, 5, 5),
            Vector3(6, 5, 5),
            Vector3(5, 6, 5)
        ])
        var scales = PackedVector3Array([
            Vector3(2, 2, 2),
            Vector3(2, 2, 2),
            Vector3(2, 2, 2)
        ])
        var opacities = PackedFloat32Array([1.0, 1.0, 1.0])

        _populate_gaussians(data, positions, scales, opacities)

        data.build_octree(6)

        # Query that should find all three
        var results = data.query_octree(AABB(Vector3(0, 0, 0), Vector3(10, 10, 10)))
        assert_equal(results.size(), 3, "Large query finds all Gaussians")

        # Query that should find only first two
        results = data.query_octree(AABB(Vector3(4, 4, 4), Vector3(3, 2, 2)))
        assert_equal(results.size() >= 2, true, "Focused query finds overlapping Gaussians")

    ## Exercises queries along octree boundaries to ensure splats on volume edges stay
    ## discoverable without duplicating references.
    func test_boundary_cases():
        print("\n--- Testing Boundary Cases ---")
        var data = GaussianData.new()
        var positions = PackedVector3Array([
            Vector3(0, 0, 0),
            Vector3(10, 0, 0),
            Vector3(0, 10, 0),
            Vector3(0, 0, 10)
        ])
        var scales = PackedVector3Array([
            Vector3(0.5, 0.5, 0.5),
            Vector3(0.5, 0.5, 0.5),
            Vector3(0.5, 0.5, 0.5),
            Vector3(0.5, 0.5, 0.5)
        ])
        var opacities = PackedFloat32Array([1.0, 1.0, 1.0, 1.0])
        _populate_gaussians(data, positions, scales, opacities)

        data.build_octree(4)

        # Query each corner
        var results = data.query_octree(AABB(Vector3(-1, -1, -1), Vector3(2, 2, 2)))
        assert_equal(results.size(), 1, "Corner query finds corner Gaussian")

        # Query diagonal
        results = data.query_octree(AABB(Vector3(-1, -1, -1), Vector3(12, 12, 12)))
        assert_equal(results.size(), 4, "Diagonal query finds all Gaussians")

    ## Verifies that extremely large splats expand query coverage correctly while small
    ## neighbors continue operating normally.
    func test_large_scale_gaussians():
        print("\n--- Testing Large Scale Gaussians ---")
        var data = GaussianData.new()
        # One small, one large Gaussian
        var positions = PackedVector3Array([
            Vector3(5, 5, 5),
            Vector3(15, 15, 15)
        ])
        var scales = PackedVector3Array([
            Vector3(0.1, 0.1, 0.1),
            Vector3(5, 5, 5)
        ])
        var opacities = PackedFloat32Array([1.0, 1.0])
        _populate_gaussians(data, positions, scales, opacities)

        data.build_octree(6)

        # Query that only touches the large Gaussian's extent
        var results = data.query_octree(AABB(Vector3(25, 15, 15), Vector3(2, 2, 2)))
        assert_equal(results.size(), 1, "Query at edge of large Gaussian finds it")

        # Query between the two
        results = data.query_octree(AABB(Vector3(8, 8, 8), Vector3(4, 4, 4)))
        # Should find the large one due to its extent
        assert_equal(results.size() >= 1, true, "Query between Gaussians may find large one")

    ## Checks repeated queries of a dense grid return consistent counts and omit
    ## duplicates, guarding against nondeterminism.
    func test_query_consistency():
        print("\n--- Testing Query Consistency ---")
        var data = GaussianData.new()

        # Create grid of 125 Gaussians (5x5x5)
        var positions = PackedVector3Array()
        var scales = PackedVector3Array()
        var opacities = PackedFloat32Array()
        for x in range(5):
            for y in range(5):
                for z in range(5):
                    positions.append(Vector3(x * 3.0, y * 3.0, z * 3.0))
                    scales.append(Vector3(0.5, 0.5, 0.5))
                    opacities.append(1.0)

        _populate_gaussians(data, positions, scales, opacities)

        data.build_octree(8)

        # Multiple queries of the same region should give same results
        var query_bounds = AABB(Vector3(2, 2, 2), Vector3(8, 8, 8))
        var results1 = data.query_octree(query_bounds)
        var results2 = data.query_octree(query_bounds)

        assert_equal(results1.size(), results2.size(), "Repeated queries return same count")

        # Verify no duplicates in results
        var unique = {}
        for idx in results1:
            unique[idx] = true
        assert_equal(unique.size(), results1.size(), "No duplicate indices in results")

## Entry point when run from Godot's headless test runner; exits with code 0 on success
## and 1 on failure for CI integration.
func _ready():
    var tester = TestOctree.new()
    var success = tester.run_tests()

    # Exit with appropriate code
    if success:
        get_tree().quit(0)
    else:
        get_tree().quit(1)
