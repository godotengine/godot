extends Node

const TEST_SIZES := [1000, 5000, 10000, 50000, 100000, 500000, 1000000, 2000000]
const WARMUP_RUNS := 3
const BENCHMARK_RUNS := 10
const TARGET_SORT_TIME_MS := 2.0
const TARGET_IMPROVEMENT := 3.0
const ALGORITHMS := ["RADIX"]

var results := {}
var renderer: GaussianSplatRenderer
var exit_code := 0

## Initializes the renderer and runs GPU sorting benchmarks for each algorithm.
func _ready() -> void:
    print("\n[GPU SORT BENCHMARK] Starting Gaussian Splat sorting benchmarks (Phase 2)")
    print("[GPU SORT BENCHMARK] Using renderer-specified GPU sorting configuration")
    print("[GPU SORT BENCHMARK] Test sizes: %s" % str(TEST_SIZES))

    renderer = GaussianSplatRenderer.new()
    renderer.initialize()
    print("[GPU SORT BENCHMARK] Renderer initialized and ready")

    for algorithm_name in ALGORITHMS:
        await benchmark_algorithm(algorithm_name)

    validate_speedups()
    generate_performance_report()
    save_results_to_file()

    print("\n[GPU SORT BENCHMARK] ===== BENCHMARK COMPLETE =====")
    if exit_code != 0:
        push_error("[GPU SORT BENCHMARK] ❌ One or more checks failed; review logs")
    get_tree().quit(exit_code)

## Benchmarks the configured sorting method across all test sizes.
## @param algorithm_name: Label for the algorithm under test.
func benchmark_algorithm(algorithm_name: String) -> void:
    print("\n[GPU SORT BENCHMARK] Testing algorithm: %s" % algorithm_name)
    results[algorithm_name] = {}

    for size in TEST_SIZES:
        print("[GPU SORT BENCHMARK] Testing %s with %d elements..." % [algorithm_name, size])

        var dataset := create_test_data(size)
        renderer.set_gaussian_data(dataset["data"])
        var positions: PackedVector3Array = dataset["positions"]

        for i in range(WARMUP_RUNS):
            await get_tree().process_frame()
            renderer.force_sort_for_view(Transform3D.IDENTITY)
            renderer.get_sort_time_ms()

        var times: Array = []
        for i in range(BENCHMARK_RUNS):
            await get_tree().process_frame()
            renderer.force_sort_for_view(Transform3D.IDENTITY)
            var sort_time := renderer.get_sort_time_ms()
            var stats := renderer.get_render_stats()

            var algorithm_label := String(stats.get("gpu_sorter_algorithm", ""))
            if algorithm_label.is_empty() or algorithm_label.to_lower().find("radix") == -1:
                push_error("[GPU SORT BENCHMARK] ❌ GPU sorter algorithm missing or incorrect at size %d" % size)
                exit_code = 1

            var preview: PackedInt32Array = stats.get("sorted_indices_preview", PackedInt32Array())
            if preview.size() > 1:
                var prev_dist := -INF
                for idx in preview:
                    if idx < 0 or idx >= positions.size():
                        continue
                    var dist := positions[idx].distance_squared_to(Vector3.ZERO)
                    if dist < prev_dist - 0.0001:
                        push_error("[GPU SORT BENCHMARK] ❌ Preview ordering broke monotonicity at size %d" % size)
                        exit_code = 1
                        break
                    prev_dist = dist

            if stats.get("sort_submission_time_ms", 0.0) <= 0.0 and stats.get("sort_wait_time_ms", 0.0) <= 0.0:
                push_error("[GPU SORT BENCHMARK] ❌ GPU timing metrics missing at size %d" % size)
                exit_code = 1

            times.append(sort_time)

        if times.is_empty():
            continue

        var sum_time := 0.0
        var min_time := INF
        var max_time := -INF
        for time in times:
            sum_time += time
            min_time = min(min_time, time)
            max_time = max(max_time, time)
        var avg_time := sum_time / times.size()

        var variance := 0.0
        for time in times:
            variance += (time - avg_time) * (time - avg_time)
        variance /= times.size()
        var std_dev := sqrt(variance)

        var throughput := size / max(avg_time, 0.0001) / 1000.0

        results[algorithm_name][size] = {
            "avg_time_ms": avg_time,
            "min_time_ms": min_time,
            "max_time_ms": max_time,
            "std_dev_ms": std_dev,
            "throughput_msps": throughput,
            "meets_target": avg_time <= TARGET_SORT_TIME_MS,
            "raw_times": times,
        }

        var status := "✓" if avg_time <= TARGET_SORT_TIME_MS else "⚠"
        print("[GPU SORT BENCHMARK]   %s %.2f±%.2f ms (%.1f M splats/s)" % [
            status,
            avg_time,
            std_dev,
            throughput,
        ])

## Validates speedup expectations for the largest dataset.
func validate_speedups() -> void:
    var largest_size := TEST_SIZES.back()
    if "RADIX" in results and largest_size in results["RADIX"]:
        var radix_time := results["RADIX"][largest_size]["avg_time_ms"]
        print("[GPU SORT BENCHMARK] Radix average at %d elements: %.2fms" % [largest_size, radix_time])

## Generates deterministic Gaussian test data for benchmarking.
## @param size: Number of splats to generate.
## @return Dictionary with GaussianData and positions.
func create_test_data(size: int) -> Dictionary:
    var data := GaussianData.new()

    var positions := PackedVector3Array()
    var colors := PackedColorArray()
    var scales := PackedVector3Array()

    var rng := RandomNumberGenerator.new()
    rng.seed = 12345

    for i in range(size):
        var cluster_center := Vector3(
            rng.randf_range(-20, 20),
            rng.randf_range(-20, 20),
            rng.randf_range(-50, 50)
        )
        positions.append(cluster_center + Vector3(
            rng.randf_range(-2, 2),
            rng.randf_range(-2, 2),
            rng.randf_range(-2, 2)
        ))

        colors.append(Color(rng.randf(), rng.randf(), rng.randf(), 0.8))

        var scale := rng.randf_range(0.1, 1.0)
        scales.append(Vector3(scale, scale, scale))

    data.set_positions(positions)
    data.set_colors(colors)
    data.set_scales(scales)

    return {
        "data": data,
        "positions": positions,
    }

## Prints a summary table of benchmark results to stdout.
func generate_performance_report() -> void:
    print("\n[GPU SORT BENCHMARK] ========== PERFORMANCE REPORT ==========")
    print("[GPU SORT BENCHMARK] Algorithm Performance Summary:")
    print("[GPU SORT BENCHMARK] Size\t\tRadix")
    print("[GPU SORT BENCHMARK] " + "-" * 24)

    for size in TEST_SIZES:
        var line := "[GPU SORT BENCHMARK] %d\t" % size
        for algorithm in ALGORITHMS:
            if algorithm in results and size in results[algorithm]:
                var avg_time := results[algorithm][size]["avg_time_ms"]
                var status := "✓" if avg_time <= TARGET_SORT_TIME_MS else "⚠"
                line += "\t%s%.1f" % [status, avg_time]
            else:
                line += "\t---"
        print(line)

## Writes benchmark results to a JSON file in the user directory.
func save_results_to_file() -> void:
    var file := FileAccess.open("user://gpu_sort_benchmark.json", FileAccess.WRITE)
    if file:
        file.store_string(JSON.stringify(results, "  "))
        file.close()
        print("[GPU SORT BENCHMARK] Results saved to user://gpu_sort_benchmark.json")
