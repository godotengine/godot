extends Node3D

# Automatic threshold optimization for Issue #126
# Finds optimal algorithm selection thresholds to achieve <2ms target

# Optimization parameters
const TARGET_TIME_MS = 2.0
const THRESHOLD_SEARCH_RANGES = {
	"bitonic_max": [8192, 32768],      # Test bitonic threshold range
	"radix_max": [1000000, 4000000]    # Test radix threshold range
}
const OPTIMIZATION_SIZES = [10000, 50000, 100000, 500000, 1000000, 2000000]
const RUNS_PER_TEST = 10

var renderer: GaussianSplatRenderer
var optimization_results = {}
var optimal_thresholds = {}

## Runs the GPU sorting threshold optimization workflow end-to-end.
func _ready():
	print("\n=== GPU Sorting Threshold Optimization for Issue #126 ===")
	print("Goal: Find optimal algorithm thresholds for <2ms sorting")
	print("Method: Grid search across threshold combinations\n")

	renderer = GaussianSplatRenderer.new()
	renderer.initialize()

	await run_threshold_optimization()
	analyze_optimization_results()
	apply_optimal_thresholds()

## Sweeps threshold combinations and records performance scores per dataset size.
func run_threshold_optimization():
	print("Starting threshold optimization...")

	# Test bitonic thresholds
	var bitonic_candidates = range(
		THRESHOLD_SEARCH_RANGES.bitonic_max[0],
		THRESHOLD_SEARCH_RANGES.bitonic_max[1] + 1,
		4096  # Step size
	)

	# Test radix thresholds
	var radix_candidates = range(
		THRESHOLD_SEARCH_RANGES.radix_max[0],
		THRESHOLD_SEARCH_RANGES.radix_max[1] + 1,
		250000  # Step size
	)

	print("Testing %d bitonic × %d radix threshold combinations..." % [
		bitonic_candidates.size(), radix_candidates.size()
	])

	var test_count = 0
	var total_tests = bitonic_candidates.size() * radix_candidates.size()

	for bitonic_threshold in bitonic_candidates:
		for radix_threshold in radix_candidates:
			if radix_threshold <= bitonic_threshold:
				continue  # Invalid configuration

			test_count += 1
			print("[Optimization] Test %d/%d: bitonic=%d, radix=%d" % [
				test_count, total_tests, bitonic_threshold, radix_threshold
			])

			# Test this threshold combination
			var performance_scores = await test_threshold_combination(
				bitonic_threshold, radix_threshold
			)

			# Store results
			var key = "%d_%d" % [bitonic_threshold, radix_threshold]
			optimization_results[key] = {
				"bitonic_threshold": bitonic_threshold,
				"radix_threshold": radix_threshold,
				"performance_scores": performance_scores,
				"overall_score": calculate_overall_score(performance_scores)
			}

			# Progress update
			if test_count % 5 == 0:
				var best_so_far = find_current_best()
				print("[Optimization] Best so far: score=%.3f, thresholds=(%d,%d)" % [
					best_so_far.overall_score,
					best_so_far.bitonic_threshold,
					best_so_far.radix_threshold
				])

## Tests a specific threshold combination across all dataset sizes.
## @param bitonic_threshold: Max size for bitonic sorting.
## @param radix_threshold: Max size for radix sorting.
## @return Performance score dictionary keyed by dataset size.
func test_threshold_combination(bitonic_threshold: int, radix_threshold: int) -> Dictionary:
	var scores = {}

	# Temporarily set thresholds (would normally use project settings)
	# For testing, we'll simulate by directly testing each algorithm

	for size in OPTIMIZATION_SIZES:
		var algorithm = determine_algorithm(size, bitonic_threshold, radix_threshold)
		var times = []

		# Set the specific algorithm
		match algorithm:
			"BITONIC":
				renderer.set_sorting_method(GaussianSplatRenderer.SORT_BITONIC)
			"RADIX":
				renderer.set_sorting_method(GaussianSplatRenderer.SORT_RADIX)
			"ONESWEEP":
				renderer.set_sorting_method(GaussianSplatRenderer.SORT_ONESWEEP)

		# Create test data
		var data = create_test_data(size)
		renderer.set_gaussian_data(data)

		# Warmup
		for i in 2:
			await get_tree().process_frame
			renderer.get_sort_time_ms()

		# Performance test
		for run in RUNS_PER_TEST:
			await get_tree().process_frame
			var sort_time = renderer.get_sort_time_ms()
			times.append(sort_time)

		# Calculate score for this size
		var avg_time = times.reduce(func(a, b): return a + b) / times.size()
		var meets_target = avg_time <= TARGET_TIME_MS
		var score = calculate_size_score(avg_time, size)

		scores[size] = {
			"algorithm": algorithm,
			"avg_time": avg_time,
			"meets_target": meets_target,
			"score": score,
			"raw_times": times
		}

	return scores

## Determines which algorithm would be selected for the given thresholds.
## @param size: Dataset size.
## @param bitonic_threshold: Max size for bitonic sorting.
## @param radix_threshold: Max size for radix sorting.
## @return Algorithm name.
func determine_algorithm(size: int, bitonic_threshold: int, radix_threshold: int) -> String:
	if size <= bitonic_threshold:
		return "BITONIC"
	elif size <= radix_threshold:
		return "RADIX"
	else:
		return "ONESWEEP"

## Calculates a weighted performance score for a dataset size.
## @param time_ms: Average sort time in milliseconds.
## @param size: Dataset size.
## @return Score between 0 and 1.
func calculate_size_score(time_ms: float, size: int) -> float:
	# Base score: higher is better
	var time_score = max(0.0, (TARGET_TIME_MS - time_ms) / TARGET_TIME_MS)

	# Throughput bonus
	var throughput_msps = size / time_ms / 1000.0
	var throughput_score = min(1.0, throughput_msps / 1000.0)  # Normalize to ~1000 M splats/s

	# Combined score with time being most important
	return time_score * 0.8 + throughput_score * 0.2

## Aggregates per-size scores into a weighted overall score.
## @param performance_scores: Per-size performance results.
## @return Weighted overall score.
func calculate_overall_score(performance_scores: Dictionary) -> float:
	var total_score = 0.0
	var weight_sum = 0.0

	for size in performance_scores.keys():
		var score = performance_scores[size].score
		var weight = sqrt(size)  # Larger sizes get more weight
		total_score += score * weight
		weight_sum += weight

	return total_score / weight_sum if weight_sum > 0 else 0.0

## Finds the best-performing threshold combination collected so far.
## @return Dictionary describing the best result.
func find_current_best() -> Dictionary:
	var best_result = null
	var best_score = -1.0

	for key in optimization_results.keys():
		var result = optimization_results[key]
		if result.overall_score > best_score:
			best_score = result.overall_score
			best_result = result

	return best_result

## Summarizes the best thresholds and prints a performance breakdown.
func analyze_optimization_results():
	print("\n=== Optimization Analysis ===")

	var best_result = find_current_best()
	if best_result == null:
		print("No valid results found!")
		return

	optimal_thresholds = {
		"bitonic_threshold": best_result.bitonic_threshold,
		"radix_threshold": best_result.radix_threshold
	}

	print("Optimal Thresholds Found:")
	print("  Bitonic threshold: %d elements" % optimal_thresholds.bitonic_threshold)
	print("  Radix threshold: %d elements" % optimal_thresholds.radix_threshold)
	print("  Overall score: %.3f" % best_result.overall_score)

	print("\nPerformance breakdown:")
	for size in OPTIMIZATION_SIZES:
		if size in best_result.performance_scores:
			var perf = best_result.performance_scores[size]
			var status = "✓" if perf.meets_target else "✗"
			print("  %s %d elements: %s, %.2f ms (score: %.3f)" % [
				status, size, perf.algorithm, perf.avg_time, perf.score
			])

	# Compare with default thresholds
	print("\nComparison with defaults:")
	print("  Default bitonic: 16384 → Optimal: %d (%+d)" % [
		optimal_thresholds.bitonic_threshold,
		optimal_thresholds.bitonic_threshold - 16384
	])
	print("  Default radix: 2000000 → Optimal: %d (%+d)" % [
		optimal_thresholds.radix_threshold,
		optimal_thresholds.radix_threshold - 2000000
	])

## Writes the optimal threshold configuration to a JSON file for reuse.
func apply_optimal_thresholds():
	print("\n=== Applying Optimal Configuration ===")

	# Save optimal configuration to project settings
	var config_file = FileAccess.open("res://optimal_gpu_sorting_config.json", FileAccess.WRITE)
	if config_file:
		var json = JSON.new()
		var config = {
			"issue": "126",
			"optimization_timestamp": Time.get_datetime_string_from_system(),
			"optimal_thresholds": optimal_thresholds,
			"performance_validation": {},
			"godot_project_settings": {
				"rendering/gaussian_splatting/gpu_sorting/bitonic_threshold": optimal_thresholds.bitonic_threshold,
				"rendering/gaussian_splatting/gpu_sorting/radix_threshold": optimal_thresholds.radix_threshold,
				"rendering/gaussian_splatting/gpu_sorting/target_sort_time_ms": TARGET_TIME_MS,
				"rendering/gaussian_splatting/gpu_sorting/enable_adaptive_thresholds": true,
				"rendering/gaussian_splatting/gpu_sorting/enable_performance_logging": true
			}
		}

		# Add validation data
		var best_result = find_current_best()
		config.performance_validation = best_result.performance_scores

		config_file.store_string(json.stringify(config, "\t"))
		config_file.close()

		print("✓ Optimal configuration saved to optimal_gpu_sorting_config.json")
		print("✓ Apply these settings to your project.godot file:")
		print()

		for setting_path in config.godot_project_settings.keys():
			var value = config.godot_project_settings[setting_path]
			print("  %s=%s" % [setting_path, str(value)])

	else:
		print("✗ Failed to save optimal configuration")

	print("\n=== Optimization Complete ===")
	print("Thresholds optimized for <2ms sorting performance target")

## Creates deterministic Gaussian data for threshold optimization runs.
## @param size: Number of splats to generate.
## @return GaussianData populated with positions/colors/scales.
func create_test_data(size: int) -> GaussianData:
	var data = GaussianData.new()

	var positions = PackedVector3Array()
	var colors = PackedColorArray()
	var scales = PackedVector3Array()

	var rng = RandomNumberGenerator.new()
	rng.seed = 12345  # Consistent seed

	for i in range(size):
		positions.append(Vector3(
			rng.randf_range(-20, 20),
			rng.randf_range(-20, 20),
			rng.randf_range(-100, -1)
		))
		colors.append(Color(rng.randf(), rng.randf(), rng.randf(), 0.8))

		var scale = rng.randf_range(0.1, 1.0)
		scales.append(Vector3(scale, scale, scale))

	data.set_positions(positions)
	data.set_colors(colors)
	data.set_scales(scales)

	return data
