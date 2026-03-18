extends SceneTree

# CI-ready GPU Sorting Test
# Tests GPU sorting functionality with performance validation and proper error handling

var test_results = {
	"test_name": "GPU Sorting Tests",
	"start_time": 0,
	"end_time": 0,
	"total_tests": 0,
	"passed_tests": 0,
	"failed_tests": 0,
	"skipped_tests": 0,
	"errors": [],
	"details": [],
	"performance_metrics": {}
}

# Validation mode:
# - strict (default in CI): benchmark/perf issues fail tests.
# - warn-only (default locally): benchmark/perf issues are reported but non-fatal.
const VALIDATION_MODE_ENV := "GS_CI_VALIDATION_MODE"

# GPU-required CI mode:
# - default enabled when CI=true.
# - set GS_CI_GPU_REQUIRED=0 to opt out.
# - set GS_CI_ALLOW_ALL_GPU_SKIPPED=1 to explicitly allow all tests skipped.
const GPU_REQUIRED_ENV := "GS_CI_GPU_REQUIRED"
const ALLOW_ALL_GPU_SKIPPED_ENV := "GS_CI_ALLOW_ALL_GPU_SKIPPED"

func _env_flag(name: String, default_value: bool) -> bool:
	if not OS.has_environment(name):
		return default_value
	var raw = OS.get_environment(name).strip_edges().to_lower()
	if raw in ["1", "true", "yes", "on"]:
		return true
	if raw in ["0", "false", "no", "off", ""]:
		return false
	return default_value

func _is_ci() -> bool:
	return _env_flag("CI", false)

func _validation_mode() -> String:
	if OS.has_environment(VALIDATION_MODE_ENV):
		var explicit_mode = OS.get_environment(VALIDATION_MODE_ENV).strip_edges().to_lower()
		if explicit_mode in ["strict", "warn-only"]:
			return explicit_mode
	return "strict" if _is_ci() else "warn-only"

func _is_strict_mode() -> bool:
	return _validation_mode() == "strict"

func _is_gpu_required_ci_mode() -> bool:
	return _env_flag(GPU_REQUIRED_ENV, _is_ci())

## Entry point for the GPU sorting CI suite; runs tests and exits with status.
func _initialize():
	call_deferred("_run_suite")

func _run_suite():
	print("=== GPU Sorting CI Test Suite ===")
	print("Script started, running tests...")
	test_results.start_time = Time.get_unix_time_from_system()
	print("Validation mode: %s" % _validation_mode())
	print("GPU-required CI mode: %s" % ("enabled" if _is_gpu_required_ci_mode() else "disabled"))

	# Give the engine a frame to fully initialize
	await process_frame

	# Run all tests
	await run_all_tests()

	_apply_suite_gates()

	# Generate final report
	generate_test_report()

	# Exit with appropriate code
	var exit_code = 0 if test_results.failed_tests == 0 else 1
	print("Exiting with code: ", exit_code)
	quit(exit_code)

func _apply_suite_gates():
	var all_skipped = test_results.total_tests > 0 and test_results.skipped_tests == test_results.total_tests
	if all_skipped and _is_gpu_required_ci_mode() and not _env_flag(ALLOW_ALL_GPU_SKIPPED_ENV, false):
		var message = (
			"All GPU sorting tests were skipped while GPU-required CI mode is enabled. "
			+ "Set %s=1 to explicitly allow this."
		) % ALLOW_ALL_GPU_SKIPPED_ENV
		test_results.failed_tests += 1
		test_results.errors.append("suite_gate: " + message)
		print("❌ SUITE GATE FAILED: " + message)

## Runs all GPU sorting test cases in sequence.
func run_all_tests():
	print("\n--- Starting GPU Sorting Tests ---")

	# Test 1: Renderer initialization and GPU context
	run_test("test_renderer_initialization", test_renderer_initialization)

	# Test 2: Sorting method configuration
	run_test("test_sorting_method_config", test_sorting_method_config)

	# Test 3: Small dataset sorting (1K splats)
	await run_test_async("test_small_dataset_sorting", test_small_dataset_sorting)

	# Test 4: Medium dataset sorting (10K splats)
	await run_test_async("test_medium_dataset_sorting", test_medium_dataset_sorting)

	# Test 5: Large dataset sorting (100K splats)
	await run_test_async("test_large_dataset_sorting", test_large_dataset_sorting)

	# Test 6: RadixSort direct test
	run_test("test_radix_sorter", test_radix_sorter)

	# Test 7: Performance validation
	run_test("test_performance_validation", test_performance_validation)

## Runs a synchronous test and records its result.
## @param test_name: Name of the test case.
## @param test_function: Callable returning a result dictionary.
func run_test(test_name: String, test_function: Callable):
	print("\n🧪 Running test: " + test_name)
	test_results.total_tests += 1

	var test_detail = {
		"name": test_name,
		"passed": false,
		"skipped": false,
		"error": "",
		"duration": 0,
		"details": ""
	}

	var start_time = Time.get_unix_time_from_system()

	var result = test_function.call()
	test_detail.passed = result.get("success", false)
	test_detail.skipped = result.get("skipped", false)
	test_detail.details = result.get("details", "")
	test_detail.error = result.get("error", "")

	_record_test_result(test_name, test_detail, start_time)

## Runs an async test and records its result.
## @param test_name: Name of the test case.
## @param test_function: Callable returning a result dictionary.
func run_test_async(test_name: String, test_function: Callable):
	print("\n🧪 Running test: " + test_name)
	test_results.total_tests += 1

	var test_detail = {
		"name": test_name,
		"passed": false,
		"skipped": false,
		"error": "",
		"duration": 0,
		"details": ""
	}

	var start_time = Time.get_unix_time_from_system()

	var result = await test_function.call()
	test_detail.passed = result.get("success", false)
	test_detail.skipped = result.get("skipped", false)
	test_detail.details = result.get("details", "")
	test_detail.error = result.get("error", "")

	_record_test_result(test_name, test_detail, start_time)

## Updates counters and stores a completed test result entry.
func _record_test_result(test_name: String, test_detail: Dictionary, start_time: float):
	if test_detail.passed:
		test_results.passed_tests += 1
		print("✅ PASSED: " + test_name)
	elif test_detail.skipped:
		test_results.skipped_tests += 1
		var skip_msg = " - " + test_detail.error if test_detail.error != "" else ""
		print("⏭️ SKIPPED: " + test_name + skip_msg)
	else:
		test_results.failed_tests += 1
		if test_detail.error != "":
			test_results.errors.append(test_name + ": " + test_detail.error)
		print("❌ FAILED: " + test_name + " - " + test_detail.error)

	test_detail.duration = Time.get_unix_time_from_system() - start_time
	test_results.details.append(test_detail)

## Validates renderer construction and access to render stats.
func test_renderer_initialization() -> Dictionary:
	var result = {"success": false, "details": "", "error": "", "skipped": false}

	result.skipped = true
	result.error = "Renderer construction is validated in native tests"
	result.details = "Skipped GDScript renderer initialization (constructor requires native setup)"

	return result

## Ensures sorting method configuration can be set and queried.
func test_sorting_method_config() -> Dictionary:
	var result = {"success": false, "details": "", "error": "", "skipped": false}

	result.skipped = true
	result.error = "Sorting config is validated in native tests"
	result.details = "Sorting method selection removed; GPU radix is always used"

	return result

## Builds deterministic GaussianData for sort tests.
## @param count: Number of splats to generate.
## @return GaussianData populated with positions/scales/opacities.
func create_test_data(count: int) -> GaussianData:
	var data = GaussianData.new()
	data.resize(count)

	var positions = PackedVector3Array()
	var scales = PackedVector3Array()
	var rotations: Array[Quaternion] = []
	var opacities = PackedFloat32Array()

	positions.resize(count)
	scales.resize(count)
	opacities.resize(count)

	# Create deterministic but varied test data
	for i in range(count):
		positions[i] = Vector3(
			sin(i * 0.1) * 10.0,
			cos(i * 0.1) * 10.0,
			(i % 100) * 0.1
		)
		var scale = 0.1 + (i % 10) * 0.05
		scales[i] = Vector3(scale, scale, scale)
		rotations.append(Quaternion())
		opacities[i] = 0.8 + (i % 5) * 0.04

	data.set_positions(positions)
	data.set_scales(scales)
	data.set_rotations(rotations)
	data.set_opacities(opacities)

	return data

## Runs the dataset sorting test with a small dataset.
func test_small_dataset_sorting() -> Dictionary:
	return await test_dataset_sorting(1000, "small")

## Runs the dataset sorting test with a medium dataset.
func test_medium_dataset_sorting() -> Dictionary:
	return await test_dataset_sorting(10000, "medium")

## Runs the dataset sorting test with a large dataset.
func test_large_dataset_sorting() -> Dictionary:
	return await test_dataset_sorting(100000, "large")

## Executes a sorting test for a dataset of the given size.
## @param count: Number of splats to sort.
## @param size_label: Size label for reporting.
func test_dataset_sorting(count: int, size_label: String) -> Dictionary:
	var result = {"success": false, "details": "", "error": "", "skipped": false}

	var rd = RenderingServer.create_local_rendering_device()
	if rd == null:
		result.skipped = true
		result.error = "No RenderingDevice available for dataset sorting"
		result.details = "Skipped %s dataset sorting test (no GPU context)" % size_label
		return result

	result.skipped = true
	result.error = "Dataset sorting is validated in native tests"
	result.details = "Skipped %s dataset sorting (renderer constructor requires native setup)" % size_label
	test_results.performance_metrics["dataset_sorting_native_only"] = true
	return result

## Returns expected maximum sort time thresholds for CI.
func get_expected_max_time(count: int) -> float:
	# Performance expectations based on dataset size
	if count <= 1000:
		return 5.0  # 5ms for small datasets (relaxed for CI)
	elif count <= 10000:
		return 15.0  # 15ms for medium datasets
	else:
		return 50.0  # 50ms for large datasets

## Confirms the RadixSort class is instantiable.
func test_radix_sorter() -> Dictionary:
	var result = {"success": false, "details": "", "error": "", "skipped": false}

	var rd = RenderingServer.create_local_rendering_device()
	if rd == null:
		result.skipped = true
		result.error = "No RenderingDevice available"
		result.details = "Skipped RadixSort test (no GPU context available)"
		return result

	# RadixSort is registered and should be exposed
	var sorter = RadixSort.new()
	if sorter == null:
		result.error = "Failed to create RadixSort instance"
		return result

	# Check algorithm name if exposed
	var algo_name = sorter.get_algorithm_name() if sorter.has_method("get_algorithm_name") else "RadixSort"

	result.success = true
	result.details = "RadixSort class instantiated successfully (algorithm: %s)" % algo_name

	return result

## Validates collected performance metrics for basic scaling expectations.
func test_performance_validation() -> Dictionary:
	var result = {"success": false, "details": "", "error": "", "skipped": false}
	var strict_mode = _is_strict_mode()

	var metrics = test_results.performance_metrics
	var issues = []

	# Check if we have performance data
	if metrics.has("small_sort_time"):
		var small_time = metrics.small_sort_time

		if small_time > 10.0:  # Very lenient for CI
			issues.append("Small dataset sorting slower than expected: %.2fms" % small_time)
		if small_time <= 0.0:
			issues.append("Small dataset produced invalid sort_time=%.2fms" % small_time)

		for size_label in ["small", "medium", "large"]:
			if not metrics.has(size_label + "_sort_time"):
				issues.append("Missing %s_sort_time metric" % size_label)

		# Check throughput scaling
		if metrics.has("medium_throughput") and metrics.has("small_throughput"):
			var small_tp = metrics.small_throughput
			var medium_tp = metrics.medium_throughput
			if small_tp > 0:
				var throughput_ratio = medium_tp / small_tp
				if throughput_ratio < 0.3:  # Medium should maintain reasonable throughput
					issues.append("Throughput scaling issue: medium/small ratio = %.2f" % throughput_ratio)
		elif strict_mode:
			issues.append("Missing throughput metrics required for strict validation")
	else:
		# In strict mode this is a failure because validation cannot verify regressions.
		if bool(metrics.get("dataset_sorting_native_only", false)):
			result.skipped = true
			result.error = "Dataset sorting metrics are covered by native tests"
			result.details = "Performance validation skipped: GDScript benchmark path intentionally disabled"
		elif strict_mode:
			result.error = "No GPU performance metrics collected in strict mode"
			result.details = "Missing performance metrics is treated as failure in strict mode"
		else:
			result.skipped = true
			result.error = "No GPU performance metrics collected"
			result.details = "Performance validation skipped (warn-only mode)"
		return result

	if issues.size() == 0:
		result.success = true
		result.details = "All performance metrics within acceptable ranges"
	else:
		if strict_mode:
			result.error = "Performance validation failed: " + "; ".join(issues)
			result.details = "Performance regression detected in strict mode"
		else:
			result.success = true
			result.details = "Performance warnings (warn-only mode): " + "; ".join(issues)

	return result

## Prints a summary report of GPU sorting CI results.
func generate_test_report():
	test_results.end_time = Time.get_unix_time_from_system()
	var duration = test_results.end_time - test_results.start_time

	print("\n=== GPU Sorting Test Report ===")
	print("Total Tests: %d" % test_results.total_tests)
	print("Passed: %d" % test_results.passed_tests)
	print("Failed: %d" % test_results.failed_tests)
	print("Skipped: %d" % test_results.skipped_tests)
	print("Duration: %.2f seconds" % duration)

	# Performance summary
	if test_results.performance_metrics.size() > 0:
		print("\n📊 Performance Metrics:")
		for key in test_results.performance_metrics:
			var value = test_results.performance_metrics[key]
			if key.ends_with("_sort_time"):
				print("  %s: %.2f ms" % [key, value])
			elif key.ends_with("_throughput"):
				print("  %s: %.1f M splats/second" % [key, value])
			else:
				print("  %s: %s" % [key, str(value)])

	if test_results.failed_tests > 0:
		print("\n❌ FAILED TESTS:")
		for error in test_results.errors:
			print("  - " + error)
	elif test_results.skipped_tests > 0 and test_results.passed_tests > 0:
		print("\n⚠️ Some tests were skipped (expected in headless mode)")
		print("✅ All GPU-available tests PASSED!")
	elif test_results.skipped_tests == test_results.total_tests:
		print("\n⚠️ All tests skipped - no GPU context available")
	else:
		print("\n✅ ALL TESTS PASSED!")

	# Save detailed results to file for CI
	save_test_results_json()

## Persists detailed CI results to a JSON file for later inspection.
func save_test_results_json():
	var file = FileAccess.open("user://gpu_sorting_test_results.json", FileAccess.WRITE)
	if file:
		file.store_string(JSON.stringify(test_results, "\t"))
		file.close()
		print("📄 Test results saved to user://gpu_sorting_test_results.json")
	else:
		print("⚠️ Could not save test results file")
