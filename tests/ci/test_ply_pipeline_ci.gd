extends SceneTree

# CI-ready PLY Pipeline Test
# Tests complete PLY data pipeline with asset loading and renderer integration

var test_results = {
	"test_name": "PLY Pipeline Tests",
	"start_time": 0,
	"end_time": 0,
	"total_tests": 0,
	"passed_tests": 0,
	"failed_tests": 0,
	"skipped_tests": 0,
	"errors": [],
	"details": []
}

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

func _is_gpu_required_ci_mode() -> bool:
	return _env_flag(GPU_REQUIRED_ENV, _is_ci())

## Entry point for the PLY pipeline CI suite; runs tests and exits with status.
func _initialize():
	print("=== PLY Pipeline CI Test Suite ===")
	test_results.start_time = Time.get_unix_time_from_system()
	print("GPU-required CI mode: %s" % ("enabled" if _is_gpu_required_ci_mode() else "disabled"))

	# Run all tests
	run_all_tests()
	_apply_suite_gates()

	# Generate final report
	generate_test_report()

	# Exit with appropriate code
	var exit_code = 0 if test_results.failed_tests == 0 else 1
	quit(exit_code)

func _apply_suite_gates():
	var all_skipped = test_results.total_tests > 0 and test_results.skipped_tests == test_results.total_tests
	if all_skipped and _is_gpu_required_ci_mode() and not _env_flag(ALLOW_ALL_GPU_SKIPPED_ENV, false):
		var message = (
			"All PLY pipeline tests were skipped while GPU-required CI mode is enabled. "
			+ "Set %s=1 to explicitly allow this."
		) % ALLOW_ALL_GPU_SKIPPED_ENV
		test_results.failed_tests += 1
		test_results.errors.append("suite_gate: " + message)
		print("❌ SUITE GATE FAILED: " + message)

## Runs the full set of PLY pipeline integration tests.
func run_all_tests():
	print("\n--- Starting PLY Pipeline Tests ---")

	# Test 1: GaussianSplatAsset PLY loading
	run_test("test_gaussian_splat_asset_loading", test_gaussian_splat_asset_loading)

	# Test 2: PLYLoader integration
	run_test("test_ply_loader_integration", test_ply_loader_integration)

	# Test 3: Renderer pipeline integration
	run_test("test_renderer_integration", test_renderer_integration)

	# Test 4: Data consistency validation
	run_test("test_data_consistency", test_data_consistency)

## Executes a test case and records pass/fail details.
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
	if typeof(result) != TYPE_DICTIONARY:
		test_detail.passed = false
		test_detail.error = "Test did not return a Dictionary result"
	else:
		test_detail.passed = result.get("success", false)
		test_detail.skipped = result.get("skipped", false)
		test_detail.details = result.get("details", "")
		test_detail.error = result.get("error", "")

	if test_detail.passed:
		test_results.passed_tests += 1
		print("✅ PASSED: " + test_name)
	elif test_detail.skipped:
		test_results.skipped_tests += 1
		var skip_msg = " - " + test_detail.error if test_detail.error != "" else ""
		print("⏭️ SKIPPED: " + test_name + skip_msg)
	else:
		test_results.failed_tests += 1
		test_results.errors.append(test_name + ": " + test_detail.error)
		print("❌ FAILED: " + test_name + " - " + test_detail.error)

	test_detail.duration = Time.get_unix_time_from_system() - start_time
	test_results.details.append(test_detail)

## Validates loading a PLY file into GaussianSplatAsset.
func test_gaussian_splat_asset_loading() -> Dictionary:
	var result = {"success": false, "details": "", "error": "", "skipped": false}

	# Create GaussianSplatAsset
	var asset = GaussianSplatAsset.new()
	if not asset:
		result.error = "Failed to create GaussianSplatAsset instance"
		return result

	# Test loading PLY file
	var ply_path = "res://tests/fixtures/test_splats.ply"
	var load_result = asset.load_from_file(ply_path)

	if load_result == OK:
		var splat_count = asset.get_splat_count()
		var positions = asset.get_positions()
		var colors = asset.get_colors()
		var scales = asset.get_scales()
		var rotations = asset.get_rotations()

		result.details = "Loaded %d splats from %s" % [splat_count, ply_path]

		# Validate array sizes
		var errors = []
		if positions.size() != splat_count * 3:
			errors.append("Position array size mismatch: expected %d, got %d" % [splat_count * 3, positions.size()])

		if colors.size() != splat_count:
			errors.append("Color array size mismatch: expected %d, got %d" % [splat_count, colors.size()])

		if scales.size() != splat_count * 3:
			errors.append("Scale array size mismatch: expected %d, got %d" % [splat_count * 3, scales.size()])

		if rotations.size() != splat_count * 4:
			errors.append("Rotation array size mismatch: expected %d, got %d" % [splat_count * 4, rotations.size()])

		if errors.size() > 0:
			result.error = "Data validation errors: " + "; ".join(errors)
		else:
			result.success = true
			result.details += ". All array sizes valid."

	elif load_result == ERR_FILE_NOT_FOUND:
		# Missing fixture should not be treated as a pass. Skip the coverage claim
		# and surface the reason so CI output stays truthful.
		result.skipped = true
		result.error = "Fixture missing: %s" % ply_path
		result.details = "Skipped fixture-backed PLY load validation because the fixture file was not present."
	else:
		result.error = "Failed to load PLY file: Error code " + str(load_result)

	return result

## Ensures the PLYLoader populates GaussianData correctly.
func test_ply_loader_integration() -> Dictionary:
	var result = {"success": false, "details": "", "error": "", "skipped": false}

	# Create PLYLoader
	var ply_loader = PLYLoader.new()
	if not ply_loader:
		result.error = "Failed to create PLYLoader instance"
		return result

	# First, create test data to ensure we have something to load
	if not create_test_ply_file():
		result.error = "Failed to create test PLY file for integration test"
		return result

	# Test loading with PLYLoader
	var load_result = ply_loader.load_file("user://test_pipeline_integration.ply")

	if load_result == OK:
		var gaussian_data = ply_loader.get_gaussian_data()

		if gaussian_data and gaussian_data.get_count() > 0:
			var count = gaussian_data.get_count()
			result.success = true
			result.details = "PLYLoader successfully loaded %d splats into GaussianData" % count

			# Validate data integrity
			var loaded_aabb = gaussian_data.get_aabb()
			if loaded_aabb.size.length() <= 0.0:
				result.success = false
				result.error = "Invalid GaussianData AABB after PLYLoader: " + str(loaded_aabb)
		else:
			result.error = "PLYLoader returned invalid or empty GaussianData"
	else:
		result.error = "PLYLoader failed to load test file: Error code " + str(load_result)

	return result

## Exercises the end-to-end pipeline through the renderer.
func test_renderer_integration() -> Dictionary:
	var result = {"success": false, "details": "", "error": "", "skipped": false}

	# GaussianSplatRenderer currently has no script-safe constructor that accepts a RenderingDevice.
	# Renderer integration is covered by C++ doctests and runtime harnesses.
	result.skipped = true
	result.error = "Renderer integration is validated in native tests"
	result.details = "Skipped GDScript renderer integration (constructor requires native setup)"
	return result

## Checks round-trip data consistency across asset and data loaders.
func test_data_consistency() -> Dictionary:
	var result = {"success": false, "details": "", "error": "", "skipped": false}

	# Create test data with known values
	var test_count = 5
	var original_positions = []

	# Generate deterministic test data
	for i in range(test_count):
		original_positions.append(Vector3(i * 2.0, i * 1.5, i * 1.0))

	# Save through GaussianData
	var gaussian_data = GaussianData.new()
	gaussian_data.resize(test_count)

	var positions = PackedVector3Array()
	for i in range(test_count):
		positions.append(original_positions[i])

	gaussian_data.set_positions(positions)

	var save_result = gaussian_data.save_to_file("user://test_consistency.ply")
	if save_result != OK:
		result.error = "Failed to save test data: Error code " + str(save_result)
		return result

	# Load back through asset
	var asset = GaussianSplatAsset.new()
	var load_result = asset.load_from_file("user://test_consistency.ply")
	if load_result != OK:
		result.error = "Failed to load back test data through asset: Error code " + str(load_result)
		return result

	# Validate consistency
	var loaded_count = asset.get_splat_count()
	if loaded_count != test_count:
		result.error = "Count mismatch: expected %d, got %d" % [test_count, loaded_count]
		return result

	var loaded_positions = asset.get_positions()
	# Check position consistency
	for i in range(test_count):
		var expected_pos = original_positions[i]
		var loaded_pos = Vector3(
			loaded_positions[i * 3],
			loaded_positions[i * 3 + 1],
			loaded_positions[i * 3 + 2]
		)

		if expected_pos.distance_to(loaded_pos) > 0.001:
			result.error = "Position %d inconsistency: expected %s, got %s" % [i, expected_pos, loaded_pos]
			return result

	result.success = true
	result.details = "Data consistency verified through complete pipeline: GaussianData -> PLY file -> GaussianSplatAsset"
	return result

## Builds and saves a temporary PLY file for integration tests.
## @return True when the file is saved successfully.
func create_test_ply_file() -> bool:
	var gaussian_data = GaussianData.new()
	gaussian_data.resize(20)

	var positions = PackedVector3Array()
	var scales = PackedVector3Array()
	var rotations: Array[Quaternion] = []
	var opacities = PackedFloat32Array()

	for i in range(20):
		positions.append(Vector3(i * 0.5, i * 0.3, i * 0.1))
		scales.append(Vector3(0.5, 0.5, 0.5))
		rotations.append(Quaternion())
		opacities.append(0.8)

	gaussian_data.set_positions(positions)
	gaussian_data.set_scales(scales)
	gaussian_data.set_rotations(rotations)
	gaussian_data.set_opacities(opacities)

	var save_result = gaussian_data.save_to_file("user://test_pipeline_integration.ply")
	return save_result == OK

## Prints the PLY pipeline test summary and persists results.
func generate_test_report():
	test_results.end_time = Time.get_unix_time_from_system()
	var duration = test_results.end_time - test_results.start_time

	print("\n=== PLY Pipeline Test Report ===")
	print("Total Tests: %d" % test_results.total_tests)
	print("Passed: %d" % test_results.passed_tests)
	print("Failed: %d" % test_results.failed_tests)
	print("Skipped: %d" % test_results.skipped_tests)
	print("Duration: %.2f seconds" % duration)

	if test_results.failed_tests > 0:
		print("\n❌ FAILED TESTS:")
		for error in test_results.errors:
			print("  - " + error)
	elif test_results.skipped_tests > 0 and test_results.passed_tests > 0:
		print("\n⚠️ Some tests were skipped")
		print("✅ All executed tests passed")
	elif test_results.skipped_tests == test_results.total_tests:
		print("\n⚠️ All tests skipped")
	else:
		print("\n✅ ALL TESTS PASSED!")

	# Save detailed results to file for CI
	save_test_results_json()

## Writes detailed PLY pipeline CI results to a JSON file.
func save_test_results_json():
	var file = FileAccess.open("user://ply_pipeline_test_results.json", FileAccess.WRITE)
	if file:
		file.store_string(JSON.stringify(test_results))
		file.close()
		print("📄 Test results saved to user://ply_pipeline_test_results.json")
	else:
		print("⚠️ Could not save test results file")
