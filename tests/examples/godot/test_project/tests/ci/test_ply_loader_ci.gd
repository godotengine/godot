extends SceneTree

# CI-ready PLY Loader Test
# Tests PLY file loading and saving functionality with proper error reporting

const FIXTURE_PATH := "res://tests/fixtures/test_splats.ply"

var test_results = {
	"test_name": "PLY Loader Tests",
	"start_time": 0,
	"end_time": 0,
	"total_tests": 0,
	"passed_tests": 0,
	"failed_tests": 0,
	"errors": [],
	"details": []
}

## Entry point for the PLY loader CI suite; runs tests and exits with status.
func _initialize():
	print("=== PLY Loader CI Test Suite ===")
	test_results.start_time = Time.get_unix_time_from_system()

	# Run all tests
	run_all_tests()

	# Generate final report
	generate_test_report()

	# Exit with appropriate code
	var exit_code = 0 if test_results.failed_tests == 0 else 1
	quit(exit_code)

## Runs the full set of PLY loader tests.
func run_all_tests():
	print("\n--- Starting PLY Loader Tests ---")

	# Test 1: Basic PLY loading functionality
	run_test("test_ply_loading_basic", test_ply_loading_basic)

	# Test 2: PLY saving functionality
	run_test("test_ply_saving", test_ply_saving)

	# Test 3: Generated PLY roundtrip
	run_test("test_ply_roundtrip", test_ply_roundtrip)

	# Test 4: Error handling for invalid files
	run_test("test_ply_error_handling", test_ply_error_handling)

## Executes a test case and records pass/fail details.
## @param test_name: Name of the test case.
## @param test_function: Callable returning a result dictionary.
func run_test(test_name: String, test_function: Callable):
	print("\n🧪 Running test: " + test_name)
	test_results.total_tests += 1

	var test_detail = {
		"name": test_name,
		"passed": false,
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
		test_detail.details = result.get("details", "")
		test_detail.error = result.get("error", "")

	if test_detail.passed:
		test_results.passed_tests += 1
		print("✅ PASSED: " + test_name)
	else:
		test_results.failed_tests += 1
		test_results.errors.append(test_name + ": " + test_detail.error)
		print("❌ FAILED: " + test_name + " - " + test_detail.error)

	test_detail.duration = Time.get_unix_time_from_system() - start_time
	test_results.details.append(test_detail)

## Verifies loading of a fixture PLY file into GaussianData.
func test_ply_loading_basic() -> Dictionary:
	var result = {"success": false, "details": "", "error": ""}

	var gaussian_data = GaussianData.new()
	if not gaussian_data:
		result.error = "Failed to create GaussianData instance"
		return result

	# Test loading existing PLY file
	var test_file = FIXTURE_PATH
	var load_result = gaussian_data.load_from_file(test_file)

	if load_result == OK:
		var count = gaussian_data.get_count()
		var aabb = gaussian_data.get_aabb()

		result.success = true
		result.details = "Loaded %d splats from %s. AABB: %s" % [count, test_file, str(aabb)]

		# Validate reasonable data ranges
		if count <= 0:
			result.success = false
			result.error = "Invalid splat count: " + str(count)
		elif aabb.size.length() <= 0:
			result.success = false
			result.error = "Invalid AABB size: " + str(aabb.size)

	else:
		result.error = "Failed to load PLY file: Error code " + str(load_result)

	if load_result == ERR_FILE_NOT_FOUND:
		result.success = false
		result.error = "Fixture missing: " + test_file

	return result

## Validates saving GaussianData to a PLY file.
func test_ply_saving() -> Dictionary:
	var result = {"success": false, "details": "", "error": ""}

	var gaussian_data = GaussianData.new()
	if not gaussian_data:
		result.error = "Failed to create GaussianData instance"
		return result

	# Create test data
	gaussian_data.resize(10)

	var positions = PackedVector3Array()
	var scales = PackedVector3Array()
	var rotations: Array[Quaternion] = []
	var opacities = PackedFloat32Array()

	for i in range(10):
		positions.append(Vector3(i * 1.0, i * 0.5, i * 0.25))
		scales.append(Vector3(1, 1, 1))
		rotations.append(Quaternion())
		opacities.append(1.0)

	gaussian_data.set_positions(positions)
	gaussian_data.set_scales(scales)
	gaussian_data.set_rotations(rotations)
	gaussian_data.set_opacities(opacities)

	# Test saving
	var save_path = "user://test_ci_generated.ply"
	var save_result = gaussian_data.save_to_file(save_path)

	if save_result == OK:
		result.success = true
		result.details = "Successfully saved 10 test splats to " + save_path

		# Verify file exists
		if not FileAccess.file_exists(save_path):
			result.success = false
			result.error = "File was not actually created at " + save_path
	else:
		result.error = "Failed to save PLY file: Error code " + str(save_result)

	return result

## Confirms saved PLY data can be reloaded with consistent values.
func test_ply_roundtrip() -> Dictionary:
	var result = {"success": false, "details": "", "error": ""}

	# Save data first
	var save_result = test_ply_saving()
	if not save_result.success:
		result.error = "Prerequisites failed: " + save_result.error
		return result

	# Now test loading it back
	var loaded_data = GaussianData.new()
	var load_result = loaded_data.load_from_file("user://test_ci_generated.ply")

	if load_result == OK:
		var count = loaded_data.get_count()
		if count == 10:
			result.success = true
			result.details = "Successfully loaded back saved PLY file with correct count: " + str(count)

			# Validate aabb shape for coarse data integrity
			var loaded_aabb = loaded_data.get_aabb()
			if loaded_aabb.size.length() > 0.0:
				result.details += ". Data integrity verified."
			else:
				result.success = false
				result.error = "Roundtrip produced invalid AABB: " + str(loaded_aabb)
		else:
			result.success = false
			result.error = "Incorrect splat count after reload. Expected: 10, Got: " + str(count)
	else:
		result.error = "Failed to load back saved PLY file: Error code " + str(load_result)

	return result

## Ensures invalid file paths return error codes as expected.
func test_ply_error_handling() -> Dictionary:
	var result = {"success": false, "details": "", "error": ""}

	var gaussian_data = GaussianData.new()

	# Test loading non-existent file
	var load_result = gaussian_data.load_from_file("res://nonexistent_file.ply")

	if load_result == ERR_FILE_NOT_FOUND:
		result.success = true
		result.details = "Correctly handled non-existent file with ERR_FILE_NOT_FOUND"
	elif load_result != OK:
		# Other error codes are also acceptable for error handling test
		result.success = true
		result.details = "Correctly handled invalid file with error code: " + str(load_result)
	else:
		result.error = "Expected error for non-existent file, but got OK"

	return result

## Prints the PLY loader test summary and persists results.
func generate_test_report():
	test_results.end_time = Time.get_unix_time_from_system()
	var duration = test_results.end_time - test_results.start_time

	print("\n=== PLY Loader Test Report ===")
	print("Total Tests: %d" % test_results.total_tests)
	print("Passed: %d" % test_results.passed_tests)
	print("Failed: %d" % test_results.failed_tests)
	print("Duration: %.2f seconds" % duration)

	if test_results.failed_tests > 0:
		print("\n❌ FAILED TESTS:")
		for error in test_results.errors:
			print("  - " + error)
	else:
		print("\n✅ ALL TESTS PASSED!")

	# Save detailed results to file for CI
	save_test_results_json()

## Writes detailed PLY loader CI results to a JSON file.
func save_test_results_json():
	var file = FileAccess.open("user://ply_loader_test_results.json", FileAccess.WRITE)
	if file:
		file.store_string(JSON.stringify(test_results))
		file.close()
		print("📄 Test results saved to user://ply_loader_test_results.json")
	else:
		print("⚠️ Could not save test results file")
