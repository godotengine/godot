extends SceneTree

# Importer zero-initialization regression test (catches commit 546d3d6d25's bug).
#
# Background: GaussianSplatAsset::_ensure_buffer_sizes() unconditionally resizes
# several optional PackedArrays (opacity_logits, palette_ids, painterly_flags,
# normals, brush_axes, stroke_ages, sh_first/high_order_coefficients) to
# splat_count slots. The PLY importer does not always populate them; when it
# does not, those slots must be exactly zero so the runtime opacity-aware
# tile-binning pass treats them as "no value." The bug was that Vector<T>::resize()
# for trivially-constructible types (POD floats, ints) skips zero-init, leaving
# whatever bytes the allocator returned. On Windows dev builds that pattern is
# 0xC0C0C0C0, which as float is -6.023529. Those bytes were serialized into the
# .tres cache and later loaded as plausible-but-wrong opacity logits, causing
# the binning shader to reject every splat.
#
# This test forces a fresh import path and asserts every optional Packed*Array
# slot is exactly 0 / 0.0. If a future change re-introduces .resize() without
# resize_initialized() OR without an explicit fill loop, this test fails.

const FIXTURE_PATH := "res://tests/fixtures/test_splats.ply"

# 0xC0C0C0C0 as float32. If the bug regresses, optional fields will contain
# exactly this value (or another non-zero pattern). Either way, we just check
# for non-zero.
const POISON_FLOAT := -6.023529

var test_results = {
	"test_name": "Importer Zero-Init Tests",
	"start_time": 0,
	"end_time": 0,
	"total_tests": 0,
	"passed_tests": 0,
	"failed_tests": 0,
	"errors": [],
	"details": []
}


func _initialize() -> void:
	print("=== Importer Zero-Init CI Test ===")
	test_results.start_time = Time.get_unix_time_from_system()

	run_test("test_optional_arrays_are_zero", test_optional_arrays_are_zero)
	run_test("test_no_poison_pattern_in_serialized_cache", test_no_poison_pattern_in_serialized_cache)
	run_test("test_loaded_asset_renders_nonzero_opacity", test_loaded_asset_renders_nonzero_opacity)

	generate_test_report()
	var exit_code := 0 if test_results.failed_tests == 0 else 1
	quit(exit_code)


func run_test(test_name: String, test_func: Callable) -> void:
	test_results.total_tests += 1
	print("\n--- Running %s ---" % test_name)
	var ok := false
	var err_msg := ""
	var result = test_func.call()
	if result is Dictionary:
		ok = result.get("ok", false)
		err_msg = result.get("error", "")
	else:
		ok = bool(result)
	if ok:
		test_results.passed_tests += 1
		test_results.details.append({"name": test_name, "status": "PASS"})
		print("    PASS")
	else:
		test_results.failed_tests += 1
		test_results.errors.append("%s: %s" % [test_name, err_msg])
		test_results.details.append({"name": test_name, "status": "FAIL", "error": err_msg})
		print("    FAIL: %s" % err_msg)


# ---- TESTS ----

func test_optional_arrays_are_zero() -> Dictionary:
	# Load the imported .tres (via the PLY's import handle).
	var asset := load(FIXTURE_PATH) as GaussianSplatAsset
	if asset == null:
		return {"ok": false, "error": "load(%s) returned null" % FIXTURE_PATH}

	var checks := [
		["opacity_logits", asset.get_opacity_logits()],
		["palette_ids", asset.get_palette_ids()],
		["painterly_flags", asset.get_painterly_flags()],
		["normals", asset.get_normals()],
		["brush_axes", asset.get_brush_axes()],
		["stroke_ages", asset.get_stroke_ages()],
		["sh_first_order_coefficients", asset.get_sh_first_order_coefficients()],
		["sh_high_order_coefficients", asset.get_sh_high_order_coefficients()],
	]
	for entry in checks:
		var name: String = entry[0]
		var arr = entry[1]
		if arr == null:
			continue
		var n: int = arr.size()
		for i in range(n):
			var v = arr[i]
			# All of these are PackedFloat32Array or PackedInt32Array; for
			# floats compare exact zero (not approximate — the bug pattern
			# is bit-for-bit reproducible 0xC0C0C0C0). For ints exact 0.
			if typeof(v) == TYPE_FLOAT:
				if v != 0.0:
					return {
						"ok": false,
						"error": "%s[%d] = %f (expected 0.0; uninit-pattern serialized into .tres)" % [name, i, v]
					}
			elif typeof(v) == TYPE_INT:
				if v != 0:
					return {
						"ok": false,
						"error": "%s[%d] = %d (expected 0; uninit-pattern serialized into .tres)" % [name, i, v]
					}
	return {"ok": true}


func test_no_poison_pattern_in_serialized_cache() -> Dictionary:
	# Sanity guard against the specific 0xC0C0C0C0 -> -6.023529 pattern.
	# A future regression that produces a DIFFERENT non-zero garbage pattern
	# is also caught by test_optional_arrays_are_zero above; this one
	# specifically pinpoints the original bug signature.
	var asset := load(FIXTURE_PATH) as GaussianSplatAsset
	if asset == null:
		return {"ok": false, "error": "load failed"}
	var arr := asset.get_opacity_logits()
	for i in range(arr.size()):
		if absf(arr[i] - POISON_FLOAT) < 0.001:
			return {
				"ok": false,
				"error": "opacity_logits[%d] = %f matches the 0xC0C0C0C0 poison signature" % [i, arr[i]]
			}
	return {"ok": true}


func test_loaded_asset_renders_nonzero_opacity() -> Dictionary:
	# End-to-end: a valid asset should report sensible per-splat opacities
	# (they live in get_colors().a). If the bug regresses, opacity_logits
	# may still be junk while colors stay valid — but if BOTH paths use the
	# bad data, this catches it.
	var asset := load(FIXTURE_PATH) as GaussianSplatAsset
	if asset == null:
		return {"ok": false, "error": "load failed"}
	var colors := asset.get_colors()
	if colors.size() == 0:
		return {"ok": false, "error": "colors array empty after load"}
	var any_visible := false
	for c in colors:
		if c.a > 0.01:
			any_visible = true
			break
	if not any_visible:
		return {"ok": false, "error": "no visible splat (every color.a < 0.01)"}
	return {"ok": true}


# ---- REPORT ----

func generate_test_report() -> void:
	test_results.end_time = Time.get_unix_time_from_system()
	var elapsed: float = test_results.end_time - test_results.start_time
	print("\n=== Importer Zero-Init Test Report ===")
	print("Total: %d  Passed: %d  Failed: %d  Time: %.2fs" % [
		test_results.total_tests,
		test_results.passed_tests,
		test_results.failed_tests,
		elapsed,
	])
	if test_results.failed_tests > 0:
		print("\nErrors:")
		for e in test_results.errors:
			print("  - %s" % e)
	# Optional: write JSON artifact next to the existing qa_results.json
	var report_path := "user://importer_zero_init_results.json"
	var f := FileAccess.open(report_path, FileAccess.WRITE)
	if f != null:
		f.store_string(JSON.stringify(test_results, "\t"))
		f.close()
		print("Report: %s" % report_path)
