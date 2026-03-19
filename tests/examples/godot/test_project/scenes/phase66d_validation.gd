extends Node3D
## Phase 6.6d Validation: Legacy vs Instance Pipeline A/B Test
## Measures FPS on both paths with same asset for fair comparison.

@export var test_duration_per_path: float = 30.0
@export var warmup_frames: int = 60

var legacy_node: GaussianSplatWorld3D
var instance_node: GaussianSplatNode3D

var current_path: String = ""
var frame_count: int = 0
var fps_samples: Array[float] = []
var test_start_time: float = 0.0
var warmup_complete: bool = false
var results: Dictionary = {}

enum TestPhase { IDLE, WARMUP_LEGACY, TEST_LEGACY, WARMUP_INSTANCE, TEST_INSTANCE, DONE }
var phase: TestPhase = TestPhase.IDLE

func _ready():
	legacy_node = $LegacyPath
	instance_node = $InstancePath

	# Start with both disabled
	legacy_node.visible = false
	instance_node.visible = false

	print("")
	print("=" .repeat(60))
	print("[PHASE 6.6d] Performance Validation: Legacy vs Instance Pipeline")
	print("=" .repeat(60))
	print("Asset: cabin (84,528 splats)")
	print("Test duration per path: %.0fs" % test_duration_per_path)
	print("Warmup frames: %d" % warmup_frames)
	print("")

	# Start testing after a short delay
	await get_tree().create_timer(1.0).timeout
	_start_legacy_test()

func _start_legacy_test():
	print("[LEGACY] Starting warmup...")
	phase = TestPhase.WARMUP_LEGACY
	current_path = "LEGACY"
	frame_count = 0
	fps_samples.clear()

	legacy_node.visible = true
	instance_node.visible = false

func _start_instance_test():
	print("[INSTANCE] Starting warmup...")
	phase = TestPhase.WARMUP_INSTANCE
	current_path = "INSTANCE"
	frame_count = 0
	fps_samples.clear()

	legacy_node.visible = false
	instance_node.visible = true

func _process(delta: float):
	if phase == TestPhase.IDLE or phase == TestPhase.DONE:
		return

	frame_count += 1

	# Warmup phase
	if phase == TestPhase.WARMUP_LEGACY:
		if frame_count >= warmup_frames:
			print("[LEGACY] Warmup complete, starting measurement...")
			phase = TestPhase.TEST_LEGACY
			frame_count = 0
			test_start_time = Time.get_ticks_msec() / 1000.0
		return

	if phase == TestPhase.WARMUP_INSTANCE:
		if frame_count >= warmup_frames:
			print("[INSTANCE] Warmup complete, starting measurement...")
			phase = TestPhase.TEST_INSTANCE
			frame_count = 0
			test_start_time = Time.get_ticks_msec() / 1000.0
		return

	# Measurement phase
	var fps = 1.0 / delta if delta > 0 else 0.0
	fps_samples.append(fps)

	var elapsed = Time.get_ticks_msec() / 1000.0 - test_start_time

	# Progress report every 5 seconds
	if frame_count % (5 * 60) < 2:  # Roughly every 5 seconds
		var avg = _calc_avg(fps_samples)
		print("[%s] t=%.0fs | FPS: %.1f | avg: %.1f | samples: %d" % [
			current_path, elapsed, fps, avg, fps_samples.size()
		])

	# Check if test phase complete
	if elapsed >= test_duration_per_path:
		_finish_current_test()

func _finish_current_test():
	var avg = _calc_avg(fps_samples)
	var min_fps = fps_samples.min() if fps_samples.size() > 0 else 0.0
	var max_fps = fps_samples.max() if fps_samples.size() > 0 else 0.0
	var p1 = _percentile(fps_samples, 1)
	var p99 = _percentile(fps_samples, 99)

	results[current_path] = {
		"avg": avg,
		"min": min_fps,
		"max": max_fps,
		"p1": p1,
		"p99": p99,
		"samples": fps_samples.size()
	}

	print("")
	print("[%s] RESULTS:" % current_path)
	print("  Average FPS: %.1f" % avg)
	print("  Min/Max: %.1f / %.1f" % [min_fps, max_fps])
	print("  P1/P99: %.1f / %.1f" % [p1, p99])
	print("")

	if phase == TestPhase.TEST_LEGACY:
		_start_instance_test()
	elif phase == TestPhase.TEST_INSTANCE:
		_print_final_results()

func _print_final_results():
	phase = TestPhase.DONE

	legacy_node.visible = false
	instance_node.visible = false

	var legacy_avg = results["LEGACY"]["avg"]
	var instance_avg = results["INSTANCE"]["avg"]
	var diff_pct = ((instance_avg - legacy_avg) / legacy_avg) * 100.0 if legacy_avg > 0 else 0.0

	print("")
	print("=" .repeat(60))
	print("[PHASE 6.6d] FINAL RESULTS")
	print("=" .repeat(60))
	print("")
	print("  LEGACY PATH:   %.1f FPS (P1: %.1f, P99: %.1f)" % [
		legacy_avg, results["LEGACY"]["p1"], results["LEGACY"]["p99"]
	])
	print("  INSTANCE PATH: %.1f FPS (P1: %.1f, P99: %.1f)" % [
		instance_avg, results["INSTANCE"]["p1"], results["INSTANCE"]["p99"]
	])
	print("")
	print("  DIFFERENCE: %+.1f%%" % diff_pct)
	print("")

	if abs(diff_pct) <= 10.0:
		print("  STATUS: PASS - Within 10% parity target")
	elif diff_pct > 0:
		print("  STATUS: PASS - Instance path FASTER than legacy")
	else:
		print("  STATUS: FAIL - Instance path %.1f%% slower than legacy" % abs(diff_pct))

	print("")
	print("=" .repeat(60))
	print("")

	# Quit after showing results
	await get_tree().create_timer(3.0).timeout
	get_tree().quit()

func _calc_avg(samples: Array[float]) -> float:
	if samples.size() == 0:
		return 0.0
	var sum = 0.0
	for s in samples:
		sum += s
	return sum / samples.size()

func _percentile(samples: Array[float], p: float) -> float:
	if samples.size() == 0:
		return 0.0
	var sorted = samples.duplicate()
	sorted.sort()
	var idx = int((p / 100.0) * (sorted.size() - 1))
	return sorted[idx]
