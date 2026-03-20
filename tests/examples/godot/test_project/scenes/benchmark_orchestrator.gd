extends Node

const DEFAULT_PLAN_PATH := "user://benchmark_plan.json"
const DEFAULT_RESULT_PATH := "user://benchmark_orchestrator_report.json"

var _plan_path := DEFAULT_PLAN_PATH
var _result_path := DEFAULT_RESULT_PATH
var _fail_fast := false
var _headless_summary := false

func _ready() -> void:
	_parse_args()
	call_deferred("_run_orchestration")

func _parse_args() -> void:
	var args := OS.get_cmdline_args()
	var i := 0
	while i < args.size():
		var arg := str(args[i])
		match arg:
			"--benchmark-fail-fast":
				_fail_fast = true
			"--benchmark-headless-summary":
				_headless_summary = true
			"--benchmark-plan":
				i += 1
				if i < args.size():
					_plan_path = str(args[i])
			"--benchmark-orchestrator-report":
				i += 1
				if i < args.size():
					_result_path = str(args[i])
			_:
				if arg.begins_with("--benchmark-plan="):
					_plan_path = arg.substr(len("--benchmark-plan="))
				elif arg.begins_with("--benchmark-orchestrator-report="):
					_result_path = arg.substr(len("--benchmark-orchestrator-report="))
		i += 1

func _run_orchestration() -> void:
	var started_unix := Time.get_unix_time_from_system()
	var started_ticks := Time.get_ticks_msec()
	var plan_payload = _load_json(_plan_path)
	if not (plan_payload is Dictionary):
		_finish_with_report({
			"success": false,
			"error": "invalid benchmark plan payload",
			"plan_path": _plan_path,
			"result_path": _result_path,
		}, 1)
		return

	var plan: Dictionary = plan_payload
	var entries_variant = plan.get("entries", [])
	if not (entries_variant is Array):
		_finish_with_report({
			"success": false,
			"error": "benchmark plan missing 'entries' array",
			"plan_path": _plan_path,
			"result_path": _result_path,
		}, 1)
		return

	var entries: Array = entries_variant
	var results: Array[Dictionary] = []
	var run_failed := false
	for entry_variant in entries:
		if not (entry_variant is Dictionary):
			var malformed_result := {
				"success": false,
				"timed_out": false,
				"exit_code": 1,
				"error": "malformed plan entry (expected dictionary)",
			}
			results.append(malformed_result)
			run_failed = true
			if _fail_fast:
				break
			continue

		var entry: Dictionary = entry_variant
		var lane_id := str(entry.get("lane_id", "unknown_lane"))
		var scene_path := str(entry.get("scene", ""))
		var contract_variant = entry.get("contract", {})
		var contract: Dictionary = {}
		if contract_variant is Dictionary:
			contract = contract_variant
		var timeout_s := float(entry.get("timeout_s", 0.0))
		var expected_report_path := str(entry.get("report_path", contract.get("output_path", "")))

		var lane_result := await _run_one_scene(
			lane_id,
			scene_path,
			contract,
			timeout_s,
			expected_report_path,
		)
		results.append(lane_result)
		if not bool(lane_result.get("success", false)):
			run_failed = true
			if _fail_fast:
				break

	var report := {
		"name": "GodotGS Single-Process Benchmark Orchestrator",
		"success": not run_failed,
		"profile": str(plan.get("profile", "")),
		"plan_path": _plan_path,
		"result_path": _result_path,
		"fail_fast": _fail_fast,
		"entry_count": entries.size(),
		"completed_count": results.size(),
		"timestamp_unix": started_unix,
		"duration_s": maxf(0.0, float(Time.get_ticks_msec() - started_ticks) / 1000.0),
		"entries": results,
	}
	_finish_with_report(report, 1 if run_failed else 0)

func _run_one_scene(
	lane_id: String,
	scene_path: String,
	contract: Dictionary,
	timeout_s: float,
	expected_report_path: String,
) -> Dictionary:
	var started_ticks := Time.get_ticks_msec()
	var packed := load(scene_path)
	if packed == null:
		return {
			"lane_id": lane_id,
			"scene": scene_path,
			"success": false,
			"timed_out": false,
			"exit_code": 1,
			"error": "failed to load scene",
			"report_path": expected_report_path,
		}

	var instance = packed.instantiate()
	if instance == null:
		return {
			"lane_id": lane_id,
			"scene": scene_path,
			"success": false,
			"timed_out": false,
			"exit_code": 1,
			"error": "failed to instantiate scene",
			"report_path": expected_report_path,
		}

	if instance.has_method("apply_benchmark_contract"):
		instance.call("apply_benchmark_contract", contract)
	else:
		return {
			"lane_id": lane_id,
			"scene": scene_path,
			"success": false,
			"timed_out": false,
			"exit_code": 1,
			"error": "scene missing apply_benchmark_contract(contract)",
			"report_path": expected_report_path,
		}

	if not instance.has_signal("benchmark_scene_finished"):
		return {
			"lane_id": lane_id,
			"scene": scene_path,
			"success": false,
			"timed_out": false,
			"exit_code": 1,
			"error": "scene missing benchmark_scene_finished signal",
			"report_path": expected_report_path,
		}

	add_child(instance)
	var done := false
	var payload: Dictionary = {}
	instance.connect("benchmark_scene_finished", func(result):
		if result is Dictionary:
			payload = result
		else:
			payload = {"result": result}
		done = true
	, CONNECT_ONE_SHOT)

	var timed_out := false
	while not done:
		await get_tree().process_frame
		if timeout_s > 0.0:
			var elapsed_s := float(Time.get_ticks_msec() - started_ticks) / 1000.0
			if elapsed_s > timeout_s:
				timed_out = true
				break
		if not is_instance_valid(instance):
			break

	if timed_out and is_instance_valid(instance):
		instance.queue_free()
		done = true

	var success := false
	var error := ""
	var report_path := expected_report_path
	if timed_out:
		error = "scene timeout exceeded"
	elif payload.is_empty():
		error = "scene finished without payload"
	else:
		success = bool(payload.get("success", false))
		error = str(payload.get("error", ""))
		report_path = str(payload.get("report_path", expected_report_path))

	return {
		"lane_id": lane_id,
		"scene": scene_path,
		"success": success and not timed_out,
		"timed_out": timed_out,
		"exit_code": 124 if timed_out else (0 if success else 1),
		"error": error,
		"report_path": report_path,
		"duration_s": maxf(0.0, float(Time.get_ticks_msec() - started_ticks) / 1000.0),
		"result": payload,
	}

func _finish_with_report(report: Dictionary, exit_code: int) -> void:
	_write_json(_result_path, report)
	if _headless_summary:
		print("[BENCH-ORCH] profile=%s success=%s completed=%d" % [
			str(report.get("profile", "")),
			str(report.get("success", false)),
			int(report.get("completed_count", 0)),
		])
	print("[BENCH-ORCH] report=%s" % _result_path)
	get_tree().quit(exit_code)

func _load_json(path: String):
	if not FileAccess.file_exists(path):
		return {}
	var handle := FileAccess.open(path, FileAccess.READ)
	if handle == null:
		return {}
	return JSON.parse_string(handle.get_as_text())

func _write_json(path: String, payload: Dictionary) -> void:
	var handle := FileAccess.open(path, FileAccess.WRITE)
	if handle == null:
		return
	handle.store_string(JSON.stringify(payload, "  "))
