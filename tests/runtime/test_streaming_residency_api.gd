extends SceneTree

const FAIL_MARKER := "[RUNTIME_FAIL]"

func _init() -> void:
	call_deferred("_run")

func _record_failure(reason: String, context: Dictionary = {}) -> void:
	var message := reason
	if not context.is_empty():
		message = "%s | context=%s" % [reason, str(context)]
	push_error("%s %s" % [FAIL_MARKER, message])
	quit(1)

func _build_minimal_gaussian_data(count: int) -> GaussianData:
	var data := GaussianData.new()
	data.resize(count)

	var positions := PackedVector3Array()
	var scales := PackedVector3Array()
	var opacities := PackedFloat32Array()
	var sh_dc := PackedFloat32Array()

	positions.resize(count)
	scales.resize(count)
	opacities.resize(count)
	sh_dc.resize(count * 3)

	for i in range(count):
		positions[i] = Vector3(float(i), 0.0, 0.0)
		scales[i] = Vector3.ONE * 0.25
		opacities[i] = 0.9
		var sh_base := i * 3
		sh_dc[sh_base + 0] = 0.8
		sh_dc[sh_base + 1] = 0.7
		sh_dc[sh_base + 2] = 0.6

	data.set_positions(positions)
	data.set_scales(scales)
	data.set_opacities(opacities)
	data.set_spherical_harmonics(sh_dc)
	return data

func _run() -> void:
	var streaming := GaussianStreamingSystem.new()
	if streaming == null:
		_record_failure("Failed to create GaussianStreamingSystem")
		return

	var required_methods := [
		"begin_residency_requests",
		"request_chunk_residency",
		"request_asset_residency",
		"finalize_residency_requests",
		"get_residency_request_status"
	]
	for method_name in required_methods:
		if not streaming.has_method(method_name):
			_record_failure("Streaming residency API method missing", {"method": method_name})
			return

	var data := _build_minimal_gaussian_data(32)
	streaming.initialize(data)

	streaming.begin_residency_requests()
	streaming.finalize_residency_requests()
	var idle_status := streaming.get_residency_request_status(0, 0)
	if bool(idle_status.get("request_pending", true)):
		_record_failure("No-op finalize should not create pending residency work", idle_status)
		return
	if int(idle_status.get("request_state", -1)) != 0:
		_record_failure("No-op finalize should leave request state idle", idle_status)
		return
	if bool(idle_status.get("request_status_current_generation", true)):
		_record_failure("Idle residency status should not report a current-generation completion", idle_status)
		return

	streaming.begin_residency_requests()
	var chunk_result := streaming.request_chunk_residency(0, 0, 0)
	if chunk_result != OK:
		_record_failure("Primary explicit residency request was rejected", {"error": chunk_result})
		return
	var status := streaming.get_residency_request_status(0, 0)
	if not bool(status.get("requested", false)):
		_record_failure("Primary explicit residency request was not recorded", status)
		return
	if status.get("request_state_name", "") != "collected":
		_record_failure("Primary explicit residency request did not expose collected state", status)
		return
	if status.get("request_result_name", "") != "collected":
		_record_failure("Primary explicit residency request did not expose collected result", status)
		return
	if not bool(status.get("request_status_current_generation", false)):
		_record_failure("Collected residency request should report a current-generation status", status)
		return
	if not bool(status.get("request_pending", false)):
		_record_failure("Primary explicit residency request should mark pending work", status)
		return

	var asset_result := streaming.request_asset_residency(0, 0)
	if asset_result != OK:
		_record_failure("Primary explicit residency asset request was rejected", {"error": asset_result})
		return
	streaming.finalize_residency_requests()

	var final_status := streaming.get_residency_request_status(0, 0)
	if int(final_status.get("lod_mask", 0)) == 0:
		_record_failure("Primary explicit residency request did not populate a lod mask", final_status)
		return

	print("✅ Streaming residency API request state coverage passed")
	quit(0)
