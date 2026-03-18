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
		"finalize_residency_requests"
	]
	for method_name in required_methods:
		if not streaming.has_method(method_name):
			_record_failure("Streaming residency API method missing", {"method": method_name})
			return

	var data := _build_minimal_gaussian_data(32)
	streaming.initialize(data)

	# Script-driven residency request flow should remain callable even when runtime
	# streaming availability depends on device context.
	streaming.begin_residency_requests()
	streaming.request_chunk_residency(0, 0, 0)
	streaming.request_asset_residency(0, 0)
	streaming.finalize_residency_requests()

	print("✅ Streaming residency API script bindings are callable")
	quit(0)
