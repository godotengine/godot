#include "compute_infrastructure.h"

#include "core/templates/hash_set.h"

namespace GaussianSplatting {
namespace ComputeInfrastructure {

static StageResult _stage_ok(const String &p_stage_name) {
	StageResult result;
	result.code = StageErrorCode::OK;
	result.stage_name = p_stage_name;
	return result;
}

static StageResult _stage_fail(StageErrorCode p_code, const String &p_stage_name, const String &p_detail) {
	StageResult result;
	result.code = p_code;
	result.stage_name = p_stage_name;
	result.detail = p_detail;
	return result;
}

static StageResult _validate_bindings(const String &p_stage_name, const Vector<UniformBindingContract> &p_bindings) {
	if (p_bindings.is_empty()) {
		return _stage_fail(StageErrorCode::UNIFORM_BINDING_INVALID, p_stage_name,
				"missing uniform binding contract");
	}

	HashSet<uint32_t> seen_bindings;
	for (int i = 0; i < p_bindings.size(); i++) {
		const UniformBindingContract &binding = p_bindings[i];
		if (binding.type == RD::UNIFORM_TYPE_MAX) {
			return _stage_fail(StageErrorCode::UNIFORM_BINDING_INVALID, p_stage_name,
					vformat("uniform[%d] has invalid type", i));
		}
		if (!binding.resource.is_valid()) {
			const String label = binding.label.is_empty() ? String("unnamed") : binding.label;
			return _stage_fail(StageErrorCode::UNIFORM_BINDING_INVALID, p_stage_name,
					vformat("uniform[%d] (%s) has invalid RID", i, label));
		}
		if (seen_bindings.has(binding.binding)) {
			return _stage_fail(StageErrorCode::UNIFORM_BINDING_CONFLICT, p_stage_name,
					vformat("duplicate uniform binding index %d", binding.binding));
		}
		seen_bindings.insert(binding.binding);
	}

	return _stage_ok(p_stage_name);
}

static StageResult _validate_dispatch(const String &p_stage_name, uint32_t p_dispatch_x,
		uint32_t p_dispatch_y, uint32_t p_dispatch_z) {
	if (p_dispatch_x == 0 || p_dispatch_y == 0 || p_dispatch_z == 0) {
		return _stage_fail(StageErrorCode::DISPATCH_DIMENSIONS_INVALID, p_stage_name,
				vformat("invalid dispatch groups (%d, %d, %d)", p_dispatch_x, p_dispatch_y, p_dispatch_z));
	}
	return _stage_ok(p_stage_name);
}

Error StageResult::to_error() const {
	switch (code) {
		case StageErrorCode::OK:
			return OK;
		case StageErrorCode::DEVICE_UNAVAILABLE:
		case StageErrorCode::COMPUTE_UNSUPPORTED:
			return ERR_UNAVAILABLE;
		case StageErrorCode::INVALID_SHADER:
		case StageErrorCode::SHADER_SOURCE_MISSING:
		case StageErrorCode::SHADER_COMPILE_FAILED:
		case StageErrorCode::SHADER_BINARY_ASSEMBLY_FAILED:
		case StageErrorCode::SHADER_CREATE_FAILED:
		case StageErrorCode::PIPELINE_CREATE_FAILED:
		case StageErrorCode::UNIFORM_SET_CREATE_FAILED:
			return ERR_CANT_CREATE;
		case StageErrorCode::UNIFORM_BINDING_INVALID:
		case StageErrorCode::UNIFORM_BINDING_CONFLICT:
		case StageErrorCode::DISPATCH_DIMENSIONS_INVALID:
			return ERR_INVALID_PARAMETER;
	}
	return ERR_BUG;
}

void StageValidationHarness::reset() {
	records.clear();
}

StageResult StageValidationHarness::validate(const StageValidationInput &p_input) {
	StageResult result = validate_stage_contract(p_input);
	StageValidationRecord record;
	record.stage_name = p_input.stage_name;
	record.status = result;
	records.push_back(record);
	return result;
}

bool StageValidationHarness::all_valid() const {
	for (int i = 0; i < records.size(); i++) {
		if (!records[i].status.ok()) {
			return false;
		}
	}
	return true;
}

String StageValidationHarness::summarize_failures() const {
	String summary;
	for (int i = 0; i < records.size(); i++) {
		const StageResult &status = records[i].status;
		if (status.ok()) {
			continue;
		}
		if (!summary.is_empty()) {
			summary += "; ";
		}
		summary += vformat("%s:%s", records[i].stage_name, status.detail);
	}
	return summary;
}

StageResult check_compute_capabilities(RenderingDevice *p_device, const String &p_stage_name) {
	if (!p_device) {
		return _stage_fail(StageErrorCode::DEVICE_UNAVAILABLE, p_stage_name,
				"RenderingDevice is null");
	}

	const uint64_t max_invocations = p_device->limit_get(RD::LIMIT_MAX_COMPUTE_WORKGROUP_INVOCATIONS);
	const uint64_t max_x = p_device->limit_get(RD::LIMIT_MAX_COMPUTE_WORKGROUP_SIZE_X);
	const uint64_t max_y = p_device->limit_get(RD::LIMIT_MAX_COMPUTE_WORKGROUP_SIZE_Y);
	const uint64_t max_z = p_device->limit_get(RD::LIMIT_MAX_COMPUTE_WORKGROUP_SIZE_Z);
	if (max_invocations == 0 || max_x == 0 || max_y == 0 || max_z == 0) {
		return _stage_fail(StageErrorCode::COMPUTE_UNSUPPORTED, p_stage_name,
				"compute workgroup limits are unavailable on this device");
	}

	return _stage_ok(p_stage_name);
}

FallbackDecision resolve_fallback(const StageResult &p_result, const CapabilityGatePolicy &p_policy) {
	FallbackDecision decision;
	if (p_result.ok()) {
		decision.route = FallbackDecision::Route::NONE;
		decision.reason = "stage ready";
		return decision;
	}

	const bool hard_unavailable =
			p_result.code == StageErrorCode::DEVICE_UNAVAILABLE ||
			p_result.code == StageErrorCode::COMPUTE_UNSUPPORTED;
	if (hard_unavailable && p_policy.allow_cpu_fallback) {
		decision.route = FallbackDecision::Route::USE_CPU_FALLBACK;
		decision.reason = "compute unavailable, using CPU fallback";
		return decision;
	}

	if (!hard_unavailable && p_policy.allow_retry) {
		decision.route = FallbackDecision::Route::RETRY_NEXT_FRAME;
		decision.reason = "transient stage failure, retry next frame";
		return decision;
	}

	decision.route = FallbackDecision::Route::DISABLE_STAGE;
	decision.reason = hard_unavailable ? "compute unavailable; stage disabled" : "stage failed; disabled";
	return decision;
}

StageResult validate_stage_contract(const StageValidationInput &p_input) {
	if (p_input.stage_name.is_empty()) {
		return _stage_fail(StageErrorCode::UNIFORM_BINDING_INVALID, String(),
				"stage_name is required");
	}

	StageResult binding_result = _validate_bindings(p_input.stage_name, p_input.bindings);
	if (!binding_result.ok()) {
		return binding_result;
	}

	if (p_input.validate_dispatch) {
		StageResult dispatch_result = _validate_dispatch(
				p_input.stage_name, p_input.dispatch_x, p_input.dispatch_y, p_input.dispatch_z);
		if (!dispatch_result.ok()) {
			return dispatch_result;
		}
	}

	return _stage_ok(p_input.stage_name);
}

StageResult create_uniform_set_checked(RenderingDevice *p_device, RID p_shader, uint32_t p_set_index,
		const Vector<UniformBindingContract> &p_bindings, const String &p_stage_name,
		const String &p_resource_name, RID &r_uniform_set) {
	r_uniform_set = RID();

	if (!p_device) {
		return _stage_fail(StageErrorCode::DEVICE_UNAVAILABLE, p_stage_name,
				"cannot create uniform set without RenderingDevice");
	}
	if (!p_shader.is_valid()) {
		return _stage_fail(StageErrorCode::INVALID_SHADER, p_stage_name,
				"cannot create uniform set with invalid shader");
	}

	StageResult validation_result = _validate_bindings(p_stage_name, p_bindings);
	if (!validation_result.ok()) {
		return validation_result;
	}

	Vector<RD::Uniform> uniforms;
	uniforms.resize(p_bindings.size());
	for (int i = 0; i < p_bindings.size(); i++) {
		const UniformBindingContract &binding = p_bindings[i];
		RD::Uniform uniform;
		uniform.uniform_type = binding.type;
		uniform.binding = binding.binding;
		uniform.append_id(binding.resource);
		uniforms.write[i] = uniform;
	}

	r_uniform_set = p_device->uniform_set_create(uniforms, p_shader, p_set_index);
	if (!r_uniform_set.is_valid()) {
		return _stage_fail(StageErrorCode::UNIFORM_SET_CREATE_FAILED, p_stage_name,
				vformat("uniform_set_create failed for set %d", p_set_index));
	}

	if (!p_resource_name.is_empty()) {
		p_device->set_resource_name(r_uniform_set, p_resource_name);
	}

	return _stage_ok(p_stage_name);
}

StageResult create_pipeline_checked(RenderingDevice *p_device, RID p_shader, const String &p_stage_name,
		RID &r_pipeline) {
	r_pipeline = RID();

	if (!p_device) {
		return _stage_fail(StageErrorCode::DEVICE_UNAVAILABLE, p_stage_name,
				"cannot create compute pipeline without RenderingDevice");
	}
	if (!p_shader.is_valid()) {
		return _stage_fail(StageErrorCode::INVALID_SHADER, p_stage_name,
				"cannot create compute pipeline with invalid shader");
	}

	r_pipeline = p_device->compute_pipeline_create(p_shader);
	if (!r_pipeline.is_valid()) {
		return _stage_fail(StageErrorCode::PIPELINE_CREATE_FAILED, p_stage_name,
				"compute_pipeline_create returned invalid RID");
	}

	return _stage_ok(p_stage_name);
}

StageResult compile_compute_shader_from_source(RenderingDevice *p_device, const String &p_stage_name,
		const String &p_source, const String &p_debug_name, RID &r_shader) {
	r_shader = RID();
	if (!p_device) {
		return _stage_fail(StageErrorCode::DEVICE_UNAVAILABLE, p_stage_name,
				"cannot compile shader without RenderingDevice");
	}
	if (p_source.is_empty()) {
		return _stage_fail(StageErrorCode::SHADER_SOURCE_MISSING, p_stage_name,
				"compute shader source is empty");
	}

	String compile_error;
	Vector<uint8_t> spirv = p_device->shader_compile_spirv_from_source(
			RenderingDevice::SHADER_STAGE_COMPUTE, p_source,
			RenderingDevice::SHADER_LANGUAGE_GLSL, &compile_error);
	if (spirv.is_empty()) {
		return _stage_fail(StageErrorCode::SHADER_COMPILE_FAILED, p_stage_name,
				vformat("SPIR-V compile failed: %s",
						compile_error.is_empty() ? String("unknown error") : compile_error));
	}

	RenderingDevice::ShaderStageSPIRVData stage_data;
	stage_data.shader_stage = RenderingDevice::SHADER_STAGE_COMPUTE;
	stage_data.spirv = spirv;

	Vector<RenderingDevice::ShaderStageSPIRVData> shader_stages;
	shader_stages.push_back(stage_data);

	Vector<uint8_t> shader_binary = p_device->shader_compile_binary_from_spirv(shader_stages, p_debug_name);
	if (shader_binary.is_empty()) {
		return _stage_fail(StageErrorCode::SHADER_BINARY_ASSEMBLY_FAILED, p_stage_name,
				"shader_compile_binary_from_spirv returned empty bytecode");
	}

	r_shader = p_device->shader_create_from_bytecode(shader_binary);
	if (!r_shader.is_valid()) {
		return _stage_fail(StageErrorCode::SHADER_CREATE_FAILED, p_stage_name,
				"shader_create_from_bytecode returned invalid RID");
	}

	return _stage_ok(p_stage_name);
}

const char *stage_error_code_name(StageErrorCode p_code) {
	switch (p_code) {
		case StageErrorCode::OK:
			return "ok";
		case StageErrorCode::DEVICE_UNAVAILABLE:
			return "device_unavailable";
		case StageErrorCode::COMPUTE_UNSUPPORTED:
			return "compute_unsupported";
		case StageErrorCode::INVALID_SHADER:
			return "invalid_shader";
		case StageErrorCode::SHADER_SOURCE_MISSING:
			return "shader_source_missing";
		case StageErrorCode::SHADER_COMPILE_FAILED:
			return "shader_compile_failed";
		case StageErrorCode::SHADER_BINARY_ASSEMBLY_FAILED:
			return "shader_binary_assembly_failed";
		case StageErrorCode::SHADER_CREATE_FAILED:
			return "shader_create_failed";
		case StageErrorCode::PIPELINE_CREATE_FAILED:
			return "pipeline_create_failed";
		case StageErrorCode::UNIFORM_BINDING_INVALID:
			return "uniform_binding_invalid";
		case StageErrorCode::UNIFORM_BINDING_CONFLICT:
			return "uniform_binding_conflict";
		case StageErrorCode::UNIFORM_SET_CREATE_FAILED:
			return "uniform_set_create_failed";
		case StageErrorCode::DISPATCH_DIMENSIONS_INVALID:
			return "dispatch_dimensions_invalid";
	}
	return "unknown";
}

const char *fallback_route_name(FallbackDecision::Route p_route) {
	switch (p_route) {
		case FallbackDecision::Route::NONE:
			return "none";
		case FallbackDecision::Route::RETRY_NEXT_FRAME:
			return "retry_next_frame";
		case FallbackDecision::Route::USE_CPU_FALLBACK:
			return "use_cpu_fallback";
		case FallbackDecision::Route::DISABLE_STAGE:
			return "disable_stage";
	}
	return "unknown";
}

String format_stage_error(const String &p_owner, const StageResult &p_result) {
	String message = vformat("[%s] %s: code=%s", p_owner, p_result.stage_name,
			String(stage_error_code_name(p_result.code)));
	if (!p_result.detail.is_empty()) {
		message += vformat(" detail=%s", p_result.detail);
	}
	return message;
}

} // namespace ComputeInfrastructure
} // namespace GaussianSplatting
