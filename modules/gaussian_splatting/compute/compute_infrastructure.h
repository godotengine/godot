#ifndef GS_COMPUTE_INFRASTRUCTURE_H
#define GS_COMPUTE_INFRASTRUCTURE_H

#include "core/error/error_list.h"
#include "core/string/ustring.h"
#include "core/templates/vector.h"
#include "servers/rendering/rendering_device.h"

namespace GaussianSplatting {
namespace ComputeInfrastructure {

enum class StageErrorCode : uint8_t {
	OK = 0,
	DEVICE_UNAVAILABLE,
	COMPUTE_UNSUPPORTED,
	INVALID_SHADER,
	SHADER_SOURCE_MISSING,
	SHADER_COMPILE_FAILED,
	SHADER_BINARY_ASSEMBLY_FAILED,
	SHADER_CREATE_FAILED,
	PIPELINE_CREATE_FAILED,
	UNIFORM_BINDING_INVALID,
	UNIFORM_BINDING_CONFLICT,
	UNIFORM_SET_CREATE_FAILED,
	DISPATCH_DIMENSIONS_INVALID
};

struct StageResult {
	StageErrorCode code = StageErrorCode::OK;
	String stage_name;
	String detail;

	_FORCE_INLINE_ bool ok() const { return code == StageErrorCode::OK; }
	Error to_error() const;
};

struct CapabilityGatePolicy {
	bool allow_cpu_fallback = false;
	bool allow_retry = true;
};

struct FallbackDecision {
	enum class Route : uint8_t {
		NONE = 0,
		RETRY_NEXT_FRAME,
		USE_CPU_FALLBACK,
		DISABLE_STAGE
	};

	Route route = Route::NONE;
	String reason;
};

struct UniformBindingContract {
	RenderingDevice::UniformType type = RD::UNIFORM_TYPE_MAX;
	uint32_t binding = 0;
	RID resource;
	String label;
};

struct StageValidationInput {
	String stage_name;
	Vector<UniformBindingContract> bindings;
	bool validate_dispatch = false;
	uint32_t dispatch_x = 0;
	uint32_t dispatch_y = 0;
	uint32_t dispatch_z = 0;
};

struct StageValidationRecord {
	String stage_name;
	StageResult status;
};

class StageValidationHarness {
	Vector<StageValidationRecord> records;

public:
	void reset();
	StageResult validate(const StageValidationInput &p_input);
	bool all_valid() const;
	String summarize_failures() const;
	const Vector<StageValidationRecord> &get_records() const { return records; }
};

StageResult check_compute_capabilities(RenderingDevice *p_device, const String &p_stage_name);
FallbackDecision resolve_fallback(const StageResult &p_result, const CapabilityGatePolicy &p_policy);

StageResult validate_stage_contract(const StageValidationInput &p_input);

StageResult create_uniform_set_checked(RenderingDevice *p_device, RID p_shader, uint32_t p_set_index,
		const Vector<UniformBindingContract> &p_bindings, const String &p_stage_name,
		const String &p_resource_name, RID &r_uniform_set);

StageResult create_pipeline_checked(RenderingDevice *p_device, RID p_shader, const String &p_stage_name,
		RID &r_pipeline);

StageResult compile_compute_shader_from_source(RenderingDevice *p_device, const String &p_stage_name,
		const String &p_source, const String &p_debug_name, RID &r_shader);

const char *stage_error_code_name(StageErrorCode p_code);
const char *fallback_route_name(FallbackDecision::Route p_route);
String format_stage_error(const String &p_owner, const StageResult &p_result);

} // namespace ComputeInfrastructure
} // namespace GaussianSplatting

#endif // GS_COMPUTE_INFRASTRUCTURE_H
