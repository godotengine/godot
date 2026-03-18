#pragma once

#include "test_macros.h"

#include "../compute/compute_infrastructure.h"

namespace TestGaussianSplatting {

static GaussianSplatting::ComputeInfrastructure::UniformBindingContract _make_test_binding(
		uint32_t p_binding, uint64_t p_rid_value, RenderingDevice::UniformType p_type = RD::UNIFORM_TYPE_STORAGE_BUFFER) {
	GaussianSplatting::ComputeInfrastructure::UniformBindingContract contract;
	contract.binding = p_binding;
	contract.resource = RID::from_uint64(p_rid_value);
	contract.type = p_type;
	contract.label = vformat("binding_%d", p_binding);
	return contract;
}

TEST_CASE("[GaussianSplatting][ComputeInfra] Stage harness rejects duplicate bindings deterministically") {
	GaussianSplatting::ComputeInfrastructure::StageValidationHarness harness;
	GaussianSplatting::ComputeInfrastructure::StageValidationInput input;
	input.stage_name = "Test.DuplicateBindings";
	input.bindings.push_back(_make_test_binding(0, 11));
	input.bindings.push_back(_make_test_binding(0, 12));

	GaussianSplatting::ComputeInfrastructure::StageResult result = harness.validate(input);
	CHECK(!result.ok());
	CHECK(result.code == GaussianSplatting::ComputeInfrastructure::StageErrorCode::UNIFORM_BINDING_CONFLICT);
	CHECK(result.detail.contains("duplicate uniform binding index 0"));
	CHECK(!harness.all_valid());
	CHECK(harness.summarize_failures().contains("Test.DuplicateBindings"));
}

TEST_CASE("[GaussianSplatting][ComputeInfra] Stage harness validates dispatch dimensions") {
	GaussianSplatting::ComputeInfrastructure::StageValidationHarness harness;
	GaussianSplatting::ComputeInfrastructure::StageValidationInput input;
	input.stage_name = "Test.Dispatch";
	input.bindings.push_back(_make_test_binding(0, 21));
	input.validate_dispatch = true;
	input.dispatch_x = 0;
	input.dispatch_y = 1;
	input.dispatch_z = 1;

	GaussianSplatting::ComputeInfrastructure::StageResult result = harness.validate(input);
	CHECK(!result.ok());
	CHECK(result.code == GaussianSplatting::ComputeInfrastructure::StageErrorCode::DISPATCH_DIMENSIONS_INVALID);
	CHECK(result.detail.contains("invalid dispatch groups"));
}

TEST_CASE("[GaussianSplatting][ComputeInfra] Stage contract accepts dispatch dimensions only when all groups are non-zero") {
	GaussianSplatting::ComputeInfrastructure::StageValidationInput input;
	input.stage_name = "Test.DispatchGuards";
	input.bindings.push_back(_make_test_binding(0, 22));
	input.validate_dispatch = true;

	input.dispatch_x = 4;
	input.dispatch_y = 1;
	input.dispatch_z = 2;
	GaussianSplatting::ComputeInfrastructure::StageResult valid_result =
			GaussianSplatting::ComputeInfrastructure::validate_stage_contract(input);
	CHECK(valid_result.ok());

	input.dispatch_x = 1;
	input.dispatch_y = 0;
	input.dispatch_z = 1;
	GaussianSplatting::ComputeInfrastructure::StageResult invalid_result =
			GaussianSplatting::ComputeInfrastructure::validate_stage_contract(input);
	CHECK(!invalid_result.ok());
	CHECK(invalid_result.code == GaussianSplatting::ComputeInfrastructure::StageErrorCode::DISPATCH_DIMENSIONS_INVALID);
}

TEST_CASE("[GaussianSplatting][ComputeInfra] Fallback policy routes compute-unavailable failures to CPU path when enabled") {
	GaussianSplatting::ComputeInfrastructure::StageResult unsupported;
	unsupported.code = GaussianSplatting::ComputeInfrastructure::StageErrorCode::COMPUTE_UNSUPPORTED;
	unsupported.stage_name = "Test.Capability";
	unsupported.detail = "compute limits unavailable";

	GaussianSplatting::ComputeInfrastructure::CapabilityGatePolicy fallback_policy;
	fallback_policy.allow_cpu_fallback = true;
	fallback_policy.allow_retry = false;

	GaussianSplatting::ComputeInfrastructure::FallbackDecision decision =
			GaussianSplatting::ComputeInfrastructure::resolve_fallback(unsupported, fallback_policy);
	CHECK(decision.route == GaussianSplatting::ComputeInfrastructure::FallbackDecision::Route::USE_CPU_FALLBACK);
}

TEST_CASE("[GaussianSplatting][ComputeInfra] Fallback policy routes transient setup failures to retry path") {
	GaussianSplatting::ComputeInfrastructure::StageResult transient_error;
	transient_error.code = GaussianSplatting::ComputeInfrastructure::StageErrorCode::SHADER_COMPILE_FAILED;
	transient_error.stage_name = "Test.ShaderCompile";
	transient_error.detail = "bad source";

	GaussianSplatting::ComputeInfrastructure::CapabilityGatePolicy fallback_policy;
	fallback_policy.allow_cpu_fallback = false;
	fallback_policy.allow_retry = true;

	GaussianSplatting::ComputeInfrastructure::FallbackDecision decision =
			GaussianSplatting::ComputeInfrastructure::resolve_fallback(transient_error, fallback_policy);
	CHECK(decision.route == GaussianSplatting::ComputeInfrastructure::FallbackDecision::Route::RETRY_NEXT_FRAME);
}

TEST_CASE("[GaussianSplatting][ComputeInfra] Fallback routing disables stage when fallback and retry are unavailable") {
	GaussianSplatting::ComputeInfrastructure::StageResult unsupported;
	unsupported.code = GaussianSplatting::ComputeInfrastructure::StageErrorCode::COMPUTE_UNSUPPORTED;
	unsupported.stage_name = "Test.NoFallback";

	GaussianSplatting::ComputeInfrastructure::CapabilityGatePolicy fallback_policy;
	fallback_policy.allow_cpu_fallback = false;
	fallback_policy.allow_retry = false;

	GaussianSplatting::ComputeInfrastructure::FallbackDecision decision =
			GaussianSplatting::ComputeInfrastructure::resolve_fallback(unsupported, fallback_policy);
	CHECK(decision.route == GaussianSplatting::ComputeInfrastructure::FallbackDecision::Route::DISABLE_STAGE);
	CHECK(decision.reason.contains("compute unavailable"));
}

TEST_CASE("[GaussianSplatting][ComputeInfra] Fallback routing keeps successful stages on the normal path") {
	GaussianSplatting::ComputeInfrastructure::StageResult ok_result;
	ok_result.code = GaussianSplatting::ComputeInfrastructure::StageErrorCode::OK;
	ok_result.stage_name = "Test.Success";

	GaussianSplatting::ComputeInfrastructure::CapabilityGatePolicy fallback_policy;
	fallback_policy.allow_cpu_fallback = true;
	fallback_policy.allow_retry = true;

	GaussianSplatting::ComputeInfrastructure::FallbackDecision decision =
			GaussianSplatting::ComputeInfrastructure::resolve_fallback(ok_result, fallback_policy);
	CHECK(decision.route == GaussianSplatting::ComputeInfrastructure::FallbackDecision::Route::NONE);
	CHECK(decision.reason == "stage ready");
}

TEST_CASE("[GaussianSplatting][ComputeInfra] Null-device capability checks provide deterministic unsupported errors") {
	GaussianSplatting::ComputeInfrastructure::StageResult result =
			GaussianSplatting::ComputeInfrastructure::check_compute_capabilities(nullptr, "Test.CapabilityNull");
	CHECK(!result.ok());
	CHECK(result.code == GaussianSplatting::ComputeInfrastructure::StageErrorCode::DEVICE_UNAVAILABLE);
	CHECK(result.detail.contains("RenderingDevice is null"));
}

TEST_CASE("[GaussianSplatting][ComputeInfra] Uniform set creation returns explicit device-unavailable error") {
	Vector<GaussianSplatting::ComputeInfrastructure::UniformBindingContract> bindings;
	bindings.push_back(_make_test_binding(0, 31));

	RID uniform_set;
	GaussianSplatting::ComputeInfrastructure::StageResult result =
			GaussianSplatting::ComputeInfrastructure::create_uniform_set_checked(
					nullptr, RID::from_uint64(100), 0, bindings,
					"Test.UniformSet", "TestUniformSet", uniform_set);
	CHECK(!result.ok());
	CHECK(result.code == GaussianSplatting::ComputeInfrastructure::StageErrorCode::DEVICE_UNAVAILABLE);
	CHECK(!uniform_set.is_valid());
}

TEST_CASE("[GaussianSplatting][ComputeInfra] Stage error formatting includes owner, stage, code, and detail") {
	GaussianSplatting::ComputeInfrastructure::StageResult result;
	result.code = GaussianSplatting::ComputeInfrastructure::StageErrorCode::UNIFORM_BINDING_CONFLICT;
	result.stage_name = "Test.ErrorFormatting";
	result.detail = "duplicate binding";

	String formatted = GaussianSplatting::ComputeInfrastructure::format_stage_error("TileRenderer", result);
	CHECK(formatted.contains("[TileRenderer] Test.ErrorFormatting"));
	CHECK(formatted.contains("code=uniform_binding_conflict"));
	CHECK(formatted.contains("detail=duplicate binding"));
}

TEST_CASE("[GaussianSplatting][ComputeInfra] Stage error formatting omits detail payload when empty") {
	GaussianSplatting::ComputeInfrastructure::StageResult result;
	result.code = GaussianSplatting::ComputeInfrastructure::StageErrorCode::SHADER_CREATE_FAILED;
	result.stage_name = "Test.ErrorFormatting.NoDetail";

	String formatted = GaussianSplatting::ComputeInfrastructure::format_stage_error("Sorter", result);
	CHECK(formatted.contains("code=shader_create_failed"));
	CHECK(!formatted.contains("detail="));
}

} // namespace TestGaussianSplatting
