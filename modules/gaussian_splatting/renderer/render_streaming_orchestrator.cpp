#include "render_streaming_orchestrator.h"

#include "gaussian_gpu_layout.h"
#include "instance_pipeline_contract.h"
#include "pipeline_io_contracts.h"
#include "render_data_orchestrator.h"
#include "render_device_orchestrator.h"
#include "render_pipeline_stages.h"
#include "resource_owner_mismatch_contract.h"
#include "gpu_sorting_config.h"
#include "../core/gaussian_splat_scene_director.h"
#include "../interfaces/gpu_sorting_pipeline.h"
#include "../logger/gs_debug_trace.h"
#include "../logger/gs_logger.h"

#include "core/error/error_macros.h"
#include "core/math/math_defs.h"
#include "servers/rendering/renderer_rd/storage_rd/render_data_rd.h"
#include <algorithm>
#include <cstdint>

using StreamingState = GaussianSplatRenderer::StreamingState;
using PerformanceSettings = GaussianSplatRenderer::PerformanceSettings;
using ResourceState = GaussianSplatRenderer::ResourceState;
using RenderFallbackReason = GaussianSplatRenderer::RenderFallbackReason;
using InstancePipelineBuffers = GaussianSplatRenderer::InstancePipelineBuffers;
using InvariantViolation = GaussianSplatting::InstancePipelineContract::InvariantViolation;
using InvariantViolationReason = GaussianSplatting::InstancePipelineContract::InvariantViolationReason;

static uint64_t _mix_content_generation(uint64_t a, uint64_t b) {
	uint64_t x = a + 0x9e3779b97f4a7c15ULL;
	x ^= b + (x << 6) + (x >> 2);
	return x;
}

static uint64_t _mix_u32_generation(uint64_t p_generation, uint32_t p_value) {
	return _mix_content_generation(p_generation, uint64_t(p_value));
}

static uint64_t _mix_rid_generation(uint64_t p_generation, const RID &p_rid) {
	return _mix_content_generation(p_generation, p_rid.is_valid() ? p_rid.get_id() : 0ULL);
}

static uint64_t _compute_instance_pipeline_resource_fingerprint(const ResourceState &p_resource_state, const InstancePipelineBuffers &p_buffers) {
	uint64_t generation = 0x6a09e667f3bcc909ULL;
	generation = _mix_rid_generation(generation, p_resource_state.instance_buffer);
	generation = _mix_u32_generation(generation, p_resource_state.instance_buffer_capacity);
	generation = _mix_rid_generation(generation, p_resource_state.instance_visible_chunk_buffer);
	generation = _mix_u32_generation(generation, p_resource_state.instance_visible_chunk_capacity);
	generation = _mix_rid_generation(generation, p_resource_state.instance_splat_ref_buffer);
	generation = _mix_u32_generation(generation, p_resource_state.instance_splat_ref_capacity);
	generation = _mix_rid_generation(generation, p_resource_state.instance_counter_buffer);
	generation = _mix_rid_generation(generation, p_resource_state.instance_chunk_dispatch_buffer);
	generation = _mix_rid_generation(generation, p_resource_state.instance_indirect_count_buffer);
	generation = _mix_rid_generation(generation, p_resource_state.instance_count_buffer);
	generation = _mix_rid_generation(generation, p_buffers.atlas_gaussian_buffer);
	generation = _mix_u32_generation(generation, p_buffers.atlas_gaussian_count);
	generation = _mix_rid_generation(generation, p_buffers.asset_meta_buffer);
	generation = _mix_rid_generation(generation, p_buffers.chunk_meta_buffer);
	generation = _mix_rid_generation(generation, p_buffers.asset_chunk_index_buffer);
	generation = _mix_u32_generation(generation, p_buffers.quantization_required ? 1u : 0u);
	generation = _mix_rid_generation(generation, p_buffers.quantization_buffer);
	generation = _mix_rid_generation(generation, p_buffers.visible_chunk_buffer);
	generation = _mix_rid_generation(generation, p_buffers.splat_ref_buffer);
	generation = _mix_rid_generation(generation, p_buffers.sort_key_buffer);
	generation = _mix_rid_generation(generation, p_buffers.sort_value_buffer);
	generation = _mix_rid_generation(generation, p_buffers.counter_buffer);
	generation = _mix_rid_generation(generation, p_buffers.chunk_dispatch_buffer);
	generation = _mix_rid_generation(generation, p_buffers.indirect_count_buffer);
	generation = _mix_rid_generation(generation, p_buffers.instance_count_buffer);
	generation = _mix_u32_generation(generation, p_buffers.instance_count);
	generation = _mix_u32_generation(generation, p_buffers.dispatch_chunk_count);
	generation = _mix_u32_generation(generation, p_buffers.max_visible_chunks);
	generation = _mix_u32_generation(generation, p_buffers.max_visible_splats);
	generation = _mix_u32_generation(generation, p_buffers.max_chunk_splats);
	return generation;
}

constexpr uint32_t kInstanceAssetEmptySyncGraceFrames = 3u;

enum class LayoutHintValidationUsage : uint8_t {
	IO = 0,
	PRIMARY = 1,
};

enum class LayoutHintFailureReason : uint8_t {
	NONE = 0,
	HINTS_EMPTY,
	HINT_COUNT_ZERO,
	HINT_NON_CONTIGUOUS_COVERAGE,
	HINT_OVERLAPPING_RANGES,
	REMAP_SOURCE_COUNT_MISMATCH,
	REMAP_TOTAL_COUNT_MISMATCH,
	REMAP_SOURCE_INDEX_OUT_OF_RANGE,
	COUNT,
};

enum class LayoutHintFailureCategory : uint8_t {
	INPUT = 0,
	NON_CONTIGUOUS,
	INDEX_RANGE,
	REMAP,
	OTHER,
};

static const char *_layout_hint_usage_code(LayoutHintValidationUsage p_usage) {
	return p_usage == LayoutHintValidationUsage::PRIMARY ? "primary" : "io";
}

static const char *_layout_hint_reason_code(LayoutHintFailureReason p_reason) {
	switch (p_reason) {
		case LayoutHintFailureReason::NONE:
			return "none";
		case LayoutHintFailureReason::HINTS_EMPTY:
			return "hints_empty";
		case LayoutHintFailureReason::HINT_COUNT_ZERO:
			return "hint_count_zero";
		case LayoutHintFailureReason::HINT_NON_CONTIGUOUS_COVERAGE:
			return "hint_non_contiguous_coverage";
		case LayoutHintFailureReason::HINT_OVERLAPPING_RANGES:
			return "hint_overlapping_ranges";
		case LayoutHintFailureReason::REMAP_SOURCE_COUNT_MISMATCH:
			return "remap_source_count_mismatch";
		case LayoutHintFailureReason::REMAP_TOTAL_COUNT_MISMATCH:
			return "remap_total_count_mismatch";
		case LayoutHintFailureReason::REMAP_SOURCE_INDEX_OUT_OF_RANGE:
			return "remap_source_index_out_of_range";
		case LayoutHintFailureReason::COUNT:
			break;
	}
	return "unknown";
}

static LayoutHintFailureCategory _layout_hint_reason_category(LayoutHintFailureReason p_reason) {
	switch (p_reason) {
		case LayoutHintFailureReason::HINTS_EMPTY:
		case LayoutHintFailureReason::HINT_COUNT_ZERO:
			return LayoutHintFailureCategory::INPUT;
		case LayoutHintFailureReason::HINT_NON_CONTIGUOUS_COVERAGE:
		case LayoutHintFailureReason::HINT_OVERLAPPING_RANGES:
			return LayoutHintFailureCategory::NON_CONTIGUOUS;
		case LayoutHintFailureReason::REMAP_SOURCE_INDEX_OUT_OF_RANGE:
			return LayoutHintFailureCategory::INDEX_RANGE;
		case LayoutHintFailureReason::REMAP_SOURCE_COUNT_MISMATCH:
		case LayoutHintFailureReason::REMAP_TOTAL_COUNT_MISMATCH:
			return LayoutHintFailureCategory::REMAP;
		case LayoutHintFailureReason::NONE:
		case LayoutHintFailureReason::COUNT:
			break;
	}
	return LayoutHintFailureCategory::OTHER;
}

static const char *_layout_hint_category_code(LayoutHintFailureCategory p_category) {
	switch (p_category) {
		case LayoutHintFailureCategory::INPUT:
			return "input";
		case LayoutHintFailureCategory::NON_CONTIGUOUS:
			return "non_contiguous";
		case LayoutHintFailureCategory::INDEX_RANGE:
			return "index_range";
		case LayoutHintFailureCategory::REMAP:
			return "remap";
		case LayoutHintFailureCategory::OTHER:
			return "other";
	}
	return "other";
}

static String _layout_hint_failure_detail(int p_hint_index, uint64_t p_detail_a, uint64_t p_detail_b) {
	if (p_hint_index < 0 && p_detail_a == 0 && p_detail_b == 0) {
		return String();
	}
	return vformat("hint_idx=%d detail_a=%d detail_b=%d",
			p_hint_index,
			static_cast<int64_t>(p_detail_a),
			static_cast<int64_t>(p_detail_b));
}

static void _apply_static_layout_fallback_diagnostics(Dictionary &r_streaming_metrics,
		uint64_t p_fallback_total,
		uint64_t p_fallback_io_total,
		uint64_t p_fallback_primary_total,
		const HashMap<String, uint64_t> &p_reason_counts,
		const HashMap<String, uint64_t> &p_category_counts,
		const String &p_last_usage,
		const String &p_last_reason,
		const String &p_last_reason_category,
		const String &p_last_context,
		const String &p_last_detail) {
	Dictionary reason_counts;
	for (int i = 1; i < static_cast<int>(LayoutHintFailureReason::COUNT); i++) {
		const LayoutHintFailureReason reason = static_cast<LayoutHintFailureReason>(i);
		reason_counts[_layout_hint_reason_code(reason)] = static_cast<int64_t>(0);
	}
	for (const KeyValue<String, uint64_t> &E : p_reason_counts) {
		reason_counts[E.key] = static_cast<int64_t>(E.value);
	}

	Dictionary category_counts;
	category_counts["input"] = static_cast<int64_t>(0);
	category_counts["non_contiguous"] = static_cast<int64_t>(0);
	category_counts["index_range"] = static_cast<int64_t>(0);
	category_counts["remap"] = static_cast<int64_t>(0);
	category_counts["other"] = static_cast<int64_t>(0);
	for (const KeyValue<String, uint64_t> &E : p_category_counts) {
		category_counts[E.key] = static_cast<int64_t>(E.value);
	}

	Dictionary orchestrator_validation;
	orchestrator_validation["fallback_total"] = static_cast<int64_t>(p_fallback_total);
	orchestrator_validation["fallback_io_total"] = static_cast<int64_t>(p_fallback_io_total);
	orchestrator_validation["fallback_primary_total"] = static_cast<int64_t>(p_fallback_primary_total);
	orchestrator_validation["last_usage"] = p_last_usage;
	orchestrator_validation["last_reason"] = p_last_reason;
	orchestrator_validation["last_reason_category"] = p_last_reason_category;
	orchestrator_validation["last_context"] = p_last_context;
	orchestrator_validation["last_detail"] = p_last_detail;
	orchestrator_validation["reason_counts"] = reason_counts;
	orchestrator_validation["category_counts"] = category_counts;
	r_streaming_metrics["layout_hint_orchestrator_validation"] = orchestrator_validation;

	Dictionary layout_hint_validation;
	if (r_streaming_metrics.has("layout_hint_validation")) {
		const Variant existing_layout_validation = r_streaming_metrics["layout_hint_validation"];
		if (existing_layout_validation.get_type() == Variant::DICTIONARY) {
			layout_hint_validation = existing_layout_validation;
		}
	}
	layout_hint_validation["orchestrator_fallback_total"] = static_cast<int64_t>(p_fallback_total);
	layout_hint_validation["orchestrator_fallback_io_total"] = static_cast<int64_t>(p_fallback_io_total);
	layout_hint_validation["orchestrator_fallback_primary_total"] = static_cast<int64_t>(p_fallback_primary_total);
	layout_hint_validation["orchestrator_last_usage"] = p_last_usage;
	layout_hint_validation["orchestrator_last_reason"] = p_last_reason;
	layout_hint_validation["orchestrator_last_reason_category"] = p_last_reason_category;
	layout_hint_validation["orchestrator_last_context"] = p_last_context;
	layout_hint_validation["orchestrator_last_detail"] = p_last_detail;
	layout_hint_validation["orchestrator_reason_counts"] = reason_counts;
	layout_hint_validation["orchestrator_category_counts"] = category_counts;
	r_streaming_metrics["layout_hint_validation"] = layout_hint_validation;

	r_streaming_metrics["layout_hint_orchestrator_fallback_total"] = static_cast<int64_t>(p_fallback_total);
	r_streaming_metrics["layout_hint_orchestrator_fallback_io_total"] = static_cast<int64_t>(p_fallback_io_total);
	r_streaming_metrics["layout_hint_orchestrator_fallback_primary_total"] = static_cast<int64_t>(p_fallback_primary_total);
	r_streaming_metrics["layout_hint_orchestrator_last_usage"] = p_last_usage;
	r_streaming_metrics["layout_hint_orchestrator_last_reason"] = p_last_reason;
	r_streaming_metrics["layout_hint_orchestrator_last_reason_category"] = p_last_reason_category;
	r_streaming_metrics["layout_hint_orchestrator_last_context"] = p_last_context;
	r_streaming_metrics["layout_hint_orchestrator_last_detail"] = p_last_detail;
	r_streaming_metrics["layout_hint_orchestrator_reason_counts"] = reason_counts;
	r_streaming_metrics["layout_hint_orchestrator_category_counts"] = category_counts;

	const String existing_last_reason = r_streaming_metrics.get("layout_hint_last_reason", String("none"));
	if (p_fallback_total > 0 && existing_last_reason == String("none")) {
		r_streaming_metrics["layout_hint_last_reason"] = p_last_reason;
		r_streaming_metrics["layout_hint_last_reason_category"] = p_last_reason_category;
		r_streaming_metrics["layout_hint_last_context"] = p_last_context;
		r_streaming_metrics["layout_hint_last_usage"] = p_last_usage;
	}
}

enum class StreamingReadinessState : uint8_t {
	READY = 0,
	MISSING_STREAMING_SYSTEM,
	RUNTIME_BOOTSTRAP_PENDING,
	REGISTRATION_PENDING,
	MISSING_ATLAS_INPUTS,
	MISSING_CULL_INPUTS,
	MISSING_SORT_INPUTS,
	MISSING_RASTER_INPUTS,
};

static const char *_streaming_readiness_state_token(StreamingReadinessState p_state) {
	switch (p_state) {
		case StreamingReadinessState::READY:
			return "READY";
		case StreamingReadinessState::MISSING_STREAMING_SYSTEM:
			return "MISSING_STREAMING_SYSTEM";
		case StreamingReadinessState::RUNTIME_BOOTSTRAP_PENDING:
			return "RUNTIME_BOOTSTRAP_PENDING";
		case StreamingReadinessState::REGISTRATION_PENDING:
			return "REGISTRATION_PENDING";
		case StreamingReadinessState::MISSING_ATLAS_INPUTS:
			return "MISSING_ATLAS_INPUTS";
		case StreamingReadinessState::MISSING_CULL_INPUTS:
			return "MISSING_CULL_INPUTS";
		case StreamingReadinessState::MISSING_SORT_INPUTS:
			return "MISSING_SORT_INPUTS";
		case StreamingReadinessState::MISSING_RASTER_INPUTS:
			return "MISSING_RASTER_INPUTS";
		default:
			return "UNKNOWN";
	}
}

static String _streaming_not_ready_route_uid(StreamingReadinessState p_state) {
	return String("COMMON.SKIP.STREAMING_NOT_READY.") + String(_streaming_readiness_state_token(p_state));
}

static String _streaming_not_ready_reason(StreamingReadinessState p_state) {
	switch (p_state) {
		case StreamingReadinessState::READY:
			return "Streaming path ready";
		case StreamingReadinessState::MISSING_STREAMING_SYSTEM:
			return "Streaming system unavailable";
		case StreamingReadinessState::RUNTIME_BOOTSTRAP_PENDING:
			return "Streaming runtime bootstrap pending";
		case StreamingReadinessState::REGISTRATION_PENDING:
			return "Streaming asset registration pending";
		case StreamingReadinessState::MISSING_ATLAS_INPUTS:
			return "Streaming atlas inputs unavailable";
		case StreamingReadinessState::MISSING_CULL_INPUTS:
			return "Streaming cull inputs unavailable";
		case StreamingReadinessState::MISSING_SORT_INPUTS:
			return "Streaming sort inputs unavailable";
		case StreamingReadinessState::MISSING_RASTER_INPUTS:
			return "Streaming raster inputs unavailable";
		default:
			return "Streaming readiness unknown";
	}
}

static void _apply_streaming_render_readiness_diagnostics(Dictionary &r_streaming_metrics,
		StreamingReadinessState p_state,
		const String &p_detail = String()) {
	const bool ready = p_state == StreamingReadinessState::READY;
	const String state_token = String(_streaming_readiness_state_token(p_state));
	const String reason = _streaming_not_ready_reason(p_state);
	const String route_uid = ready ? String("INSTANCE.STREAMING") : _streaming_not_ready_route_uid(p_state);
	Dictionary readiness;
	readiness["state"] = state_token;
	readiness["reason"] = reason;
	readiness["ready"] = ready;
	readiness["route_uid"] = route_uid;
	if (!p_detail.is_empty()) {
		readiness["detail"] = p_detail;
	}
	r_streaming_metrics["render_readiness"] = readiness;
	r_streaming_metrics["render_readiness_state"] = state_token;
	r_streaming_metrics["render_readiness_reason"] = reason;
	r_streaming_metrics["render_readiness_ready"] = ready;
	r_streaming_metrics["render_readiness_route_uid"] = route_uid;
	if (!p_detail.is_empty()) {
		r_streaming_metrics["render_readiness_detail"] = p_detail;
	}
	if (!ready) {
		r_streaming_metrics["diagnostics_category"] = String("render_not_ready");
		r_streaming_metrics["diagnostics_fingerprint"] =
				String("render_not_ready.") + state_token.to_lower();
		r_streaming_metrics["diagnostics_has_failure"] = true;
		Dictionary diagnostics;
		if (r_streaming_metrics.has("diagnostics")) {
			const Variant existing = r_streaming_metrics["diagnostics"];
			if (existing.get_type() == Variant::DICTIONARY) {
				diagnostics = existing;
			}
		}
		diagnostics["reason"] = reason;
		diagnostics["render_readiness_state"] = state_token;
		diagnostics["render_readiness_route_uid"] = route_uid;
		if (!p_detail.is_empty()) {
			diagnostics["detail"] = p_detail;
		}
		r_streaming_metrics["diagnostics"] = diagnostics;
	}
}

static bool _is_debug_or_test_invariant_hard_fail_enabled() {
#if defined(DEBUG_ENABLED) || defined(TESTS_ENABLED)
	return true;
#else
	return false;
#endif
}

static bool _is_impossible_streaming_activation_violation(uint32_t p_instance_count,
		uint32_t p_registered_assets_with_data,
		bool p_allow_runtime_fallback_instance,
		const InvariantViolation &p_violation) {
	if (!p_violation.has_violation()) {
		return false;
	}
	if (p_instance_count == 0) {
		return false;
	}
	// Runtime fallback can legitimately run with a synthetic primary instance before
	// atlas-backed streaming assets become fully resident.
	if (p_allow_runtime_fallback_instance && p_registered_assets_with_data == 0) {
		return false;
	}
	return true;
}

struct OwnerMismatchRemediationResult {
	bool mismatch_detected = false;
	bool remediated = false;
	bool forced_invalidation = false;
};

static OwnerMismatchRemediationResult _remediate_instance_pipeline_buffer_owner_mismatch(
		GaussianSplatRenderer *p_renderer,
		RenderingDevice *p_active_rd,
		RID &r_buffer,
		uint32_t *r_capacity,
		const char *p_label) {
	OwnerMismatchRemediationResult result;
	if (!p_renderer || !p_active_rd || !r_buffer.is_valid()) {
		return result;
	}

	RenderingDevice *owner = p_renderer->get_resource_owner(r_buffer, nullptr);
	ResourceOwnerMismatchContract::Inputs contract_inputs;
	contract_inputs.rid_valid = r_buffer.is_valid();
	contract_inputs.has_owner = owner != nullptr;
	contract_inputs.owner_instance_id = owner ? owner->get_device_instance_id() : 0;
	contract_inputs.active_instance_id = p_active_rd->get_device_instance_id();
	ResourceOwnerMismatchContract::Decision decision =
			ResourceOwnerMismatchContract::evaluate(contract_inputs);
	String decision_error;
	if (!ResourceOwnerMismatchContract::validate(contract_inputs, decision, &decision_error)) {
		ERR_PRINT(vformat("[GaussianSplatRenderer] Owner-mismatch contract violation for %s: %s",
				p_label ? String(p_label) : String("instance_buffer"),
				decision_error));
		decision.mismatch_detected = true;
		decision.should_attempt_release = true;
		decision.should_force_invalidate_after_release = true;
	}
	if (!decision.mismatch_detected) {
		return result;
	}

	result.mismatch_detected = true;
	if (decision.should_attempt_release) {
		p_renderer->free_owned_resource(owner ? owner : p_active_rd, r_buffer);
	}
	if (r_capacity) {
		*r_capacity = 0;
	}
	if (!r_buffer.is_valid()) {
		result.remediated = true;
		return result;
	}

	RenderingDevice *post_owner = p_renderer->get_resource_owner(r_buffer, nullptr);
	if (decision.should_force_invalidate_after_release ||
			!post_owner ||
			post_owner != p_active_rd) {
		result.forced_invalidation = true;
		p_renderer->forget_resource_owner(r_buffer);
		r_buffer = RID();
		if (r_capacity) {
			*r_capacity = 0;
		}
	}

	if (r_buffer.is_valid()) {
		RenderingDevice *validated_owner = p_renderer->get_resource_owner(r_buffer, nullptr);
		result.remediated = validated_owner == p_active_rd;
	} else {
		result.remediated = true;
	}

	if (!result.remediated) {
		ERR_PRINT(vformat("[GaussianSplatRenderer] Owner-mismatch remediation failed for %s; forcing handle invalidation.",
				p_label ? String(p_label) : String("instance_buffer")));
		p_renderer->forget_resource_owner(r_buffer);
		r_buffer = RID();
		if (r_capacity) {
			*r_capacity = 0;
		}
		result.forced_invalidation = true;
		result.remediated = true;
	}

	return result;
}

static void _collect_instance_pipeline_residency_requests(const LocalVector<InstanceDataGPU> &p_instance_cache,
		HashMap<uint32_t, uint32_t> &r_lod_mask_cache,
		GaussianStreamingSystem *p_streaming_system,
		int *r_request_count = nullptr) {
	if (r_request_count) {
		*r_request_count = 0;
	}
	if (p_instance_cache.is_empty()) {
		return;
	}

	r_lod_mask_cache.clear();
	r_lod_mask_cache.reserve(p_instance_cache.size());
	for (const InstanceDataGPU &entry : p_instance_cache) {
		const uint32_t asset_id = entry.ids[0];
		if (asset_id == 0) {
			continue;
		}
		const uint32_t lod_level = MIN<uint32_t>(entry.lod[0], GS_MAX_ASSET_LODS - 1);
		uint32_t *mask = r_lod_mask_cache.getptr(asset_id);
		if (!mask) {
			r_lod_mask_cache.insert(asset_id, 1u << lod_level);
		} else {
			*mask |= (1u << lod_level);
		}
	}

	for (const KeyValue<uint32_t, uint32_t> &E : r_lod_mask_cache) {
		const uint32_t asset_id = E.key;
		uint32_t mask = E.value;
		for (uint32_t lod = 0; lod < GS_MAX_ASSET_LODS; lod++) {
			if (mask & (1u << lod)) {
				p_streaming_system->request_asset_residency(asset_id, lod);
				if (r_request_count) {
					(*r_request_count)++;
				}
			}
		}
	}
}

RenderStreamingOrchestrator::RenderStreamingOrchestrator(GaussianSplatRenderer *p_renderer,
		RenderDataOrchestrator *p_data_orchestrator,
		RenderDeviceOrchestrator *p_device_orchestrator) :
		renderer(p_renderer),
		data_orchestrator(p_data_orchestrator),
		device_orchestrator(p_device_orchestrator) {
	ERR_FAIL_NULL(renderer);
	ERR_FAIL_NULL(data_orchestrator);
	ERR_FAIL_NULL(device_orchestrator);
}

const RenderStreamingOrchestrator::VisibleLODSelection &RenderStreamingOrchestrator::produce_visible_lod_selection(
		GaussianSplatSceneDirector *p_director,
		GaussianStreamingSystem *p_streaming_system,
		const Vector3 &p_camera_origin) {
	visible_lod_selection.instances.clear();
	visible_lod_selection.residency_request_count = 0;

	if (!p_director || !p_streaming_system) {
		return visible_lod_selection;
	}

	const LODConfig &lod_config = p_streaming_system->get_lod_config();
	const float hysteresis_zone = p_streaming_system->get_lod_hysteresis_zone();
	p_director->update_instance_lods_for_renderer(renderer, p_camera_origin, lod_config, hysteresis_zone);
	p_director->build_instance_buffer_for_renderer(renderer, visible_lod_selection.instances);

	return visible_lod_selection;
}

void RenderStreamingOrchestrator::consume_visible_lod_selection_for_residency(
		const VisibleLODSelection &p_selection,
		GaussianStreamingSystem *p_streaming_system,
		bool p_trace_enabled) {
	if (!p_streaming_system) {
		return;
	}

	p_streaming_system->begin_residency_requests();
	if (p_trace_enabled) {
		GaussianSplatting::debug_trace_record_event("instance_pipeline",
				vformat("Residency instance_cache_size=%d", p_selection.get_instances().size()),
				false);
	}

	visible_lod_selection.residency_request_count = 0;
	if (p_selection.has_instances()) {
		int request_count = 0;
		_collect_instance_pipeline_residency_requests(
				p_selection.get_instances(),
				instance_pipeline_lod_mask_cache,
				p_streaming_system,
				&request_count);
		visible_lod_selection.residency_request_count = static_cast<uint32_t>(MAX(request_count, 0));
	}

	if (p_trace_enabled) {
		GaussianSplatting::debug_trace_record_event("instance_pipeline",
				vformat("Residency lod_masks=%d requests=%d",
						instance_pipeline_lod_mask_cache.size(),
						visible_lod_selection.residency_request_count),
				false);
	}
	p_streaming_system->finalize_residency_requests();
}

bool RenderStreamingOrchestrator::should_throttle_streaming_rebuild(uint32_t p_chunks_loaded, uint32_t p_chunks_evicted,
		uint32_t p_visible_evicted, uint64_t p_current_frame) {
	StreamingState &streaming_state = renderer->get_streaming_state();

	uint32_t total_chunks_changed = p_chunks_loaded + p_chunks_evicted;

	// Only throttle if:
	// 1. We have a valid cache to use
	// 2. Changes are small (below threshold)
	// 3. We rebuilt recently
	if (!streaming_state.cached_streamed_indices_valid) {
		return false;
	}

	// CRITICAL FIX: Never throttle when chunks have JUST loaded!
	// The previous logic had a bug where:
	// - Frame N: Chunk 1 loads, cache rebuilt with chunk 1 only
	// - Frame N+1: Chunk 0 loads, but throttled (within MIN_FRAMES window)
	// - Frame N+30+: No more chunks_loaded events, cache NEVER rebuilt!
	// This caused chunk 0's indices to be permanently missing.
	//
	// Fix: Only throttle evictions, not loads. When chunks load, we MUST
	// update the cache immediately to include the new data.
	if (p_chunks_loaded > 0) {
		return false; // Never throttle loads - new data must be visible immediately!
	}

	bool is_small_change = total_chunks_changed > 0 &&
			total_chunks_changed <= StreamingState::SMALL_CHANGE_THRESHOLD;
	uint64_t frames_since_rebuild = p_current_frame - streaming_state.last_rebuild_frame;

	if (is_small_change && frames_since_rebuild < StreamingState::MIN_FRAMES_BETWEEN_SMALL_REBUILDS) {
		// Log throttle decision periodically
		if (p_current_frame - streaming_state.last_throttle_log_frame >= StreamingState::LOG_THROTTLE_FRAMES) {
			GS_LOG_STREAMING_DEBUG(vformat("[PERF-THROTTLE] Skipping rebuild (changed=%d, frames_since=%d)",
					total_chunks_changed, frames_since_rebuild));
			streaming_state.last_throttle_log_frame = p_current_frame;
		}
		return true;
	}

	return false;
}

bool RenderStreamingOrchestrator::ensure_instance_streaming_system() {
	StreamingState &streaming_state = renderer->get_streaming_state();
	if (streaming_state.current_streaming_system.is_valid()) {
		return true;
	}
	if (!renderer->ensure_rendering_device("instance_pipeline_streaming")) {
		return false;
	}
	RenderingDevice *rd = renderer->get_device_state().rd;
	if (!rd) {
		return false;
	}

	Ref<GaussianStreamingSystem> streaming_system;
	streaming_system.instantiate();
	GaussianStreamingSystem::ConfigOverrides overrides;
	if (data_orchestrator) {
		overrides = data_orchestrator->get_streaming_config_overrides();
	}
	// Instance pipeline: disable VRAM auto-regulation.
	// All instances share the same atlas chunks, so thrashing detection
	// falsely triggers (load+evict during initialization → regulation spiral).
	// The atlas for instance pipeline is small and stable — regulation is unnecessary.
	// Load max_chunks from project settings to avoid coupling with max_splat_count (Issue #798)
	VRAMBudgetConfig instance_budget = VRAMBudgetConfig::load_from_project_settings();
	instance_budget.auto_regulate_enabled = false;
	instance_budget.min_chunks = 4;
	overrides.override_vram_budget = true;
	overrides.vram_budget_config = instance_budget;
	streaming_system->set_config_overrides(overrides);
	streaming_system->set_chunk_radius_multiplier(
			renderer->get_cull_radius_multiplier() * renderer->get_cull_frustum_plane_slack());
	const Ref<GaussianData> primary_data = renderer->get_scene_state().gaussian_data;
	if (primary_data.is_valid() && primary_data->get_count() > 0) {
		// Recover world/static-only renderers that populated gaussian_data before
		// the RenderingDevice was available. In that case set_gaussian_data() can
		// leave scene data valid while streaming bootstrap is missing.
		streaming_system->initialize_with_device(primary_data, rd);
	} else {
		streaming_system->initialize_empty(rd);
	}
	if (streaming_state.memory_stream.is_valid()) {
		streaming_system->attach_memory_stream(streaming_state.memory_stream);
	}
	String init_error;
	if (!streaming_system->is_runtime_ready(&init_error)) {
		ERR_PRINT(vformat("[GaussianSplatRenderer] Failed to initialize streaming bootstrap: %s", init_error));
		return false;
	}
	streaming_state.current_streaming_system = streaming_system;
	streaming_state.use_streamed_data = false;
	streaming_state.cached_streamed_indices_valid = false;
	streaming_state.current_stream_gpu_buffer = RID();
	streaming_state.streaming_gpu_splat_count = 0;
	streaming_state.streaming_gpu_total_capacity = 0;
	streaming_state.streamed_indices_are_local = false;
	instance_pipeline_assets.clear();
	instance_pipeline_asset_versions.clear();
	// Streaming system can be recreated while static chunk revision stays unchanged.
	// Force one sync cache miss so primary/io chunk layout hints are reapplied.
	static_layout_cache_revision = UINT64_MAX;
	static_layout_cache_hints.clear();
	static_layout_cache_valid = false;
	static_layout_warned_non_contiguous = false;
	static_layout_fallback_total = 0;
	static_layout_fallback_io_total = 0;
	static_layout_fallback_primary_total = 0;
	static_layout_fallback_reason_counts.clear();
	static_layout_fallback_category_counts.clear();
	static_layout_fallback_last_usage = "none";
	static_layout_fallback_last_reason = "none";
	static_layout_fallback_last_reason_category = "other";
	static_layout_fallback_last_context = "none";
	static_layout_fallback_last_detail = "none";
	static_layout_bound_asset_id = UINT32_MAX;
	primary_layout_cache_hints.clear();
	primary_layout_cache_source_indices.clear();
	primary_layout_cache_valid = false;
	primary_layout_warned_invalid = false;
	return streaming_state.current_streaming_system.is_valid();
}

void RenderStreamingOrchestrator::sync_instance_pipeline_assets(GaussianStreamingSystem *p_streaming_system) {
	const bool trace_enabled = GaussianSplatting::debug_trace_is_enabled();
	if (!p_streaming_system) {
		if (trace_enabled) {
			GaussianSplatting::debug_trace_record_event("instance_pipeline", "SyncAssets FAIL: streaming_system=NULL", true);
		}
		return;
	}
	GaussianSplatSceneDirector *director = GaussianSplatSceneDirector::get_singleton();
	if (!director) {
		if (trace_enabled) {
			GaussianSplatting::debug_trace_record_event("instance_pipeline", "SyncAssets FAIL: director=NULL", true);
		}
		return;
	}
	instance_pipeline_assets_cache.clear();
	director->collect_instance_assets_for_renderer(renderer, instance_pipeline_assets_cache);
	if (trace_enabled) {
		GaussianSplatting::debug_trace_record_event("instance_pipeline",
				vformat("SyncAssets collected=%d", instance_pipeline_assets_cache.size()),
				false);
	}
	bool transient_empty_asset_sync = false;
	if (instance_pipeline_assets_cache.is_empty()) {
		if (!instance_pipeline_assets.is_empty()) {
			instance_pipeline_empty_asset_sync_streak++;
			transient_empty_asset_sync = instance_pipeline_empty_asset_sync_streak < kInstanceAssetEmptySyncGraceFrames;
			if (trace_enabled && transient_empty_asset_sync) {
				GaussianSplatting::debug_trace_record_event("instance_pipeline",
						vformat("SyncAssets transient-empty streak=%d/%d (keeping previous asset registration)",
								instance_pipeline_empty_asset_sync_streak, kInstanceAssetEmptySyncGraceFrames - 1u),
						false);
			}
		} else {
			instance_pipeline_empty_asset_sync_streak = 0;
		}
	} else {
		instance_pipeline_empty_asset_sync_streak = 0;
	}

	const auto &cull_state = renderer->get_subsystem_state().gpu_culler->get_state();
	const Vector<GaussianSplatRenderer::StaticChunk> &static_chunks = cull_state.static_chunks;
	const uint64_t static_chunks_revision = cull_state.static_chunks_revision;
	const bool static_layout_cache_miss = static_layout_cache_revision != static_chunks_revision;
	const uint32_t prev_bound_asset_id = static_layout_bound_asset_id;
	bool static_layout_rebind_required = false;
	LayoutHintFailureReason primary_layout_failure_reason = LayoutHintFailureReason::NONE;
	int primary_layout_failure_hint_index = -1;
	uint64_t primary_layout_failure_detail_a = 0;
	uint64_t primary_layout_failure_detail_b = 0;
	auto set_primary_layout_failure = [&](LayoutHintFailureReason p_reason,
			int p_hint_index = -1,
			uint64_t p_detail_a = 0,
			uint64_t p_detail_b = 0) {
		if (primary_layout_failure_reason == LayoutHintFailureReason::NONE) {
			primary_layout_failure_reason = p_reason;
			primary_layout_failure_hint_index = p_hint_index;
			primary_layout_failure_detail_a = p_detail_a;
			primary_layout_failure_detail_b = p_detail_b;
		}
	};
	LayoutHintFailureReason io_layout_failure_reason = LayoutHintFailureReason::NONE;
	int io_layout_failure_hint_index = -1;
	uint64_t io_layout_failure_detail_a = 0;
	uint64_t io_layout_failure_detail_b = 0;
	auto set_io_layout_failure = [&](LayoutHintFailureReason p_reason,
			int p_hint_index = -1,
			uint64_t p_detail_a = 0,
			uint64_t p_detail_b = 0) {
		if (io_layout_failure_reason == LayoutHintFailureReason::NONE) {
			io_layout_failure_reason = p_reason;
			io_layout_failure_hint_index = p_hint_index;
			io_layout_failure_detail_a = p_detail_a;
			io_layout_failure_detail_b = p_detail_b;
		}
	};
	auto record_static_layout_fallback = [&](LayoutHintValidationUsage p_usage,
			LayoutHintFailureReason p_reason,
			const String &p_context,
			int p_hint_index = -1,
			uint64_t p_detail_a = 0,
			uint64_t p_detail_b = 0) {
		if (p_reason == LayoutHintFailureReason::NONE) {
			return;
		}
		static_layout_fallback_total++;
		if (p_usage == LayoutHintValidationUsage::PRIMARY) {
			static_layout_fallback_primary_total++;
		} else {
			static_layout_fallback_io_total++;
		}
		const String reason_code = _layout_hint_reason_code(p_reason);
		const String category_code = _layout_hint_category_code(_layout_hint_reason_category(p_reason));
		uint64_t *reason_count = static_layout_fallback_reason_counts.getptr(reason_code);
		if (!reason_count) {
			static_layout_fallback_reason_counts.insert(reason_code, 1);
		} else {
			(*reason_count)++;
		}
		uint64_t *category_count = static_layout_fallback_category_counts.getptr(category_code);
		if (!category_count) {
			static_layout_fallback_category_counts.insert(category_code, 1);
		} else {
			(*category_count)++;
		}
		static_layout_fallback_last_usage = _layout_hint_usage_code(p_usage);
		static_layout_fallback_last_reason = reason_code;
		static_layout_fallback_last_reason_category = category_code;
		static_layout_fallback_last_context = p_context;
		const String detail = _layout_hint_failure_detail(p_hint_index, p_detail_a, p_detail_b);
		static_layout_fallback_last_detail = detail.is_empty() ? String("none") : detail;
	};
	if (static_layout_cache_miss) {
		static_layout_cache_revision = static_chunks_revision;
		static_layout_cache_hints.clear();
		static_layout_cache_valid = false;
		static_layout_warned_non_contiguous = false;
		primary_layout_cache_hints.clear();
		primary_layout_cache_source_indices.clear();
		primary_layout_cache_valid = false;
		primary_layout_warned_invalid = false;
		// Layout changed: force deterministic rebind of the target asset.
		static_layout_bound_asset_id = UINT32_MAX;
		static_layout_rebind_required = true;

		if (static_chunks.is_empty()) {
			static_layout_cache_valid = true;
			primary_layout_cache_valid = true;
		} else {
			bool primary_layout_valid = true;
			uint64_t total_source_indices = 0;
			for (int chunk_idx = 0; chunk_idx < static_chunks.size(); chunk_idx++) {
				const GaussianSplatRenderer::StaticChunk &chunk = static_chunks[chunk_idx];
				if (chunk.indices.is_empty()) {
					set_primary_layout_failure(LayoutHintFailureReason::HINT_COUNT_ZERO, chunk_idx);
					primary_layout_valid = false;
					break;
				}
				total_source_indices += static_cast<uint64_t>(chunk.indices.size());
				if (total_source_indices > static_cast<uint64_t>(UINT32_MAX)) {
					set_primary_layout_failure(LayoutHintFailureReason::REMAP_SOURCE_COUNT_MISMATCH,
							chunk_idx,
							total_source_indices,
							UINT32_MAX);
					primary_layout_valid = false;
					break;
				}
			}
			if (primary_layout_valid && total_source_indices > 0) {
				primary_layout_cache_source_indices.resize(static_cast<int>(total_source_indices));
				const Ref<GaussianData> primary_data = renderer->get_scene_state().gaussian_data;
				const uint32_t primary_splat_count = primary_data.is_valid() ? primary_data->get_count() : 0;
				uint32_t write_offset = 0;
				for (int chunk_idx = 0; chunk_idx < static_chunks.size() && primary_layout_valid; chunk_idx++) {
					const GaussianSplatRenderer::StaticChunk &chunk = static_chunks[chunk_idx];
					uint32_t chunk_read_offset = 0;
					while (chunk_read_offset < static_cast<uint32_t>(chunk.indices.size())) {
						const uint32_t remaining = static_cast<uint32_t>(chunk.indices.size()) - chunk_read_offset;
						const uint32_t split_count = MIN(GaussianStreamingSystem::CHUNK_SIZE, remaining);
						GaussianStreamingSystem::ChunkLayoutHint hint;
						hint.start_idx = 0;
						hint.count = split_count;
						hint.source_index_offset = write_offset;
						hint.source_indices_remapped = true;
						hint.bounds = chunk.bounds;
						hint.center = chunk.center;
						hint.radius = chunk.radius;
						primary_layout_cache_hints.push_back(hint);
						for (uint32_t local_idx = 0; local_idx < split_count; local_idx++) {
							const uint32_t source_idx = chunk.indices[chunk_read_offset + local_idx];
							if (primary_splat_count > 0 && source_idx >= primary_splat_count) {
								set_primary_layout_failure(LayoutHintFailureReason::REMAP_SOURCE_INDEX_OUT_OF_RANGE,
										chunk_idx,
										source_idx,
										primary_splat_count);
								primary_layout_valid = false;
								break;
							}
							primary_layout_cache_source_indices.write[write_offset + local_idx] = source_idx;
						}
						write_offset += split_count;
						chunk_read_offset += split_count;
						if (!primary_layout_valid) {
							break;
						}
					}
				}
				if (primary_layout_valid && write_offset != static_cast<uint32_t>(total_source_indices)) {
					set_primary_layout_failure(LayoutHintFailureReason::REMAP_TOTAL_COUNT_MISMATCH,
							-1,
							write_offset,
							total_source_indices);
					primary_layout_valid = false;
				}
			} else if (primary_layout_valid) {
				set_primary_layout_failure(LayoutHintFailureReason::HINTS_EMPTY);
				primary_layout_valid = false;
			}
			primary_layout_cache_valid = primary_layout_valid;
			if (!primary_layout_cache_valid) {
				primary_layout_cache_hints.clear();
				primary_layout_cache_source_indices.clear();
			}

			Vector<GaussianStreamingSystem::ChunkLayoutHint> layout_hints;
			layout_hints.resize(static_chunks.size());
			struct ChunkLayoutRange {
				uint32_t start = 0;
				uint32_t end = 0;
				int hint_index = -1;
			};
			LocalVector<ChunkLayoutRange> layout_ranges;
			layout_ranges.resize(static_chunks.size());
			bool layout_valid = true;
			for (int chunk_idx = 0; chunk_idx < static_chunks.size(); chunk_idx++) {
				const GaussianSplatRenderer::StaticChunk &chunk = static_chunks[chunk_idx];
				if (chunk.indices.is_empty()) {
					set_io_layout_failure(LayoutHintFailureReason::HINT_COUNT_ZERO, chunk_idx);
					layout_valid = false;
					break;
				}
				uint32_t min_idx = UINT32_MAX;
				uint32_t max_idx = 0;
				for (int i = 0; i < chunk.indices.size(); i++) {
					const uint32_t idx = chunk.indices[i];
					min_idx = MIN(min_idx, idx);
					max_idx = MAX(max_idx, idx);
				}
				const uint64_t expected_count = uint64_t(max_idx) - uint64_t(min_idx) + 1;
				if (expected_count != static_cast<uint64_t>(chunk.indices.size())) {
					set_io_layout_failure(LayoutHintFailureReason::HINT_NON_CONTIGUOUS_COVERAGE,
							chunk_idx,
							expected_count,
							static_cast<uint64_t>(chunk.indices.size()));
					layout_valid = false;
					break;
				}
				Vector<uint8_t> seen;
				seen.resize(chunk.indices.size());
				for (int i = 0; i < seen.size(); i++) {
					seen.write[i] = 0;
				}
				for (int i = 0; i < chunk.indices.size(); i++) {
					const uint32_t idx = chunk.indices[i];
					const uint32_t offset = idx - min_idx;
					if (offset >= static_cast<uint32_t>(seen.size()) || seen[offset] != 0) {
						set_io_layout_failure(LayoutHintFailureReason::HINT_NON_CONTIGUOUS_COVERAGE,
								chunk_idx,
								offset,
								static_cast<uint64_t>(seen.size()));
						layout_valid = false;
						break;
					}
					seen.write[offset] = 1;
				}
				if (!layout_valid) {
					break;
				}
				layout_ranges[chunk_idx].start = min_idx;
				layout_ranges[chunk_idx].end = max_idx;
				layout_ranges[chunk_idx].hint_index = chunk_idx;
				GaussianStreamingSystem::ChunkLayoutHint hint;
				hint.start_idx = min_idx;
				hint.count = chunk.indices.size();
				hint.bounds = chunk.bounds;
				hint.center = chunk.center;
				hint.radius = chunk.radius;
				layout_hints.write[chunk_idx] = hint;
			}
			if (layout_valid && layout_ranges.size() > 1) {
				ChunkLayoutRange *ranges = layout_ranges.ptr();
				std::sort(ranges, ranges + layout_ranges.size(), [](const ChunkLayoutRange &a, const ChunkLayoutRange &b) {
					if (a.start == b.start) {
						return a.end < b.end;
					}
					return a.start < b.start;
				});
				uint32_t prev_end = ranges[0].end;
				for (uint32_t i = 1; i < layout_ranges.size(); i++) {
					if (ranges[i].start <= prev_end) {
						set_io_layout_failure(LayoutHintFailureReason::HINT_OVERLAPPING_RANGES,
								ranges[i].hint_index,
								ranges[i].start,
								prev_end);
						layout_valid = false;
						break;
					}
					prev_end = MAX(prev_end, ranges[i].end);
				}
			}
			static_layout_cache_valid = layout_valid;
			if (layout_valid) {
				static_layout_cache_hints = layout_hints;
			} else {
				static_layout_cache_hints.clear();
			}
		}
	}

	if (static_layout_cache_miss) {
		if (primary_layout_cache_valid) {
			p_streaming_system->set_primary_chunk_layout(primary_layout_cache_hints, primary_layout_cache_source_indices);
		} else {
			if (primary_layout_failure_reason == LayoutHintFailureReason::NONE) {
				primary_layout_failure_reason = LayoutHintFailureReason::HINTS_EMPTY;
			}
			record_static_layout_fallback(LayoutHintValidationUsage::PRIMARY,
					primary_layout_failure_reason,
					"orchestrator.primary_static_layout",
					primary_layout_failure_hint_index,
					primary_layout_failure_detail_a,
					primary_layout_failure_detail_b);
			if (!primary_layout_warned_invalid) {
				const String detail = _layout_hint_failure_detail(
						primary_layout_failure_hint_index,
						primary_layout_failure_detail_a,
						primary_layout_failure_detail_b);
				WARN_PRINT(vformat("[Streaming] Static chunk layout indices are invalid for primary streaming; "
								   "falling back to contiguous runtime chunk partitioning "
								   "(usage=%s reason=%s category=%s%s%s).",
						_layout_hint_usage_code(LayoutHintValidationUsage::PRIMARY),
						_layout_hint_reason_code(primary_layout_failure_reason),
						_layout_hint_category_code(_layout_hint_reason_category(primary_layout_failure_reason)),
						detail.is_empty() ? "" : " ",
						detail));
				primary_layout_warned_invalid = true;
			}
			p_streaming_system->set_primary_chunk_layout(Vector<GaussianStreamingSystem::ChunkLayoutHint>(), Vector<uint32_t>());
		}
	}

	bool bound_asset_present = false;
	uint32_t deterministic_asset_id = UINT32_MAX;
	for (const InstanceAssetRegistration &entry : instance_pipeline_assets_cache) {
		if (entry.asset_id == 0 || entry.data.is_null()) {
			continue;
		}
		if (deterministic_asset_id == UINT32_MAX || entry.asset_id < deterministic_asset_id) {
			deterministic_asset_id = entry.asset_id;
		}
		if (entry.asset_id == static_layout_bound_asset_id) {
			bound_asset_present = true;
		}
	}
	if (!transient_empty_asset_sync && !bound_asset_present) {
		static_layout_bound_asset_id = deterministic_asset_id;
	}
	if (static_layout_bound_asset_id != prev_bound_asset_id) {
		static_layout_rebind_required = true;
	}

	if (static_chunks.is_empty()) {
		p_streaming_system->set_io_chunk_layout_hints(Vector<GaussianStreamingSystem::ChunkLayoutHint>());
	} else if (static_layout_cache_valid) {
		p_streaming_system->set_io_chunk_layout_hints(static_layout_cache_hints, static_layout_bound_asset_id);
	} else {
		if (static_layout_cache_miss) {
			if (io_layout_failure_reason == LayoutHintFailureReason::NONE) {
				io_layout_failure_reason = LayoutHintFailureReason::HINT_NON_CONTIGUOUS_COVERAGE;
			}
			record_static_layout_fallback(LayoutHintValidationUsage::IO,
					io_layout_failure_reason,
					"orchestrator.io_static_layout",
					io_layout_failure_hint_index,
					io_layout_failure_detail_a,
					io_layout_failure_detail_b);
		}
		if (!static_layout_warned_non_contiguous) {
			const String detail = _layout_hint_failure_detail(
					io_layout_failure_hint_index,
					io_layout_failure_detail_a,
					io_layout_failure_detail_b);
			const LayoutHintFailureReason emit_reason = io_layout_failure_reason == LayoutHintFailureReason::NONE
					? LayoutHintFailureReason::HINT_NON_CONTIGUOUS_COVERAGE
					: io_layout_failure_reason;
			WARN_PRINT(vformat("[Streaming] Static chunk layout is non-contiguous; falling back to contiguous runtime chunk partitioning "
							   "(usage=%s reason=%s category=%s%s%s).",
					_layout_hint_usage_code(LayoutHintValidationUsage::IO),
					_layout_hint_reason_code(emit_reason),
					_layout_hint_category_code(_layout_hint_reason_category(emit_reason)),
					detail.is_empty() ? "" : " ",
					detail));
			static_layout_warned_non_contiguous = true;
		}
		p_streaming_system->set_io_chunk_layout_hints(Vector<GaussianStreamingSystem::ChunkLayoutHint>());
	}

	instance_pipeline_assets_next.clear();
	if (transient_empty_asset_sync) {
		instance_pipeline_assets_next.reserve(instance_pipeline_assets.size());
		for (uint32_t asset_id : instance_pipeline_assets) {
			instance_pipeline_assets_next.insert(asset_id);
		}
	} else {
		instance_pipeline_assets_next.reserve(instance_pipeline_assets_cache.size());
		for (const InstanceAssetRegistration &entry : instance_pipeline_assets_cache) {
			if (entry.asset_id == 0 || entry.data.is_null()) {
				if (trace_enabled) {
					GaussianSplatting::debug_trace_record_event("instance_pipeline",
							vformat("SyncAssets SKIP asset_id=%d data=%s",
									entry.asset_id, entry.data.is_valid() ? "valid" : "null"),
							true);
				}
				continue;
			}
			instance_pipeline_assets_next.insert(entry.asset_id);
			const uint32_t asset_id = entry.asset_id;
			uint32_t *version_ptr = instance_pipeline_asset_versions.getptr(asset_id);
			const bool version_changed = !version_ptr || *version_ptr != entry.edited_version;
			const bool layout_rebind_for_asset = static_layout_rebind_required &&
					(asset_id == static_layout_bound_asset_id || asset_id == prev_bound_asset_id);
			const bool needs_register = !instance_pipeline_assets.has(asset_id) ||
					!p_streaming_system->is_asset_registered(asset_id) ||
					version_changed ||
					layout_rebind_for_asset;
			if (trace_enabled) {
				GaussianSplatting::debug_trace_record_event("instance_pipeline",
						vformat("SyncAssets asset_id=%d count=%d needs_register=%s version_changed=%s layout_rebind=%s",
									asset_id, entry.data.is_valid() ? entry.data->get_count() : 0,
									needs_register ? "YES" : "no",
									version_changed ? "YES" : "no",
									layout_rebind_for_asset ? "YES" : "no"),
						false);
			}
			if (needs_register) {
				p_streaming_system->register_asset(entry.asset_id, entry.data);
				if (trace_enabled) {
					GaussianSplatting::debug_trace_record_event("instance_pipeline",
							vformat("SyncAssets REGISTERED asset_id=%d", asset_id),
							false);
				}
				instance_pipeline_asset_versions.insert(asset_id, entry.edited_version);
			}
		}
	}

	if (!transient_empty_asset_sync && !instance_pipeline_assets.is_empty()) {
		instance_pipeline_assets_to_remove.clear();
		instance_pipeline_assets_to_remove.reserve(instance_pipeline_assets.size());
		for (uint32_t asset_id : instance_pipeline_assets) {
			if (!instance_pipeline_assets_next.has(asset_id)) {
				instance_pipeline_assets_to_remove.push_back(asset_id);
			}
		}
		for (uint32_t asset_id : instance_pipeline_assets_to_remove) {
			p_streaming_system->unregister_asset(asset_id);
			instance_pipeline_asset_versions.erase(asset_id);
		}
	}

	instance_pipeline_assets.clear();
	instance_pipeline_assets.reserve(instance_pipeline_assets_next.size());
	for (uint32_t asset_id : instance_pipeline_assets_next) {
		instance_pipeline_assets.insert(asset_id);
	}
}

bool RenderStreamingOrchestrator::render_streaming_frame(RenderDataRD *p_render_data, const Transform3D &p_camera_to_world_transform,
			const Transform3D &p_world_to_camera_transform, const Projection &p_projection, const Projection &p_render_projection,
			RenderSceneBuffersRD *p_render_buffers, bool p_allow_runtime_fallback_instance) {
	const bool trace_enabled = GaussianSplatting::debug_trace_is_enabled();
	// DEBUG: Track if this function runs every frame
	static int streaming_frame_counter = 0;
	if (trace_enabled && ++streaming_frame_counter % 60 == 1) {
		GaussianSplatting::debug_trace_record_event("render_streaming",
				vformat("frame=%d", streaming_frame_counter),
				false);
	}

	StreamingState &streaming_state = renderer->get_streaming_state();
	auto publish_not_ready_route = [&](StreamingReadinessState p_state) {
		renderer->get_debug_state().route_uid = _streaming_not_ready_route_uid(p_state);
	};
	if (!streaming_state.current_streaming_system.is_valid()) {
		const StreamingReadinessState readiness_state = StreamingReadinessState::MISSING_STREAMING_SYSTEM;
		publish_not_ready_route(readiness_state);
		Dictionary streaming_metrics = renderer->get_performance_state().metrics.streaming_state;
		_apply_streaming_render_readiness_diagnostics(streaming_metrics, readiness_state,
				"current_streaming_system invalid");
		renderer->get_performance_state().metrics.streaming_state = streaming_metrics;
		return false;
	}
	auto log_streaming_reset = [&](const char *p_reason) {
		const uint64_t system_id = streaming_state.current_streaming_system.is_valid()
				? uint64_t(reinterpret_cast<uintptr_t>(streaming_state.current_streaming_system.ptr()))
				: 0u;
		ERR_PRINT(vformat("[GaussianSplatRenderer] Streaming system reset (%s), previous_system=%s",
				p_reason ? String(p_reason) : String("unknown"),
				String::num_uint64(system_id)));
	};

	String instance_invariant_route = "instance_pipeline/invariant/not_checked";
	String instance_invariant_class = "none";
	String instance_invariant_reason = "none";
	bool instance_invariant_failed = false;
	uint32_t owner_mismatch_detected_count = 0;
	uint32_t owner_mismatch_forced_invalidation_count = 0;
	auto set_instance_invariant_status = [&](const InvariantViolation &p_violation) {
		instance_invariant_failed = p_violation.has_violation();
		if (!instance_invariant_failed) {
			instance_invariant_route = "instance_pipeline/invariant/ok";
			instance_invariant_class = "none";
			instance_invariant_reason = "none";
			return;
		}
		instance_invariant_route = GaussianSplatting::InstancePipelineContract::get_violation_route(p_violation.reason);
		instance_invariant_class = GaussianSplatting::InstancePipelineContract::get_violation_class_name(p_violation.violation_class);
		instance_invariant_reason = GaussianSplatting::InstancePipelineContract::get_violation_reason_name(p_violation.reason);
	};

	auto finalize_streaming_frame = [&](StreamingReadinessState p_readiness_state = StreamingReadinessState::READY,
			const String &p_readiness_detail = String()) {
		streaming_state.current_streaming_system->end_frame();
		Dictionary streaming_metrics = streaming_state.current_streaming_system->get_streaming_analytics();
		_apply_static_layout_fallback_diagnostics(streaming_metrics,
				static_layout_fallback_total,
				static_layout_fallback_io_total,
				static_layout_fallback_primary_total,
				static_layout_fallback_reason_counts,
				static_layout_fallback_category_counts,
				static_layout_fallback_last_usage,
				static_layout_fallback_last_reason,
				static_layout_fallback_last_reason_category,
				static_layout_fallback_last_context,
				static_layout_fallback_last_detail);
		_apply_streaming_render_readiness_diagnostics(streaming_metrics, p_readiness_state, p_readiness_detail);
		streaming_metrics["instance_pipeline_invariant_route"] = instance_invariant_route;
		streaming_metrics["instance_pipeline_invariant_class"] = instance_invariant_class;
		streaming_metrics["instance_pipeline_invariant_reason"] = instance_invariant_reason;
		streaming_metrics["instance_pipeline_invariant_failed"] = instance_invariant_failed;
		streaming_metrics["instance_pipeline_owner_mismatch_detected"] =
				static_cast<int64_t>(owner_mismatch_detected_count);
		streaming_metrics["instance_pipeline_owner_mismatch_forced_invalidation"] =
				static_cast<int64_t>(owner_mismatch_forced_invalidation_count);
		renderer->get_performance_state().metrics.streaming_state = streaming_metrics;
	};

	streaming_state.current_streaming_system->begin_frame();

	Projection cull_projection = renderer->build_cull_projection(p_render_data, p_projection);
	renderer->validate_cull_projection_contract(p_render_data, p_projection, cull_projection,
			"render_streaming_orchestrator::render_streaming_frame");

	// Update streaming based on camera transform and projection.
	Transform3D streaming_camera_transform = p_camera_to_world_transform;

	GaussianSplatSceneDirector *director = GaussianSplatSceneDirector::get_singleton();
	GaussianStreamingSystem *streaming_system = streaming_state.current_streaming_system.ptr();
	if (!streaming_system) {
		const StreamingReadinessState readiness_state = StreamingReadinessState::MISSING_STREAMING_SYSTEM;
		finalize_streaming_frame(readiness_state, "current_streaming_system.ptr() returned null");
		publish_not_ready_route(readiness_state);
		log_streaming_reset("null_system_ptr");
		streaming_state.current_streaming_system.unref();
		renderer->clear_instance_pipeline_buffers();
		return false;
	}
	String bootstrap_error;
	if (!streaming_system->is_runtime_ready(&bootstrap_error)) {
		const StreamingReadinessState readiness_state = StreamingReadinessState::RUNTIME_BOOTSTRAP_PENDING;
		ERR_PRINT(vformat("[GaussianSplatRenderer] Streaming runtime invalid; resetting system (%s).", bootstrap_error));
		finalize_streaming_frame(readiness_state, bootstrap_error);
		publish_not_ready_route(readiness_state);
		log_streaming_reset("runtime_not_ready");
		streaming_state.current_streaming_system.unref();
		renderer->clear_instance_pipeline_buffers();
		return false;
	}
	instance_pipeline_instance_cache.clear();
	if (director && streaming_system) {
		sync_instance_pipeline_assets(streaming_system);
		const VisibleLODSelection &selection = produce_visible_lod_selection(
				director,
				streaming_system,
				p_camera_to_world_transform.origin);
		instance_pipeline_instance_cache = selection.get_instances();
		consume_visible_lod_selection_for_residency(selection, streaming_system, trace_enabled);
	}
	// Some callsites (runtime force-sort, world/static-chunk rendering) can execute
	// without SceneDirector instances. In those explicit opt-in paths, inject a
	// synthetic primary instance so cull/sort buffer contracts remain valid.
	if (p_allow_runtime_fallback_instance &&
			instance_pipeline_instance_cache.is_empty() &&
			renderer->get_scene_state().gaussian_data.is_valid() &&
			renderer->get_scene_state().gaussian_data->get_count() > 0) {
		InstanceDataGPU fallback_instance = {};
		fallback_instance.rotation[3] = 1.0f;
		fallback_instance.inv_rotation[3] = 1.0f;
		fallback_instance.translation_scale[3] = 1.0f;
		fallback_instance.params[0] = 1.0f;
		fallback_instance.params[1] = 1.0f;
		fallback_instance.params[2] = 1.0f;
		fallback_instance.params[3] = 0.0f;
		fallback_instance.ids[0] = 0u;
		fallback_instance.ids[1] = GS_INSTANCE_FLAG_ROTATION_IDENTITY |
				GS_INSTANCE_FLAG_SCALE_IDENTITY |
				GS_INSTANCE_FLAG_TRANSLATION_ZERO;
		fallback_instance.lod[0] = 0;
		fallback_instance.lod[1] = 0;
		fallback_instance.wind_params[0] = 0.0f;
		fallback_instance.wind_params[1] = 0.0f;
		fallback_instance.wind_params[2] = 0.0f;
		fallback_instance.wind_params[3] = 1.0f;
		instance_pipeline_instance_cache.push_back(fallback_instance);
		if (trace_enabled) {
			GaussianSplatting::debug_trace_record_event("instance_pipeline",
					"Injected primary fallback instance for empty-instance streaming path",
					false);
		}
	}
	const uint32_t registered_assets_with_data = streaming_system->get_registered_asset_count_with_data();
	if (!instance_pipeline_instance_cache.is_empty() &&
			registered_assets_with_data == 0 &&
			!p_allow_runtime_fallback_instance) {
		const StreamingReadinessState readiness_state = StreamingReadinessState::REGISTRATION_PENDING;
		ERR_PRINT("[GaussianSplatRenderer] Streaming bootstrap produced instances but no registered streaming assets with data; resetting streaming system.");
		finalize_streaming_frame(readiness_state,
				vformat("instances=%d registered_assets_with_data=%d",
						instance_pipeline_instance_cache.size(),
						registered_assets_with_data));
		publish_not_ready_route(readiness_state);
		log_streaming_reset("instances_without_registered_assets");
		streaming_state.current_streaming_system.unref();
		renderer->clear_instance_pipeline_buffers();
		return false;
	}
	renderer->update_instance_buffer(instance_pipeline_instance_cache);

	streaming_state.current_streaming_system->set_chunk_radius_multiplier(
			renderer->get_cull_radius_multiplier() * renderer->get_cull_frustum_plane_slack());

	streaming_state.current_streaming_system->update_streaming(streaming_camera_transform, cull_projection);

	uint64_t instance_generation = 0;
	if (GaussianSplatSceneDirector *generation_director = GaussianSplatSceneDirector::get_singleton()) {
		instance_generation = generation_director->get_instance_generation_for_renderer(renderer);
	}
	uint64_t atlas_generation = 0;
	if (streaming_state.current_streaming_system.is_valid()) {
		atlas_generation = streaming_state.current_streaming_system->get_atlas_generation();
	}
	const uint64_t base_content_generation = _mix_content_generation(instance_generation, atlas_generation);
	ResourceState &resource_state = renderer->get_resource_state();
	resource_state.instance_pipeline_content_generation = base_content_generation;
	bool instance_buffers_atlas_ready = false;
	bool instance_buffers_cull_ready = false;
	bool instance_buffers_sort_ready = false;
	bool instance_buffers_raster_ready = false;
	bool instance_buffers_atlas_required = false;
	uint32_t instance_pipeline_instance_count = 0;
	uint32_t instance_pipeline_dispatch_chunk_count = 0;
	uint32_t instance_pipeline_max_visible_chunks = 0;
	uint32_t instance_pipeline_max_visible_splats = 0;
	uint32_t instance_pipeline_max_chunk_splats = 0;
	StreamingReadinessState stream_readiness_state = StreamingReadinessState::MISSING_STREAMING_SYSTEM;

	if (streaming_system) {
		InstancePipelineBuffers buffers = renderer->instance_pipeline_buffers;
		const GlobalAtlasState &atlas_state = streaming_system->get_global_atlas_state();
		const bool quantization_required = streaming_system->is_per_chunk_quantization_enabled();

		buffers.atlas_gaussian_buffer = atlas_state.atlas_gaussian_buffer;
		buffers.atlas_gaussian_count = atlas_state.atlas_gaussian_count;
		if (trace_enabled) {
			GaussianSplatting::debug_trace_record_event("instance_pipeline",
					vformat("AtlasState gaussian_count=%d buffer_valid=%s",
								atlas_state.atlas_gaussian_count,
								atlas_state.atlas_gaussian_buffer.is_valid() ? "YES" : "no"),
					false);
		}
			buffers.asset_meta_buffer = atlas_state.asset_meta_buffer;
			buffers.chunk_meta_buffer = atlas_state.chunk_meta_buffer;
			buffers.asset_chunk_index_buffer = atlas_state.asset_chunk_index_buffer;
			static int instance_buffer_diag_counter = 0;
			if (++instance_buffer_diag_counter <= 20) {
				print_line(vformat("[Streaming DIAG] instance buffer handoff: chunk_meta_rid=%d asset_meta_rid=%d chunk_index_rid=%d atlas_gen=%d",
						buffers.chunk_meta_buffer.get_id(),
						buffers.asset_meta_buffer.get_id(),
						buffers.asset_chunk_index_buffer.get_id(),
						int64_t(atlas_state.atlas_generation)));
			}
			buffers.quantization_required = quantization_required;
			buffers.quantization_buffer = quantization_required ? atlas_state.quantization_buffer : RID();

		const uint32_t instance_count = buffers.instance_count;
		instance_pipeline_instance_count = instance_count;
		buffers.dispatch_chunk_count = streaming_system->get_max_chunk_count_per_asset();
		instance_pipeline_dispatch_chunk_count = buffers.dispatch_chunk_count;
		buffers.max_chunk_splats = streaming_system->get_max_chunk_splats();
		instance_pipeline_max_chunk_splats = buffers.max_chunk_splats;

		const PerformanceSettings &perf_settings = renderer->get_performance_settings();
		uint64_t max_visible_splats_u64 = perf_settings.max_splats > 0
				? uint64_t(perf_settings.max_splats)
				: uint64_t(atlas_state.atlas_gaussian_count);

		// Recompute the budget every frame from current scene state.
		// Do not reuse stale values from prior renderer/scene frames.
		if (instance_count > 1 && buffers.dispatch_chunk_count > 0 && buffers.max_chunk_splats > 0) {
			const uint64_t needed = uint64_t(instance_count)
					* uint64_t(buffers.dispatch_chunk_count)
					* uint64_t(buffers.max_chunk_splats);
			max_visible_splats_u64 = MAX(max_visible_splats_u64, needed);
		}
		const uint64_t sort_cap = g_gpu_sorting_config.max_sort_elements > 0
				? uint64_t(g_gpu_sorting_config.max_sort_elements)
				: uint64_t(UINT32_MAX);
		max_visible_splats_u64 = MIN(max_visible_splats_u64, sort_cap);
		uint32_t max_visible_splats = max_visible_splats_u64 > UINT32_MAX
				? UINT32_MAX
				: uint32_t(max_visible_splats_u64);
		buffers.max_visible_splats = max_visible_splats;

		uint64_t max_visible_chunks_u64 = uint64_t(instance_count) * uint64_t(buffers.dispatch_chunk_count);
		uint32_t max_visible_chunks = max_visible_chunks_u64 > UINT32_MAX ? UINT32_MAX : uint32_t(max_visible_chunks_u64);
		if (max_visible_chunks > 0 && max_visible_splats > 0) {
			max_visible_chunks = MIN(max_visible_chunks, max_visible_splats);
		}
		buffers.max_visible_chunks = max_visible_chunks;
		instance_pipeline_max_visible_chunks = buffers.max_visible_chunks;

		ResourceState &resource_state_mut = renderer->get_resource_state();
		RenderingDevice *rd = renderer->get_device_state().rd;
		if (rd) {
			const auto remediate_instance_pipeline_buffer = [&](RID &r_buffer, uint32_t *r_capacity, const char *p_label) {
				const OwnerMismatchRemediationResult remediation =
						_remediate_instance_pipeline_buffer_owner_mismatch(
								renderer, rd, r_buffer, r_capacity, p_label);
				if (remediation.mismatch_detected) {
					owner_mismatch_detected_count++;
				}
				if (remediation.forced_invalidation) {
					owner_mismatch_forced_invalidation_count++;
				}
			};

			remediate_instance_pipeline_buffer(resource_state_mut.instance_visible_chunk_buffer,
					&resource_state_mut.instance_visible_chunk_capacity,
					"instance_visible_chunk_buffer");
			remediate_instance_pipeline_buffer(resource_state_mut.instance_splat_ref_buffer,
					&resource_state_mut.instance_splat_ref_capacity,
					"instance_splat_ref_buffer");
			remediate_instance_pipeline_buffer(resource_state_mut.instance_counter_buffer, nullptr, "instance_counter_buffer");
			remediate_instance_pipeline_buffer(resource_state_mut.instance_chunk_dispatch_buffer, nullptr, "instance_chunk_dispatch_buffer");
			remediate_instance_pipeline_buffer(resource_state_mut.instance_indirect_count_buffer, nullptr, "instance_indirect_count_buffer");
			remediate_instance_pipeline_buffer(resource_state_mut.instance_count_buffer, nullptr, "instance_count_buffer");
		}
		if (rd && buffers.max_visible_chunks > 0) {
			if (!resource_state_mut.instance_visible_chunk_buffer.is_valid() ||
					resource_state_mut.instance_visible_chunk_capacity < buffers.max_visible_chunks) {
				renderer->free_owned_resource(rd, resource_state_mut.instance_visible_chunk_buffer);
				const uint32_t new_capacity = buffers.max_visible_chunks;
				const uint32_t buffer_size = new_capacity * sizeof(VisibleChunkRefGPU);
				resource_state_mut.instance_visible_chunk_buffer = rd->storage_buffer_create(buffer_size);
				if (resource_state_mut.instance_visible_chunk_buffer.is_valid()) {
					rd->set_resource_name(resource_state_mut.instance_visible_chunk_buffer, "GS_InstanceVisibleChunks");
					renderer->track_resource_owner(resource_state_mut.instance_visible_chunk_buffer, rd);
					resource_state_mut.instance_visible_chunk_capacity = new_capacity;
				} else {
					resource_state_mut.instance_visible_chunk_capacity = 0;
				}
			}
			buffers.visible_chunk_buffer = resource_state_mut.instance_visible_chunk_buffer;
		}

		if (rd && buffers.max_visible_splats > 0) {
			if (!resource_state_mut.instance_splat_ref_buffer.is_valid() ||
					resource_state_mut.instance_splat_ref_capacity < buffers.max_visible_splats) {
				renderer->free_owned_resource(rd, resource_state_mut.instance_splat_ref_buffer);
				const uint32_t new_capacity = buffers.max_visible_splats;
				const uint32_t buffer_size = new_capacity * sizeof(SplatRefGPU);
				resource_state_mut.instance_splat_ref_buffer = rd->storage_buffer_create(buffer_size);
				if (resource_state_mut.instance_splat_ref_buffer.is_valid()) {
					rd->set_resource_name(resource_state_mut.instance_splat_ref_buffer, "GS_InstanceSplatRefs");
					renderer->track_resource_owner(resource_state_mut.instance_splat_ref_buffer, rd);
					resource_state_mut.instance_splat_ref_capacity = new_capacity;
				} else {
					resource_state_mut.instance_splat_ref_capacity = 0;
				}
			}
			buffers.splat_ref_buffer = resource_state_mut.instance_splat_ref_buffer;
		}

		if (rd) {
			if (!resource_state_mut.instance_counter_buffer.is_valid()) {
				resource_state_mut.instance_counter_buffer = rd->storage_buffer_create(sizeof(uint32_t) * 2);
				if (resource_state_mut.instance_counter_buffer.is_valid()) {
					rd->set_resource_name(resource_state_mut.instance_counter_buffer, "GS_InstanceCounters");
					renderer->track_resource_owner(resource_state_mut.instance_counter_buffer, rd);
				}
			}
			buffers.counter_buffer = resource_state_mut.instance_counter_buffer;
		}

		if (rd) {
			if (!resource_state_mut.instance_chunk_dispatch_buffer.is_valid()) {
				resource_state_mut.instance_chunk_dispatch_buffer = rd->storage_buffer_create(
						sizeof(uint32_t) * 3, Vector<uint8_t>(), RD::STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT);
				if (resource_state_mut.instance_chunk_dispatch_buffer.is_valid()) {
					rd->set_resource_name(resource_state_mut.instance_chunk_dispatch_buffer, "GS_InstanceChunkDispatch");
					renderer->track_resource_owner(resource_state_mut.instance_chunk_dispatch_buffer, rd);
				}
			}
			buffers.chunk_dispatch_buffer = resource_state_mut.instance_chunk_dispatch_buffer;
		}

		if (rd) {
			if (!resource_state_mut.instance_indirect_count_buffer.is_valid()) {
				resource_state_mut.instance_indirect_count_buffer =
						rd->storage_buffer_create(sizeof(GaussianSplatting::IndirectDispatchLayout),
								Vector<uint8_t>(), RD::STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT);
				if (resource_state_mut.instance_indirect_count_buffer.is_valid()) {
					rd->set_resource_name(resource_state_mut.instance_indirect_count_buffer, "GS_InstanceIndirectCount");
					renderer->track_resource_owner(resource_state_mut.instance_indirect_count_buffer, rd);
				}
			}
			buffers.indirect_count_buffer = resource_state_mut.instance_indirect_count_buffer;
		}
		if (rd) {
			if (!resource_state_mut.instance_count_buffer.is_valid()) {
				resource_state_mut.instance_count_buffer =
						rd->storage_buffer_create(sizeof(GaussianSplatting::IndirectDispatchLayout));
				if (resource_state_mut.instance_count_buffer.is_valid()) {
					rd->set_resource_name(resource_state_mut.instance_count_buffer, "GS_InstanceCount");
					renderer->track_resource_owner(resource_state_mut.instance_count_buffer, rd);
				}
			}
			buffers.instance_count_buffer = resource_state_mut.instance_count_buffer;
		}

		GPUSortingPipeline *sorting_pipeline = renderer->get_subsystem_state().sorting_pipeline.ptr();
		if (sorting_pipeline) {
			// Instance pipeline may need more sort capacity than the per-instance
			// max_splats budget used to initialize the sorter. Rebuild the sorter
			// with the actual required capacity if it is insufficient.
			const uint32_t required_capacity = buffers.max_visible_splats;
			const uint32_t current_capacity = sorting_pipeline->get_max_elements();
			if (required_capacity > current_capacity && required_capacity > 0) {
				sorting_pipeline->rebuild_sorter(required_capacity);
			}
			sorting_pipeline->ensure_buffers(buffers.max_visible_splats);
			SortBufferHandles handles = sorting_pipeline->get_buffer_handles();
			if (handles.valid) {
				buffers.sort_key_buffer = handles.keys_buffer;
				buffers.sort_value_buffer = handles.indices_buffer;
				// Clamp max_visible_splats to sort buffer capacity to prevent
				// GPU OOB writes in depth_compute and OOB reads in rasterizer.
				if (handles.capacity > 0 && buffers.max_visible_splats > handles.capacity) {
					WARN_PRINT_ONCE(vformat("[GaussianSplat] Sort buffer capacity (%d) < required (%d); clamping max_visible_splats.", handles.capacity, buffers.max_visible_splats));
					buffers.max_visible_splats = handles.capacity;
				}
				if (buffers.max_visible_chunks > 0 && buffers.max_visible_splats > 0 &&
						buffers.max_visible_chunks > buffers.max_visible_splats) {
					buffers.max_visible_chunks = buffers.max_visible_splats;
				}
			}
		}
		instance_pipeline_max_visible_splats = buffers.max_visible_splats;
		instance_pipeline_max_visible_chunks = buffers.max_visible_chunks;

		const bool atlas_buffers_required = !instance_pipeline_assets_cache.is_empty();
		instance_buffers_atlas_required = atlas_buffers_required;
		const bool atlas_ready = GaussianSplatting::InstancePipelineContract::has_atlas_buffers(buffers);
		const bool cull_ready = GaussianSplatting::InstancePipelineContract::has_cull_buffers(buffers);
		const bool sort_ready = GaussianSplatting::InstancePipelineContract::has_sort_buffers(buffers);
		const bool raster_ready = GaussianSplatting::InstancePipelineContract::has_raster_buffers(buffers);
			stream_readiness_state = StreamingReadinessState::READY;
			if (!atlas_ready) {
				stream_readiness_state = StreamingReadinessState::MISSING_ATLAS_INPUTS;
		} else if (!cull_ready) {
			stream_readiness_state = StreamingReadinessState::MISSING_CULL_INPUTS;
		} else if (!sort_ready) {
			stream_readiness_state = StreamingReadinessState::MISSING_SORT_INPUTS;
			} else if (!raster_ready) {
				stream_readiness_state = StreamingReadinessState::MISSING_RASTER_INPUTS;
			}
			const InvariantViolationReason atlas_violation_reason = atlas_buffers_required
					? GaussianSplatting::InstancePipelineContract::first_atlas_violation(buffers)
					: InvariantViolationReason::NONE;
		const InvariantViolationReason cull_violation_reason = cull_ready
				? InvariantViolationReason::NONE
				: GaussianSplatting::InstancePipelineContract::first_cull_violation(buffers);
		const InvariantViolationReason sort_violation_reason = sort_ready
				? InvariantViolationReason::NONE
					: GaussianSplatting::InstancePipelineContract::first_sort_violation(buffers);
			const InvariantViolationReason raster_violation_reason = raster_ready
					? InvariantViolationReason::NONE
					: GaussianSplatting::InstancePipelineContract::first_raster_violation(buffers);
			instance_buffers_atlas_ready = atlas_ready;
		instance_buffers_cull_ready = cull_ready;
		instance_buffers_sort_ready = sort_ready;
		instance_buffers_raster_ready = raster_ready;

		if (trace_enabled) {
			GaussianSplatting::debug_trace_record_event("instance_pipeline",
					vformat("InstanceBufferCheck atlas=%s cull=%s sort=%s raster=%s",
								atlas_ready ? "YES" : "no",
								cull_ready ? "YES" : "no",
								sort_ready ? "YES" : "no",
								raster_ready ? "YES" : "no"),
					false);
		}
		if (trace_enabled && !atlas_ready) {
			GaussianSplatting::debug_trace_record_event("instance_pipeline",
					vformat("InstanceBufferCheck atlas_gaussian=%s asset_meta=%s chunk_meta=%s asset_chunk_idx=%s quant=%s(req=%s)",
								buffers.atlas_gaussian_buffer.is_valid() ? "Y" : "N",
								buffers.asset_meta_buffer.is_valid() ? "Y" : "N",
								buffers.chunk_meta_buffer.is_valid() ? "Y" : "N",
								buffers.asset_chunk_index_buffer.is_valid() ? "Y" : "N",
								buffers.quantization_buffer.is_valid() ? "Y" : "N",
								quantization_required ? "YES" : "no"),
					true);
		}
		if (trace_enabled && !cull_ready) {
			GaussianSplatting::debug_trace_record_event("instance_pipeline",
					vformat("InstanceBufferCheck cull inst_buf=%s vis_chunk=%s counter=%s inst_cnt=%d chunk_cnt=%d max_vis_chunks=%d",
								buffers.instance_buffer.is_valid() ? "Y" : "N",
								buffers.visible_chunk_buffer.is_valid() ? "Y" : "N",
								buffers.counter_buffer.is_valid() ? "Y" : "N",
								buffers.instance_count, buffers.dispatch_chunk_count, buffers.max_visible_chunks),
					true);
		}
		if (trace_enabled && !sort_ready) {
			GaussianSplatting::debug_trace_record_event("instance_pipeline",
					vformat("InstanceBufferCheck sort splat_ref=%s sort_key=%s sort_val=%s chunk_dispatch=%s indirect=%s count=%s max_vis_splats=%d max_chunk_splats=%d",
								buffers.splat_ref_buffer.is_valid() ? "Y" : "N",
								buffers.sort_key_buffer.is_valid() ? "Y" : "N",
								buffers.sort_value_buffer.is_valid() ? "Y" : "N",
								buffers.chunk_dispatch_buffer.is_valid() ? "Y" : "N",
								buffers.indirect_count_buffer.is_valid() ? "Y" : "N",
								buffers.instance_count_buffer.is_valid() ? "Y" : "N",
								buffers.max_visible_splats, buffers.max_chunk_splats),
					true);
		}
		if (trace_enabled && !raster_ready) {
			GaussianSplatting::debug_trace_record_event("instance_pipeline",
					vformat("InstanceBufferCheck raster atlas=%s instance=%s splat_ref=%s count=%s quant=%s(req=%s)",
								buffers.atlas_gaussian_buffer.is_valid() ? "Y" : "N",
								buffers.instance_buffer.is_valid() ? "Y" : "N",
								buffers.splat_ref_buffer.is_valid() ? "Y" : "N",
								buffers.instance_count_buffer.is_valid() ? "Y" : "N",
								buffers.quantization_buffer.is_valid() ? "Y" : "N",
								quantization_required ? "YES" : "no"),
					true);
		}

		InvariantViolation activation_violation;
		if (atlas_violation_reason != InvariantViolationReason::NONE) {
			activation_violation = GaussianSplatting::InstancePipelineContract::make_violation(atlas_violation_reason);
		} else if (cull_violation_reason != InvariantViolationReason::NONE) {
			activation_violation = GaussianSplatting::InstancePipelineContract::make_violation(cull_violation_reason);
		} else if (sort_violation_reason != InvariantViolationReason::NONE) {
			activation_violation = GaussianSplatting::InstancePipelineContract::make_violation(sort_violation_reason);
		} else if (raster_violation_reason != InvariantViolationReason::NONE) {
			activation_violation = GaussianSplatting::InstancePipelineContract::make_violation(raster_violation_reason);
		} else {
			activation_violation = GaussianSplatting::InstancePipelineContract::evaluate_streaming_activation(
					buffers, atlas_buffers_required);
		}
		set_instance_invariant_status(activation_violation);
		const bool fallback_warmup_allowed =
				p_allow_runtime_fallback_instance &&
				registered_assets_with_data == 0 &&
				!_is_impossible_streaming_activation_violation(
						instance_pipeline_instance_count,
						registered_assets_with_data,
						p_allow_runtime_fallback_instance,
						activation_violation);

		const auto emit_invariant_violation_diagnostic = [&](InvariantViolationReason p_reason) {
			if (p_reason == InvariantViolationReason::NONE) {
				return;
			}
			static bool emitted_reasons[static_cast<int>(InvariantViolationReason::COUNT)] = {};
			const int reason_index = static_cast<int>(p_reason);
			bool should_emit_log = true;
			if (reason_index >= 0 && reason_index < static_cast<int>(InvariantViolationReason::COUNT)) {
				should_emit_log = !emitted_reasons[reason_index];
				emitted_reasons[reason_index] = true;
			}
			const char *route = GaussianSplatting::InstancePipelineContract::get_violation_route(p_reason);
			const char *violation_class_name = GaussianSplatting::InstancePipelineContract::get_violation_class_name(
					GaussianSplatting::InstancePipelineContract::get_violation_class(p_reason));
			const char *reason_name = GaussianSplatting::InstancePipelineContract::get_violation_reason_name(p_reason);
			const bool expected_zero_instance_warmup_reason =
					p_reason == InvariantViolationReason::CULL_INSTANCE_COUNT_ZERO;
			if (should_emit_log) {
				if (expected_zero_instance_warmup_reason || fallback_warmup_allowed) {
					GS_LOG_RENDERER_DEBUG(vformat("[GaussianSplatRenderer] Instance pipeline warmup invariant route=%s class=%s reason=%s inst=%d registered_assets=%d atlas_required=%s",
							route,
							violation_class_name,
							reason_name,
							instance_pipeline_instance_count,
							registered_assets_with_data,
							atlas_buffers_required ? "yes" : "no"));
				} else {
					ERR_PRINT(vformat("[GaussianSplatRenderer] Instance pipeline invariant violation route=%s class=%s reason=%s inst=%d registered_assets=%d atlas_required=%s",
							route,
							violation_class_name,
							reason_name,
							instance_pipeline_instance_count,
							registered_assets_with_data,
							atlas_buffers_required ? "yes" : "no"));
				}
			}
			if (trace_enabled) {
				GaussianSplatting::debug_trace_record_event("instance_pipeline",
						vformat("InvariantViolation route=%s class=%s reason=%s", route, violation_class_name, reason_name),
						true);
			}
		};
		emit_invariant_violation_diagnostic(atlas_violation_reason);
		emit_invariant_violation_diagnostic(cull_violation_reason);
		emit_invariant_violation_diagnostic(sort_violation_reason);
		emit_invariant_violation_diagnostic(raster_violation_reason);

			if (_is_debug_or_test_invariant_hard_fail_enabled() &&
					_is_impossible_streaming_activation_violation(
							instance_pipeline_instance_count,
							registered_assets_with_data,
							p_allow_runtime_fallback_instance,
							activation_violation)) {
				renderer->instance_pipeline_buffers = buffers;
				renderer->instance_pipeline_buffers_valid = false;
				resource_state.instance_pipeline_content_generation = _mix_content_generation(
						base_content_generation,
						_compute_instance_pipeline_resource_fingerprint(resource_state_mut, buffers));
				StreamingReadinessState hard_fail_state = stream_readiness_state;
				if (hard_fail_state == StreamingReadinessState::READY) {
					if (atlas_violation_reason != InvariantViolationReason::NONE) {
						hard_fail_state = StreamingReadinessState::MISSING_ATLAS_INPUTS;
					} else if (cull_violation_reason != InvariantViolationReason::NONE) {
						hard_fail_state = StreamingReadinessState::MISSING_CULL_INPUTS;
					} else if (sort_violation_reason != InvariantViolationReason::NONE) {
						hard_fail_state = StreamingReadinessState::MISSING_SORT_INPUTS;
					} else {
						hard_fail_state = StreamingReadinessState::MISSING_RASTER_INPUTS;
					}
				}
				publish_not_ready_route(hard_fail_state);
				finalize_streaming_frame(
						hard_fail_state,
						vformat("hard_fail invariant route=%s class=%s reason=%s",
								instance_invariant_route,
								instance_invariant_class,
								instance_invariant_reason));
				ERR_FAIL_V_MSG(false, vformat("[GaussianSplatRenderer] Hard-fail: impossible instance pipeline activation invariant violation route=%s class=%s reason=%s",
						instance_invariant_route,
						instance_invariant_class,
					instance_invariant_reason));
		}

			if (stream_readiness_state == StreamingReadinessState::MISSING_ATLAS_INPUTS) {
				if (atlas_buffers_required) {
					WARN_PRINT_ONCE("[GaussianSplatRenderer] Instance pipeline requires global atlas buffers; streaming not ready.");
				}
			renderer->instance_pipeline_buffers = buffers;
			renderer->instance_pipeline_buffers_valid = false;
		} else if (stream_readiness_state == StreamingReadinessState::READY) {
			if (trace_enabled) {
				GaussianSplatting::debug_trace_record_event("instance_pipeline",
						"InstanceBufferCheck READY - set_instance_pipeline_buffers",
						false);
			}
			renderer->set_instance_pipeline_buffers(buffers);
		} else {
			if (trace_enabled) {
				GaussianSplatting::debug_trace_record_event("instance_pipeline",
						vformat("InstanceBufferCheck NOT READY state=%s",
								String(_streaming_readiness_state_token(stream_readiness_state))),
						true);
			}
			renderer->instance_pipeline_buffers = buffers;
			renderer->instance_pipeline_buffers_valid = false;
		}
		resource_state.instance_pipeline_content_generation = _mix_content_generation(
				base_content_generation,
				_compute_instance_pipeline_resource_fingerprint(resource_state_mut, buffers));
	} else {
		renderer->instance_pipeline_buffers_valid = false;
		resource_state.instance_pipeline_content_generation = base_content_generation;
	}

	// Reset legacy streaming buffers, but do not publish a pre-cull visible count.
	// The active cull->sort path is the authoritative source for frame visibility.
	renderer->_reset_legacy_streaming_data_path_state();

	// Legacy instance transforms removed; use the view transform directly.
	Transform3D effective_view_transform = p_world_to_camera_transform;

	if (!renderer->instance_pipeline_buffers_valid && stream_readiness_state == StreamingReadinessState::READY) {
		stream_readiness_state = StreamingReadinessState::MISSING_RASTER_INPUTS;
	}
	const bool stream_ready = stream_readiness_state == StreamingReadinessState::READY &&
			renderer->instance_pipeline_buffers_valid;
	if (trace_enabled && stream_ready) {
		GaussianSplatting::debug_trace_record_event("instance_pipeline",
				"InstancePipeline readiness=READY",
				false);
	}
	if (!stream_ready) {
		String readiness_detail;
		if (trace_enabled) {
			Dictionary chunk_stats = streaming_state.current_streaming_system->get_chunk_culling_stats();
			const int64_t total_chunks = int64_t(chunk_stats.get("total_chunks", int64_t(0)));
			const int64_t visible_chunks = int64_t(chunk_stats.get("visible_chunks", int64_t(0)));
			const int64_t loaded_chunks = int64_t(chunk_stats.get("loaded_chunks", int64_t(0)));
				readiness_detail = vformat("state=%s atlas=%s cull=%s sort=%s raster=%s atlas_required=%s inst=%d chunk_count=%d max_vis_chunks=%d max_vis_splats=%d max_chunk_splats=%d chunks=%d/%d (visible=%d)",
						String(_streaming_readiness_state_token(stream_readiness_state)),
						instance_buffers_atlas_ready ? "Y" : "N",
					instance_buffers_cull_ready ? "Y" : "N",
					instance_buffers_sort_ready ? "Y" : "N",
					instance_buffers_raster_ready ? "Y" : "N",
					instance_buffers_atlas_required ? "Y" : "N",
					instance_pipeline_instance_count,
					instance_pipeline_dispatch_chunk_count,
					instance_pipeline_max_visible_chunks,
					instance_pipeline_max_visible_splats,
					instance_pipeline_max_chunk_splats,
					loaded_chunks,
						total_chunks,
						visible_chunks);
				GaussianSplatting::debug_trace_record_event("instance_pipeline",
						vformat("InstancePipeline readiness NOT READY %s invariant_route=%s invariant_class=%s invariant_reason=%s",
								readiness_detail,
								instance_invariant_route,
								instance_invariant_class,
								instance_invariant_reason),
						true);
			}
		publish_not_ready_route(stream_readiness_state);
		finalize_streaming_frame(stream_readiness_state, readiness_detail);
		return false;
	}

	renderer->_run_cull_sort_pipeline_frame(p_render_data, effective_view_transform, p_projection, p_render_projection,
			p_render_buffers, stream_ready,
			"Cull skipped: streaming data unavailable",
			"Sort skipped: streaming data unavailable",
			RenderFallbackReason::STREAMING_DATA_UNAVAILABLE,
			RenderFallbackReason::STREAMING_DATA_UNAVAILABLE,
			true, false);

	finalize_streaming_frame(StreamingReadinessState::READY);
	return true;
}

void RenderStreamingOrchestrator::tick_streaming_only(const Transform3D &p_camera_to_world_transform, const Projection &p_projection) {
	StreamingState &streaming_state = renderer->get_streaming_state();
	if (!streaming_state.current_streaming_system.is_valid()) {
		if (!ensure_instance_streaming_system()) {
			return;
		}
	}

	GaussianStreamingSystem *streaming_system = streaming_state.current_streaming_system.ptr();
	if (!streaming_system) {
		return;
	}
	String bootstrap_error;
	if (!streaming_system->is_runtime_ready(&bootstrap_error)) {
		ERR_PRINT(vformat("[GaussianSplatRenderer] Streaming tick skipped: invalid runtime bootstrap (%s).", bootstrap_error));
		const uint64_t system_id = streaming_state.current_streaming_system.is_valid()
				? uint64_t(reinterpret_cast<uintptr_t>(streaming_state.current_streaming_system.ptr()))
				: 0u;
		ERR_PRINT(vformat("[GaussianSplatRenderer] Streaming system reset (tick_runtime_not_ready), previous_system=%s",
				String::num_uint64(system_id)));
		streaming_state.current_streaming_system.unref();
		renderer->clear_instance_pipeline_buffers();
		return;
	}

	streaming_system->begin_frame();

	GaussianSplatSceneDirector *director = GaussianSplatSceneDirector::get_singleton();
	if (director) {
		sync_instance_pipeline_assets(streaming_system);
		const VisibleLODSelection &selection = produce_visible_lod_selection(
				director,
				streaming_system,
				p_camera_to_world_transform.origin);
		instance_pipeline_instance_cache = selection.get_instances();
		const uint32_t registered_assets_with_data = streaming_system->get_registered_asset_count_with_data();
		if (!instance_pipeline_instance_cache.is_empty() && registered_assets_with_data == 0) {
			ERR_PRINT("[GaussianSplatRenderer] Streaming tick found instances but no registered streaming assets with data; resetting streaming system.");
			streaming_system->end_frame();
			const uint64_t system_id = streaming_state.current_streaming_system.is_valid()
					? uint64_t(reinterpret_cast<uintptr_t>(streaming_state.current_streaming_system.ptr()))
					: 0u;
			ERR_PRINT(vformat("[GaussianSplatRenderer] Streaming system reset (tick_instances_without_registered_assets), previous_system=%s",
					String::num_uint64(system_id)));
			streaming_state.current_streaming_system.unref();
			renderer->clear_instance_pipeline_buffers();
			return;
		}

		consume_visible_lod_selection_for_residency(selection, streaming_system, false);
	}

	streaming_system->set_chunk_radius_multiplier(
			renderer->get_cull_radius_multiplier() * renderer->get_cull_frustum_plane_slack());
	const Projection cull_projection = renderer->build_cull_projection(nullptr, p_projection);
	renderer->validate_cull_projection_contract(nullptr, p_projection, cull_projection,
			"render_streaming_orchestrator::tick_streaming_only");
	streaming_system->update_streaming(p_camera_to_world_transform, cull_projection);
	renderer->get_frame_state().visible_splat_count.store(
			streaming_system->get_visible_count(), std::memory_order_release);

	streaming_system->end_frame();

	Dictionary streaming_metrics = streaming_system->get_streaming_analytics();
	_apply_static_layout_fallback_diagnostics(streaming_metrics,
			static_layout_fallback_total,
			static_layout_fallback_io_total,
			static_layout_fallback_primary_total,
			static_layout_fallback_reason_counts,
			static_layout_fallback_category_counts,
			static_layout_fallback_last_usage,
			static_layout_fallback_last_reason,
			static_layout_fallback_last_reason_category,
			static_layout_fallback_last_context,
			static_layout_fallback_last_detail);
	auto &perf_metrics = renderer->get_performance_state().metrics;
	const auto &frame_state = renderer->get_frame_state();
	const auto &debug_state = renderer->get_debug_state();

	bool stage_metrics_valid = debug_state.last_stage_metrics_valid;
	float stage_cull_time_ms = perf_metrics.culling_time_ms;
	float stage_sort_time_ms = frame_state.sort_time_ms;
	float stage_raster_time_ms = frame_state.render_time_ms;
	float stage_composite_time_ms = 0.0f;
	bool stage_composite_executed = false;
	if (stage_metrics_valid) {
		const auto &stage_metrics = debug_state.last_stage_metrics;
		stage_cull_time_ms = stage_metrics.cull.cull_time_ms;
		stage_sort_time_ms = stage_metrics.sort.sort_time_ms;
		stage_raster_time_ms = stage_metrics.raster.render_time_ms;
		stage_composite_time_ms = stage_metrics.composite_time_ms;
		stage_composite_executed = stage_metrics.composite_executed;
	}

	streaming_metrics["stage_metrics_valid"] = stage_metrics_valid;
	streaming_metrics["stage_cull_time_ms"] = stage_cull_time_ms;
	streaming_metrics["stage_sort_time_ms"] = stage_sort_time_ms;
	streaming_metrics["stage_raster_time_ms"] = stage_raster_time_ms;
	streaming_metrics["stage_composite_time_ms"] = stage_composite_time_ms;
	streaming_metrics["stage_composite_executed"] = stage_composite_executed;
	streaming_metrics["cull_ms"] = stage_cull_time_ms;
	streaming_metrics["sort_ms"] = stage_sort_time_ms;
	streaming_metrics["raster_ms"] = stage_raster_time_ms;
	streaming_metrics["composite_ms"] = stage_composite_time_ms;
	streaming_metrics["frame_sort_time_ms"] = frame_state.sort_time_ms;
	streaming_metrics["frame_render_time_ms"] = frame_state.render_time_ms;

	streaming_metrics["gpu_frame_time_ms"] = perf_metrics.gpu_frame_time_ms;
	streaming_metrics["gpu_tile_binning_time_ms"] = perf_metrics.gpu_tile_binning_time_ms;
	streaming_metrics["gpu_tile_prefix_time_ms"] = perf_metrics.gpu_tile_prefix_time_ms;
	streaming_metrics["gpu_tile_raster_time_ms"] = perf_metrics.gpu_tile_raster_time_ms;
	streaming_metrics["gpu_tile_resolve_time_ms"] = perf_metrics.gpu_tile_resolve_time_ms;
	streaming_metrics["gpu_timing_frame_serial"] = static_cast<int64_t>(perf_metrics.gpu_timing_frame_serial);
	streaming_metrics["gpu_timing_frames_behind"] = static_cast<int64_t>(perf_metrics.gpu_timing_frames_behind);
	streaming_metrics["gpu_pass_breakdown_available"] = perf_metrics.gpu_tile_binning_time_ms > 0.0f ||
			perf_metrics.gpu_tile_prefix_time_ms > 0.0f ||
			perf_metrics.gpu_tile_raster_time_ms > 0.0f ||
			perf_metrics.gpu_tile_resolve_time_ms > 0.0f;

	perf_metrics.streaming_state = streaming_metrics;
}
