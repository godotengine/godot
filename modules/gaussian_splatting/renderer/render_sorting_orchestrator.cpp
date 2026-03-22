#include "render_sorting_orchestrator.h"

#include "gaussian_splat_renderer.h"
#include "render_debug_state_orchestrator.h"
#include "core/error/error_macros.h"
#include "core/math/math_funcs.h"
#include "core/math/math_defs.h"
#include "core/math/random_pcg.h"
#include "core/math/quaternion.h"
#include "core/templates/sort_array.h"
#include "core/os/os.h"
#include "core/string/ustring.h"
#include "servers/rendering/rendering_device.h"
#include "../interfaces/gpu_culler.h"
#include "../interfaces/gpu_sorting_pipeline.h"
#include "../logger/gs_logger.h"
#include "gpu_buffer_raii.h"
#include "gpu_sorting_config.h"
#include "gpu_sorter.h"
#include "instance_pipeline_contract.h"
#include "render_streaming_orchestrator.h"
#include "sort_benchmark_metrics.h"
#include "sort_fallback_policy.h"
#include "sorting_config.h"
#include "sorting_contract.h"

namespace {

constexpr float kSortCameraPositionEpsilon = 1e-3f;
constexpr float kSortCameraRotationEpsilon = 1e-3f;

// Ignore sub-millimeter/milliradian jitter to avoid re-sorting every frame.
static bool _is_sort_camera_move_significant(const Transform3D &a, const Transform3D &b) {
	Vector3 delta = a.origin - b.origin;
	if (delta.length_squared() > kSortCameraPositionEpsilon * kSortCameraPositionEpsilon) {
		return true;
	}

	Quaternion qa(a.basis);
	Quaternion qb(b.basis);
	float angle = qa.angle_to(qb);
	return angle > kSortCameraRotationEpsilon;
}

static bool _get_sort_position(const GaussianSplatRenderer &p_renderer, uint32_t p_index, Vector3 &r_position) {
	const auto &streaming_state = p_renderer.get_streaming_state();
	if (streaming_state.use_streamed_data && streaming_state.streamed_indices_are_local &&
			streaming_state.current_streaming_system.is_valid()) {
		uint32_t source_index = 0;
		if (streaming_state.current_streaming_system->map_buffer_index_to_source(p_index, source_index)) {
			const auto &streamed_scene_state = p_renderer.get_scene_state();
			if (streamed_scene_state.gaussian_data.is_valid()) {
				const LocalVector<Gaussian> &gaussians = streamed_scene_state.gaussian_data->get_gaussian_storage();
				if (source_index < static_cast<uint32_t>(gaussians.size())) {
					r_position = gaussians[source_index].position;
					return true;
				}
			}
		}
	}
	if (streaming_state.use_streamed_data && !streaming_state.cached_streamed_gaussians.is_empty()) {
		const uint32_t streamed_count = static_cast<uint32_t>(streaming_state.cached_streamed_gaussians.size());
		if (p_index < streamed_count) {
			r_position = streaming_state.cached_streamed_gaussians[p_index].position;
			return true;
		}
		if (const uint32_t *mapped = streaming_state.cached_streamed_index_lookup.getptr(p_index)) {
			uint32_t mapped_index = *mapped;
			if (mapped_index < streamed_count) {
				r_position = streaming_state.cached_streamed_gaussians[mapped_index].position;
				return true;
			}
		}
	}

	const auto &scene_state = p_renderer.get_scene_state();
	if (scene_state.gaussian_data.is_valid()) {
		const LocalVector<Gaussian> &gaussians = scene_state.gaussian_data->get_gaussian_storage();
		if (p_index < static_cast<uint32_t>(gaussians.size())) {
			r_position = gaussians[p_index].position;
			return true;
		}
	}

	const auto &test_data = p_renderer.get_test_data_state();
	if (!test_data.positions.is_empty() && p_index < static_cast<uint32_t>(test_data.positions.size())) {
		r_position = test_data.positions[p_index];
		return true;
	}

	return false;
}

static void _set_instance_sort_inputs(GaussianSplatRenderer *p_renderer, GPUSortingPipeline *p_sorting_pipeline,
		uint32_t p_visible_chunk_count) {
	const auto &buffers = p_renderer->get_instance_pipeline_buffers();
	GPUSortingPipeline::InstancePipelineInputs instance_inputs;
	instance_inputs.atlas_gaussian_buffer = buffers.atlas_gaussian_buffer;
	instance_inputs.quantization_buffer = buffers.quantization_required ? buffers.quantization_buffer : RID();
	instance_inputs.instance_buffer = buffers.instance_buffer;
	instance_inputs.chunk_meta_buffer = buffers.chunk_meta_buffer;
	instance_inputs.visible_chunk_buffer = buffers.visible_chunk_buffer;
	instance_inputs.splat_ref_buffer = buffers.splat_ref_buffer;
	instance_inputs.sort_key_buffer = buffers.sort_key_buffer;
	instance_inputs.sort_value_buffer = buffers.sort_value_buffer;
	instance_inputs.counter_buffer = buffers.counter_buffer;
	instance_inputs.chunk_dispatch_buffer = buffers.chunk_dispatch_buffer;
	instance_inputs.indirect_count_buffer = buffers.indirect_count_buffer;
	instance_inputs.instance_count_buffer = buffers.instance_count_buffer;
	instance_inputs.visible_chunk_count = MIN(p_visible_chunk_count, buffers.max_visible_chunks);
	instance_inputs.max_visible_chunks = buffers.max_visible_chunks;
	instance_inputs.max_visible_splats = buffers.max_visible_splats;
	instance_inputs.max_chunk_splats = buffers.max_chunk_splats;
	instance_inputs.device = p_renderer->get_device_state().rd;
	p_sorting_pipeline->set_instance_pipeline_inputs(instance_inputs);
}

static bool _sync_instance_sort_inputs(GaussianSplatRenderer *p_renderer, GPUCuller *p_gpu_culler,
		GPUSortingPipeline *p_sorting_pipeline, uint32_t *r_visible_chunk_count) {
	if (r_visible_chunk_count) {
		*r_visible_chunk_count = 0;
	}
	if (!p_renderer || !p_sorting_pipeline || !p_renderer->has_instance_pipeline_buffers()) {
		if (p_sorting_pipeline) {
			p_sorting_pipeline->clear_instance_pipeline_inputs();
		}
		return false;
	}

	const auto &buffers = p_renderer->get_instance_pipeline_buffers();
	if (!GaussianSplatting::InstancePipelineContract::has_sort_buffers(buffers)) {
		p_sorting_pipeline->clear_instance_pipeline_inputs();
		WARN_PRINT_ONCE("[GPU Sort] Instance pipeline buffers invalid for sort; using fallback path.");
		return false;
	}

	// FIX: Use buffer capacity instead of stale async readback.
	// GPU-side counter drives actual dispatch; this is a structural guard only.
	uint32_t visible_chunk_count = buffers.max_visible_chunks;
	_set_instance_sort_inputs(p_renderer, p_sorting_pipeline, visible_chunk_count);
	if (r_visible_chunk_count) {
		*r_visible_chunk_count = visible_chunk_count;
	}
	return true;
}

static GPUSorterFactory::SortingAlgorithm _get_forced_sort_algorithm(const SortingStrategyConfig &p_config) {
	switch (p_config.force_algorithm) {
		case SortingStrategyConfig::ForcedAlgorithm::RADIX:
			return GPUSorterFactory::ALGORITHM_RADIX;
		case SortingStrategyConfig::ForcedAlgorithm::BITONIC:
			return GPUSorterFactory::ALGORITHM_BITONIC;
		case SortingStrategyConfig::ForcedAlgorithm::ONESWEEP:
			return GPUSorterFactory::ALGORITHM_ONESWEEP;
		case SortingStrategyConfig::ForcedAlgorithm::AUTO:
		default:
			return GPUSorterFactory::ALGORITHM_AUTO;
	}
}

static String _algorithm_override_label(const SortingStrategyConfig &p_config) {
	return p_config.get_forced_algorithm_name();
}
} // namespace

RenderSortingOrchestrator::RenderSortingOrchestrator(GaussianSplatRenderer *p_renderer, GPUCuller *p_gpu_culler,
		GPUSortingPipeline *p_sorting_pipeline,
		CullForViewFn p_cull_for_view, RecordRenderingErrorFn p_record_rendering_error) :
		renderer(p_renderer),
		gpu_culler(p_gpu_culler),
		sorting_pipeline(p_sorting_pipeline),
		cull_for_view(p_cull_for_view),
		record_rendering_error(p_record_rendering_error) {
	ERR_FAIL_NULL(renderer);
	ERR_FAIL_NULL(gpu_culler);
	ERR_FAIL_NULL(sorting_pipeline);
	ERR_FAIL_COND_MSG(!cull_for_view, "RenderSortingOrchestrator requires a cull callback.");
	ERR_FAIL_COND_MSG(!record_rendering_error, "RenderSortingOrchestrator requires an error reporting callback.");
}

void RenderSortingOrchestrator::refresh_gpu_sorter(const char *p_context) {
	if (!renderer->ensure_rendering_device(p_context)) {
		sorting_state.sorter_needs_rebuild = false;
		return;
	}

	// Phase 4: Backoff on repeated sorter init failures to prevent OOM crash
	const uint64_t current_frame = renderer->get_frame_state().frame_counter;
	if (sorting_state.sorter_init_failure_count > 0) {
		// Check if we've exceeded max failures - disable GPU sorting entirely
		if (sorting_state.sorter_init_failure_count >= sorting_state.kSorterInitMaxFailures) {
			// GPU sorting permanently disabled for this session due to repeated OOM
			sorting_state.sorter_needs_rebuild = false;
			return;
		}
		// Check if we're still in backoff period
		const uint64_t frames_since_failure = current_frame - sorting_state.last_sorter_init_failure_frame;
		if (frames_since_failure < sorting_state.kSorterInitBackoffFrames) {
			// Still in backoff, skip init attempt
			return;
		}
	}

	uint32_t capacity = MAX<uint32_t>(renderer->get_performance_settings().max_splats,
			renderer->get_test_data_state().positions.size());
	if (capacity == 0) {
		capacity = 1;
	}
	if (g_gpu_sorting_config.enable_performance_logging) {
		const uint64_t device_id = renderer->get_device_state().rd ? renderer->get_device_state().rd->get_device_instance_id() : 0;
		GS_LOG_GPU_SORT_DEBUG(vformat("[GPU Sort] Refresh (%s): max_splats=%d capacity=%u device_id=%s",
				p_context ? p_context : "unknown",
				renderer->get_performance_settings().max_splats,
				capacity,
				device_id > 0 ? String::num_uint64(device_id) : String("none")));
	}

	Ref<IGPUSorter> new_sorter;
	if (sorting_pipeline) {
		SortingStrategyConfig sort_config = SortingStrategyConfig::load_from_project_settings();
		sorting_pipeline->set_forced_sort_algorithm(_get_forced_sort_algorithm(sort_config));
		new_sorter = sorting_pipeline->rebuild_sorter_if_needed(
				renderer->get_device_state().rd, capacity, sorting_state.sorter_needs_rebuild);
	}

	sorting_state.gpu_sorter = new_sorter;
	sorting_state.sorter_needs_rebuild = false;

	if (!sorting_state.gpu_sorter.is_valid()) {
		// Sorter creation failed - record failure for backoff
		sorting_state.sorter_init_failure_count++;
		sorting_state.last_sorter_init_failure_frame = current_frame;
		if (sorting_state.sorter_init_failure_count >= sorting_state.kSorterInitMaxFailures) {
			GS_LOG_ERROR_DEFAULT(vformat("[GPU Sort] Sorter init failed %d times - disabling GPU sorting (use CPU fallback)",
					sorting_state.sorter_init_failure_count));
		} else {
			GS_LOG_WARN_DEFAULT(vformat("[GPU Sort] Sorter init failed (attempt %d/%d) - backing off for %d frames",
					sorting_state.sorter_init_failure_count,
					sorting_state.kSorterInitMaxFailures,
					sorting_state.kSorterInitBackoffFrames));
		}
		return;
	}

	// Success - reset failure counter
	if (sorting_state.sorter_init_failure_count > 0) {
		GS_LOG_INFO_DEFAULT("[GPU Sort] Sorter init succeeded after previous failures - resetting backoff");
		sorting_state.sorter_init_failure_count = 0;
	}

	bool pipeline_manages_buffers = sorting_pipeline &&
			sorting_pipeline->is_managing_buffers();
	if (!pipeline_manages_buffers && sorting_pipeline) {
		sorting_pipeline->release_sort_buffers(renderer);
	}
	if (sorting_pipeline) {
		sorting_pipeline->ensure_sort_buffers(
				renderer, sorting_state.gpu_sorter->get_max_elements());
	}
}

void RenderSortingOrchestrator::initialize_sorting() {
	sorting_state.sorter_needs_rebuild = true;
	if (sorting_pipeline) {
		sorting_pipeline->mark_sorter_dirty();
	}

	if (!renderer->ensure_rendering_device("initialize_sorting")) {
		sorting_state.sorting_initialized = false;
		return;
	}

	const RDD::Capabilities &caps = renderer->get_device_state().rd->get_device_capabilities();
	uint64_t max_workgroup_x = renderer->get_device_state().rd->limit_get(RenderingDevice::LIMIT_MAX_COMPUTE_WORKGROUP_SIZE_X);
	uint64_t max_invocations = renderer->get_device_state().rd->limit_get(RenderingDevice::LIMIT_MAX_COMPUTE_WORKGROUP_INVOCATIONS);
	bool has_compute_limits = max_workgroup_x > 0 && max_invocations > 0;
	bool compute_supported = has_compute_limits && max_workgroup_x >= 256 && max_invocations >= 256;
	if (!compute_supported) {
		GS_LOG_ERROR_DEFAULT("[GPU Sort] Rendering device lacks compute shader capacity; GPU sorting unavailable");
		RenderingError error(RenderingErrorCodes::gpu_sort_unavailable(), RenderingError::Severity::RECOVERABLE,
				"GPU sorting unavailable: compute shader limits below requirements");
		error.add_context("max_workgroup_size_x", static_cast<int64_t>(max_workgroup_x));
		error.add_context("max_workgroup_invocations", static_cast<int64_t>(max_invocations));
		error.add_context("device_name", renderer->get_device_state().rd ? renderer->get_device_state().rd->get_device_name() : String());
		error.add_recovery_step("Enable force_cpu_sort to use CPU sorting fallback");
		error.add_recovery_step("Use a device with compute workgroup size >= 256");
		record_rendering_error(error);
		sorting_state.sorter_needs_rebuild = false;
		sorting_state.sorting_initialized = false;
		if (sorting_state.gpu_sorter.is_valid()) {
			sorting_state.gpu_sorter->shutdown();
			sorting_state.gpu_sorter.unref();
		}
		if (sorting_pipeline) {
			sorting_pipeline->release_sort_buffers(renderer);
		}
		return;
	}

	refresh_gpu_sorter("initialize_sorting");
	sorting_state.sorting_initialized = sorting_state.gpu_sorter.is_valid();
}

Array RenderSortingOrchestrator::run_sort_benchmark(const PackedInt32Array &p_sizes) {
	Array results;

	if (!renderer->ensure_rendering_device("run_sort_benchmark")) {
		return results;
	}

	if (sorting_state.sorter_needs_rebuild) {
		refresh_gpu_sorter("run_sort_benchmark");
	}

	if (!sorting_state.gpu_sorter.is_valid()) {
		GS_LOG_WARN_DEFAULT("[GPU Sort Benchmark] GPU sorter not initialized; skipping benchmark");
		return results;
	}

	const uint32_t sorter_capacity = sorting_state.gpu_sorter->get_max_elements();
	if (sorter_capacity == 0) {
		GS_LOG_WARN_DEFAULT("[GPU Sort Benchmark] GPU sorter has zero capacity; skipping benchmark");
		return results;
	}

	RandomPCG rng;
	rng.randomize();

	const int size_count = p_sizes.size();
	for (int i = 0; i < size_count; i++) {
		int requested = p_sizes[i];
		if (requested <= 0) {
			continue;
		}

		uint32_t size = (uint32_t)requested;
		if (size > sorter_capacity) {
			GS_LOG_WARN_DEFAULT(vformat("[GPU Sort Benchmark] Requested size %d exceeds sorter capacity %d; skipping",
					size, sorter_capacity));
			continue;
		}

		Vector<uint8_t> key_data;
		Vector<uint8_t> value_data;
		key_data.resize(sorter_capacity * sizeof(float));
		value_data.resize(sorter_capacity * sizeof(uint32_t));

		float *keys = reinterpret_cast<float *>(key_data.ptrw());
		uint32_t *values = reinterpret_cast<uint32_t *>(value_data.ptrw());

		for (uint32_t j = 0; j < size; j++) {
			keys[j] = rng.random(0.0f, 1000.0f);
			values[j] = j;
		}
		GaussianSplatting::fill_sort_padding(keys, values, size, sorter_capacity);

		RID keys_rid = renderer->get_device_state().rd->storage_buffer_create(key_data.size(), key_data);
		if (keys_rid.is_valid()) {
			renderer->get_device_state().rd->set_resource_name(keys_rid, "GS_RenderSortingOrchestrator_BenchmarkKeys");
		}
		RID values_rid = renderer->get_device_state().rd->storage_buffer_create(value_data.size(), value_data);
		if (values_rid.is_valid()) {
			renderer->get_device_state().rd->set_resource_name(values_rid, "GS_RenderSortingOrchestrator_BenchmarkValues");
		}
		GPUBuffer keys_buffer(renderer->get_device_state().rd, keys_rid);
		GPUBuffer values_buffer(renderer->get_device_state().rd, values_rid);

		if (!keys_buffer.is_valid() || !values_buffer.is_valid()) {
			GS_LOG_WARN_DEFAULT("[GPU Sort Benchmark] Failed to allocate GPU buffers for benchmark");
			continue;
		}

		uint64_t submit_start = OS::get_singleton()->get_ticks_usec();
		uint64_t timeline_value = sorting_state.gpu_sorter->sort_async(
				keys_buffer.get(), values_buffer.get(), size);
		uint64_t submit_end = OS::get_singleton()->get_ticks_usec();
		float submit_time_ms = (submit_end - submit_start) / 1000.0f;

		uint64_t wait_start = OS::get_singleton()->get_ticks_usec();
		sorting_state.gpu_sorter->wait_for_completion();
		uint64_t wait_end = OS::get_singleton()->get_ticks_usec();
		const float measured_wait_ms = (wait_end - wait_start) / 1000.0f;
		const float reported_gpu_ms = sorting_state.gpu_sorter->get_last_sort_time_ms();
		const GaussianSplatting::SortBenchmarkTimingMetrics timing =
				GaussianSplatting::compute_sort_benchmark_timing(
						submit_time_ms, measured_wait_ms, reported_gpu_ms, timeline_value);
		if (!timing.async_requested && timing.waited_for_completion) {
			GS_LOG_WARN_DEFAULT(vformat("[GPU Sort Benchmark] sort_async returned no timeline token but wait_for_completion blocked for %.3fms",
					timing.wait_ms));
		}
		if (timing.gpu_ms <= 0.0f) {
			GS_LOG_WARN_DEFAULT("[GPU Sort Benchmark] Sort benchmark sample produced non-positive timing; skipping");
			continue;
		}

		Dictionary sample;
		sample["elements"] = (int)size;
		sample["submit_ms"] = timing.submit_ms;
		sample["wait_ms"] = timing.wait_ms;
		sample["gpu_ms"] = timing.gpu_ms;
		sample["throughput_mpps"] = timing.gpu_ms > 0.0f ? (double(size) / timing.gpu_ms) / 1000.0 : 0.0;
		sample["used_async"] = timing.used_async;
		sample["async_requested"] = timing.async_requested;
		sample["waited_for_completion"] = timing.waited_for_completion;
		sample["algorithm"] = sorting_state.gpu_sorter->get_algorithm_name();
		sample["capacity"] = (int)sorter_capacity;

		results.push_back(sample);

	}

	return results;
}

void RenderSortingOrchestrator::benchmark_sorting_performance() {
	if (!renderer->ensure_rendering_device("benchmark_sorting_performance")) {
		return;
	}

	if (sorting_state.sorter_needs_rebuild) {
		refresh_gpu_sorter("benchmark_sorting_performance");
	}

	if (!sorting_state.gpu_sorter.is_valid()) {
		GS_LOG_INFO_DEFAULT("[GPU Sort Benchmark] GPU sorter not initialized; skipping benchmark");
		return;
	}

	const uint32_t sorter_capacity = sorting_state.gpu_sorter->get_max_elements();
	GS_LOG_INFO_DEFAULT("\n=== GPU Sorting Benchmark ===");
	GS_LOG_INFO_DEFAULT("Algorithm: " + sorting_state.gpu_sorter->get_algorithm_name());
	GS_LOG_INFO_DEFAULT("Capacity: " + itos(sorter_capacity) + " elements");

	uint32_t test_sizes[] = {1000, 10000, 50000, 100000, 500000};
	RandomPCG rng;
	rng.randomize();

	for (uint32_t size : test_sizes) {
		if (size > sorter_capacity) {
			continue;
		}

		Vector<uint8_t> key_data;
		Vector<uint8_t> value_data;
		key_data.resize(sorter_capacity * sizeof(float));
		value_data.resize(sorter_capacity * sizeof(uint32_t));

		float *keys = reinterpret_cast<float *>(key_data.ptrw());
		uint32_t *values = reinterpret_cast<uint32_t *>(value_data.ptrw());

		for (uint32_t i = 0; i < size; i++) {
			keys[i] = rng.random(0.0f, 1000.0f);
			values[i] = i;
		}
		GaussianSplatting::fill_sort_padding(keys, values, size, sorter_capacity);

		RID keys_buffer = renderer->get_device_state().rd->storage_buffer_create(key_data.size(), key_data);
		if (keys_buffer.is_valid()) {
			renderer->get_device_state().rd->set_resource_name(keys_buffer, "GS_RenderSortingOrchestrator_PerfBenchmarkKeys");
		}
		RID values_buffer = renderer->get_device_state().rd->storage_buffer_create(value_data.size(), value_data);
		if (values_buffer.is_valid()) {
			renderer->get_device_state().rd->set_resource_name(values_buffer, "GS_RenderSortingOrchestrator_PerfBenchmarkValues");
		}

		auto queue_free = [&](RID &r_buffer) {
			renderer->get_resource_state().deletion_queue.queue_free(renderer->get_device_state().rd, r_buffer);
			r_buffer = RID();
		};

		if (!keys_buffer.is_valid() || !values_buffer.is_valid()) {
			GS_LOG_WARN_DEFAULT("[GPU Sort Benchmark] Failed to allocate buffers for benchmark");
			queue_free(keys_buffer);
			queue_free(values_buffer);
			continue;
		}

		const float reported_before_ms = sorting_state.gpu_sorter->get_last_sort_time_ms();
		uint64_t submit_start = OS::get_singleton()->get_ticks_usec();
		uint64_t timeline_value = sorting_state.gpu_sorter->sort_async(keys_buffer, values_buffer, size);
		uint64_t submit_end = OS::get_singleton()->get_ticks_usec();
		float submit_time_ms = (submit_end - submit_start) / 1000.0f;

		uint64_t wait_start = OS::get_singleton()->get_ticks_usec();
		sorting_state.gpu_sorter->wait_for_completion();
		uint64_t wait_end = OS::get_singleton()->get_ticks_usec();
		float measured_wait_ms = (wait_end - wait_start) / 1000.0f;
		float reported_ms = sorting_state.gpu_sorter->get_last_sort_time_ms();
		const GaussianSplatting::SortBenchmarkTimingMetrics timing =
				GaussianSplatting::compute_sort_benchmark_timing(
						submit_time_ms, measured_wait_ms, reported_ms, timeline_value);
		if (timeline_value == 0 && !timing.waited_for_completion &&
				(reported_ms <= 0.0f || Math::is_equal_approx(reported_ms, reported_before_ms))) {
			GS_LOG_WARN_DEFAULT(vformat("[GPU Sort Benchmark] sort_async submit failed for %d elements; skipping benchmark sample", size));
			queue_free(keys_buffer);
			queue_free(values_buffer);
			continue;
		}
		if (!timing.async_requested && timing.waited_for_completion) {
			GS_LOG_WARN_DEFAULT(vformat("[GPU Sort Benchmark] sort_async returned no timeline token but wait_for_completion blocked for %.3fms",
					timing.wait_ms));
		}
		if (timing.gpu_ms <= 0.0f) {
			GS_LOG_WARN_DEFAULT("[GPU Sort Benchmark] Sort invocation produced non-positive timing");
			queue_free(keys_buffer);
			queue_free(values_buffer);
			continue;
		}

		float throughput = timing.gpu_ms > 0.0f ? (size / timing.gpu_ms) / 1000.0f : 0.0f;

		GS_LOG_INFO_DEFAULT(vformat("  %6d elements: submit %.2fms, wait %.2fms, GPU %.2fms, %.2fM elements/sec (async=%s)",
				size,
				timing.submit_ms,
				timing.wait_ms,
				timing.gpu_ms,
				throughput,
				timing.used_async ? "true" : "false"));

		queue_free(keys_buffer);
		queue_free(values_buffer);
	}

	GS_LOG_INFO_DEFAULT("==============================\n");
}

GaussianRenderState::SortStageSummary RenderSortingOrchestrator::sort_gaussians_for_view(
		const Transform3D &p_world_to_camera_transform, GaussianRenderState::IndexDomain p_input_domain) {
	auto &cull_state = gpu_culler->get_state();
	uint32_t available_splats = cull_state.culled_indices.size();
	auto resolve_output_domain_for_input = [](GaussianRenderState::IndexDomain p_domain) {
		switch (p_domain) {
			case GaussianRenderState::IndexDomain::CHUNK_REF:
				return GaussianRenderState::IndexDomain::SPLAT_REF;
			case GaussianRenderState::IndexDomain::GAUSSIAN_GLOBAL:
				return GaussianRenderState::IndexDomain::GAUSSIAN_GLOBAL;
			case GaussianRenderState::IndexDomain::UNKNOWN:
			case GaussianRenderState::IndexDomain::SPLAT_REF:
			default:
				return GaussianRenderState::IndexDomain::UNKNOWN;
		}
	};
	auto build_summary = [&]() {
		GaussianRenderState::SortStageSummary summary;
		summary.sorted_count = sorting_state.sorted_splat_count;
		summary.sort_time_ms = renderer->get_frame_state().sort_time_ms;
		summary.input_domain = p_input_domain;
		summary.output_domain = resolve_output_domain_for_input(p_input_domain);
		return summary;
	};
	auto set_active_sort_algorithm = [&](const String &p_algorithm, const String &p_reason) {
		if (sorting_state.active_sort_algorithm != p_algorithm) {
			sorting_state.active_sort_algorithm = p_algorithm;
			sorting_state.sort_switch_reason = p_reason;
		} else if (!p_reason.is_empty()) {
			sorting_state.sort_switch_reason = p_reason;
		} else if (sorting_state.sort_switch_reason.is_empty() ||
				sorting_state.sort_switch_reason == "uninitialized") {
			sorting_state.sort_switch_reason = p_reason;
		}
	};

	SortingStrategyConfig sort_config = SortingStrategyConfig::load_from_project_settings();
	const bool force_cpu_sort = sort_config.force_cpu_sort;
	const bool strict_global_sort = g_gpu_sorting_config.strict_global_sort;
	const bool force_algorithm = sort_config.is_algorithm_forced();
	const String forced_algorithm_name = _algorithm_override_label(sort_config);
	const int force_algorithm_value = static_cast<int>(sort_config.force_algorithm);
	if (sorting_pipeline) {
		sorting_pipeline->set_forced_sort_algorithm(_get_forced_sort_algorithm(sort_config));
	}

	const bool tracking_initialized = runtime_override_tracking_initialized;
	const bool cpu_override_changed = tracking_initialized && last_force_cpu_override != force_cpu_sort;
	const bool algorithm_override_changed = tracking_initialized && last_force_algorithm_override != force_algorithm_value;
	runtime_override_tracking_initialized = true;
	last_force_cpu_override = force_cpu_sort;
	last_force_algorithm_override = force_algorithm_value;
	const bool override_state_changed = cpu_override_changed || algorithm_override_changed;
	if (algorithm_override_changed) {
		sorting_state.sorter_needs_rebuild = true;
		instance_sort_cache.valid = false;
		if (sorting_pipeline) {
			sorting_pipeline->mark_sorter_dirty();
		}
		GS_LOG_GPU_SORT_INFO(vformat("[GPU Sort] Runtime algorithm override changed: %s",
				forced_algorithm_name));
	}
	if (cpu_override_changed) {
		instance_sort_cache.valid = false;
		GS_LOG_GPU_SORT_INFO(vformat("[GPU Sort] Runtime CPU fallback override %s",
				force_cpu_sort ? "enabled" : "disabled"));
	}
	if (override_state_changed) {
		if (force_cpu_sort) {
			sorting_state.sort_switch_reason = "force_cpu_sort override enabled";
		} else if (force_algorithm) {
			sorting_state.sort_switch_reason = vformat("forced GPU algorithm override: %s", forced_algorithm_name);
		} else {
			sorting_state.sort_switch_reason = "runtime sorting overrides disabled";
		}
	}
	sorting_state.override_force_cpu = force_cpu_sort;
	sorting_state.override_force_algorithm = force_algorithm;
	sorting_state.override_forced_algorithm = forced_algorithm_name;

	renderer->get_performance_state().metrics.sort_input_build_time_ms = 0.0f;
	const bool instance_pipeline_buffers_valid = renderer->has_instance_pipeline_buffers();
	uint64_t instance_content_generation = 0;
	uint32_t instance_max_visible_splats = 0;
	uint32_t instance_visible_chunk_count = 0;
	uint32_t instance_max_chunk_splats = 0;
	const bool instance_pipeline_active = _sync_instance_sort_inputs(renderer, gpu_culler, sorting_pipeline,
			&instance_visible_chunk_count);
	const bool input_domain_is_chunk = p_input_domain == GaussianRenderState::IndexDomain::CHUNK_REF;
	const bool input_domain_is_global = p_input_domain == GaussianRenderState::IndexDomain::GAUSSIAN_GLOBAL;
	const bool input_domain_known = input_domain_is_chunk || input_domain_is_global;
	Transform3D instance_camera_to_world;
	bool instance_camera_valid = false;
	if (instance_pipeline_buffers_valid) {
		instance_content_generation = renderer->get_instance_pipeline_content_generation();
		instance_max_visible_splats = renderer->get_instance_pipeline_buffers().max_visible_splats;
		instance_max_chunk_splats = renderer->get_instance_pipeline_buffers().max_chunk_splats;
	}
	renderer->get_debug_state().sort_route_uid = RenderRouteUID::COMMON_UNSET_SORT_ROUTE;
	if (!input_domain_known) {
		GS_LOG_ERROR_DEFAULT("[GPU Sort] Index-domain contract violation: sort input domain is unknown");
		renderer->get_frame_state().sort_time_ms = 0.0f;
		renderer->get_performance_state().metrics.sort_submission_time_ms = 0.0f;
		renderer->get_performance_state().metrics.sort_wait_time_ms = 0.0f;
		renderer->get_performance_state().metrics.sort_input_build_time_ms = 0.0f;
		renderer->get_performance_state().metrics.async_sort_used = false;
		renderer->get_performance_state().metrics.async_sort_waited = false;
		renderer->get_performance_state().metrics.async_overlap_efficiency = 0.0f;
		sorting_state.sorted_splat_count = 0;
		renderer->get_frame_state().visible_splat_count.store(0, std::memory_order_release);
		renderer->get_debug_state().sort_route_uid = RenderRouteUID::COMMON_FAIL_SORT_FAILED;
		return build_summary();
	}
	if (input_domain_is_chunk && !instance_pipeline_active) {
		GS_LOG_ERROR_DEFAULT("[GPU Sort] Index-domain contract violation: chunk-domain sort input requires instance pipeline inputs");
		renderer->get_frame_state().sort_time_ms = 0.0f;
		renderer->get_performance_state().metrics.sort_submission_time_ms = 0.0f;
		renderer->get_performance_state().metrics.sort_wait_time_ms = 0.0f;
		renderer->get_performance_state().metrics.sort_input_build_time_ms = 0.0f;
		renderer->get_performance_state().metrics.async_sort_used = false;
		renderer->get_performance_state().metrics.async_sort_waited = false;
		renderer->get_performance_state().metrics.async_overlap_efficiency = 0.0f;
		sorting_state.sorted_splat_count = 0;
		renderer->get_frame_state().visible_splat_count.store(0, std::memory_order_release);
		renderer->get_debug_state().sort_route_uid = RenderRouteUID::COMMON_FAIL_SORT_FAILED;
		return build_summary();
	}
	if (input_domain_is_global && instance_pipeline_active) {
		GS_LOG_ERROR_DEFAULT("[GPU Sort] Index-domain contract violation: global-domain sort input is incompatible with instance pipeline sort");
		renderer->get_frame_state().sort_time_ms = 0.0f;
		renderer->get_performance_state().metrics.sort_submission_time_ms = 0.0f;
		renderer->get_performance_state().metrics.sort_wait_time_ms = 0.0f;
		renderer->get_performance_state().metrics.sort_input_build_time_ms = 0.0f;
		renderer->get_performance_state().metrics.async_sort_used = false;
		renderer->get_performance_state().metrics.async_sort_waited = false;
		renderer->get_performance_state().metrics.async_overlap_efficiency = 0.0f;
		sorting_state.sorted_splat_count = 0;
		renderer->get_frame_state().visible_splat_count.store(0, std::memory_order_release);
		renderer->get_debug_state().sort_route_uid = RenderRouteUID::COMMON_FAIL_SORT_FAILED;
		return build_summary();
	}

	const bool instance_sort_inputs_ready = instance_pipeline_active;
	uint64_t current_cull_signature = 0;
	bool current_cull_signature_computed = false;
	auto compute_current_cull_signature = [&]() -> uint64_t {
		if (current_cull_signature_computed) {
			return current_cull_signature;
		}
		constexpr uint64_t fnv_offset = 1469598103934665603ULL;
		constexpr uint64_t fnv_prime = 1099511628211ULL;
		uint64_t signature = fnv_offset;
		const uint32_t cull_count = MIN(available_splats, static_cast<uint32_t>(cull_state.culled_indices.size()));
		for (uint32_t i = 0; i < cull_count; i++) {
			signature ^= (uint64_t(cull_state.culled_indices[i]) << 32) ^ uint64_t(i);
			signature *= fnv_prime;
		}
		signature ^= uint64_t(cull_count);
		signature *= fnv_prime;
		current_cull_signature = signature;
		current_cull_signature_computed = true;
		return current_cull_signature;
	};
	auto refresh_cull_signature_tracking = [&]() {
		if (instance_pipeline_active) {
			sorting_state.last_cull_indices_signature = 0;
			sorting_state.last_cull_indices_signature_valid = false;
			return;
		}
		if (available_splats == 0 || cull_state.culled_indices.is_empty()) {
			sorting_state.last_cull_indices_signature = 0;
			sorting_state.last_cull_indices_signature_valid = true;
			return;
		}
		sorting_state.last_cull_indices_signature = compute_current_cull_signature();
		sorting_state.last_cull_indices_signature_valid = true;
	};

	auto reset_sort_metrics = [&]() {
		renderer->get_frame_state().sort_time_ms = 0.0f;
		renderer->get_performance_state().metrics.sort_submission_time_ms = 0.0f;
		renderer->get_performance_state().metrics.sort_wait_time_ms = 0.0f;
		renderer->get_performance_state().metrics.sort_input_build_time_ms = 0.0f;
		renderer->get_performance_state().metrics.async_sort_used = false;
		renderer->get_performance_state().metrics.async_sort_waited = false;
		renderer->get_performance_state().metrics.async_overlap_efficiency = 0.0f;
	};

	auto record_gpu_sort_sample = [&]() {
		GaussianSplatRenderer::SortFrameMetrics sample;
		sample.frame_index = renderer->get_frame_state().frame_counter;
		sample.element_count = sorting_state.sorted_splat_count;
		sample.total_ms = renderer->get_frame_state().sort_time_ms;
		sample.gpu_ms = renderer->get_frame_state().sort_time_ms;
		sample.cpu_ms = 0.0f;
		sample.cpu_selection_ms = renderer->get_performance_state().metrics.sort_input_build_time_ms;
		if (sorting_state.gpu_sorter.is_valid()) {
			sample.algorithm = sorting_state.gpu_sorter->get_algorithm_name();
		} else if (sorting_pipeline) {
			sample.algorithm = sorting_pipeline->get_algorithm_name();
		} else {
			sample.algorithm = StringName("GPU");
		}
		sample.used_gpu = true;
		sample.used_cpu_fallback = false;
		sample.used_hybrid = false;
		renderer->record_sort_sample(sample);
	};

	auto reuse_previous_sort = [&](const String &p_reason, const char *p_route_uid) -> bool {
		const bool has_valid_sort_buffer = sorting_pipeline && sorting_pipeline->get_sort_indices_buffer().is_valid();
		if (strict_global_sort && has_valid_sort_buffer) {
			return false;
		}
		if (!sorting_pipeline || sorting_state.sorted_splat_count == 0) {
			return false;
		}
		RID sort_indices_buffer = sorting_pipeline->get_sort_indices_buffer();
		if (!sort_indices_buffer.is_valid()) {
			return false;
		}
		GS_LOG_WARN_DEFAULT(vformat("[GPU Sort] %s; reusing previous sorted indices", p_reason));
		reset_sort_metrics();
		renderer->get_debug_state().sort_route_uid = p_route_uid;
		renderer->get_frame_state().visible_splat_count.store(sorting_state.sorted_splat_count, std::memory_order_release);
		renderer->get_performance_state().metrics.rendered_splat_count =
				renderer->get_frame_state().visible_splat_count.load(std::memory_order_acquire);
		set_active_sort_algorithm(sorting_state.active_sort_algorithm, p_reason);
		return true;
	};

	const auto compute_instance_visible_splat_budget = [&](uint32_t p_visible_chunk_count) -> uint32_t {
		if (instance_max_visible_splats == 0 || instance_max_chunk_splats == 0 || p_visible_chunk_count == 0) {
			return 0;
		}
		const uint64_t chunk_budget_u64 = uint64_t(p_visible_chunk_count) * uint64_t(instance_max_chunk_splats);
		const uint32_t chunk_budget = chunk_budget_u64 > uint64_t(UINT32_MAX) ? UINT32_MAX : uint32_t(chunk_budget_u64);
		return MIN(instance_max_visible_splats, chunk_budget);
	};

	// Instance sort-cache reuse with identical camera is exact (not approximate),
	// so it is safe even in strict_global_sort mode.
	if (!force_cpu_sort && instance_pipeline_active && instance_max_visible_splats > 0 &&
			instance_visible_chunk_count > 0 && instance_max_chunk_splats > 0) {
		instance_camera_to_world = p_world_to_camera_transform.affine_inverse();
		instance_camera_valid = true;
		uint32_t cached_count = 0;
		if (_try_reuse_instance_sort_cache_with_camera(instance_camera_to_world, instance_content_generation,
					instance_max_visible_splats, instance_visible_chunk_count, cached_count)) {
			const uint32_t current_visible_budget = compute_instance_visible_splat_budget(instance_visible_chunk_count);
			// Active-path correctness guard: never reuse cached sort domains larger
			// than the current chunk-domain splat budget.
			// In the instance pipeline, cull_state.culled_indices is not splat-domain,
			// so relying on it can admit stale tails when visibility drops.
			if (cached_count > current_visible_budget) {
				// Do not reuse; fall through to execute a fresh sort below.
			} else {
				sorting_state.sorted_splat_count = cached_count;
				renderer->get_frame_state().visible_splat_count.store(cached_count, std::memory_order_release);
				renderer->get_performance_state().metrics.rendered_splat_count =
						renderer->get_frame_state().visible_splat_count.load(std::memory_order_acquire);
				reset_sort_metrics();
				renderer->get_debug_state().sort_route_uid = RenderRouteUID::INSTANCE_SORT_CACHED;
				sorting_state.last_sort_world_to_camera_transform = p_world_to_camera_transform;
				sorting_state.last_sort_transform_valid = true;
				gpu_culler->get_config().cull_params_dirty = false;
				if (sorting_state.active_sort_algorithm == "uninitialized") {
					set_active_sort_algorithm("GPU (cached)", "reused cached sort indices");
				}
				refresh_cull_signature_tracking();
				return build_summary();
			}
		}
	}

	if (!force_cpu_sort && instance_pipeline_active && instance_sort_inputs_ready) {
		if (sorting_pipeline && sorting_pipeline->sort_gaussians_gpu(renderer, p_world_to_camera_transform)) {
			renderer->get_debug_state().sort_route_uid = RenderRouteUID::INSTANCE_SORT_GPU;
			sorting_state.last_sort_world_to_camera_transform = p_world_to_camera_transform;
			sorting_state.last_sort_transform_valid = true;
			gpu_culler->get_config().cull_params_dirty = false;
			String gpu_algorithm = sorting_state.gpu_sorter.is_valid()
					? sorting_state.gpu_sorter->get_algorithm_name()
					: sorting_pipeline->get_algorithm_name();
			if (gpu_algorithm.is_empty()) {
				gpu_algorithm = "GPU";
			}
			String gpu_switch_reason = force_algorithm
					? vformat("forced GPU algorithm override: %s", forced_algorithm_name)
					: String("automatic GPU algorithm selection");
			set_active_sort_algorithm(gpu_algorithm, gpu_switch_reason);
			record_gpu_sort_sample();
			if (instance_max_visible_splats > 0) {
				if (!instance_camera_valid) {
					instance_camera_to_world = p_world_to_camera_transform.affine_inverse();
					instance_camera_valid = true;
				}
				_update_instance_sort_cache_with_camera(instance_camera_to_world, instance_content_generation,
						instance_max_visible_splats, instance_visible_chunk_count, sorting_state.sorted_splat_count);
			}
			refresh_cull_signature_tracking();
			return build_summary();
		}
	}

	if (available_splats == 0) {
		renderer->get_debug_state().sort_route_uid = RenderRouteUID::COMMON_SKIP_NO_VISIBLE;
		renderer->get_frame_state().visible_splat_count.store(0, std::memory_order_release);
		sorting_state.sorted_splat_count = 0;
		renderer->get_frame_state().sort_time_ms = 0.0f;
		renderer->get_performance_state().metrics.sort_submission_time_ms = 0.0f;
		renderer->get_performance_state().metrics.sort_wait_time_ms = 0.0f;
		renderer->get_performance_state().metrics.sort_input_build_time_ms = 0.0f;
		renderer->get_performance_state().metrics.async_sort_used = false;
		renderer->get_performance_state().metrics.async_sort_waited = false;
		renderer->get_performance_state().metrics.async_overlap_efficiency = 0.0f;
		// Fix: Reset cull_params_dirty on early return to prevent stuck dirty state
		gpu_culler->get_config().cull_params_dirty = false;
		refresh_cull_signature_tracking();
		return build_summary();
	}

	bool camera_moved = true;
	if (sorting_state.last_sort_transform_valid) {
		camera_moved = _is_sort_camera_move_significant(
			sorting_state.last_sort_world_to_camera_transform,
			p_world_to_camera_transform);
	}
	// Async CPU readback can lag the actual sorted index buffer by a frame.
	// Keep the mismatch for diagnostics only; do not let it drive re-sort decisions.
	const bool stale_cpu_count_mismatch = (sorting_state.sorted_splat_count != available_splats);
	uint32_t instance_visible_splats = (instance_pipeline_active && sorting_pipeline && instance_max_visible_splats > 0)
			? MIN(sorting_pipeline->get_last_instance_visible_splat_count(), instance_max_visible_splats)
			: 0u;
	if (instance_visible_splats > 0 && instance_visible_chunk_count > 0 && instance_max_chunk_splats > 0) {
		const uint64_t chunk_budget_u64 = uint64_t(instance_visible_chunk_count) * uint64_t(instance_max_chunk_splats);
		const uint32_t chunk_budget = chunk_budget_u64 > uint64_t(UINT32_MAX) ? UINT32_MAX : uint32_t(chunk_budget_u64);
		instance_visible_splats = MIN(instance_visible_splats, chunk_budget);
	}

	bool cull_signature_untracked = false;
	bool cull_signature_mismatch = false;
	const bool can_validate_previous_global_sort =
			!instance_pipeline_active &&
			available_splats > 0 &&
			sorting_state.last_sort_transform_valid &&
			sorting_pipeline &&
			sorting_pipeline->get_sort_indices_buffer().is_valid();
	if (can_validate_previous_global_sort) {
		cull_signature_untracked = !sorting_state.last_cull_indices_signature_valid;
		if (!cull_signature_untracked &&
				!camera_moved &&
				!gpu_culler->get_config().cull_params_dirty &&
				!override_state_changed) {
			cull_signature_mismatch =
					sorting_state.last_cull_indices_signature != compute_current_cull_signature();
		}
	}

	bool need_sort = camera_moved ||
			!sorting_state.last_sort_transform_valid ||
			gpu_culler->get_config().cull_params_dirty ||
			override_state_changed ||
			cull_signature_untracked ||
			cull_signature_mismatch;

	// DEBUG: Track which path we're taking
	if (renderer->get_debug_config().enable_sort_path_logs) {
		static int debug_frame = 0;
		const int log_interval = renderer->get_debug_config().frame_log_frequency;
		if (log_interval > 0 && (++debug_frame == 1 || (debug_frame % log_interval == 0))) {
			GS_LOG_GPU_SORT_DEBUG(vformat("[SORT-PATH] need_sort=%s camera_moved=%s count_mismatch=%s transform_valid=%s cull_dirty=%s override_changed=%s available=%d",
				need_sort ? "YES" : "no",
				camera_moved ? "YES" : "no",
				stale_cpu_count_mismatch ? "cpu-ignored" : "no",
				sorting_state.last_sort_transform_valid ? "yes" : "NO",
				gpu_culler->get_config().cull_params_dirty ? "YES" : "no",
				override_state_changed ? "YES" : "no",
				available_splats));
		}
	}

	auto publish_instance_identity_fallback = [&](const String &p_reason) -> bool {
		if (strict_global_sort) {
			// Even in strict mode, allow the identity fallback when the sort
			// buffer is missing/invalid — refusing to render at all is worse
			// than showing an unsorted frame.
			if (sorting_pipeline && sorting_pipeline->get_sort_indices_buffer().is_valid()) {
				return false;
			}
		}
		if (!instance_pipeline_active || !sorting_pipeline) {
			return false;
		}
		if (instance_visible_splats == 0) {
			return false;
		}

		sorting_state.sort_index_bytes.resize(instance_visible_splats * sizeof(uint32_t));
		uint32_t *indices_ptr = reinterpret_cast<uint32_t *>(sorting_state.sort_index_bytes.ptrw());
		for (uint32_t i = 0; i < instance_visible_splats; i++) {
			indices_ptr[i] = i;
		}
		sorting_pipeline->ensure_sort_buffers(renderer, instance_visible_splats);
		RID sort_indices_buffer = sorting_pipeline->get_sort_indices_buffer();
		if (!sort_indices_buffer.is_valid()) {
			return false;
		}
		RenderingDevice *target_device = renderer->get_resource_owner(sort_indices_buffer, renderer->get_device_state().rd);
		if (!target_device) {
			target_device = renderer->get_device_state().rd;
		}
		if (!target_device) {
			return false;
		}
		target_device->buffer_update(sort_indices_buffer, 0,
				sorting_state.sort_index_bytes.size(),
				sorting_state.sort_index_bytes.ptr());
		sorting_state.sorted_splat_count = instance_visible_splats;
		renderer->get_frame_state().visible_splat_count.store(instance_visible_splats, std::memory_order_release);
		renderer->get_performance_state().metrics.rendered_splat_count =
				renderer->get_frame_state().visible_splat_count.load(std::memory_order_acquire);
		renderer->get_debug_state().sort_route_uid = RenderRouteUID::INSTANCE_SORT_IDENTITY_FALLBACK;
		sorting_state.last_sort_transform_valid = false;
		set_active_sort_algorithm("GPU (identity fallback)", p_reason);
		GS_LOG_WARN_DEFAULT(vformat("[GPU Sort] %s; publishing identity order for instance sort domain", p_reason));
		return true;
	};

	if (!need_sort) {
		const bool can_reuse_previous_sorted =
				sorting_pipeline &&
				sorting_pipeline->get_sort_indices_buffer().is_valid() &&
				((!instance_pipeline_active && available_splats > 0) ||
						(instance_pipeline_active && sorting_state.sorted_splat_count > 0));
		if (can_reuse_previous_sorted) {
			// In the global path the current cull domain is authoritative. Reuse the
			// previous sorted buffer without trusting a possibly stale CPU-side count.
			const uint32_t reused_count = instance_pipeline_active ? sorting_state.sorted_splat_count : available_splats;
			sorting_state.sorted_splat_count = reused_count;
			renderer->get_frame_state().visible_splat_count.store(reused_count, std::memory_order_release);
			renderer->get_performance_state().metrics.rendered_splat_count =
					renderer->get_frame_state().visible_splat_count.load(std::memory_order_acquire);
		} else if (publish_instance_identity_fallback("Missing previous sorted buffer on camera-stable frame")) {
			reset_sort_metrics();
			return build_summary();
		} else if ((!strict_global_sort || !sorting_pipeline->get_sort_indices_buffer().is_valid()) && !instance_pipeline_active && !cull_state.culled_indices.is_empty() && sorting_pipeline) {
			// Last resort bootstrap: if the previous sorted buffer is unavailable,
			// keep rendering progress with current cull order instead of showing zero splats.
			// This fallback is reachable even in strict mode when the cached sort buffer is missing,
			// since rendering zero splats is worse than rendering with approximate cull order.
			const uint32_t copy_count = MIN<uint32_t>(available_splats,
					static_cast<uint32_t>(cull_state.culled_indices.size()));
			if (copy_count > 0) {
				sorting_state.sort_index_bytes.resize(copy_count * sizeof(uint32_t));
				uint32_t *indices_ptr = reinterpret_cast<uint32_t *>(sorting_state.sort_index_bytes.ptrw());
				for (uint32_t i = 0; i < copy_count; i++) {
					indices_ptr[i] = cull_state.culled_indices[i];
				}
				sorting_pipeline->ensure_sort_buffers(renderer, copy_count);
				RID sort_indices_buffer = sorting_pipeline->get_sort_indices_buffer();
				if (sort_indices_buffer.is_valid()) {
					RenderingDevice *target_device = renderer->get_resource_owner(sort_indices_buffer, renderer->get_device_state().rd);
					if (!target_device) {
						target_device = renderer->get_device_state().rd;
					}
					if (target_device) {
						target_device->buffer_update(sort_indices_buffer, 0,
								sorting_state.sort_index_bytes.size(),
								sorting_state.sort_index_bytes.ptr());
					}
				}
				sorting_state.sorted_splat_count = copy_count;
				renderer->get_frame_state().visible_splat_count.store(copy_count, std::memory_order_release);
				renderer->get_performance_state().metrics.rendered_splat_count =
						renderer->get_frame_state().visible_splat_count.load(std::memory_order_acquire);
				// Force a real re-sort on the next frame; this bootstrap order is not depth-sorted.
				sorting_state.last_sort_transform_valid = false;
			}
		} else if (strict_global_sort && sorting_pipeline && available_splats > 0) {
			// Strict mode cannot publish fallback ordering. If the stable-frame fast path
			// has no reusable sorted buffer, force a real sort this frame instead of
			// returning with stale/invalid state.
			need_sort = true;
			sorting_state.last_sort_transform_valid = false;
			GS_LOG_WARN_DEFAULT("[GPU Sort] Strict mode missing previous sorted buffer on camera-stable frame; forcing immediate re-sort");
		}
		if (!need_sort) {
			refresh_cull_signature_tracking();
			reset_sort_metrics();
			renderer->get_debug_state().sort_route_uid = RenderRouteUID::COMMON_SKIP_CAMERA_STABLE;
			return build_summary();
		}
	}

	auto apply_sort_buffer_update = [&](uint32_t p_count) {
		if (!sorting_pipeline) {
			return;
		}
		sorting_pipeline->ensure_sort_buffers(renderer, p_count);
		RID sort_indices_buffer = sorting_pipeline->get_sort_indices_buffer();
		if (!sort_indices_buffer.is_valid() || sorting_state.sort_index_bytes.is_empty()) {
			return;
		}
		RenderingDevice *target_device = renderer->get_resource_owner(sort_indices_buffer, renderer->get_device_state().rd);
		if (!target_device) {
			target_device = renderer->get_device_state().rd;
		}
		if (target_device) {
			target_device->buffer_update(sort_indices_buffer, 0,
					sorting_state.sort_index_bytes.size(),
					sorting_state.sort_index_bytes.ptr());
		}
	};

	auto record_cpu_sort_sample = [&](uint32_t p_count, float p_cpu_time_ms, const StringName &p_algorithm) {
		GaussianSplatRenderer::SortFrameMetrics sample;
		sample.frame_index = renderer->get_frame_state().frame_counter;
		sample.element_count = p_count;
		sample.total_ms = p_cpu_time_ms;
		sample.gpu_ms = 0.0f;
		sample.cpu_ms = p_cpu_time_ms;
		sample.cpu_selection_ms = renderer->get_performance_state().metrics.sort_input_build_time_ms;
		sample.algorithm = p_algorithm;
		sample.used_gpu = false;
		sample.used_cpu_fallback = true;
		sample.used_hybrid = false;
		renderer->record_sort_sample(sample);
	};

	auto run_cpu_sort = [&](const String &p_reason, bool p_forced) -> bool {
		if (!input_domain_is_global) {
			GS_LOG_ERROR_DEFAULT(vformat("[CPU Sort] Index-domain contract violation: CPU fallback requires gaussian_global input (got %s)",
					GaussianRenderState::index_domain_to_string(p_input_domain)));
			return false;
		}
		uint64_t sort_start = OS::get_singleton()->get_ticks_usec();
		auto &cpu_cull_state = gpu_culler->get_state();

		if ((uint32_t)cpu_sort_original_indices_scratch.size() < available_splats) {
			cpu_sort_original_indices_scratch.resize(available_splats);
		}
		for (uint32_t i = 0; i < available_splats; i++) {
			cpu_sort_original_indices_scratch[i] = cpu_cull_state.culled_indices[i];
		}

		const uint32_t original_distances_count = cpu_cull_state.culled_distances_sq.size();
		if ((uint32_t)cpu_sort_original_distances_scratch.size() < original_distances_count) {
			cpu_sort_original_distances_scratch.resize(original_distances_count);
		}
		for (uint32_t i = 0; i < original_distances_count; i++) {
			cpu_sort_original_distances_scratch[i] = cpu_cull_state.culled_distances_sq[i];
		}

		const uint32_t original_importance_count = cpu_cull_state.culled_importance_weights.size();
		if ((uint32_t)cpu_sort_original_importance_scratch.size() < original_importance_count) {
			cpu_sort_original_importance_scratch.resize(original_importance_count);
		}
		for (uint32_t i = 0; i < original_importance_count; i++) {
			cpu_sort_original_importance_scratch[i] = cpu_cull_state.culled_importance_weights[i];
		}

		if ((uint32_t)cpu_sort_entries_scratch.size() < available_splats) {
			cpu_sort_entries_scratch.resize(available_splats);
		}
		CpuSortEntry *entries = cpu_sort_entries_scratch.ptr();
		bool positions_ready = true;
		for (uint32_t i = 0; i < available_splats; i++) {
			Vector3 position;
			if (!_get_sort_position(*renderer, cpu_sort_original_indices_scratch[i], position)) {
				positions_ready = false;
				break;
			}
			Vector3 view_pos = p_world_to_camera_transform.xform(position);
			float depth = -view_pos.z;
			if (Math::is_nan(depth) || Math::is_inf(depth)) {
				depth = 1e10f;
			}
			entries[i].depth = depth;
			entries[i].index = cpu_sort_original_indices_scratch[i];
			entries[i].source_index = i;
		}

		if (strict_global_sort && !positions_ready) {
			GS_LOG_WARN_DEFAULT(vformat("[CPU Sort] Strict global sort: positions unavailable, rendering unsorted fallback (%s).",
					p_reason));
		}

		if (positions_ready) {
			struct CpuSortComparator {
				_FORCE_INLINE_ bool operator()(const CpuSortEntry &a, const CpuSortEntry &b) const {
					return a.depth < b.depth;
				}
			};
			SortArray<CpuSortEntry, CpuSortComparator> sorter;
			sorter.sort(entries, static_cast<int>(available_splats));

			if ((uint32_t)cpu_cull_state.culled_indices.size() != available_splats) {
				cpu_cull_state.culled_indices.resize(available_splats);
			}
			if ((uint32_t)cpu_cull_state.culled_distances_sq.size() < available_splats) {
				cpu_cull_state.culled_distances_sq.resize(available_splats);
			}
			if ((uint32_t)cpu_cull_state.culled_importance_weights.size() < available_splats) {
				cpu_cull_state.culled_importance_weights.resize(available_splats);
			}

			for (uint32_t i = 0; i < available_splats; i++) {
				const CpuSortEntry &entry = entries[i];
				cpu_cull_state.culled_indices[i] = entry.index;
				if (entry.source_index < original_distances_count) {
					cpu_cull_state.culled_distances_sq[i] = cpu_sort_original_distances_scratch[entry.source_index];
				}
				if (entry.source_index < original_importance_count) {
					cpu_cull_state.culled_importance_weights[i] = cpu_sort_original_importance_scratch[entry.source_index];
				}
			}
		}

		sorting_state.sort_index_bytes.resize(available_splats * sizeof(uint32_t));
		uint32_t *sorted_indices = reinterpret_cast<uint32_t *>(sorting_state.sort_index_bytes.ptrw());
		for (uint32_t i = 0; i < available_splats; i++) {
			sorted_indices[i] = cpu_cull_state.culled_indices[i];
		}

		apply_sort_buffer_update(available_splats);

		uint64_t sort_end = OS::get_singleton()->get_ticks_usec();
		float cpu_time_ms = (sort_end - sort_start) / 1000.0f;

		sorting_state.sorted_splat_count = available_splats;
		renderer->get_frame_state().visible_splat_count.store(available_splats, std::memory_order_release);
		renderer->get_performance_state().metrics.rendered_splat_count =
				renderer->get_frame_state().visible_splat_count.load(std::memory_order_acquire);
		renderer->get_frame_state().sort_time_ms = cpu_time_ms;
		renderer->get_performance_state().metrics.sort_submission_time_ms = cpu_time_ms;
		renderer->get_performance_state().metrics.sort_wait_time_ms = 0.0f;
		renderer->get_performance_state().metrics.async_sort_used = false;
		renderer->get_performance_state().metrics.async_sort_waited = false;
		renderer->get_performance_state().metrics.async_overlap_efficiency = 0.0f;

		sorting_state.last_sort_world_to_camera_transform = p_world_to_camera_transform;
		sorting_state.last_sort_transform_valid = true;
		gpu_culler->get_config().cull_params_dirty = false;

		StringName algorithm = positions_ready ? StringName(p_forced ? "CPU (forced)" : "CPU") : StringName("CPU (unsorted)");
		String switch_reason = p_forced
				? String("force_cpu_sort override enabled")
				: p_reason;
		if (!positions_ready) {
			switch_reason += " (positions unavailable)";
		}
		set_active_sort_algorithm(String(algorithm), switch_reason);
		record_cpu_sort_sample(available_splats, cpu_time_ms, algorithm);
		refresh_cull_signature_tracking();

		if (!positions_ready) {
			GS_LOG_WARN_DEFAULT(vformat("[CPU Sort] %s; using unsorted culled order", p_reason));
		}
		return true;
	};

	auto execute_sort_fallback_policy = [&](GaussianSplatting::SortFallbackScenario p_scenario,
											const String &p_reason,
											const String &p_failure_log = String()) {
		const GaussianSplatting::SortFallbackPolicyDecision policy =
				GaussianSplatting::build_sort_fallback_policy(p_scenario, instance_pipeline_active);
		const char *reuse_route_uid = instance_pipeline_active
				? RenderRouteUID::INSTANCE_SORT_CACHED
				: RenderRouteUID::COMMON_FAIL_SORT_FAILED;
		for (uint32_t i = 0; i < policy.action_count; i++) {
			switch (policy.actions[i]) {
				case GaussianSplatting::SortFallbackAction::REUSE_PREVIOUS_SORT:
					if (reuse_previous_sort(p_reason, reuse_route_uid)) {
						return;
					}
					break;
				case GaussianSplatting::SortFallbackAction::PUBLISH_INSTANCE_IDENTITY:
					if (publish_instance_identity_fallback(p_reason)) {
						reset_sort_metrics();
						return;
					}
					break;
				case GaussianSplatting::SortFallbackAction::RUN_CPU_SORT:
					if (run_cpu_sort(p_reason, policy.cpu_sort_forced)) {
						renderer->get_debug_state().sort_route_uid = RenderRouteUID::INSTANCE_SORT_CPU_FALLBACK;
						return;
					}
					break;
				case GaussianSplatting::SortFallbackAction::FAIL:
				default:
					if (!p_failure_log.is_empty()) {
						GS_LOG_WARN_DEFAULT(p_failure_log);
					}
					reset_sort_metrics();
					renderer->get_debug_state().sort_route_uid = RenderRouteUID::COMMON_FAIL_SORT_FAILED;
					sorting_state.sorted_splat_count = 0;
					sorting_state.last_sort_transform_valid = false;
					renderer->get_frame_state().visible_splat_count.store(0, std::memory_order_release);
					return;
			}
		}
		reset_sort_metrics();
		renderer->get_debug_state().sort_route_uid = RenderRouteUID::COMMON_FAIL_SORT_FAILED;
		sorting_state.sorted_splat_count = 0;
		sorting_state.last_sort_transform_valid = false;
		renderer->get_frame_state().visible_splat_count.store(0, std::memory_order_release);
	};

	if (force_cpu_sort) {
		const String reason = instance_pipeline_active
				? String("force_cpu_sort override incompatible with instance pipeline")
				: String("CPU sort forced");
		const String failure_log = instance_pipeline_active
				? String("[GPU Sort] force_cpu_sort override requested with instance pipeline but no valid fallback was available")
				: String();
		execute_sort_fallback_policy(GaussianSplatting::SortFallbackScenario::FORCE_CPU_OVERRIDE, reason, failure_log);
		return build_summary();
	}

	if (!renderer->ensure_rendering_device("sort_gaussians_for_view")) {
		GS_LOG_ERROR_DEFAULT("[GPU Sort] Rendering device unavailable; skipping sort");
		reset_sort_metrics();
		renderer->get_debug_state().sort_route_uid = RenderRouteUID::COMMON_FAIL_NO_DEVICE;
		sorting_state.sorted_splat_count = 0;
		sorting_state.last_sort_transform_valid = false;
		renderer->get_frame_state().visible_splat_count.store(0, std::memory_order_release);
		return build_summary();
	}

	if (sorting_state.sorter_needs_rebuild) {
		refresh_gpu_sorter("sort_gaussians_for_view");
	}

	if (!sorting_state.gpu_sorter.is_valid()) {
		GS_LOG_WARN_DEFAULT(vformat("[GPU Sort] Requested sorter unavailable (requested=%s); falling back",
				forced_algorithm_name));
		execute_sort_fallback_policy(
				GaussianSplatting::SortFallbackScenario::SORTER_UNAVAILABLE,
				vformat("Requested sorter unavailable (requested=%s)", forced_algorithm_name));
		return build_summary();
	}

	if (sorting_pipeline &&
			sorting_pipeline->sort_gaussians_gpu(renderer, p_world_to_camera_transform)) {
		renderer->get_debug_state().sort_route_uid = RenderRouteUID::INSTANCE_SORT_GPU;
		sorting_state.last_sort_world_to_camera_transform = p_world_to_camera_transform;
		sorting_state.last_sort_transform_valid = true;
		gpu_culler->get_config().cull_params_dirty = false;
		refresh_cull_signature_tracking();
		String gpu_algorithm = sorting_state.gpu_sorter->get_algorithm_name();
		if (gpu_algorithm.is_empty()) {
			gpu_algorithm = "GPU";
		}
		String gpu_switch_reason = force_algorithm
				? vformat("forced GPU algorithm override: %s", forced_algorithm_name)
				: String("automatic GPU algorithm selection");
		set_active_sort_algorithm(gpu_algorithm, gpu_switch_reason);
		record_gpu_sort_sample();
		return build_summary();
	}

	GS_LOG_WARN_DEFAULT("[GPU Sort] GPU sort failed; falling back");
	execute_sort_fallback_policy(
			GaussianSplatting::SortFallbackScenario::GPU_SORT_FAILED,
			"GPU sort failed");
	return build_summary();
}

void RenderSortingOrchestrator::force_sort_for_view(const Transform3D &p_world_to_camera_transform) {
	if (!renderer->ensure_rendering_device("force_sort_for_view")) {
		return;
	}

	Transform3D view_transform = p_world_to_camera_transform.affine_inverse();

	Projection projection = renderer->get_view_state().last_camera_projection;
	if (projection.get_z_far() <= projection.get_z_near()) {
		projection.set_perspective(60.0f, 16.0f / 9.0f, 0.1f, 1000.0f);
	}

	Size2i viewport_size = renderer->get_view_state().manual_viewport_override;
	if (viewport_size.x <= 0 || viewport_size.y <= 0) {
		viewport_size = Size2i(1280, 720);
	}

	// Runtime validation invokes force_sort_for_view outside render_scene_instance().
	// Route through streaming orchestration first so instance cull/sort buffers are
	// built and GPU radix has valid inputs.
	if (renderer->streaming_orchestrator &&
			renderer->streaming_orchestrator->ensure_instance_streaming_system()) {
		auto &streaming_state = renderer->get_streaming_state();
		if (streaming_state.current_streaming_system.is_valid()) {
			const bool stream_rendered = renderer->streaming_orchestrator->render_streaming_frame(
					nullptr,
					p_world_to_camera_transform,
					view_transform,
					projection,
					projection,
					nullptr,
					true);
			if (stream_rendered) {
				return;
			}
			WARN_PRINT_ONCE("[GaussianSplatRenderer] force_sort_for_view streaming path not ready; using resident cull+sort fallback.");
		}
	}

	GaussianRenderState::CullStageOutput cull_output = cull_for_view(view_transform, projection, viewport_size);
	(void)sort_gaussians_for_view(view_transform, cull_output.visible_domain);
}

void GaussianSplatRenderer::refresh_gpu_sorter(const char *p_context) {
	sorting_orchestrator->refresh_gpu_sorter(p_context);
}

void GaussianSplatRenderer::initialize_sorting() {
	sorting_orchestrator->initialize_sorting();
}

Array GaussianSplatRenderer::run_sort_benchmark(const PackedInt32Array &p_sizes) {
	return sorting_orchestrator->run_sort_benchmark(p_sizes);
}

void GaussianSplatRenderer::benchmark_sorting_performance() {
	sorting_orchestrator->benchmark_sorting_performance();
}

void GaussianSplatRenderer::force_sort_for_view(const Transform3D &p_world_to_camera_transform) {
	RenderingServer *rs = RenderingServer::get_singleton();
	bool dispatch_submitted = false;
	if (rs && !rs->is_on_render_thread()) {
		if (_dispatch_call_on_render_thread_blocking(
					callable_mp(this, &GaussianSplatRenderer::_force_sort_for_view_on_render_thread).bind(p_world_to_camera_transform),
					&dispatch_submitted)) {
			return;
		}
		if (dispatch_submitted) {
			GS_LOG_RENDERER_WARN("[GaussianSplatRenderer] force_sort_for_view dispatch timed out after submit; skipping unsafe local fallback");
			return;
		}
	}
	sorting_orchestrator->force_sort_for_view(p_world_to_camera_transform);
}

void GaussianSplatRenderer::_force_sort_for_view_on_render_thread(const Transform3D &p_world_to_camera_transform, uint64_t p_request_id) {
	sorting_orchestrator->force_sort_for_view(p_world_to_camera_transform);
	_notify_render_thread_dispatch_completed(p_request_id);
}

// Sort cache methods (merged from RenderSortCacheOrchestrator)

void RenderSortingOrchestrator::set_static_sort_cache_enabled(bool p_enabled) {
	gpu_culler->get_state().static_sort_cache_enabled = p_enabled;
	if (!p_enabled) {
		instance_sort_cache.valid = false;
	}
}

void RenderSortingOrchestrator::invalidate_static_chunk_caches(bool p_free_rids) {
	instance_sort_cache.valid = false;

	if (p_free_rids) {
		gpu_culler->get_state().visible_static_chunk_indices.clear();
	}
}

const Transform3D &RenderSortingOrchestrator::_get_camera_to_world_cached(const Transform3D &p_world_to_camera_transform) {
	if (!cached_camera_to_world_valid ||
			!p_world_to_camera_transform.origin.is_equal_approx(cached_world_to_camera.origin) ||
			!p_world_to_camera_transform.basis.is_equal_approx(cached_world_to_camera.basis)) {
		cached_world_to_camera = p_world_to_camera_transform;
		cached_camera_to_world = p_world_to_camera_transform.affine_inverse();
		cached_camera_to_world_valid = true;
	}
	return cached_camera_to_world;
}

bool RenderSortingOrchestrator::try_reuse_instance_sort_cache(const Transform3D &p_world_to_camera_transform,
		uint64_t p_content_generation, uint32_t p_max_visible_splats, uint32_t p_visible_chunk_count,
		uint32_t &r_sorted_count) {
	const Transform3D &camera_to_world = _get_camera_to_world_cached(p_world_to_camera_transform);
	return _try_reuse_instance_sort_cache_with_camera(camera_to_world, p_content_generation,
			p_max_visible_splats, p_visible_chunk_count, r_sorted_count);
}

void RenderSortingOrchestrator::update_instance_sort_cache(const Transform3D &p_world_to_camera_transform,
		uint64_t p_content_generation, uint32_t p_max_visible_splats, uint32_t p_visible_chunk_count,
		uint32_t p_sorted_count) {
	const Transform3D &camera_to_world = _get_camera_to_world_cached(p_world_to_camera_transform);
	_update_instance_sort_cache_with_camera(camera_to_world, p_content_generation, p_max_visible_splats,
			p_visible_chunk_count, p_sorted_count);
}

bool RenderSortingOrchestrator::_try_reuse_instance_sort_cache_with_camera(const Transform3D &p_camera_to_world,
		uint64_t p_content_generation, uint32_t p_max_visible_splats, uint32_t p_visible_chunk_count,
		uint32_t &r_sorted_count) {
	if (!gpu_culler->get_state().static_sort_cache_enabled) {
		return false;
	}
	if (!instance_sort_cache.valid || p_max_visible_splats == 0) {
		renderer->get_performance_state().metrics.sort_cache_misses++;
		return false;
	}
	if (instance_sort_cache.content_generation != p_content_generation ||
			instance_sort_cache.max_visible_splats != p_max_visible_splats ||
			instance_sort_cache.visible_chunk_count != p_visible_chunk_count) {
		renderer->get_performance_state().metrics.sort_cache_misses++;
		return false;
	}
	if (!sorting_pipeline || !sorting_pipeline->get_sort_indices_buffer().is_valid()) {
		renderer->get_performance_state().metrics.sort_cache_misses++;
		return false;
	}

	Vector3 camera_forward = -p_camera_to_world.basis.get_column(2);
	if (!camera_forward.is_zero_approx()) {
		camera_forward.normalize();
	} else {
		camera_forward = Vector3(0, 0, -1);
	}
	Vector3 camera_position = p_camera_to_world.origin;

	float dot = instance_sort_cache.camera_direction.is_zero_approx()
			? -1.0f
			: camera_forward.dot(instance_sort_cache.camera_direction);
	if (gpu_culler->get_state().sort_cache_angle_cos_threshold > 0.0f &&
			dot < gpu_culler->get_state().sort_cache_angle_cos_threshold) {
		renderer->get_performance_state().metrics.sort_cache_misses++;
		return false;
	}

	if (gpu_culler->get_state().sort_cache_position_threshold_sq > 0.0f) {
		Vector3 delta = instance_sort_cache.camera_position - camera_position;
		if (delta.length_squared() > gpu_culler->get_state().sort_cache_position_threshold_sq) {
			renderer->get_performance_state().metrics.sort_cache_misses++;
			return false;
		}
	}

	instance_sort_cache.camera_direction = camera_forward;
	instance_sort_cache.camera_position = camera_position;
	r_sorted_count = p_visible_chunk_count == 0
			? 0u
			: MIN(instance_sort_cache.sorted_count, p_max_visible_splats);
	renderer->get_performance_state().metrics.sort_cache_hits++;
	return true;
}

void RenderSortingOrchestrator::_update_instance_sort_cache_with_camera(const Transform3D &p_camera_to_world,
		uint64_t p_content_generation, uint32_t p_max_visible_splats, uint32_t p_visible_chunk_count,
		uint32_t p_sorted_count) {
	if (!gpu_culler->get_state().static_sort_cache_enabled) {
		instance_sort_cache.valid = false;
		return;
	}
	if (p_max_visible_splats == 0) {
		instance_sort_cache.valid = false;
		return;
	}

	Vector3 camera_forward = -p_camera_to_world.basis.get_column(2);
	if (!camera_forward.is_zero_approx()) {
		camera_forward.normalize();
	} else {
		camera_forward = Vector3(0, 0, -1);
	}

	instance_sort_cache.camera_direction = camera_forward;
	instance_sort_cache.camera_position = p_camera_to_world.origin;
	instance_sort_cache.content_generation = p_content_generation;
	instance_sort_cache.max_visible_splats = p_max_visible_splats;
	instance_sort_cache.visible_chunk_count = p_visible_chunk_count;
	instance_sort_cache.sorted_count = p_sorted_count;
	instance_sort_cache.valid = true;
}

void GaussianSplatRenderer::set_static_sort_cache_enabled(bool p_enabled) {
	sorting_orchestrator->set_static_sort_cache_enabled(p_enabled);
}

void GaussianSplatRenderer::_invalidate_static_chunk_caches(bool p_free_rids) {
	sorting_orchestrator->invalidate_static_chunk_caches(p_free_rids);
}

bool GaussianSplatRenderer::_try_reuse_instance_sort_cache(const Transform3D &p_world_to_camera_transform,
		uint64_t p_content_generation, uint32_t p_max_visible_splats, uint32_t p_visible_chunk_count,
		uint32_t &r_sorted_count) {
	return sorting_orchestrator->try_reuse_instance_sort_cache(p_world_to_camera_transform, p_content_generation,
			p_max_visible_splats, p_visible_chunk_count, r_sorted_count);
}

void GaussianSplatRenderer::_update_instance_sort_cache(const Transform3D &p_world_to_camera_transform,
		uint64_t p_content_generation, uint32_t p_max_visible_splats, uint32_t p_visible_chunk_count,
		uint32_t p_sorted_count) {
	sorting_orchestrator->update_instance_sort_cache(p_world_to_camera_transform, p_content_generation,
			p_max_visible_splats, p_visible_chunk_count, p_sorted_count);
}
