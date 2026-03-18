/**
 * tile_render_binning.cpp — TileRenderer::TileBinningStage method implementations.
 *
 * Companion .cpp for tile_renderer.h / tile_render_stages.h.
 * Contains tile binning count/emit dispatches, uniform set acquisition
 * (buffer, count, param, lighting), and tile count clearing.
 *
 * Pattern 10 (Flyweight + GPU resource cache): Binning uniform sets are
 * cached flyweight references into shared GPU resources; invalidated when
 * dependencies (gaussian_buffer, sorted_indices, etc.) change.
 */

#include "tile_renderer.h"
#include "core/error/error_macros.h"
#include "core/os/os.h"
#include "core/math/vector3.h"
#include "core/math/vector4.h"
#include "core/math/math_funcs.h"
#include "core/object/callable_method_pointer.h"
#include "servers/rendering/rendering_device.h"
#include "core/templates/hash_map.h"
#include "servers/rendering/renderer_rd/storage_rd/light_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/texture_storage.h"

#include "gpu_debug_utils.h"
#include "gpu_performance_monitor.h"
#include "gpu_sorter.h"
#include "gpu_sorting_config.h"
#include "quantization_config.h"
#include "resource_owner_mismatch_contract.h"
#include "../logger/gs_logger.h"
#include "gaussian_gpu_layout.h"
#include "pipeline_io_contracts.h"
#include "shader_compilation_helper.h"
#include "sh_config.h"
#include "tile_prefix_scan_utils.h"
#include "../interfaces/sync_policy.h"
#include "../shaders/tile_resolve.glsl.gen.h"

using GaussianSplatting::PassColors;
using GaussianSplatting::ScopedGpuMarker;
using GaussianSplatting::ScopedGpuMarkerEx;

#include <algorithm>
#include <cmath>
#include <cstring>

namespace {

static void _release_uniform_set(RenderingDevice *p_device, RID &p_uniform_set) {
	if (!p_uniform_set.is_valid()) {
		return;
	}
	// ISSUE-010: Guard against null device pointers. The uniform_set_is_valid
	// call also serves as a liveness check - if the device has been destroyed,
	// its RID table is cleared and this returns false, preventing the free() call.
	if (p_device && p_device->uniform_set_is_valid(p_uniform_set)) {
		p_device->free(p_uniform_set);
	}
	p_uniform_set = RID();
}

} // namespace

uint64_t TileRenderer::TileBinningStage::dispatch_tile_binning(uint32_t p_gaussian_count, RID p_buffer_uniform_set, RID p_param_uniform_set,
		RID p_lighting_uniform_set, RenderingDevice *p_submission_device, bool p_requires_sync) {
	if (!owner.shader_resources.tile_binning_pipeline.is_valid()) {
		WARN_PRINT_ONCE("[TileRenderer] Binning pipeline invalid - shader compilation failed");
		return 0;
	}
	const bool use_instance_indirect = owner.instance_pipeline_buffers.indirect_dispatch_buffer.is_valid();
	if (p_gaussian_count == 0 && !use_instance_indirect) {
		WARN_PRINT_ONCE("[TileRenderer] Binning skipped - zero gaussians");
		return 0;
	}

	uint32_t dispatch_x = 0;
	if (!use_instance_indirect) {
		dispatch_x = (p_gaussian_count + TileRenderer::BINNING_GROUP_SIZE - 1) / TileRenderer::BINNING_GROUP_SIZE;
		if (dispatch_x == 0) {
			return 0;
		}
	}

	RenderingDevice *submission_device = p_submission_device;
	if (!submission_device) {
		return 0;
	}

	uint32_t timestamp_base = submission_device->get_captured_timestamps_count();
	String binning_label = "TileBinning_" + String::num_uint64(owner.frame_state.current_frame_serial);
	submission_device->capture_timestamp(binning_label + String("_Begin"));
	ScopedGpuMarker binning_marker(submission_device, "GS_TileBinning_EMIT", Color(0.3f, 0.9f, 0.3f, 1.0f));

	RD::ComputeListID compute_list = submission_device->compute_list_begin();
	if (compute_list == RD::INVALID_ID) {
		ERR_PRINT_ONCE("[TileRenderer] Failed to begin binning emit compute list");
		return 0;
	}
	submission_device->compute_list_bind_compute_pipeline(compute_list, owner.shader_resources.tile_binning_pipeline);
	submission_device->compute_list_bind_uniform_set(compute_list, p_buffer_uniform_set, 0);
	submission_device->compute_list_bind_uniform_set(compute_list, p_param_uniform_set, 1);
	if (p_lighting_uniform_set.is_valid()) {
		submission_device->compute_list_bind_uniform_set(compute_list, p_lighting_uniform_set, 2);
	}
	if (use_instance_indirect) {
		submission_device->compute_list_dispatch_indirect(compute_list,
				owner.instance_pipeline_buffers.indirect_dispatch_buffer, 0);
	} else {
		submission_device->compute_list_dispatch(compute_list, dispatch_x, 1, 1);
	}
	submission_device->compute_list_add_barrier(compute_list);
	submission_device->compute_list_end();
	submission_device->capture_timestamp(binning_label + String("_End"));

	owner.timing_state.binning_timestamp.device = submission_device;
	owner.timing_state.binning_timestamp.start_index = timestamp_base;
	owner.timing_state.binning_timestamp.end_index = timestamp_base + 1;
	owner.timing_state.binning_timestamp.label = binning_label;

	owner._queue_submission(submission_device, p_requires_sync);

	if (p_lighting_uniform_set.is_valid() && submission_device->uniform_set_is_valid(p_lighting_uniform_set)) {
		submission_device->free(p_lighting_uniform_set);
	}

	return 0;
}

uint64_t TileRenderer::TileBinningStage::dispatch_tile_binning_count(uint32_t p_gaussian_count, RID p_buffer_uniform_set,
		RID p_param_uniform_set, RID p_lighting_uniform_set, RenderingDevice *p_submission_device, bool p_requires_sync) {
	if (!owner.shader_resources.tile_binning_count_pipeline.is_valid()) {
		WARN_PRINT_ONCE("[TileRenderer] Global count pipeline invalid - shader compilation failed");
		return 0;
	}
	const bool use_instance_indirect = owner.instance_pipeline_buffers.indirect_dispatch_buffer.is_valid();
	if (p_gaussian_count == 0 && !use_instance_indirect) {
		return 0;
	}

	uint32_t dispatch_x = 0;
	if (!use_instance_indirect) {
		dispatch_x = (p_gaussian_count + TileRenderer::BINNING_GROUP_SIZE - 1) / TileRenderer::BINNING_GROUP_SIZE;
		if (dispatch_x == 0) {
			return 0;
		}
	}

	RenderingDevice *submission_device = p_submission_device;
	if (!submission_device) {
		return 0;
	}

	uint32_t timestamp_base = submission_device->get_captured_timestamps_count();
	String count_label = "TileOverlapCount_" + String::num_uint64(owner.frame_state.current_frame_serial);
	submission_device->capture_timestamp(count_label + String("_Begin"));
	ScopedGpuMarker overlap_marker(submission_device, "GS_TileBinning_COUNT", Color(0.4f, 0.85f, 0.35f, 1.0f));

#ifdef DEV_ENABLED
    static int binning_dispatch_count = 0;
    if (GaussianSplatting::is_debug_frame_logging_enabled() && ++binning_dispatch_count <= 5) {
        GS_LOG_RENDERER_DEBUG(vformat("[BINNING-COUNT] dispatch #%d: gaussians=%d dispatch_x=%d buffer_set=%s param_set=%s",
            binning_dispatch_count, p_gaussian_count, dispatch_x,
            p_buffer_uniform_set.is_valid() ? "valid" : "INVALID",
			p_param_uniform_set.is_valid() ? "valid" : "INVALID"));
	}
#endif

	RD::ComputeListID compute_list = submission_device->compute_list_begin();
	if (compute_list == RD::INVALID_ID) {
		ERR_PRINT_ONCE("[TileRenderer] Failed to begin binning count compute list");
		return 0;
	}

	submission_device->compute_list_bind_compute_pipeline(compute_list, owner.shader_resources.tile_binning_count_pipeline);
	submission_device->compute_list_bind_uniform_set(compute_list, p_buffer_uniform_set, 0);
	submission_device->compute_list_bind_uniform_set(compute_list, p_param_uniform_set, 1);
	if (p_lighting_uniform_set.is_valid()) {
		submission_device->compute_list_bind_uniform_set(compute_list, p_lighting_uniform_set, 2);
	}
	if (use_instance_indirect) {
		submission_device->compute_list_dispatch_indirect(compute_list,
				owner.instance_pipeline_buffers.indirect_dispatch_buffer, 0);
	} else {
		submission_device->compute_list_dispatch(compute_list, dispatch_x, 1, 1);
	}
	submission_device->compute_list_add_barrier(compute_list);
	submission_device->compute_list_end();
	submission_device->capture_timestamp(count_label + String("_End"));

	owner._queue_submission(submission_device, p_requires_sync);

	if (p_lighting_uniform_set.is_valid() && submission_device->uniform_set_is_valid(p_lighting_uniform_set)) {
		submission_device->free(p_lighting_uniform_set);
	}

	return 0;
}

RID TileRenderer::TileBinningStage::acquire_binning_buffer_uniform_set(RenderingDevice *p_device, const RID &p_gaussian_buffer,
		const RID &p_sorted_indices) {
	ERR_FAIL_NULL_V(p_device, RID());
	ERR_FAIL_COND_V(!p_gaussian_buffer.is_valid(), RID());
	ERR_FAIL_COND_V(!p_sorted_indices.is_valid(), RID());
	ERR_FAIL_COND_V(!owner.debug_stats.overflow_statistics_buffer.is_valid(), RID());
	ERR_FAIL_COND_V(!owner.debug_stats.debug_counter_buffer.is_valid(), RID());
	ERR_FAIL_COND_V(!owner.debug_stats.debug_splat_audit_buffer.is_valid(), RID());
	ERR_FAIL_COND_V(!owner.projection_buffers.projection_buffer.is_valid(), RID());
	ERR_FAIL_COND_V(!owner.subpixel_history_buffers.subpixel_history_buffer.is_valid(), RID());
	ERR_FAIL_COND_V(!owner.subpixel_history_buffers.subpixel_history_buffer.is_valid(), RID());
	ERR_FAIL_COND_V(!owner.subpixel_visibility_buffers.subpixel_visibility_buffer.is_valid(), RID());
	if (owner.render_settings.enable_sh_amortization) {
		ERR_FAIL_COND_V(!owner.sh_cache_buffers.sh_color_cache.is_valid(), RID());
	}
	if (owner.render_settings.global_sort_enabled) {
		ERR_FAIL_COND_V(!owner.global_sort_resources.keys_buffer.is_valid(), RID());
		ERR_FAIL_COND_V(!owner.global_sort_resources.values_buffer.is_valid(), RID());
		ERR_FAIL_COND_V(!owner.global_sort_resources.get_tile_counts_buffer().is_valid(), RID());
		ERR_FAIL_COND_V(!owner.global_sort_resources.tile_ranges_buffer.is_valid(), RID());
		ERR_FAIL_COND_V(!owner.global_sort_resources.indirect_dispatch_buffer.is_valid(), RID());
	}

	const bool quantization_required = g_quantization_config.per_chunk_quantization;
	ERR_FAIL_COND_V(!owner.instance_pipeline_buffers.splat_ref_buffer.is_valid(), RID());
	ERR_FAIL_COND_V(!owner.instance_pipeline_buffers.instance_buffer.is_valid(), RID());
	ERR_FAIL_COND_V(!owner.instance_pipeline_buffers.indirect_count_buffer.is_valid(), RID());
	ERR_FAIL_COND_V(!owner.instance_pipeline_buffers.indirect_dispatch_buffer.is_valid(), RID());
	ERR_FAIL_COND_V(quantization_required && !owner.instance_pipeline_buffers.quantization_buffer.is_valid(), RID());

	RID tile_counts_buffer = owner.global_sort_resources.get_tile_counts_buffer();
	const bool cache_deps_match = cached_generation == owner.descriptor_generation &&
			cached_binning_buffer_device == p_device &&
			cached_binning_gaussian_buffer == p_gaussian_buffer &&
			cached_binning_sorted_indices == p_sorted_indices;
	if (!cache_deps_match) {
		if (cached_binning_buffer_uniform_set.is_valid()) {
			RenderingDevice *owner_device = cached_binning_buffer_device ? cached_binning_buffer_device : p_device;
			_release_uniform_set(owner_device, cached_binning_buffer_uniform_set);
		}
		if (cached_binning_buffer_uniform_set_alt.is_valid()) {
			RenderingDevice *owner_device = cached_binning_buffer_device ? cached_binning_buffer_device : p_device;
			_release_uniform_set(owner_device, cached_binning_buffer_uniform_set_alt);
		}
		cached_binning_buffer_uniform_set = RID();
		cached_binning_buffer_uniform_set_alt = RID();
		cached_binning_buffer_device = nullptr;
		cached_binning_tile_counts = RID();
		cached_binning_tile_counts_alt = RID();
	}

	if (cache_deps_match && cached_binning_buffer_uniform_set.is_valid() &&
			cached_binning_tile_counts == tile_counts_buffer) {
		return cached_binning_buffer_uniform_set;
	}
	if (cache_deps_match && cached_binning_buffer_uniform_set_alt.is_valid() &&
			cached_binning_tile_counts_alt == tile_counts_buffer) {
		return cached_binning_buffer_uniform_set_alt;
	}

    Vector<RD::Uniform> uniforms;

    RD::Uniform gaussian_uniform;
    gaussian_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
    gaussian_uniform.binding = 0;
    gaussian_uniform.append_id(p_gaussian_buffer);
    uniforms.push_back(gaussian_uniform);

    RD::Uniform indices_uniform;
    indices_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
    indices_uniform.binding = 1;
    indices_uniform.append_id(p_sorted_indices);
    uniforms.push_back(indices_uniform);

	RD::Uniform splat_ref_uniform;
	splat_ref_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	splat_ref_uniform.binding = 12;
	splat_ref_uniform.append_id(owner.instance_pipeline_buffers.splat_ref_buffer);
	uniforms.push_back(splat_ref_uniform);

	RD::Uniform instance_uniform;
	instance_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	instance_uniform.binding = 13;
	instance_uniform.append_id(owner.instance_pipeline_buffers.instance_buffer);
	uniforms.push_back(instance_uniform);

	if (quantization_required) {
		RD::Uniform quant_uniform;
		quant_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
		quant_uniform.binding = 14;
		quant_uniform.append_id(owner.instance_pipeline_buffers.quantization_buffer);
		uniforms.push_back(quant_uniform);
	}

	RD::Uniform indirect_uniform;
	indirect_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	indirect_uniform.binding = 15;
	indirect_uniform.append_id(owner.instance_pipeline_buffers.indirect_count_buffer);
	uniforms.push_back(indirect_uniform);

	RD::Uniform history_uniform;
	history_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	history_uniform.binding = 16;
	history_uniform.append_id(owner.subpixel_history_buffers.subpixel_history_buffer);
	uniforms.push_back(history_uniform);

	RD::Uniform visibility_uniform;
	visibility_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	visibility_uniform.binding = 17;
	visibility_uniform.append_id(owner.subpixel_visibility_buffers.subpixel_visibility_buffer);
	uniforms.push_back(visibility_uniform);

	if (owner.render_settings.global_sort_enabled) {
		RD::Uniform keys_uniform;
		keys_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
		keys_uniform.binding = 2;
		keys_uniform.append_id(owner.global_sort_resources.keys_buffer);
		uniforms.push_back(keys_uniform);

		RD::Uniform stats_uniform;
		stats_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
		stats_uniform.binding = 3;
		stats_uniform.append_id(owner.debug_stats.overflow_statistics_buffer);
		uniforms.push_back(stats_uniform);

		RD::Uniform values_uniform;
		values_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
		values_uniform.binding = 4;
		values_uniform.append_id(owner.global_sort_resources.values_buffer);
		uniforms.push_back(values_uniform);

		RD::Uniform counts_uniform;
		counts_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
		counts_uniform.binding = 5;
		counts_uniform.append_id(owner.global_sort_resources.get_tile_counts_buffer());
		uniforms.push_back(counts_uniform);

			RD::Uniform projection_uniform;
			projection_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			projection_uniform.binding = 7;
			projection_uniform.append_id(owner.projection_buffers.projection_buffer);
			uniforms.push_back(projection_uniform);

			RD::Uniform ranges_uniform;
			ranges_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			ranges_uniform.binding = 8;
			ranges_uniform.append_id(owner.global_sort_resources.tile_ranges_buffer);
			uniforms.push_back(ranges_uniform);

			RD::Uniform indirect_uniform;
			indirect_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			indirect_uniform.binding = 9;
			indirect_uniform.append_id(owner.global_sort_resources.indirect_dispatch_buffer);
			uniforms.push_back(indirect_uniform);
	}

	RD::Uniform debug_uniform;
	debug_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	debug_uniform.binding = 6;
	debug_uniform.append_id(owner.debug_stats.debug_counter_buffer);
    uniforms.push_back(debug_uniform);

	RD::Uniform audit_uniform;
	audit_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	audit_uniform.binding = 10;
	audit_uniform.append_id(owner.debug_stats.debug_splat_audit_buffer);
	uniforms.push_back(audit_uniform);

	if (owner.render_settings.enable_sh_amortization) {
		RD::Uniform sh_cache_uniform;
		sh_cache_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
		sh_cache_uniform.binding = 11;
		sh_cache_uniform.append_id(owner.sh_cache_buffers.sh_color_cache);
		uniforms.push_back(sh_cache_uniform);
	}

    // ISSUE-002: Verify key buffers belong to p_device before creating uniform set.
    ERR_FAIL_COND_V(!_verify_buffer_device_ownership(p_device, p_gaussian_buffer, "binning:gaussian_buffer"), RID());
    ERR_FAIL_COND_V(!_verify_buffer_device_ownership(p_device, p_sorted_indices, "binning:sorted_indices"), RID());
    ERR_FAIL_COND_V(!_verify_buffer_device_ownership(p_device, owner.projection_buffers.projection_buffer, "binning:projection_buffer"), RID());

    RID new_uniform_set = p_device->uniform_set_create(uniforms, owner.shader_resources.tile_binning_shader, 0);
    if (!new_uniform_set.is_valid()) {
        ERR_FAIL_V_MSG(RID(), "[TileRenderer] Failed to create binning buffer uniform set (projection buffer path)");
    }
    p_device->set_resource_name(new_uniform_set, "GS_TileRenderer_BinningBufferSet");

    RID *target_set = &cached_binning_buffer_uniform_set;
    RID *target_counts = &cached_binning_tile_counts;
    if (cached_binning_buffer_uniform_set.is_valid()) {
        if (!cached_binning_buffer_uniform_set_alt.is_valid()) {
            target_set = &cached_binning_buffer_uniform_set_alt;
            target_counts = &cached_binning_tile_counts_alt;
        }
    }
    *target_set = new_uniform_set;
    *target_counts = tile_counts_buffer;

    cached_binning_gaussian_buffer = p_gaussian_buffer;
    cached_binning_sorted_indices = p_sorted_indices;
    cached_binning_buffer_device = p_device;
    cached_generation = owner.descriptor_generation;

    return *target_set;
}

RID TileRenderer::TileBinningStage::acquire_binning_count_uniform_set(RenderingDevice *p_device, const RID &p_gaussian_buffer,
		const RID &p_sorted_indices) {
	ERR_FAIL_NULL_V(p_device, RID());
	ERR_FAIL_COND_V(!owner.shader_resources.tile_binning_count_shader.is_valid(), RID());
	ERR_FAIL_COND_V(!p_gaussian_buffer.is_valid(), RID());
	ERR_FAIL_COND_V(!p_sorted_indices.is_valid(), RID());
	ERR_FAIL_COND_V(!owner.global_sort_resources.get_tile_counts_buffer().is_valid(), RID());
	ERR_FAIL_COND_V(!owner.debug_stats.overflow_statistics_buffer.is_valid(), RID());
	ERR_FAIL_COND_V(!owner.debug_stats.debug_counter_buffer.is_valid(), RID());
	ERR_FAIL_COND_V(!owner.debug_stats.debug_splat_audit_buffer.is_valid(), RID());
	ERR_FAIL_COND_V(!owner.projection_buffers.projection_buffer.is_valid(), RID());
	ERR_FAIL_COND_V(!owner.subpixel_history_buffers.subpixel_history_buffer.is_valid(), RID());
	ERR_FAIL_COND_V(!owner.subpixel_visibility_buffers.subpixel_visibility_buffer.is_valid(), RID());

	const bool quantization_required = g_quantization_config.per_chunk_quantization;
	ERR_FAIL_COND_V(!owner.instance_pipeline_buffers.splat_ref_buffer.is_valid(), RID());
	ERR_FAIL_COND_V(!owner.instance_pipeline_buffers.instance_buffer.is_valid(), RID());
	ERR_FAIL_COND_V(!owner.instance_pipeline_buffers.indirect_count_buffer.is_valid(), RID());
	ERR_FAIL_COND_V(!owner.instance_pipeline_buffers.indirect_dispatch_buffer.is_valid(), RID());
	ERR_FAIL_COND_V(quantization_required && !owner.instance_pipeline_buffers.quantization_buffer.is_valid(), RID());

	RID tile_counts_buffer = owner.global_sort_resources.get_tile_counts_buffer();
	const bool cache_deps_match = cached_generation == owner.descriptor_generation &&
			cached_binning_count_device == p_device &&
			cached_binning_count_gaussian_buffer == p_gaussian_buffer &&
			cached_binning_count_sorted_indices == p_sorted_indices;
	if (!cache_deps_match) {
		if (cached_binning_count_uniform_set.is_valid()) {
			RenderingDevice *owner_device = cached_binning_count_device ? cached_binning_count_device : p_device;
			_release_uniform_set(owner_device, cached_binning_count_uniform_set);
		}
		if (cached_binning_count_uniform_set_alt.is_valid()) {
			RenderingDevice *owner_device = cached_binning_count_device ? cached_binning_count_device : p_device;
			_release_uniform_set(owner_device, cached_binning_count_uniform_set_alt);
		}
		cached_binning_count_uniform_set = RID();
		cached_binning_count_uniform_set_alt = RID();
		cached_binning_count_device = nullptr;
		cached_binning_count_tile_counts = RID();
		cached_binning_count_tile_counts_alt = RID();
	}

	if (cache_deps_match && cached_binning_count_uniform_set.is_valid() &&
			cached_binning_count_tile_counts == tile_counts_buffer) {
		return cached_binning_count_uniform_set;
	}
	if (cache_deps_match && cached_binning_count_uniform_set_alt.is_valid() &&
			cached_binning_count_tile_counts_alt == tile_counts_buffer) {
		return cached_binning_count_uniform_set_alt;
	}

	Vector<RD::Uniform> uniforms;

	RD::Uniform gaussian_uniform;
	gaussian_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	gaussian_uniform.binding = 0;
	gaussian_uniform.append_id(p_gaussian_buffer);
	uniforms.push_back(gaussian_uniform);

	RD::Uniform indices_uniform;
	indices_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	indices_uniform.binding = 1;
	indices_uniform.append_id(p_sorted_indices);
	uniforms.push_back(indices_uniform);

	RD::Uniform splat_ref_uniform;
	splat_ref_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	splat_ref_uniform.binding = 12;
	splat_ref_uniform.append_id(owner.instance_pipeline_buffers.splat_ref_buffer);
	uniforms.push_back(splat_ref_uniform);

	RD::Uniform instance_uniform;
	instance_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	instance_uniform.binding = 13;
	instance_uniform.append_id(owner.instance_pipeline_buffers.instance_buffer);
	uniforms.push_back(instance_uniform);

	if (quantization_required) {
		RD::Uniform quant_uniform;
		quant_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
		quant_uniform.binding = 14;
		quant_uniform.append_id(owner.instance_pipeline_buffers.quantization_buffer);
		uniforms.push_back(quant_uniform);
	}

	RD::Uniform indirect_uniform;
	indirect_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	indirect_uniform.binding = 15;
	indirect_uniform.append_id(owner.instance_pipeline_buffers.indirect_count_buffer);
	uniforms.push_back(indirect_uniform);

	RD::Uniform history_uniform;
	history_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	history_uniform.binding = 16;
	history_uniform.append_id(owner.subpixel_history_buffers.subpixel_history_buffer);
	uniforms.push_back(history_uniform);

	RD::Uniform visibility_uniform;
	visibility_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	visibility_uniform.binding = 17;
	visibility_uniform.append_id(owner.subpixel_visibility_buffers.subpixel_visibility_buffer);
	uniforms.push_back(visibility_uniform);

	RD::Uniform stats_uniform;
	stats_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	stats_uniform.binding = 3;
	stats_uniform.append_id(owner.debug_stats.overflow_statistics_buffer);
	uniforms.push_back(stats_uniform);

	RD::Uniform counts_uniform;
	counts_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	counts_uniform.binding = 5;
	counts_uniform.append_id(owner.global_sort_resources.get_tile_counts_buffer());
	uniforms.push_back(counts_uniform);

	RD::Uniform debug_uniform;
	debug_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	debug_uniform.binding = 6;
	debug_uniform.append_id(owner.debug_stats.debug_counter_buffer);
	uniforms.push_back(debug_uniform);

	RD::Uniform audit_uniform;
	audit_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	audit_uniform.binding = 10;
	audit_uniform.append_id(owner.debug_stats.debug_splat_audit_buffer);
	uniforms.push_back(audit_uniform);

	RID new_uniform_set = p_device->uniform_set_create(uniforms, owner.shader_resources.tile_binning_count_shader, 0);
	ERR_FAIL_COND_V_MSG(!new_uniform_set.is_valid(), RID(),
			"[TileRenderer] Failed to create global count uniform set (tile_binning.glsl count pass)");
	p_device->set_resource_name(new_uniform_set, "GS_TileRenderer_BinningCountSet");

	RID *target_set = &cached_binning_count_uniform_set;
	RID *target_counts = &cached_binning_count_tile_counts;
	if (cached_binning_count_uniform_set.is_valid()) {
		if (!cached_binning_count_uniform_set_alt.is_valid()) {
			target_set = &cached_binning_count_uniform_set_alt;
			target_counts = &cached_binning_count_tile_counts_alt;
		}
	}
	*target_set = new_uniform_set;
	*target_counts = tile_counts_buffer;

	cached_binning_count_gaussian_buffer = p_gaussian_buffer;
	cached_binning_count_sorted_indices = p_sorted_indices;
	cached_binning_count_device = p_device;
	cached_generation = owner.descriptor_generation;
	return *target_set;
}

void TileRenderer::TileBinningStage::prepare_count_uniform_sets(RenderingDevice *p_device, const RID &p_gaussian_buffer,
		const RID &p_sorted_indices, const RenderParams &p_params, BinningUniformSets &r_sets) {
	r_sets.param_uniform_set = acquire_binning_param_uniform_set(p_device);
	r_sets.buffer_uniform_set = acquire_binning_count_uniform_set(p_device, p_gaussian_buffer, p_sorted_indices);
	r_sets.lighting_uniform_set = create_binning_lighting_uniform_set(p_device, p_params);
}

void TileRenderer::TileBinningStage::prepare_emit_uniform_sets(RenderingDevice *p_device, const RID &p_gaussian_buffer,
		const RID &p_sorted_indices, const RenderParams &p_params, BinningUniformSets &r_sets) {
	r_sets.param_uniform_set = acquire_binning_param_uniform_set(p_device);
	r_sets.buffer_uniform_set = acquire_binning_buffer_uniform_set(p_device, p_gaussian_buffer, p_sorted_indices);
	r_sets.lighting_uniform_set = create_binning_lighting_uniform_set(p_device, p_params);
}

void TileRenderer::TileBinningStage::clear_tile_counts(RenderingDevice *p_device) const {
	if (!p_device || owner.grid_state.total_tiles == 0) {
		return;
	}
	if (!owner.global_sort_resources.ensure_tile_counts_ready(p_device)) {
		return;
	}
}

RID TileRenderer::TileBinningStage::acquire_binning_param_uniform_set(RenderingDevice *p_device) {
	ERR_FAIL_NULL_V(p_device, RID());
	ERR_FAIL_COND_V(!owner.uniform_buffers.param_uniform_buffer.is_valid(), RID());

	if (cached_binning_param_uniform_set.is_valid() && cached_binning_param_device == p_device) {
		return cached_binning_param_uniform_set;
	}

	if (cached_binning_param_uniform_set.is_valid()) {
		RenderingDevice *owner_device = cached_binning_param_device ? cached_binning_param_device : p_device;
		_release_uniform_set(owner_device, cached_binning_param_uniform_set);
		cached_binning_param_uniform_set = RID();
		cached_binning_param_device = nullptr;
	}

	Vector<RD::Uniform> uniforms;
	RD::Uniform params_uniform;
	params_uniform.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
	params_uniform.binding = 0;
	params_uniform.append_id(owner.uniform_buffers.param_uniform_buffer);
	uniforms.push_back(params_uniform);

	cached_binning_param_uniform_set = p_device->uniform_set_create(uniforms, owner.shader_resources.tile_binning_shader, 1);
	ERR_FAIL_COND_V_MSG(!cached_binning_param_uniform_set.is_valid(), RID(), "[TileRenderer] Failed to create binning param uniform set");
	p_device->set_resource_name(cached_binning_param_uniform_set, "GS_TileRenderer_BinningParamSet");
	cached_binning_param_device = p_device;

	return cached_binning_param_uniform_set;
}

RID TileRenderer::TileBinningStage::create_binning_lighting_uniform_set(RenderingDevice *p_device, const RenderParams &p_params) {
	ERR_FAIL_NULL_V(p_device, RID());
	if (!owner.shader_resources.tile_binning_shader.is_valid()) {
		return RID();
	}
	if (!owner.resolve_stage.ensure_resolve_sampler(p_device)) {
		return RID();
	}
	if (!owner.resolve_stage.ensure_shadow_sampler(p_device)) {
		return RID();
	}

	RID scene_buffer = p_params.scene_uniform_buffer;
	if (!scene_buffer.is_valid()) {
		if (!owner.resolve_stage.ensure_fallback_lighting_buffers(p_device)) {
			return RID();
		}
		scene_buffer = owner.resolve_stage.fallback_scene_uniform_buffer;
	}

	RendererRD::LightStorage *light_storage = RendererRD::LightStorage::get_singleton();
	RID directional_buffer = p_params.directional_light_buffer;
	if (!directional_buffer.is_valid()) {
		directional_buffer = light_storage ? light_storage->get_directional_light_buffer() : RID();
	}
	if (!directional_buffer.is_valid()) {
		if (!owner.resolve_stage.ensure_fallback_lighting_buffers(p_device)) {
			return RID();
		}
		directional_buffer = owner.resolve_stage.fallback_directional_light_buffer;
	}

	RID omni_buffer = light_storage ? light_storage->get_omni_light_buffer() : RID();
	if (!omni_buffer.is_valid()) {
		if (!owner.resolve_stage.ensure_fallback_lighting_buffers(p_device)) {
			return RID();
		}
		omni_buffer = owner.resolve_stage.fallback_omni_light_buffer;
	}

	RID spot_buffer = light_storage ? light_storage->get_spot_light_buffer() : RID();
	if (!spot_buffer.is_valid()) {
		if (!owner.resolve_stage.ensure_fallback_lighting_buffers(p_device)) {
			return RID();
		}
		spot_buffer = owner.resolve_stage.fallback_spot_light_buffer;
	}

	RID reflection_buffer = light_storage ? light_storage->get_reflection_probe_buffer() : RID();
	if (!reflection_buffer.is_valid()) {
		if (!owner.resolve_stage.ensure_fallback_lighting_buffers(p_device)) {
			return RID();
		}
		reflection_buffer = owner.resolve_stage.fallback_reflection_buffer;
	}

	RID cluster_buffer = p_params.cluster_buffer;
	if (!cluster_buffer.is_valid()) {
		if (!owner.resolve_stage.ensure_fallback_lighting_buffers(p_device)) {
			return RID();
		}
		cluster_buffer = owner.resolve_stage.fallback_cluster_buffer;
	}

	RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();
	RID decal_texture;
	RID reflection_texture;
	RID shadow_atlas_texture;
	RID directional_shadow_texture;
	RID default_depth_texture;
	if (texture_storage) {
		decal_texture = texture_storage->decal_atlas_get_texture_srgb();
		if (!decal_texture.is_valid()) {
			decal_texture = texture_storage->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_BLACK);
		}
		reflection_texture = texture_storage->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_CUBEMAP_ARRAY_BLACK);
		default_depth_texture = texture_storage->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_DEPTH);
	}

	if (light_storage && p_params.shadow_atlas.is_valid()) {
		shadow_atlas_texture = light_storage->shadow_atlas_get_texture(p_params.shadow_atlas);
	}
	if (light_storage) {
		directional_shadow_texture = light_storage->directional_shadow_get_texture();
	}
	if (!shadow_atlas_texture.is_valid()) {
		shadow_atlas_texture = default_depth_texture;
	}
	if (!directional_shadow_texture.is_valid()) {
		directional_shadow_texture = default_depth_texture;
	}

	if (!decal_texture.is_valid() || !p_device->texture_is_valid(decal_texture)) {
		if (!owner.resolve_stage.ensure_fallback_lighting_buffers(p_device)) {
			return RID();
		}
		decal_texture = owner.resolve_stage.fallback_decal_texture;
	}

	if (!reflection_texture.is_valid() || !p_device->texture_is_valid(reflection_texture)) {
		if (!owner.resolve_stage.ensure_fallback_lighting_buffers(p_device)) {
			return RID();
		}
		reflection_texture = owner.resolve_stage.fallback_reflection_texture;
	}

	if (!shadow_atlas_texture.is_valid() || !p_device->texture_is_valid(shadow_atlas_texture)) {
		if (!owner.resolve_stage.ensure_fallback_lighting_buffers(p_device)) {
			return RID();
		}
		shadow_atlas_texture = owner.resolve_stage.fallback_shadow_texture;
	}

	if (!directional_shadow_texture.is_valid() || !p_device->texture_is_valid(directional_shadow_texture)) {
		if (!owner.resolve_stage.ensure_fallback_lighting_buffers(p_device)) {
			return RID();
		}
		directional_shadow_texture = owner.resolve_stage.fallback_directional_shadow_texture;
	}

	Vector<RD::Uniform> uniforms;
	RD::Uniform scene_uniform;
	scene_uniform.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
	scene_uniform.binding = 0;
	scene_uniform.append_id(scene_buffer);
	uniforms.push_back(scene_uniform);

	RD::Uniform directional_uniform;
	directional_uniform.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
	directional_uniform.binding = 1;
	directional_uniform.append_id(directional_buffer);
	uniforms.push_back(directional_uniform);

	RD::Uniform omni_uniform;
	omni_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	omni_uniform.binding = 2;
	omni_uniform.append_id(omni_buffer);
	uniforms.push_back(omni_uniform);

	RD::Uniform spot_uniform;
	spot_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	spot_uniform.binding = 3;
	spot_uniform.append_id(spot_buffer);
	uniforms.push_back(spot_uniform);

	RD::Uniform reflection_uniform;
	reflection_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	reflection_uniform.binding = 4;
	reflection_uniform.append_id(reflection_buffer);
	uniforms.push_back(reflection_uniform);

	RD::Uniform cluster_uniform;
	cluster_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	cluster_uniform.binding = 9;
	cluster_uniform.append_id(cluster_buffer);
	uniforms.push_back(cluster_uniform);

	RD::Uniform decal_uniform;
	decal_uniform.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
	decal_uniform.binding = 5;
	decal_uniform.append_id(decal_texture);
	uniforms.push_back(decal_uniform);

	RD::Uniform reflection_atlas_uniform;
	reflection_atlas_uniform.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
	reflection_atlas_uniform.binding = 6;
	reflection_atlas_uniform.append_id(reflection_texture);
	uniforms.push_back(reflection_atlas_uniform);

	RD::Uniform projector_sampler_uniform;
	projector_sampler_uniform.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
	projector_sampler_uniform.binding = 7;
	projector_sampler_uniform.append_id(owner.resolve_stage.resolve_sampler);
	uniforms.push_back(projector_sampler_uniform);

	RD::Uniform default_sampler_uniform;
	default_sampler_uniform.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
	default_sampler_uniform.binding = 8;
	default_sampler_uniform.append_id(owner.resolve_stage.resolve_sampler);
	uniforms.push_back(default_sampler_uniform);

	RD::Uniform shadow_sampler_uniform;
	shadow_sampler_uniform.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
	shadow_sampler_uniform.binding = 10;
	shadow_sampler_uniform.append_id(owner.resolve_stage.shadow_sampler);
	uniforms.push_back(shadow_sampler_uniform);

	RD::Uniform shadow_atlas_uniform;
	shadow_atlas_uniform.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
	shadow_atlas_uniform.binding = 11;
	shadow_atlas_uniform.append_id(shadow_atlas_texture);
	uniforms.push_back(shadow_atlas_uniform);

	RD::Uniform directional_shadow_uniform;
	directional_shadow_uniform.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
	directional_shadow_uniform.binding = 12;
	directional_shadow_uniform.append_id(directional_shadow_texture);
	uniforms.push_back(directional_shadow_uniform);

	RD::Uniform linear_clamp_uniform;
	linear_clamp_uniform.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
	linear_clamp_uniform.binding = 13;
	linear_clamp_uniform.append_id(owner.resolve_stage.resolve_sampler);
	uniforms.push_back(linear_clamp_uniform);

	RID lighting_uniform_set = p_device->uniform_set_create(uniforms, owner.shader_resources.tile_binning_shader, 2);
	if (lighting_uniform_set.is_valid()) {
		p_device->set_resource_name(lighting_uniform_set, "GS_TileRenderer_BinningLightingSet");
	}
	return lighting_uniform_set;
}
