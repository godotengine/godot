/**
 * tile_render_rasterizer_stage.cpp — TileRenderer::TileRasterizerStage method implementations.
 *
 * Companion .cpp for tile_renderer.h / tile_render_stages.h.
 * Contains the tile rasterizer dispatches (compute and fragment paths),
 * uniform set acquisition (buffer, compute buffer, param, compute param,
 * image), and uniform set preparation helpers.
 *
 * Pattern 10 (Flyweight + GPU resource cache): Rasterizer uniform sets are
 * cached flyweight references into shared GPU resources; invalidated when
 * dependencies (gaussian_buffer, sorted_indices, render targets) change.
 * Pattern 12 (Ownership graph): RIDs for render targets (output_texture,
 * depth_texture, normal_texture) are borrowed from the TileRenderer owner.
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

uint64_t TileRenderer::TileRasterizerStage::dispatch_tile_rasterizer_compute(uint32_t p_gaussian_count, RID p_buffer_uniform_set,
        RID p_param_uniform_set, RID p_image_uniform_set, RenderingDevice *p_submission_device) {
    if (!owner.shader_resources.tile_raster_compute_pipeline.is_valid() || owner.grid_state.tiles_x == 0 || owner.grid_state.tiles_y == 0) {
        return 0;
    }
    RenderingDevice *submission_device = p_submission_device;
    if (!submission_device) {
        return 0;
    }
    if (!p_buffer_uniform_set.is_valid() || !p_param_uniform_set.is_valid() || !p_image_uniform_set.is_valid()) {
        return 0;
    }

    uint32_t dispatch_x = owner.grid_state.tiles_x;
    uint32_t dispatch_y = owner.grid_state.tiles_y;

    uint32_t timestamp_base = submission_device->get_captured_timestamps_count();
    String raster_label = "TileRaster_" + String::num_uint64(owner.frame_state.current_frame_serial);
    submission_device->capture_timestamp(raster_label + String("_Begin"));
    ScopedGpuMarker raster_marker(submission_device, "GS_TileRaster", Color(0.2f, 0.5f, 1.0f, 1.0f));

    auto clear_raster_target = [&](const RID &p_texture, const Color &p_clear) {
        if (!p_texture.is_valid() || !submission_device->texture_is_valid(p_texture)) {
            return;
        }
        RD::TextureFormat fmt = submission_device->texture_get_format(p_texture);
        const uint32_t mipmaps = MAX<uint32_t>(1, fmt.mipmaps);
        const uint32_t layers = MAX<uint32_t>(1, fmt.array_layers);
        submission_device->texture_clear(p_texture, p_clear, 0, mipmaps, 0, layers);
    };
    clear_raster_target(owner.render_targets.output_texture, Color(0.0f, 0.0f, 0.0f, 0.0f));
    clear_raster_target(owner.render_targets.depth_texture, Color(1.0f, 0.0f, 0.0f, 0.0f));
    clear_raster_target(owner.render_targets.normal_texture, Color(0.0f, 0.0f, 0.0f, 0.0f));

    RD::ComputeListID compute_list = submission_device->compute_list_begin();
    if (compute_list == RD::INVALID_ID) {
        ERR_PRINT_ONCE("[TileRenderer] Failed to begin compute raster list");
        return 0;
    }

    submission_device->compute_list_bind_compute_pipeline(compute_list, owner.shader_resources.tile_raster_compute_pipeline);
    submission_device->compute_list_bind_uniform_set(compute_list, p_buffer_uniform_set, 0);
    submission_device->compute_list_bind_uniform_set(compute_list, p_param_uniform_set, 1);
    submission_device->compute_list_bind_uniform_set(compute_list, p_image_uniform_set, 2);
    submission_device->compute_list_dispatch(compute_list, dispatch_x, dispatch_y, 1);
    submission_device->compute_list_end();
    submission_device->capture_timestamp(raster_label + String("_End"));

    owner.timing_state.raster_timestamp.device = submission_device;
    owner.timing_state.raster_timestamp.start_index = timestamp_base;
    owner.timing_state.raster_timestamp.end_index = timestamp_base + 1;
    owner.timing_state.raster_timestamp.label = raster_label;

    owner._queue_submission(submission_device, false);

    return 0;
}

uint64_t TileRenderer::TileRasterizerStage::dispatch_tile_rasterizer(uint32_t p_gaussian_count, RID p_buffer_uniform_set,
        RID p_param_uniform_set, RenderingDevice *p_submission_device) {
    if (!owner.shader_resources.tile_raster_shader.is_valid() || !owner.render_targets.tile_framebuffer.is_valid() || owner.grid_state.tiles_x == 0 || owner.grid_state.tiles_y == 0) {
        return 0;
    }
    RenderingDevice *submission_device = p_submission_device;
    if (!submission_device) {
        return 0;
    }
    if (!p_buffer_uniform_set.is_valid() || !p_param_uniform_set.is_valid()) {
        return 0;
    }

    if (owner.shader_resources.tile_raster_pipeline.is_valid() &&
            owner.render_targets.tile_framebuffer_format != submission_device->framebuffer_get_format(owner.render_targets.tile_framebuffer)) {
        if (submission_device->render_pipeline_is_valid(owner.shader_resources.tile_raster_pipeline)) {
            submission_device->free(owner.shader_resources.tile_raster_pipeline);
        }
        owner.shader_resources.tile_raster_pipeline = RID();
    }

    uint32_t timestamp_base = submission_device->get_captured_timestamps_count();
    String raster_label = "TileRaster_" + String::num_uint64(owner.frame_state.current_frame_serial);
    submission_device->capture_timestamp(raster_label + String("_Begin"));
    ScopedGpuMarker raster_marker(submission_device, "GS_TileRaster", Color(0.2f, 0.5f, 1.0f, 1.0f));

    if (owner.render_targets.tile_framebuffer_format == RD::INVALID_ID) {
        owner.render_targets.tile_framebuffer_format = submission_device->framebuffer_get_format(owner.render_targets.tile_framebuffer);
    }

    if (!owner.shader_resources.tile_raster_pipeline.is_valid()) {
        RD::PipelineRasterizationState raster_state;
        raster_state.cull_mode = RD::POLYGON_CULL_DISABLED;
        RD::PipelineMultisampleState ms_state;
        RD::PipelineDepthStencilState depth_state;
        depth_state.enable_depth_test = false;
        depth_state.enable_depth_write = false;

        // Configure blend state for THREE color attachments (out_color, out_depth, out_normal)
        RD::PipelineColorBlendState blend_state;
        blend_state.attachments.resize(3);

        // Attachment 0: out_color with alpha blending
        blend_state.attachments.write[0] = RD::PipelineColorBlendState::Attachment();
        blend_state.attachments.write[0].enable_blend = true;
        // Premultiplied alpha: src color already includes alpha, so use ONE.
        blend_state.attachments.write[0].src_color_blend_factor = RD::BLEND_FACTOR_ONE;
        blend_state.attachments.write[0].dst_color_blend_factor = RD::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blend_state.attachments.write[0].color_blend_op = RD::BLEND_OP_ADD;
        blend_state.attachments.write[0].src_alpha_blend_factor = RD::BLEND_FACTOR_ONE;
        blend_state.attachments.write[0].dst_alpha_blend_factor = RD::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blend_state.attachments.write[0].alpha_blend_op = RD::BLEND_OP_ADD;

        // Attachment 1: out_depth (no blending, just write)
        blend_state.attachments.write[1] = RD::PipelineColorBlendState::Attachment();
        blend_state.attachments.write[1].enable_blend = false;

        // Attachment 2: out_normal (premultiplied alpha accumulation)
        blend_state.attachments.write[2] = RD::PipelineColorBlendState::Attachment();
        blend_state.attachments.write[2].enable_blend = true;
        blend_state.attachments.write[2].src_color_blend_factor = RD::BLEND_FACTOR_ONE;
        blend_state.attachments.write[2].dst_color_blend_factor = RD::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blend_state.attachments.write[2].color_blend_op = RD::BLEND_OP_ADD;
        blend_state.attachments.write[2].src_alpha_blend_factor = RD::BLEND_FACTOR_ONE;
        blend_state.attachments.write[2].dst_alpha_blend_factor = RD::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blend_state.attachments.write[2].alpha_blend_op = RD::BLEND_OP_ADD;

        owner.shader_resources.tile_raster_pipeline = submission_device->render_pipeline_create(owner.shader_resources.tile_raster_shader,
                owner.render_targets.tile_framebuffer_format, RD::INVALID_ID, RD::RENDER_PRIMITIVE_TRIANGLES, raster_state, ms_state,
                depth_state, blend_state, 0);
        ERR_FAIL_COND_V(!owner.shader_resources.tile_raster_pipeline.is_valid(), 0);
    }

    Rect2i viewport_rect(Vector2i(), owner.grid_state.viewport_size);
    Vector<Color> clear_colors;
    clear_colors.push_back(Color(0.0f, 0.0f, 0.0f, 0.0f));
    clear_colors.push_back(Color(1.0f, 0.0f, 0.0f, 0.0f));
    clear_colors.push_back(Color(0.0f, 0.0f, 0.0f, 0.0f));

    RD::DrawListID draw_list = submission_device->draw_list_begin(owner.render_targets.tile_framebuffer, RD::DRAW_CLEAR_ALL,
            clear_colors, 1.0f, 0, viewport_rect);

    submission_device->draw_list_bind_render_pipeline(draw_list, owner.shader_resources.tile_raster_pipeline);
    submission_device->draw_list_bind_uniform_set(draw_list, p_buffer_uniform_set, 0);
    submission_device->draw_list_bind_uniform_set(draw_list, p_param_uniform_set, 1);
    submission_device->draw_list_draw(draw_list, false, 1, 3);
    submission_device->draw_list_end();

#ifdef DEV_ENABLED
    static uint64_t raster_call_count = 0;
    raster_call_count++;
    if (owner.diagnostics.debug_tile_dispatch_logs_enabled && owner.diagnostics.debug_frame_log_frequency > 0 &&
            (raster_call_count % static_cast<uint64_t>(owner.diagnostics.debug_frame_log_frequency) == 0)) {
        GS_LOG_RENDERER_DEBUG(vformat("[TILE_RASTER_DRAW #%d] viewport=%dx%d tiles=%dx%d fb=%s pipeline=%s splat_count=%d",
                raster_call_count, owner.grid_state.viewport_size.x, owner.grid_state.viewport_size.y, owner.grid_state.tiles_x, owner.grid_state.tiles_y,
                owner.render_targets.tile_framebuffer.is_valid() ? "Y" : "N",
                owner.shader_resources.tile_raster_pipeline.is_valid() ? "Y" : "N",
                p_gaussian_count));
    }
#endif

    submission_device->capture_timestamp(raster_label + String("_End"));

    // NOTE: Draw lists are recorded into the device's command buffer and submitted
    // as part of the frame. We do NOT call _queue_submission() here because:
    // 1. draw_list_end() already completed the draw list recording
    // 2. The work will be submitted when the frame ends
    // 3. Calling submit()/sync() here would fail with "sync can only be called after submit"
    //    because there's no separate compute/transfer work to submit
    // The timestamp queries will be resolved when we read them back later.

    owner.timing_state.raster_timestamp.device = submission_device;
    owner.timing_state.raster_timestamp.start_index = timestamp_base;
    owner.timing_state.raster_timestamp.end_index = timestamp_base + 1;
    owner.timing_state.raster_timestamp.label = raster_label;

    return 0;
}

bool TileRenderer::TileRasterizerStage::prepare_compute_uniform_sets(RenderingDevice *p_device, const RID &p_state_uniform,
		const RID &p_gaussian_buffer, const RID &p_sorted_indices, RasterUniformSets &r_sets) {
	r_sets.param_uniform_set = acquire_raster_compute_param_uniform_set(p_device, p_state_uniform);
	r_sets.buffer_uniform_set = acquire_raster_compute_buffer_uniform_set(p_device, p_gaussian_buffer, p_sorted_indices);
	r_sets.image_uniform_set = acquire_raster_image_uniform_set(p_device);
	return r_sets.param_uniform_set.is_valid() && r_sets.buffer_uniform_set.is_valid() && r_sets.image_uniform_set.is_valid();
}

bool TileRenderer::TileRasterizerStage::prepare_fragment_uniform_sets(RenderingDevice *p_device, const RID &p_state_uniform,
		const RID &p_gaussian_buffer, const RID &p_sorted_indices, RasterUniformSets &r_sets) {
	r_sets.param_uniform_set = acquire_raster_param_uniform_set(p_device, p_state_uniform);
	r_sets.buffer_uniform_set = acquire_raster_buffer_uniform_set(p_device, p_gaussian_buffer, p_sorted_indices);
	r_sets.image_uniform_set = RID();
	return r_sets.param_uniform_set.is_valid() && r_sets.buffer_uniform_set.is_valid();
}

RID TileRenderer::TileRasterizerStage::acquire_raster_buffer_uniform_set(RenderingDevice *p_device,
		const RID &p_gaussian_buffer, const RID &p_sorted_indices) {
	ERR_FAIL_NULL_V(p_device, RID());
	ERR_FAIL_COND_V(!owner.debug_stats.overflow_statistics_buffer.is_valid(), RID());
	ERR_FAIL_COND_V(!p_gaussian_buffer.is_valid(), RID());
	ERR_FAIL_COND_V(!p_sorted_indices.is_valid(), RID());
	ERR_FAIL_COND_V(!owner.projection_buffers.projection_buffer.is_valid(), RID());
	ERR_FAIL_COND_V(!owner.debug_stats.debug_splat_audit_buffer.is_valid(), RID());
	if (owner.render_settings.global_sort_enabled) {
		ERR_FAIL_COND_V(!owner.global_sort_resources.tile_ranges_buffer.is_valid(), RID());
		ERR_FAIL_COND_V(!owner.global_sort_resources.values_buffer.is_valid(), RID());
	}

	ERR_FAIL_COND_V(!owner.instance_pipeline_buffers.splat_ref_buffer.is_valid(), RID());
	ERR_FAIL_COND_V(!owner.instance_pipeline_buffers.indirect_count_buffer.is_valid(), RID());

    if (cached_raster_buffer_uniform_set.is_valid() &&
            cached_generation == owner.descriptor_generation &&
            cached_raster_buffer_device == p_device &&
            cached_raster_gaussian_buffer == p_gaussian_buffer &&
            cached_raster_sorted_indices == p_sorted_indices &&
            cached_raster_splat_ref_buffer == owner.instance_pipeline_buffers.splat_ref_buffer &&
            cached_raster_indirect_count_buffer == owner.instance_pipeline_buffers.indirect_count_buffer) {
        return cached_raster_buffer_uniform_set;
    }

    if (cached_raster_buffer_uniform_set.is_valid()) {
        RenderingDevice *owner_device = cached_raster_buffer_device ? cached_raster_buffer_device : p_device;
        _release_uniform_set(owner_device, cached_raster_buffer_uniform_set);
        cached_raster_buffer_uniform_set = RID();
        cached_raster_buffer_device = nullptr;
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

	RD::Uniform indirect_uniform_instance;
	indirect_uniform_instance.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	indirect_uniform_instance.binding = 15;
	indirect_uniform_instance.append_id(owner.instance_pipeline_buffers.indirect_count_buffer);
	uniforms.push_back(indirect_uniform_instance);

	RD::Uniform ranges_uniform;
	ranges_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	ranges_uniform.binding = 2;
	ranges_uniform.append_id(owner.global_sort_resources.tile_ranges_buffer);
	uniforms.push_back(ranges_uniform);

    RD::Uniform stats_uniform;
    stats_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
    stats_uniform.binding = 3;
    stats_uniform.append_id(owner.debug_stats.overflow_statistics_buffer);
    uniforms.push_back(stats_uniform);

	RD::Uniform projection_uniform;
	projection_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	projection_uniform.binding = 4;
	projection_uniform.append_id(owner.projection_buffers.projection_buffer);
	uniforms.push_back(projection_uniform);

	RD::Uniform values_uniform;
	values_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	values_uniform.binding = 5;
	values_uniform.append_id(owner.global_sort_resources.values_buffer);
	uniforms.push_back(values_uniform);

	// Binding 6: IndirectDispatch buffer for GPU-side element_count
	ERR_FAIL_COND_V(!owner.global_sort_resources.indirect_dispatch_buffer.is_valid(), RID());
	RD::Uniform indirect_uniform;
	indirect_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	indirect_uniform.binding = 6;
	indirect_uniform.append_id(owner.global_sort_resources.indirect_dispatch_buffer);
	uniforms.push_back(indirect_uniform);

	RD::Uniform audit_uniform;
	audit_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	audit_uniform.binding = 10;
	audit_uniform.append_id(owner.debug_stats.debug_splat_audit_buffer);
	uniforms.push_back(audit_uniform);

	// ISSUE-002: Verify all buffers belong to p_device before creating uniform set.
	ERR_FAIL_COND_V(!_verify_buffer_device_ownership(p_device, p_gaussian_buffer, "raster:gaussian_buffer"), RID());
	ERR_FAIL_COND_V(!_verify_buffer_device_ownership(p_device, p_sorted_indices, "raster:sorted_indices"), RID());
	ERR_FAIL_COND_V(!_verify_buffer_device_ownership(p_device, owner.projection_buffers.projection_buffer, "raster:projection_buffer"), RID());
	ERR_FAIL_COND_V(!_verify_buffer_device_ownership(p_device, owner.global_sort_resources.tile_ranges_buffer, "raster:tile_ranges_buffer"), RID());
	ERR_FAIL_COND_V(!_verify_buffer_device_ownership(p_device, owner.global_sort_resources.values_buffer, "raster:values_buffer"), RID());
	ERR_FAIL_COND_V(!_verify_buffer_device_ownership(p_device, owner.global_sort_resources.indirect_dispatch_buffer, "raster:indirect_dispatch_buffer"), RID());

	cached_raster_buffer_uniform_set = p_device->uniform_set_create(uniforms, owner.shader_resources.tile_raster_shader, 0);
	if (!cached_raster_buffer_uniform_set.is_valid()) {
		ERR_FAIL_V_MSG(RID(), "[TileRenderer] Failed to create raster buffer uniform set (projection buffer path)");
	}
	p_device->set_resource_name(cached_raster_buffer_uniform_set, "GS_TileRenderer_RasterBufferSet");
    cached_raster_gaussian_buffer = p_gaussian_buffer;
    cached_raster_sorted_indices = p_sorted_indices;
    cached_raster_splat_ref_buffer = owner.instance_pipeline_buffers.splat_ref_buffer;
    cached_raster_indirect_count_buffer = owner.instance_pipeline_buffers.indirect_count_buffer;
    cached_raster_buffer_device = p_device;
    cached_generation = owner.descriptor_generation;

    return cached_raster_buffer_uniform_set;
}

RID TileRenderer::TileRasterizerStage::acquire_raster_compute_buffer_uniform_set(RenderingDevice *p_device,
		const RID &p_gaussian_buffer, const RID &p_sorted_indices) {
    ERR_FAIL_NULL_V(p_device, RID());
    ERR_FAIL_COND_V(!owner.shader_resources.tile_raster_compute_shader.is_valid(), RID());
    ERR_FAIL_COND_V(!owner.debug_stats.overflow_statistics_buffer.is_valid(), RID());
    ERR_FAIL_COND_V(!p_gaussian_buffer.is_valid(), RID());
    ERR_FAIL_COND_V(!p_sorted_indices.is_valid(), RID());
    ERR_FAIL_COND_V(!owner.projection_buffers.projection_buffer.is_valid(), RID());
    ERR_FAIL_COND_V(!owner.debug_stats.debug_splat_audit_buffer.is_valid(), RID());
    if (owner.render_settings.global_sort_enabled) {
        ERR_FAIL_COND_V(!owner.global_sort_resources.tile_ranges_buffer.is_valid(), RID());
        ERR_FAIL_COND_V(!owner.global_sort_resources.values_buffer.is_valid(), RID());
    }

    ERR_FAIL_COND_V(!owner.instance_pipeline_buffers.splat_ref_buffer.is_valid(), RID());
    ERR_FAIL_COND_V(!owner.instance_pipeline_buffers.indirect_count_buffer.is_valid(), RID());

    if (cached_raster_compute_buffer_uniform_set.is_valid() &&
            cached_generation == owner.descriptor_generation &&
            cached_raster_compute_buffer_device == p_device &&
            cached_raster_compute_gaussian_buffer == p_gaussian_buffer &&
            cached_raster_compute_sorted_indices == p_sorted_indices &&
            cached_raster_compute_splat_ref_buffer == owner.instance_pipeline_buffers.splat_ref_buffer &&
            cached_raster_compute_indirect_count_buffer == owner.instance_pipeline_buffers.indirect_count_buffer) {
        return cached_raster_compute_buffer_uniform_set;
    }

    if (cached_raster_compute_buffer_uniform_set.is_valid()) {
        RenderingDevice *owner_device = cached_raster_compute_buffer_device ? cached_raster_compute_buffer_device : p_device;
        _release_uniform_set(owner_device, cached_raster_compute_buffer_uniform_set);
        cached_raster_compute_buffer_uniform_set = RID();
        cached_raster_compute_buffer_device = nullptr;
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

    RD::Uniform indirect_uniform_instance;
    indirect_uniform_instance.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
    indirect_uniform_instance.binding = 15;
    indirect_uniform_instance.append_id(owner.instance_pipeline_buffers.indirect_count_buffer);
    uniforms.push_back(indirect_uniform_instance);

    RD::Uniform ranges_uniform;
    ranges_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
    ranges_uniform.binding = 2;
    ranges_uniform.append_id(owner.global_sort_resources.tile_ranges_buffer);
    uniforms.push_back(ranges_uniform);

    RD::Uniform stats_uniform;
    stats_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
    stats_uniform.binding = 3;
    stats_uniform.append_id(owner.debug_stats.overflow_statistics_buffer);
    uniforms.push_back(stats_uniform);

    RD::Uniform projection_uniform;
    projection_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
    projection_uniform.binding = 4;
    projection_uniform.append_id(owner.projection_buffers.projection_buffer);
    uniforms.push_back(projection_uniform);

    RD::Uniform values_uniform;
    values_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
    values_uniform.binding = 5;
    values_uniform.append_id(owner.global_sort_resources.values_buffer);
    uniforms.push_back(values_uniform);

    // Binding 6: IndirectDispatch buffer for GPU-side element_count
    ERR_FAIL_COND_V(!owner.global_sort_resources.indirect_dispatch_buffer.is_valid(), RID());
    RD::Uniform indirect_uniform;
    indirect_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
    indirect_uniform.binding = 6;
    indirect_uniform.append_id(owner.global_sort_resources.indirect_dispatch_buffer);
    uniforms.push_back(indirect_uniform);

    RD::Uniform audit_uniform;
    audit_uniform.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
    audit_uniform.binding = 10;
    audit_uniform.append_id(owner.debug_stats.debug_splat_audit_buffer);
    uniforms.push_back(audit_uniform);

    // ISSUE-002: Verify key buffers belong to p_device before creating uniform set.
    ERR_FAIL_COND_V(!_verify_buffer_device_ownership(p_device, p_gaussian_buffer, "compute_raster:gaussian_buffer"), RID());
    ERR_FAIL_COND_V(!_verify_buffer_device_ownership(p_device, p_sorted_indices, "compute_raster:sorted_indices"), RID());
    ERR_FAIL_COND_V(!_verify_buffer_device_ownership(p_device, owner.projection_buffers.projection_buffer, "compute_raster:projection_buffer"), RID());
    ERR_FAIL_COND_V(!_verify_buffer_device_ownership(p_device, owner.global_sort_resources.tile_ranges_buffer, "compute_raster:tile_ranges_buffer"), RID());
    ERR_FAIL_COND_V(!_verify_buffer_device_ownership(p_device, owner.global_sort_resources.values_buffer, "compute_raster:values_buffer"), RID());
    ERR_FAIL_COND_V(!_verify_buffer_device_ownership(p_device, owner.global_sort_resources.indirect_dispatch_buffer, "compute_raster:indirect_dispatch_buffer"), RID());

    cached_raster_compute_buffer_uniform_set = p_device->uniform_set_create(uniforms, owner.shader_resources.tile_raster_compute_shader, 0);
    if (!cached_raster_compute_buffer_uniform_set.is_valid()) {
        ERR_FAIL_V_MSG(RID(), "[TileRenderer] Failed to create compute raster buffer uniform set");
    }
    p_device->set_resource_name(cached_raster_compute_buffer_uniform_set, "GS_TileRenderer_RasterComputeBufferSet");
    cached_raster_compute_gaussian_buffer = p_gaussian_buffer;
    cached_raster_compute_sorted_indices = p_sorted_indices;
    cached_raster_compute_splat_ref_buffer = owner.instance_pipeline_buffers.splat_ref_buffer;
    cached_raster_compute_indirect_count_buffer = owner.instance_pipeline_buffers.indirect_count_buffer;
    cached_raster_compute_buffer_device = p_device;
    cached_generation = owner.descriptor_generation;

    return cached_raster_compute_buffer_uniform_set;
}

RID TileRenderer::TileRasterizerStage::acquire_raster_param_uniform_set(RenderingDevice *p_device, const RID &p_state_uniform) {
    ERR_FAIL_NULL_V(p_device, RID());
    ERR_FAIL_COND_V(!owner.uniform_buffers.param_uniform_buffer.is_valid(), RID());

    if (cached_raster_param_uniform_set.is_valid() && cached_state_uniform == p_state_uniform) {
        if (cached_raster_param_device != p_device) {
            // Device mismatch requires rebuild.
        } else {
            return cached_raster_param_uniform_set;
        }
    }

    if (cached_raster_param_uniform_set.is_valid()) {
        RenderingDevice *owner_device = cached_raster_param_device ? cached_raster_param_device : p_device;
        _release_uniform_set(owner_device, cached_raster_param_uniform_set);
        cached_raster_param_uniform_set = RID();
        cached_raster_param_device = nullptr;
    }

    Vector<RD::Uniform> uniforms;
    RD::Uniform params_uniform;
    params_uniform.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
    params_uniform.binding = 0;
    params_uniform.append_id(owner.uniform_buffers.param_uniform_buffer);
    uniforms.push_back(params_uniform);

    if (p_state_uniform.is_valid()) {
        RD::Uniform state_uniform;
        state_uniform.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
        state_uniform.binding = 1;
        state_uniform.append_id(p_state_uniform);
        uniforms.push_back(state_uniform);
    }

    cached_raster_param_uniform_set = p_device->uniform_set_create(uniforms, owner.shader_resources.tile_raster_shader, 1);
    if (!cached_raster_param_uniform_set.is_valid()) {
        ERR_FAIL_V_MSG(RID(), "[TileRenderer] Failed to create raster param uniform set");
    }
    p_device->set_resource_name(cached_raster_param_uniform_set, "GS_TileRenderer_RasterParamSet");
    cached_state_uniform = p_state_uniform;
    cached_raster_param_device = p_device;

    return cached_raster_param_uniform_set;
}

RID TileRenderer::TileRasterizerStage::acquire_raster_compute_param_uniform_set(RenderingDevice *p_device, const RID &p_state_uniform) {
    ERR_FAIL_NULL_V(p_device, RID());
    ERR_FAIL_COND_V(!owner.shader_resources.tile_raster_compute_shader.is_valid(), RID());
    ERR_FAIL_COND_V(!owner.uniform_buffers.param_uniform_buffer.is_valid(), RID());

    if (cached_raster_compute_param_uniform_set.is_valid() && cached_raster_compute_state_uniform == p_state_uniform) {
        if (cached_raster_compute_param_device != p_device) {
            // Device mismatch requires rebuild.
        } else {
            return cached_raster_compute_param_uniform_set;
        }
    }

    if (cached_raster_compute_param_uniform_set.is_valid()) {
        RenderingDevice *owner_device = cached_raster_compute_param_device ? cached_raster_compute_param_device : p_device;
        _release_uniform_set(owner_device, cached_raster_compute_param_uniform_set);
        cached_raster_compute_param_uniform_set = RID();
        cached_raster_compute_param_device = nullptr;
    }

    Vector<RD::Uniform> uniforms;
    RD::Uniform params_uniform;
    params_uniform.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
    params_uniform.binding = 0;
    params_uniform.append_id(owner.uniform_buffers.param_uniform_buffer);
    uniforms.push_back(params_uniform);

    if (p_state_uniform.is_valid()) {
        RD::Uniform state_uniform;
        state_uniform.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
        state_uniform.binding = 1;
        state_uniform.append_id(p_state_uniform);
        uniforms.push_back(state_uniform);
    }

    cached_raster_compute_param_uniform_set = p_device->uniform_set_create(uniforms, owner.shader_resources.tile_raster_compute_shader, 1);
    if (!cached_raster_compute_param_uniform_set.is_valid()) {
        ERR_FAIL_V_MSG(RID(), "[TileRenderer] Failed to create compute raster param uniform set");
    }
    p_device->set_resource_name(cached_raster_compute_param_uniform_set, "GS_TileRenderer_RasterComputeParamSet");
    cached_raster_compute_state_uniform = p_state_uniform;
    cached_raster_compute_param_device = p_device;

    return cached_raster_compute_param_uniform_set;
}

RID TileRenderer::TileRasterizerStage::acquire_raster_image_uniform_set(RenderingDevice *p_device) {
    ERR_FAIL_NULL_V(p_device, RID());
    ERR_FAIL_COND_V(!owner.shader_resources.tile_raster_compute_shader.is_valid(), RID());
    ERR_FAIL_COND_V(!owner.render_targets.output_texture.is_valid(), RID());
    ERR_FAIL_COND_V(!owner.render_targets.depth_texture.is_valid(), RID());
    ERR_FAIL_COND_V(!owner.render_targets.normal_texture.is_valid(), RID());

    // ISSUE-002: Verify ALL render target textures belong to p_device before
    // binding into a uniform set. Cross-device binding causes GPU hangs.
    ERR_FAIL_COND_V(!_verify_texture_device_ownership(p_device, owner.render_targets.output_texture, "raster_image:output_texture"), RID());
    ERR_FAIL_COND_V(!_verify_texture_device_ownership(p_device, owner.render_targets.depth_texture, "raster_image:depth_texture"), RID());
    ERR_FAIL_COND_V(!_verify_texture_device_ownership(p_device, owner.render_targets.normal_texture, "raster_image:normal_texture"), RID());

    if (cached_raster_image_uniform_set.is_valid() &&
            cached_generation == owner.descriptor_generation &&
            cached_raster_image_device == p_device &&
            cached_raster_image_output == owner.render_targets.output_texture &&
            cached_raster_image_depth == owner.render_targets.depth_texture &&
            cached_raster_image_normal == owner.render_targets.normal_texture) {
        return cached_raster_image_uniform_set;
    }

    if (cached_raster_image_uniform_set.is_valid()) {
        RenderingDevice *owner_device = cached_raster_image_device ? cached_raster_image_device : p_device;
        _release_uniform_set(owner_device, cached_raster_image_uniform_set);
        cached_raster_image_uniform_set = RID();
        cached_raster_image_device = nullptr;
    }

    Vector<RD::Uniform> uniforms;
    RD::Uniform color_uniform;
    color_uniform.uniform_type = RD::UNIFORM_TYPE_IMAGE;
    color_uniform.binding = 0;
    color_uniform.append_id(owner.render_targets.output_texture);
    uniforms.push_back(color_uniform);

    RD::Uniform depth_uniform;
    depth_uniform.uniform_type = RD::UNIFORM_TYPE_IMAGE;
    depth_uniform.binding = 1;
    depth_uniform.append_id(owner.render_targets.depth_texture);
    uniforms.push_back(depth_uniform);

    RD::Uniform normal_uniform;
    normal_uniform.uniform_type = RD::UNIFORM_TYPE_IMAGE;
    normal_uniform.binding = 2;
    normal_uniform.append_id(owner.render_targets.normal_texture);
    uniforms.push_back(normal_uniform);

    cached_raster_image_uniform_set = p_device->uniform_set_create(uniforms, owner.shader_resources.tile_raster_compute_shader, 2);
    if (!cached_raster_image_uniform_set.is_valid()) {
        ERR_FAIL_V_MSG(RID(), "[TileRenderer] Failed to create compute raster image uniform set");
    }
    p_device->set_resource_name(cached_raster_image_uniform_set, "GS_TileRenderer_RasterImageSet");
    cached_raster_image_device = p_device;
    cached_raster_image_output = owner.render_targets.output_texture;
    cached_raster_image_depth = owner.render_targets.depth_texture;
    cached_raster_image_normal = owner.render_targets.normal_texture;
    cached_generation = owner.descriptor_generation;

    return cached_raster_image_uniform_set;
}
