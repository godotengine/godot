#include "painterly_renderer.h"
#include "../core/gs_project_settings.h"
#include "rasterizer_interfaces.h"
#include "interactive_state_manager.h"
#include "overflow_auto_tuner.h"
#include "painterly_material_manager.h"
#include "gpu_sorting_pipeline.h"
#include "../renderer/gaussian_splat_renderer.h"
#include "servers/rendering/renderer_rd/storage_rd/render_data_rd.h"
#include "../renderer/painterly_pass_graph.h"
#include "../painterly/painterly_material.h"
#include "../logger/gs_logger.h"
#include "core/error/error_macros.h"
#include "core/config/project_settings.h"
#include "core/math/math_funcs.h"
#include "core/os/os.h"
#include "servers/rendering/renderer_rd/framebuffer_cache_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/render_scene_buffers_rd.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"

// Painterly compute shader sources
#include "../shaders/sobel_outline.glsl.gen.h"
#include "../shaders/brush_accumulate.glsl.gen.h"
#include "../shaders/painterly_composite.glsl.gen.h"

namespace {

// Project settings helpers provided by gs_project_settings.h (gs::settings namespace).
static bool _get_bool_setting(ProjectSettings *ps, const StringName &name, bool fallback) {
	return gs::settings::get_bool(ps, name, fallback);
}

static RenderingDevice *_ensure_local_device(RenderingDevice *p_candidate) {
	return p_candidate;
}

static RenderingDevice *_acquire_manager_local_device() {
	if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
		if (RenderingDevice *primary = _ensure_local_device(manager->get_primary_rendering_device())) {
			return primary;
		}
		if (RenderingDevice *shared = _ensure_local_device(manager->get_shared_submission_device())) {
			return shared;
		}
	}

	return nullptr;
}

} // namespace

void PainterlyRenderer::_bind_methods() {
    // Bind methods for script access if needed
}

PainterlyRenderer::PainterlyRenderer() {
    pass_graph = memnew(PainterlyPassGraph);
    for (int i = 0; i < kTextureSlotCount; i++) {
        tracked_local_textures[i] = RID();
        tracked_shared_textures[i] = RID();
    }
}

PainterlyRenderer::~PainterlyRenderer() {
    shutdown();
    if (pass_graph) {
        memdelete(pass_graph);
        pass_graph = nullptr;
    }
}

Error PainterlyRenderer::initialize(RenderingDevice *p_device, const Vector2i &p_initial_size) {
    if (!p_device) {
        return ERR_INVALID_PARAMETER;
    }

    rd = p_device;
    pass_graph->setup(p_device);

    if (p_initial_size.x > 0 && p_initial_size.y > 0) {
        pass_graph->configure(p_initial_size, current_config.internal_scale,
                current_config.enable_stylization, current_config.low_end_mode);
    }

    return OK;
}

RenderingDevice *PainterlyRenderer::_resolve_tracked_device(const RidOwner &p_owner, GaussianSplatRenderer *p_renderer) const {
    if (p_owner.device == nullptr && p_owner.device_id == 0) {
        return nullptr;
    }

    RenderingDevice *candidates[6];
    int candidate_count = 0;

    auto add_candidate = [&](RenderingDevice *p_device) {
        if (!p_device) {
            return;
        }
        for (int i = 0; i < candidate_count; i++) {
            if (candidates[i] == p_device) {
                return;
            }
        }
        candidates[candidate_count++] = p_device;
    };

    add_candidate(rd);
    if (p_renderer) {
        GaussianSplatRenderer::FrameStateProvider frame_provider(p_renderer);
        const GaussianSplatRenderer::IFrameStateView &state_view = frame_provider;
        add_candidate(p_renderer->get_main_rendering_device());
        add_candidate(state_view.get_rendering_device());
    }
    if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
        add_candidate(_ensure_local_device(manager->get_primary_rendering_device()));
        add_candidate(_ensure_local_device(manager->get_shared_submission_device()));
    }
    add_candidate(RenderingDevice::get_singleton());

    for (int i = 0; i < candidate_count; i++) {
        RenderingDevice *candidate = candidates[i];
        if (p_owner.device && candidate != p_owner.device) {
            continue;
        }
        if (p_owner.device_id != 0 && p_owner.device_id != candidate->get_device_instance_id()) {
            continue;
        }
        return candidate;
    }

    return nullptr;
}

void PainterlyRenderer::_free_tracked_rid(RID &p_rid, RidOwner &p_owner, GaussianSplatRenderer *p_renderer, bool p_forget_renderer_owner) {
    if (!p_rid.is_valid()) {
        p_owner.clear();
        return;
    }

    RenderingDevice *owner_device = _resolve_tracked_device(p_owner, p_renderer);
    if (!owner_device) {
        // Keep RID + owner so a later pass with a valid device context can release it safely.
        return;
    }
    owner_device->free(p_rid);

    if (p_forget_renderer_owner && p_renderer) {
        p_renderer->forget_resource_owner(p_rid);
    }

    p_rid = RID();
    p_owner.clear();
}

void PainterlyRenderer::shutdown() {
    _shutdown_internal(nullptr);
}

void PainterlyRenderer::_shutdown_internal(GaussianSplatRenderer *p_renderer) {
    _forget_painterly_texture_tracking(p_renderer);

    // Check if our device is still valid by comparing with current shared device
    GaussianSplatManager *manager = GaussianSplatManager::get_singleton();
    RenderingDevice *current_shared = manager ? manager->get_shared_submission_device() : nullptr;
    bool device_still_valid = (rd != nullptr) && (rd == current_shared);

    // Free GPU resources if device is still valid
    if (device_still_valid) {
        _free_tracked_rid(painterly_sampler, painterly_sampler_owner, p_renderer, false);
        _free_tracked_rid(composite_sampler, composite_sampler_owner, p_renderer, false);
        _free_tracked_rid(composite_depth_sampler, composite_depth_sampler_owner, p_renderer, false);
        _free_tracked_rid(painterly_depth_sampler, painterly_depth_sampler_owner, p_renderer, true);
        _free_tracked_rid(painterly_color_sampler, painterly_color_sampler_owner, p_renderer, true);
        _free_tracked_rid(painterly_composite_shader, painterly_composite_shader_owner, p_renderer, true);

        // Free pipelines first (they depend on shaders)
        // Use device validity checks to avoid double-free from PR 103113 auto-free
        if (sobel_pipeline.is_valid()) {
            if (rd->compute_pipeline_is_valid(sobel_pipeline)) {
                rd->free(sobel_pipeline);
            }
            sobel_pipeline = RID();
        }
        if (brush_pipeline.is_valid()) {
            if (rd->compute_pipeline_is_valid(brush_pipeline)) {
                rd->free(brush_pipeline);
            }
            brush_pipeline = RID();
        }
        if (composite_pipeline.is_valid()) {
            if (rd->compute_pipeline_is_valid(composite_pipeline)) {
                rd->free(composite_pipeline);
            }
            composite_pipeline = RID();
        }

        // Free uniform sets
        if (sobel_uniform_set.is_valid() && rd->uniform_set_is_valid(sobel_uniform_set)) {
            rd->free(sobel_uniform_set);
        }
        if (brush_uniform_set.is_valid() && rd->uniform_set_is_valid(brush_uniform_set)) {
            rd->free(brush_uniform_set);
        }

        // Do not free ShaderRD-backed shader RIDs directly here.
        // They are owned by shader versions and released via version_free() below.
        // Explicit frees in this path can trigger deterministic invalid RID errors
        // during shutdown teardown ordering.

        // Note: PipelineCacheRD manages its own resources - clear() is called below
    }

    // Always invalidate RIDs
    sobel_pipeline = RID();
    brush_pipeline = RID();
    sobel_uniform_set = RID();
    brush_uniform_set = RID();
    sobel_uniform_sampler = RID();
    sobel_uniform_color_input = RID();
    sobel_uniform_edge_texture = RID();
    brush_uniform_sampler = RID();
    brush_uniform_color_input = RID();
    brush_uniform_edge_input = RID();
    brush_uniform_stylized_texture = RID();
    painterly_sampler = RID();
    composite_pipeline = RID();
    composite_shader = RID();
    composite_sampler = RID();
    composite_depth_sampler = RID();
    painterly_sampler_owner.clear();
    composite_sampler_owner.clear();
    composite_depth_sampler_owner.clear();

    // Free shader versions and sources
    if (composite_shader_source) {
        if (composite_shader_version.is_valid()) {
            composite_shader_source->version_free(composite_shader_version);
            composite_shader_version = RID();
        }
        memdelete(composite_shader_source);
        composite_shader_source = nullptr;
    }
    composite_failed = false;

    if (sobel_shader_source) {
        if (sobel_shader_version.is_valid()) {
            sobel_shader_source->version_free(sobel_shader_version);
            sobel_shader_version = RID();
        }
        memdelete(sobel_shader_source);
        sobel_shader_source = nullptr;
    }
    sobel_shader = RID();
    sobel_shader_initialized = false;

    if (brush_shader_source) {
        if (brush_shader_version.is_valid()) {
            brush_shader_source->version_free(brush_shader_version);
            brush_shader_version = RID();
        }
        memdelete(brush_shader_source);
        brush_shader_source = nullptr;
    }
    brush_shader = RID();
    brush_shader_initialized = false;
    composite_shader_initialized = false;

    // Shutdown internal rasterizer if we own it
    if (owns_rasterizer && internal_rasterizer.is_valid()) {
        internal_rasterizer->shutdown();
    }
    internal_rasterizer.unref();
    owns_rasterizer = false;

    // Clear material texture caches
    cached_palette_textures.clear();
    cached_noise_luts.clear();
    cached_stroke_density_buffer = RID();

    if (pass_graph) {
        pass_graph->reset();
    }

    rd = nullptr;
    shaders_compiled = false;
    composite_initialized = false;
    painterly_composite_pipeline.clear();
    painterly_composite_pipeline_initialized = false;
    painterly_composite_failed = false;
    painterly_composite_shader = RID();
    painterly_composite_shader_owner.clear();
    painterly_depth_sampler = RID();
    painterly_depth_sampler_owner.clear();
    painterly_color_sampler = RID();
    painterly_color_sampler_owner.clear();
}

bool PainterlyRenderer::is_ready() const {
    return pass_graph && pass_graph->is_ready();
}

void PainterlyRenderer::configure(const PainterlyConfig &p_config) {
    current_config = p_config;

    if (pass_graph) {
        pass_graph->set_color_format(p_config.color_format);
        pass_graph->configure(p_config.viewport_size, p_config.internal_scale,
                p_config.enable_stylization, p_config.low_end_mode);
    }
}

PainterlyConfig PainterlyRenderer::get_config() const {
    return current_config;
}

void PainterlyRenderer::set_material(const Ref<PainterlyMaterial> &p_material) {
    if (material == p_material) {
        return;
    }
    material = p_material;
    material_dirty = true;
}

Ref<PainterlyMaterial> PainterlyRenderer::get_material() const {
    return material;
}

void PainterlyRenderer::set_color_format(RD::DataFormat p_format) {
    current_config.color_format = p_format;
    if (pass_graph) {
        pass_graph->set_color_format(p_format);
    }
}

RD::DataFormat PainterlyRenderer::get_color_format() const {
    if (pass_graph) {
        return pass_graph->get_color_format();
    }
    return current_config.color_format;
}

Vector2i PainterlyRenderer::get_internal_size() const {
    if (pass_graph) {
        return pass_graph->get_internal_size();
    }
    return Vector2i();
}

Vector2i PainterlyRenderer::get_requested_size() const {
    if (pass_graph) {
        return pass_graph->get_requested_size();
    }
    return current_config.viewport_size;
}

float PainterlyRenderer::get_internal_scale() const {
    if (pass_graph) {
        return pass_graph->get_internal_scale();
    }
    return current_config.internal_scale;
}

bool PainterlyRenderer::is_stylization_enabled() const {
    if (pass_graph) {
        return pass_graph->is_stylization_enabled();
    }
    return current_config.enable_stylization;
}

bool PainterlyRenderer::is_low_end_mode() const {
    if (pass_graph) {
        return pass_graph->is_low_end_mode();
    }
    return current_config.low_end_mode;
}

RID PainterlyRenderer::get_texture(PainterlyTextureSlot p_slot) const {
    if (pass_graph) {
        return pass_graph->get_texture(static_cast<PainterlyPassGraph::TextureSlot>(static_cast<int>(p_slot)));
    }
    return RID();
}

RID PainterlyRenderer::get_shared_texture(PainterlyTextureSlot p_slot) const {
    if (pass_graph) {
        return pass_graph->get_shared_texture(static_cast<PainterlyPassGraph::TextureSlot>(static_cast<int>(p_slot)));
    }
    return RID();
}

RenderingDevice *PainterlyRenderer::get_shared_texture_owner(PainterlyTextureSlot p_slot) const {
    if (pass_graph) {
        return pass_graph->get_shared_texture_owner(static_cast<PainterlyPassGraph::TextureSlot>(static_cast<int>(p_slot)));
    }
    return nullptr;
}

PainterlyTextureInfo PainterlyRenderer::get_texture_info(PainterlyTextureSlot p_slot) const {
    PainterlyTextureInfo info;
    if (!pass_graph) {
        return info;
    }

    const PainterlyPassGraph::TextureInfo &graph_info =
            pass_graph->get_texture_info(static_cast<PainterlyPassGraph::TextureSlot>(static_cast<int>(p_slot)));

    info.size = graph_info.size;
    info.format = graph_info.format.format;
    info.texture = graph_info.texture;
    info.shared_texture = graph_info.shared_texture;
    info.shared_owner = pass_graph->get_shared_texture_owner(static_cast<PainterlyPassGraph::TextureSlot>(static_cast<int>(p_slot)));
    info.valid = graph_info.valid;

    return info;
}

PainterlyRenderResult PainterlyRenderer::execute_passes(RID p_color_input, RID p_depth_input) {
    PainterlyRenderResult result;

    if (!is_ready() || !p_color_input.is_valid()) {
        return result;
    }

    // CPU-side timing for painterly passes.
    // NOTE: This measures command recording time, not actual GPU execution time.
    // Godot's RenderingDevice does not expose per-dispatch GPU timestamps to
    // user code.  For accurate GPU profiling use an external tool (RenderDoc,
    // Nsight, etc.) or the frame-level timestamps captured by TileRenderer.
    const uint64_t painterly_start_usec = OS::get_singleton()->get_ticks_usec();

    // Execute edge detection pass (uniform set created lazily inside pass)
    if (is_stylization_enabled() && sobel_pipeline.is_valid()) {
        _execute_sobel_pass(p_color_input);
    }

    // Execute brush accumulation pass (uniform set created lazily inside pass)
    RID edge_texture = get_texture(PainterlyTextureSlot::EDGE);
    if (current_config.enable_strokes && brush_pipeline.is_valid() && edge_texture.is_valid()) {
        _execute_brush_pass(p_color_input, edge_texture);
    }

    const uint64_t painterly_end_usec = OS::get_singleton()->get_ticks_usec();
    cached_performance.last_pass_cpu_time_ms = (painterly_end_usec - painterly_start_usec) / 1000.0f;

    // Determine final output
    if (is_stylization_enabled()) {
        RID stylized = get_shared_texture(PainterlyTextureSlot::STYLIZED);
        if (stylized.is_valid()) {
            result.final_texture = stylized;
            result.final_texture_owner = get_shared_texture_owner(PainterlyTextureSlot::STYLIZED);
            result.stylization_applied = true;
        } else {
            result.final_texture = get_shared_texture(PainterlyTextureSlot::COLOR);
            result.final_texture_owner = get_shared_texture_owner(PainterlyTextureSlot::COLOR);
        }
    } else {
        result.final_texture = get_shared_texture(PainterlyTextureSlot::COLOR);
        result.final_texture_owner = get_shared_texture_owner(PainterlyTextureSlot::COLOR);
    }

    result.success = result.final_texture.is_valid();
    return result;
}

Error PainterlyRenderer::compile_shaders() {
    Error err = _compile_sobel_shader();
    if (err != OK) {
        return err;
    }

    err = _compile_brush_shader();
    if (err != OK) {
        return err;
    }

    shaders_compiled = true;
    return OK;
}

bool PainterlyRenderer::are_shaders_ready() const {
    return shaders_compiled && sobel_pipeline.is_valid() && brush_pipeline.is_valid();
}

PainterlyPerformance PainterlyRenderer::get_performance() const {
    return cached_performance;
}

void PainterlyRenderer::reset_performance() {
    cached_performance = PainterlyPerformance();
}

uint64_t PainterlyRenderer::get_version() const {
    if (pass_graph) {
        return pass_graph->get_version();
    }
    return 0;
}

Error PainterlyRenderer::_compile_sobel_shader() {
    if (!rd) {
        return ERR_UNCONFIGURED;
    }

    // Create shader source if not exists
    if (!sobel_shader_source) {
        sobel_shader_source = memnew(SobelOutlineShaderRD);
    }

    // Initialize shader source with variants (only once)
    if (!sobel_shader_initialized) {
        Vector<String> variants;
        variants.push_back("");  // Default variant
        sobel_shader_source->initialize(variants);
        sobel_shader_initialized = true;
    }

    // Create shader version
    if (!sobel_shader_version.is_valid()) {
        sobel_shader_version = sobel_shader_source->version_create();
        if (!sobel_shader_version.is_valid()) {
            ERR_PRINT("[PainterlyRenderer] Failed to create sobel shader version");
            return ERR_CANT_CREATE;
        }
    }

    // Get shader RID (variant 0 = default)
    sobel_shader = sobel_shader_source->version_get_shader(sobel_shader_version, 0);
    if (!sobel_shader.is_valid()) {
        ERR_PRINT("[PainterlyRenderer] Failed to get sobel shader");
        return ERR_CANT_CREATE;
    }

    // Create compute pipeline
    sobel_pipeline = rd->compute_pipeline_create(sobel_shader);
    if (!sobel_pipeline.is_valid()) {
        ERR_PRINT("[PainterlyRenderer] Failed to create sobel compute pipeline");
        return ERR_CANT_CREATE;
    }

    return OK;
}

Error PainterlyRenderer::_compile_brush_shader() {
    if (!rd) {
        return ERR_UNCONFIGURED;
    }

    // Create shader source if not exists
    if (!brush_shader_source) {
        brush_shader_source = memnew(BrushAccumulateShaderRD);
    }

    // Initialize shader source with variants (only once)
    if (!brush_shader_initialized) {
        Vector<String> variants;
        variants.push_back("");  // Default variant
        brush_shader_source->initialize(variants);
        brush_shader_initialized = true;
    }

    // Create shader version
    if (!brush_shader_version.is_valid()) {
        brush_shader_version = brush_shader_source->version_create();
        if (!brush_shader_version.is_valid()) {
            ERR_PRINT("[PainterlyRenderer] Failed to create brush shader version");
            return ERR_CANT_CREATE;
        }
    }

    // Get shader RID (variant 0 = default)
    brush_shader = brush_shader_source->version_get_shader(brush_shader_version, 0);
    if (!brush_shader.is_valid()) {
        ERR_PRINT("[PainterlyRenderer] Failed to get brush shader");
        return ERR_CANT_CREATE;
    }

    // Create compute pipeline
    brush_pipeline = rd->compute_pipeline_create(brush_shader);
    if (!brush_pipeline.is_valid()) {
        ERR_PRINT("[PainterlyRenderer] Failed to create brush compute pipeline");
        return ERR_CANT_CREATE;
    }

    return OK;
}

void PainterlyRenderer::_execute_sobel_pass(RID p_color_input) {
    if (!rd || !sobel_pipeline.is_valid() || !pass_graph) {
        return;
    }

    if (painterly_sampler.is_valid() && !painterly_sampler_owner.matches(rd)) {
        _free_tracked_rid(painterly_sampler, painterly_sampler_owner, nullptr, false);
        if (sobel_uniform_set.is_valid() && rd->uniform_set_is_valid(sobel_uniform_set)) {
            rd->free(sobel_uniform_set);
        }
        if (brush_uniform_set.is_valid() && rd->uniform_set_is_valid(brush_uniform_set)) {
            rd->free(brush_uniform_set);
        }
        sobel_uniform_set = RID();
        brush_uniform_set = RID();
        sobel_uniform_sampler = RID();
        sobel_uniform_color_input = RID();
        sobel_uniform_edge_texture = RID();
        brush_uniform_sampler = RID();
        brush_uniform_color_input = RID();
        brush_uniform_edge_input = RID();
        brush_uniform_stylized_texture = RID();
    }

    // Ensure sampler exists
    if (!painterly_sampler.is_valid()) {
        RD::SamplerState sampler_state;
        sampler_state.mag_filter = RD::SAMPLER_FILTER_LINEAR;
        sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
        sampler_state.mip_filter = RD::SAMPLER_FILTER_LINEAR;
        sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
        sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
        sampler_state.repeat_w = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
        painterly_sampler = rd->sampler_create(sampler_state);
        if (painterly_sampler.is_valid()) {
            rd->set_resource_name(painterly_sampler, "GS_PainterlyRenderer_PainterlySampler");
            painterly_sampler_owner.set(rd);
        }
    }

    RID edge_texture = pass_graph->get_texture(PainterlyPassGraph::TEXTURE_EDGE);
    if (!p_color_input.is_valid() || !edge_texture.is_valid()) {
        return;
    }

    const bool sobel_needs_recreate = !sobel_uniform_set.is_valid() ||
            sobel_uniform_sampler != painterly_sampler ||
            sobel_uniform_color_input != p_color_input ||
            sobel_uniform_edge_texture != edge_texture;
    if (sobel_needs_recreate) {
        if (sobel_uniform_set.is_valid() && rd->uniform_set_is_valid(sobel_uniform_set)) {
            rd->free(sobel_uniform_set);
        }
        sobel_uniform_set = RID();

        Vector<RD::Uniform> uniforms;

        RD::Uniform color_uniform;
        color_uniform.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
        color_uniform.binding = 0;
        color_uniform.append_id(painterly_sampler);
        color_uniform.append_id(p_color_input);
        uniforms.push_back(color_uniform);

        RD::Uniform edge_uniform;
        edge_uniform.uniform_type = RD::UNIFORM_TYPE_IMAGE;
        edge_uniform.binding = 1;
        edge_uniform.append_id(edge_texture);
        uniforms.push_back(edge_uniform);

        sobel_uniform_set = rd->uniform_set_create(uniforms, sobel_shader, 0);
        if (sobel_uniform_set.is_valid()) {
            rd->set_resource_name(sobel_uniform_set, "GS_PainterlyRenderer_SobelUniformSet");
            sobel_uniform_sampler = painterly_sampler;
            sobel_uniform_color_input = p_color_input;
            sobel_uniform_edge_texture = edge_texture;
        }
    }

    if (!sobel_uniform_set.is_valid()) {
        return;
    }

    // Get internal size for dispatch
    Vector2i internal_size = pass_graph->get_internal_size();
    uint32_t groups_x = (uint32_t)((internal_size.x + 7) / 8);
    uint32_t groups_y = (uint32_t)((internal_size.y + 7) / 8);

    // Push constants struct matching shader layout
    struct SobelParams {
        float texel_size[2];
        float intensity;
        float threshold;
    } params;

    params.texel_size[0] = internal_size.x > 0 ? 1.0f / float(internal_size.x) : 1.0f;
    params.texel_size[1] = internal_size.y > 0 ? 1.0f / float(internal_size.y) : 1.0f;
    params.intensity = current_config.edge_intensity;
    params.threshold = current_config.edge_threshold;

    // Execute compute pass
    RD::ComputeListID compute_list = rd->compute_list_begin();
    if (compute_list == RD::INVALID_ID) {
        return;
    }

    rd->compute_list_bind_compute_pipeline(compute_list, sobel_pipeline);
    rd->compute_list_bind_uniform_set(compute_list, sobel_uniform_set, 0);
    rd->compute_list_set_push_constant(compute_list, &params, sizeof(SobelParams));
    rd->compute_list_dispatch(compute_list, groups_x, groups_y, 1);
    rd->compute_list_add_barrier(compute_list);
    rd->compute_list_end();
}

void PainterlyRenderer::_execute_brush_pass(RID p_color_input, RID p_edge_input) {
    if (!rd || !brush_pipeline.is_valid() || !pass_graph) {
        return;
    }

    if (painterly_sampler.is_valid() && !painterly_sampler_owner.matches(rd)) {
        _free_tracked_rid(painterly_sampler, painterly_sampler_owner, nullptr, false);
        if (sobel_uniform_set.is_valid() && rd->uniform_set_is_valid(sobel_uniform_set)) {
            rd->free(sobel_uniform_set);
        }
        if (brush_uniform_set.is_valid() && rd->uniform_set_is_valid(brush_uniform_set)) {
            rd->free(brush_uniform_set);
        }
        sobel_uniform_set = RID();
        brush_uniform_set = RID();
        sobel_uniform_sampler = RID();
        sobel_uniform_color_input = RID();
        sobel_uniform_edge_texture = RID();
        brush_uniform_sampler = RID();
        brush_uniform_color_input = RID();
        brush_uniform_edge_input = RID();
        brush_uniform_stylized_texture = RID();
    }

    // Ensure sampler exists
    if (!painterly_sampler.is_valid()) {
        RD::SamplerState sampler_state;
        sampler_state.mag_filter = RD::SAMPLER_FILTER_LINEAR;
        sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
        sampler_state.mip_filter = RD::SAMPLER_FILTER_LINEAR;
        sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
        sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
        sampler_state.repeat_w = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
        painterly_sampler = rd->sampler_create(sampler_state);
        if (painterly_sampler.is_valid()) {
            rd->set_resource_name(painterly_sampler, "GS_PainterlyRenderer_PainterlySampler");
            painterly_sampler_owner.set(rd);
        }
    }

    RID stylized_texture = pass_graph->get_texture(PainterlyPassGraph::TEXTURE_STYLIZED);
    if (!p_color_input.is_valid() || !p_edge_input.is_valid() || !stylized_texture.is_valid()) {
        return;
    }

    const bool brush_needs_recreate = !brush_uniform_set.is_valid() ||
            brush_uniform_sampler != painterly_sampler ||
            brush_uniform_color_input != p_color_input ||
            brush_uniform_edge_input != p_edge_input ||
            brush_uniform_stylized_texture != stylized_texture;
    if (brush_needs_recreate) {
        if (brush_uniform_set.is_valid() && rd->uniform_set_is_valid(brush_uniform_set)) {
            rd->free(brush_uniform_set);
        }
        brush_uniform_set = RID();

        Vector<RD::Uniform> uniforms;

        RD::Uniform color_uniform;
        color_uniform.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
        color_uniform.binding = 0;
        color_uniform.append_id(painterly_sampler);
        color_uniform.append_id(p_color_input);
        uniforms.push_back(color_uniform);

        RD::Uniform edge_uniform;
        edge_uniform.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
        edge_uniform.binding = 1;
        edge_uniform.append_id(painterly_sampler);
        edge_uniform.append_id(p_edge_input);
        uniforms.push_back(edge_uniform);

        RD::Uniform stylized_uniform;
        stylized_uniform.uniform_type = RD::UNIFORM_TYPE_IMAGE;
        stylized_uniform.binding = 2;
        stylized_uniform.append_id(stylized_texture);
        uniforms.push_back(stylized_uniform);

        brush_uniform_set = rd->uniform_set_create(uniforms, brush_shader, 0);
        if (brush_uniform_set.is_valid()) {
            rd->set_resource_name(brush_uniform_set, "GS_PainterlyRenderer_BrushUniformSet");
            brush_uniform_sampler = painterly_sampler;
            brush_uniform_color_input = p_color_input;
            brush_uniform_edge_input = p_edge_input;
            brush_uniform_stylized_texture = stylized_texture;
        }
    }

    if (!brush_uniform_set.is_valid()) {
        return;
    }

    // Get internal size for dispatch
    Vector2i internal_size = pass_graph->get_internal_size();
    uint32_t groups_x = (uint32_t)((internal_size.x + 7) / 8);
    uint32_t groups_y = (uint32_t)((internal_size.y + 7) / 8);

    // Push constants struct matching shader layout
    struct BrushParams {
        float stroke_opacity;
        float edge_strength;
        float stroke_length;
        float gamma;
    } params;

    params.stroke_opacity = current_config.stroke_opacity;
    params.edge_strength = current_config.edge_intensity;
    params.stroke_length = current_config.stroke_length;
    params.gamma = current_config.gamma;

    // Execute compute pass
    RD::ComputeListID compute_list = rd->compute_list_begin();
    if (compute_list == RD::INVALID_ID) {
        return;
    }

    rd->compute_list_bind_compute_pipeline(compute_list, brush_pipeline);
    rd->compute_list_bind_uniform_set(compute_list, brush_uniform_set, 0);
    rd->compute_list_set_push_constant(compute_list, &params, sizeof(BrushParams));
    rd->compute_list_dispatch(compute_list, groups_x, groups_y, 1);
    rd->compute_list_end();
}

// Phase 2: Full render pipeline with internal rasterizer
PainterlyRenderResult PainterlyRenderer::render(const PainterlyRenderInput &p_input) {
    PainterlyRenderResult result;

    if (!rd) {
        return result;
    }

    // Ensure internal rasterizer exists
    if (!internal_rasterizer.is_valid()) {
        internal_rasterizer.instantiate();
        Error err = internal_rasterizer->initialize(rd, p_input.viewport_size,
                                                    current_config.enable_stylization ? 8 : -1,
                                                    current_config.color_format);
        if (err != OK) {
            return result;
        }
        owns_rasterizer = true;
    }

    // Ensure pass graph is configured for this viewport size
    if (pass_graph) {
        Vector2i internal_size = Vector2i(
            MAX(1, int(p_input.viewport_size.x * current_config.internal_scale)),
            MAX(1, int(p_input.viewport_size.y * current_config.internal_scale)));
        pass_graph->configure(internal_size, current_config.internal_scale,
                              current_config.enable_stylization, current_config.low_end_mode);
    }

    // Build raster params from input
    RasterParams raster_params;
    raster_params.device = rd;
    raster_params.gaussian_buffer = p_input.gaussian_buffer;
    raster_params.sorted_indices = p_input.sorted_indices;
    raster_params.splat_count = p_input.splat_count;
    raster_params.total_gaussians = p_input.total_gaussians;
    raster_params.world_to_camera_transform = p_input.world_to_camera_transform;
    raster_params.camera_to_world_transform = p_input.camera_to_world_transform; // PERF (#659): Pass pre-computed inverse
    raster_params.projection = p_input.projection;
    raster_params.render_projection = p_input.render_projection;
    raster_params.viewport_size = p_input.viewport_size;
    raster_params.interactive_state_uniform = p_input.interactive_state_uniform;
    raster_params.tile_size = 8;
    raster_params.output_is_premultiplied = true;

    // Rasterize gaussians to G-buffer
    RasterResult gbuffer = internal_rasterizer->render(raster_params);
    if (!gbuffer.success) {
        return result;
    }

    // Execute post-processing passes (sobel + brush)
    return execute_passes(gbuffer.output_texture, gbuffer.depth_texture);
}

// Phase 1 Extension: Material texture management
void PainterlyRenderer::set_material_textures(const LocalVector<RID> &p_palette,
                                               const LocalVector<RID> &p_noise_luts,
                                               RID p_stroke_density_buffer) {
    cached_palette_textures.clear();
    cached_noise_luts.clear();

    for (uint32_t i = 0; i < p_palette.size(); i++) {
        cached_palette_textures.push_back(p_palette[i]);
    }
    for (uint32_t i = 0; i < p_noise_luts.size(); i++) {
        cached_noise_luts.push_back(p_noise_luts[i]);
    }
    cached_stroke_density_buffer = p_stroke_density_buffer;
    material_textures_dirty = true;
}

void PainterlyRenderer::clear_material_textures() {
    cached_palette_textures.clear();
    cached_noise_luts.clear();
    cached_stroke_density_buffer = RID();
    material_textures_dirty = true;
}

// Phase 1 Extension: Rasterizer access
void PainterlyRenderer::set_rasterizer(Ref<TileRasterizer> p_rasterizer) {
    // If we own the current rasterizer, shut it down
    if (owns_rasterizer && internal_rasterizer.is_valid()) {
        internal_rasterizer->shutdown();
    }

    internal_rasterizer = p_rasterizer;
    owns_rasterizer = false;  // External rasterizer, we don't own it
}

Ref<TileRasterizer> PainterlyRenderer::get_rasterizer() const {
    return internal_rasterizer;
}

// Phase 5: Composite shader compilation
Error PainterlyRenderer::_compile_composite_shader() {
    if (!rd) {
        return ERR_UNCONFIGURED;
    }

    if (composite_failed) {
        return ERR_COMPILATION_FAILED;
    }

    // Create shader source if not exists
    if (!composite_shader_source) {
        composite_shader_source = memnew(PainterlyCompositeShaderRD);
    }

    // Initialize shader source with variants (only once)
    if (!composite_shader_initialized) {
        Vector<String> variants;
        variants.push_back("");  // Default variant
        composite_shader_source->initialize(variants);
        composite_shader_initialized = true;
    }

    // Create shader version
    if (!composite_shader_version.is_valid()) {
        composite_shader_version = composite_shader_source->version_create();
        if (!composite_shader_version.is_valid()) {
            ERR_PRINT("[PainterlyRenderer] Failed to create composite shader version");
            composite_failed = true;
            return ERR_CANT_CREATE;
        }
    }

    // Get shader RID (variant 0 = default)
    composite_shader = composite_shader_source->version_get_shader(composite_shader_version, 0);
    if (!composite_shader.is_valid()) {
        ERR_PRINT("[PainterlyRenderer] Failed to get composite shader");
        composite_failed = true;
        return ERR_CANT_CREATE;
    }

    return OK;
}

// Phase 5: Ensure composite resources
void PainterlyRenderer::_ensure_composite_resources() {
    if (!rd || composite_failed) {
        return;
    }

    // Compile shader if needed
    if (!composite_shader.is_valid()) {
        if (_compile_composite_shader() != OK) {
            return;
        }
    }

    if (composite_sampler.is_valid() && !composite_sampler_owner.matches(rd)) {
        _free_tracked_rid(composite_sampler, composite_sampler_owner, nullptr, false);
    }

    // Create composite sampler if needed
    if (!composite_sampler.is_valid()) {
        RD::SamplerState sampler_state;
        sampler_state.mag_filter = RD::SAMPLER_FILTER_LINEAR;
        sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
        sampler_state.mip_filter = RD::SAMPLER_FILTER_LINEAR;
        sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
        sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
        sampler_state.repeat_w = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
        composite_sampler = rd->sampler_create(sampler_state);
        if (composite_sampler.is_valid()) {
            rd->set_resource_name(composite_sampler, "GS_PainterlyRenderer_CompositeSampler");
            composite_sampler_owner.set(rd);
        }
    }

    if (composite_depth_sampler.is_valid() && !composite_depth_sampler_owner.matches(rd)) {
        _free_tracked_rid(composite_depth_sampler, composite_depth_sampler_owner, nullptr, false);
    }

    // Create depth sampler if needed
    if (!composite_depth_sampler.is_valid()) {
        RD::SamplerState sampler_state;
        sampler_state.mag_filter = RD::SAMPLER_FILTER_NEAREST;
        sampler_state.min_filter = RD::SAMPLER_FILTER_NEAREST;
        sampler_state.mip_filter = RD::SAMPLER_FILTER_NEAREST;
        sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
        sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
        sampler_state.repeat_w = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
        sampler_state.enable_compare = false;
        composite_depth_sampler = rd->sampler_create(sampler_state);
        if (composite_depth_sampler.is_valid()) {
            rd->set_resource_name(composite_depth_sampler, "GS_PainterlyRenderer_CompositeDepthSampler");
            composite_depth_sampler_owner.set(rd);
        }
    }

    composite_initialized = true;
}

void PainterlyRenderer::_update_painterly_texture_tracking(GaussianSplatRenderer *p_renderer) {
    if (!p_renderer || !pass_graph || !pass_graph->is_ready()) {
        _forget_painterly_texture_tracking(p_renderer);
        return;
    }

    GaussianSplatRenderer::FrameStateProvider frame_provider(p_renderer);
    const GaussianSplatRenderer::IFrameStateView &state_view = frame_provider;
    RenderingDevice *fallback_device = state_view.get_rendering_device();
    RenderingDevice *local_tracking_device = rd ? rd : fallback_device;
    for (int slot = 0; slot < PainterlyPassGraph::TEXTURE_COUNT; slot++) {
        PainterlyPassGraph::TextureSlot texture_slot = static_cast<PainterlyPassGraph::TextureSlot>(slot);
        const PainterlyPassGraph::TextureInfo &info = pass_graph->get_texture_info(texture_slot);

        RID local_texture = info.texture;
        if (local_texture.is_valid()) {
            p_renderer->track_resource_owner(local_texture, local_tracking_device, false, "painterly_local_texture");
            tracked_local_textures[slot] = local_texture;
        } else if (tracked_local_textures[slot].is_valid()) {
            p_renderer->forget_resource_owner(tracked_local_textures[slot]);
            tracked_local_textures[slot] = RID();
        }

        RID shared_texture = pass_graph->get_shared_texture(texture_slot);
        RenderingDevice *shared_owner = pass_graph->get_shared_texture_owner(texture_slot);
        if (shared_texture.is_valid()) {
            p_renderer->track_resource_owner(shared_texture, shared_owner ? shared_owner : local_tracking_device, false, "painterly_shared_texture");
            tracked_shared_textures[slot] = shared_texture;
        } else if (tracked_shared_textures[slot].is_valid()) {
            p_renderer->forget_resource_owner(tracked_shared_textures[slot]);
            tracked_shared_textures[slot] = RID();
        }
    }
}

void PainterlyRenderer::_forget_painterly_texture_tracking(GaussianSplatRenderer *p_renderer) {
    if (!p_renderer) {
        for (int slot = 0; slot < PainterlyPassGraph::TEXTURE_COUNT; slot++) {
            tracked_local_textures[slot] = RID();
            tracked_shared_textures[slot] = RID();
        }
        return;
    }

    for (int slot = 0; slot < PainterlyPassGraph::TEXTURE_COUNT; slot++) {
        if (tracked_local_textures[slot].is_valid()) {
            p_renderer->forget_resource_owner(tracked_local_textures[slot]);
            tracked_local_textures[slot] = RID();
        }
        if (tracked_shared_textures[slot].is_valid()) {
            p_renderer->forget_resource_owner(tracked_shared_textures[slot]);
            tracked_shared_textures[slot] = RID();
        }
    }
}

void PainterlyRenderer::execute_painterly_passes(GaussianSplatRenderer *p_renderer, const Size2i &p_internal_size) {
    if (!p_renderer || !pass_graph) {
        return;
    }
    if (!p_renderer->ensure_rendering_device("execute_painterly_passes") || !pass_graph->is_ready()) {
        return;
    }

    const auto &painterly_config = p_renderer->get_painterly_config();
    const bool stylization_requested = painterly_config.enabled && painterly_config.enable_strokes;
    if (!are_shaders_ready()) {
        if (stylization_requested) {
            GS_LOG_WARN_DEFAULT("[Painterly] PainterlyRenderer not ready; painterly passes skipped");
        }
        return;
    }

    // Update config to match current settings
    PainterlyConfig config = get_config();
    config.viewport_size = p_internal_size;
    config.internal_scale = painterly_config.internal_scale;
    config.enable_stylization = stylization_requested;
    config.low_end_mode = painterly_config.low_end_mode;
    config.edge_threshold = painterly_config.edge_threshold;
    config.edge_intensity = painterly_config.edge_intensity;
    config.stroke_length = painterly_config.stroke_length;
    config.stroke_opacity = painterly_config.stroke_opacity;
    config.gamma = painterly_config.gamma;
    configure(config);

    // Get color texture from our pass graph and execute passes via PainterlyRenderer
    RID color_texture = pass_graph->get_texture(PainterlyPassGraph::TEXTURE_COLOR);
    RID depth_texture = pass_graph->get_texture(PainterlyPassGraph::TEXTURE_DEPTH);
    if (color_texture.is_valid()) {
        PainterlyRenderResult result = execute_passes(color_texture, depth_texture);
        if (!result.success) {
            GS_LOG_WARN_DEFAULT("[Painterly] PainterlyRenderer execute_passes failed");
        }
    }
}

void PainterlyRenderer::ensure_painterly_resources(GaussianSplatRenderer *p_renderer, const Size2i &p_viewport_size,
        RD::DataFormat p_target_format) {
    if (!p_renderer || !pass_graph) {
        return;
    }
    if (!p_renderer->ensure_rendering_device("ensure_painterly_resources")) {
        return;
    }

    const auto &view_state = p_renderer->get_view_state();
    RD::DataFormat effective_target_format = p_target_format != RD::DATA_FORMAT_MAX ? p_target_format : view_state.active_viewport_color_format;
    if (effective_target_format != RD::DATA_FORMAT_MAX) {
        pass_graph->set_color_format(effective_target_format);
    }

    const auto &painterly_config = p_renderer->get_painterly_config();
    bool stylization_requested = painterly_config.enabled && painterly_config.enable_strokes;
    pass_graph->configure(p_viewport_size, painterly_config.internal_scale, stylization_requested,
            painterly_config.low_end_mode);
    _update_painterly_texture_tracking(p_renderer);
    // Uniform sets for sobel/brush passes are now managed by PainterlyRenderer
}

Error PainterlyRenderer::render_painterly_frame(GaussianSplatRenderer *p_renderer, const Size2i &p_viewport_size,
        RD::DataFormat p_target_format, const Transform3D &p_world_to_camera_transform, const Projection &p_projection,
        const Projection &p_render_projection, RID &r_final_output, Size2i &r_internal_size, float &r_render_time_ms) {
    r_final_output = RID();
    r_internal_size = p_viewport_size;
    r_render_time_ms = 0.0f;

    if (!p_renderer || !pass_graph) {
        return ERR_UNCONFIGURED;
    }

    ensure_painterly_resources(p_renderer, p_viewport_size, p_target_format);

    if (!pass_graph->is_ready()) {
        const auto &painterly_config = p_renderer->get_painterly_config();
        const bool stylization_requested = painterly_config.enabled && painterly_config.enable_strokes;
        pass_graph->configure(p_viewport_size, painterly_config.internal_scale,
                stylization_requested, painterly_config.low_end_mode);
        if (!pass_graph->is_ready()) {
            return ERR_UNAVAILABLE;
        }
    }

    r_internal_size = pass_graph->get_internal_size();

    uint64_t populate_start = OS::get_singleton()->get_ticks_usec();
    Error populate_err = populate_painterly_gbuffer(p_renderer, r_internal_size, p_world_to_camera_transform, p_projection, p_render_projection);
    uint64_t populate_end = OS::get_singleton()->get_ticks_usec();
    if (populate_err != OK) {
        return populate_err;
    }

    float render_time_ms = (populate_end - populate_start) / 1000.0f;

    auto use_painterly_color = [&]() {
        RID color_texture = pass_graph->get_texture(PainterlyPassGraph::TEXTURE_COLOR);

        if (color_texture.is_valid()) {
            uint64_t start_time = OS::get_singleton()->get_ticks_usec();
            execute_painterly_passes(p_renderer, r_internal_size);
            uint64_t end_time = OS::get_singleton()->get_ticks_usec();
            render_time_ms += (end_time - start_time) / 1000.0f;

            r_final_output = pass_graph->get_shared_texture(PainterlyPassGraph::TEXTURE_COLOR);

            if (pass_graph->is_stylization_enabled()) {
                RID stylized_texture = pass_graph->get_shared_texture(PainterlyPassGraph::TEXTURE_STYLIZED);
                if (stylized_texture.is_valid()) {
                    r_final_output = stylized_texture;
                }
            }
        }
    };

    GaussianSplatRenderer::FrameStateProvider frame_provider(p_renderer);
    const GaussianSplatRenderer::IFrameStateView &state_view = frame_provider;
    const auto &subsystem_state = state_view.get_subsystem_state_view();
    if (subsystem_state.rasterizer.is_valid()) {
        RID tile_color = subsystem_state.rasterizer->get_output_texture();
        RID tile_depth = subsystem_state.rasterizer->has_depth_output()
                ? subsystem_state.rasterizer->get_depth_texture()
                : RID();
        RenderingDevice *color_owner = subsystem_state.rasterizer->get_output_texture_owner();
        RenderingDevice *depth_owner = subsystem_state.rasterizer->has_depth_output()
                ? subsystem_state.rasterizer->get_depth_texture_owner()
                : nullptr;
        p_renderer->update_tile_renderer_output_tracking(tile_color, color_owner, tile_depth, depth_owner);

        if (tile_color.is_valid()) {
            r_final_output = tile_color;

            if (pass_graph->is_stylization_enabled()) {
                uint64_t start_time = OS::get_singleton()->get_ticks_usec();
                execute_painterly_passes(p_renderer, r_internal_size);
                uint64_t end_time = OS::get_singleton()->get_ticks_usec();
                render_time_ms += (end_time - start_time) / 1000.0f;

                RID stylized_texture = pass_graph->get_shared_texture(PainterlyPassGraph::TEXTURE_STYLIZED);
                if (stylized_texture.is_valid()) {
                    r_final_output = stylized_texture;
                }
            }
        } else {
            use_painterly_color();
        }
    } else {
        use_painterly_color();
    }

    r_render_time_ms = render_time_ms;
    if (!r_final_output.is_valid()) {
        return ERR_UNCONFIGURED;
    }

    return OK;
}

void PainterlyRenderer::_ensure_painterly_composite_resources(GaussianSplatRenderer *p_renderer, RD::FramebufferFormatID p_framebuffer_format) {
    if (!p_renderer) {
        return;
    }

    if (painterly_composite_failed) {
        return;
    }

    if (!p_renderer->ensure_rendering_device("_ensure_painterly_composite_resources")) {
        return;
    }

    GaussianSplatRenderer::FrameStateProvider frame_provider(p_renderer);
    const GaussianSplatRenderer::IFrameStateView &state_view = frame_provider;

    if (!painterly_composite_shader.is_valid()) {
        Vector<String> vertex_paths;
        vertex_paths.push_back("res://modules/gaussian_splatting/shaders/painterly_composite.vert.glsl");
        vertex_paths.push_back("modules/gaussian_splatting/shaders/painterly_composite.vert.glsl");

        Vector<String> fragment_paths;
        fragment_paths.push_back("res://modules/gaussian_splatting/shaders/painterly_composite.frag.glsl");
        fragment_paths.push_back("modules/gaussian_splatting/shaders/painterly_composite.frag.glsl");

        painterly_composite_shader = p_renderer->load_graphics_shader(vertex_paths, fragment_paths);
        if (!painterly_composite_shader.is_valid()) {
            WARN_PRINT_ONCE("[Painterly] Failed to load composite shader for fullscreen pass");
            painterly_composite_failed = true;
            return;
        }
    }
    if (painterly_composite_shader.is_valid()) {
        painterly_composite_shader_owner.set(
                p_renderer->get_resource_owner(painterly_composite_shader, state_view.get_rendering_device()));
    }

    if (!painterly_composite_pipeline_initialized && painterly_composite_shader.is_valid()) {
        RD::PipelineRasterizationState raster_state;
        RD::PipelineMultisampleState multisample_state;
        RD::PipelineDepthStencilState depth_state;
        depth_state.enable_depth_test = false;
        depth_state.enable_depth_write = false;

        RD::PipelineColorBlendState blend_state;
        RD::PipelineColorBlendState::Attachment attachment;
        attachment.enable_blend = true;
        // Composite assumes premultiplied input (tile raster output)
        attachment.src_color_blend_factor = RD::BLEND_FACTOR_ONE;
        attachment.dst_color_blend_factor = RD::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        attachment.color_blend_op = RD::BLEND_OP_ADD;
        attachment.src_alpha_blend_factor = RD::BLEND_FACTOR_ONE;
        attachment.dst_alpha_blend_factor = RD::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        attachment.alpha_blend_op = RD::BLEND_OP_ADD;
        blend_state.attachments.push_back(attachment);

        painterly_composite_pipeline.setup(painterly_composite_shader, RD::RENDER_PRIMITIVE_TRIANGLES,
                raster_state, multisample_state, depth_state, blend_state);
        painterly_composite_pipeline_initialized = true;
    }

    RenderingDevice *viewport_device = p_renderer->get_main_rendering_device();

    if (painterly_depth_sampler.is_valid() && !painterly_depth_sampler_owner.matches(viewport_device)) {
        _free_tracked_rid(painterly_depth_sampler, painterly_depth_sampler_owner, p_renderer, true);
    }

    if (!painterly_depth_sampler.is_valid() && viewport_device) {
        RD::SamplerState sampler_state;
        sampler_state.mag_filter = RD::SAMPLER_FILTER_NEAREST;
        sampler_state.min_filter = RD::SAMPLER_FILTER_NEAREST;
        sampler_state.mip_filter = RD::SAMPLER_FILTER_NEAREST;
        sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
        sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
        sampler_state.repeat_w = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
        sampler_state.enable_compare = false;
        painterly_depth_sampler = viewport_device->sampler_create(sampler_state);
        if (painterly_depth_sampler.is_valid()) {
            viewport_device->set_resource_name(painterly_depth_sampler, "GS_PainterlyRenderer_PainterlyDepthSampler");
            painterly_depth_sampler_owner.set(viewport_device);
            p_renderer->track_resource_owner(painterly_depth_sampler, viewport_device);
        }
    }

    if (painterly_color_sampler.is_valid() && !painterly_color_sampler_owner.matches(viewport_device)) {
        _free_tracked_rid(painterly_color_sampler, painterly_color_sampler_owner, p_renderer, true);
    }

    // Create color sampler with LINEAR filtering for composite color texture
    if (!painterly_color_sampler.is_valid() && viewport_device) {
        RD::SamplerState sampler_state;
        sampler_state.mag_filter = RD::SAMPLER_FILTER_LINEAR;
        sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
        sampler_state.mip_filter = RD::SAMPLER_FILTER_LINEAR;
        sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
        sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
        sampler_state.repeat_w = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
        sampler_state.enable_compare = false;
        painterly_color_sampler = viewport_device->sampler_create(sampler_state);
        if (painterly_color_sampler.is_valid()) {
            viewport_device->set_resource_name(painterly_color_sampler, "GS_PainterlyRenderer_PainterlyColorSampler");
            painterly_color_sampler_owner.set(viewport_device);
            p_renderer->track_resource_owner(painterly_color_sampler, viewport_device);
        }
    }

    if (painterly_composite_pipeline_initialized) {
        painterly_composite_pipeline.get_render_pipeline(RD::INVALID_ID, p_framebuffer_format);
    }
}

bool PainterlyRenderer::composite_painterly_output(GaussianSplatRenderer *p_renderer, RenderDataRD *p_render_data, RID p_color_texture,
        RID p_depth_texture, const Size2i &p_viewport_size) {
    if (!p_renderer || !p_renderer->ensure_rendering_device("_composite_painterly_output") || !p_render_data || !p_render_data->render_buffers.is_valid()) {
        return false;
    }

    RenderingDevice *viewport_device = p_renderer->get_main_rendering_device();
    if (!viewport_device) {
        GS_LOG_WARN_DEFAULT("[Painterly] No RenderingDevice available for viewport composite submission");
        return false;
    }

    RenderSceneBuffersRD *render_buffers_rd = Object::cast_to<RenderSceneBuffersRD>(p_render_data->render_buffers.ptr());
    if (!render_buffers_rd) {
        return false;
    }

    if (!render_buffers_rd->has_internal_texture() || !render_buffers_rd->has_depth_texture()) {
        return false;
    }

    RID scene_color = render_buffers_rd->get_internal_texture();
    RID scene_depth = render_buffers_rd->get_depth_texture();
    if (!scene_color.is_valid() || !scene_depth.is_valid()) {
        return false;
    }

    if (!p_color_texture.is_valid() || !p_depth_texture.is_valid()) {
        return false;
    }

    RenderingDevice *color_device = p_renderer->get_resource_owner(scene_color, viewport_device);
    RenderingDevice *depth_device = p_renderer->get_resource_owner(scene_depth, viewport_device);
    RenderingDevice *painterly_color_device = p_renderer->get_resource_owner(p_color_texture, viewport_device);
    RenderingDevice *painterly_depth_device = p_renderer->get_resource_owner(p_depth_texture, viewport_device);

    RenderingDevice *draw_device = color_device ? color_device : viewport_device;
    if (painterly_color_device && draw_device != painterly_color_device) {
        GS_LOG_ERROR_DEFAULT("[Painterly] Color texture device mismatch during composite");
        return false;
    }
    if (painterly_depth_device && draw_device != painterly_depth_device) {
        GS_LOG_ERROR_DEFAULT("[Painterly] Depth texture device mismatch during composite");
        return false;
    }
    if (depth_device && draw_device != depth_device) {
        GS_LOG_ERROR_DEFAULT("[Painterly] Scene depth texture device mismatch during composite");
        return false;
    }

    FramebufferCacheRD *framebuffer_cache = FramebufferCacheRD::get_singleton();
    UniformSetCacheRD *uniform_cache = UniformSetCacheRD::get_singleton();
    if (!framebuffer_cache || !uniform_cache) {
        return false;
    }

    RID framebuffer = framebuffer_cache->get_cache(scene_color);
    if (!framebuffer.is_valid()) {
        return false;
    }

    RD::FramebufferFormatID fb_format = draw_device->framebuffer_get_format(framebuffer);
    _ensure_painterly_composite_resources(p_renderer, fb_format);

    if (!painterly_composite_pipeline_initialized || !painterly_composite_shader.is_valid() ||
            !painterly_depth_sampler.is_valid() || !painterly_color_sampler.is_valid()) {
        return false;
    }

    RD::Uniform color_uniform;
    color_uniform.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
    color_uniform.binding = 0;
    color_uniform.append_id(painterly_color_sampler);
    color_uniform.append_id(p_color_texture);

    RD::Uniform painterly_depth_uniform;
    painterly_depth_uniform.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
    painterly_depth_uniform.binding = 1;
    painterly_depth_uniform.append_id(painterly_depth_sampler);
    painterly_depth_uniform.append_id(p_depth_texture);

    RD::Uniform scene_depth_uniform;
    scene_depth_uniform.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
    scene_depth_uniform.binding = 2;
    scene_depth_uniform.append_id(painterly_depth_sampler);
    scene_depth_uniform.append_id(scene_depth);

    RID uniform_set = uniform_cache->get_cache(
            painterly_composite_shader, 0, color_uniform, painterly_depth_uniform, scene_depth_uniform);
    if (!uniform_set.is_valid()) {
        return false;
    }

    Rect2i viewport_rect(Vector2i(), p_viewport_size);
    RD::DrawListID draw_list = draw_device->draw_list_begin(framebuffer, RD::DRAW_DEFAULT_ALL, Vector<Color>(), 1.0f, 0, viewport_rect);

    RID pipeline = painterly_composite_pipeline.get_render_pipeline(RD::INVALID_ID, fb_format);
    if (!pipeline.is_valid()) {
        draw_device->draw_list_end();
        return false;
    }

    draw_device->draw_list_bind_render_pipeline(draw_list, pipeline);
    draw_device->draw_list_bind_uniform_set(draw_list, uniform_set, 0);

    GaussianSplatRenderer::PainterlyCompositePushConstant push_constant;
    push_constant.inv_viewport_size[0] = p_viewport_size.x > 0 ? 1.0f / float(p_viewport_size.x) : 1.0f;
    push_constant.inv_viewport_size[1] = p_viewport_size.y > 0 ? 1.0f / float(p_viewport_size.y) : 1.0f;
    push_constant.depth_bias = 0.0005f;
    push_constant.blend_strength = 1.0f;
    const auto &view_state = p_renderer->get_view_state();
    push_constant.near_plane = view_state.last_camera_projection.get_z_near();
    push_constant.far_plane = view_state.last_camera_projection.get_z_far();
    const Projection &proj = view_state.last_camera_projection;
    push_constant.proj_22 = proj.columns[2][2];
    push_constant.proj_32 = proj.columns[3][2];
    push_constant.proj_23 = proj.columns[2][3];

    draw_device->draw_list_set_push_constant(draw_list, &push_constant, sizeof(GaussianSplatRenderer::PainterlyCompositePushConstant));
    draw_device->draw_list_draw(draw_list, false, 1, 3);
    draw_device->draw_list_end();

    return true;
}

void PainterlyRenderer::free_painterly_resources(GaussianSplatRenderer *p_renderer) {
    _shutdown_internal(p_renderer);
}

void PainterlyRenderer::clear_painterly_gpu_resources(GaussianSplatRenderer *p_renderer) {
    if (!p_renderer) {
        return;
    }
    GaussianSplatRenderer::FrameStateProvider frame_provider(p_renderer);
    GaussianSplatRenderer::IFrameMutationAccess &state_mut = frame_provider;
    // Phase 15: Fully delegated to PainterlyMaterialManager
    auto &subsystem_state = state_mut.get_subsystem_state_mut();
    if (subsystem_state.painterly_material_manager.is_valid()) {
        subsystem_state.painterly_material_manager->clear_resources();
    }
}

void PainterlyRenderer::update_painterly_gpu_resources(GaussianSplatRenderer *p_renderer) {
    if (!p_renderer) {
        return;
    }
    if (!material_dirty) {
        return;
    }

    if (!p_renderer->ensure_rendering_device("update_painterly_gpu_resources")) {
        return;
    }

    GaussianSplatRenderer::FrameStateProvider frame_provider(p_renderer);
    const GaussianSplatRenderer::IFrameStateView &state_view = frame_provider;
    GaussianSplatRenderer::IFrameMutationAccess &state_mut = frame_provider;
    // Phase 15: Delegate resource management to PainterlyMaterialManager
    auto &subsystem_state = state_mut.get_subsystem_state_mut();
    RenderingDevice *rendering_device = state_view.get_rendering_device();
    if (subsystem_state.painterly_material_manager.is_valid()) {
        // Initialize manager if needed (requires RenderingDevice)
        if (!subsystem_state.painterly_material_manager->is_initialized() && rendering_device) {
            subsystem_state.painterly_material_manager->initialize(rendering_device);
        }

        // Manager handles material signal connection and resource updates
        subsystem_state.painterly_material_manager->update_resources();

        // Get resources from manager and pass to painterly renderer
        PainterlyMaterialResources resources = subsystem_state.painterly_material_manager->get_resources();
        if (resources.valid) {
            set_material_textures(
                resources.palette_texture_rids,
                resources.noise_lut_rids,
                resources.stroke_density_buffer
            );
        } else {
            clear_material_textures();
        }

        // Log any missing resources
        Vector<String> missing = subsystem_state.painterly_material_manager->get_missing_resources();
        if (!missing.is_empty()) {
            String warning = "[Painterly] PainterlyMaterial missing required resources: ";
            const int missing_count = missing.size();
            for (int i = 0; i < missing_count; i++) {
                if (i > 0) {
                    warning += ", ";
                }
                warning += missing[i];
            }
            GS_LOG_WARN_DEFAULT(warning);
        }
    }

    material_dirty = false;
}

Error PainterlyRenderer::populate_painterly_gbuffer(GaussianSplatRenderer *p_renderer, const Size2i &p_internal_size,
        const Transform3D &p_world_to_camera_transform, const Projection &p_projection, const Projection &p_render_projection) {
    if (!p_renderer || !pass_graph) {
        return ERR_UNCONFIGURED;
    }

    if (!p_renderer->ensure_rendering_device("_populate_painterly_gbuffer")) {
        return ERR_UNCONFIGURED;
    }

    if (!pass_graph->is_ready()) {
        GS_LOG_ERROR_DEFAULT("[Painterly] Pass graph not ready; skipping G-buffer population");
        return ERR_UNCONFIGURED;
    }

    GaussianSplatRenderer::FrameStateProvider frame_provider(p_renderer);
    const GaussianSplatRenderer::IFrameStateView &state_view = frame_provider;
    GaussianSplatRenderer::IFrameMutationAccess &state_mut = frame_provider;

    const auto &scene_state = state_view.get_scene_state();
    const auto &resource_state = state_view.get_resource_state_view();
    const auto &streaming_state = state_view.get_streaming_state();
    const auto &sorting_state = state_view.get_sorting_state_view();
    auto &culling_config = p_renderer->get_culling_config();
    auto &view_state = p_renderer->get_view_state();
    auto &tile_renderer_state = p_renderer->get_tile_renderer_state();
    auto &subsystem_state = state_mut.get_subsystem_state_mut();
    auto &debug_state = p_renderer->get_debug_state();
    const auto &jacobian_debug = state_view.get_jacobian_debug_view();
    const auto &frame_state = state_view.get_frame_state_view();
    auto &performance_state = state_mut.get_performance_state_mut();
    RenderingDevice *rendering_device = state_view.get_rendering_device();

    RID color_texture = pass_graph->get_texture(PainterlyPassGraph::TEXTURE_COLOR);
    RID depth_texture = pass_graph->get_texture(PainterlyPassGraph::TEXTURE_DEPTH);
    if (!color_texture.is_valid()) {
        GS_LOG_ERROR_DEFAULT("[Painterly] Color buffer unavailable; painterly output cannot be populated");
        return ERR_UNCONFIGURED;
    }

    const int width = MAX(1, p_internal_size.x);
    const int height = MAX(1, p_internal_size.y);

    RD::DataFormat target_output_format = RD::DATA_FORMAT_MAX;
    const PainterlyPassGraph::TextureInfo &color_info = pass_graph->get_texture_info(PainterlyPassGraph::TEXTURE_COLOR);
    if (color_info.valid && color_info.format.format != RD::DATA_FORMAT_MAX) {
        target_output_format = color_info.format.format;
    }
    if (target_output_format == RD::DATA_FORMAT_MAX && view_state.active_viewport_color_format != RD::DATA_FORMAT_MAX) {
        target_output_format = view_state.active_viewport_color_format;
    }
    if (target_output_format == RD::DATA_FORMAT_MAX && view_state.manual_viewport_format_override != RD::DATA_FORMAT_MAX) {
        target_output_format = view_state.manual_viewport_format_override;
    }
    if (target_output_format == RD::DATA_FORMAT_MAX) {
        target_output_format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
    }

    RID manager_gaussian_buffer;
    RID manager_sorted_indices;
    GPUBufferManager::BufferHandle manager_gaussian_handle;
    GPUBufferManager::BufferHandle manager_sorted_handle;
    uint32_t manager_visible_splat_count = 0;
    bool has_buffer_manager_data = scene_state.gaussian_data.is_valid() && resource_state.buffer_manager.is_valid() && resource_state.buffer_manager_initialized;
    if (has_buffer_manager_data) {
        manager_gaussian_handle = resource_state.buffer_manager->get_gaussian_handle();
        manager_sorted_handle = resource_state.buffer_manager->get_sorted_indices_handle();
        if (manager_gaussian_handle.is_valid()) {
            p_renderer->track_resource_owner(manager_gaussian_handle.buffer, manager_gaussian_handle.device);
        }
        if (manager_sorted_handle.is_valid()) {
            p_renderer->track_resource_owner(manager_sorted_handle.buffer, manager_sorted_handle.device);
        }
        manager_gaussian_buffer = manager_gaussian_handle.buffer;
        manager_sorted_indices = manager_sorted_handle.buffer;
        manager_visible_splat_count = resource_state.buffer_manager->get_visible_count();
        if (manager_visible_splat_count == 0) {
            manager_visible_splat_count = resource_state.buffer_manager->get_gaussian_count();
        }
        if (!manager_gaussian_buffer.is_valid() || !manager_sorted_indices.is_valid() || manager_visible_splat_count == 0) {
            has_buffer_manager_data = false;
            manager_gaussian_buffer = RID();
            manager_sorted_indices = RID();
            manager_visible_splat_count = 0;
            manager_gaussian_handle = GPUBufferManager::BufferHandle();
            manager_sorted_handle = GPUBufferManager::BufferHandle();
        }
    }

    RID instance_gaussian_buffer;
    RID instance_sorted_indices;
    uint32_t instance_total_gaussians = 0;
    bool has_instance_data = false;
    if (p_renderer->has_instance_pipeline_buffers()) {
        const auto &instance_buffers = p_renderer->get_instance_pipeline_buffers();
        instance_gaussian_buffer = instance_buffers.atlas_gaussian_buffer;
        if (subsystem_state.sorting_pipeline.is_valid()) {
            instance_sorted_indices = subsystem_state.sorting_pipeline->get_sort_indices_buffer();
        }
        if (instance_gaussian_buffer.is_valid() && instance_sorted_indices.is_valid()) {
            has_instance_data = true;
            instance_total_gaussians = instance_buffers.atlas_gaussian_count;
            if (instance_total_gaussians == 0 && instance_buffers.max_visible_splats > 0) {
                instance_total_gaussians = instance_buffers.max_visible_splats;
            }
        }
    }

    RID stream_gaussian_buffer;
    RID stream_sorted_indices;
    bool has_stream_data = streaming_state.use_streamed_data;
    if (has_stream_data) {
        stream_gaussian_buffer = streaming_state.current_stream_gpu_buffer;
        if (subsystem_state.sorting_pipeline.is_valid()) {
            stream_sorted_indices = subsystem_state.sorting_pipeline->get_sort_indices_buffer();
        }
        has_stream_data = stream_gaussian_buffer.is_valid() && stream_sorted_indices.is_valid();
    }

    uint32_t visible_splat_count = sorting_state.sorted_splat_count;
    if (!has_buffer_manager_data && !has_instance_data && !has_stream_data) {
        GS_LOG_ERROR_DEFAULT("[Painterly] No GPU gaussian buffers available; painterly splatting requires valid GPU data");
        return ERR_UNCONFIGURED;
    }

    TileRenderer::RenderParams render_params;
    if (has_buffer_manager_data) {
        render_params.gaussian_buffer = manager_gaussian_buffer;
        render_params.sorted_indices = manager_sorted_indices;
        if (visible_splat_count > 0) {
            visible_splat_count = MIN(manager_visible_splat_count, visible_splat_count);
        }
    } else if (has_instance_data) {
        render_params.gaussian_buffer = instance_gaussian_buffer;
        render_params.sorted_indices = instance_sorted_indices;
        if (instance_total_gaussians > 0 && visible_splat_count > instance_total_gaussians) {
            visible_splat_count = instance_total_gaussians;
        }
    } else {
        render_params.gaussian_buffer = stream_gaussian_buffer;
        render_params.sorted_indices = stream_sorted_indices;
    }

    render_params.splat_count = visible_splat_count;
    render_params.overlap_record_count = render_params.splat_count; // actual overlap count determined on GPU; this seeds buffer sizing

    uint32_t total_gaussians = 0;
    if (has_buffer_manager_data) {
        if (resource_state.buffer_manager.is_valid() && resource_state.buffer_manager_initialized) {
            total_gaussians = resource_state.buffer_manager->get_gaussian_count();
            if (total_gaussians == 0 && scene_state.gaussian_data.is_valid()) {
                total_gaussians = scene_state.gaussian_data->get_count();
            }
        }
    } else if (has_instance_data) {
        total_gaussians = instance_total_gaussians;
        if (total_gaussians == 0 && visible_splat_count > 0) {
            total_gaussians = visible_splat_count;
        }
    } else {
        // FIX: Use full buffer count, not visible count, for bounds checking
        if (scene_state.gaussian_data.is_valid()) {
            total_gaussians = scene_state.gaussian_data->get_count();
        } else {
            total_gaussians = streaming_state.streaming_gpu_splat_count > 0 ? streaming_state.streaming_gpu_splat_count : visible_splat_count;
        }
    }
    render_params.total_gaussians = total_gaussians;
    if (total_gaussians == 0 && visible_splat_count > 0) {
        GS_LOG_ERROR_DEFAULT("[Painterly] total_gaussians must be set when splat_count > 0");
        return ERR_INVALID_DATA;
    }
	render_params.viewport_size = Vector2i(width, height);
	render_params.world_to_camera_transform = p_world_to_camera_transform;
	render_params.projection = p_projection;
	render_params.render_projection = p_render_projection;
	render_params.tile_size = TileRenderer::DEFAULT_TILE_SIZE;
    render_params.opacity_multiplier = state_view.get_render_config_view().opacity_multiplier;

	// Read debug options from interface subsystem (Phase 8 migration)
	p_renderer->apply_debug_options_to_render_params(render_params);
    render_params.output_is_premultiplied = true;
    render_params.alpha_floor = culling_config.solid_coverage_enabled ? culling_config.solid_coverage_alpha_floor : 0.0f;
    render_params.force_solid_coverage = culling_config.solid_coverage_enabled;
    render_params.cull_far_tolerance = subsystem_state.gpu_culler->get_config().cull_far_tolerance;
    render_params.tiny_splat_screen_radius = subsystem_state.gpu_culler->get_state().tiny_splat_screen_radius_px;
    render_params.opacity_aware_culling = subsystem_state.gpu_culler->get_config().opacity_aware_culling;
    render_params.visibility_threshold = subsystem_state.gpu_culler->get_config().visibility_threshold;
    render_params.distance_cull_enabled = subsystem_state.gpu_culler->get_config().distance_cull_enabled;
    render_params.distance_cull_start = subsystem_state.gpu_culler->get_config().distance_cull_start;
    render_params.distance_cull_max_rate = subsystem_state.gpu_culler->get_config().distance_cull_max_rate;
    float low_pass_filter = render_params.low_pass_filter;
    if (ProjectSettings *ps = ProjectSettings::get_singleton()) {
        static const StringName low_pass_filter_path("rendering/gaussian_splatting/rasterization/low_pass_filter");
        if (ps->has_setting(low_pass_filter_path)) {
            low_pass_filter = (float)ps->get_setting_with_override(low_pass_filter_path);
        }
    }
    render_params.low_pass_filter = CLAMP(low_pass_filter, 0.05f, 2.0f);

    // Jacobian diagnostic toggles for radial stretching investigation
    render_params.jacobian_bypass_radius_depth_floor = jacobian_debug.bypass_radius_depth_floor;
    render_params.jacobian_bypass_j_col2_clamp = jacobian_debug.bypass_j_col2_clamp;
    render_params.jacobian_invert_j_col2_sign = jacobian_debug.invert_j_col2_sign;
    render_params.max_conic_aspect = jacobian_debug.max_conic_aspect;

    // Instance pipeline supplies per-instance inverse rotations; keep global rotation identity.
    render_params.instance_rotation_inverse = Basis();
    render_params.instance_rotation_valid = false;

    render_params.frame_serial = frame_state.frame_counter;
    if (subsystem_state.rasterizer.is_valid()) {
        if (tile_renderer_state.renderer.is_valid()) {
            tile_renderer_state.renderer->set_performance_monitor(&tile_renderer_state.gpu_performance_monitor);
        }
        subsystem_state.rasterizer->set_frame_serial(frame_state.frame_counter);
    }

    RenderingDevice *gaussian_owner_device = nullptr;
    RenderingDevice *index_owner_device = nullptr;
    if (has_buffer_manager_data) {
        if (manager_gaussian_handle.is_valid()) {
            gaussian_owner_device = manager_gaussian_handle.device;
        }
        if (manager_sorted_handle.is_valid()) {
            index_owner_device = manager_sorted_handle.device;
        }
    }

    if (render_params.gaussian_buffer.is_valid()) {
        gaussian_owner_device = p_renderer->get_resource_owner(render_params.gaussian_buffer, gaussian_owner_device);
    }
    if (render_params.sorted_indices.is_valid()) {
        index_owner_device = p_renderer->get_resource_owner(render_params.sorted_indices, index_owner_device ? index_owner_device : gaussian_owner_device);
    }

    gaussian_owner_device = _ensure_local_device(gaussian_owner_device);
    index_owner_device = _ensure_local_device(index_owner_device);

    RenderingDevice *previous_tile_device = nullptr;
    if (subsystem_state.rasterizer.is_valid()) {
        previous_tile_device = subsystem_state.rasterizer->get_output_texture_owner();
        if (!previous_tile_device && subsystem_state.rasterizer->has_depth_output()) {
            previous_tile_device = subsystem_state.rasterizer->get_depth_texture_owner();
        }
    } else if (tile_renderer_state.renderer.is_valid()) {
        previous_tile_device = tile_renderer_state.renderer->get_output_texture_owner();
        if (!previous_tile_device && tile_renderer_state.renderer->has_depth_output()) {
            previous_tile_device = tile_renderer_state.renderer->get_depth_texture_owner();
        }
    }

    // CRITICAL FIX: Always use the main RenderingDevice for TileRenderer
    // The tile subsystem_state.rasterizer's draw list MUST be recorded on the main device
    // so that the output is visible in Godot's presentation pipeline.
    // Using a different device causes the draws to be recorded on an
    // isolated command buffer that never reaches the screen.
    RenderingDevice *tile_device = p_renderer->get_main_rendering_device();
    if (!tile_device) {
        // Fallback to previously-used logic only if main device unavailable
        tile_device = _ensure_local_device(gaussian_owner_device ? gaussian_owner_device : index_owner_device);
    }
    if (!tile_device) {
        tile_device = _ensure_local_device(index_owner_device);
    }
    if (!tile_device) {
        tile_device = _ensure_local_device(previous_tile_device);
    }
    if (!tile_device) {
        tile_device = _ensure_local_device(rendering_device);
    }
    if (!tile_device) {
        tile_device = p_renderer->get_submission_device();
    }
    if (!tile_device) {
        tile_device = _acquire_manager_local_device();
    }
    if (!tile_device) {
        ERR_PRINT_ONCE("[Painterly] Unable to determine RenderingDevice for tile renderer");
        return ERR_UNCONFIGURED;
    }

    if (index_owner_device && index_owner_device != tile_device) {
        WARN_PRINT_ONCE("[Painterly] Gaussian and index buffers originate from different RenderingDevices; using gaussian buffer owner");
        index_owner_device = tile_device;
    }

    bool tile_device_changed = previous_tile_device && previous_tile_device != tile_device;
    if (tile_device_changed) {
        if (tile_renderer_state.renderer.is_valid()) {
            p_renderer->forget_tile_renderer_outputs();
            tile_renderer_state.renderer->cleanup();
            tile_renderer_state.renderer.unref();
        }
        tile_renderer_state.init_failed = false;
    }

    if (!tile_renderer_state.renderer.is_valid() && !tile_renderer_state.init_failed) {
        tile_renderer_state.renderer.instantiate();
        tile_renderer_state.renderer->set_performance_monitor(&tile_renderer_state.gpu_performance_monitor);
        p_renderer->synchronize_tile_submission(tile_device, "TileRenderer initialization");
        Error init_err = tile_device ? tile_renderer_state.renderer->initialize(tile_device, Vector2i(width, height), TileRenderer::DEFAULT_TILE_SIZE, target_output_format, tile_device)
                                     : ERR_UNCONFIGURED;
        if (init_err != OK) {
            GS_LOG_ERROR_DEFAULT(vformat("[Painterly] Failed to initialize tile renderer: %d (will not retry)", init_err));
            p_renderer->forget_tile_renderer_outputs();
            tile_renderer_state.renderer->cleanup();
            tile_renderer_state.renderer.unref();
            tile_renderer_state.init_failed = true;
            return init_err;
        }

        // Wrap tile_renderer_state.renderer with interface adapter (Phase 8 migration)
        if (!subsystem_state.rasterizer.is_valid()) {
            subsystem_state.rasterizer.instantiate();
        }
        subsystem_state.rasterizer->set_device_manager(subsystem_state.device_manager);
        subsystem_state.rasterizer->set_tile_renderer(tile_renderer_state.renderer);

        // Initialize InteractiveStateManager when RenderingDevice is available (Phase 8)
        if (subsystem_state.interactive_state_manager.is_valid() && !subsystem_state.interactive_state_manager->is_initialized()) {
            Error state_err = subsystem_state.interactive_state_manager->initialize(tile_device);
            if (state_err != OK) {
                GS_LOG_WARN_DEFAULT(vformat("[Painterly] Failed to initialize interactive state manager: %d", state_err));
            }
        }
    }

    if (tile_renderer_state.init_failed) {
        return ERR_UNCONFIGURED;
    }

    // Use subsystem_state.rasterizer interface for output format (Phase 8 - remove direct tile_renderer_state.renderer access)
    if (subsystem_state.rasterizer.is_valid()) {
        subsystem_state.rasterizer->set_output_format(target_output_format);
    } else if (tile_renderer_state.renderer.is_valid()) {
        tile_renderer_state.renderer->set_output_format(target_output_format);
    }
    // Update interactive state GPU buffer before rendering (Phase 8)
    if (subsystem_state.interactive_state_manager.is_valid() && subsystem_state.interactive_state_manager->is_initialized()) {
        subsystem_state.interactive_state_manager->update_gpu_state();
    }
    render_params.interactive_state_uniform = RID();
    if (subsystem_state.interactive_state_manager.is_valid()) {
        render_params.interactive_state_uniform =
                subsystem_state.interactive_state_manager->ensure_state_uniform_buffer(p_renderer, tile_device);
    }
    render_params.frame_serial = frame_state.frame_counter;
    if (subsystem_state.rasterizer.is_valid()) {
        if (tile_renderer_state.renderer.is_valid()) {
            tile_renderer_state.renderer->set_performance_monitor(&tile_renderer_state.gpu_performance_monitor);
        }
        subsystem_state.rasterizer->set_frame_serial(frame_state.frame_counter);
    }
    if (tile_renderer_state.renderer.is_valid() && subsystem_state.rasterizer.is_valid()) {
        TileRenderer::ResolveDebugMode resolve_mode = TileRenderer::RESOLVE_DEBUG_NONE;
        if (debug_state.show_resolve_input) {
            resolve_mode = TileRenderer::RESOLVE_DEBUG_INPUT;
        } else if (debug_state.show_resolve_output) {
            resolve_mode = TileRenderer::RESOLVE_DEBUG_OUTPUT;
        }
        subsystem_state.rasterizer->set_resolve_debug_mode(static_cast<int>(resolve_mode));
    }

    // Safety check: ensure subsystem_state.rasterizer and tile_renderer_state.renderer are valid
    if (!subsystem_state.rasterizer.is_valid()) {
        GS_LOG_ERROR_DEFAULT("[Painterly] rasterizer is null - initialization failed");
        return ERR_UNCONFIGURED;
    }

    // If subsystem_state.rasterizer's tile_renderer_state.renderer might be stale, update it
    if (tile_renderer_state.renderer.is_valid() && subsystem_state.rasterizer->get_tile_renderer() != tile_renderer_state.renderer) {
        subsystem_state.rasterizer->set_tile_renderer(tile_renderer_state.renderer);
    }

    // Render through TileRasterizer interface (Phase 8 migration)
    RasterResult raster_result = subsystem_state.rasterizer->render_direct(tile_device, render_params);
    if (!raster_result.success || !raster_result.output_texture.is_valid()) {
        GS_LOG_ERROR_DEFAULT("[Painterly] Tile renderer failed to produce output texture");
        return FAILED;
    }
    RID rendered_texture = raster_result.output_texture;

    RenderingDevice *render_texture_owner = raster_result.output_owner;
    if (render_texture_owner) {
        p_renderer->track_resource_owner(rendered_texture, render_texture_owner, false, "tile_renderer_output");
    } else if (tile_device) {
        p_renderer->track_resource_owner(rendered_texture, tile_device, false, "tile_renderer_output");
    }

    Vector3i copy_region(width, height, 1);
    RenderingDevice *render_texture_device = p_renderer->get_resource_owner(rendered_texture, tile_device);
    RenderingDevice *color_device = p_renderer->get_resource_owner(color_texture, tile_device);
    RenderingDevice *copy_device = render_texture_device ? render_texture_device : color_device;
    if (render_texture_device && color_device && render_texture_device != color_device) {
        GS_LOG_ERROR_DEFAULT("[Painterly] Tile renderer color copy attempted between mismatched devices");
        return ERR_UNCONFIGURED;
    }
    if (!copy_device) {
        copy_device = tile_device;
    }

    Error copy_err = copy_device ? copy_device->texture_copy(rendered_texture, color_texture,
                                    Vector3i(), Vector3i(), copy_region, 0, 0, 0, 0)
                                 : ERR_UNCONFIGURED;
    if (copy_err != OK) {
        GS_LOG_ERROR_DEFAULT(vformat("[Painterly] Failed to copy tile renderer output into painterly color buffer: %d", copy_err));
        return copy_err;
    }

    if (depth_texture.is_valid() && subsystem_state.rasterizer->has_depth_output()) {
        bool tile_depth_copy_supported = subsystem_state.rasterizer->is_depth_copy_compatible();

        if (!tile_depth_copy_supported) {
            const PainterlyPassGraph::TextureInfo &depth_info =
                    pass_graph->get_texture_info(PainterlyPassGraph::TEXTURE_DEPTH);
            tile_depth_copy_supported = depth_info.valid && depth_info.format.format == RD::DATA_FORMAT_R32_SFLOAT;
        }

        if (tile_depth_copy_supported) {
            RID tile_depth = subsystem_state.rasterizer->get_depth_texture();
            if (tile_depth.is_valid()) {
                RenderingDevice *tile_depth_device = p_renderer->get_resource_owner(tile_depth, tile_device);
                RenderingDevice *depth_device = p_renderer->get_resource_owner(depth_texture, tile_device);
                RenderingDevice *depth_copy_device = tile_depth_device ? tile_depth_device : depth_device;
                if (tile_depth_device && depth_device && tile_depth_device != depth_device) {
                    GS_LOG_WARN_DEFAULT("[Painterly] Tile renderer depth copy attempted between mismatched devices");
                }
                Error depth_err = depth_copy_device ? depth_copy_device->texture_copy(tile_depth, depth_texture,
                        Vector3i(), Vector3i(), copy_region, 0, 0, 0, 0)
                                                    : ERR_UNCONFIGURED;
                if (depth_err != OK) {
                    GS_LOG_WARN_DEFAULT(vformat("[Painterly] Failed to copy tile renderer depth buffer: %d", depth_err));
                }
            }
        } else {
            p_renderer->warn_tile_depth_copy_incompatible();
        }
    }

    debug_state.tile_density_cache.clear();
    debug_state.tile_density_width = 0;
    debug_state.tile_density_height = 0;
    debug_state.tile_density_peak = 0;
    debug_state.tile_density_average = 0.0f;
    debug_state.last_tile_assignment_ms = 0.0f;
    debug_state.last_tile_rasterization_ms = 0.0f;
    debug_state.overlay_dirty = false;
    debug_state.hud_dirty = false;
    debug_state.last_stage_metrics = {};
    debug_state.last_stage_metrics_valid = false;

    if (tile_renderer_state.renderer.is_valid()) {
        float utilization = tile_renderer_state.gpu_performance_monitor.get_gpu_utilization_async();
        performance_state.metrics.gpu_utilization = utilization * 100.0f;
    }

    if (subsystem_state.rasterizer.is_valid() && subsystem_state.gpu_culler->get_state().overflow_autotune_enabled) {
        RasterOverflowStats overflow_stats = subsystem_state.rasterizer->get_overflow_stats();
        RasterStats raster_stats = subsystem_state.rasterizer->get_render_stats();
        subsystem_state.gpu_culler->apply_overflow_feedback(overflow_stats, render_params.splat_count,
                subsystem_state.rasterizer->get_tile_count(), static_cast<IOverflowAutoTuner *>(subsystem_state.overflow_auto_tuner.ptr()),
                raster_stats.average_splats_per_tile);
    }

    p_renderer->update_gpu_pass_metrics_from_tile_renderer();

    return OK;
}
