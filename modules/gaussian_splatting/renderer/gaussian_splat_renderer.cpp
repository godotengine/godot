#include "gaussian_splat_renderer.h"
#include "../core/gs_project_settings.h"
#include "servers/rendering/rendering_server_default.h"
#include "gpu_debug_utils.h"
#include "render_pipeline_stages.h"
#include "render_device_orchestrator.h"
#include "render_debug_state_orchestrator.h"
#include "render_diagnostics_orchestrator.h"
#include "render_sorting_orchestrator.h"
#include "render_streaming_orchestrator.h"
// RenderCullingOrchestrator removed (ISSUE-016) — merged into RenderQualityOrchestrator.
#include "render_quality_orchestrator.h"
#include "render_config_orchestrator.h"
#include "render_instancing_orchestrator.h"
#include "render_resource_orchestrator.h"
#include "render_data_orchestrator.h"
#include "render_output_orchestrator.h"
#include "pipeline_io_contracts.h"
#include "../logger/gs_debug_trace.h"
#include "core/object/callable_method_pointer.h"

using GaussianSplatting::ScopedGpuMarker;
#include "core/error/error_macros.h"
#include "core/config/project_settings.h"
#include "core/object/object.h"
#include "core/math/math_defs.h"
#include "core/math/math_funcs.h"
#include "core/os/os.h"
#include "core/variant/variant.h"
#include "core/templates/hash_set.h"
#include "core/math/random_pcg.h"
#include "core/templates/hashfuncs.h"
#include "core/typedefs.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include "core/math/vector3i.h"
// Note: scene/ includes not available in modules - using fallback rendering
// #include "scene/resources/primitive_meshes.h"
#include "servers/rendering/rendering_server_globals.h"
#include <limits>


#include "servers/rendering_server.h"
#include "servers/rendering/rendering_device.h"
#include "servers/rendering/renderer_rd/framebuffer_cache_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/render_data_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/texture_storage.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"
#include "servers/rendering/renderer_scene_cull.h"
#include "../core/gaussian_splat_manager.h"
#include "../core/gaussian_splat_scene_director.h"
#include "../core/performance_monitors.h"
#include "../interfaces/debug_overlay_system.h"
#include "../interfaces/interactive_state_manager.h"
#include "../interfaces/tile_rasterizer.h"
#include "../interfaces/gpu_culler.h"
#include "../interfaces/output_compositor.h"
#include "../interfaces/gpu_sorting_pipeline.h"
#include "../interfaces/overflow_auto_tuner.h"
#include "../interfaces/painterly_material_manager.h"
#include "gpu_buffer_raii.h"
#include "gpu_memory_stream.h"
#include "gpu_sorter.h"
#include "gaussian_gpu_layout.h"
#include "rendering_diagnostics.h"
#include "shader_compilation_helper.h"
#include "sorting_config.h"
#include "gpu_sorting_config.h"
#include "sorting_contract.h"
#include "../logger/gs_logger.h"
#include "../shaders/gaussian_splat.glsl.gen.h"
#include "../shaders/gs_shadow_blit.glsl.gen.h"
#include <cstdint>
#include <algorithm>
#include <memory>
#include <type_traits>
#include <initializer_list>

#include "core/object/callable_method_pointer.h"

#ifndef kLogFrameDebug
#define kLogFrameDebug 0
#endif

namespace {

constexpr uint32_t FRUSTUM_PLANE_COUNT = 6;
static constexpr real_t SORT_TRANSFORM_TOLERANCE = 1e-4f;

// Project settings helpers provided by gs_project_settings.h (gs::settings namespace).
static bool _get_bool_setting(ProjectSettings *ps, const StringName &name, bool fallback) {
    return gs::settings::get_bool(ps, name, fallback);
}

static int _get_int_setting(ProjectSettings *ps, const StringName &name, int fallback) {
    return static_cast<int>(gs::settings::get_uint(ps, name, static_cast<uint32_t>(fallback)));
}

// Convert SH band level (0-3) to coefficient count limit
static uint8_t _sh_band_to_coeff_limit(int p_band) {
    // SH bands: 0 -> 1 coeff (DC only), 1 -> 4, 2 -> 9, 3 -> 16
    int clamped = CLAMP(p_band, 0, 3);
    return static_cast<uint8_t>((clamped + 1) * (clamped + 1));
}

static void _initialize_lighting_project_settings_defaults() {
    ProjectSettings *ps = ProjectSettings::get_singleton();
    if (!ps) {
        return;
    }

    static const StringName direct_light_scale_path("rendering/gaussian_splatting/lighting/direct_light_scale");
    static const StringName indirect_sh_scale_path("rendering/gaussian_splatting/lighting/indirect_sh_scale");
    static const StringName shadow_strength_path("rendering/gaussian_splatting/lighting/shadow_strength");
    static const StringName sh_dc_logit_path("rendering/gaussian_splatting/lighting/dc_logit");
    static const StringName shadow_bias_scale_path("rendering/gaussian_splatting/lighting/shadow_receiver_bias_scale");
    static const StringName shadow_bias_min_path("rendering/gaussian_splatting/lighting/shadow_receiver_bias_min");
    static const StringName shadow_bias_max_path("rendering/gaussian_splatting/lighting/shadow_receiver_bias_max");

    if (!ps->has_setting(direct_light_scale_path)) {
        ps->set_setting(direct_light_scale_path, 0.5f); // GS_CI_ALLOW_RENDER_PATH_SETTING_MUTATION
    }
    ps->set_initial_value(direct_light_scale_path, 0.5f);

    if (!ps->has_setting(indirect_sh_scale_path)) {
        ps->set_setting(indirect_sh_scale_path, 1.0f); // GS_CI_ALLOW_RENDER_PATH_SETTING_MUTATION
    }
    ps->set_initial_value(indirect_sh_scale_path, 1.0f);

    if (!ps->has_setting(shadow_strength_path)) {
        ps->set_setting(shadow_strength_path, 1.0f); // GS_CI_ALLOW_RENDER_PATH_SETTING_MUTATION
    }
    ps->set_initial_value(shadow_strength_path, 1.0f);

    if (!ps->has_setting(sh_dc_logit_path)) {
        ps->set_setting(sh_dc_logit_path, false); // GS_CI_ALLOW_RENDER_PATH_SETTING_MUTATION
    }
    ps->set_initial_value(sh_dc_logit_path, false);

    if (!ps->has_setting(shadow_bias_scale_path)) {
        ps->set_setting(shadow_bias_scale_path, 0.2f); // GS_CI_ALLOW_RENDER_PATH_SETTING_MUTATION
    }
    ps->set_initial_value(shadow_bias_scale_path, 0.2f);

    if (!ps->has_setting(shadow_bias_min_path)) {
        ps->set_setting(shadow_bias_min_path, 0.0f); // GS_CI_ALLOW_RENDER_PATH_SETTING_MUTATION
    }
    ps->set_initial_value(shadow_bias_min_path, 0.0f);

    if (!ps->has_setting(shadow_bias_max_path)) {
        ps->set_setting(shadow_bias_max_path, 0.0f); // GS_CI_ALLOW_RENDER_PATH_SETTING_MUTATION
    }
    ps->set_initial_value(shadow_bias_max_path, 0.0f);
}

class FrameLogSettingsRegistry {
public:
    static FrameLogSettingsRegistry &get_singleton() {
        static FrameLogSettingsRegistry instance;
        return instance;
    }

    void ensure_initialized() {
        if (initialized) {
            return;
        }
        ProjectSettings *ps = ProjectSettings::get_singleton();
        if (!ps) {
            return;
        }
        Callable callback = callable_mp_static(&FrameLogSettingsRegistry::_on_project_settings_changed);
        if (!ps->is_connected("settings_changed", callback)) {
            ps->connect("settings_changed", callback);
        }
        initialized = true;
        refresh();
    }

    bool is_frame_log_enabled() {
        ensure_initialized();
        return enable_all_debug || enable_frame_logging;
    }

    int get_frame_log_frequency() {
        ensure_initialized();
        return frame_log_frequency;
    }

private:
    static void _on_project_settings_changed() {
        get_singleton().refresh();
    }

    void refresh() {
        enable_all_debug = false;
        enable_frame_logging = false;
        frame_log_frequency = 300;
        if (ProjectSettings *ps = ProjectSettings::get_singleton()) {
            enable_all_debug = gs::settings::get_bool(ps,
                    "rendering/gaussian_splatting/debug/enable_all_debug", false);
            enable_frame_logging = gs::settings::get_bool(ps,
                    "rendering/gaussian_splatting/debug/enable_frame_logging", false);
            frame_log_frequency = static_cast<int>(gs::settings::get_uint(ps,
                    "rendering/gaussian_splatting/debug/frame_log_frequency",
                    static_cast<uint32_t>(frame_log_frequency)));
        }
        if (enable_all_debug && frame_log_frequency <= 0) {
            frame_log_frequency = 1;
        }
    }

    bool initialized = false;
    bool enable_all_debug = false;
    bool enable_frame_logging = false;
    int frame_log_frequency = 300;
};

static bool _is_frame_log_enabled() {
#ifdef GS_SILENCE_LOGS
    // Compile-time silencing for performance testing
    return false;
#elif kLogFrameDebug
    // Compile-time override for active development
    return true;
#elif defined(DEBUG_ENABLED)
    // PROD-1 (#626): Respect project setting even in debug builds
    // Default is false - set rendering/gaussian_splatting/debug/enable_frame_logging to true if needed
    return FrameLogSettingsRegistry::get_singleton().is_frame_log_enabled();
#else
    return false;
#endif
}

static int _get_frame_log_frequency() {
    return FrameLogSettingsRegistry::get_singleton().get_frame_log_frequency();
}

static bool _should_log_frame(uint64_t p_frame) {
    if (!_is_frame_log_enabled()) {
        return false;
    }
    const int freq = _get_frame_log_frequency();
    if (freq <= 0) {
        return false;
    }
    return p_frame == 0 || (p_frame % static_cast<uint64_t>(freq) == 0);
}

static void _trace_render_path(bool p_enabled, uint64_t p_frame, bool p_streaming_enabled, bool p_streaming_ready) {
    if (!p_enabled) {
        return;
    }
    const char *mode = (p_streaming_enabled && p_streaming_ready) ? "streaming" : "resident";
    GS_LOG_RENDERER_DEBUG(vformat("[RENDER-PATH] frame=%s mode=%s streaming_enabled=%s streaming_ready=%s",
            String::num_uint64(p_frame),
            mode, p_streaming_enabled ? "yes" : "no", p_streaming_ready ? "yes" : "no"));
}

static String _streaming_not_ready_route_uid(const char *p_state_token) {
    return String("COMMON.SKIP.STREAMING_NOT_READY.") + String(p_state_token ? p_state_token : "UNKNOWN");
}

static bool _is_typed_streaming_not_ready_route(const String &p_route_uid) {
    return p_route_uid.begins_with("COMMON.SKIP.STREAMING_NOT_READY.");
}

static bool _projection_nearly_equal(const Projection &p_a, const Projection &p_b, real_t p_tolerance = 1e-6f) {
	for (int col = 0; col < 4; col++) {
		for (int row = 0; row < 4; row++) {
			const real_t diff = Math::abs(p_a.columns[col][row] - p_b.columns[col][row]);
			if (diff > p_tolerance) {
				return false;
			}
		}
	}
	return true;
}

} // namespace

Error GaussianSplatRenderer::get_active_data_source(const SceneState &p_scene_state,
        const StreamingState &p_streaming_state,
        const SortingState &p_sorting_state,
        const ResourceState &p_resource_state,
        const SubsystemState &p_subsystem_state,
        SplatDataSource &r_source,
        String &r_error) {
    r_source = SplatDataSource();
    r_error = String();

    (void)p_scene_state;
    (void)p_resource_state;

    if (!p_streaming_state.current_streaming_system.is_valid()) {
        r_error = "Streaming system unavailable";
        return ERR_UNCONFIGURED;
    }

    const GlobalAtlasState &atlas_state = p_streaming_state.current_streaming_system->get_global_atlas_state();
    if (!atlas_state.atlas_gaussian_buffer.is_valid()) {
        r_error = "Instance atlas buffer unavailable";
        return ERR_UNAVAILABLE;
    }
    if (!p_subsystem_state.sorting_pipeline.is_valid()) {
        r_error = "Sorting pipeline unavailable";
        return ERR_UNAVAILABLE;
    }

    RID sort_indices = p_subsystem_state.sorting_pipeline->get_sort_indices_buffer();
    if (!sort_indices.is_valid()) {
        r_error = "Sorted indices buffer unavailable";
        return ERR_UNAVAILABLE;
    }

    r_source.source_name = SplatDataSource::kSourceStreaming;
    r_source.gaussian_buffer = atlas_state.atlas_gaussian_buffer;
    r_source.sorted_indices = sort_indices;
    r_source.splat_count = p_sorting_state.sorted_splat_count;
    if (r_source.splat_count == 0) {
        r_source.splat_count = p_streaming_state.current_streaming_system->get_visible_count();
    }
    r_source.total_gaussians = atlas_state.atlas_gaussian_count;
    if (r_source.total_gaussians == 0 && r_source.splat_count > 0) {
        r_source.total_gaussians = r_source.splat_count;
    }

    return OK;
}

GaussianSplatRenderer::DataSourcePlan GaussianSplatRenderer::build_data_source_plan(const SceneState &p_scene_state,
        const StreamingState &p_streaming_state,
        const SortingState &p_sorting_state,
        const ResourceState &p_resource_state,
        const SubsystemState &p_subsystem_state) {
    // Delegate to RenderPipelineStages (T4-PR3)
    return RenderPipelineStages::build_data_source_plan(p_scene_state, p_streaming_state, p_sorting_state,
            p_resource_state, p_subsystem_state);
}

void GaussianSplatRenderer::apply_data_source_plan(const DataSourcePlan &p_plan, PerformanceMetrics &p_metrics,
        const ResourceState &p_resource_state) {
    // Delegate to RenderPipelineStages (T4-PR3)
    RenderPipelineStages::apply_data_source_plan(p_plan, p_metrics, p_resource_state);
}

GaussianSplatRenderer::RenderFramePlan GaussianSplatRenderer::build_frame_plan(const SceneState &p_scene_state,
        const StreamingState &p_streaming_state,
        const SortingState &p_sorting_state,
        const ResourceState &p_resource_state,
        const SubsystemState &p_subsystem_state,
        const PipelineFeatureSet *p_pipeline_features,
        bool p_has_render_data,
        const String &p_cull_skip_reason,
        const String &p_sort_skip_reason,
        RenderFallbackReason p_cull_skip_reason_code,
        RenderFallbackReason p_sort_skip_reason_code,
        bool p_set_skip_metrics,
        bool p_clear_cull_state_on_skip) {
    // Delegate to RenderPipelineStages (T4-PR3)
    return RenderPipelineStages::build_frame_plan(p_scene_state, p_streaming_state, p_sorting_state,
            p_resource_state, p_subsystem_state, p_pipeline_features, p_has_render_data, p_cull_skip_reason,
            p_sort_skip_reason, p_cull_skip_reason_code, p_sort_skip_reason_code, p_set_skip_metrics,
            p_clear_cull_state_on_skip);
}

Projection GaussianSplatRenderer::build_cull_projection(RenderDataRD *p_render_data, const Projection &p_projection) const {
	Projection cull_projection = p_projection;
	if (p_render_data && p_render_data->scene_data && p_render_data->scene_data->flip_y) {
		// Frustum plane extraction must use the same flip convention as cull/sort paths.
		cull_projection.columns[1][1] = -cull_projection.columns[1][1];
	}
	return cull_projection;
}

bool GaussianSplatRenderer::validate_cull_projection_contract(RenderDataRD *p_render_data, const Projection &p_projection,
		const Projection &p_cull_projection, const char *p_context) {
	const Projection expected = build_cull_projection(p_render_data, p_projection);
	if (_projection_nearly_equal(expected, p_cull_projection)) {
		return true;
	}

	get_performance_state().metrics.cull_projection_contract_mismatch_count++;
	const String context = p_context ? String(p_context) : String("unknown");
	WARN_PRINT_ONCE(vformat("[GaussianSplatRenderer] Cull projection contract mismatch in %s; expected shared flip_y-aware cull projection.",
			context));
	return false;
}

void GaussianSplatRenderer::_forget_tile_renderer_outputs() {
    if (subsystem_state.rasterizer.is_valid()) {
        subsystem_state.rasterizer->clear_output_resource_tracking();
    }
    if (subsystem_state.output_compositor.is_valid()) {
        subsystem_state.output_compositor->invalidate_cached_render();
    }
}

void GaussianSplatRenderer::_warn_tile_depth_copy_incompatible() {
    WARN_PRINT_ONCE("[GaussianSplatRenderer] Tile depth copy contract not satisfied; depth compositing is deterministically disabled for this frame");
}

GaussianSplatRenderer::FrameStateProvider::FrameStateProvider(GaussianSplatRenderer *p_renderer,
        const RenderFrameContext::FrameDeps *p_deps) :
        renderer(p_renderer),
        deps(p_deps) {
}

OutputCompositor *GaussianSplatRenderer::FrameStateProvider::get_output_compositor() const {
    ERR_FAIL_NULL_V(renderer, nullptr);
    if (deps && deps->output_compositor) {
        return deps->output_compositor;
    }
    return renderer->get_subsystem_state().output_compositor.ptr();
}

GPUCuller *GaussianSplatRenderer::FrameStateProvider::get_gpu_culler() const {
    ERR_FAIL_NULL_V(renderer, nullptr);
    if (deps && deps->gpu_culler) {
        return deps->gpu_culler;
    }
    return renderer->get_subsystem_state().gpu_culler.ptr();
}

PainterlyRenderer *GaussianSplatRenderer::FrameStateProvider::get_painterly_renderer() const {
    ERR_FAIL_NULL_V(renderer, nullptr);
    if (deps && deps->painterly_renderer) {
        return deps->painterly_renderer;
    }
    return renderer->get_subsystem_state().painterly_renderer.ptr();
}

GPUSortingPipeline *GaussianSplatRenderer::FrameStateProvider::get_sorting_pipeline() const {
    ERR_FAIL_NULL_V(renderer, nullptr);
    if (deps && deps->sorting_pipeline) {
        return deps->sorting_pipeline;
    }
    return renderer->get_subsystem_state().sorting_pipeline.ptr();
}

RenderingDevice *GaussianSplatRenderer::FrameStateProvider::get_rendering_device() const {
    ERR_FAIL_NULL_V(renderer, nullptr);
    if (deps && deps->rendering_device) {
        return deps->rendering_device;
    }
    return renderer->get_device_state().rd;
}

const GaussianSplatRenderer::SceneState &GaussianSplatRenderer::FrameStateProvider::get_scene_state() const {
    static SceneState fallback;
    ERR_FAIL_NULL_V(renderer, fallback);
    if (deps && deps->scene_state) {
        return *deps->scene_state;
    }
    return renderer->get_scene_state();
}

const GaussianSplatRenderer::StreamingState &GaussianSplatRenderer::FrameStateProvider::get_streaming_state() const {
    static StreamingState fallback;
    ERR_FAIL_NULL_V(renderer, fallback);
    if (deps && deps->streaming_state) {
        return *deps->streaming_state;
    }
    return renderer->get_streaming_state();
}

const GaussianSplatRenderer::SortingState &GaussianSplatRenderer::FrameStateProvider::get_sorting_state_view() const {
    static SortingState fallback;
    ERR_FAIL_NULL_V(renderer, fallback);
    if (deps && deps->sorting_state) {
        return *deps->sorting_state;
    }
    return renderer->get_sorting_state();
}

const GaussianSplatRenderer::RenderConfig &GaussianSplatRenderer::FrameStateProvider::get_render_config_view() const {
    static RenderConfig fallback;
    ERR_FAIL_NULL_V(renderer, fallback);
    if (deps && deps->render_config) {
        return *deps->render_config;
    }
    return renderer->get_render_config();
}

const GaussianSplatRenderer::JacobianDebugConfig &GaussianSplatRenderer::FrameStateProvider::get_jacobian_debug_view() const {
    static JacobianDebugConfig fallback;
    ERR_FAIL_NULL_V(renderer, fallback);
    if (deps && deps->jacobian_debug) {
        return *deps->jacobian_debug;
    }
    return renderer->get_jacobian_debug();
}

const GaussianSplatRenderer::ResourceState &GaussianSplatRenderer::FrameStateProvider::get_resource_state_view() const {
    static ResourceState fallback;
    ERR_FAIL_NULL_V(renderer, fallback);
    if (deps && deps->resource_state) {
        return *deps->resource_state;
    }
    return renderer->get_resource_state();
}

const GaussianSplatRenderer::FrameState &GaussianSplatRenderer::FrameStateProvider::get_frame_state_view() const {
    static FrameState fallback;
    ERR_FAIL_NULL_V(renderer, fallback);
    if (deps && deps->frame_state) {
        return *deps->frame_state;
    }
    return renderer->get_frame_state();
}

const GaussianSplatRenderer::PerformanceState &GaussianSplatRenderer::FrameStateProvider::get_performance_state_view() const {
    static PerformanceState fallback;
    ERR_FAIL_NULL_V(renderer, fallback);
    if (deps && deps->performance_state) {
        return *deps->performance_state;
    }
    return renderer->get_performance_state();
}

const GaussianSplatRenderer::SubsystemState &GaussianSplatRenderer::FrameStateProvider::get_subsystem_state_view() const {
    static SubsystemState fallback;
    ERR_FAIL_NULL_V(renderer, fallback);
    if (deps && deps->subsystem_state) {
        return *deps->subsystem_state;
    }
    return renderer->get_subsystem_state();
}

GaussianSplatRenderer::SortingState &GaussianSplatRenderer::FrameStateProvider::get_sorting_state_mut() {
    static SortingState fallback;
    ERR_FAIL_NULL_V(renderer, fallback);
    if (deps && deps->sorting_state) {
        return *deps->sorting_state;
    }
    return renderer->get_sorting_state();
}

GaussianSplatRenderer::RenderConfig &GaussianSplatRenderer::FrameStateProvider::get_render_config_mut() {
    static RenderConfig fallback;
    ERR_FAIL_NULL_V(renderer, fallback);
    if (deps && deps->render_config) {
        return *deps->render_config;
    }
    return renderer->get_render_config();
}

GaussianSplatRenderer::ResourceState &GaussianSplatRenderer::FrameStateProvider::get_resource_state_mut() {
    static ResourceState fallback;
    ERR_FAIL_NULL_V(renderer, fallback);
    if (deps && deps->resource_state) {
        return *deps->resource_state;
    }
    return renderer->get_resource_state();
}

GaussianSplatRenderer::FrameState &GaussianSplatRenderer::FrameStateProvider::get_frame_state_mut() {
    static FrameState fallback;
    ERR_FAIL_NULL_V(renderer, fallback);
    if (deps && deps->frame_state) {
        return *deps->frame_state;
    }
    return renderer->get_frame_state();
}

GaussianSplatRenderer::PerformanceState &GaussianSplatRenderer::FrameStateProvider::get_performance_state_mut() {
    static PerformanceState fallback;
    ERR_FAIL_NULL_V(renderer, fallback);
    if (deps && deps->performance_state) {
        return *deps->performance_state;
    }
    return renderer->get_performance_state();
}

GaussianSplatRenderer::SubsystemState &GaussianSplatRenderer::FrameStateProvider::get_subsystem_state_mut() {
    static SubsystemState fallback;
    ERR_FAIL_NULL_V(renderer, fallback);
    if (deps && deps->subsystem_state) {
        return *deps->subsystem_state;
    }
    return renderer->get_subsystem_state();
}

const PipelineFeatureSet *GaussianSplatRenderer::FrameStateProvider::get_pipeline_features() const {
    ERR_FAIL_NULL_V(renderer, nullptr);
    if (deps && deps->pipeline_features) {
        return deps->pipeline_features;
    }
    return &renderer->pipeline_features_effective;
}

const GaussianSplatRenderer::RenderFramePlan *GaussianSplatRenderer::FrameStateProvider::get_frame_plan() const {
    return deps ? deps->frame_plan : nullptr;
}

bool GaussianSplatRenderer::_dispatch_call_on_render_thread_blocking(
        const Callable &p_callable, bool *r_dispatched, bool p_allow_timeout, uint64_t *r_request_id) {
    ERR_FAIL_NULL_V(render_thread_dispatcher, false);
    return render_thread_dispatcher->dispatch_call_on_render_thread_blocking(
            p_callable, r_dispatched, p_allow_timeout, r_request_id, "[GaussianSplatRenderer] Render-thread dispatch");
}

void GaussianSplatRenderer::_notify_render_thread_dispatch_completed(uint64_t p_request_id) {
    ERR_FAIL_NULL(render_thread_dispatcher);
    render_thread_dispatcher->notify_completed(p_request_id);
}

GaussianSplatRenderer::GaussianSplatRenderer(RenderingDevice *p_device) {
    // State buckets: frame_context_manager (view + frame metrics), device state (orchestrator-owned),
    // resource orchestrator state, streaming state (data orchestrator), debug_state_orchestrator, diagnostics_orchestrator.
    RenderingDevice *device = p_device;
    if (!device) {
        if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
            device = manager->get_primary_rendering_device();
        }
    }

    _initialize_lighting_project_settings_defaults();
    render_thread_dispatcher = std::make_unique<RenderThreadDispatcher>();

    // Initialize modular interface subsystems (Phase 8 migration)
    subsystem_state.device_manager.instantiate();
    subsystem_state.device_manager->initialize(device);  // Pass current device (may be null, will acquire later)
    subsystem_state.debug_overlay_system.instantiate();
    subsystem_state.debug_overlay_system->initialize();
    subsystem_state.interactive_state_manager.instantiate();
    // Note: subsystem_state.interactive_state_manager->initialize() requires RenderingDevice,
    // will be called when device becomes available
    subsystem_state.gpu_culler.instantiate();
    // Respect project setting for LOD - don't hardcode to true
    bool lod_enabled_setting = gs::settings::get_bool(ProjectSettings::get_singleton(),
            "rendering/gaussian_splatting/lod/enabled", true);
    subsystem_state.gpu_culler->get_config().lod_enabled = lod_enabled_setting;
    subsystem_state.gpu_culler->get_config().lod_bias = 1.0f;
    subsystem_state.gpu_culler->get_config().frustum_culling = true;
    subsystem_state.gpu_culler->get_config().temporal_coherence = true;
    // Note: subsystem_state.gpu_culler->initialize() requires RenderingDevice,
    // will be called when device becomes available
    subsystem_state.output_compositor.instantiate();
    subsystem_state.output_compositor->set_device_manager(subsystem_state.device_manager);
    // Phase 15: Wire up callbacks for texture format and owner lookup
    subsystem_state.output_compositor->set_texture_format_callback([this](RenderingDevice *p_device, RID p_texture) -> RD::TextureFormat {
        return _get_texture_format(p_device, p_texture);
    });
    // Note: subsystem_state.output_compositor->initialize() requires RenderingDevice,
    // will be called when device becomes available
    subsystem_state.sorting_pipeline.instantiate();
    subsystem_state.sorting_pipeline->set_device_manager(subsystem_state.device_manager);
    subsystem_state.sorting_pipeline->set_manage_buffers(true);
    // Note: subsystem_state.sorting_pipeline->initialize() requires RenderingDevice,
    // will be called when device becomes available
    subsystem_state.overflow_auto_tuner.instantiate();
    subsystem_state.overflow_auto_tuner->set_baselines(subsystem_state.gpu_culler->get_config().importance_cull_baseline, subsystem_state.gpu_culler->get_state().tiny_splat_screen_radius_baseline);
    subsystem_state.painterly_renderer.instantiate();
    subsystem_state.painterly_material_manager.instantiate();
    // Note: subsystem_state.painterly_material_manager->initialize() requires RenderingDevice,
    // will be called when device becomes available

    // Don't instantiate painterly material here to avoid "default value" warning
    // It will be lazy-initialized when first accessed via set_painterly_material()

    // Test splat buffers remain empty until explicitly populated for debugging.
    test_data_state.positions.clear();
    test_data_state.colors.clear();
    test_data_state.scales.clear();
    subsystem_state.gpu_culler->get_state().sort_cache_angle_cos_threshold = Math::cos(Math::deg_to_rad(5.0f));
    float position_threshold = 0.05f;
    subsystem_state.gpu_culler->get_state().sort_cache_position_threshold_sq = position_threshold * position_threshold;

    // Initialize orchestrators.
    pipeline_stages = std::make_unique<RenderPipelineStages>(this);
    debug_state_orchestrator = std::make_unique<RenderDebugStateOrchestrator>(
            this, &tile_renderer_state.renderer, &subsystem_state.debug_overlay_system, &jacobian_debug);
    diagnostics_orchestrator = std::make_unique<RenderDiagnosticsOrchestrator>(
            this,
            debug_state_orchestrator.get(),
            [this]() { return _build_device_capability_report(); });
    pipeline_stages->set_debug_state_orchestrator(debug_state_orchestrator.get());
    pipeline_stages->set_diagnostics_orchestrator(diagnostics_orchestrator.get());
    device_orchestrator = std::make_unique<RenderDeviceOrchestrator>(
            this, subsystem_state.device_manager.ptr(), subsystem_state.sorting_pipeline.ptr(),
            [this](const CrossDeviceOperation &p_operation) {
                if (diagnostics_orchestrator) {
                    diagnostics_orchestrator->record_cross_device_operation(p_operation);
                }
            },
            [this](const RenderingError &p_error) {
                if (diagnostics_orchestrator) {
                    diagnostics_orchestrator->record_rendering_error(p_error);
                }
            },
            [this]() {
                if (diagnostics_orchestrator) {
                    diagnostics_orchestrator->emit_runtime_diagnostics_if_requested();
                }
            });

    device_orchestrator->initialize_device_state(device);
    if (!get_device_state().rd) {
        GS_LOG_WARN_DEFAULT("[GaussianSplatRenderer] Local RenderingDevice unavailable; GPU operations will be deferred until device is available");
    }

    sorting_orchestrator = std::make_unique<RenderSortingOrchestrator>(
            this, subsystem_state.gpu_culler.ptr(), subsystem_state.sorting_pipeline.ptr(),
            [this](const Transform3D &p_world_to_camera_transform, const Projection &p_projection,
                    const Size2i &p_viewport_size) {
                return _cull_for_view(p_world_to_camera_transform, p_projection, p_viewport_size);
            },
            [this](const RenderingError &p_error) {
                if (diagnostics_orchestrator) {
                    diagnostics_orchestrator->record_rendering_error(p_error);
                }
            });
    quality_orchestrator = std::make_unique<RenderQualityOrchestrator>(this, subsystem_state.gpu_culler.ptr());
    config_orchestrator = std::make_unique<RenderConfigOrchestrator>(
            this, &subsystem_state.interactive_state_manager, &subsystem_state.painterly_renderer);
    instancing_orchestrator = std::make_unique<RenderInstancingOrchestrator>(
            this, subsystem_state.output_compositor.ptr(), pipeline_stages.get(),
            [this](RenderDataRD *p_render_data, const Transform3D &p_world_to_camera_transform,
                    const Projection &p_projection, const Projection &p_render_projection,
                    bool p_defer_render_buffers_commit, RenderFrameContext &r_context) {
                _prepare_render_frame_context(p_render_data, p_world_to_camera_transform, p_projection,
                        p_render_projection, p_defer_render_buffers_commit, r_context);
            },
            [this](RenderDataRD *p_render_data, const Transform3D &p_world_to_camera_transform,
                    const Projection &p_projection, const Projection &p_render_projection,
                    bool p_defer_render_buffers_commit) {
                render_sorted_splats(p_render_data, p_world_to_camera_transform, p_projection,
                        p_render_projection, p_defer_render_buffers_commit);
            });
    resource_orchestrator = std::make_unique<RenderResourceOrchestrator>(
            this, &get_device_state(), &pipeline_features_effective, &pipeline_features_warning_cache);
    data_orchestrator = std::make_unique<RenderDataOrchestrator>(
            this,
            [this]() { _release_shared_dynamic_asset(); },
            [this]() { return _acquire_rendering_device(); },
            [this](bool p_free_rids) { _invalidate_static_chunk_caches(p_free_rids); });
    streaming_orchestrator = std::make_unique<RenderStreamingOrchestrator>(
            RenderStreamingOrchestratorDependencies{this, data_orchestrator.get(), device_orchestrator.get()});
    RenderOutputOrchestrator::Dependencies output_dependencies;
    output_dependencies.renderer = this;
    output_dependencies.output_compositor = subsystem_state.output_compositor.ptr();
    output_dependencies.painterly_renderer = subsystem_state.painterly_renderer.ptr();
    output_dependencies.gpu_culler = subsystem_state.gpu_culler.ptr();
    output_dependencies.runtime_ports.create_gpu_resources = [this]() { _create_gpu_resources_safe(); };
    output_dependencies.runtime_ports.ensure_rendering_device = &GaussianSplatRenderer::ensure_rendering_device;
    output_dependencies.runtime_ports.get_texture_format = &GaussianSplatRenderer::get_texture_format;
    output_dependencies.runtime_ports.set_active_viewport_format = &GaussianSplatRenderer::set_active_viewport_format;
    output_dependencies.runtime_ports.set_manual_viewport_format = &GaussianSplatRenderer::set_manual_viewport_format;
    output_orchestrator = std::make_unique<RenderOutputOrchestrator>(output_dependencies);

    GaussianRenderingDiagnostics::ensure_singleton();
    if (GaussianRenderingDiagnostics::get_singleton()) {
        GaussianRenderingDiagnostics::get_singleton()->register_renderer(this);
    }

    // Register with performance monitors for streaming/VRAM telemetry
    GaussianSplattingPerformanceMonitors::create_singleton();
    if (GaussianSplattingPerformanceMonitors *monitors = GaussianSplattingPerformanceMonitors::get_singleton()) {
        monitors->register_splat_renderer(this);
    }
}

bool GaussianSplatRenderer::ensure_sort_rendering_device(const char *p_context) {
    return ensure_rendering_device(p_context);
}

RenderingDevice *GaussianSplatRenderer::get_sort_rendering_device() const {
    return get_device_state().rd;
}

SortExternalBufferState GaussianSplatRenderer::get_sort_external_buffer_state() const {
    SortExternalBufferState state;
    const ResourceState &resource_state = get_resource_state();
    if (!resource_state.buffer_manager.is_valid() || !resource_state.buffer_manager_initialized) {
        return state;
    }

    GPUBufferManager::BufferHandle manager_keys = resource_state.buffer_manager->get_sort_key_handle();
    GPUBufferManager::BufferHandle manager_indices = resource_state.buffer_manager->get_sorted_indices_handle();
    if (!manager_keys.is_valid() || !manager_indices.is_valid() || manager_keys.device != manager_indices.device) {
        return state;
    }

    state.keys_buffer = manager_keys.buffer;
    state.indices_buffer = manager_indices.buffer;
    state.device = manager_keys.device;
    state.capacity = resource_state.buffer_manager->get_buffer_capacity();
    state.valid = true;
    return state;
}

bool GaussianSplatRenderer::resize_sort_state_byte_vectors(uint32_t p_cpu_capacity, uint32_t p_key_stride_bytes, const char *p_context) {
    const uint64_t key_bytes_u64 = uint64_t(p_cpu_capacity) * uint64_t(p_key_stride_bytes);
    const uint64_t index_bytes_u64 = uint64_t(p_cpu_capacity) * uint64_t(sizeof(uint32_t));
    const uint64_t byte_limit = MIN(uint64_t(std::numeric_limits<uint32_t>::max()), uint64_t(std::numeric_limits<int>::max()));
    if (key_bytes_u64 > byte_limit || index_bytes_u64 > byte_limit) {
        GS_LOG_WARN_DEFAULT(vformat("[GaussianSplatRenderer] %s requested oversized renderer sort buffers (capacity=%s key_stride=%s key_bytes=%s index_bytes=%s limit=%s)",
                String(p_context),
                String::num_uint64(p_cpu_capacity),
                String::num_uint64(p_key_stride_bytes),
                String::num_uint64(key_bytes_u64),
                String::num_uint64(index_bytes_u64),
                String::num_uint64(byte_limit)));
        return false;
    }

    SortingState &sorting_state = get_sorting_state();
    const int key_bytes = int(key_bytes_u64);
    const int index_bytes = int(index_bytes_u64);
    if (sorting_state.sort_key_bytes.size() != key_bytes) {
        sorting_state.sort_key_bytes.resize(key_bytes);
    }
    if (sorting_state.sort_index_bytes.size() != index_bytes) {
        sorting_state.sort_index_bytes.resize(index_bytes);
    }
    return true;
}

void GaussianSplatRenderer::set_sort_buffer_binding_state(bool p_keys_external, bool p_indices_external,
        bool p_pipeline_managed, uint32_t p_capacity) {
    SortingState &sorting_state = get_sorting_state();
    sorting_state.sort_keys_external = p_keys_external;
    sorting_state.sort_indices_external = p_indices_external;
    sorting_state.sort_buffers_pipeline_managed = p_pipeline_managed;
    sorting_state.sort_buffer_capacity = p_capacity;
}

void GaussianSplatRenderer::clear_sort_buffer_binding_state() {
    SortingState &sorting_state = get_sorting_state();
    sorting_state.sort_buffer_capacity = 0;
    sorting_state.local_sort_buffer_capacity = 0;
    sorting_state.culled_position_capacity = 0;
    sorting_state.sort_key_bytes.clear();
    sorting_state.sort_index_bytes.clear();
    sorting_state.culled_position_bytes.clear();
    sorting_state.sort_keys_external = false;
    sorting_state.sort_indices_external = false;
    sorting_state.sort_buffers_pipeline_managed = false;
}

void GaussianSplatRenderer::publish_sorted_indices(const SortPublicationPayload &p_payload) {
    const uint32_t available_splats = static_cast<uint32_t>(p_payload.sorted_indices.size());
    if (!subsystem_state.gpu_culler.is_valid()) {
        return;
    }

    GPUCuller::CullingState &cull_state = subsystem_state.gpu_culler->get_state();
    if (static_cast<uint32_t>(cull_state.culled_indices.size()) != available_splats) {
        cull_state.culled_indices.resize(available_splats);
    }

    SortingState &sorting_state = get_sorting_state();
    sorting_state.sort_index_bytes.resize(int(available_splats * sizeof(uint32_t)));
    uint32_t *final_indices = reinterpret_cast<uint32_t *>(sorting_state.sort_index_bytes.ptrw());
    for (uint32_t i = 0; i < available_splats; i++) {
        const uint32_t index = p_payload.sorted_indices[i];
        cull_state.culled_indices[i] = index;
        final_indices[i] = index;
    }

    if (p_payload.sort_indices_buffer.is_valid() && !sorting_state.sort_index_bytes.is_empty()) {
        RenderingDevice *target_device = get_resource_owner(p_payload.sort_indices_buffer,
                p_payload.default_device ? p_payload.default_device : get_device_state().rd);
        if (!target_device) {
            target_device = p_payload.default_device ? p_payload.default_device : get_device_state().rd;
        }
        if (target_device) {
            target_device->buffer_update(p_payload.sort_indices_buffer, 0,
                    sorting_state.sort_index_bytes.size(), sorting_state.sort_index_bytes.ptr());
        }
    }

    sorting_state.sorted_splat_count = available_splats;
    get_frame_state().visible_splat_count.store(available_splats, std::memory_order_release);
    get_performance_state().metrics.rendered_splat_count =
            get_frame_state().visible_splat_count.load(std::memory_order_acquire);
}

GaussianSplatRenderer::~GaussianSplatRenderer() {
    if (GaussianRenderingDiagnostics::get_singleton()) {
        GaussianRenderingDiagnostics::get_singleton()->unregister_renderer(this);
    }

    // Unregister from performance monitors
    if (GaussianSplattingPerformanceMonitors *monitors = GaussianSplattingPerformanceMonitors::get_singleton()) {
        monitors->unregister_splat_renderer(this);
    }

    bool dispatch_submitted = false;
    if (_dispatch_call_on_render_thread_blocking(
                callable_mp(this, &GaussianSplatRenderer::_teardown_on_render_thread),
                &dispatch_submitted,
                false)) {
        return;
    }
    _teardown_resources();
}

void GaussianSplatRenderer::_teardown_resources() {
    bool expected = false;
    if (!teardown_resources_started.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        return;
    }

    _release_shared_dynamic_asset();
    get_resource_state().deletion_queue.flush_all();

    StreamingState &streaming_state_ref = get_streaming_state();

    if (subsystem_state.output_compositor.is_valid()) {
        subsystem_state.output_compositor->clear_cached_framebuffers();
        subsystem_state.output_compositor->clear_viewport_blit_resources();
    }
    if (shadow_output_compositor.is_valid()) {
        shadow_output_compositor->clear_cached_framebuffers();
        shadow_output_compositor->clear_viewport_blit_resources();
        shadow_output_compositor->shutdown();
        shadow_output_compositor.unref();
    }
    shadow_output_device_id = 0;
    if (get_device_state().rd) {
        shadow_blit_state.clear(get_device_state().rd);
    }
    // Phase 8: frustum cull resources now managed by GPUCuller interface

    if (streaming_state_ref.registered_gaussian_buffer.is_valid()) {
        GaussianSplatManager *manager = GaussianSplatManager::get_singleton();
        if (manager) {
            manager->unregister_gaussian_buffer(streaming_state_ref.registered_gaussian_buffer);
        }
        streaming_state_ref.registered_gaussian_buffer = RID();
        streaming_state_ref.registered_gaussian_data_id = ObjectID();
    }

    // Clean up tile renderer
    if (tile_renderer_state.renderer.is_valid()) {
        _forget_tile_renderer_outputs();
        tile_renderer_state.renderer->cleanup();
        tile_renderer_state.renderer.unref();
    }

    // Clean up mesh instances (disabled - scene classes not available in modules)
    test_data_state.mesh_instances.clear();

    // Clean up streaming system
    if (streaming_state_ref.current_streaming_system.is_valid()) {
        streaming_state_ref.current_streaming_system.unref();
    }
    if (streaming_state_ref.memory_stream.is_valid()) {
        streaming_state_ref.memory_stream->shutdown();
        streaming_state_ref.memory_stream.unref();
    }

    _invalidate_static_chunk_caches(true);

    // Clean up modular interface subsystems
    if (subsystem_state.gpu_culler.is_valid()) {
        subsystem_state.gpu_culler->shutdown();
        subsystem_state.gpu_culler.unref();
    }
    if (subsystem_state.interactive_state_manager.is_valid()) {
        subsystem_state.interactive_state_manager->shutdown();
        subsystem_state.interactive_state_manager.unref();
    }
    if (subsystem_state.debug_overlay_system.is_valid()) {
        subsystem_state.debug_overlay_system->shutdown();
        subsystem_state.debug_overlay_system.unref();
    }
    if (subsystem_state.sorting_pipeline.is_valid()) {
        subsystem_state.sorting_pipeline->set_sort_result_sink(this);
        subsystem_state.sorting_pipeline->set_sort_buffer_host_context(this);
        subsystem_state.sorting_pipeline->release_sort_buffers();
        subsystem_state.sorting_pipeline->shutdown();
        subsystem_state.sorting_pipeline.unref();
    }

    if (subsystem_state.painterly_renderer.is_valid()) {
        subsystem_state.painterly_renderer->free_painterly_resources(this);
        subsystem_state.painterly_renderer.unref();
    }
    // Clean up GPU buffer manager
    if (get_resource_state().buffer_manager.is_valid()) {
        get_resource_state().buffer_manager.unref();
    }

    if (get_sorting_state().gpu_sorter.is_valid()) {
        get_sorting_state().gpu_sorter->shutdown();
        get_sorting_state().gpu_sorter.unref();
    }

    // Clean up GPU resources
    if (get_device_state().rd) {
        ResourceState &resource_state = get_resource_state();
        if (get_pipeline_state().gaussian_shader_source && get_pipeline_state().gaussian_shader_version.is_valid()) {
            get_pipeline_state().gaussian_shader_source->version_free(get_pipeline_state().gaussian_shader_version);
            get_pipeline_state().gaussian_shader_version = RID();
        }
        get_pipeline_state().gaussian_shader = RID();
        get_pipeline_state().gaussian_shader_initialized = false;
        _free_owned_resource(get_device_state().rd, resource_state.instance_buffer);
        resource_state.instance_buffer_capacity = 0;
        _free_owned_resource(get_device_state().rd, resource_state.instance_visible_chunk_buffer);
        resource_state.instance_visible_chunk_capacity = 0;
        _free_owned_resource(get_device_state().rd, resource_state.instance_splat_ref_buffer);
        resource_state.instance_splat_ref_capacity = 0;
        _free_owned_resource(get_device_state().rd, resource_state.instance_counter_buffer);
        _free_owned_resource(get_device_state().rd, resource_state.instance_chunk_dispatch_buffer);
        _free_owned_resource(get_device_state().rd, resource_state.instance_indirect_count_buffer);
        _free_owned_resource(get_device_state().rd, resource_state.instance_count_buffer);
        _free_owned_resource(get_device_state().rd, test_data_state.vertex_buffer);
        _free_owned_resource(get_device_state().rd, test_data_state.position_buffer);
        _free_owned_resource(get_device_state().rd, test_data_state.scale_buffer);
        _free_owned_resource(get_device_state().rd, test_data_state.rotation_buffer);
        _free_owned_resource(get_device_state().rd, test_data_state.sh_buffer);
        // Phase 15: painterly_stroke_density_buffer now managed by PainterlyMaterialManager

    }

    if (get_pipeline_state().gaussian_shader_source) {
        memdelete(get_pipeline_state().gaussian_shader_source);
        get_pipeline_state().gaussian_shader_source = nullptr;
    }

    // Keep device manager alive until all owned RID frees have completed.
    if (subsystem_state.device_manager.is_valid()) {
        subsystem_state.device_manager->shutdown();
        subsystem_state.device_manager.unref();
    }
}

void GaussianSplatRenderer::_teardown_on_render_thread(uint64_t p_request_id) {
    _teardown_resources();
    _notify_render_thread_dispatch_completed(p_request_id);
}

void GaussianSplatRenderer::_release_shared_dynamic_asset() {
    StreamingState &streaming_state_ref = get_streaming_state();

    if (!streaming_state_ref.shared_dynamic_asset_handle.is_valid()) {
        return;
    }

    if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
        manager->release_dynamic_asset(streaming_state_ref.shared_dynamic_asset_handle);
    }

    streaming_state_ref.shared_dynamic_asset_handle = GaussianSplatManager::SharedDynamicAssetHandle();
}

// Phase 13: _bind_methods() moved to gaussian_splat_renderer_bindings.cpp

void GaussianSplatRenderer::_notification(int p_what) {
    // Notification handling disabled for Object-based implementation
    // Will be called manually from module initialization
}

void GaussianSplatRenderer::initialize() {
    bool dispatch_submitted = false;
    if (_dispatch_call_on_render_thread_blocking(
                callable_mp(this, &GaussianSplatRenderer::_initialize_on_render_thread),
                &dispatch_submitted)) {
        return;
    }
    if (dispatch_submitted) {
        // A render-thread callback is already queued; avoid racing it with a
        // local fallback initialization path.
        get_resource_state().gpu_initialization_pending = true;
        GS_LOG_RENDERER_WARN("[GaussianSplatRenderer] initialize fallback skipped because render-thread request is still pending");
        return;
    }

    // Manual initialization called from module
    // Try to initialize GPU resources
    _create_gpu_resources_safe();

    if (get_device_state().rd) {
        initialize_sorting();
    }

    if (get_scene_state().gaussian_data.is_valid()) {
        Error data_err = _update_gpu_buffers_with_real_data();
        if (data_err == OK) {
            get_frame_state().visible_splat_count.store(get_scene_state().gaussian_data->get_count(), std::memory_order_release);
        } else {
            get_frame_state().visible_splat_count.store(0, std::memory_order_release);
            GS_LOG_ERROR_DEFAULT(vformat("[Hello Splat] Failed to activate gaussian data during initialization: %d", data_err));
        }
    } else {
        get_frame_state().visible_splat_count.store(0, std::memory_order_release);
    }
}

void GaussianSplatRenderer::_initialize_on_render_thread(uint64_t p_request_id) {
    initialize();
    _notify_render_thread_dispatch_completed(p_request_id);
}

void GaussianSplatRenderer::set_painterly_material(const Ref<PainterlyMaterial> &p_material) {
    if (subsystem_state.painterly_renderer.is_valid() && subsystem_state.painterly_renderer->get_material() == p_material) {
        return;
    }

    if (subsystem_state.painterly_renderer.is_valid()) {
        subsystem_state.painterly_renderer->set_material(p_material);
    }

    if (subsystem_state.painterly_material_manager.is_valid()) {
        subsystem_state.painterly_material_manager->set_material(p_material);
    }

    if (subsystem_state.painterly_renderer.is_valid()) {
        subsystem_state.painterly_renderer->mark_material_dirty();
        subsystem_state.painterly_renderer->update_painterly_gpu_resources(this);
    }
}

Ref<PainterlyMaterial> GaussianSplatRenderer::get_painterly_material() const {
    if (subsystem_state.painterly_renderer.is_valid()) {
        return subsystem_state.painterly_renderer->get_material();
    }
    return Ref<PainterlyMaterial>();
}

void GaussianSplatRenderer::update_depth_range(float p_near, float p_far) {
    // Called from renderer_scene_render_rd.cpp before rendering.
    // Currently a no-op stub - depth range is handled via projection matrix.
    (void)p_near;
    (void)p_far;
}

void GaussianRenderFacadeState::ShadowBlitState::clear(RenderingDevice *p_device) {
    if (sampler.is_valid()) {
        RenderingDevice *owner = sampler_owner.device ? sampler_owner.device : p_device;
        if (owner) {
            owner->free(sampler);
        }
    }
    sampler = RID();
    sampler_owner.clear();

    if (shader.is_valid()) {
        RenderingDevice *owner = p_device ? p_device : sampler_owner.device;
        if (owner) {
            owner->free(shader);
        }
    }
    shader = RID();
    pipeline_cache.clear();
    device_id = 0;
}

bool GaussianSplatRenderer::_ensure_shadow_output_compositor(RenderingDevice *p_device) {
    if (!p_device) {
        return false;
    }
    if (!shadow_output_compositor.is_valid()) {
        shadow_output_compositor.instantiate();
        shadow_output_compositor->set_device_manager(subsystem_state.device_manager);
        shadow_output_compositor->set_texture_format_callback([this](RenderingDevice *p_dev, RID p_texture) -> RD::TextureFormat {
            return _get_texture_format(p_dev, p_texture);
        });
    }
    uint64_t device_id = p_device->get_device_instance_id();
    if (shadow_output_device_id != device_id) {
        shadow_output_compositor->shutdown();
        shadow_output_device_id = device_id;
    }
    if (!shadow_output_compositor->is_initialized()) {
        shadow_output_compositor->initialize(p_device);
    }
    return shadow_output_compositor->is_initialized();
}

bool GaussianSplatRenderer::_ensure_shadow_blit_resources(RenderingDevice *p_device) {
    if (!p_device) {
        return false;
    }

    uint64_t device_id = p_device->get_device_instance_id();
    if (shadow_blit_state.device_id != 0 && shadow_blit_state.device_id != device_id) {
        shadow_blit_state.clear(shadow_blit_state.sampler_owner.device ? shadow_blit_state.sampler_owner.device : p_device);
    }
    shadow_blit_state.device_id = device_id;

    if (!shadow_blit_state.shader_source_initialized) {
        if (!shadow_blit_state.shader_source) {
            shadow_blit_state.shader_source = std::make_unique<GsShadowBlitShaderRD>();
        }
        Vector<String> versions;
        versions.push_back(String());
        shadow_blit_state.shader_source->initialize(versions);
        shadow_blit_state.shader_source_initialized = true;
    }

    if (!shadow_blit_state.shader.is_valid()) {
        RID shader_version = shadow_blit_state.shader_source->version_create();
        if (!shader_version.is_valid()) {
            return false;
        }
        Vector<String> stage_sources = shadow_blit_state.shader_source->version_build_variant_stage_sources(shader_version, 0);
        shadow_blit_state.shader_source->version_free(shader_version);
        if (stage_sources.size() <= RD::SHADER_STAGE_FRAGMENT) {
            return false;
        }
        String compile_error;
        shadow_blit_state.shader = ShaderCompilationHelper::compile_graphics_shader_on_device(
                p_device,
                stage_sources[RD::SHADER_STAGE_VERTEX],
                stage_sources[RD::SHADER_STAGE_FRAGMENT],
                "gs_shadow_blit",
                Vector<String>(),
                &compile_error);
        if (!shadow_blit_state.shader.is_valid()) {
            GS_LOG_ERROR_DEFAULT(vformat("[GS Shadow] Failed to compile shadow blit shader: %s", compile_error));
            return false;
        }

        RD::PipelineRasterizationState raster_state;
        raster_state.cull_mode = RD::POLYGON_CULL_DISABLED;
        RD::PipelineDepthStencilState depth_state;
        depth_state.enable_depth_test = true;
        depth_state.enable_depth_write = true;
        depth_state.depth_compare_operator = RD::COMPARE_OP_GREATER_OR_EQUAL;
        RD::PipelineColorBlendState blend_state = RD::PipelineColorBlendState::create_disabled(0);
        shadow_blit_state.pipeline_cache.setup(shadow_blit_state.shader, RD::RENDER_PRIMITIVE_TRIANGLES,
                raster_state, RD::PipelineMultisampleState(), depth_state, blend_state, 0);
    }

    if (!shadow_blit_state.sampler.is_valid() || !shadow_blit_state.sampler_owner.matches(p_device)) {
        if (shadow_blit_state.sampler.is_valid() && shadow_blit_state.sampler_owner.device &&
                !shadow_blit_state.sampler_owner.matches(p_device)) {
            shadow_blit_state.sampler_owner.device->free(shadow_blit_state.sampler);
        }
        RD::SamplerState sampler_state;
        sampler_state.mag_filter = RD::SAMPLER_FILTER_NEAREST;
        sampler_state.min_filter = RD::SAMPLER_FILTER_NEAREST;
        sampler_state.mip_filter = RD::SAMPLER_FILTER_NEAREST;
        sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
        sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
        sampler_state.repeat_w = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
        sampler_state.enable_compare = false;
        shadow_blit_state.sampler = p_device->sampler_create(sampler_state);
        shadow_blit_state.sampler_owner.set(p_device);
    }

    return shadow_blit_state.shader.is_valid() && shadow_blit_state.sampler.is_valid();
}

bool GaussianSplatRenderer::_blit_shadow_depth(RID p_source_depth, RID p_shadow_fb, const Rect2i &p_atlas_rect, bool p_flip_y) {
    RenderingDevice *rd = RenderingDevice::get_singleton();
    if (!rd || !p_source_depth.is_valid() || !p_shadow_fb.is_valid()) {
        return false;
    }
    if (!rd->framebuffer_is_valid(p_shadow_fb) || !rd->texture_is_valid(p_source_depth)) {
        return false;
    }
    if (!_ensure_shadow_blit_resources(rd)) {
        return false;
    }

    UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
    RendererRD::MaterialStorage *material_storage = RendererRD::MaterialStorage::get_singleton();
    ERR_FAIL_NULL_V(uniform_set_cache, false);
    ERR_FAIL_NULL_V(material_storage, false);

    RD::Uniform u_source;
    u_source.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
    u_source.binding = 0;
    u_source.append_id(shadow_blit_state.sampler);
    u_source.append_id(p_source_depth);

    RID uniform_set = uniform_set_cache->get_cache(shadow_blit_state.shader, 0, u_source);
    RD::FramebufferFormatID fb_format = rd->framebuffer_get_format(p_shadow_fb);
    RID pipeline = shadow_blit_state.pipeline_cache.get_render_pipeline(RD::INVALID_ID, fb_format);

    struct ShadowBlitPushConstant {
        float uv_scale_offset[4];
        float invert_depth;
        float pad0;
        float pad1;
        float pad2;
    } push{};

    const float y_scale = p_flip_y ? -1.0f : 1.0f;
    const float y_offset = p_flip_y ? 1.0f : 0.0f;
    push.uv_scale_offset[0] = 1.0f;
    push.uv_scale_offset[1] = y_scale;
    push.uv_scale_offset[2] = 0.0f;
    push.uv_scale_offset[3] = y_offset;
    push.invert_depth = 1.0f;

    RD::DrawListID draw_list = rd->draw_list_begin(p_shadow_fb, RD::DRAW_DEFAULT_ALL, Vector<Color>(), 1.0f, 0, p_atlas_rect);
    rd->draw_list_bind_render_pipeline(draw_list, pipeline);
    rd->draw_list_bind_uniform_set(draw_list, uniform_set, 0);
    rd->draw_list_bind_index_array(draw_list, material_storage->get_quad_index_array());
    rd->draw_list_set_push_constant(draw_list, &push, sizeof(push));
    rd->draw_list_draw(draw_list, true);
    rd->draw_list_end();

    return true;
}

bool GaussianSplatRenderer::render_directional_shadow_map(const Projection &p_light_projection, const Transform3D &p_light_transform,
        const Rect2i &p_atlas_rect, RID p_shadow_framebuffer, bool p_flip_y) {
    if (p_atlas_rect.size.x <= 0 || p_atlas_rect.size.y <= 0) {
        return false;
    }
    if (!ensure_rendering_device("render_directional_shadow_map")) {
        return false;
    }
    RenderingDevice *rd = get_device_state().rd;
    if (!rd) {
        return false;
    }

    if (!_ensure_shadow_output_compositor(rd)) {
        return false;
    }

    struct ViewStateRestore {
        ViewState &state;
        Size2i manual_viewport;
        RD::DataFormat manual_format;
        Transform3D cam_transform;
        Projection cam_projection;
        bool using_scene_data;
        ViewStateRestore(ViewState &p_state) :
                state(p_state),
                manual_viewport(p_state.manual_viewport_override),
                manual_format(p_state.manual_viewport_format_override),
                cam_transform(p_state.last_camera_to_world_transform),
                cam_projection(p_state.last_camera_projection),
                using_scene_data(p_state.using_scene_data) {}
        ~ViewStateRestore() {
            state.manual_viewport_override = manual_viewport;
            state.manual_viewport_format_override = manual_format;
            state.last_camera_to_world_transform = cam_transform;
            state.last_camera_projection = cam_projection;
            state.using_scene_data = using_scene_data;
        }
    } view_state_restore(get_view_state());

    ViewState &view_state = get_view_state();
    view_state.manual_viewport_override = p_atlas_rect.size;
    view_state.manual_viewport_format_override = RD::DATA_FORMAT_R8G8B8A8_UNORM;
    view_state.last_camera_to_world_transform = p_light_transform;
    view_state.last_camera_projection = p_light_projection;
    view_state.using_scene_data = false;

    Projection projection = p_light_projection;
    Projection render_projection = p_light_projection;
    bool gs_flip_y = !p_flip_y;
    if (gs_flip_y) {
        projection.columns[1][1] = -projection.columns[1][1];
        render_projection.columns[1][1] = -render_projection.columns[1][1];
    }

    Ref<OutputCompositor> saved_output_compositor = subsystem_state.output_compositor;
    subsystem_state.output_compositor = shadow_output_compositor;

    render_sorted_splats(nullptr, p_light_transform.affine_inverse(), projection, render_projection, false);

    subsystem_state.output_compositor = saved_output_compositor;

    if (!subsystem_state.rasterizer.is_valid()) {
        return false;
    }
    RID depth_texture = subsystem_state.rasterizer->get_depth_texture();
    RenderingDevice *depth_owner = subsystem_state.rasterizer->get_depth_texture_owner();
    if (!depth_texture.is_valid() || !depth_owner) {
        return false;
    }
    RenderingDevice *main_device = RenderingDevice::get_singleton();
    if (main_device && depth_owner && depth_owner != main_device) {
        if (!main_device->texture_is_valid(depth_texture)) {
            GS_LOG_WARN_DEFAULT("[GS Shadow] Depth texture is not visible on the main RenderingDevice; skipping shadow blit.");
            return false;
        }
        GS_LOG_WARN_DEFAULT("[GS Shadow] Depth texture owner mismatch; using main RenderingDevice alias for shadow blit");
    }

    return _blit_shadow_depth(depth_texture, p_shadow_framebuffer, p_atlas_rect, p_flip_y);
}

void GaussianSplatRenderer::_on_painterly_material_changed() {
    if (subsystem_state.painterly_renderer.is_valid()) {
        subsystem_state.painterly_renderer->mark_material_dirty();
        subsystem_state.painterly_renderer->update_painterly_gpu_resources(this);
    }
}

GaussianSplatRenderer::SortStageSummary GaussianSplatRenderer::sort_gaussians_for_view(
        const Transform3D &p_world_to_camera_transform, IndexDomain p_input_domain) {
    return sorting_orchestrator->sort_gaussians_for_view(p_world_to_camera_transform, p_input_domain);
}

GaussianSplatRenderer::CullStageOutput GaussianSplatRenderer::_cull_for_view(const Transform3D &p_world_to_camera_transform,
        const Projection &p_projection, const Size2i &p_viewport_size) {
    return quality_orchestrator->cull_for_view(p_world_to_camera_transform, p_projection, p_viewport_size);
}

RID GaussianSplatRenderer::_get_painterly_depth_texture() const {
    if (subsystem_state.rasterizer.is_valid() && subsystem_state.rasterizer->has_depth_output()) {
        RID tile_depth = subsystem_state.rasterizer->get_depth_texture();
        if (tile_depth.is_valid()) {
            RenderingDevice *main_device = get_main_rendering_device();
            RenderingDevice *depth_owner = subsystem_state.rasterizer->get_depth_texture_owner();
            if (main_device && depth_owner && depth_owner != main_device) {
                if (!main_device->texture_is_valid(tile_depth)) {
                    GS_LOG_WARN_DEFAULT("[Painterly] Depth texture owner mismatch and no main-device alias; skipping tile depth");
                    return RID();
                }
                GS_LOG_WARN_DEFAULT("[Painterly] Depth texture owner mismatch; using main RenderingDevice alias for painterly depth");
                return tile_depth;
            } else {
                return tile_depth;
            }
        }
    }

    if (subsystem_state.painterly_renderer.is_valid()) {
        PainterlyPassGraph *pass_graph = subsystem_state.painterly_renderer->get_pass_graph();
        if (pass_graph) {
            return pass_graph->get_shared_texture(PainterlyPassGraph::TEXTURE_DEPTH);
        }
    }

    return RID();
}

const Gaussian *GaussianSplatRenderer::_get_streamed_gaussian(uint32_t p_index) const {
    const StreamingState &streaming_state = get_streaming_state();
    if (!streaming_state.use_streamed_data || streaming_state.cached_streamed_gaussians.is_empty()) {
        return nullptr;
    }

    if (const uint32_t *offset = streaming_state.cached_streamed_index_lookup.getptr(p_index)) {
        uint32_t position = *offset;
        if (position < (uint32_t)streaming_state.cached_streamed_gaussians.size()) {
            return &streaming_state.cached_streamed_gaussians[position];
        }
    }

    return nullptr;
}

RID GaussianSplatRenderer::_get_viewport_color_target(RenderSceneBuffersRD *p_render_buffers) {
    if (!p_render_buffers) {
        return RID();
    }

    if (p_render_buffers->has_internal_texture()) {
        RID internal_texture = p_render_buffers->get_internal_texture();
        if (internal_texture.is_valid()) {
            RenderingDevice *main_device = get_main_rendering_device();
            if (!main_device || main_device->texture_is_valid(internal_texture)) {
                if (main_device) {
                    track_resource_owner(internal_texture, main_device);
                }
                return internal_texture;
            }
        }
    }

    RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();
    if (texture_storage) {
        RID render_target_rid = p_render_buffers->get_render_target();
        if (render_target_rid.is_valid()) {
            RID render_target_texture = texture_storage->render_target_get_rd_texture(render_target_rid);
            if (render_target_texture.is_valid()) {
                if (RenderingDevice *main_device = get_main_rendering_device()) {
                    track_resource_owner(render_target_texture, main_device);
                }
                return render_target_texture;
            }
        }
    }

    return RID();
}

void GaussianSplatRenderer::_check_dual_state_sync(const char *p_context) const {
#ifdef DEV_ENABLED
    if (!debug_state_orchestrator || !get_debug_config().enable_state_guardrails) {
        return;
    }
    if (!config_orchestrator || !data_orchestrator || !instancing_orchestrator || !device_orchestrator || !resource_orchestrator
            || !sorting_orchestrator || !quality_orchestrator) {
        return;
    }

    (void)p_context;
#endif
}

void GaussianSplatRenderer::_set_manual_viewport_format(RD::DataFormat p_format, const char *p_context) {
    (void)p_context;
    get_view_state().manual_viewport_format_override = p_format;
}

void GaussianSplatRenderer::_set_active_viewport_format(RD::DataFormat p_format, const char *p_context) {
    (void)p_context;
    get_view_state().active_viewport_color_format = p_format;
}

void GaussianSplatRenderer::_prepare_render_frame_context(RenderDataRD *p_render_data, const Transform3D &p_world_to_camera_transform,
        const Projection &p_projection, const Projection &p_render_projection, bool p_defer_render_buffers_commit,
        RenderFrameContext &r_context) {
    // Delegate to RenderPipelineStages (T4-PR3)
    pipeline_stages->prepare_frame_context(p_render_data, p_world_to_camera_transform, p_projection,
            p_render_projection, p_defer_render_buffers_commit, r_context);
}

void GaussianSplatRenderer::_run_pipeline_entry(const RenderFrameContext &p_frame_context,
        bool p_has_render_data, const String &p_cull_skip_reason, const String &p_sort_skip_reason,
        RenderFallbackReason p_cull_skip_reason_code, RenderFallbackReason p_sort_skip_reason_code,
        bool p_set_skip_metrics, bool p_clear_cull_state_on_skip) {
    // Delegate to RenderPipelineStages (T4-PR3)
    pipeline_stages->execute_frame_entry(p_frame_context, p_has_render_data, p_cull_skip_reason,
            p_sort_skip_reason, p_cull_skip_reason_code, p_sort_skip_reason_code, p_set_skip_metrics,
            p_clear_cull_state_on_skip);
}

void GaussianSplatRenderer::_run_cull_sort_pipeline_frame(RenderDataRD *p_render_data,
        const Transform3D &p_world_to_camera_transform, const Projection &p_projection,
        const Projection &p_render_projection, RenderSceneBuffersRD *p_render_buffers,
        bool p_has_render_data, const String &p_cull_skip_reason, const String &p_sort_skip_reason,
        RenderFallbackReason p_cull_skip_reason_code, RenderFallbackReason p_sort_skip_reason_code,
        bool p_set_skip_metrics, bool p_clear_cull_state_on_skip) {
    StageMetrics stage_metrics{};
    RenderFrameContext frame_context;
    _prepare_render_frame_context(p_render_data, p_world_to_camera_transform, p_projection, p_render_projection,
            p_render_buffers != nullptr, frame_context);
    frame_context.metrics = &stage_metrics;
    FrameStateProvider frame_provider(this, &frame_context.deps);
    const IFrameStateView &state_view = frame_provider;
    IFrameMutationAccess &state_mut = frame_provider;
    frame_context.state_view = &state_view;
    frame_context.mutation_access = &state_mut;
    _run_pipeline_entry(frame_context, p_has_render_data, p_cull_skip_reason, p_sort_skip_reason,
            p_cull_skip_reason_code, p_sort_skip_reason_code, p_set_skip_metrics, p_clear_cull_state_on_skip);
}

void GaussianSplatRenderer::_reset_legacy_streaming_data_path_state() {
    StreamingState &streaming_state = get_streaming_state();
    streaming_state.use_streamed_data = false;
    streaming_state.cached_streamed_gaussians.clear();
    streaming_state.cached_streamed_indices.clear();
    streaming_state.cached_streamed_source_indices.clear();
    streaming_state.cached_streamed_sh_limits.clear();
    streaming_state.cached_streamed_index_lookup.clear();
    streaming_state.current_stream_gpu_buffer = RID();
    streaming_state.streaming_gpu_splat_count = 0;
    streaming_state.streaming_gpu_total_capacity = 0;
    streaming_state.streamed_indices_generation = 0;
    streaming_state.streamed_indices_are_local = false;
    streaming_state.cached_streamed_indices_valid = false;
}


void GaussianSplatRenderer::_render_resident_frame(RenderDataRD *p_render_data, const Transform3D &p_world_to_camera_transform,
        const Projection &p_projection, const Projection &p_render_projection, RenderSceneBuffersRD *p_render_buffers) {
    // Fallback to test data if streaming is not active.
    _reset_legacy_streaming_data_path_state();

    StreamingState &streaming_state = get_streaming_state();

    const bool has_gaussian_dataset = get_scene_state().gaussian_data.is_valid();
    bool has_buffer_manager_data = get_resource_state().buffer_manager.is_valid() &&
            get_resource_state().buffer_manager_initialized;
    if (has_buffer_manager_data) {
        has_buffer_manager_data = get_resource_state().buffer_manager->get_gaussian_count() > 0;
    }

    // Legacy instance transforms removed; use the view transform directly.
    Transform3D effective_view_transform = p_world_to_camera_transform;

#if defined(DEBUG_ENABLED) || kLogFrameDebug
    // DEBUG: Log instance transform in resident path
    const bool frame_logs_enabled = _is_frame_log_enabled();
    const uint64_t log_frame = get_frame_state().frame_counter;
    const bool should_log_frame = _should_log_frame(log_frame);
    if (should_log_frame) {
        const bool instance_buffers_valid = has_instance_pipeline_buffers();
        GS_LOG_RENDERER_DEBUG(vformat("[RESIDENT-DEBUG] Frame %d: instance_pipeline_buffers_valid=%s",
                log_frame, instance_buffers_valid ? "TRUE" : "FALSE"));
    }
#endif

    if (!has_gaussian_dataset && !has_buffer_manager_data) {
        _run_cull_sort_pipeline_frame(p_render_data, effective_view_transform, p_projection, p_render_projection,
                p_render_buffers, false,
                "Cull skipped: no render data",
                "Sort skipped: no render data",
                RenderFallbackReason::DATA_UNAVAILABLE,
                RenderFallbackReason::DATA_UNAVAILABLE,
                true, true);
        return;
    }

    const bool has_dynamic_asset_handle = streaming_state.shared_dynamic_asset_handle.is_valid();
    const bool instanced_path_ready = streaming_state.current_streaming_system.is_valid() &&
            has_instance_pipeline_buffers();

    if (has_dynamic_asset_handle && instanced_path_ready) {
        if (debug_state_orchestrator) {
            get_debug_state().route_uid = RenderRouteUID::INSTANCE_ENTRY_INSTANCED_FAST;
        }
        LocalVector<Transform3D> instance_transforms;
        instance_transforms.push_back(Transform3D());
        // NOTE: Pass p_world_to_camera_transform (not effective_view_transform) because render_instanced()
        // will multiply the instance_transform into the view transform internally.
        // Passing effective_view_transform would apply instance_transform TWICE.
        render_instanced(p_render_data, streaming_state.shared_dynamic_asset_handle,
                p_world_to_camera_transform, p_projection, p_render_projection, instance_transforms);
        return;
    }
    if (has_dynamic_asset_handle && !instanced_path_ready) {
        WARN_PRINT_ONCE("[GaussianSplatRenderer] Resident fallback bypassed instanced path because instance pipeline is not ready.");
    }
    const bool resident_pipeline_ready = has_instance_pipeline_buffers();
    if (!resident_pipeline_ready) {
        WARN_PRINT_ONCE("[GaussianSplatRenderer] Resident fallback skipped: instance pipeline buffers unavailable.");
    }
    _run_cull_sort_pipeline_frame(p_render_data, effective_view_transform, p_projection, p_render_projection,
            p_render_buffers, resident_pipeline_ready,
            "Cull skipped: resident fallback instance pipeline unavailable",
            "Sort skipped: resident fallback instance pipeline unavailable",
            RenderFallbackReason::STREAMING_DATA_UNAVAILABLE,
            RenderFallbackReason::STREAMING_DATA_UNAVAILABLE,
            false, false);
}

void GaussianSplatRenderer::render_scene_instance(RenderDataRD *p_render_data) {
    // Render flow: render_scene_instance -> (streaming ? render_streaming_frame : render_resident_frame)
    // -> RenderPipelineStages::render_sorted_splats_with_context -> raster/composite.

    if (debug_state_orchestrator) {
        DebugState &debug_state = get_debug_state();
        debug_state.route_uid = RenderRouteUID::COMMON_UNSET_ROUTE;
        debug_state.sort_route_uid = RenderRouteUID::COMMON_UNSET_SORT_ROUTE;
    }

    get_resource_state().deletion_queue.process_frame();
    _check_dual_state_sync("render_scene_instance");

    if (get_subsystem_state().output_compositor.is_valid()) {
        auto &output_cache = get_subsystem_state().output_compositor->get_cache_state();
        output_cache.render_buffers_commit_pending = false;
        output_cache.pending_render_buffers_size = Size2i();
        output_cache.pending_painterly_commit = false;
    }

    if (!ensure_rendering_device("render_scene_instance")) {
        if (debug_state_orchestrator) {
            get_debug_state().route_uid = RenderRouteUID::COMMON_FAIL_NO_DEVICE;
        }
        get_frame_state().visible_splat_count.store(0, std::memory_order_release);
        get_frame_state().render_time_ms = 0.0f;
        get_frame_state().sort_time_ms = 0.0f;
        get_sorting_state().sorted_splat_count = 0;
        if (get_subsystem_state().gpu_culler.is_valid()) {
            get_subsystem_state().gpu_culler->get_state().culled_indices.clear();
            get_subsystem_state().gpu_culler->get_state().culled_distances_sq.clear();
            get_subsystem_state().gpu_culler->get_state().culled_importance_weights.clear();
        }
        pipeline_stages->reset_render_state_for_frame();
        if (debug_state_orchestrator) {
            debug_state_orchestrator->store_stage_metrics(StageMetrics());
        }
        if (get_streaming_state().memory_stream.is_valid()) {
            get_streaming_state().memory_stream->end_frame();
        }
        if (get_streaming_state().current_streaming_system.is_valid()) {
            get_streaming_state().current_streaming_system->end_frame();
        }
        return;
    }

    Transform3D cam_transform = get_view_state().last_camera_to_world_transform;
    Projection cam_projection = get_view_state().last_camera_projection;

    RenderSceneBuffersRD *render_buffers_rd = nullptr;
    get_view_state().using_scene_data = false;

#if defined(DEBUG_ENABLED) || kLogFrameDebug
    // DEBUG: Track camera source
    const bool frame_logs_enabled = _is_frame_log_enabled();
    const uint64_t log_frame = get_frame_state().frame_counter;
    const bool should_log_frame = _should_log_frame(log_frame);
    String cam_source = frame_logs_enabled ? "FALLBACK (cached)" : String();
#endif

    if (p_render_data) {
        if (p_render_data->scene_data) {
            cam_transform = p_render_data->scene_data->cam_transform;
            cam_projection = p_render_data->scene_data->cam_projection;
            get_view_state().using_scene_data = true;
#if defined(DEBUG_ENABLED) || kLogFrameDebug
            if (frame_logs_enabled) {
                cam_source = "scene_data (viewport camera)";
            }
#endif
        }

        if (p_render_data->render_buffers.is_valid()) {
            render_buffers_rd = Object::cast_to<RenderSceneBuffersRD>(p_render_data->render_buffers.ptr());
        }
    }
    if (GaussianSplatting::is_debug_frame_logging_enabled()) {
        static int rb_dbg = 0;
        if (++rb_dbg <= 3) {
            GS_LOG_RENDERER_DEBUG(vformat("[RB-DBG] render_buffers_rd=%s", render_buffers_rd ? "valid" : "NULL"));
        }
    }

    get_view_state().last_camera_position = cam_transform.origin;
    get_view_state().last_camera_to_world_transform = cam_transform;
    get_view_state().last_camera_projection = cam_projection;

    // Always derive a view-space transform (world -> camera) once and reuse it throughout the frame.
    Transform3D view_transform = cam_transform.affine_inverse();
    // Apply flip_y only; gaussian pipeline uses linear view-space depth.
    Projection render_projection = cam_projection;
    if (get_view_state().using_scene_data && p_render_data && p_render_data->scene_data && p_render_data->scene_data->flip_y) {
        render_projection.columns[1][1] = -render_projection.columns[1][1];
    }

#if defined(DEBUG_ENABLED) || kLogFrameDebug
    // DEBUG: Log camera source at the configured frame-log interval (and frame 0)
    if (should_log_frame) {
        GS_LOG_RENDERER_DEBUG(vformat("[CAM-SOURCE] Frame %d: %s", log_frame, cam_source));
        GS_LOG_RENDERER_DEBUG(vformat("  cam_transform.origin (WORLD): (%.2f, %.2f, %.2f)", cam_transform.origin.x, cam_transform.origin.y, cam_transform.origin.z));
        GS_LOG_RENDERER_DEBUG(vformat("  cam_transform.basis[0]: (%.4f, %.4f, %.4f)", cam_transform.basis[0][0], cam_transform.basis[0][1], cam_transform.basis[0][2]));
        GS_LOG_RENDERER_DEBUG(vformat("  view_transform.origin: (%.2f, %.2f, %.2f)", view_transform.origin.x, view_transform.origin.y, view_transform.origin.z));
        // Debug projection values - all columns
        GS_LOG_RENDERER_DEBUG(vformat("  cam_projection[0]: (%.4f, %.4f, %.4f, %.4f)", cam_projection.columns[0][0], cam_projection.columns[0][1], cam_projection.columns[0][2], cam_projection.columns[0][3]));
        GS_LOG_RENDERER_DEBUG(vformat("  cam_projection[1]: (%.4f, %.4f, %.4f, %.4f)", cam_projection.columns[1][0], cam_projection.columns[1][1], cam_projection.columns[1][2], cam_projection.columns[1][3]));
        GS_LOG_RENDERER_DEBUG(vformat("  cam_projection[2]: (%.4f, %.4f, %.4f, %.4f)", cam_projection.columns[2][0], cam_projection.columns[2][1], cam_projection.columns[2][2], cam_projection.columns[2][3]));
        GS_LOG_RENDERER_DEBUG(vformat("  cam_projection[3]: (%.4f, %.4f, %.4f, %.4f)", cam_projection.columns[3][0], cam_projection.columns[3][1], cam_projection.columns[3][2], cam_projection.columns[3][3]));
        GS_LOG_RENDERER_DEBUG(vformat("  render_projection[0]: (%.4f, %.4f, %.4f, %.4f)", render_projection.columns[0][0], render_projection.columns[0][1], render_projection.columns[0][2], render_projection.columns[0][3]));
        GS_LOG_RENDERER_DEBUG(vformat("  render_projection[1]: (%.4f, %.4f, %.4f, %.4f)", render_projection.columns[1][0], render_projection.columns[1][1], render_projection.columns[1][2], render_projection.columns[1][3]));
        GS_LOG_RENDERER_DEBUG(vformat("  render_projection[2]: (%.4f, %.4f, %.4f, %.4f)", render_projection.columns[2][0], render_projection.columns[2][1], render_projection.columns[2][2], render_projection.columns[2][3]));
        GS_LOG_RENDERER_DEBUG(vformat("  render_projection[3]: (%.4f, %.4f, %.4f, %.4f)", render_projection.columns[3][0], render_projection.columns[3][1], render_projection.columns[3][2], render_projection.columns[3][3]));
        GS_LOG_RENDERER_DEBUG(vformat("  cam_projection.near/far: %.4f / %.4f", cam_projection.get_z_near(), cam_projection.get_z_far()));
        if (p_render_data && p_render_data->scene_data) {
            GS_LOG_RENDERER_DEBUG(vformat("  scene_data->flip_y: %s", p_render_data->scene_data->flip_y ? "TRUE" : "FALSE"));
        }
    }
#endif

    if (get_subsystem_state().painterly_renderer.is_valid() &&
            get_subsystem_state().painterly_renderer->is_material_dirty()) {
        get_subsystem_state().painterly_renderer->update_painterly_gpu_resources(this);
    }

    // Update interactive state if needed
    if (get_interactive_state_config().state_dirty && get_subsystem_state().interactive_state_manager.is_valid()) {
        get_subsystem_state().interactive_state_manager->update_renderer_state_uniforms(this);
    }

    // Update streaming system if active.
    const bool streaming_requested = true;
    bool streaming_ready = get_streaming_state().current_streaming_system.is_valid();
    static uint64_t render_debug_count = 0;
    const auto &debug_config = get_debug_config();
    const bool trace_enabled = GaussianSplatting::debug_trace_is_enabled();
    const bool render_log_enabled = debug_config.enable_frame_logging ||
            debug_config.enable_data_logging ||
            debug_config.enable_all_debug;
    if (render_log_enabled && (++render_debug_count % 300) == 1) {
        GS_LOG_RENDERER_DEBUG(vformat("[RENDER-DBG] streaming_requested=%s streaming_ready=%s",
                streaming_requested ? "yes" : "no",
                streaming_ready ? "yes" : "no"));
    }
#if defined(DEBUG_ENABLED) || kLogFrameDebug
    if (should_log_frame) {
        _trace_render_path(true, log_frame, streaming_requested, streaming_ready);
    }
#endif
    if (streaming_requested && !streaming_ready && streaming_orchestrator) {
        if (streaming_orchestrator->ensure_instance_streaming_system()) {
            streaming_ready = get_streaming_state().current_streaming_system.is_valid();
        }
    }
    // DEBUG: Track why render_streaming_frame might not be called
    static int render_gate_counter = 0;
    if (trace_enabled && ++render_gate_counter % 60 == 1) {
        GaussianSplatting::debug_trace_record_event("render_gate",
                vformat("streaming_requested=%s streaming_ready=%s",
                        streaming_requested ? "YES" : "no",
                        streaming_ready ? "YES" : "no"),
                false);
    }
    if (streaming_requested && streaming_ready) {
        const bool has_primary_gaussian_data =
                get_scene_state().gaussian_data.is_valid() &&
                get_scene_state().gaussian_data->get_count() > 0;
        // Some renderers (for example world/static scenes) can render primary gaussian data
        // without SceneDirector instances. Allow a synthetic primary instance in those cases
        // so streaming cull/sort/raster buffers can still initialize.
        const bool allow_primary_fallback_instance = has_primary_gaussian_data;
        if (debug_state_orchestrator) {
            get_debug_state().route_uid = RenderRouteUID::INSTANCE_STREAMING;
        }
        const bool streaming_frame_rendered = streaming_orchestrator &&
                streaming_orchestrator->render_streaming_frame(
                        p_render_data, cam_transform, view_transform, cam_projection, render_projection, render_buffers_rd,
                        allow_primary_fallback_instance);
        if (streaming_frame_rendered) {
            return;
        }
        String fallback_route_uid;
        if (debug_state_orchestrator) {
            DebugState &debug_state = get_debug_state();
            if (debug_state.route_uid.is_empty() ||
                    debug_state.route_uid == RenderRouteUID::INSTANCE_STREAMING ||
                    debug_state.route_uid == RenderRouteUID::COMMON_SKIP_STREAMING_NOT_READY) {
                debug_state.route_uid = _streaming_not_ready_route_uid("UNKNOWN");
            }
            fallback_route_uid = debug_state.route_uid;
        }
        if (fallback_route_uid.is_empty()) {
            WARN_PRINT_ONCE("[GaussianSplatRenderer] Streaming resources not ready; falling back to resident render path.");
        } else {
            WARN_PRINT_ONCE(vformat("[GaussianSplatRenderer] Streaming resources not ready (route=%s); falling back to resident render path.",
                    fallback_route_uid));
        }
    } else {
        String fallback_route_uid = _streaming_not_ready_route_uid("MISSING_STREAMING_SYSTEM");
        if (debug_state_orchestrator) {
            DebugState &debug_state = get_debug_state();
            if (debug_state.route_uid.is_empty() ||
                    debug_state.route_uid == RenderRouteUID::COMMON_SKIP_STREAMING_NOT_READY ||
                    !_is_typed_streaming_not_ready_route(debug_state.route_uid)) {
                debug_state.route_uid = fallback_route_uid;
            } else {
                fallback_route_uid = debug_state.route_uid;
            }
        }
        if (streaming_requested) {
            WARN_PRINT_ONCE(vformat("[GaussianSplatRenderer] Streaming unavailable (route=%s); falling back to resident render path.",
                    fallback_route_uid));
        }
    }
    _render_resident_frame(p_render_data, view_transform, cam_projection, render_projection, render_buffers_rd);
}

void GaussianSplatRenderer::tick_streaming_only(const Transform3D &p_camera_to_world_transform, const Projection &p_projection) {
    if (!streaming_orchestrator) {
        return;
    }

    get_view_state().last_camera_to_world_transform = p_camera_to_world_transform;
    get_view_state().last_camera_projection = p_projection;

    streaming_orchestrator->tick_streaming_only(p_camera_to_world_transform, p_projection);
}

void GaussianSplatRenderer::render_gaussians(RenderDataRD *p_render_data, const PagedArray<RID> &p_instances) {
    render_scene_instance(p_render_data);
    // Force continuous redraw for gaussian splats (sorting depends on camera)
    RenderingServerDefault::redraw_request();
}

void GaussianSplatRenderer::render_sorted_splats(RenderDataRD *p_render_data,
		const Transform3D &p_world_to_camera_transform, const Projection &p_projection, const Projection &p_render_projection,
	bool p_defer_render_buffers_commit) {
	if (debug_state_orchestrator) {
		DebugState &debug_state = get_debug_state();
		debug_state.sort_route_uid = RenderRouteUID::COMMON_UNSET_SORT_ROUTE;
	}
	StageMetrics stage_metrics{};
	RenderFrameContext frame_context;
	_prepare_render_frame_context(p_render_data, p_world_to_camera_transform, p_projection, p_render_projection,
			p_defer_render_buffers_commit, frame_context);
	frame_context.metrics = &stage_metrics;
	FrameStateProvider frame_provider(this, &frame_context.deps);
	const IFrameStateView &state_view = frame_provider;
	IFrameMutationAccess &state_mut = frame_provider;
	frame_context.state_view = &state_view;
	frame_context.mutation_access = &state_mut;
	RenderFramePlan frame_plan = build_frame_plan(state_view.get_scene_state(), state_view.get_streaming_state(), state_view.get_sorting_state_view(),
			state_view.get_resource_state_view(), state_view.get_subsystem_state_view(), state_view.get_pipeline_features(),
			true, String(), String(), RenderFallbackReason::NONE, RenderFallbackReason::NONE, false, false);
    frame_context.deps.frame_plan = &frame_plan;
    DEV_ASSERT(frame_context.deps.frame_plan);
	ERR_FAIL_COND(!frame_context.deps.validate());
	const FrameState &frame_state = state_view.get_frame_state_view();
	const SortingState &sorting_state = state_view.get_sorting_state_view();
	frame_context.snapshot.valid = true;
	frame_context.snapshot.visible_splats = frame_state.visible_splat_count.load(std::memory_order_acquire);
	frame_context.snapshot.sorted_splats = sorting_state.sorted_splat_count;
	frame_context.snapshot.cull_visible_domain = has_instance_pipeline_buffers() ?
			IndexDomain::CHUNK_REF :
			IndexDomain::GAUSSIAN_GLOBAL;
	frame_context.snapshot.sorted_index_domain = has_instance_pipeline_buffers() ?
			IndexDomain::SPLAT_REF :
			IndexDomain::GAUSSIAN_GLOBAL;
	pipeline_stages->render_sorted_splats_with_context(frame_context);
}

RID GaussianSplatRenderer::get_final_texture() const {
    if (subsystem_state.output_compositor.is_valid()) {
        return subsystem_state.output_compositor->get_final_render_texture();
    }
    return RID();
}

bool GaussianSplatRenderer::has_rendered_content() const {
    if (subsystem_state.output_compositor.is_valid()) {
        return subsystem_state.output_compositor->get_has_valid_render()
                && get_frame_state().visible_splat_count.load(std::memory_order_acquire) > 0;
    }
    return false;
}

AABB GaussianSplatRenderer::get_aabb() const {
    if (get_scene_state().gaussian_data.is_valid()) {
        return get_scene_state().gaussian_data->get_aabb();
    }
    return AABB(Vector3(-1, -1, -1), Vector3(2, 2, 2)); // Default AABB
}

GaussianSplatRenderer::SceneState &GaussianSplatRenderer::get_scene_state() {
    static SceneState fallback;
    ERR_FAIL_NULL_V(data_orchestrator, fallback);
    return data_orchestrator->access_scene_state_mutable();
}

const GaussianSplatRenderer::SceneState &GaussianSplatRenderer::get_scene_state() const {
    static SceneState fallback;
    ERR_FAIL_NULL_V(data_orchestrator, fallback);
    return data_orchestrator->get_scene_state();
}

GaussianSplatRenderer::RenderConfig &GaussianSplatRenderer::get_render_config() {
    static RenderConfig fallback;
    ERR_FAIL_NULL_V(config_orchestrator, fallback);
    return config_orchestrator->get_render_config();
}

const GaussianSplatRenderer::RenderConfig &GaussianSplatRenderer::get_render_config() const {
    static RenderConfig fallback;
    ERR_FAIL_NULL_V(config_orchestrator, fallback);
    return config_orchestrator->get_render_config();
}

GaussianSplatRenderer::DeviceState &GaussianSplatRenderer::get_device_state() {
    static DeviceState fallback;
    ERR_FAIL_NULL_V(device_orchestrator, fallback);
    return device_orchestrator->get_device_state();
}

const GaussianSplatRenderer::DeviceState &GaussianSplatRenderer::get_device_state() const {
    static DeviceState fallback;
    ERR_FAIL_NULL_V(device_orchestrator, fallback);
    return device_orchestrator->get_device_state();
}

GaussianSplatRenderer::PipelineState &GaussianSplatRenderer::get_pipeline_state() {
    static PipelineState fallback;
    ERR_FAIL_NULL_V(resource_orchestrator, fallback);
    return resource_orchestrator->get_pipeline_state();
}

const GaussianSplatRenderer::PipelineState &GaussianSplatRenderer::get_pipeline_state() const {
    static PipelineState fallback;
    ERR_FAIL_NULL_V(resource_orchestrator, fallback);
    return resource_orchestrator->get_pipeline_state();
}

GaussianSplatRenderer::PerformanceSettings &GaussianSplatRenderer::get_performance_settings() {
    static PerformanceSettings fallback;
    ERR_FAIL_NULL_V(quality_orchestrator, fallback);
    return quality_orchestrator->get_performance_settings();
}

const GaussianSplatRenderer::PerformanceSettings &GaussianSplatRenderer::get_performance_settings() const {
    static PerformanceSettings fallback;
    ERR_FAIL_NULL_V(quality_orchestrator, fallback);
    return quality_orchestrator->get_performance_settings();
}

GaussianSplatRenderer::CullingConfig &GaussianSplatRenderer::get_culling_config() {
    static CullingConfig fallback;
    ERR_FAIL_NULL_V(config_orchestrator, fallback);
    return config_orchestrator->get_culling_config();
}

const GaussianSplatRenderer::CullingConfig &GaussianSplatRenderer::get_culling_config() const {
    static CullingConfig fallback;
    ERR_FAIL_NULL_V(config_orchestrator, fallback);
    return config_orchestrator->get_culling_config();
}

GaussianSplatRenderer::PainterlyConfig &GaussianSplatRenderer::get_painterly_config() {
    static PainterlyConfig fallback;
    ERR_FAIL_NULL_V(config_orchestrator, fallback);
    return config_orchestrator->get_painterly_config();
}

const GaussianSplatRenderer::PainterlyConfig &GaussianSplatRenderer::get_painterly_config() const {
    static PainterlyConfig fallback;
    ERR_FAIL_NULL_V(config_orchestrator, fallback);
    return config_orchestrator->get_painterly_config();
}

GaussianSplatRenderer::InteractiveStateConfig &GaussianSplatRenderer::get_interactive_state_config() {
    static InteractiveStateConfig fallback;
    ERR_FAIL_NULL_V(config_orchestrator, fallback);
    return config_orchestrator->get_interactive_state();
}

const GaussianSplatRenderer::InteractiveStateConfig &GaussianSplatRenderer::get_interactive_state_config() const {
    static InteractiveStateConfig fallback;
    ERR_FAIL_NULL_V(config_orchestrator, fallback);
    return config_orchestrator->get_interactive_state();
}

GaussianSplatRenderer::SortingState &GaussianSplatRenderer::get_sorting_state() {
    static SortingState fallback;
    ERR_FAIL_NULL_V(sorting_orchestrator, fallback);
    return sorting_orchestrator->access_sorting_state_mutable();
}

GaussianSplatRenderer::StreamingState &GaussianSplatRenderer::get_streaming_state() {
    static StreamingState fallback;
    ERR_FAIL_NULL_V(data_orchestrator, fallback);
    return data_orchestrator->access_streaming_state_mutable();
}

const GaussianSplatRenderer::StreamingState &GaussianSplatRenderer::get_streaming_state() const {
    static StreamingState fallback;
    ERR_FAIL_NULL_V(data_orchestrator, fallback);
    return data_orchestrator->get_streaming_state();
}

GaussianSplatRenderer::ResourceState &GaussianSplatRenderer::get_resource_state() {
    static ResourceState fallback;
    ERR_FAIL_NULL_V(resource_orchestrator, fallback);
    return resource_orchestrator->get_resource_state();
}

const GaussianSplatRenderer::ResourceState &GaussianSplatRenderer::get_resource_state() const {
    static ResourceState fallback;
    ERR_FAIL_NULL_V(resource_orchestrator, fallback);
    return resource_orchestrator->get_resource_state();
}

void GaussianSplatRenderer::set_instance_pipeline_buffers(const InstancePipelineBuffers &p_buffers) {
    instance_pipeline_buffers = p_buffers;
    StreamingState &streaming_state = get_streaming_state();
    if (instance_pipeline_buffers.atlas_gaussian_count == 0 &&
            streaming_state.shared_dynamic_asset_handle.is_valid()) {
        instance_pipeline_buffers.atlas_gaussian_count = streaming_state.shared_dynamic_asset_handle.gaussian_count;
    }
    if (instance_pipeline_buffers.atlas_gaussian_count == 0 &&
            instance_pipeline_buffers.max_visible_splats > 0) {
        instance_pipeline_buffers.atlas_gaussian_count = instance_pipeline_buffers.max_visible_splats;
        WARN_PRINT_ONCE("[GaussianSplatRenderer] atlas_gaussian_count missing; using max_visible_splats as bounds.");
    }
    instance_pipeline_buffers_valid = true;
}

void GaussianSplatRenderer::clear_instance_pipeline_buffers() {
    instance_pipeline_buffers = InstancePipelineBuffers();
    instance_pipeline_buffers_valid = false;
}

bool GaussianSplatRenderer::update_instance_buffer(LocalVector<InstanceDataGPU> &p_instances) {
    if (!_ensure_rendering_device("update_instance_buffer")) {
        return false;
    }

    StreamingState &streaming_state = get_streaming_state();
    if (!streaming_state.current_streaming_system.is_valid()) {
        WARN_PRINT_ONCE("[GaussianSplatRenderer] Instance buffer upload skipped; streaming system not available.");
        instance_pipeline_buffers.instance_count = 0;
        return false;
    }

    streaming_state.current_streaming_system->remap_instance_asset_ids(p_instances);

    const uint32_t instance_count = p_instances.size();
    if (instance_count == 0) {
        instance_pipeline_buffers.instance_count = 0;
        return true;
    }

    RenderingDevice *rd = get_device_state().rd;
    if (!rd) {
        WARN_PRINT_ONCE("[GaussianSplatRenderer] Instance buffer upload skipped; rendering device not available.");
        instance_pipeline_buffers.instance_count = 0;
        return false;
    }

    ResourceState &resource_state = get_resource_state();
    if (!resource_state.instance_buffer.is_valid() || resource_state.instance_buffer_capacity < instance_count) {
        const uint32_t new_capacity = next_power_of_2(MAX(instance_count, (uint32_t)1));
        const uint64_t buffer_size = static_cast<uint64_t>(new_capacity) * sizeof(InstanceDataGPU);
        if (resource_state.instance_buffer.is_valid()) {
            _free_owned_resource(rd, resource_state.instance_buffer);
        }
        resource_state.instance_buffer = rd->storage_buffer_create(buffer_size);
        if (!resource_state.instance_buffer.is_valid()) {
            resource_state.instance_buffer_capacity = 0;
            WARN_PRINT_ONCE("[GaussianSplatRenderer] Failed to allocate instance buffer.");
            instance_pipeline_buffers.instance_count = 0;
            return false;
        }
        rd->set_resource_name(resource_state.instance_buffer, "GS_InstanceBuffer");
        track_resource_owner(resource_state.instance_buffer, rd);
        resource_state.instance_buffer_capacity = new_capacity;
    }

    const uint64_t upload_size = static_cast<uint64_t>(instance_count) * sizeof(InstanceDataGPU);
    rd->buffer_update(resource_state.instance_buffer, 0, upload_size, p_instances.ptr());
    instance_pipeline_buffers.instance_buffer = resource_state.instance_buffer;
    instance_pipeline_buffers.instance_count = instance_count;

    return true;
}

GaussianSplatRenderer::DebugConfig &GaussianSplatRenderer::get_debug_config() {
    return debug_state_orchestrator->get_config();
}

const GaussianSplatRenderer::DebugConfig &GaussianSplatRenderer::get_debug_config() const {
    return debug_state_orchestrator->get_config();
}

GaussianSplatRenderer::DebugState &GaussianSplatRenderer::get_debug_state() {
    return debug_state_orchestrator->get_state();
}

const GaussianSplatRenderer::DebugState &GaussianSplatRenderer::get_debug_state() const {
    return debug_state_orchestrator->get_state();
}

#ifdef TESTS_ENABLED
void GaussianSplatRenderer::_test_dispatch_noop_callback(uint64_t p_request_id) {
    (void)p_request_id;
}

void GaussianSplatRenderer::test_force_disable_streaming() {
    StreamingState &streaming_state = get_streaming_state();
    streaming_state.use_streamed_data = false;
    streaming_state.streaming_gpu_splat_count = 0;
    streaming_state.streaming_gpu_total_capacity = 0;
    streaming_state.streamed_indices_generation = 0;
    streaming_state.streamed_indices_are_local = false;
    streaming_state.cached_streamed_indices_valid = false;
    get_sorting_state().sorted_splat_count = 0;
    if (streaming_state.current_streaming_system.is_valid()) {
        streaming_state.current_streaming_system->end_frame();
        streaming_state.current_streaming_system.unref();
    }
    streaming_state.cached_streamed_gaussians.clear();
    streaming_state.cached_streamed_indices.clear();
    streaming_state.cached_streamed_source_indices.clear();
    streaming_state.cached_streamed_sh_limits.clear();
    streaming_state.cached_streamed_index_lookup.clear();
    streaming_state.current_stream_gpu_buffer = RID();
    if (streaming_state.memory_stream.is_valid()) {
        streaming_state.memory_stream->shutdown();
    }
}

void GaussianSplatRenderer::test_release_current_streaming_system() {
    StreamingState &streaming_state = get_streaming_state();
    if (streaming_state.current_streaming_system.is_valid()) {
        streaming_state.current_streaming_system->end_frame();
        streaming_state.current_streaming_system.unref();
    }
}

bool GaussianSplatRenderer::test_has_current_streaming_system() const {
    return get_streaming_state().current_streaming_system.is_valid();
}

Ref<OutputCompositor> GaussianSplatRenderer::test_get_output_compositor() const {
    return get_subsystem_state().output_compositor;
}

void GaussianSplatRenderer::test_set_test_splats(const Vector<Vector3> &p_positions, const Vector<Vector3> &p_scales) {
    const int count = p_positions.size();
    get_test_data_state().positions.resize(count);
    get_test_data_state().colors.resize(count);
    get_test_data_state().scales.resize(count);

    const int scale_count = p_scales.size();
    for (int i = 0; i < count; i++) {
        get_test_data_state().positions[i] = p_positions[i];
        get_test_data_state().colors[i] = Color(1.0f, 1.0f, 1.0f, 1.0f);
        if (i < scale_count) {
            get_test_data_state().scales[i] = p_scales[i];
        } else {
            get_test_data_state().scales[i] = Vector3(1.0f, 1.0f, 1.0f);
        }
    }

    get_test_data_state().content_generation++;
}

int GaussianSplatRenderer::test_cull_visible_count(const Transform3D &p_world_to_camera_transform,
        const Projection &p_projection, const Size2i &p_viewport_size) {
    get_scene_state().gaussian_data.unref();
    get_subsystem_state().gpu_culler->get_config().last_cull_viewport_size = p_viewport_size;
    Transform3D view_transform = p_world_to_camera_transform.affine_inverse();
    CullStageOutput output = _cull_for_view(view_transform, p_projection, p_viewport_size);
    return static_cast<int>(output.visible_count);
}

void GaussianSplatRenderer::test_disable_gpu_culler() {
    subsystem_state.gpu_culler.unref();
}

void GaussianSplatRenderer::test_disable_rasterizer() {
    if (tile_renderer_state.renderer.is_valid()) {
        tile_renderer_state.renderer->cleanup();
    }
    tile_renderer_state.renderer.unref();
    tile_renderer_state.init_failed = true;
    subsystem_state.rasterizer.unref();
}

void GaussianSplatRenderer::test_set_render_thread_dispatch_timeout_usec(uint64_t p_timeout_usec) {
    if (render_thread_dispatcher) {
        render_thread_dispatcher->set_timeout_usec(p_timeout_usec);
    }
}

uint64_t GaussianSplatRenderer::test_get_render_thread_dispatch_timeout_usec() const {
    return render_thread_dispatcher ? render_thread_dispatcher->get_timeout_usec() : 0;
}

bool GaussianSplatRenderer::test_dispatch_call_on_render_thread_blocking_without_completion() {
    return _dispatch_call_on_render_thread_blocking(callable_mp(this, &GaussianSplatRenderer::_test_dispatch_noop_callback));
}

bool GaussianSplatRenderer::test_dispatch_call_on_render_thread_blocking_with_completion() {
    return _dispatch_call_on_render_thread_blocking(callable_mp(this, &GaussianSplatRenderer::_notify_render_thread_dispatch_completed));
}

void GaussianSplatRenderer::test_notify_render_thread_dispatch_completed(uint64_t p_request_id) {
    _notify_render_thread_dispatch_completed(p_request_id);
}

uint64_t GaussianSplatRenderer::test_get_render_thread_dispatch_completed_request_id() const {
    return render_thread_dispatcher ? render_thread_dispatcher->get_completed_request_id() : 0;
}
#endif
