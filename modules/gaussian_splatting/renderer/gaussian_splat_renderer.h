/**
 * @file gaussian_splat_renderer.h
 * @brief Core rendering implementation for Gaussian Splatting.
 *
 * This file contains the GaussianSplatRenderer class which manages the complete
 * GPU rendering pipeline for Gaussian splats including:
 * - Tile-based rasterization via the global composite sort pipeline
 * - GPU radix sorting for depth ordering
 * - Frustum culling and LOD
 * - Painterly post-processing effects
 * - Performance monitoring and debug visualization
 */

#ifndef GAUSSIAN_SPLAT_RENDERER_H
#define GAUSSIAN_SPLAT_RENDERER_H

// Scene includes not available in modules - using core Node instead
#include "core/math/aabb.h"
#include "core/object/object.h"
#include "core/object/ref_counted.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"
#include "core/templates/local_vector.h"
#include "core/templates/vector.h"
#include "core/string/string_name.h"
#include "core/variant/typed_array.h"
#include "core/os/mutex.h"
#include "core/os/semaphore.h"
// #include "scene/3d/node_3d.h"
// #include "scene/3d/mesh_instance_3d.h"
// #include "scene/resources/mesh.h"
// #include "scene/resources/material.h"
#include "servers/rendering/renderer_rd/pipeline_cache_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/render_scene_buffers_rd.h"
#include "servers/rendering_server.h"
#include "core/math/random_pcg.h" // For test data generation
#include "../core/gaussian_data.h"
#include "../core/gaussian_splat_asset.h"
#include "../core/gaussian_splat_manager.h"
#include "../painterly/painterly_material.h"
#include "painterly_pass_graph.h"
#include "../interfaces/painterly_renderer.h"
#include "gpu_memory_stream.h"
#include "gaussian_gpu_layout.h"
#include "../core/gaussian_streaming.h"
#include "tile_renderer.h"
#include "gpu_buffer_manager.h"
#include "gpu_performance_monitor.h"
#include "render_frame_context_manager.h"
#include "pipeline_feature_set.h"
#include "../lod/hierarchical_splat_structure.h"
#include "rendering_error.h"
#include "../interfaces/debug_overlay_interfaces.h"
#include "../interfaces/interactive_state_interfaces.h"
#include "../interfaces/rasterizer_interfaces.h"
#include "../interfaces/culler_interfaces.h"
#include "../interfaces/gpu_culler.h"
#include "../interfaces/gpu_sorting_pipeline_interfaces.h"
#include "../interfaces/render_thread_dispatcher.h"
#include "../interfaces/render_device_manager.h"
#include "../interfaces/renderer_interfaces.h"
#include "render_types/render_config_types.h"
#include "render_types/render_debug_types.h"
#include "render_types/render_facade_state_types.h"
#include "render_types/render_frame_types.h"
#include "render_types/render_performance_types.h"
#include "render_types/render_pipeline_io_types.h"
#include "render_types/render_state_types.h"
#include <atomic>
#include <memory>

// Forward declarations for interface implementations
class DebugOverlaySystem;
class InteractiveStateManager;
class TileRasterizer;
class GPUCuller;
class OutputCompositor;
class GPUSortingPipeline;
class OverflowAutoTuner;
class PainterlyMaterialManager;
class RenderPipelineStages;
class RenderDeviceOrchestrator;
class RenderDebugStateOrchestrator;
class RenderDiagnosticsOrchestrator;
class RenderSortingOrchestrator;
class RenderStreamingOrchestrator;
class RenderQualityOrchestrator;
class RenderConfigOrchestrator;
class RenderInstancingOrchestrator;
class RenderResourceOrchestrator;
class RenderDataOrchestrator;
class RenderOutputOrchestrator;
class IGPUSorter;

class RendererSceneRenderRD;
class RenderDataRD;
class GaussianSplatShaderRD;
class GsShadowBlitShaderRD;

struct InstanceAssetRegistration {
    uint32_t asset_id = 0;
    Ref<GaussianData> data;
    uint32_t edited_version = 0;
};

/**
 * @class GaussianSplatRenderer
 * @brief GPU rendering backend for Gaussian Splatting.
 *
 * GaussianSplatRenderer implements the complete rendering pipeline for Gaussian splats
 * using Godot's RenderingDevice API. It is designed for integration with the engine's
 * RendererSceneRenderRD and can be used standalone for viewport rendering.
 *
 * ## Pipeline Overview
 *
 * The renderer uses a global composite sort approach:
 * 1. **Frustum Cull**: GPU hierarchical culling to reject off-screen splats
 * 2. **Tile Binning**: Projects splats and counts per-tile overlaps
 * 3. **Prefix Scan**: GPU parallel scan to build per-tile ranges
 * 4. **Radix Sort**: O(n) GPU sort on composite (tile_id, depth) keys
 * 5. **Tile Rasterize**: Per-pixel Gaussian weight evaluation and alpha blending
 *
 * ## Usage
 *
 * Typically created and managed by GaussianSplatNode3D:
 * @code
 * Ref<GaussianSplatRenderer> renderer;
 * renderer.instantiate();
 * renderer->initialize();
 * renderer->set_gaussian_data(my_data);
 * renderer->render_for_view(camera_transform, projection, render_target, viewport_size);
 * @endcode
 *
 * @note This class is RefCounted and should be used with Ref<GaussianSplatRenderer>.
 */
class GaussianSplatRenderer : public RefCounted, public IRenderer, public ISortResultSink, public ISortBufferHostContext {
    GDCLASS(GaussianSplatRenderer, RefCounted);

public:
    /**
     * @enum RenderMode
     * @brief Rendering mode selection for 2D/3D Gaussian types.
     */
    enum RenderMode {
        MODE_3D,
        MODE_2D,
        MODE_HYBRID
    };

    /**
     * @enum InteractiveState
     * @brief Visual feedback states for editor selection/hover.
     */
    enum InteractiveState {
        STATE_NORMAL = 0,
        STATE_HOVERED = 1,
        STATE_SELECTED = 2,
        STATE_DISABLED = 3
    };

    /**
     * @enum DebugPreviewMode
     * @brief Debug visualization modes for development and troubleshooting.
     */
    enum DebugPreviewMode {
        DEBUG_PREVIEW_OFF = 0,
        DEBUG_PREVIEW_WIREFRAME,
        DEBUG_PREVIEW_POINTS,
        DEBUG_PREVIEW_DEPTH,
        DEBUG_PREVIEW_HEATMAP,
        DEBUG_PREVIEW_RUNTIME_MODIFICATIONS
    };
    using StaticChunk = ::StaticChunk;

private:
    friend struct FrameSyncGuard;

    // Phase 15: _push_texture_trace removed (dead code)
    void _push_cross_device_operation(const String &p_context, RenderingDevice *p_source, RenderingDevice *p_target);
    Dictionary _build_device_capability_report() const;
    void _check_dual_state_sync(const char *p_context) const;

public:
    // State types (public for orchestrator access)
    using IndexDomain = GaussianRenderState::IndexDomain;
    using SceneState = GaussianRenderState::SceneState;
    using StreamingState = GaussianRenderState::StreamingState;
    using SortingState = GaussianRenderState::SortingState;
    using CullStageOutput = GaussianRenderState::CullStageOutput;
    using SortStageSummary = GaussianRenderState::SortStageSummary;

    struct PipelineState {
        RID gaussian_shader;
        RID gaussian_shader_version;
        GaussianSplatShaderRD *gaussian_shader_source = nullptr;
        bool gaussian_shader_initialized = false;
    };

    // Phase 15: painterly_stroke_density_buffer removed - now managed by PainterlyMaterialManager

    // Phase 8: Legacy frustum cull buffers removed - now managed by GPUCuller interface
    // Phase 8: Resource tracking HashMaps removed - now managed by RenderDeviceManager

    // Extracted pipeline I/O types (ISSUE-029)
    using RenderFallbackReason = GaussianRenderPipeline::RenderFallbackReason;
    using StageResult = GaussianRenderPipeline::StageResult;
    using StageIO = GaussianRenderPipeline::StageIO;
    using SortStageOutput = GaussianRenderPipeline::SortStageOutput;
    using SplatDataSource = GaussianRenderPipeline::SplatDataSource;
    using InstancePipelineBuffers = GaussianRenderPipeline::InstancePipelineBuffers;
    using RenderFrameSnapshot = GaussianRenderPipeline::RenderFrameSnapshot;
    using DataSourcePlan = GaussianRenderPipeline::DataSourcePlan;
    using RasterStageOutput = GaussianRenderPipeline::RasterStageOutput;

    // Extracted performance types (ISSUE-029)
    using PerformanceSettings = GaussianRenderPerformance::PerformanceSettings;
    using SortFrameMetrics = GaussianRenderPerformance::SortFrameMetrics;
    using PerformanceMetrics = GaussianRenderPerformance::PerformanceMetrics;
    using PerformanceState = GaussianRenderPerformance::PerformanceState;

    // Rendering state
    using FrameState = RenderFrameContextManager::FrameState;
    using ViewState = RenderFrameContextManager::ViewState;
    RenderFrameContextManager frame_context_manager;


public:
    // Extracted types (ISSUE-029 Phase 2)
    using ErrorRecoveryStateMachine = GaussianRenderFrame::ErrorRecoveryStateMachine;
    using RuntimeErrorStatistics = GaussianRenderFrame::RuntimeErrorStatistics;
    using TextureTraceEntry = GaussianRenderFrame::TextureTraceEntry;
    using CrossDeviceOperation = GaussianRenderFrame::CrossDeviceOperation;
    using FrameTimingSample = GaussianRenderFrame::FrameTimingSample;
    using DiagnosticsState = GaussianRenderFrame::DiagnosticsState;
    using JacobianDebugConfig = GaussianRenderDebug::JacobianDebugConfig;
    using SplatAuditSummary = GaussianRenderDebug::SplatAuditSummary;
    using PainterlyCompositePushConstant = GaussianRenderPipeline::PainterlyCompositePushConstant;
    using RenderFramePlan = GaussianRenderPipeline::RenderFramePlan;
    using StageMetrics = GaussianRenderPipeline::StageMetrics;
    using PipelineEvent = GaussianRenderPipeline::PipelineEvent;
    using RenderConfig = GaussianRenderConfig::RenderConfig<RenderMode>;
    using CullingConfig = GaussianRenderConfig::CullingConfig;
    using PainterlyConfig = GaussianRenderConfig::PainterlyConfig;
    using StateUniformData = GaussianRenderConfig::StateUniformData;
    using InteractiveStateConfig = GaussianRenderConfig::InteractiveStateConfig<InteractiveState>;
    using DebugConfig = GaussianRenderDebug::DebugConfig;
    using DebugState = GaussianRenderDebug::DebugState<DebugPreviewMode, StageMetrics, PipelineEvent>;
    using DeviceState = GaussianRenderFacadeState::DeviceState;
    using ResourceState = GaussianRenderFacadeState::ResourceState;
    using TestDataState = GaussianRenderFacadeState::TestDataState;
    using TileRendererState = GaussianRenderFacadeState::TileRendererState;
    using SubsystemState = GaussianRenderFacadeState::SubsystemState;
    using ShadowBlitState = GaussianRenderFacadeState::ShadowBlitState;

    // Stage types exposed for RenderPipelineStages and orchestrators
    class IFrameStateView;
    class IFrameMutationAccess;

    struct RenderFrameContext {
        uint64_t frame_id = 0;
        RenderDataRD *render_data = nullptr;
        RenderSceneBuffersRD *render_buffers = nullptr;
        RID render_target;
        Transform3D world_to_camera_transform;
        Projection projection;
        Projection cull_projection; // Flip/depth-corrected projection for culling (no jitter).
        Projection render_projection; // GPU projection with depth/jitter correction applied.
        Size2i viewport_size;
        RD::DataFormat viewport_format = RD::DATA_FORMAT_MAX;
        bool defer_commit = false;
        bool painterly_enabled = false;
        StageMetrics *metrics = nullptr;
        const IFrameStateView *state_view = nullptr;
        IFrameMutationAccess *mutation_access = nullptr;
        RenderFrameSnapshot snapshot;
        struct FrameDeps {
            // -- Required: must be non-null for any render path ---------------
            RenderingDevice *rendering_device = nullptr;
            SceneState *scene_state = nullptr;
            ResourceState *resource_state = nullptr;
            FrameState *frame_state = nullptr;
            PerformanceState *performance_state = nullptr;
            SubsystemState *subsystem_state = nullptr;
            RenderConfig *render_config = nullptr;
            const PipelineFeatureSet *pipeline_features = nullptr;

            // -- Required for sort/raster/composite ---------------------------
            GPUSortingPipeline *sorting_pipeline = nullptr;
            SortingState *sorting_state = nullptr;
            StreamingState *streaming_state = nullptr;
            OutputCompositor *output_compositor = nullptr;
            GPUCuller *gpu_culler = nullptr;
            const RenderFramePlan *frame_plan = nullptr; // Set before render_sorted_splats_with_context().

            // -- Optional: null disables the corresponding feature ------------
            PainterlyRenderer *painterly_renderer = nullptr;
            JacobianDebugConfig *jacobian_debug = nullptr;

            /// Validate that all critical pointers are populated.
            /// Call at render-path entry points; returns false and logs the
            /// first null field on failure.
            bool validate() const {
                ERR_FAIL_NULL_V_MSG(rendering_device, false, "FrameDeps: rendering_device is null");
                ERR_FAIL_NULL_V_MSG(scene_state, false, "FrameDeps: scene_state is null");
                ERR_FAIL_NULL_V_MSG(resource_state, false, "FrameDeps: resource_state is null");
                ERR_FAIL_NULL_V_MSG(frame_state, false, "FrameDeps: frame_state is null");
                ERR_FAIL_NULL_V_MSG(performance_state, false, "FrameDeps: performance_state is null");
                ERR_FAIL_NULL_V_MSG(subsystem_state, false, "FrameDeps: subsystem_state is null");
                ERR_FAIL_NULL_V_MSG(render_config, false, "FrameDeps: render_config is null");
                ERR_FAIL_NULL_V_MSG(pipeline_features, false, "FrameDeps: pipeline_features is null");
                ERR_FAIL_NULL_V_MSG(sorting_pipeline, false, "FrameDeps: sorting_pipeline is null");
                ERR_FAIL_NULL_V_MSG(sorting_state, false, "FrameDeps: sorting_state is null");
                ERR_FAIL_NULL_V_MSG(streaming_state, false, "FrameDeps: streaming_state is null");
                ERR_FAIL_NULL_V_MSG(output_compositor, false, "FrameDeps: output_compositor is null");
                ERR_FAIL_NULL_V_MSG(gpu_culler, false, "FrameDeps: gpu_culler is null");
                // frame_plan validated separately (set later in some paths).
                // painterly_renderer and jacobian_debug are optional.
                return true;
            }
        } deps;
    };

    class IFrameStateView {
    public:
        virtual ~IFrameStateView() = default;

        virtual OutputCompositor *get_output_compositor() const = 0;
        virtual GPUCuller *get_gpu_culler() const = 0;
        virtual PainterlyRenderer *get_painterly_renderer() const = 0;
        virtual GPUSortingPipeline *get_sorting_pipeline() const = 0;
        virtual RenderingDevice *get_rendering_device() const = 0;

        virtual const SceneState &get_scene_state() const = 0;
        virtual const StreamingState &get_streaming_state() const = 0;
        virtual const DebugState &get_debug_state_view() const = 0;
        virtual const SortingState &get_sorting_state_view() const = 0;
        virtual const RenderConfig &get_render_config_view() const = 0;
        virtual const JacobianDebugConfig &get_jacobian_debug_view() const = 0;
        virtual const ResourceState &get_resource_state_view() const = 0;
        virtual const FrameState &get_frame_state_view() const = 0;
        virtual const PerformanceState &get_performance_state_view() const = 0;
        virtual const SubsystemState &get_subsystem_state_view() const = 0;
        virtual const PipelineFeatureSet *get_pipeline_features() const = 0;
        virtual const RenderFramePlan *get_frame_plan() const = 0;
    };

    class IFrameMutationAccess {
    public:
        virtual ~IFrameMutationAccess() = default;

        virtual SortingState &get_sorting_state_mut() = 0;
        virtual StreamingState &get_streaming_state_mut() = 0;
        virtual DebugState &get_debug_state_mut() = 0;
        virtual RenderConfig &get_render_config_mut() = 0;
        virtual ResourceState &get_resource_state_mut() = 0;
        virtual FrameState &get_frame_state_mut() = 0;
        virtual PerformanceState &get_performance_state_mut() = 0;
        virtual SubsystemState &get_subsystem_state_mut() = 0;
    };

    class FrameStateProvider : public IFrameStateView, public IFrameMutationAccess {
        GaussianSplatRenderer *renderer = nullptr;
        const RenderFrameContext::FrameDeps *deps = nullptr;

    public:
        FrameStateProvider(GaussianSplatRenderer *p_renderer, const RenderFrameContext::FrameDeps *p_deps = nullptr);

        OutputCompositor *get_output_compositor() const override;
        GPUCuller *get_gpu_culler() const override;
        PainterlyRenderer *get_painterly_renderer() const override;
        GPUSortingPipeline *get_sorting_pipeline() const override;
        RenderingDevice *get_rendering_device() const override;

        const SceneState &get_scene_state() const override;
        const StreamingState &get_streaming_state() const override;
        const DebugState &get_debug_state_view() const override;
        const SortingState &get_sorting_state_view() const override;
        const RenderConfig &get_render_config_view() const override;
        const JacobianDebugConfig &get_jacobian_debug_view() const override;
        const ResourceState &get_resource_state_view() const override;
        const FrameState &get_frame_state_view() const override;
        const PerformanceState &get_performance_state_view() const override;
        const SubsystemState &get_subsystem_state_view() const override;
        SortingState &get_sorting_state_mut() override;
        StreamingState &get_streaming_state_mut() override;
        DebugState &get_debug_state_mut() override;
        RenderConfig &get_render_config_mut() override;
        ResourceState &get_resource_state_mut() override;
        FrameState &get_frame_state_mut() override;
        PerformanceState &get_performance_state_mut() override;
        SubsystemState &get_subsystem_state_mut() override;
        const PipelineFeatureSet *get_pipeline_features() const override;
        const RenderFramePlan *get_frame_plan() const override;
    };

    struct CullStageInput {
        uint64_t frame_id = 0;
        Transform3D world_to_camera_transform;
        Projection projection;
        Size2i viewport_size;
        StageMetrics *metrics = nullptr;
        const IFrameStateView *state_view = nullptr;
    };

    struct SortStageInput {
        uint64_t frame_id = 0;
        Transform3D world_to_camera_transform;
        uint32_t input_count = 0;
        StageMetrics *metrics = nullptr;
        const IFrameStateView *state_view = nullptr;
        IndexDomain input_domain = IndexDomain::UNKNOWN;
    };

    static Error get_active_data_source(const SceneState &p_scene_state,
            const StreamingState &p_streaming_state,
            const SortingState &p_sorting_state,
            const ResourceState &p_resource_state,
            const SubsystemState &p_subsystem_state,
            SplatDataSource &r_source,
            String &r_error);

    static DataSourcePlan build_data_source_plan(const SceneState &p_scene_state,
            const StreamingState &p_streaming_state,
            const SortingState &p_sorting_state,
            const ResourceState &p_resource_state,
            const SubsystemState &p_subsystem_state);
    static void apply_data_source_plan(const DataSourcePlan &p_plan, PerformanceMetrics &p_metrics,
            const ResourceState &p_resource_state);

    static RenderFramePlan build_frame_plan(const SceneState &p_scene_state,
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
            bool p_clear_cull_state_on_skip);

    struct RasterStageInput {
        uint64_t frame_id = 0;
        RenderDataRD *render_data = nullptr;
        Transform3D world_to_camera_transform;
        Projection projection;
        Projection render_projection;
        Size2i viewport_size;
        RD::DataFormat viewport_format = RD::DATA_FORMAT_MAX;
        uint32_t sorted_splat_count = 0;
        uint64_t content_generation = 0;
        uint64_t cull_config_signature = 0;
        uint64_t color_grading_signature = 0;
        uint64_t lighting_signature = 0;
        float sort_time_ms = 0.0f;
        IndexDomain sorted_index_domain = IndexDomain::UNKNOWN;
        bool painterly_requested = false;
        StageMetrics *metrics = nullptr;
        const IFrameStateView *state_view = nullptr;
        IFrameMutationAccess *mutation_access = nullptr;
    };

    struct CompositeStageInput {
        uint64_t frame_id = 0;
        RenderDataRD *render_data = nullptr;
        RenderSceneBuffersRD *render_buffers = nullptr;
        RID render_target;
        Size2i viewport_size;
        bool defer_commit = false;
        RasterStageOutput raster_output;
        StageMetrics *metrics = nullptr;
        const IFrameStateView *state_view = nullptr;
    };

public:
    PerformanceState performance_state;
    TestDataState test_data_state;

    // Phase 15: painterly texture RIDs removed - now managed by PainterlyMaterialManager

    // Tile-based renderer
    TileRendererState tile_renderer_state;

    void _update_tile_renderer_output_tracking(const RID &p_color_output, RenderingDevice *p_color_device,
            const RID &p_depth_output, RenderingDevice *p_depth_device);
    void _forget_tile_renderer_outputs();
    void _warn_tile_depth_copy_incompatible();

    // Modular interface subsystems (Phase 8 migration)
    // These will gradually replace the inline debug/interactive state management
    SubsystemState subsystem_state;

    std::unique_ptr<RenderPipelineStages> pipeline_stages;
    std::unique_ptr<RenderDeviceOrchestrator> device_orchestrator;
    std::unique_ptr<RenderDebugStateOrchestrator> debug_state_orchestrator;
    std::unique_ptr<RenderDiagnosticsOrchestrator> diagnostics_orchestrator;
    std::unique_ptr<RenderSortingOrchestrator> sorting_orchestrator;
    std::unique_ptr<RenderStreamingOrchestrator> streaming_orchestrator;
    std::unique_ptr<RenderQualityOrchestrator> quality_orchestrator;
    std::unique_ptr<RenderConfigOrchestrator> config_orchestrator;
    std::unique_ptr<RenderInstancingOrchestrator> instancing_orchestrator;
    std::unique_ptr<RenderResourceOrchestrator> resource_orchestrator;
    std::unique_ptr<RenderDataOrchestrator> data_orchestrator;
    std::unique_ptr<RenderOutputOrchestrator> output_orchestrator;
    std::unique_ptr<IRenderThreadDispatcher> render_thread_dispatcher;
    std::atomic<bool> teardown_resources_started{false};

    InstancePipelineBuffers instance_pipeline_buffers;
    bool instance_pipeline_buffers_valid = false;
    PipelineFeatureSet pipeline_features_effective;
    String pipeline_features_warning_cache;

    JacobianDebugConfig jacobian_debug;

    void _invalidate_static_chunk_caches(bool p_free_rids);
    bool _try_reuse_instance_sort_cache(const Transform3D &p_world_to_camera_transform, uint64_t p_content_generation,
            uint32_t p_max_visible_splats, uint32_t p_visible_chunk_count, uint32_t &r_sorted_count);
    void _update_instance_sort_cache(const Transform3D &p_world_to_camera_transform, uint64_t p_content_generation,
            uint32_t p_max_visible_splats, uint32_t p_visible_chunk_count, uint32_t p_sorted_count);

    // Phase 15: _copy_final_output_compute removed (dead code)

    // Removed: _generate_test_splats (dead code)
    Error _update_gpu_buffers_with_real_data(); // Replace test data with real PLY data
    void _initialize_shaders();
    void _initialize_painterly_pipeline();
    void _create_gpu_resources_safe();
    RenderingDevice *_get_submission_device();
    RenderingDevice *_peek_submission_device() const;
    bool _dispatch_call_on_render_thread_blocking(const Callable &p_callable, bool *r_dispatched = nullptr,
            bool p_allow_timeout = true, uint64_t *r_request_id = nullptr);
    void _notify_render_thread_dispatch_completed(uint64_t p_request_id);
    void _initialize_on_render_thread(uint64_t p_request_id);
    void _teardown_resources();
    void _teardown_on_render_thread(uint64_t p_request_id);
#ifdef TESTS_ENABLED
    void _test_dispatch_noop_callback(uint64_t p_request_id);
#endif
    void _set_max_splats_on_render_thread(int p_count, uint64_t p_request_id);
    void _set_gaussian_data_on_render_thread(const Ref<::GaussianData> &p_data, uint64_t p_request_id);
    void _force_sort_for_view_on_render_thread(const Transform3D &p_world_to_camera_transform, uint64_t p_request_id);
    RenderingDevice *_acquire_rendering_device();
    RenderingDevice *_get_main_rendering_device() const;
    bool _ensure_rendering_device(const char *p_context);
    bool _ensure_submission_device(const char *p_context);

    void _safe_submit_sync(RenderingDevice *p_device);
    void _track_resource_owner(const RID &p_rid, RenderingDevice *p_device, bool p_owned = true, const char *p_label = nullptr);
    RenderingDevice *_get_resource_owner(const RID &p_rid, RenderingDevice *p_fallback) const;
    RID _get_viewport_color_target(RenderSceneBuffersRD *p_render_buffers);
    RenderingDevice *_acquire_submission_device_for(RenderingDevice *p_device,
            GaussianSplatManager::ScopedSubmissionLock &r_lock) const;
    RenderingDevice *get_texture_owner_device(const RID &p_texture) const;
    RD::TextureFormat _get_texture_format(RenderingDevice *p_device, RID p_texture) const;
    void _set_manual_viewport_format(RD::DataFormat p_format, const char *p_context);
    void _set_active_viewport_format(RD::DataFormat p_format, const char *p_context);
    void _free_owned_resource(RenderingDevice *p_fallback_device, RID &p_rid);
    // Removed: _upload_test_splats_to_gpu (dead code)
    // Removed: _flush_depth_submission (dead code)
    void _update_gpu_pass_metrics_from_tile_renderer();
    void _prepare_render_frame_context(RenderDataRD *p_render_data, const Transform3D &p_world_to_camera_transform,
            const Projection &p_projection, const Projection &p_render_projection, bool p_defer_render_buffers_commit,
            RenderFrameContext &r_context);
    void _run_pipeline_entry(const RenderFrameContext &p_frame_context,
            bool p_has_render_data, const String &p_cull_skip_reason, const String &p_sort_skip_reason,
            RenderFallbackReason p_cull_skip_reason_code, RenderFallbackReason p_sort_skip_reason_code,
            bool p_set_skip_metrics, bool p_clear_cull_state_on_skip);
    void _run_cull_sort_pipeline_frame(RenderDataRD *p_render_data, const Transform3D &p_world_to_camera_transform,
            const Projection &p_projection, const Projection &p_render_projection, RenderSceneBuffersRD *p_render_buffers,
            bool p_has_render_data, const String &p_cull_skip_reason, const String &p_sort_skip_reason,
            RenderFallbackReason p_cull_skip_reason_code, RenderFallbackReason p_sort_skip_reason_code,
            bool p_set_skip_metrics, bool p_clear_cull_state_on_skip);
    void _reset_legacy_streaming_data_path_state();
    void _render_resident_frame(RenderDataRD *p_render_data, const Transform3D &p_world_to_camera_transform,
            const Projection &p_projection, const Projection &p_render_projection, RenderSceneBuffersRD *p_render_buffers);
    const Gaussian *_get_streamed_gaussian(uint32_t p_index) const;
    SortStageSummary sort_gaussians_for_view(const Transform3D &p_world_to_camera_transform,
            IndexDomain p_input_domain = IndexDomain::UNKNOWN);
    void render_sorted_splats(RenderDataRD *p_render_data, const Transform3D &p_world_to_camera_transform,
            const Projection &p_projection, const Projection &p_render_projection,
            bool p_defer_render_buffers_commit = false);
    void render_instanced(RenderDataRD *p_render_data, const GaussianSplatManager::SharedDynamicAssetHandle &p_handle,
            const Transform3D &p_world_to_camera_transform, const Projection &p_projection, const Projection &p_render_projection,
            const LocalVector<Transform3D> &p_instance_transforms);
    CullStageOutput _cull_for_view(const Transform3D &p_world_to_camera_transform, const Projection &p_projection, const Size2i &p_viewport_size);
    RID _get_painterly_depth_texture() const;
    RID _load_graphics_shader(const Vector<String> &p_vertex_paths, const Vector<String> &p_fragment_paths);
    void _on_painterly_material_changed();
    void _synchronize_tile_submission(RenderingDevice *p_device, const char *p_context);

    Ref<class OutputCompositor> shadow_output_compositor;
    uint64_t shadow_output_device_id = 0;
    ShadowBlitState shadow_blit_state;

    bool _ensure_shadow_output_compositor(RenderingDevice *p_device);
    bool _ensure_shadow_blit_resources(RenderingDevice *p_device);
    bool _blit_shadow_depth(RID p_source_depth, RID p_shadow_fb, const Rect2i &p_atlas_rect, bool p_flip_y);


    // Interactive state system methods
    void _release_shared_dynamic_asset();

protected:
    static void _bind_methods();
    void _notification(int p_what);

public:
    /**
     * @brief Constructs a new renderer instance.
     * @param p_device Optional RenderingDevice to use. If nullptr, the default device is used.
     */
    GaussianSplatRenderer(RenderingDevice *p_device = nullptr);
    ~GaussianSplatRenderer();

    /// @name Internal State Access
    /// @brief Accessors for subsystem orchestrators (internal use).
    /// @{
    SceneState &get_scene_state();
    const SceneState &get_scene_state() const;
    PipelineState &get_pipeline_state();
    const PipelineState &get_pipeline_state() const;
    PerformanceSettings &get_performance_settings();
    const PerformanceSettings &get_performance_settings() const;
    FrameState &get_frame_state() { return frame_context_manager.get_frame_state(); }
    const FrameState &get_frame_state() const { return frame_context_manager.get_frame_state(); }
    PerformanceState &get_performance_state() { return performance_state; }
    const PerformanceState &get_performance_state() const { return performance_state; }
    DeviceState &get_device_state();
    const DeviceState &get_device_state() const;
    RenderConfig &get_render_config();
    const RenderConfig &get_render_config() const;
    CullingConfig &get_culling_config();
    const CullingConfig &get_culling_config() const;
    PainterlyConfig &get_painterly_config();
    const PainterlyConfig &get_painterly_config() const;
    StreamingState &get_streaming_state();
    const StreamingState &get_streaming_state() const;
    ResourceState &get_resource_state();
    const ResourceState &get_resource_state() const;
    uint64_t get_instance_pipeline_content_generation() const { return get_resource_state().instance_pipeline_content_generation; }
    TestDataState &get_test_data_state() { return test_data_state; }
    const TestDataState &get_test_data_state() const { return test_data_state; }
    SortingState &get_sorting_state();
    ViewState &get_view_state() { return frame_context_manager.get_view_state(); }
    const ViewState &get_view_state() const { return frame_context_manager.get_view_state(); }
    TileRendererState &get_tile_renderer_state() { return tile_renderer_state; }
    InteractiveStateConfig &get_interactive_state_config();
    const InteractiveStateConfig &get_interactive_state_config() const;
    DebugConfig &get_debug_config();
    const DebugConfig &get_debug_config() const;
    DebugState &get_debug_state();
    const DebugState &get_debug_state() const;
    SubsystemState &get_subsystem_state() { return subsystem_state; }
    const SubsystemState &get_subsystem_state() const { return subsystem_state; }
    JacobianDebugConfig &get_jacobian_debug() { return jacobian_debug; }
    void set_instance_pipeline_buffers(const InstancePipelineBuffers &p_buffers);
    void clear_instance_pipeline_buffers();
    bool has_instance_pipeline_buffers() const { return instance_pipeline_buffers_valid; }
    const InstancePipelineBuffers &get_instance_pipeline_buffers() const { return instance_pipeline_buffers; }
    bool update_instance_buffer(LocalVector<InstanceDataGPU> &p_instances);

    bool ensure_rendering_device(const char *p_context) { return _ensure_rendering_device(p_context); }
    bool ensure_submission_device(const char *p_context) { return _ensure_submission_device(p_context); }
    RenderingDevice *get_submission_device() { return _get_submission_device(); }
    RenderingDevice *get_main_rendering_device() const { return _get_main_rendering_device(); }
    RenderingDevice *get_resource_owner(const RID &p_rid, RenderingDevice *p_fallback) const { return _get_resource_owner(p_rid, p_fallback); }
    void track_resource_owner(const RID &p_rid, RenderingDevice *p_device, bool p_owned = true, const char *p_label = nullptr) {
        _track_resource_owner(p_rid, p_device, p_owned, p_label);
    }
    void forget_resource_owner(const RID &p_rid);
    void free_owned_resource(RenderingDevice *p_fallback_device, RID &p_rid) { _free_owned_resource(p_fallback_device, p_rid); }
    void refresh_gpu_sorter(const char *p_context);
    void update_gpu_pass_metrics_from_tile_renderer() { _update_gpu_pass_metrics_from_tile_renderer(); }
    void update_debug_raster_metrics(const RasterPerformance &p_perf, const RasterStats &p_stats);
    void clear_debug_overlay_dirty_flags();
    const Gaussian *get_streamed_gaussian(uint32_t p_index) const { return _get_streamed_gaussian(p_index); }
    void apply_debug_options_to_render_params(TileRenderer::RenderParams &r_params) const;
    RID get_painterly_depth_texture() const { return _get_painterly_depth_texture(); }
    RID load_graphics_shader(const Vector<String> &p_vertex_paths, const Vector<String> &p_fragment_paths) {
        return _load_graphics_shader(p_vertex_paths, p_fragment_paths);
    }
    void synchronize_tile_submission(RenderingDevice *p_device, const char *p_context) { _synchronize_tile_submission(p_device, p_context); }
    bool ensure_sort_rendering_device(const char *p_context) override;
    RenderingDevice *get_sort_rendering_device() const override;
    SortExternalBufferState get_sort_external_buffer_state() const override;
    bool resize_sort_state_byte_vectors(uint32_t p_cpu_capacity, uint32_t p_key_stride_bytes, const char *p_context) override;
    void set_sort_buffer_binding_state(bool p_keys_external, bool p_indices_external,
            bool p_pipeline_managed, uint32_t p_capacity) override;
    void clear_sort_buffer_binding_state() override;
    void publish_sorted_indices(const SortPublicationPayload &p_payload) override;
    void forget_tile_renderer_outputs() { _forget_tile_renderer_outputs(); }
    void warn_tile_depth_copy_incompatible() { _warn_tile_depth_copy_incompatible(); }
    void update_tile_renderer_output_tracking(const RID &p_color_output, RenderingDevice *p_color_device,
            const RID &p_depth_output, RenderingDevice *p_depth_device) {
        _update_tile_renderer_output_tracking(p_color_output, p_color_device, p_depth_output, p_depth_device);
    }
    bool try_reuse_instance_sort_cache(const Transform3D &p_world_to_camera_transform, uint64_t p_content_generation,
            uint32_t p_max_visible_splats, uint32_t p_visible_chunk_count, uint32_t &r_sorted_count) {
        return _try_reuse_instance_sort_cache(p_world_to_camera_transform, p_content_generation, p_max_visible_splats,
                p_visible_chunk_count, r_sorted_count);
    }
    void update_instance_sort_cache(const Transform3D &p_world_to_camera_transform, uint64_t p_content_generation,
            uint32_t p_max_visible_splats, uint32_t p_visible_chunk_count, uint32_t p_sorted_count) {
        _update_instance_sort_cache(p_world_to_camera_transform, p_content_generation, p_max_visible_splats,
                p_visible_chunk_count, p_sorted_count);
    }
    void record_sort_sample(const SortFrameMetrics &p_sample);
    bool is_main_rendering_device(RenderingDevice *p_device) const;

    // Internal accessors for pipeline orchestrators (not user-facing API).
    RD::TextureFormat get_texture_format(RenderingDevice *p_device, RID p_texture) const { return _get_texture_format(p_device, p_texture); }
    void set_manual_viewport_format(RD::DataFormat p_format, const char *p_context) { _set_manual_viewport_format(p_format, p_context); }
    void set_active_viewport_format(RD::DataFormat p_format, const char *p_context) { _set_active_viewport_format(p_format, p_context); }
    void update_pipeline_features(RenderingDevice *p_device);
    CullStageOutput cull_for_view(const Transform3D &p_world_to_camera_transform, const Projection &p_projection, const Size2i &p_viewport_size) {
        return _cull_for_view(p_world_to_camera_transform, p_projection, p_viewport_size);
    }
    void reset_legacy_streaming_data_path_state() { _reset_legacy_streaming_data_path_state(); }
    void run_cull_sort_pipeline_frame(RenderDataRD *p_render_data, const Transform3D &p_world_to_camera_transform,
            const Projection &p_projection, const Projection &p_render_projection, RenderSceneBuffersRD *p_render_buffers,
            bool p_has_render_data, const String &p_cull_skip_reason, const String &p_sort_skip_reason,
            RenderFallbackReason p_cull_skip_reason_code, RenderFallbackReason p_sort_skip_reason_code,
            bool p_set_skip_metrics, bool p_clear_cull_state_on_skip) {
        _run_cull_sort_pipeline_frame(p_render_data, p_world_to_camera_transform, p_projection, p_render_projection,
                p_render_buffers, p_has_render_data, p_cull_skip_reason, p_sort_skip_reason,
                p_cull_skip_reason_code, p_sort_skip_reason_code, p_set_skip_metrics, p_clear_cull_state_on_skip);
    }

    /// @}

    /// @name Initialization
    /// @{

    /**
     * @brief Initializes the renderer and creates GPU resources.
     *
     * Must be called before rendering. Creates shader programs, allocates buffers,
     * and initializes the tile-based rendering pipeline.
     */
    void initialize() override;

    /**
     * @brief Initializes the GPU sorting subsystem.
     *
     * Called automatically by initialize(). Can be called separately if sorting
     * needs to be reinitialized after configuration changes.
     */
    void initialize_sorting();

    /// @}

    /// @name Data Management
    /// @{

    /**
     * @brief Sets the Gaussian data to render.
     * @param p_data GaussianData resource containing splat data.
     * @return OK on success, or an error code if data cannot be loaded.
     */
    Error set_gaussian_data(const Ref<::GaussianData> &p_data) override;

    /** @brief Returns the currently assigned Gaussian data. */
    Ref<::GaussianData> get_gaussian_data() const override { return get_scene_state().gaussian_data; }

    /**
     * @brief Sets a GaussianSplatAsset to render.
     * @param p_asset Asset containing pre-processed splat data.
     */
    void set_gaussian_asset(const Ref<GaussianSplatAsset> &p_asset) override;

    /** @brief Returns the currently assigned Gaussian asset. */
    Ref<GaussianSplatAsset> get_gaussian_asset() const { return get_scene_state().active_asset; }

    /// @}

    /// @name Rendering Configuration
    /// @{

    /**
     * @brief Sets the rendering mode (2D, 3D, or Hybrid).
     * @param p_mode Render mode selection.
     */
    void set_render_mode(RenderMode p_mode);

    /** @brief Returns the current render mode. */
    RenderMode get_render_mode() const { return get_render_config().render_mode; }

    /**
     * @brief Sets a global opacity multiplier for all splats.
     * @param p_opacity Multiplier in range [0, 1].
     */
    void set_opacity_multiplier(float p_opacity) override;
    float get_opacity_multiplier() const override { return get_render_config().opacity_multiplier; }

    /**
     * @brief Sets color grading resource for real-time color adjustments.
     * @param p_grading Color grading settings resource.
     */
    void set_color_grading(const Ref<class ColorGradingResource> &p_grading) override;
    Ref<class ColorGradingResource> get_color_grading() const { return get_render_config().color_grading; }

    /**
     * @brief Enables or disables static sort caching.
     * @param p_enabled When true, caches sort results for static cameras.
     */
    void set_static_sort_cache_enabled(bool p_enabled);

    /** @brief Returns true if static sort caching is enabled. */
    bool is_static_sort_cache_enabled() const { return subsystem_state.gpu_culler->get_state().static_sort_cache_enabled; }

    /** @brief Enables or disables cached render reuse in the output compositor. */
    void set_cached_render_reuse_enabled(bool p_enabled);

    /** @brief Returns true if cached render reuse is enabled. */
    bool is_cached_render_reuse_enabled() const;

    /** @brief Invalidates cached final-frame reuse state in the output compositor. */
    void invalidate_cached_render();

    /**
     * @brief Sets the painterly material for stylized rendering.
     * @param p_material Painterly material resource.
     */
    void set_painterly_material(const Ref<PainterlyMaterial> &p_material);

    /** @brief Returns the current painterly material. */
    Ref<PainterlyMaterial> get_painterly_material() const;

    /// @}

    /// @name LOD and Culling
    /// @{

    /**
     * @brief Enables or disables level-of-detail culling.
     * @param p_enabled When true, applies LOD-based splat culling.
     */
    void set_lod_enabled(bool p_enabled) override;

    /** @brief Returns true if LOD culling is enabled. */
    bool get_lod_enabled() const override { return subsystem_state.gpu_culler->get_config().lod_enabled; }

    /** @brief Sets LOD bias (higher values keep more detail at distance). */
    void set_lod_bias(float p_bias) override;
    float get_lod_bias() const override { return subsystem_state.gpu_culler->get_config().lod_bias; }

    /** @brief Sets minimum screen-space size in pixels before a splat is culled. */
    void set_lod_min_screen_size(float p_pixels);
    float get_lod_min_screen_size() const { return subsystem_state.gpu_culler->get_config().lod_min_screen_size; }

    /** @brief Sets maximum render distance for splats. */
    void set_lod_max_distance(float p_distance) override;
    float get_lod_max_distance() const override { return subsystem_state.gpu_culler->get_config().lod_max_distance; }

    /**
     * @brief Sets the importance threshold for splat culling.
     * @param p_threshold Minimum importance value in [0, 1] to keep a splat.
     */
    void set_importance_cull_threshold(float p_threshold);

    /** @brief Returns the importance cull threshold. */
    float get_importance_cull_threshold() const { return subsystem_state.gpu_culler->get_config().importance_cull_threshold; }

    /**
     * @brief Sets the radius multiplier for frustum culling bounds.
     * @param p_multiplier Multiplier applied to splat bounding radius.
     */
    void set_cull_radius_multiplier(float p_multiplier);

    /** @brief Returns the cull radius multiplier. */
    float get_cull_radius_multiplier() const { return subsystem_state.gpu_culler->get_config().cull_radius_multiplier; }

    /**
     * @brief Sets extra slack distance for frustum plane culling.
     * @param p_slack Additional distance in world units added to frustum planes.
     */
    void set_cull_frustum_plane_slack(float p_slack);

    /** @brief Returns the frustum plane slack distance. */
    float get_cull_frustum_plane_slack() const { return subsystem_state.gpu_culler->get_config().cull_frustum_plane_slack; }

    /**
     * @brief Sets the near plane tolerance for depth culling.
     * @param p_tolerance Distance in world units added to the near plane.
     */
    void set_cull_near_tolerance(float p_tolerance);

    /** @brief Returns the near plane tolerance. */
    float get_cull_near_tolerance() const { return subsystem_state.gpu_culler->get_config().cull_near_tolerance; }

    /**
     * @brief Sets the far plane tolerance for depth culling.
     * @param p_tolerance Distance in world units subtracted from the far plane.
     */
    void set_cull_far_tolerance(float p_tolerance);

    /**
     * @brief Updates the near/far depth range for culling.
     * @param p_near Near plane distance.
     * @param p_far Far plane distance.
     * @note Called automatically by the engine's scene renderer.
     */
    void update_depth_range(float p_near, float p_far);
    float get_cull_far_tolerance() const { return subsystem_state.gpu_culler->get_config().cull_far_tolerance; }

    /**
     * @brief Sets the minimum screen-space radius for tiny splat culling.
     * @param p_pixels Minimum radius in pixels; splats smaller than this are culled.
     */
    void set_tiny_splat_screen_radius(float p_pixels);

    /** @brief Returns the tiny splat screen radius threshold. */
    float get_tiny_splat_screen_radius() const { return subsystem_state.gpu_culler->get_state().tiny_splat_screen_radius_px; }

    /**
     * @brief Enables or disables opacity-aware bounding (FlashGS optimization).
     * @param p_enabled When true, splat radii are computed based on opacity, reducing tile-Gaussian pairs by ~94%.
     */
    void set_opacity_aware_culling(bool p_enabled);

    /** @brief Returns true if opacity-aware culling is enabled. */
    bool is_opacity_aware_culling() const { return subsystem_state.gpu_culler->get_config().opacity_aware_culling; }

    /**
     * @brief Sets the visibility threshold (tau) for opacity-aware culling.
     * @param p_threshold Minimum visible contribution; typical range 0.001-0.05, default 0.01.
     */
    void set_visibility_threshold(float p_threshold);

    /** @brief Returns the visibility threshold for opacity-aware culling. */
    float get_visibility_threshold() const { return subsystem_state.gpu_culler->get_config().visibility_threshold; }

    /**
     * @brief Enables or disables distance-based probabilistic culling during tile binning.
     * @param p_enabled When true, splats beyond the start distance are probabilistically culled.
     */
    void set_distance_cull_enabled(bool p_enabled);

    /** @brief Returns true if distance-based culling is enabled. */
    bool is_distance_cull_enabled() const { return subsystem_state.gpu_culler->get_config().distance_cull_enabled; }

    /**
     * @brief Sets the distance (world units) where distance-based culling starts ramping.
     * @param p_distance Start distance in world units.
     */
    void set_distance_cull_start(float p_distance);

    /** @brief Returns the distance where distance-based culling starts. */
    float get_distance_cull_start() const { return subsystem_state.gpu_culler->get_config().distance_cull_start; }

    /**
     * @brief Sets the maximum cull probability at far distances (0-1).
     * @param p_rate Maximum cull rate.
     */
    void set_distance_cull_max_rate(float p_rate);

    /** @brief Returns the maximum distance-based cull rate. */
    float get_distance_cull_max_rate() const { return subsystem_state.gpu_culler->get_config().distance_cull_max_rate; }

    /**
     * @brief Enables or disables automatic overflow tuning.
     * @param p_enabled When true, dynamically adjusts culling to prevent tile overflow.
     */
    void set_overflow_autotune_enabled(bool p_enabled);

    /** @brief Returns true if overflow auto-tuning is enabled. */
    bool is_overflow_autotune_enabled() const { return subsystem_state.gpu_culler->get_state().overflow_autotune_enabled; }

    /**
     * @brief Sets the maximum number of splats to render per frame.
     * @param p_count Maximum splat count (default 2,000,000).
     */
    void set_max_splats(int p_count) override;
    int get_max_splats() const override { return get_performance_settings().max_splats; }

    /**
     * @brief Enables or disables frustum culling.
     * @param p_enabled When true, splats outside the view frustum are culled.
     */
    void set_frustum_culling(bool p_enabled) override;

    /** @brief Returns true if frustum culling is enabled. */
    bool get_frustum_culling() const override { return subsystem_state.gpu_culler->get_config().frustum_culling; }

    /**
     * @brief Enables or disables async GPU uploads for streaming.
     * @param p_enabled When true, uses async transfers for better performance.
     *
     * This is typically set based on the "optimize_for_gpu" import setting.
     * When disabled, uploads are synchronous which may reduce throughput but
     * can be useful for debugging or on systems with limited async transfer support.
     */
    void set_async_upload_enabled(bool p_enabled) override;

    /** @brief Returns true if async GPU uploads are enabled. */
    bool get_async_upload_enabled() const override;

    /// @}

    /// @name Painterly Rendering
    /// @brief Controls for stylized brush-stroke rendering effects.
    /// @{

    /**
     * @brief Enables or disables painterly rendering mode.
     * @param p_enabled When true, applies brush-stroke post-processing.
     */
    void set_painterly_enabled(bool p_enabled) override;

    /** @brief Returns true if painterly rendering is enabled. */
    bool get_painterly_enabled() const override { return get_painterly_config().enabled; }

    /**
     * @brief Enables low-end mode for painterly rendering.
     * @param p_enabled When true, uses simplified effects for better performance.
     */
    void set_painterly_low_end_mode(bool p_enabled);

    /** @brief Returns true if painterly low-end mode is enabled. */
    bool get_painterly_low_end_mode() const { return get_painterly_config().low_end_mode; }

    /**
     * @brief Enables or disables brush stroke overlay.
     * @param p_enabled When true, renders visible brush strokes.
     */
    void set_painterly_enable_strokes(bool p_enabled);

    /** @brief Returns true if brush strokes are enabled. */
    bool get_painterly_enable_strokes() const { return get_painterly_config().enable_strokes; }

    /**
     * @brief Sets the internal rendering scale for painterly effects.
     * @param p_scale Scale factor (1.0 = full resolution, 0.5 = half resolution).
     */
    void set_painterly_internal_scale(float p_scale);

    /** @brief Returns the painterly internal scale. */
    float get_painterly_internal_scale() const { return get_painterly_config().internal_scale; }

    /**
     * @brief Sets the edge detection threshold for painterly outlines.
     * @param p_threshold Threshold in [0, 1]; lower values detect more edges.
     */
    void set_painterly_edge_threshold(float p_threshold) override;

    /** @brief Returns the painterly edge threshold. */
    float get_painterly_edge_threshold() const override { return get_painterly_config().edge_threshold; }

    /**
     * @brief Sets the edge intensity for painterly outlines.
     * @param p_intensity Multiplier for edge visibility.
     */
    void set_painterly_edge_intensity(float p_intensity);

    /** @brief Returns the painterly edge intensity. */
    float get_painterly_edge_intensity() const { return get_painterly_config().edge_intensity; }

    /**
     * @brief Sets the maximum stroke length in pixels.
     * @param p_length Stroke length in screen pixels.
     */
    void set_painterly_stroke_length(float p_length) override;

    /** @brief Returns the painterly stroke length. */
    float get_painterly_stroke_length() const override { return get_painterly_config().stroke_length; }

    /**
     * @brief Sets the opacity of painterly brush strokes.
     * @param p_opacity Opacity in [0, 1].
     */
    void set_painterly_stroke_opacity(float p_opacity) override;

    /** @brief Returns the painterly stroke opacity. */
    float get_painterly_stroke_opacity() const override { return get_painterly_config().stroke_opacity; }

    /**
     * @brief Sets the gamma correction for painterly output.
     * @param p_gamma Gamma value (typically 2.2 for sRGB).
     */
    void set_painterly_gamma(float p_gamma) override;

    /** @brief Returns the painterly gamma value. */
    float get_painterly_gamma() const override { return get_painterly_config().gamma; }

    /// @}

    /// @name Quality Presets
    /// @{

    /**
     * @brief Applies a named quality preset configuration.
     * @param p_preset Accepted values (case-insensitive):
     *   - "ultra", "quality", "high" - High quality (max splats, LOD bias 0.8)
     *   - "balanced", "medium" - Medium quality (LOD bias 1.0)
     *   - "performance", "low" - Low quality (LOD bias 1.5)
     *
     * @note get_quality_preset() returns normalized names: "ultra", "high", "medium", "low"
     */
    void set_quality_preset(const String &p_preset);
    String get_quality_preset() const;

    /// @}

    /// @name Debug and Benchmarking
    /// @{

    /** @brief Runs sorting performance benchmarks and logs results. */
    void benchmark_sorting_performance();

    /// @}

    /// @name Interactive State
    /// @brief Editor selection and hover visual feedback.
    /// @{

    /**
     * @brief Sets the interactive visual state for editor feedback.
     * @param p_state One of STATE_NORMAL, STATE_HOVERED, STATE_SELECTED, or STATE_DISABLED.
     */
    void set_interactive_state(InteractiveState p_state);

    /** @brief Returns the current interactive state. */
    InteractiveState get_interactive_state() const { return get_interactive_state_config().current_state; }

    /**
     * @brief Enables a highlight effect on the rendered splats.
     * @param p_color Highlight tint color.
     */
    void enable_highlight_effect(const Color &p_color);

    /**
     * @brief Enables an outline effect around rendered splats.
     * @param p_color Outline color.
     * @param p_width Outline width in pixels.
     */
    void enable_outline_effect(const Color &p_color, float p_width);

    /** @brief Removes all interactive visual effects. */
    void remove_visual_effects();

    /// @}

    // Debug overlay controls for projection verification (Issue #125)
    // Getters delegate to DebugOverlaySystem (Phase 8 migration)
    void set_debug_show_tile_bounds(bool p_enabled);
    bool get_debug_show_tile_bounds() const;
    void set_debug_show_splat_coverage(bool p_enabled);
    bool get_debug_show_splat_coverage() const;
    void set_debug_show_overflow_tiles(bool p_enabled);
    bool get_debug_show_overflow_tiles() const;
    void set_debug_show_projection_issues(bool p_enabled);
    bool get_debug_show_projection_issues() const;
    void set_debug_show_white_albedo(bool p_enabled);
    bool get_debug_show_white_albedo() const;
    void set_debug_show_shadow_opacity(bool p_enabled);
    bool get_debug_show_shadow_opacity() const;
    void set_debug_dump_gpu_counters(bool p_enabled);
    bool get_debug_dump_gpu_counters() const;
    void set_debug_binning_counters_enabled(bool p_enabled);
    bool get_debug_binning_counters_enabled() const;
    void set_debug_pipeline_trace_enabled(bool p_enabled);
    bool get_debug_pipeline_trace_enabled() const;
    void set_debug_state_guardrails_enabled(bool p_enabled);
    bool get_debug_state_guardrails_enabled() const;
    void set_debug_cull_guardrails_enabled(bool p_enabled);
    bool get_debug_cull_guardrails_enabled() const;
    void set_debug_splat_audit_enabled(bool p_enabled);
    bool get_debug_splat_audit_enabled() const;
    void set_debug_splat_audit_sample_count(int p_count);
    int get_debug_splat_audit_sample_count() const;
    void set_debug_overlay_opacity(float p_opacity) override;
    float get_debug_overlay_opacity() const override { return get_debug_config().overlay_opacity; }
    void set_solid_coverage_enabled(bool p_enabled);
    bool is_solid_coverage_enabled() const { return get_culling_config().solid_coverage_enabled; }
    void set_solid_coverage_alpha_floor(float p_alpha);
    float get_solid_coverage_alpha_floor() const { return get_culling_config().solid_coverage_alpha_floor; }
    void set_debug_show_resolve_input(bool p_enabled);
    bool get_debug_show_resolve_input() const;
    void set_debug_show_resolve_output(bool p_enabled);
    bool get_debug_show_resolve_output() const;

    void reload_pipeline_feature_set();

    // Jacobian diagnostic toggles for radial stretching investigation
    void set_jacobian_bypass_radius_depth_floor(bool p_enabled);
    bool get_jacobian_bypass_radius_depth_floor() const { return jacobian_debug.bypass_radius_depth_floor; }
    void set_jacobian_bypass_j_col2_clamp(bool p_enabled);
    bool get_jacobian_bypass_j_col2_clamp() const { return jacobian_debug.bypass_j_col2_clamp; }
    void set_jacobian_invert_j_col2_sign(bool p_enabled);
    bool get_jacobian_invert_j_col2_sign() const { return jacobian_debug.invert_j_col2_sign; }
    void set_max_conic_aspect(float p_aspect);
    float get_max_conic_aspect() const { return jacobian_debug.max_conic_aspect; }

    /// @name Performance Monitoring
    /// @{

    /**
     * @brief Read-only snapshot consumed by performance monitor callbacks.
     *
     * The snapshot intentionally flattens streaming-system query data so
     * monitor consumers avoid broad direct access to renderer mutable state.
     */
    struct MonitorStreamingSnapshot {
        bool has_streaming_system = false;
        bool has_streaming_data = false;
        bool runtime_ready = false;
        Dictionary streaming_analytics;
        Dictionary vram_debug_stats;
        Dictionary chunk_culling_stats;
        Dictionary lod_debug_stats;
        String route_uid;
        String sort_route_uid;
        bool stage_metrics_valid = false;
        float stage_cull_time_ms = 0.0f;
        float stage_sort_time_ms = 0.0f;
        float frame_sort_time_ms = 0.0f;
        float metrics_gpu_frame_time_ms = 0.0f;
        float metrics_culling_time_ms = 0.0f;
        float metrics_gpu_tile_binning_time_ms = 0.0f;
        float metrics_gpu_tile_prefix_time_ms = 0.0f;
        float metrics_gpu_tile_raster_time_ms = 0.0f;
        float metrics_gpu_tile_resolve_time_ms = 0.0f;
        uint32_t metrics_visible_after_culling = 0;
        uint32_t metrics_rendered_splat_count = 0;
        uint32_t frame_visible_splat_count = 0;
        uint64_t vram_usage_bytes = 0;
        uint32_t chunks_loaded_this_frame = 0;
        uint32_t chunks_evicted_this_frame = 0;
        uint32_t visible_splat_count = 0;
        uint32_t buffer_capacity_splats = 0;
        uint32_t effective_splat_count = 0;
        float visible_chunk_change_ratio = 0.0f;
        float global_lod_blend_factor = 0.0f;
        int global_sh_band_level = 0;
        float lod_hysteresis_zone = 0.0f;
        float lod_blend_distance = 0.0f;
        float lod_distance_multiplier = 1.0f;
        bool memory_stream_valid = false;
        StreamingStats memory_stream_stats;
        bool sh_compression_metrics_valid = false;
        SHCompressionMetrics sh_compression_metrics;
    };

    /**
     * @brief Builds a read-only streaming/LOD monitor snapshot for tooling.
     */
    MonitorStreamingSnapshot get_monitor_streaming_snapshot() const;

    /**
     * @brief Returns comprehensive rendering statistics.
     * @return Dictionary containing timing, memory, and splat count metrics.
     */
    Dictionary get_render_stats() const override;

    /**
     * @brief Returns debug counters from the tile binning pass.
     * @return Dictionary with projection success rates and rejection stats.
     */
    Dictionary get_binning_debug_counters() const;

    /**
     * @brief Returns metrics from the last sorting pass.
     * @return Dictionary with timing and algorithm info.
     */
    Dictionary get_last_sort_metrics() const;

    /**
     * @brief Returns history of sort metrics for performance analysis.
     * @return Array of SortFrameMetrics dictionaries.
     */
    Array get_sort_metrics_history() const;

    /**
     * @brief Runs sorting benchmarks with various element counts.
     * @param p_sizes Array of element counts to benchmark.
     * @return Array of benchmark results.
     */
    Array run_sort_benchmark(const PackedInt32Array &p_sizes);

    /** @brief Returns the last frame's sorting time in milliseconds. */
    float get_sort_time_ms() const;

    /** @brief Returns the last frame's total render time in milliseconds. */
    float get_render_time_ms() const;

    /** @brief Returns the number of visible splats after culling. */
    uint32_t get_visible_splat_count() const override { return get_frame_state().visible_splat_count.load(std::memory_order_acquire); }

    /**
     * @brief Returns cached overflow tile count from the raster pipeline.
     * @note This is cached data; call get_overflow_stats() to request a fresh readback.
     */
    int get_overflow_tile_count() const;

    /**
     * @brief Returns cached count of clamped overlap records.
     * @note This is cached data; call get_overflow_stats() to request a fresh readback.
     */
    int get_clamped_records() const;

    /**
     * @brief Returns cached count of aggregated overlap records (total overlaps emitted).
     * @note This is cached data; call get_overflow_stats() to request a fresh readback.
     */
    int get_aggregated_count() const;

    /**
     * @brief Returns a dictionary of overflow stats, triggering a GPU readback.
     * @note This may return the previous frame's data if a readback is pending.
     */
    Dictionary get_overflow_stats() const;

    /**
     * @brief Forces a depth sort for the given view transform.
     * @param p_world_to_camera_transform World-to-camera (view) matrix for depth calculation.
     * @note Bypasses sort caching and always performs a fresh sort.
     */
    void force_sort_for_view(const Transform3D &p_world_to_camera_transform);

    /// @}

    /// @name Engine Integration
    /// @brief Methods called by RendererSceneRenderRD for scene rendering.
    /// @{

    /**
     * @brief Renders Gaussian splats for a scene instance.
     * @param p_render_data Render data from the scene renderer.
     * @note Called automatically by the engine's rendering pipeline.
     */
    void render_scene_instance(RenderDataRD *p_render_data);

    /**
     * @brief Renders Gaussian splats for multiple instances.
     * @param p_render_data Render data from the scene renderer.
     * @param p_instances Array of instance RIDs to render.
     * @note Called by the engine's scene rendering pipeline.
     */
    void render_gaussians(RenderDataRD *p_render_data, const PagedArray<RID> &p_instances);

    /**
     * @brief Renders a directional shadow map into the light's shadow atlas.
     * @param p_light_projection Light projection matrix for the shadow pass.
     * @param p_light_transform Light camera transform (camera-to-world).
     * @param p_atlas_rect Target atlas rectangle for this cascade.
     * @param p_shadow_framebuffer Shadow atlas framebuffer RID.
     * @param p_flip_y Whether the atlas rect should be sampled with a Y flip.
     * @return True if shadow depth was rendered and blitted.
     */
    bool render_directional_shadow_map(const Projection &p_light_projection, const Transform3D &p_light_transform,
            const Rect2i &p_atlas_rect, RID p_shadow_framebuffer, bool p_flip_y);

    /**
     * @brief Renders splats for a specific camera view.
     * @param p_world_to_camera_transform World-to-camera (view) matrix.
     * @param p_cam_projection Camera projection matrix.
     * @param p_render_target Target RID for output.
     * @param p_viewport_size Viewport dimensions in pixels.
     * @return True if rendering was performed (data available and device ready),
     *         false if no gaussian data loaded or rendering device unavailable.
     */
    bool render_for_view(const Transform3D &p_world_to_camera_transform, const Projection &p_cam_projection, RID p_render_target, const Size2i &p_viewport_size) override;

    /**
     * @brief Copies the final rendered texture to a target RID.
     * @param p_render_target Destination render target RID.
     * @param p_viewport_size Viewport dimensions for the copy.
     * @return True if the copy succeeded.
     */
    bool copy_final_texture_to_target(RID p_render_target, const Size2i &p_viewport_size);

    /**
     * @brief Commits rendered content to the scene render buffers.
     * @param p_render_data Render data containing the target buffers.
     */
    void commit_to_render_buffers(RenderDataRD *p_render_data);

    /// @}

    /// @name Camera State
    /// @brief Camera state used by GaussianSplatNode3D for render_scene_instance.
    /// @{

    /**
     * @brief Sets the camera transform for the next render pass.
     * @param p_transform Camera world transform.
     */
    void set_camera_transform(const Transform3D &p_transform) override { get_view_state().last_camera_to_world_transform = p_transform; }

    /**
     * @brief Sets the camera projection for the next render pass.
     * @param p_projection Camera projection matrix.
     */
    void set_camera_projection(const Projection &p_projection) override { get_view_state().last_camera_projection = p_projection; }

    /**
     * @brief Updates the streaming system without rendering.
     * @param p_camera_to_world_transform Camera world transform.
     * @param p_projection Camera projection matrix.
     */
    void tick_streaming_only(const Transform3D &p_camera_to_world_transform, const Projection &p_projection);
    Projection build_cull_projection(RenderDataRD *p_render_data, const Projection &p_projection) const;
    bool validate_cull_projection_contract(RenderDataRD *p_render_data, const Projection &p_projection,
            const Projection &p_cull_projection, const char *p_context);

    /** @brief Returns the last camera transform. */
    Transform3D get_camera_transform() const override { return get_view_state().last_camera_to_world_transform; }

    /** @brief Returns the last camera projection. */
    Projection get_camera_projection() const { return get_view_state().last_camera_projection; }

    /// @}

    /// @name Output Access
    /// @{

    /** @brief Returns the RID of the final rendered color texture. */
    RID get_final_texture() const override;

    /** @brief Returns true if there is valid rendered content available. */
    bool has_rendered_content() const override;

    /** @brief Returns the internal tile renderer for advanced debugging. */
    Ref<TileRenderer> get_tile_renderer() const { return tile_renderer_state.renderer; }

    /** @brief Returns true if the last viewport copy operation succeeded. */
    bool was_last_viewport_copy_successful() const;

    /** @brief Returns the source size of the last viewport copy. */
    Size2i get_last_viewport_copy_source_size() const;

    /** @brief Returns the destination size of the last viewport copy. */
    Size2i get_last_viewport_copy_dest_size() const;

    /// @}

    void set_debug_show_tile_grid(bool p_enabled) override;
    bool is_debug_show_tile_grid() const override { return get_debug_state().show_tile_grid; }
    void set_debug_show_density_heatmap(bool p_enabled) override;
    bool is_debug_show_density_heatmap() const override { return get_debug_state().show_density_heatmap; }
    void set_debug_show_performance_hud(bool p_enabled) override;
    bool is_debug_show_performance_hud() const override { return get_debug_state().show_performance_hud; }
    void set_debug_show_residency_hud(bool p_enabled) override;
    bool is_debug_show_residency_hud() const override { return get_debug_state().show_residency_hud; }
    void set_debug_show_device_boundaries(bool p_enabled);
    bool is_debug_show_device_boundaries() const { return get_debug_state().show_device_boundaries; }
    void set_debug_show_texture_states(bool p_enabled);
    bool is_debug_show_texture_states() const { return get_debug_state().show_texture_states; }
    void set_debug_compute_raster_policy(int p_policy);
    int get_debug_compute_raster_policy() const;

    /**
     * @brief Returns a comprehensive runtime diagnostics snapshot.
     * @return Dictionary with error stats, timing history, and device info.
     */
    Dictionary get_runtime_diagnostic_snapshot() const;

    /**
     * @brief Returns the last pipeline trace snapshot (events + stage IO) for debugging.
     * @return Dictionary with pipeline events and stage IO summaries.
     */
    Dictionary get_pipeline_trace_snapshot() const;

    /**
     * @brief Returns the pipeline trace snapshot serialized to JSON.
     * @return JSON string containing pipeline events and stage IO summaries.
     */
    String get_pipeline_trace_json() const;

    /**
     * @brief Writes the pipeline trace JSON to a file path.
     * @param p_path Destination file path (e.g. user://pipeline_trace.json).
     * @return Error code indicating success or failure.
     */
    Error dump_pipeline_trace_to_file(const String &p_path) const;

    /**
     * @brief Sets the debug preview visualization mode.
     * @param p_mode One of DEBUG_PREVIEW_OFF, DEBUG_PREVIEW_WIREFRAME, etc.
     */
    void set_debug_preview_mode(DebugPreviewMode p_mode);
    /** @brief IRenderer override - sets debug preview mode using int. */
    void set_debug_preview_mode_int(int p_mode) override { set_debug_preview_mode(static_cast<DebugPreviewMode>(p_mode)); }

    /** @brief Returns the current debug preview mode. */
    DebugPreviewMode get_debug_preview_mode() const { return get_debug_state().preview_mode; }
    /** @brief IRenderer override - returns debug preview mode as int. */
    int get_debug_preview_mode_int() const override { return static_cast<int>(get_debug_state().preview_mode); }

    /**
     * @brief Sets static chunks for hierarchical streaming.
     * @param p_chunks Vector of static chunk definitions.
     */
    void set_static_chunks(const Vector<StaticChunk> &p_chunks) override;
    void set_streaming_config_overrides(const GaussianStreamingSystem::ConfigOverrides &p_overrides);

    /** @brief Clears all static chunk definitions. */
    void clear_static_chunks();

    /** @brief Returns the current static chunk definitions. */
    const Vector<StaticChunk> &get_static_chunks() const { return subsystem_state.gpu_culler->get_state().static_chunks; }

// Test helper methods (always available to avoid header guard issues with Godot's test framework)
    void test_override_rendering_device(RenderingDevice *p_device);
    void test_disable_gpu_culler();
    void test_disable_rasterizer();
    bool test_copy_final_output(RID p_source, RID p_destination, const Size2i &p_viewport_size);
    void test_force_disable_streaming();
    void test_release_current_streaming_system();
    bool test_has_current_streaming_system() const;
    Ref<OutputCompositor> test_get_output_compositor() const;
    void test_set_test_splats(const Vector<Vector3> &p_positions, const Vector<Vector3> &p_scales = Vector<Vector3>());
    int test_cull_visible_count(const Transform3D &p_world_to_camera_transform, const Projection &p_projection, const Size2i &p_viewport_size);
    void test_set_render_thread_dispatch_timeout_usec(uint64_t p_timeout_usec);
    uint64_t test_get_render_thread_dispatch_timeout_usec() const;
    bool test_dispatch_call_on_render_thread_blocking_without_completion();
    bool test_dispatch_call_on_render_thread_blocking_with_completion();
    void test_notify_render_thread_dispatch_completed(uint64_t p_request_id);
    uint64_t test_get_render_thread_dispatch_completed_request_id() const;

    /**
     * @brief Returns the axis-aligned bounding box of the Gaussian data.
     * @return AABB encompassing all loaded splats.
     */
    AABB get_aabb() const;
};

VARIANT_ENUM_CAST(GaussianSplatRenderer::RenderMode);
VARIANT_ENUM_CAST(GaussianSplatRenderer::InteractiveState);
VARIANT_ENUM_CAST(GaussianSplatRenderer::DebugPreviewMode);

#endif // GAUSSIAN_SPLAT_RENDERER_H
