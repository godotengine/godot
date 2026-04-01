/**
 * @file gaussian_splat_node_3d.h
 * @brief Scene node for Gaussian Splatting visualization.
 *
 * This file defines GaussianSplatNode3D, the primary scene node for adding
 * Gaussian splat content to a Godot scene. It handles asset loading, viewport
 * management, quality settings, and integration with the rendering pipeline.
 */

#ifndef GAUSSIAN_SPLAT_NODE_3D_H
#define GAUSSIAN_SPLAT_NODE_3D_H

#include "scene/3d/node_3d.h"
#include "../core/gaussian_splat_asset.h"
#include "../core/painterly_manager.h"
#include "../lod/adaptive_lod_system.h"
#include "../lod/streaming_lod_manager.h"
#include "core/math/aabb.h"
#include "core/math/vector2i.h"
#include "core/object/object_id.h"
#include "core/templates/rid.h"
#include "core/variant/typed_array.h"
#include "core/variant/variant.h"
#include "gaussian_splat_node_helpers.h"


class Node;
class GaussianData;
class GaussianSplatRenderer;
class Camera3D;
class Control;
class CanvasLayer;
class Viewport;
class ViewportTexture;
class GaussianSplatDebugHUD;

/**
 * @class GaussianSplatNode3D
 * @brief 3D scene node for displaying Gaussian splat content.
 *
 * GaussianSplatNode3D is the main scene node for integrating Gaussian Splatting
 * into Godot projects. It provides:
 *
 * - **Asset Loading**: Load .ply or .spz files via file path or GaussianSplatAsset resource
 * - **Quality Presets**: Performance, Balanced, Quality, and Custom modes
 * - **LOD Control**: Level-of-detail bias, max distance, and splat count limits
 * - **Painterly Effects**: Optional stylized brush-stroke rendering
 * - **Editor Integration**: Preview rendering, debug overlays, and drag-drop support
 *
 * ## Supported Formats
 *
 * - **PLY**: Standard Gaussian Splatting format (ASCII or binary)
 * - **SPZ**: Niantic compressed format (~10x smaller files)
 *
 * ## Basic Usage
 *
 * In the editor:
 * 1. Add a GaussianSplatNode3D to your scene
 * 2. Set the file path (.ply or .spz) or assign a "Splat Asset"
 * 3. Adjust quality preset and rendering options as needed
 *
 * From GDScript:
 * @code
 * var splat_node = GaussianSplatNode3D.new()
 * splat_node.ply_file_path = "res://my_scan.ply"  // Also accepts .spz files
 * splat_node.quality_preset = GaussianSplatNode3D.QUALITY_BALANCED
 * add_child(splat_node)
 * @endcode
 *
 * @note The property is named `ply_file_path` for historical reasons but accepts
 *       both PLY and SPZ formats.
 * @note This node participates in the shared scene renderer registry and does not
 *       own a dedicated GaussianSplatRenderer instance.
 */
class GaussianSplatNode3D : public Node3D {
    GDCLASS(GaussianSplatNode3D, Node3D);

public:
    /**
     * @enum QualityPreset
     * @brief Predefined quality/performance tradeoff configurations.
     */
    enum QualityPreset {
        QUALITY_PERFORMANCE,
        QUALITY_BALANCED,
        QUALITY_QUALITY,
        QUALITY_CUSTOM      ///< Manual configuration of all quality settings.
    };

    /**
     * @enum ViewportUpdateMode
     * @brief Controls when the splat rendering is updated.
     */
    enum ViewportUpdateMode {
        UPDATE_MODE_ALWAYS,
        UPDATE_MODE_WHEN_VISIBLE,
        UPDATE_MODE_WHEN_PARENT_VISIBLE,
        UPDATE_MODE_MANUAL  ///< Only update when update_splats() is called.
    };

    /**
     * @enum DebugDrawMode
     * @brief Debug visualization modes for development.
     */
    enum DebugDrawMode {
        DEBUG_DRAW_OFF = 0,
        DEBUG_DRAW_WIREFRAME,
        DEBUG_DRAW_POINTS,
        DEBUG_DRAW_HEATMAP
    };

private:
    // These helper classes are value-member subsystems that decompose this node's
    // implementation across multiple files (gaussian_splat_node_helpers.cpp). They
    // access 50+ private fields and methods as part of the node's own logic. Exposing
    // all accessed state through public accessors would create a larger encapsulation
    // surface than friendship. Kept intentionally as tightly-coupled decomposition.
    friend class GaussianSplatNodeAssetHelper;
    friend class GaussianSplatNodeViewportHelper;
    friend class GaussianSplatNodeDebugHelper;
    friend class GaussianSplatNodeQualityHelper;
    friend class GaussianSplatNodeVisibilityHelper;
    friend class GaussianSplatNodeRendererHelper;

    // Asset management
    String ply_file_path;
    Ref<GaussianSplatAsset> splat_asset;
    Ref<GaussianSplatAsset> runtime_asset;
    bool auto_load = true;
    bool asset_loading = false;

    // Quality settings
    QualityPreset quality_preset = QUALITY_BALANCED;
    float lod_bias = 1.0f;
    float max_render_distance = 1000.0f;
    int max_splat_count = 1000000;

    // Asset-level import settings (read from asset metadata)
    bool asset_lod_enabled = true;         // From import metadata "enable_lod"
    bool asset_optimize_for_gpu = true;    // From import metadata "optimize_for_gpu"

    GaussianSplatting::AdaptiveLODSystem::LODConfig lod_config;
    GaussianSplatting::StreamingLODManager::StreamingConfig streaming_config;

    // Painterly settings
    bool enable_painterly = false;
    float edge_threshold = 0.1f;
    float stroke_opacity = 0.8f;
    float stroke_width = 1.0f;
    float color_variation = 0.05f; // Legacy serialized compatibility only (not an exposed node property).
    float temporal_blend = 0.9f;
    uint32_t painterly_seed = 1337;
    GaussianSplatting::PainterlyManager painterly_manager;

    // Rendering settings
    ViewportUpdateMode update_mode = UPDATE_MODE_WHEN_VISIBLE;
    bool cast_shadow = false;
    bool use_frustum_culling = true;
    bool use_occlusion_culling = true; // Legacy serialized compatibility only (not an exposed node property).
    float opacity = 1.0f;
    bool wind_override_enabled = false;
    bool wind_enabled = true;
    float wind_strength = 1.0f;
    Vector3 wind_direction = Vector3();
    float wind_frequency = 1.0f;

    // Color grading
    Ref<class ColorGradingResource> color_grading;

    // Performance monitoring
    uint32_t visible_splat_count = 0;
    uint32_t total_splat_count = 0;
    float last_update_time_ms = 0.0f;
    float gpu_memory_mb = 0.0f;
    uint64_t last_stats_inspector_refresh_usec = 0;

    // Bounds and visibility
    AABB local_aabb;
    AABB world_aabb;
    bool bounds_dirty = true;

    // Rendering resources
    RID render_instance;
    RID gaussian_base;
    bool visible_in_viewport = false;
    bool parent_visible = true;
    Node *parent_visibility_target = nullptr;

    enum class ViewportTextureState {
        INACTIVE,
        WAITING_FOR_TEXTURE,
        READY
    };

    ObjectID cached_viewport_id = ObjectID();
    RID cached_viewport_render_target;
    RID cached_viewport_render_texture;
    Ref<ViewportTexture> cached_viewport_texture;
    Viewport *observed_viewport = nullptr;
    ViewportTextureState viewport_texture_state = ViewportTextureState::INACTIVE;
    Vector2i cached_viewport_size = Vector2i();
    bool viewport_bootstrap_deferred = false;
    bool first_frame_render_deferred = false;
    bool viewport_texture_missing_reported = false;

    Ref<GaussianSplatRenderer> renderer;
    Ref<::GaussianData> renderer_data;
    bool render_state_dirty = true;
    bool shared_renderer_multi_instance_state = false;

    // Editor state
    bool preview_enabled = true;
    bool show_bounds = false;
    bool show_statistics = false;
    bool show_tile_grid = false;
    bool show_density_heatmap = false;
    bool show_performance_hud = false;
    bool show_lod_spheres = true;
    bool show_performance_overlay = false;
    float debug_overlay_opacity = 0.3f;
    DebugDrawMode debug_draw_mode = DEBUG_DRAW_POINTS;
    bool runtime_preview_enabled = false;
    bool show_residency_hud = false;
    int runtime_preview_restore_mode = DEBUG_DRAW_POINTS;
    CanvasLayer *debug_hud_layer = nullptr;
    GaussianSplatDebugHUD *debug_hud_control = nullptr;

    GaussianSplatNodeAssetHelper asset_helper;
    GaussianSplatNodeViewportHelper viewport_helper;
    GaussianSplatNodeDebugHelper debug_helper;
    GaussianSplatNodeQualityHelper quality_helper;
    GaussianSplatNodeVisibilityHelper visibility_helper;
    GaussianSplatNodeRendererHelper renderer_helper;

    void _load_asset();
    void _update_asset();
    void _clear_asset();

    void _update_bounds();
    void _update_visibility();
    void _update_quality_settings();
    void _apply_painterly_settings();
    void _log_update_splats_call(int update_call_count, int asset_splat_count, int procedural_splat_count) const;
    bool _handle_empty_splat_frame(RenderingServer *rs, uint64_t start_time,
            int asset_splat_count, int procedural_splat_count);
    void _sync_renderer_splat_counts(int asset_splat_count, int procedural_splat_count);
    void _update_renderer_gpu_memory();
    void _update_viewport_render_state(RenderingServer *rs, int update_call_count);
    void _apply_render_instance_state(RenderingServer *rs);
    void _finalize_update_splats(uint64_t start_time);
    bool _ensure_renderer_for_manual_data();
    bool _validate_splat_data_inputs(int splat_count,
            const PackedVector3Array &p_positions,
            const PackedColorArray &p_colors,
            const PackedVector3Array &p_scales,
            const PackedFloat32Array &p_opacities,
            const TypedArray<Quaternion> &p_rotations,
            const PackedFloat32Array &p_spherical_harmonics,
            const PackedInt32Array &p_palette_ids,
            const PackedInt32Array &p_painterly_flags,
            const PackedVector3Array &p_normals,
            const PackedVector2Array &p_brush_axes,
            const PackedFloat32Array &p_stroke_ages) const;
    void _reset_manual_splat_state();
    void _ensure_renderer_data_for_splats(int splat_count, const PackedVector3Array &p_positions);
    void _apply_optional_splat_arrays(int splat_count,
            const PackedColorArray &p_colors,
            const PackedVector3Array &p_scales,
            const PackedFloat32Array &p_opacities,
            const TypedArray<Quaternion> &p_rotations,
            const PackedFloat32Array &p_spherical_harmonics,
            const PackedInt32Array &p_palette_ids,
            const PackedInt32Array &p_painterly_flags,
            const PackedVector3Array &p_normals,
            const PackedVector2Array &p_brush_axes,
            const PackedFloat32Array &p_stroke_ages);
    void _populate_runtime_asset_from_renderer_data();
    void _compute_manual_splat_bounds(int splat_count,
            const PackedVector3Array &p_positions,
            const PackedVector3Array &p_scales);
    void _finalize_manual_splat_setup(int splat_count);
    void _apply_quality_tier_limits(int &effective_max_splats, int &effective_max_gpu_mb,
            int &effective_target_gpu_mb, float &effective_load_ahead, float &effective_unload,
            int &effective_concurrent_loads, int &effective_stream_budget_ms) const;
    void _apply_quality_lod_config(float lod0_distance, float lod1_distance, float lod2_distance, float lod3_distance,
            float effective_distance, uint32_t max_budget, uint32_t min_budget, float importance_threshold, float size_cull_threshold,
            bool smooth_transitions, float transition_time, float target_fps, float quality_rate,
            bool temporal_coherence);
    void _apply_streaming_config_values(int effective_max_gpu_mb, int effective_target_gpu_mb,
            float effective_distance, float effective_load_ahead, float effective_unload, int effective_concurrent_loads,
            bool predictive_loading, float prediction_time, int lod_level_count, float lod_distance_multiplier,
            bool adaptive_quality, int effective_stream_budget_ms, bool async_loading, bool compression);
    void _fill_preset_config(QualityPreset p_preset, Dictionary &config) const;
    void _ensure_debug_hud_control();
    void _update_debug_hud_visibility();

    void _ensure_renderer();
    void _apply_renderer_settings();
    void _update_render_instance();
    void _upload_asset_to_renderer();
    void _mark_render_state_dirty();
    void _register_shared_renderer();
    void _unregister_shared_renderer();
    void _update_shared_transform();
    bool _resolve_is_2d_mode() const;
    uint32_t _get_instance_flags() const;
    float _get_instance_wind_intensity() const;
    uint32_t _get_instance_wind_mode() const;
    Vector3 _get_instance_wind_direction() const;
    float _get_instance_wind_frequency() const;
    void _register_instance_in_director();
    void _unregister_instance_in_director();
    void _update_instance_transform_in_director();
    void _update_instance_params_in_director();
    void _notification_enter_tree();
    void _notification_enter_world();
    void _notification_exit_tree();
    void _notification_process();
#ifdef TOOLS_ENABLED
    void _notification_editor_post_save();
#endif
    String _get_asset_source_path() const;
    bool _has_inconsistent_dual_source_configuration(String *r_asset_source_path = nullptr) const;

    void _ensure_gaussian_base();
    void _release_gaussian_base();
    void _sync_gaussian_storage();
    void _set_instance_base(const RID &p_base);
    void _update_cached_render_target(Viewport *p_viewport);
    Viewport *_find_editor_scene_viewport() const;
    bool _acquire_viewport_render_target(Viewport *p_viewport);
    void _connect_viewport_observers(Viewport *p_viewport);
    void _disconnect_viewport_observers();
    void _ensure_viewport_texture_binding(Viewport *p_viewport);
    void _queue_viewport_bootstrap();
    void _deferred_viewport_bootstrap();
    void _queue_first_frame_render();
    void _dispatch_first_frame_render();
    void _on_viewport_texture_ready();
    void _on_viewport_size_changed();
    void _on_observed_viewport_exited();

    void _on_asset_changed();
    void _on_color_grading_changed();
    void _on_transform_changed();
    void _update_parent_visibility_tracking();
    void _clear_parent_visibility_tracking();
    void _update_parent_visibility_state();
    void _on_parent_visibility_changed();
    void _on_parent_visibility_changed_with_bool(bool p_visible);

    // Quality preset configurations
    void _apply_quality_preset();
    Dictionary _get_preset_config(QualityPreset p_preset) const;

protected:
    static void _bind_methods();
    void _notification(int p_what);

    bool _set(const StringName &p_name, const Variant &p_value);
    bool _get(const StringName &p_name, Variant &r_ret) const;
    void _get_property_list(List<PropertyInfo> *p_list) const;

    void _validate_property(PropertyInfo &p_property) const;

public:
    GaussianSplatNode3D();
    ~GaussianSplatNode3D();

    /// @name Asset Management
    /// @{

    /**
     * @brief Sets the path to a Gaussian splat file to load.
     * @param p_path Resource path (res://) or absolute path to the file.
     *
     * Supported formats:
     * - .ply: Standard PLY format (ASCII or binary)
     * - .spz: Niantic compressed format
     *
     * If auto_load is enabled, the file will be loaded immediately.
     *
     * @note The property name includes "ply" for historical reasons but
     *       the loader auto-detects format based on file extension.
     */
    void set_ply_file_path(const String &p_path);
    String get_ply_file_path() const { return ply_file_path; }

    /**
     * @brief Sets a GaussianSplatAsset resource to display.
     * @param p_asset Asset containing pre-processed splat data.
     */
    void set_splat_asset(const Ref<GaussianSplatAsset> &p_asset);
    Ref<GaussianSplatAsset> get_splat_asset() const { return splat_asset; }
    String get_asset_origin_label() const;

    /**
     * @brief Enables automatic loading when ply_file_path is set.
     * @param p_enabled When true, files are loaded when the path changes.
     *
     * @note Loading only occurs when the node is inside the scene tree.
     *       If set before adding to tree, loading happens on _ready().
     */
    void set_auto_load(bool p_enabled);
    bool is_auto_load_enabled() const { return auto_load; }

    /** @brief Forces a reload of the current asset or PLY file. */
    void reload_asset();

    /** @brief Returns true if an asset is currently being loaded asynchronously. */
    bool is_asset_loading() const { return asset_loading; }

    /**
     * @brief Sets splat data directly from arrays (procedural generation).
     * @param p_positions Array of 3D positions for each splat.
     * @param p_colors Optional array of RGBA colors.
     * @param p_scales Optional array of per-axis scales.
     * @param p_opacities Optional array of opacity values.
     * @param p_rotations Optional array of rotation quaternions.
     * @param p_spherical_harmonics Optional packed SH coefficients.
     * @param p_palette_ids Optional palette indices for painterly rendering.
     * @param p_painterly_flags Optional painterly flag bitfields.
     * @param p_normals Optional surface normals for surfel mode.
     * @param p_brush_axes Optional brush axis vectors.
     * @param p_stroke_ages Optional stroke age values.
     * @param p_is_2d_mode When true, enables 2D Gaussian (surfel) mode.
     */
    void set_splat_data(const PackedVector3Array &p_positions,
            const PackedColorArray &p_colors = PackedColorArray(),
            const PackedVector3Array &p_scales = PackedVector3Array(),
            const PackedFloat32Array &p_opacities = PackedFloat32Array(),
            const TypedArray<Quaternion> &p_rotations = TypedArray<Quaternion>(),
            const PackedFloat32Array &p_spherical_harmonics = PackedFloat32Array(),
            const PackedInt32Array &p_palette_ids = PackedInt32Array(),
            const PackedInt32Array &p_painterly_flags = PackedInt32Array(),
            const PackedVector3Array &p_normals = PackedVector3Array(),
            const PackedVector2Array &p_brush_axes = PackedVector2Array(),
            const PackedFloat32Array &p_stroke_ages = PackedFloat32Array(),
            bool p_is_2d_mode = false);

    /// @}

    /// @name Quality Settings
    /// @{

    /**
     * @brief Applies a quality preset configuration.
     * @param p_preset One of QUALITY_PERFORMANCE, QUALITY_BALANCED, QUALITY_QUALITY, or QUALITY_CUSTOM.
     */
    void set_quality_preset(QualityPreset p_preset);
    QualityPreset get_quality_preset() const { return quality_preset; }

    /** @brief Returns the current LOD configuration. */
    const GaussianSplatting::AdaptiveLODSystem::LODConfig &get_lod_config() const { return lod_config; }

    /** @brief Returns the current streaming configuration. */
    const GaussianSplatting::StreamingLODManager::StreamingConfig &get_streaming_config() const { return streaming_config; }

    /** @brief Sets LOD bias (higher values keep more detail at distance). */
    void set_lod_bias(float p_bias);
    float get_lod_bias() const { return lod_bias; }

    /** @brief Sets maximum render distance in world units. */
    void set_max_render_distance(float p_distance);
    float get_max_render_distance() const { return max_render_distance; }

    /** @brief Sets maximum number of splats to render per frame. */
    void set_max_splat_count(int p_count);
    int get_max_splat_count() const { return max_splat_count; }

    /// @}

    /// @name Painterly Settings
    /// @brief Controls for stylized brush-stroke rendering.
    /// @{

    /**
     * @brief Enables or disables painterly rendering effects.
     * @param p_enabled When true, applies stylized brush-stroke rendering.
     */
    void set_enable_painterly(bool p_enabled);

    /** @brief Returns true if painterly rendering is enabled. */
    bool is_painterly_enabled() const { return enable_painterly; }

    /**
     * @brief Sets the edge detection threshold for painterly outlines.
     * @param p_threshold Threshold in [0, 1]; lower values detect more edges.
     */
    void set_edge_threshold(float p_threshold);

    /** @brief Returns the painterly edge threshold. */
    float get_edge_threshold() const { return edge_threshold; }

    /**
     * @brief Sets the opacity of painterly brush strokes.
     * @param p_opacity Opacity in [0, 1].
     */
    void set_stroke_opacity(float p_opacity);

    /** @brief Returns the stroke opacity. */
    float get_stroke_opacity() const { return stroke_opacity; }

    /**
     * @brief Sets the width of painterly brush strokes.
     * @param p_width Stroke width in pixels.
     */
    void set_stroke_width(float p_width);

    /** @brief Returns the stroke width. */
    float get_stroke_width() const { return stroke_width; }

    /** @brief Legacy compatibility setter retained for serialized scenes/scripts. Not exposed as a node property. */
    void set_color_variation(float p_variation);
    float get_color_variation() const { return color_variation; }

    /**
     * @brief Sets the temporal blend factor for painterly stability.
     * @param p_blend Blend factor in [0, 1]; higher values reduce flickering.
     */
    void set_temporal_blend(float p_blend);

    /** @brief Returns the temporal blend factor. */
    float get_temporal_blend() const { return temporal_blend; }

    /**
     * @brief Sets the random seed for painterly noise patterns.
     * @param p_seed Random seed value.
     */
    void set_painterly_seed(uint32_t p_seed);

    /** @brief Returns the painterly seed. */
    uint32_t get_painterly_seed() const { return painterly_seed; }

    /// @}

    /// @name Rendering Settings
    /// @{

    /**
     * @brief Sets when the splat rendering is updated.
     * @param p_mode One of UPDATE_MODE_ALWAYS, UPDATE_MODE_WHEN_VISIBLE, etc.
     */
    void set_update_mode(ViewportUpdateMode p_mode);

    /** @brief Returns the current update mode. */
    ViewportUpdateMode get_update_mode() const { return update_mode; }

    /**
     * @brief Enables or disables shadow casting.
     * @param p_enabled When true, splats cast shadows (if supported).
     */
    void set_cast_shadow(bool p_enabled);

    /** @brief Returns true if shadow casting is enabled. */
    bool get_cast_shadow() const { return cast_shadow; }

    /**
     * @brief Enables or disables frustum culling.
     * @param p_enabled When true, splats outside the view frustum are culled.
     */
    void set_use_frustum_culling(bool p_enabled);

    /** @brief Returns true if frustum culling is enabled. */
    bool is_frustum_culling_enabled() const { return use_frustum_culling; }

    /** @brief Legacy compatibility setter retained for serialized scenes/scripts. Not exposed as a node property. */
    void set_use_occlusion_culling(bool p_enabled);
    bool is_occlusion_culling_enabled() const { return use_occlusion_culling; }

    /** @brief Sets global opacity multiplier for all splats. */
    void set_opacity(float p_opacity);
    float get_opacity() const { return opacity; }

    /** @brief Enables per-instance wind overrides for this node. */
    void set_wind_override_enabled(bool p_enabled);
    bool is_wind_override_enabled() const { return wind_override_enabled; }

    /** @brief Overrides global wind enable state for this instance when wind override is enabled. */
    void set_wind_enabled(bool p_enabled);
    bool is_wind_enabled() const { return wind_enabled; }

    /** @brief Scales wind displacement for this instance when wind override is enabled. */
    void set_wind_strength(float p_strength);
    float get_wind_strength() const { return wind_strength; }

    /** @brief Overrides global wind direction for this instance when non-zero and wind override is enabled. */
    void set_wind_direction(const Vector3 &p_direction);
    Vector3 get_wind_direction() const { return wind_direction; }

    /** @brief Scales global wind frequency for this instance when wind override is enabled. */
    void set_wind_frequency(float p_frequency);
    float get_wind_frequency() const { return wind_frequency; }

    /** @brief Sets color grading resource for real-time or baked color adjustments. */
    void set_color_grading(const Ref<class ColorGradingResource> &p_grading);
    Ref<class ColorGradingResource> get_color_grading() const { return color_grading; }

    /**
     * @brief Bakes the current color grading into the splat data.
     * @return OK on success, error code on failure.
     *
     * This permanently applies color grading to the base colors (SH DC coefficients).
     * The original colors are backed up and can be restored via restore_color_grading().
     * Baked color grading has zero runtime cost.
     */
    Error bake_color_grading();

    /**
     * @brief Bakes a provided color grading snapshot into the splat data.
     * @param p_grading_snapshot Color grading parameters captured at action creation time.
     * @return OK on success, error code on failure.
     */
    Error bake_color_grading_snapshot(const Ref<class ColorGradingResource> &p_grading_snapshot);

    /**
     * @brief Restores original colors before any color grading was baked.
     *
     * This reverts all splat colors to their state before the first bake_color_grading() call.
     * Does nothing if no baking has been applied.
     */
    void restore_color_grading();

    /** @brief Returns true if color grading has been baked into the splat data. */
    bool is_color_grading_baked() const;

    /// @}

    /// @name Performance Monitoring
    /// @{

    /** @brief Returns the number of splats visible after culling. */
    uint32_t get_visible_splat_count() const { return visible_splat_count; }

    /** @brief Returns the total number of splats in the loaded asset. */
    uint32_t get_total_splat_count() const { return total_splat_count; }

    /** @brief Returns the last frame's update time in milliseconds. */
    float get_last_update_time_ms() const { return last_update_time_ms; }

    /** @brief Returns estimated GPU memory usage in megabytes. */
    float get_gpu_memory_mb() const { return gpu_memory_mb; }

    /**
     * @brief Returns comprehensive statistics as a Dictionary.
     * @return Dictionary containing:
     *   - "visible_splats", "total_splats" - Splat counts
     *   - "update_time_ms", "gpu_memory_mb" - Performance metrics
     *   - "bounds" - AABB of the splat data
     *   - "debug_draw_mode", "show_lod_spheres", "show_performance_overlay", "preview_enabled" - Debug state
     *   - Additional renderer statistics from get_render_stats()
     */
    Dictionary get_statistics() const;

    /// @}

    /// @name Bounds
    /// @{

    /** @brief Returns the axis-aligned bounding box in local space. */
    AABB get_aabb() const;

    /// @}

    /// @name Editor Helpers
    /// @{

    /**
     * @brief Enables or disables editor preview rendering.
     * @param p_enabled When true, renders splats in the editor viewport.
     */
    void set_preview_enabled(bool p_enabled);

    /** @brief Returns true if editor preview is enabled. */
    bool is_preview_enabled() const { return preview_enabled; }

    /**
     * @brief Shows or hides the bounding box visualization.
     * @param p_show When true, draws the AABB in the editor.
     */
    void set_show_bounds(bool p_show);

    /** @brief Returns true if bounds are being shown. */
    bool is_showing_bounds() const { return show_bounds; }

    /**
     * @brief Shows or hides performance statistics overlay.
     * @param p_show When true, displays stats in the editor.
     */
    void set_show_statistics(bool p_show);

    /** @brief Returns true if statistics are being shown. */
    bool is_showing_statistics() const { return show_statistics; }

    /// @}

    /// @name Debug Overlays
    /// @{

    /**
     * @brief Shows or hides the tile grid debug overlay.
     * @param p_show When true, renders tile boundaries.
     */
    void set_show_tile_grid(bool p_show);

    /** @brief Returns true if the tile grid is being shown. */
    bool is_showing_tile_grid() const { return show_tile_grid; }

    /**
     * @brief Shows or hides the density heatmap overlay.
     * @param p_show When true, renders splat density per tile.
     */
    void set_show_density_heatmap(bool p_show);

    /** @brief Returns true if the density heatmap is being shown. */
    bool is_showing_density_heatmap() const { return show_density_heatmap; }

    /**
     * @brief Shows or hides the performance HUD.
     * @param p_show When true, displays frame time and splat count.
     */
    void set_show_performance_hud(bool p_show);

    /** @brief Returns true if the performance HUD is being shown. */
    bool is_showing_performance_hud() const { return show_performance_hud; }

    /**
     * @brief Shows or hides LOD sphere visualization.
     * @param p_show When true, renders LOD distance spheres.
     */
    void set_show_lod_spheres(bool p_show);

    /** @brief Returns true if LOD spheres are being shown. */
    bool is_showing_lod_spheres() const { return show_lod_spheres; }

    /**
     * @brief Shows or hides the performance overlay.
     * @param p_show When true, renders performance metrics.
     */
    void set_show_performance_overlay(bool p_show);

    /** @brief Returns true if the performance overlay is being shown. */
    bool is_showing_performance_overlay() const { return show_performance_overlay; }

    /**
     * @brief Sets the opacity of debug overlays (tile grid, heatmap).
     * @param p_opacity Blend opacity (0.0-1.0). Default is 0.3.
     */
    void set_debug_overlay_opacity(float p_opacity);

    /** @brief Returns the current debug overlay opacity. */
    float get_debug_overlay_opacity() const { return debug_overlay_opacity; }

    /**
     * @brief Sets the debug visualization mode.
     * @param p_mode One of DEBUG_DRAW_OFF, DEBUG_DRAW_WIREFRAME, etc.
     */
    void set_debug_draw_mode(DebugDrawMode p_mode);

    /** @brief Returns the current debug draw mode. */
    DebugDrawMode get_debug_draw_mode() const { return debug_draw_mode; }

    /**
     * @brief Enables or disables runtime preview (in-game debug).
     * @param p_enabled When true, allows debug visualization at runtime.
     */
    void set_runtime_preview_enabled(bool p_enabled);

    /** @brief Returns true if runtime preview is enabled. */
    bool is_runtime_preview_enabled() const { return runtime_preview_enabled; }

    /**
     * @brief Shows or hides the memory residency HUD.
     * @param p_show When true, displays GPU memory residency info.
     */
    void set_show_residency_hud(bool p_show);

    /** @brief Returns true if the residency HUD is being shown. */
    bool is_showing_residency_hud() const { return show_residency_hud; }

    /// @}

    /// @name Renderer Access
    /// @{

    /**
     * @brief Returns the shared GaussianSplatRenderer for this World3D.
     * @return Reference to the renderer, or an invalid reference if not initialized.
     */
    Ref<GaussianSplatRenderer> get_renderer();

    /// @}

    /// @name Manual Update
    /// @{

    /** @brief Triggers a splat update (use with UPDATE_MODE_MANUAL). */
    void update_splats();

    /** @brief Forces a complete re-render regardless of update mode. */
    void force_update();

    /** @brief Internal method to process Gaussian rendering for the current frame. */
    void process_gaussian_render();

    /// @}

    /// @name Utility
    /// @{

    PackedStringArray get_configuration_warnings() const override;

    /// @}

#ifdef TOOLS_ENABLED
    // Editor-specific functionality
    bool _can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;
    void _drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);
#endif
};

VARIANT_ENUM_CAST(GaussianSplatNode3D::QualityPreset);
VARIANT_ENUM_CAST(GaussianSplatNode3D::ViewportUpdateMode);
VARIANT_ENUM_CAST(GaussianSplatNode3D::DebugDrawMode);

#endif // GAUSSIAN_SPLAT_NODE_3D_H
