#ifndef GAUSSIAN_SPLAT_NODE_HELPERS_H
#define GAUSSIAN_SPLAT_NODE_HELPERS_H

#include <cstdint>

class Dictionary;

class GaussianSplatNode3D;
class Viewport;

class GaussianSplatNodeAssetHelper {
public:
    explicit GaussianSplatNodeAssetHelper(GaussianSplatNode3D &p_owner) : owner(p_owner) {}

    void load_asset();
    void update_asset();
    void clear_asset();

private:
    GaussianSplatNode3D &owner;
};

class GaussianSplatNodeViewportHelper {
public:
    explicit GaussianSplatNodeViewportHelper(GaussianSplatNode3D &p_owner) : owner(p_owner) {}

    Viewport *find_editor_scene_viewport() const;
    void update_cached_render_target(Viewport *p_viewport);
    bool acquire_viewport_render_target(Viewport *p_viewport);
    void connect_viewport_observers(Viewport *p_viewport);
    void disconnect_viewport_observers();
    void ensure_viewport_texture_binding(Viewport *p_viewport);
    void queue_viewport_bootstrap();
    void deferred_viewport_bootstrap();
    void on_viewport_texture_ready();
    void on_viewport_size_changed();
    void on_observed_viewport_exited();

private:
    GaussianSplatNode3D &owner;
};

class GaussianSplatNodeDebugHelper {
public:
    explicit GaussianSplatNodeDebugHelper(GaussianSplatNode3D &p_owner) : owner(p_owner) {}

    void apply_renderer_debug_settings();
    void set_show_tile_grid(bool p_show);
    void set_show_density_heatmap(bool p_show);
    void set_show_performance_hud(bool p_show);
    void set_show_lod_spheres(bool p_show);
    void set_show_performance_overlay(bool p_show);
    void set_debug_overlay_opacity(float p_opacity);
    void set_debug_draw_mode(int p_mode);
    void set_runtime_preview_enabled(bool p_enabled);
    void set_show_residency_hud(bool p_show);

private:
    GaussianSplatNode3D &owner;
};

class GaussianSplatNodeQualityHelper {
public:
    explicit GaussianSplatNodeQualityHelper(GaussianSplatNode3D &p_owner) : owner(p_owner) {}

    void apply_quality_preset();
    void fill_preset_config(int p_preset, Dictionary &config) const;
    void apply_quality_tier_limits(int &effective_max_splats, int &effective_max_gpu_mb,
            int &effective_target_gpu_mb, float &effective_load_ahead, float &effective_unload,
            int &effective_concurrent_loads, int &effective_stream_budget_ms) const;
    void update_quality_settings();
    void apply_quality_lod_config(float lod0_distance, float lod1_distance, float lod2_distance, float lod3_distance,
            float effective_distance, uint32_t max_budget, uint32_t min_budget, float importance_threshold, float size_cull_threshold,
            bool smooth_transitions, float transition_time, float target_fps, float quality_rate,
            bool temporal_coherence);
    void apply_streaming_config_values(int effective_max_gpu_mb, int effective_target_gpu_mb,
            float effective_distance, float effective_load_ahead, float effective_unload, int effective_concurrent_loads,
            bool predictive_loading, float prediction_time, int lod_level_count, float lod_distance_multiplier,
            bool adaptive_quality, int effective_stream_budget_ms, bool async_loading, bool compression);

private:
    GaussianSplatNode3D &owner;
};

class GaussianSplatNodeVisibilityHelper {
public:
    explicit GaussianSplatNodeVisibilityHelper(GaussianSplatNode3D &p_owner) : owner(p_owner) {}

    void clear_parent_visibility_tracking();
    void update_parent_visibility_tracking();
    void update_parent_visibility_state();
    void on_parent_visibility_changed_with_bool(bool p_visible);
    void on_parent_visibility_changed();

private:
    GaussianSplatNode3D &owner;
};

class GaussianSplatNodeRendererHelper {
public:
    explicit GaussianSplatNodeRendererHelper(GaussianSplatNode3D &p_owner) : owner(p_owner) {}

    bool can_apply_renderer_settings() const;
    void release_renderer_settings_ownership();
    void ensure_renderer();
    void apply_renderer_settings();
    void mark_render_state_dirty();
    void upload_asset_to_renderer();

private:
    GaussianSplatNode3D &owner;
};

#endif // GAUSSIAN_SPLAT_NODE_HELPERS_H
