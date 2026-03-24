#ifndef GS_DEBUG_OVERLAY_SYSTEM_H
#define GS_DEBUG_OVERLAY_SYSTEM_H

#include "debug_overlay_interfaces.h"
#include "../renderer/gaussian_splat_renderer.h"
#include "core/object/ref_counted.h"
#include "servers/rendering/rendering_device.h"

class DebugOverlaySystem;

class DebugOverlayQueryView {
public:
    explicit DebugOverlayQueryView(const DebugOverlaySystem *p_system = nullptr) :
            system(p_system) {}

    DebugOverlayOptions get_options() const;
    DebugCounterSnapshot get_debug_counters() const;
    Dictionary get_binning_debug_counters() const;
    bool is_dirty() const;
    uint64_t get_version() const;
    bool has_active_overlays() const;
    const Vector<String> &get_hud_lines() const;
    uint32_t get_tile_density_peak() const;
    float get_tile_density_average() const;
    const Vector<uint32_t> &get_tile_density_cache() const;
    int get_tile_density_width() const;
    int get_tile_density_height() const;

private:
    const DebugOverlaySystem *system = nullptr;

    const GaussianSplatRenderer::DebugState *debug_state = nullptr;
    const GaussianSplatRenderer::FrameState *frame_state = nullptr;
    const GaussianSplatRenderer::SortingState *sorting_state = nullptr;
    const GaussianSplatRenderer::PerformanceState *performance_state = nullptr;
    const GaussianSplatRenderer::DeviceState *device_state = nullptr;
    const GaussianSplatRenderer::SubsystemState *subsystem_state = nullptr;
    RenderingDevice *submission_device = nullptr;
    RenderingDevice *main_rendering_device = nullptr;

    friend class DebugOverlaySystem;
};

class DebugOverlayCommandSink {
public:
    explicit DebugOverlayCommandSink(DebugOverlaySystem *p_system = nullptr) :
            system(p_system) {}

    void set_show_tile_grid(bool p_enabled) const;
    void set_show_density_heatmap(bool p_enabled) const;
    void set_show_performance_hud(bool p_enabled) const;
    void set_show_residency_hud(bool p_enabled) const;
    void set_show_device_boundaries(bool p_enabled) const;
    void set_show_texture_states(bool p_enabled) const;
    void set_show_shadow_opacity(bool p_enabled) const;
    void set_overlay_opacity(float p_opacity) const;
    void set_dump_gpu_counters(bool p_enabled) const;
    void invalidate_overlay(bool p_increment_version = true) const;
    void invalidate_hud(bool p_increment_version = true) const;
    void rebuild_overlay_statistics_from_cache() const;
    void rebuild_performance_hud_lines() const;

private:
    DebugOverlaySystem *system = nullptr;
    GaussianSplatRenderer::DebugState *debug_state = nullptr;
    GaussianSplatRenderer::DebugConfig *debug_config = nullptr;

    friend class DebugOverlaySystem;
};

// Concrete implementation of IDebugOverlaySystem
class DebugOverlaySystem : public RefCounted, public IDebugOverlaySystem {
    GDCLASS(DebugOverlaySystem, RefCounted);

public:
    DebugOverlaySystem();
    ~DebugOverlaySystem();

    // IDebugOverlaySystem interface - Lifecycle
    void initialize() override;
    void shutdown() override;

    // IDebugOverlaySystem interface - Bulk options
    void set_options(const DebugOverlayOptions &p_options) override;
    DebugOverlayOptions get_options() const override;

    // IDebugOverlaySystem interface - Individual setters
    void set_show_tile_bounds(bool p_enabled) override;
    void set_show_splat_coverage(bool p_enabled) override;
    void set_show_tile_grid(bool p_enabled) override;
    void set_show_overflow_tiles(bool p_enabled) override;
    void set_show_projection_issues(bool p_enabled) override;
    void set_show_white_albedo(bool p_enabled);
    void set_show_density_heatmap(bool p_enabled) override;
    void set_show_shadow_opacity(bool p_enabled) override;
    void set_show_resolve_input(bool p_enabled) override;
    void set_show_resolve_output(bool p_enabled) override;
    void set_show_performance_hud(bool p_enabled) override;
    void set_show_residency_hud(bool p_enabled) override;
    void set_show_device_boundaries(bool p_enabled) override;
    void set_show_texture_states(bool p_enabled) override;
    void set_overlay_opacity(float p_opacity) override;
    void set_dump_gpu_counters(bool p_enabled) override;

    // IDebugOverlaySystem interface - Individual getters
    bool get_show_tile_bounds() const override { return options.show_tile_bounds; }
    bool get_show_splat_coverage() const override { return options.show_splat_coverage; }
    bool get_show_tile_grid() const override { return options.show_tile_grid; }
    bool get_show_overflow_tiles() const override { return options.show_overflow_tiles; }
    bool get_show_projection_issues() const override { return options.show_projection_issues; }
    bool get_show_white_albedo() const { return options.show_white_albedo; }
    bool get_show_density_heatmap() const override { return options.show_density_heatmap; }
    bool get_show_shadow_opacity() const override { return options.show_shadow_opacity; }
    bool get_show_resolve_input() const override { return options.show_resolve_input; }
    bool get_show_resolve_output() const override { return options.show_resolve_output; }
    bool get_show_performance_hud() const override { return options.show_performance_hud; }
    bool get_show_residency_hud() const override { return options.show_residency_hud; }
    bool get_show_device_boundaries() const override { return options.show_device_boundaries; }
    bool get_show_texture_states() const override { return options.show_texture_states; }
    float get_overlay_opacity() const override { return options.overlay_opacity; }
    bool get_dump_gpu_counters() const override { return options.dump_gpu_counters; }

    // IDebugOverlaySystem interface - Counter access
    DebugCounterSnapshot get_debug_counters() const override;
    Dictionary get_binning_debug_counters() const override;
    void reset_counters() override;

    // IDebugOverlaySystem interface - State tracking
    bool is_dirty() const override { return dirty; }
    void clear_dirty_flag() override { dirty = false; }
    uint64_t get_version() const override { return version; }

    // IDebugOverlaySystem interface - Active check
    bool has_active_overlays() const override;

    // IDebugOverlaySystem interface - Implementation info
    String get_name() const override { return "DebugOverlaySystem"; }

    // For integration with existing renderer - update counters from GPU readback
    void update_counters(const DebugCounterSnapshot &p_counters);
    void update_binning_counters(const Dictionary &p_counters);

    // HUD building and statistics - extracted from god class
    void rebuild_overlay_statistics_from_tile_density();
    void update_tile_density_cache(const Vector<uint32_t> &p_tile_counts, const Vector2i &p_tile_grid,
            uint32_t p_peak, float p_average);
    void clear_tile_density_cache();

    DebugOverlayQueryView build_query_view(const GaussianSplatRenderer *p_renderer = nullptr) const;
    DebugOverlayCommandSink build_command_sink(GaussianSplatRenderer *p_renderer);
    DebugOverlayQueryView get_query_view(const GaussianSplatRenderer *p_renderer = nullptr) const { return build_query_view(p_renderer); }
    DebugOverlayCommandSink get_command_sink(GaussianSplatRenderer *p_renderer) { return build_command_sink(p_renderer); }

    // Explicit debug/tooling seams.
    void set_renderer_show_tile_grid(const DebugOverlayCommandSink &p_sink, bool p_enabled);
    void set_renderer_show_density_heatmap(const DebugOverlayCommandSink &p_sink, bool p_enabled);
    void set_renderer_show_performance_hud(const DebugOverlayCommandSink &p_sink, bool p_enabled);
    void set_renderer_show_residency_hud(const DebugOverlayCommandSink &p_sink, bool p_enabled);
    void set_renderer_show_device_boundaries(const DebugOverlayCommandSink &p_sink, bool p_enabled);
    void set_renderer_show_texture_states(const DebugOverlayCommandSink &p_sink, bool p_enabled);
    void set_renderer_overlay_opacity(const DebugOverlayCommandSink &p_sink, float p_opacity);
    void invalidate_renderer_overlay(const DebugOverlayCommandSink &p_sink, bool p_increment_version);
    void invalidate_renderer_hud(const DebugOverlayCommandSink &p_sink, bool p_increment_version);
    void rebuild_renderer_overlay_statistics_from_cache(const DebugOverlayQueryView &p_query_view,
            const DebugOverlayCommandSink &p_sink);
    void rebuild_renderer_performance_hud_lines(const DebugOverlayQueryView &p_query_view,
            const DebugOverlayCommandSink &p_sink);

    // Legacy renderer helpers (god class extraction)
    void set_renderer_show_tile_grid(GaussianSplatRenderer *p_renderer, bool p_enabled);
    void set_renderer_show_density_heatmap(GaussianSplatRenderer *p_renderer, bool p_enabled);
    void set_renderer_show_performance_hud(GaussianSplatRenderer *p_renderer, bool p_enabled);
    void set_renderer_show_residency_hud(GaussianSplatRenderer *p_renderer, bool p_enabled);
    void set_renderer_show_device_boundaries(GaussianSplatRenderer *p_renderer, bool p_enabled);
    void set_renderer_show_texture_states(GaussianSplatRenderer *p_renderer, bool p_enabled);
    void set_renderer_overlay_opacity(GaussianSplatRenderer *p_renderer, float p_opacity);
    void invalidate_renderer_overlay(GaussianSplatRenderer *p_renderer, bool p_increment_version);
    void invalidate_renderer_hud(GaussianSplatRenderer *p_renderer, bool p_increment_version);
    void rebuild_renderer_overlay_statistics_from_cache(GaussianSplatRenderer *p_renderer);
    void rebuild_renderer_performance_hud_lines(GaussianSplatRenderer *p_renderer);

    // HUD text access for renderer integration
    const Vector<String> &get_hud_lines() const { return hud_lines; }
    uint32_t get_tile_density_peak() const { return tile_density_peak; }
    float get_tile_density_average() const { return tile_density_average; }
    const Vector<uint32_t> &get_tile_density_cache() const { return tile_density_cache; }
    int get_tile_density_width() const { return tile_density_width; }
    int get_tile_density_height() const { return tile_density_height; }

protected:
    static void _bind_methods();

private:
    DebugOverlayOptions options;
    DebugCounterSnapshot counters;
    Dictionary binning_counters;
    bool dirty = false;
    uint64_t version = 0;

    // HUD and overlay state
    Vector<String> hud_lines;
    Vector<uint32_t> tile_density_cache;
    int tile_density_width = 0;
    int tile_density_height = 0;
    uint32_t tile_density_peak = 0;
    float tile_density_average = 0.0f;

    void _mark_dirty();
};

#endif // GS_DEBUG_OVERLAY_SYSTEM_H
