#ifndef GS_TILE_RASTERIZER_H
#define GS_TILE_RASTERIZER_H

#include "rasterizer_interfaces.h"
#include "core/object/ref_counted.h"
#include "render_device_manager.h"
#include "../renderer/tile_renderer.h"  // For TileRenderer::RenderParams

// Adapter that wraps TileRenderer to implement IRasterizer interface
class TileRasterizer : public RefCounted, public IRasterizer {
    GDCLASS(TileRasterizer, RefCounted);

public:
    TileRasterizer();
    ~TileRasterizer();

    // IRasterizer interface
    Error initialize(RenderingDevice *p_device, const Vector2i &p_initial_viewport = Vector2i(),
            int p_tile_size = -1, RD::DataFormat p_format = RD::DATA_FORMAT_MAX) override;
    void shutdown() override;
    bool is_ready() const override;

    RasterResult render(const RasterParams &p_params) override;

    // Direct render using TileRenderer::RenderParams (for incremental migration)
    RasterResult render_direct(RenderingDevice *p_device, const TileRenderer::RenderParams &p_params);

    Error resize(const Vector2i &p_size, RD::DataFormat p_format = RD::DATA_FORMAT_MAX) override;

    void set_output_format(RD::DataFormat p_format) override;
    RD::DataFormat get_output_format() const override;

    RID get_output_texture() const override;
    RID get_depth_texture() const override;
    RenderingDevice *get_output_texture_owner() const override;
    RenderingDevice *get_depth_texture_owner() const override;
    bool has_depth_output() const override;

    void set_debug_options(const RasterDebugOptions &p_options) override;
    RasterDebugOptions get_debug_options() const override;

    RasterDebugCounters get_debug_counters() const override;
    RasterOverflowStats get_overflow_stats() const override;
    RasterStats get_render_stats() const override;
    RasterPerformance get_performance() const override;

    int get_tile_size() const override;
    Vector2i get_tile_grid_size() const override;
    int get_tile_splat_capacity() const override;
    int get_tile_count() const override;
    bool is_depth_copy_compatible() const override;

    void set_frame_serial(uint64_t p_serial) override;
    void resolve_gpu_timestamps_async() override;
    void set_resolve_debug_mode(int p_mode) override;

    RID get_debug_counter_buffer() const override;
    Vector<uint32_t> get_tile_density_snapshot() const override;

    String get_name() const override { return "TileRasterizer"; }

    // Direct access to underlying TileRenderer for advanced configuration
    Ref<TileRenderer> get_tile_renderer() const { return tile_renderer; }

    // Set an external TileRenderer (for wrapping existing renderer)
    // If set, this will be used instead of creating an internal one
    void set_tile_renderer(Ref<TileRenderer> p_renderer);

    // Check if using external renderer
    bool is_using_external_renderer() const { return using_external_renderer; }

    // Resource tracking for output textures
    void set_device_manager(Ref<RenderDeviceManager> p_device_manager);
    void track_output_resources(const RID &p_color_output, RenderingDevice *p_color_device,
            const RID &p_depth_output, RenderingDevice *p_depth_device);
    void clear_output_resource_tracking();

protected:
    static void _bind_methods();

private:
    Ref<TileRenderer> tile_renderer;
    RenderingDevice *rd = nullptr;
    RasterDebugOptions current_debug_options;
    bool using_external_renderer = false;
    Ref<RenderDeviceManager> device_manager;
    RID tracked_color_output;
    RID tracked_depth_output;
    void _bind_output_invalidation_callback();
    void _unbind_output_invalidation_callback();
};

#endif // GS_TILE_RASTERIZER_H
