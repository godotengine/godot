#ifndef GS_PAINTERLY_RENDERER_H
#define GS_PAINTERLY_RENDERER_H

#include "painterly_renderer_interfaces.h"
#include "tile_rasterizer.h"
#include "core/object/ref_counted.h"
#include "servers/rendering/renderer_rd/pipeline_cache_rd.h"

// Forward declarations
class PainterlyPassGraph;
class GaussianSplatRenderer;
class RenderDataRD;
class PainterlyMaterial;

// Adapter that wraps PainterlyPassGraph to implement IPainterlyRenderer interface
class PainterlyRenderer : public RefCounted, public IPainterlyRenderer {
    GDCLASS(PainterlyRenderer, RefCounted);

public:
    PainterlyRenderer();
    ~PainterlyRenderer();

    // IPainterlyRenderer interface - Lifecycle
    Error initialize(RenderingDevice *p_device, const Vector2i &p_initial_size = Vector2i()) override;
    void shutdown() override;
    bool is_ready() const override;

    // IPainterlyRenderer interface - Configuration
    void configure(const PainterlyConfig &p_config) override;
    PainterlyConfig get_config() const override;
    void set_color_format(RD::DataFormat p_format) override;
    RD::DataFormat get_color_format() const override;

    // IPainterlyRenderer interface - Size management
    Vector2i get_internal_size() const override;
    Vector2i get_requested_size() const override;
    float get_internal_scale() const override;

    // IPainterlyRenderer interface - Stylization state
    bool is_stylization_enabled() const override;
    bool is_low_end_mode() const override;

    // IPainterlyRenderer interface - Texture access
    RID get_texture(PainterlyTextureSlot p_slot) const override;
    RID get_shared_texture(PainterlyTextureSlot p_slot) const override;
    RenderingDevice *get_shared_texture_owner(PainterlyTextureSlot p_slot) const override;
    PainterlyTextureInfo get_texture_info(PainterlyTextureSlot p_slot) const override;

    // IPainterlyRenderer interface - Rendering
    PainterlyRenderResult execute_passes(RID p_color_input, RID p_depth_input = RID()) override;

    // IPainterlyRenderer interface - Full render pipeline (Phase 1 extension)
    PainterlyRenderResult render(const PainterlyRenderInput &p_input) override;

    // IPainterlyRenderer interface - Material resources (Phase 1 extension)
    void set_material_textures(const LocalVector<RID> &p_palette,
                               const LocalVector<RID> &p_noise_luts,
                               RID p_stroke_density_buffer) override;
    void clear_material_textures() override;

    // IPainterlyRenderer interface - Rasterizer access (Phase 1 extension)
    void set_rasterizer(Ref<TileRasterizer> p_rasterizer) override;
    Ref<TileRasterizer> get_rasterizer() const override;

    // IPainterlyRenderer interface - Shader compilation
    Error compile_shaders() override;
    bool are_shaders_ready() const override;

    // IPainterlyRenderer interface - Performance
    PainterlyPerformance get_performance() const override;
    void reset_performance() override;

    // IPainterlyRenderer interface - Version tracking
    uint64_t get_version() const override;

    // IPainterlyRenderer interface - Implementation info
    String get_name() const override { return "PainterlyRenderer"; }

    // Direct access to underlying pass graph for advanced configuration
    PainterlyPassGraph *get_pass_graph() const { return pass_graph; }

    // GaussianSplatRenderer integration (Phase 4 extraction)
    void set_material(const Ref<PainterlyMaterial> &p_material);
    Ref<PainterlyMaterial> get_material() const;
    bool is_material_dirty() const { return material_dirty; }
    void mark_material_dirty() { material_dirty = true; }

    void execute_painterly_passes(GaussianSplatRenderer *p_renderer, const Size2i &p_internal_size);
    void ensure_painterly_resources(GaussianSplatRenderer *p_renderer, const Size2i &p_viewport_size,
            RD::DataFormat p_target_format = RD::DATA_FORMAT_MAX);
    Error render_painterly_frame(GaussianSplatRenderer *p_renderer, const Size2i &p_viewport_size,
            RD::DataFormat p_target_format, const Transform3D &p_view_transform, const Projection &p_projection,
            const Projection &p_render_projection, RID &r_final_output, Size2i &r_internal_size,
            float &r_render_time_ms);
    Error populate_painterly_gbuffer(GaussianSplatRenderer *p_renderer, const Size2i &p_internal_size,
            const Transform3D &p_view_transform, const Projection &p_projection, const Projection &p_render_projection);
    bool composite_painterly_output(GaussianSplatRenderer *p_renderer, RenderDataRD *p_render_data, RID p_color_texture,
            RID p_depth_texture, const Size2i &p_viewport_size);
    void free_painterly_resources(GaussianSplatRenderer *p_renderer);
    void clear_painterly_gpu_resources(GaussianSplatRenderer *p_renderer);
    void update_painterly_gpu_resources(GaussianSplatRenderer *p_renderer);

protected:
    static void _bind_methods();

private:
    struct RidOwner {
        RenderingDevice *device = nullptr;
        uint64_t device_id = 0;

        void set(RenderingDevice *p_device) {
            device = p_device;
            device_id = p_device ? p_device->get_device_instance_id() : 0;
        }

        void clear() {
            device = nullptr;
            device_id = 0;
        }

        bool matches(RenderingDevice *p_device) const {
            if (!p_device) {
                return false;
            }
            if (device && device != p_device) {
                return false;
            }
            return device_id == 0 || device_id == p_device->get_device_instance_id();
        }
    };

    static constexpr int kTextureSlotCount = static_cast<int>(PainterlyTextureSlot::COUNT);
    PainterlyPassGraph *pass_graph = nullptr;
    RenderingDevice *rd = nullptr;
    PainterlyConfig current_config;
    PainterlyPerformance cached_performance;
    bool shaders_compiled = false;
    Ref<PainterlyMaterial> material;
    bool material_dirty = true;

    // Internal rasterizer for G-buffer rendering (Phase 1 extension)
    Ref<TileRasterizer> internal_rasterizer;
    bool owns_rasterizer = false;  // True if we created it, false if externally set

    // Material texture caches (Phase 1 extension)
    LocalVector<RID> cached_palette_textures;
    LocalVector<RID> cached_noise_luts;
    RID cached_stroke_density_buffer;

    // Final composite resources (owned by PainterlyRenderer).
    RID painterly_composite_shader;
    RidOwner painterly_composite_shader_owner;
    PipelineCacheRD painterly_composite_pipeline;
    bool painterly_composite_pipeline_initialized = false;
    RID painterly_depth_sampler;
    RidOwner painterly_depth_sampler_owner;
    RID painterly_color_sampler;
    RidOwner painterly_color_sampler_owner;
    bool painterly_composite_failed = false;
    bool material_textures_dirty = false;

    // Composite pipeline resources (Phase 1 extension)
    class PainterlyCompositeShaderRD *composite_shader_source = nullptr;
    RID composite_shader_version;
    RID composite_shader;
    RID composite_pipeline;
    RID composite_sampler;
    RidOwner composite_sampler_owner;
    RID composite_depth_sampler;
    RidOwner composite_depth_sampler_owner;
    bool composite_initialized = false;
    bool composite_failed = false;

    // Shader pipelines (managed internally)
    class SobelOutlineShaderRD *sobel_shader_source = nullptr;
    class BrushAccumulateShaderRD *brush_shader_source = nullptr;
    bool sobel_shader_initialized = false;
    bool brush_shader_initialized = false;
    bool composite_shader_initialized = false;
    RID sobel_shader_version;
    RID brush_shader_version;
    RID sobel_shader;
    RID sobel_pipeline;
    RID brush_shader;
    RID brush_pipeline;
    RID sobel_uniform_set;
    RID brush_uniform_set;
    RID sobel_uniform_sampler;
    RID sobel_uniform_color_input;
    RID sobel_uniform_edge_texture;
    RID brush_uniform_sampler;
    RID brush_uniform_color_input;
    RID brush_uniform_edge_input;
    RID brush_uniform_stylized_texture;
    RID painterly_sampler;
    RidOwner painterly_sampler_owner;
    RID tracked_local_textures[kTextureSlotCount];
    RID tracked_shared_textures[kTextureSlotCount];
    RID painterly_depth_viewport_texture;
    RID painterly_depth_override_render_target;

    Error _compile_sobel_shader();
    Error _compile_brush_shader();
    Error _compile_composite_shader();
    void _execute_sobel_pass(RID p_color_input);
    void _execute_brush_pass(RID p_color_input, RID p_edge_input);
    void _ensure_composite_resources();
    RenderingDevice *_resolve_tracked_device(const RidOwner &p_owner, GaussianSplatRenderer *p_renderer) const;
    void _free_tracked_rid(RID &p_rid, RidOwner &p_owner, GaussianSplatRenderer *p_renderer, bool p_forget_renderer_owner);
    void _shutdown_internal(GaussianSplatRenderer *p_renderer);
    void _update_painterly_texture_tracking(GaussianSplatRenderer *p_renderer);
    void _forget_painterly_texture_tracking(GaussianSplatRenderer *p_renderer);
    void _ensure_painterly_composite_resources(GaussianSplatRenderer *p_renderer, RD::FramebufferFormatID p_framebuffer_format);
};

#endif // GS_PAINTERLY_RENDERER_H
