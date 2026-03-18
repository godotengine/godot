#ifndef GS_PAINTERLY_RENDERER_INTERFACES_H
#define GS_PAINTERLY_RENDERER_INTERFACES_H

#include "core/math/transform_3d.h"
#include "core/math/projection.h"
#include "core/math/vector2i.h"
#include "core/string/string_name.h"
#include "core/templates/local_vector.h"
#include "core/variant/dictionary.h"
#include "servers/rendering/rendering_device.h"

// Forward declarations
class TileRasterizer;

// Painterly texture slots
enum class PainterlyTextureSlot {
    COLOR = 0,
    DEPTH,
    EDGE,
    STYLIZED,
    COUNT
};

// Painterly pass identifiers
enum class PainterlyPassId {
    GBUFFER = 0,
    SOBEL_EDGES,
    BRUSH_ACCUMULATION
};

// Configuration for painterly rendering
struct PainterlyConfig {
    Vector2i viewport_size = Vector2i(1920, 1080);
    float internal_scale = 1.0f;
    bool enable_stylization = true;
    bool low_end_mode = false;
    bool enable_strokes = true;
    RD::DataFormat color_format = RD::DATA_FORMAT_R8G8B8A8_UNORM;

    // Stylization parameters (Phase 1 extension)
    float edge_threshold = 0.25f;
    float edge_intensity = 1.5f;
    float stroke_length = 32.0f;
    float stroke_opacity = 0.8f;
    float gamma = 2.2f;
};

// Texture information for a painterly slot
struct PainterlyTextureInfo {
    Vector2i size = Vector2i();
    RD::DataFormat format = RD::DATA_FORMAT_MAX;
    RID texture;
    RID shared_texture;
    RenderingDevice *shared_owner = nullptr;
    bool valid = false;
};

// Performance metrics for painterly rendering
struct PainterlyPerformance {
    float edge_detection_ms = 0.0f;
    float brush_accumulation_ms = 0.0f;
    float total_pass_ms = 0.0f;
    // CPU-side wall-clock time for execute_passes() command recording.
    // This does NOT include GPU execution time; Godot's RenderingDevice does not
    // expose per-dispatch timestamps.  Use RenderDoc / Nsight for GPU timing.
    float last_pass_cpu_time_ms = 0.0f;
    uint32_t edge_samples = 0;
    uint32_t brush_strokes = 0;
};

// Render result from painterly pass
struct PainterlyRenderResult {
    RID final_texture;
    RenderingDevice *final_texture_owner = nullptr;
    bool stylization_applied = false;
    bool success = false;
};

// Input data for full painterly render pipeline (Phase 1 extension)
struct PainterlyRenderInput {
    RID gaussian_buffer;
    RID sorted_indices;
    uint32_t splat_count = 0;
    uint32_t total_gaussians = 0;
    Transform3D world_to_camera_transform;
    Transform3D camera_to_world_transform; // PERF (#659): Pre-computed inverse to avoid affine_inverse() in render()
    Projection projection;
    Projection render_projection; // GPU projection with depth/jitter correction applied.
    Vector2i viewport_size;
    RID interactive_state_uniform;  // Optional for interactive features
};

// Pure abstract interface for painterly rendering pipeline
class IPainterlyRenderer {
public:
    virtual ~IPainterlyRenderer() = default;

    // Lifecycle
    virtual Error initialize(RenderingDevice *p_device, const Vector2i &p_initial_size = Vector2i()) = 0;
    virtual void shutdown() = 0;
    virtual bool is_ready() const = 0;

    // Configuration
    virtual void configure(const PainterlyConfig &p_config) = 0;
    virtual PainterlyConfig get_config() const = 0;
    virtual void set_color_format(RD::DataFormat p_format) = 0;
    virtual RD::DataFormat get_color_format() const = 0;

    // Size management
    virtual Vector2i get_internal_size() const = 0;
    virtual Vector2i get_requested_size() const = 0;
    virtual float get_internal_scale() const = 0;

    // Stylization state
    virtual bool is_stylization_enabled() const = 0;
    virtual bool is_low_end_mode() const = 0;

    // Texture access
    virtual RID get_texture(PainterlyTextureSlot p_slot) const = 0;
    virtual RID get_shared_texture(PainterlyTextureSlot p_slot) const = 0;
    virtual RenderingDevice *get_shared_texture_owner(PainterlyTextureSlot p_slot) const = 0;
    virtual PainterlyTextureInfo get_texture_info(PainterlyTextureSlot p_slot) const = 0;

    // Rendering
    virtual PainterlyRenderResult execute_passes(RID p_color_input, RID p_depth_input = RID()) = 0;

    // Full render pipeline: G-buffer → Sobel → Brush → output (Phase 1 extension)
    virtual PainterlyRenderResult render(const PainterlyRenderInput &p_input) = 0;

    // Material resource management (Phase 1 extension)
    virtual void set_material_textures(const LocalVector<RID> &p_palette,
                                       const LocalVector<RID> &p_noise_luts,
                                       RID p_stroke_density_buffer) = 0;
    virtual void clear_material_textures() = 0;

    // Internal rasterizer access for advanced use (Phase 1 extension)
    virtual void set_rasterizer(Ref<TileRasterizer> p_rasterizer) = 0;
    virtual Ref<TileRasterizer> get_rasterizer() const = 0;

    // Shader compilation (separated for async compilation)
    virtual Error compile_shaders() = 0;
    virtual bool are_shaders_ready() const = 0;

    // Performance
    virtual PainterlyPerformance get_performance() const = 0;
    virtual void reset_performance() = 0;

    // Version tracking (for detecting config changes)
    virtual uint64_t get_version() const = 0;

    // Implementation info
    virtual String get_name() const = 0;
};

#endif // GS_PAINTERLY_RENDERER_INTERFACES_H
