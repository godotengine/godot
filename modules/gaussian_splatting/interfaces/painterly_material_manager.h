#ifndef GS_PAINTERLY_MATERIAL_MANAGER_H
#define GS_PAINTERLY_MATERIAL_MANAGER_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "servers/rendering/rendering_device.h"
#include "../painterly/painterly_material.h"

// Result of material resource update
struct PainterlyMaterialResources {
    LocalVector<RID> palette_texture_rids;
    LocalVector<RID> noise_lut_rids;
    RID stroke_density_buffer;
    uint32_t stroke_density_sample_count = 0;
    bool valid = false;
};

// Pure abstract interface for painterly material resource management
class IPainterlyMaterialManager {
public:
    virtual ~IPainterlyMaterialManager() = default;

    // Lifecycle
    virtual Error initialize(RenderingDevice *p_device) = 0;
    virtual void shutdown() = 0;
    virtual bool is_initialized() const = 0;

    // Material binding
    virtual void set_material(const Ref<PainterlyMaterial> &p_material) = 0;
    virtual Ref<PainterlyMaterial> get_material() const = 0;
    virtual void clear_material() = 0;

    // Resource updates
    virtual void update_resources() = 0;
    virtual void clear_resources() = 0;
    virtual bool is_dirty() const = 0;
    virtual void mark_dirty() = 0;

    // Resource access
    virtual PainterlyMaterialResources get_resources() const = 0;
    virtual const LocalVector<RID> &get_palette_textures() const = 0;
    virtual const LocalVector<RID> &get_noise_luts() const = 0;
    virtual RID get_stroke_density_buffer() const = 0;
    virtual uint32_t get_stroke_density_sample_count() const = 0;

    // Validation
    virtual Vector<String> get_missing_resources() const = 0;

    // Implementation info
    virtual String get_name() const = 0;
};

// Concrete implementation of painterly material resource manager
class PainterlyMaterialManager : public RefCounted, public IPainterlyMaterialManager {
    GDCLASS(PainterlyMaterialManager, RefCounted);

public:
    PainterlyMaterialManager();
    ~PainterlyMaterialManager();

    // IPainterlyMaterialManager interface - Lifecycle
    Error initialize(RenderingDevice *p_device) override;
    void shutdown() override;
    bool is_initialized() const override { return initialized && rd != nullptr; }

    // IPainterlyMaterialManager interface - Material binding
    void set_material(const Ref<PainterlyMaterial> &p_material) override;
    Ref<PainterlyMaterial> get_material() const override { return material; }
    void clear_material() override;

    // IPainterlyMaterialManager interface - Resource updates
    void update_resources() override;
    void clear_resources() override;
    bool is_dirty() const override { return material_dirty; }
    void mark_dirty() override { material_dirty = true; }

    // IPainterlyMaterialManager interface - Resource access
    PainterlyMaterialResources get_resources() const override;
    const LocalVector<RID> &get_palette_textures() const override { return palette_texture_rids; }
    const LocalVector<RID> &get_noise_luts() const override { return noise_lut_rids; }
    RID get_stroke_density_buffer() const override { return stroke_density_buffer; }
    uint32_t get_stroke_density_sample_count() const override { return stroke_density_sample_count; }

    // IPainterlyMaterialManager interface - Validation
    Vector<String> get_missing_resources() const override;

    // IPainterlyMaterialManager interface - Implementation info
    String get_name() const override { return "PainterlyMaterialManager"; }

protected:
    static void _bind_methods();

private:
    // State
    bool initialized = false;
    RenderingDevice *rd = nullptr;

    // Material reference
    Ref<PainterlyMaterial> material;
    bool material_dirty = false;

    // GPU resources
    LocalVector<RID> palette_texture_rids;
    LocalVector<RID> noise_lut_rids;
    RID stroke_density_buffer;
    uint32_t stroke_density_sample_count = 0;
    uint32_t stroke_density_buffer_size = 0;

    // Material change callback
    void _on_material_changed();
};

#endif // GS_PAINTERLY_MATERIAL_MANAGER_H
