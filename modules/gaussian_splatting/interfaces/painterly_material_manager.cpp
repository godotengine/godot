// Painterly Material Manager: Manages GPU resources for PainterlyMaterial
// Extracted from GaussianSplatRenderer::_update_painterly_gpu_resources()

#include "painterly_material_manager.h"
#include "../logger/gs_logger.h"
#include <cstring>

void PainterlyMaterialManager::_bind_methods() {
    ClassDB::bind_method(D_METHOD("is_dirty"), &PainterlyMaterialManager::is_dirty);
    ClassDB::bind_method(D_METHOD("mark_dirty"), &PainterlyMaterialManager::mark_dirty);
    ClassDB::bind_method(D_METHOD("update_resources"), &PainterlyMaterialManager::update_resources);
    ClassDB::bind_method(D_METHOD("clear_resources"), &PainterlyMaterialManager::clear_resources);
}

PainterlyMaterialManager::PainterlyMaterialManager() {
}

PainterlyMaterialManager::~PainterlyMaterialManager() {
    shutdown();
}

Error PainterlyMaterialManager::initialize(RenderingDevice *p_device) {
    if (!p_device) {
        return ERR_INVALID_PARAMETER;
    }

    rd = p_device;
    initialized = true;
    return OK;
}

void PainterlyMaterialManager::shutdown() {
    clear_resources();
    clear_material();

    initialized = false;
    rd = nullptr;
}

void PainterlyMaterialManager::set_material(const Ref<PainterlyMaterial> &p_material) {
    if (material == p_material) {
        return;
    }

    // Disconnect from old material
    if (material.is_valid() && material->is_connected("changed", callable_mp(this, &PainterlyMaterialManager::_on_material_changed))) {
        material->disconnect("changed", callable_mp(this, &PainterlyMaterialManager::_on_material_changed));
    }

    material = p_material;

    // Connect to new material
    if (material.is_valid() && !material->is_connected("changed", callable_mp(this, &PainterlyMaterialManager::_on_material_changed))) {
        material->connect("changed", callable_mp(this, &PainterlyMaterialManager::_on_material_changed));
    }

    material_dirty = true;
    update_resources();
}

void PainterlyMaterialManager::clear_material() {
    if (material.is_valid() && material->is_connected("changed", callable_mp(this, &PainterlyMaterialManager::_on_material_changed))) {
        material->disconnect("changed", callable_mp(this, &PainterlyMaterialManager::_on_material_changed));
    }
    material.unref();
    material_dirty = true;
}

void PainterlyMaterialManager::_on_material_changed() {
    material_dirty = true;
    update_resources();
}

void PainterlyMaterialManager::clear_resources() {
    palette_texture_rids.clear();
    noise_lut_rids.clear();
    stroke_density_sample_count = 0;
    stroke_density_buffer_size = 0;

    // IMPORTANT: Do NOT call free() on resources here!
    // Resources belong to a specific RenderingDevice that may have been destroyed.
    // Attempting to free with a different/stale device pointer causes "invalid ID" errors.
    // The resources will be cleaned up automatically when their owning device is destroyed.
    stroke_density_buffer = RID();
}

void PainterlyMaterialManager::update_resources() {
    if (!material_dirty) {
        return;
    }

    if (!initialized || !rd) {
        return;
    }

    if (!material.is_valid()) {
        clear_resources();
        material_dirty = false;
        return;
    }

    // Check for missing resources and warn
    Vector<String> missing = material->get_missing_resources();
    if (!missing.is_empty()) {
        String warning = "[PainterlyMaterialManager] Material missing required resources: ";
        const int missing_count = missing.size();
        for (int i = 0; i < missing_count; i++) {
            if (i > 0) {
                warning += ", ";
            }
            warning += missing[i];
        }
        GS_LOG_WARN_DEFAULT(warning);
    }

    // Update palette textures
    palette_texture_rids.clear();
    TypedArray<Texture2D> palette = material->get_palette_textures();
    const int palette_count = palette.size();
    for (int i = 0; i < palette_count; i++) {
        Ref<Texture2D> texture = palette[i];
        if (texture.is_valid()) {
            palette_texture_rids.push_back(texture->get_rid());
        }
    }

    // Update noise LUTs
    noise_lut_rids.clear();
    TypedArray<Texture2D> noise = material->get_noise_luts();
    const int noise_count = noise.size();
    for (int i = 0; i < noise_count; i++) {
        Ref<Texture2D> texture = noise[i];
        if (texture.is_valid()) {
            noise_lut_rids.push_back(texture->get_rid());
        }
    }

    // Update stroke density buffer
    PackedFloat32Array stroke_density = material->get_stroke_density_lut();
    stroke_density_sample_count = stroke_density.size();

    if (stroke_density_sample_count > 0) {
        Vector<uint8_t> data;
        data.resize(stroke_density_sample_count * sizeof(float));
        memcpy(data.ptrw(), stroke_density.ptr(), data.size());

        // Check if buffer needs to be recreated due to size change
        uint32_t required_size = data.size();
        bool needs_recreation = !stroke_density_buffer.is_valid() ||
                              (stroke_density_buffer_size != required_size);

        if (needs_recreation) {
            // Invalidate existing buffer if it exists (don't free - device owns resources)
            if (stroke_density_buffer.is_valid()) {
                stroke_density_buffer = RID();
                stroke_density_buffer_size = 0;
            }

            // Create new buffer with correct size
            stroke_density_buffer = rd->storage_buffer_create(data.size(), data);
            if (!stroke_density_buffer.is_valid()) {
                GS_LOG_WARN_DEFAULT("[PainterlyMaterialManager] Failed to allocate stroke density buffer");
                stroke_density_buffer_size = 0;
            } else {
                rd->set_resource_name(stroke_density_buffer, "GS_PainterlyMaterialManager_StrokeDensityBuffer");
                stroke_density_buffer_size = required_size;
            }
        } else {
            // Buffer size is correct, just update the data
            rd->buffer_update(stroke_density_buffer, 0, data.size(), data.ptr());
        }
    } else {
        // No samples, invalidate the buffer if it exists (don't free - device owns resources)
        if (stroke_density_buffer.is_valid()) {
            stroke_density_buffer = RID();
            stroke_density_buffer_size = 0;
        }
    }

    material_dirty = false;
}

PainterlyMaterialResources PainterlyMaterialManager::get_resources() const {
    PainterlyMaterialResources resources;

    for (uint32_t i = 0; i < palette_texture_rids.size(); i++) {
        resources.palette_texture_rids.push_back(palette_texture_rids[i]);
    }
    for (uint32_t i = 0; i < noise_lut_rids.size(); i++) {
        resources.noise_lut_rids.push_back(noise_lut_rids[i]);
    }
    resources.stroke_density_buffer = stroke_density_buffer;
    resources.stroke_density_sample_count = stroke_density_sample_count;
    resources.valid = material.is_valid() && initialized;

    return resources;
}

Vector<String> PainterlyMaterialManager::get_missing_resources() const {
    if (material.is_valid()) {
        return material->get_missing_resources();
    }
    return Vector<String>();
}
