#include "gaussian_splat_asset.h"
#include "../io/gaussian_data_loader.h"
#include "../core/gaussian_data.h"
#include "core/error/error_macros.h"
#include "core/io/file_access.h"
#include "core/math/basis.h"
#include "core/math/math_funcs.h"
#include "core/math/quaternion.h"
#include "../logger/gs_logger.h"

uint32_t GaussianSplatAsset::instance_count = 0;

void GaussianSplatAsset::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_asset_type", "type"), &GaussianSplatAsset::set_asset_type);
    ClassDB::bind_method(D_METHOD("get_asset_type"), &GaussianSplatAsset::get_asset_type);

    ClassDB::bind_method(D_METHOD("is_loaded"), &GaussianSplatAsset::is_loaded);

    ClassDB::bind_method(D_METHOD("set_splat_count", "count"), &GaussianSplatAsset::set_splat_count);
    ClassDB::bind_method(D_METHOD("get_splat_count"), &GaussianSplatAsset::get_splat_count);

    // Getters
    ClassDB::bind_method(D_METHOD("get_positions"), &GaussianSplatAsset::get_positions);
    ClassDB::bind_method(D_METHOD("get_position_vectors"), &GaussianSplatAsset::get_position_vectors);
    ClassDB::bind_method(D_METHOD("get_colors"), &GaussianSplatAsset::get_colors);
    ClassDB::bind_method(D_METHOD("get_scales"), &GaussianSplatAsset::get_scales);
    ClassDB::bind_method(D_METHOD("get_scale_vectors"), &GaussianSplatAsset::get_scale_vectors);
    ClassDB::bind_method(D_METHOD("get_rotations"), &GaussianSplatAsset::get_rotations);
    ClassDB::bind_method(D_METHOD("get_rotation_quaternions"), &GaussianSplatAsset::get_rotation_quaternions);
    ClassDB::bind_method(D_METHOD("get_sh_dc_coefficients"), &GaussianSplatAsset::get_sh_dc_coefficients);
    ClassDB::bind_method(D_METHOD("get_sh_first_order_coefficients"), &GaussianSplatAsset::get_sh_first_order_coefficients);
    ClassDB::bind_method(D_METHOD("get_sh_high_order_coefficients"), &GaussianSplatAsset::get_sh_high_order_coefficients);
    ClassDB::bind_method(D_METHOD("get_spherical_harmonics_buffer"), &GaussianSplatAsset::get_spherical_harmonics_buffer);
    ClassDB::bind_method(D_METHOD("get_opacity_logits"), &GaussianSplatAsset::get_opacity_logits);
    ClassDB::bind_method(D_METHOD("get_opacities"), &GaussianSplatAsset::get_opacities);
    ClassDB::bind_method(D_METHOD("get_palette_ids"), &GaussianSplatAsset::get_palette_ids);
    ClassDB::bind_method(D_METHOD("get_palette_ids_buffer"), &GaussianSplatAsset::get_palette_ids_buffer);
    ClassDB::bind_method(D_METHOD("get_painterly_flags"), &GaussianSplatAsset::get_painterly_flags);
    ClassDB::bind_method(D_METHOD("get_painterly_flags_buffer"), &GaussianSplatAsset::get_painterly_flags_buffer);
    ClassDB::bind_method(D_METHOD("get_brush_override_ids"), &GaussianSplatAsset::get_brush_override_ids);
    ClassDB::bind_method(D_METHOD("get_brush_override_ids_buffer"), &GaussianSplatAsset::get_brush_override_ids_buffer);
    ClassDB::bind_method(D_METHOD("get_normals"), &GaussianSplatAsset::get_normals);
    ClassDB::bind_method(D_METHOD("get_normal_vectors"), &GaussianSplatAsset::get_normal_vectors);
    ClassDB::bind_method(D_METHOD("get_brush_axes"), &GaussianSplatAsset::get_brush_axes);
    ClassDB::bind_method(D_METHOD("get_brush_axes_vector2"), &GaussianSplatAsset::get_brush_axes_vector2);
    ClassDB::bind_method(D_METHOD("get_stroke_ages"), &GaussianSplatAsset::get_stroke_ages);
    ClassDB::bind_method(D_METHOD("get_stroke_ages_buffer"), &GaussianSplatAsset::get_stroke_ages_buffer);
    ClassDB::bind_method(D_METHOD("get_sh_first_order_terms"), &GaussianSplatAsset::get_sh_first_order_terms);
    ClassDB::bind_method(D_METHOD("get_sh_high_order_terms"), &GaussianSplatAsset::get_sh_high_order_terms);

    // Setters - needed for loaders to populate data
    ClassDB::bind_method(D_METHOD("set_positions", "positions"), &GaussianSplatAsset::set_positions);
    ClassDB::bind_method(D_METHOD("set_colors", "colors"), &GaussianSplatAsset::set_colors);
    ClassDB::bind_method(D_METHOD("set_scales", "scales"), &GaussianSplatAsset::set_scales);
    ClassDB::bind_method(D_METHOD("set_rotations", "rotations"), &GaussianSplatAsset::set_rotations);
    ClassDB::bind_method(D_METHOD("set_sh_dc_coefficients", "coefficients"), &GaussianSplatAsset::set_sh_dc_coefficients);
    ClassDB::bind_method(D_METHOD("set_sh_first_order_coefficients", "coefficients"), &GaussianSplatAsset::set_sh_first_order_coefficients);
    ClassDB::bind_method(D_METHOD("set_sh_high_order_coefficients", "coefficients"), &GaussianSplatAsset::set_sh_high_order_coefficients);
    ClassDB::bind_method(D_METHOD("set_opacity_logits", "opacity_logits"), &GaussianSplatAsset::set_opacity_logits);
    ClassDB::bind_method(D_METHOD("set_palette_ids", "palette_ids"), &GaussianSplatAsset::set_palette_ids);
    ClassDB::bind_method(D_METHOD("set_painterly_flags", "painterly_flags"), &GaussianSplatAsset::set_painterly_flags);
    ClassDB::bind_method(D_METHOD("set_brush_override_ids", "brush_override_ids"), &GaussianSplatAsset::set_brush_override_ids);
    ClassDB::bind_method(D_METHOD("set_normals", "normals"), &GaussianSplatAsset::set_normals);
    ClassDB::bind_method(D_METHOD("set_brush_axes", "brush_axes"), &GaussianSplatAsset::set_brush_axes);
    ClassDB::bind_method(D_METHOD("set_stroke_ages", "stroke_ages"), &GaussianSplatAsset::set_stroke_ages);
    ClassDB::bind_method(D_METHOD("set_sh_component_terms", "first_order_terms", "high_order_terms"), &GaussianSplatAsset::set_sh_component_terms);

    ClassDB::bind_method(D_METHOD("set_import_metadata", "metadata"), &GaussianSplatAsset::set_import_metadata);
    ClassDB::bind_method(D_METHOD("get_import_metadata"), &GaussianSplatAsset::get_import_metadata);
    ClassDB::bind_method(D_METHOD("set_import_quality_preset", "preset"), &GaussianSplatAsset::set_import_quality_preset);
    ClassDB::bind_method(D_METHOD("get_import_quality_preset"), &GaussianSplatAsset::get_import_quality_preset);
    ClassDB::bind_method(D_METHOD("set_compression_flags", "flags"), &GaussianSplatAsset::set_compression_flags);
    ClassDB::bind_method(D_METHOD("get_compression_flags"), &GaussianSplatAsset::get_compression_flags);
    ClassDB::bind_method(D_METHOD("set_thumbnail", "texture"), &GaussianSplatAsset::set_thumbnail);
    ClassDB::bind_method(D_METHOD("get_thumbnail"), &GaussianSplatAsset::get_thumbnail);
    ClassDB::bind_method(D_METHOD("set_source_path", "path"), &GaussianSplatAsset::set_source_path);
    ClassDB::bind_method(D_METHOD("get_source_path"), &GaussianSplatAsset::get_source_path);
    ClassDB::bind_method(D_METHOD("load_from_file", "path"), &GaussianSplatAsset::load_from_file);
    ClassDB::bind_method(D_METHOD("save_to_file", "path"), &GaussianSplatAsset::save_to_file);

    ClassDB::bind_static_method("GaussianSplatAsset", D_METHOD("get_instance_count"), &GaussianSplatAsset::get_instance_count);

    ADD_PROPERTY(PropertyInfo(Variant::INT, "asset_type", PROPERTY_HINT_ENUM, "Static,Dynamic"), "set_asset_type", "get_asset_type");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "splat_count"), "set_splat_count", "get_splat_count");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "import/quality_preset", PROPERTY_HINT_ENUM, "low,medium,high,ultra,custom"),
            "set_import_quality_preset", "get_import_quality_preset");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "import/compression_flags", PROPERTY_HINT_FLAGS, "Positions,Colors,Scales,Rotations"),
            "set_compression_flags", "get_compression_flags");
    ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "import/metadata"), "set_import_metadata", "get_import_metadata");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "import/thumbnail", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"),
            "set_thumbnail", "get_thumbnail");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "import/source_path", PROPERTY_HINT_FILE, "*.ply,*.spz"),
            "set_source_path", "get_source_path");
    ADD_PROPERTY(PropertyInfo(Variant::PACKED_FLOAT32_ARRAY, "data/positions"), "set_positions", "get_positions");
    ADD_PROPERTY(PropertyInfo(Variant::PACKED_COLOR_ARRAY, "data/colors"), "set_colors", "get_colors");
    ADD_PROPERTY(PropertyInfo(Variant::PACKED_FLOAT32_ARRAY, "data/scales"), "set_scales", "get_scales");
    ADD_PROPERTY(PropertyInfo(Variant::PACKED_FLOAT32_ARRAY, "data/rotations"), "set_rotations", "get_rotations");
    ADD_PROPERTY(PropertyInfo(Variant::PACKED_FLOAT32_ARRAY, "data/sh_dc"), "set_sh_dc_coefficients", "get_sh_dc_coefficients");
    ADD_PROPERTY(PropertyInfo(Variant::PACKED_FLOAT32_ARRAY, "data/sh_first_order"), "set_sh_first_order_coefficients", "get_sh_first_order_coefficients");
    ADD_PROPERTY(PropertyInfo(Variant::PACKED_FLOAT32_ARRAY, "data/sh_high_order"), "set_sh_high_order_coefficients", "get_sh_high_order_coefficients");
    ADD_PROPERTY(PropertyInfo(Variant::PACKED_FLOAT32_ARRAY, "data/opacity_logits"), "set_opacity_logits", "get_opacity_logits");
    ADD_PROPERTY(PropertyInfo(Variant::PACKED_INT32_ARRAY, "data/palette_ids"), "set_palette_ids", "get_palette_ids");
    ADD_PROPERTY(PropertyInfo(Variant::PACKED_INT32_ARRAY, "data/painterly_flags"), "set_painterly_flags", "get_painterly_flags");
    ADD_PROPERTY(PropertyInfo(Variant::PACKED_FLOAT32_ARRAY, "data/normals"), "set_normals", "get_normals");
    ADD_PROPERTY(PropertyInfo(Variant::PACKED_FLOAT32_ARRAY, "data/brush_axes"), "set_brush_axes", "get_brush_axes");
    ADD_PROPERTY(PropertyInfo(Variant::PACKED_FLOAT32_ARRAY, "data/stroke_ages"), "set_stroke_ages", "get_stroke_ages");

    BIND_ENUM_CONSTANT(ASSET_TYPE_STATIC);
    BIND_ENUM_CONSTANT(ASSET_TYPE_DYNAMIC);
    BIND_ENUM_CONSTANT(COMPRESSION_NONE);
    BIND_ENUM_CONSTANT(COMPRESSION_POSITIONS);
    BIND_ENUM_CONSTANT(COMPRESSION_COLORS);
    BIND_ENUM_CONSTANT(COMPRESSION_SCALES);
    BIND_ENUM_CONSTANT(COMPRESSION_ROTATIONS);
}

GaussianSplatAsset::GaussianSplatAsset() {
    instance_count++;
}

GaussianSplatAsset::~GaussianSplatAsset() {
    instance_count--;
}

void GaussianSplatAsset::_invalidate_gaussian_data_cache() {
    gaussian_data_cache.unref();
}

void GaussianSplatAsset::_invalidate_bounds_metadata() {
    import_metadata.erase(StringName("bounds"));
    import_metadata[StringName("bounds_dirty")] = true;
}

void GaussianSplatAsset::set_asset_type(AssetType p_type) {
    if (asset_type != p_type) {
        asset_type = p_type;
        emit_changed();
    }
}

void GaussianSplatAsset::set_splat_count(uint32_t p_count) {
    if (splat_count != p_count) {
        splat_count = p_count;
        _ensure_buffer_sizes();
        import_metadata[StringName("splat_count")] = (int)p_count;
        _invalidate_bounds_metadata();
        _invalidate_gaussian_data_cache();
        emit_changed();
    }
}

// ---------------------------------------------------------------------------
// Raw-array getters: warn once when the asset has no loaded data so that
// callers can distinguish "empty because unloaded" from "legitimately empty".
// These use WARN_PRINT_ONCE because they may be called per-frame.
// ---------------------------------------------------------------------------

PackedFloat32Array GaussianSplatAsset::get_positions() const {
    if (splat_count == 0 && positions.is_empty()) {
        WARN_PRINT_ONCE("[GaussianSplatAsset] get_positions() called on unloaded asset; returning empty array.");
    }
    return positions;
}

PackedColorArray GaussianSplatAsset::get_colors() const {
    if (splat_count == 0 && colors.is_empty()) {
        WARN_PRINT_ONCE("[GaussianSplatAsset] get_colors() called on unloaded asset; returning empty array.");
    }
    return colors;
}

PackedFloat32Array GaussianSplatAsset::get_scales() const {
    if (splat_count == 0 && scales.is_empty()) {
        WARN_PRINT_ONCE("[GaussianSplatAsset] get_scales() called on unloaded asset; returning empty array.");
    }
    return scales;
}

PackedFloat32Array GaussianSplatAsset::get_rotations() const {
    if (splat_count == 0 && rotations.is_empty()) {
        WARN_PRINT_ONCE("[GaussianSplatAsset] get_rotations() called on unloaded asset; returning empty array.");
    }
    return rotations;
}

PackedFloat32Array GaussianSplatAsset::get_sh_dc_coefficients() const {
    if (splat_count == 0 && !has_sh_dc_coefficients) {
        WARN_PRINT_ONCE("[GaussianSplatAsset] get_sh_dc_coefficients() called on unloaded asset; returning empty array.");
    }
    return has_sh_dc_coefficients ? sh_dc_coefficients : PackedFloat32Array();
}

PackedFloat32Array GaussianSplatAsset::get_sh_first_order_coefficients() const {
    if (splat_count == 0 && sh_first_order_coefficients.is_empty()) {
        WARN_PRINT_ONCE("[GaussianSplatAsset] get_sh_first_order_coefficients() called on unloaded asset; returning empty array.");
    }
    return sh_first_order_coefficients;
}

PackedFloat32Array GaussianSplatAsset::get_sh_high_order_coefficients() const {
    if (splat_count == 0 && sh_high_order_coefficients.is_empty()) {
        WARN_PRINT_ONCE("[GaussianSplatAsset] get_sh_high_order_coefficients() called on unloaded asset; returning empty array.");
    }
    return sh_high_order_coefficients;
}

PackedFloat32Array GaussianSplatAsset::get_opacity_logits() const {
    if (splat_count == 0 && opacity_logits.is_empty()) {
        WARN_PRINT_ONCE("[GaussianSplatAsset] get_opacity_logits() called on unloaded asset; returning empty array.");
    }
    return opacity_logits;
}

PackedInt32Array GaussianSplatAsset::get_palette_ids() const {
    if (splat_count == 0 && palette_ids.is_empty()) {
        WARN_PRINT_ONCE("[GaussianSplatAsset] get_palette_ids() called on unloaded asset; returning empty array.");
    }
    return palette_ids;
}

PackedInt32Array GaussianSplatAsset::get_painterly_flags() const {
    if (splat_count == 0 && painterly_flags.is_empty()) {
        WARN_PRINT_ONCE("[GaussianSplatAsset] get_painterly_flags() called on unloaded asset; returning empty array.");
    }
    return painterly_flags;
}

PackedInt32Array GaussianSplatAsset::get_brush_override_ids() const {
    if (splat_count == 0 && painterly_flags.is_empty()) {
        WARN_PRINT_ONCE("[GaussianSplatAsset] get_brush_override_ids() called on unloaded asset; returning empty array.");
    }
    return painterly_flags;
}

PackedFloat32Array GaussianSplatAsset::get_normals() const {
    if (splat_count == 0 && normals.is_empty()) {
        WARN_PRINT_ONCE("[GaussianSplatAsset] get_normals() called on unloaded asset; returning empty array.");
    }
    return normals;
}

PackedFloat32Array GaussianSplatAsset::get_brush_axes() const {
    if (splat_count == 0 && brush_axes.is_empty()) {
        WARN_PRINT_ONCE("[GaussianSplatAsset] get_brush_axes() called on unloaded asset; returning empty array.");
    }
    return brush_axes;
}

PackedFloat32Array GaussianSplatAsset::get_stroke_ages() const {
    if (splat_count == 0 && stroke_ages.is_empty()) {
        WARN_PRINT_ONCE("[GaussianSplatAsset] get_stroke_ages() called on unloaded asset; returning empty array.");
    }
    return stroke_ages;
}

// ---------------------------------------------------------------------------
// Structured getters: these convert raw data into higher-level types and
// silently fill fallback values when individual splat data is missing.
// They use WARN_PRINT_ONCE for the "asset not loaded at all" case.
// ---------------------------------------------------------------------------

PackedVector3Array GaussianSplatAsset::get_position_vectors() const {
    PackedVector3Array result;
    if (splat_count == 0) {
        WARN_PRINT_ONCE("[GaussianSplatAsset] get_position_vectors() called on unloaded asset (splat_count == 0); returning empty array.");
        return result;
    }

    result.resize(splat_count);
    Vector3 *write = result.ptrw();
    const float *read = positions.ptr();
    const int available = positions.size();

    if (available >= int(splat_count) * 3 && read != nullptr) {
        for (uint32_t i = 0; i < splat_count; i++) {
            const uint32_t base = i * 3u;
            write[i] = Vector3(read[base + 0], read[base + 1], read[base + 2]);
        }
        return result;
    }

    for (uint32_t i = 0; i < splat_count; i++) {
        const uint32_t base = i * 3u;
        if (available >= int(base + 3u) && read != nullptr) {
            write[i] = Vector3(read[base + 0], read[base + 1], read[base + 2]);
        } else {
            write[i] = Vector3();
        }
    }

    return result;
}

PackedVector3Array GaussianSplatAsset::get_scale_vectors() const {
    PackedVector3Array result;
    if (splat_count == 0) {
        WARN_PRINT_ONCE("[GaussianSplatAsset] get_scale_vectors() called on unloaded asset (splat_count == 0); returning empty array.");
        return result;
    }

    result.resize(splat_count);
    Vector3 *write = result.ptrw();
    const float *read = scales.ptr();
    const int available = scales.size();

    if (available >= int(splat_count) * 3 && read != nullptr) {
        for (uint32_t i = 0; i < splat_count; i++) {
            const uint32_t base = i * 3u;
            write[i] = Vector3(read[base + 0], read[base + 1], read[base + 2]);
        }
        return result;
    }

    for (uint32_t i = 0; i < splat_count; i++) {
        const uint32_t base = i * 3u;
        if (available >= int(base + 3u) && read != nullptr) {
            write[i] = Vector3(read[base + 0], read[base + 1], read[base + 2]);
        } else {
            write[i] = Vector3(1.0f, 1.0f, 1.0f);
        }
    }

    return result;
}

TypedArray<Quaternion> GaussianSplatAsset::get_rotation_quaternions() const {
    TypedArray<Quaternion> result;
    if (splat_count == 0) {
        WARN_PRINT_ONCE("[GaussianSplatAsset] get_rotation_quaternions() called on unloaded asset (splat_count == 0); returning empty array.");
        return result;
    }

    result.resize(splat_count);
    const int available = rotations.size();

    for (uint32_t i = 0; i < splat_count; i++) {
        if (available >= int(i * 4 + 4)) {
            const float w = rotations[i * 4 + 0];
            const float x = rotations[i * 4 + 1];
            const float y = rotations[i * 4 + 2];
            const float z = rotations[i * 4 + 3];
            const float len_sq = w * w + x * x + y * y + z * z;
            if (Math::is_zero_approx(len_sq)) {
                result[i] = Quaternion();
            } else {
                result[i] = Quaternion(x, y, z, w);
            }
        } else {
            result[i] = Quaternion();
        }
    }

    return result;
}

PackedFloat32Array GaussianSplatAsset::get_spherical_harmonics_buffer() const {
    PackedFloat32Array result;
    if (splat_count == 0) {
        WARN_PRINT_ONCE("[GaussianSplatAsset] get_spherical_harmonics_buffer() called on unloaded asset (splat_count == 0); returning empty array.");
        return result;
    }

    const uint32_t first_terms = sh_first_order_terms;
    const uint32_t high_terms = sh_high_order_terms;
    const uint32_t total_terms = 1 + first_terms + high_terms;

    result.resize(int64_t(splat_count) * int64_t(total_terms) * 3);
    float *write = result.ptrw();
    const float *dc_read = has_sh_dc_coefficients ? sh_dc_coefficients.ptr() : nullptr;
    const float *first_read = sh_first_order_coefficients.ptr();
    const float *high_read = sh_high_order_coefficients.ptr();
    const Color *color_read = colors.ptr();

    const int dc_available = has_sh_dc_coefficients ? sh_dc_coefficients.size() : 0;
    const int first_available = sh_first_order_coefficients.size();
    const int high_available = sh_high_order_coefficients.size();

    for (uint32_t i = 0; i < splat_count; i++) {
        int offset = int(i * total_terms * 3);

        if (dc_available >= int(i * 3 + 3) && dc_read != nullptr) {
            write[offset + 0] = dc_read[i * 3 + 0];
            write[offset + 1] = dc_read[i * 3 + 1];
            write[offset + 2] = dc_read[i * 3 + 2];
        } else if (i < (uint32_t)colors.size() && color_read != nullptr) {
            const Color color = color_read[i];
            write[offset + 0] = color.r;
            write[offset + 1] = color.g;
            write[offset + 2] = color.b;
        } else {
            write[offset + 0] = 1.0f;
            write[offset + 1] = 1.0f;
            write[offset + 2] = 1.0f;
        }

        offset += 3;

        if (first_terms > 0) {
            const int stride = int(first_terms * 3);
            const int base = int(i) * stride;
            for (uint32_t term = 0; term < first_terms; term++) {
                if (first_available >= base + int(term * 3 + 3) && first_read != nullptr) {
                    write[offset + 0] = first_read[base + term * 3 + 0];
                    write[offset + 1] = first_read[base + term * 3 + 1];
                    write[offset + 2] = first_read[base + term * 3 + 2];
                } else {
                    write[offset + 0] = 0.0f;
                    write[offset + 1] = 0.0f;
                    write[offset + 2] = 0.0f;
                }
                offset += 3;
            }
        }

        if (high_terms > 0) {
            const int stride = int(high_terms * 3);
            const int base = int(i) * stride;
            for (uint32_t term = 0; term < high_terms; term++) {
                if (high_available >= base + int(term * 3 + 3) && high_read != nullptr) {
                    write[offset + 0] = high_read[base + term * 3 + 0];
                    write[offset + 1] = high_read[base + term * 3 + 1];
                    write[offset + 2] = high_read[base + term * 3 + 2];
                } else {
                    write[offset + 0] = 0.0f;
                    write[offset + 1] = 0.0f;
                    write[offset + 2] = 0.0f;
                }
                offset += 3;
            }
        }
    }

    return result;
}

PackedFloat32Array GaussianSplatAsset::get_opacities() const {
    PackedFloat32Array result;
    if (splat_count == 0) {
        WARN_PRINT_ONCE("[GaussianSplatAsset] get_opacities() called on unloaded asset (splat_count == 0); returning empty array.");
        return result;
    }

    result.resize(splat_count);
    float *write = result.ptrw();
    const float *logit_read = opacity_logits.ptr();
    const Color *color_read = colors.ptr();

    const int logit_available = opacity_logits.size();

    for (uint32_t i = 0; i < splat_count; i++) {
        float opacity_value = 1.0f;
        if (logit_available > int(i) && logit_read != nullptr) {
            const float logit = logit_read[i];
            const float exp_value = Math::exp(-logit);
            opacity_value = 1.0f / (1.0f + exp_value);
        } else if (i < (uint32_t)colors.size() && color_read != nullptr) {
            opacity_value = color_read[i].a;
        }

        write[i] = CLAMP(opacity_value, 0.0f, 1.0f);
    }

    return result;
}

PackedInt32Array GaussianSplatAsset::get_palette_ids_buffer() const {
    PackedInt32Array result;
    if (splat_count == 0) {
        WARN_PRINT_ONCE("[GaussianSplatAsset] get_palette_ids_buffer() called on unloaded asset (splat_count == 0); returning empty array.");
        return result;
    }

    result.resize(splat_count);
    int32_t *write = result.ptrw();
    const int available = palette_ids.size();

    for (uint32_t i = 0; i < splat_count; i++) {
        int32_t value = (available > int(i)) ? palette_ids[i] : 0;
        write[i] = CLAMP(value, 0, 65535);
    }

    return result;
}

PackedInt32Array GaussianSplatAsset::get_painterly_flags_buffer() const {
    PackedInt32Array result;
    if (splat_count == 0) {
        WARN_PRINT_ONCE("[GaussianSplatAsset] get_painterly_flags_buffer() called on unloaded asset (splat_count == 0); returning empty array.");
        return result;
    }

    result.resize(splat_count);
    int32_t *write = result.ptrw();
    const int available = painterly_flags.size();

    for (uint32_t i = 0; i < splat_count; i++) {
        int32_t value = (available > int(i)) ? painterly_flags[i] : 0;
        write[i] = CLAMP(value, 0, 65535);
    }

    return result;
}

PackedInt32Array GaussianSplatAsset::get_brush_override_ids_buffer() const {
    return get_painterly_flags_buffer();
}

PackedVector3Array GaussianSplatAsset::get_normal_vectors() const {
    PackedVector3Array result;
    if (splat_count == 0) {
        WARN_PRINT_ONCE("[GaussianSplatAsset] get_normal_vectors() called on unloaded asset (splat_count == 0); returning empty array.");
        return result;
    }

    result.resize(splat_count);
    Vector3 *write = result.ptrw();
    const float *read = normals.ptr();
    const int available = normals.size();

    for (uint32_t i = 0; i < splat_count; i++) {
        if (available >= int(i * 3 + 3) && read != nullptr) {
            write[i] = Vector3(read[i * 3 + 0], read[i * 3 + 1], read[i * 3 + 2]);
        } else {
            write[i] = Vector3(0.0f, 1.0f, 0.0f);
        }
    }

    return result;
}

PackedVector2Array GaussianSplatAsset::get_brush_axes_vector2() const {
    PackedVector2Array result;
    if (splat_count == 0) {
        WARN_PRINT_ONCE("[GaussianSplatAsset] get_brush_axes_vector2() called on unloaded asset (splat_count == 0); returning empty array.");
        return result;
    }

    result.resize(splat_count);
    Vector2 *write = result.ptrw();
    const float *read = brush_axes.ptr();
    const int available = brush_axes.size();

    for (uint32_t i = 0; i < splat_count; i++) {
        if (available >= int(i * 2 + 2) && read != nullptr) {
            write[i] = Vector2(read[i * 2 + 0], read[i * 2 + 1]);
        } else {
            write[i] = Vector2(1.0f, 1.0f);
        }
    }

    return result;
}

PackedFloat32Array GaussianSplatAsset::get_stroke_ages_buffer() const {
    PackedFloat32Array result;
    if (splat_count == 0) {
        WARN_PRINT_ONCE("[GaussianSplatAsset] get_stroke_ages_buffer() called on unloaded asset (splat_count == 0); returning empty array.");
        return result;
    }

    result.resize(splat_count);
    float *write = result.ptrw();
    const int available = stroke_ages.size();

    for (uint32_t i = 0; i < splat_count; i++) {
        write[i] = (available > int(i)) ? stroke_ages[i] : 0.0f;
    }

    return result;
}

void GaussianSplatAsset::set_positions(const PackedFloat32Array &p_positions) {
    positions = p_positions;
    // Always update splat count based on position array size (3 floats per splat)
    uint32_t new_count = p_positions.size() / 3;
    if (splat_count != new_count) {
        splat_count = new_count;
    }
    _ensure_buffer_sizes();
    import_metadata[StringName("splat_count")] = (int)splat_count;
    _invalidate_bounds_metadata();
    _invalidate_gaussian_data_cache();
    emit_changed();
}

void GaussianSplatAsset::set_colors(const PackedColorArray &p_colors) {
    colors = p_colors;
    // Update splat count if not already set by positions
    if (splat_count == 0 && !p_colors.is_empty()) {
        splat_count = p_colors.size();
    }
    _ensure_buffer_sizes();
    import_metadata[StringName("splat_count")] = (int)splat_count;
    _invalidate_gaussian_data_cache();
    emit_changed();
}

void GaussianSplatAsset::set_scales(const PackedFloat32Array &p_scales) {
    scales = p_scales;
    // Update splat count if not already set
    if (splat_count == 0 && !p_scales.is_empty()) {
        splat_count = p_scales.size() / 3;
    }
    _ensure_buffer_sizes();
    import_metadata[StringName("splat_count")] = (int)splat_count;
    _invalidate_bounds_metadata();
    _invalidate_gaussian_data_cache();
    emit_changed();
}

void GaussianSplatAsset::set_rotations(const PackedFloat32Array &p_rotations) {
    rotations = p_rotations;
    // Update splat count if not already set
    if (splat_count == 0 && !p_rotations.is_empty()) {
        splat_count = p_rotations.size() / 4;
    }
    _ensure_buffer_sizes();
    import_metadata[StringName("splat_count")] = (int)splat_count;
    _invalidate_bounds_metadata();
    _invalidate_gaussian_data_cache();
    emit_changed();
}

void GaussianSplatAsset::set_sh_dc_coefficients(const PackedFloat32Array &p_coefficients) {
    sh_dc_coefficients = p_coefficients;
    has_sh_dc_coefficients = !p_coefficients.is_empty();
    if (splat_count == 0 && p_coefficients.size() >= 3) {
        splat_count = p_coefficients.size() / 3;
    }
    _ensure_buffer_sizes();
    import_metadata[StringName("splat_count")] = (int)splat_count;
    _invalidate_gaussian_data_cache();
    emit_changed();
}

void GaussianSplatAsset::set_sh_first_order_coefficients(const PackedFloat32Array &p_coefficients) {
    sh_first_order_coefficients = p_coefficients;
    if (splat_count > 0 && !p_coefficients.is_empty()) {
        sh_first_order_terms = MIN<uint32_t>(p_coefficients.size() / (splat_count * 3), 3u);
    } else if (p_coefficients.is_empty()) {
        sh_first_order_terms = 0;
    }
    _ensure_buffer_sizes();
    import_metadata[StringName("sh_first_order_terms")] = (int)sh_first_order_terms;
    _invalidate_gaussian_data_cache();
    emit_changed();
}

void GaussianSplatAsset::set_sh_high_order_coefficients(const PackedFloat32Array &p_coefficients) {
    sh_high_order_coefficients = p_coefficients;
    if (splat_count > 0 && !p_coefficients.is_empty()) {
        sh_high_order_terms = p_coefficients.size() / (splat_count * 3);
    } else if (p_coefficients.is_empty()) {
        sh_high_order_terms = 0;
    }
    _ensure_buffer_sizes();
    import_metadata[StringName("sh_high_order_terms")] = (int)sh_high_order_terms;
    _invalidate_gaussian_data_cache();
    emit_changed();
}

void GaussianSplatAsset::set_opacity_logits(const PackedFloat32Array &p_opacity_logits) {
    opacity_logits = p_opacity_logits;
    if (splat_count == 0 && !p_opacity_logits.is_empty()) {
        splat_count = p_opacity_logits.size();
    }
    _ensure_buffer_sizes();
    import_metadata[StringName("opacity_encoding")] = StringName("logit");
    _invalidate_gaussian_data_cache();
    emit_changed();
}

void GaussianSplatAsset::set_palette_ids(const PackedInt32Array &p_palette_ids) {
    palette_ids = p_palette_ids;
    if (splat_count == 0 && !p_palette_ids.is_empty()) {
        splat_count = p_palette_ids.size();
    }
    _ensure_buffer_sizes();
    import_metadata[StringName("has_palette_ids")] = palette_ids.size() == splat_count;
    _invalidate_gaussian_data_cache();
    emit_changed();
}

void GaussianSplatAsset::set_painterly_flags(const PackedInt32Array &p_flags) {
    painterly_flags = p_flags;
    if (splat_count == 0 && !p_flags.is_empty()) {
        splat_count = p_flags.size();
    }
    _ensure_buffer_sizes();
    const bool has_painterly_lane = painterly_flags.size() == splat_count;
    import_metadata[StringName("has_painterly_flags")] = has_painterly_lane;
    import_metadata[StringName("has_brush_override_ids")] = has_painterly_lane;
    _invalidate_gaussian_data_cache();
    emit_changed();
}

void GaussianSplatAsset::set_brush_override_ids(const PackedInt32Array &p_override_ids) {
    set_painterly_flags(p_override_ids);
}

void GaussianSplatAsset::set_normals(const PackedFloat32Array &p_normals) {
    normals = p_normals;
    if (splat_count == 0 && p_normals.size() >= 3) {
        splat_count = p_normals.size() / 3;
    }
    _ensure_buffer_sizes();
    import_metadata[StringName("has_normals")] = normals.size() >= splat_count * 3;
    _invalidate_gaussian_data_cache();
    emit_changed();
}

void GaussianSplatAsset::set_brush_axes(const PackedFloat32Array &p_brush_axes) {
    brush_axes = p_brush_axes;
    if (splat_count == 0 && p_brush_axes.size() >= 2) {
        splat_count = p_brush_axes.size() / 2;
    }
    _ensure_buffer_sizes();
    import_metadata[StringName("has_brush_axes")] = brush_axes.size() >= splat_count * 2;
    _invalidate_gaussian_data_cache();
    emit_changed();
}

void GaussianSplatAsset::set_stroke_ages(const PackedFloat32Array &p_stroke_ages) {
    stroke_ages = p_stroke_ages;
    if (splat_count == 0 && !p_stroke_ages.is_empty()) {
        splat_count = p_stroke_ages.size();
    }
    _ensure_buffer_sizes();
    import_metadata[StringName("has_stroke_age")] = stroke_ages.size() == splat_count;
    _invalidate_gaussian_data_cache();
    emit_changed();
}

void GaussianSplatAsset::set_sh_component_terms(uint32_t p_first_order_terms, uint32_t p_high_order_terms) {
    if (sh_first_order_terms == p_first_order_terms && sh_high_order_terms == p_high_order_terms) {
        return;
    }
    sh_first_order_terms = MIN<uint32_t>(p_first_order_terms, 3u);
    sh_high_order_terms = p_high_order_terms;
    _ensure_buffer_sizes();
    import_metadata[StringName("sh_first_order_terms")] = (int)sh_first_order_terms;
    import_metadata[StringName("sh_high_order_terms")] = (int)sh_high_order_terms;
    _invalidate_gaussian_data_cache();
    emit_changed();
}

void GaussianSplatAsset::_recalculate_sh_component_counts() {
    if (splat_count > 0) {
        if (!sh_first_order_coefficients.is_empty()) {
            sh_first_order_terms = MIN<uint32_t>(sh_first_order_coefficients.size() / (splat_count * 3), 3u);
        } else {
            sh_first_order_terms = 0;
        }
        if (!sh_high_order_coefficients.is_empty()) {
            sh_high_order_terms = sh_high_order_coefficients.size() / (splat_count * 3);
        } else {
            sh_high_order_terms = 0;
        }
    } else {
        if (sh_first_order_coefficients.is_empty()) {
            sh_first_order_terms = 0;
        }
        if (sh_high_order_coefficients.is_empty()) {
            sh_high_order_terms = 0;
        }
    }
}

void GaussianSplatAsset::_ensure_buffer_sizes() {
    const uint32_t count = splat_count;
    const int old_scale_size = scales.size();
    const int old_rotation_size = rotations.size();
    positions.resize(count * 3);
    colors.resize(count);
    scales.resize(count * 3);
    rotations.resize(count * 4);
    if (scales.size() > old_scale_size) {
        float *scale_ptr = scales.ptrw();
        int start = old_scale_size;
        if (start < 0) {
            start = 0;
        }
        for (int i = start; i < scales.size(); i += 3) {
            scale_ptr[i + 0] = 1.0f;
            if (i + 1 < scales.size()) {
                scale_ptr[i + 1] = 1.0f;
            }
            if (i + 2 < scales.size()) {
                scale_ptr[i + 2] = 1.0f;
            }
        }
    }
    if (rotations.size() > old_rotation_size) {
        float *rot_ptr = rotations.ptrw();
        int start = old_rotation_size;
        if (start < 0) {
            start = 0;
        }
        for (int i = start; i < rotations.size(); i += 4) {
            rot_ptr[i + 0] = 1.0f; // w
            if (i + 1 < rotations.size()) {
                rot_ptr[i + 1] = 0.0f;
            }
            if (i + 2 < rotations.size()) {
                rot_ptr[i + 2] = 0.0f;
            }
            if (i + 3 < rotations.size()) {
                rot_ptr[i + 3] = 0.0f;
            }
        }
    }
    if (has_sh_dc_coefficients) {
        sh_dc_coefficients.resize(count * 3);
    } else {
        sh_dc_coefficients.resize(0);
    }
    sh_first_order_coefficients.resize(count * sh_first_order_terms * 3);
    sh_high_order_coefficients.resize(count * sh_high_order_terms * 3);
    opacity_logits.resize(count);
    palette_ids.resize(count);
    painterly_flags.resize(count);
    normals.resize(count * 3);
    brush_axes.resize(count * 2);
    stroke_ages.resize(count);

    _recalculate_sh_component_counts();
}

void GaussianSplatAsset::set_import_metadata(const Dictionary &p_metadata) {
    import_metadata = p_metadata;
    import_metadata[StringName("splat_count")] = (int)splat_count;
    import_metadata[StringName("quality_preset")] = import_quality_preset;
    import_metadata[StringName("compression_flags")] = (int)compression_flags;
    _invalidate_gaussian_data_cache();
    emit_changed();
}

void GaussianSplatAsset::set_import_quality_preset(const String &p_preset) {
    String lower = p_preset.to_lower();
    if (import_quality_preset == lower) {
        return;
    }
    import_quality_preset = lower;
    import_metadata[StringName("quality_preset")] = import_quality_preset;
    emit_changed();
}

void GaussianSplatAsset::set_compression_flags(uint32_t p_flags) {
    if (compression_flags == p_flags) {
        return;
    }
    compression_flags = p_flags;
    import_metadata[StringName("compression_flags")] = (int)compression_flags;
    emit_changed();
}

void GaussianSplatAsset::set_thumbnail(const Ref<Texture2D> &p_thumbnail) {
    if (thumbnail == p_thumbnail) {
        return;
    }
    thumbnail = p_thumbnail;
    import_metadata[StringName("has_thumbnail")] = thumbnail.is_valid();
    emit_changed();
}

void GaussianSplatAsset::set_source_path(const String &p_path) {
    if (import_metadata.has(StringName("source_path")) && (String)import_metadata[StringName("source_path")] == p_path) {
        return;
    }
    import_metadata[StringName("source_path")] = p_path;
    emit_changed();
}

String GaussianSplatAsset::get_source_path() const {
    if (import_metadata.has(StringName("source_path"))) {
        return (String)import_metadata[StringName("source_path")];
    }
    return String();
}

Error GaussianSplatAsset::load_from_file(const String &p_path) {
    if (!FileAccess::exists(p_path)) {
        GS_LOG_ERROR_DEFAULT("Gaussian splat file not found: " + p_path);
        return ERR_FILE_NOT_FOUND;
    }

    GaussianDataLoadResult load_result;
    Error err = load_gaussian_data_from_file(p_path, load_result);
    if (err != OK) {
        GS_LOG_ERROR_DEFAULT("Failed to load splat file: " + p_path);
        return err;
    }

    if (load_result.used_ply) {
        if (!load_result.missing_optional.is_empty()) {
            for (int i = 0; i < load_result.missing_optional.size(); i++) {
                GS_LOG_STREAMING_DEBUG(vformat("PLY load: missing optional data %s", load_result.missing_optional[i]));
            }
        }

        if (!load_result.missing_required.is_empty()) {
            String missing_required_text;
            for (int i = 0; i < load_result.missing_required.size(); i++) {
                if (i > 0) {
                    missing_required_text += ", ";
                }
                missing_required_text += load_result.missing_required[i];
            }
            GS_LOG_ERROR_DEFAULT(vformat("PLY file missing required properties: %s", missing_required_text));
            return ERR_FILE_CORRUPT;
        }
    }

    Ref<::GaussianData> gaussian_data = load_result.data;
    Error populate_err = populate_from_gaussian_data(gaussian_data);
    if (populate_err != OK) {
        return populate_err;
    }

    const String file_label = load_result.used_spz ? "SPZ" : "PLY";
    GS_LOG_STREAMING_INFO(vformat("Loaded %s file: %d splats from %s", file_label, splat_count, p_path));
    return OK;
}

Error GaussianSplatAsset::save_to_file(const String &p_path) const {
    Ref<::GaussianData> data = get_gaussian_data();
    if (data.is_null()) {
        return ERR_INVALID_DATA;
    }
    return data->save_to_file(p_path);
}

Ref<::GaussianData> GaussianSplatAsset::get_gaussian_data() const {
    ERR_FAIL_COND_V_MSG(splat_count == 0, Ref<::GaussianData>(),
            "[GaussianSplatAsset] get_gaussian_data() called on unloaded asset (splat_count == 0); returning null.");

    if (gaussian_data_cache.is_valid()) {
        return gaussian_data_cache;
    }

    Ref<::GaussianData> data;
    if (!populate_gaussian_data(data)) {
        return Ref<::GaussianData>();
    }

    gaussian_data_cache = data;
    return gaussian_data_cache;
}

bool GaussianSplatAsset::populate_gaussian_data(Ref<::GaussianData> &r_data) const {
    if (splat_count == 0) {
        return false;
    }

    if (r_data.is_null()) {
        r_data.instantiate();
    }

    r_data->resize(splat_count);
    r_data->set_positions(get_position_vectors());
    r_data->set_scales(get_scale_vectors());
    r_data->set_rotations(get_rotation_quaternions());
    r_data->set_spherical_harmonics(get_spherical_harmonics_buffer());
    r_data->set_opacities(get_opacities());
    r_data->set_palette_ids(get_palette_ids_buffer());
    r_data->set_brush_override_ids(get_brush_override_ids_buffer());
    r_data->set_normals(get_normal_vectors());
    r_data->set_brush_axes(get_brush_axes_vector2());
    r_data->set_stroke_ages(get_stroke_ages_buffer());

    Dictionary asset_metadata = get_import_metadata();
    if (asset_metadata.has(StringName("gaussian_2d_mode"))) {
        r_data->set_2d_mode((bool)asset_metadata[StringName("gaussian_2d_mode")]);
    }

    return true;
}

Error GaussianSplatAsset::populate_from_gaussian_data(const Ref<::GaussianData> &p_gaussian_data) {
    if (p_gaussian_data.is_null()) {
        GS_LOG_ERROR_DEFAULT("populate_from_gaussian_data called with invalid GaussianData reference");
        return ERR_INVALID_PARAMETER;
    }

    _invalidate_gaussian_data_cache();

    int count = p_gaussian_data->get_count();
    if (count <= 0) {
        GS_LOG_ERROR_DEFAULT("GaussianData contains no splats");
        return ERR_FILE_CORRUPT;
    }

    splat_count = count;
    sh_first_order_terms = MIN<uint32_t>(p_gaussian_data->get_sh_first_order_count(), 3u);
    sh_high_order_terms = p_gaussian_data->get_sh_high_order_count();
    _ensure_buffer_sizes();

    const Vector3 *high_order_ptr = p_gaussian_data->get_sh_high_order_coefficients_ptr();

    float *positions_ptr = positions.is_empty() ? nullptr : positions.ptrw();
    Color *colors_ptr = colors.is_empty() ? nullptr : colors.ptrw();
    float *scales_ptr = scales.is_empty() ? nullptr : scales.ptrw();
    float *rotations_ptr = rotations.is_empty() ? nullptr : rotations.ptrw();
    float *sh_dc_ptr = sh_dc_coefficients.is_empty() ? nullptr : sh_dc_coefficients.ptrw();
    float *sh_first_order_ptr = sh_first_order_coefficients.is_empty() ? nullptr : sh_first_order_coefficients.ptrw();
    float *sh_high_order_ptr = sh_high_order_coefficients.is_empty() ? nullptr : sh_high_order_coefficients.ptrw();
    float *opacity_logits_ptr = opacity_logits.is_empty() ? nullptr : opacity_logits.ptrw();
    int32_t *palette_ids_ptr = palette_ids.is_empty() ? nullptr : palette_ids.ptrw();
    int32_t *painterly_flags_ptr = painterly_flags.is_empty() ? nullptr : painterly_flags.ptrw();
    float *normals_ptr = normals.is_empty() ? nullptr : normals.ptrw();
    float *brush_axes_ptr = brush_axes.is_empty() ? nullptr : brush_axes.ptrw();
    float *stroke_ages_ptr = stroke_ages.is_empty() ? nullptr : stroke_ages.ptrw();

    const bool has_positions = positions_ptr != nullptr;
    const bool has_colors = colors_ptr != nullptr;
    const bool has_scales = scales_ptr != nullptr;
    const bool has_rotations = rotations_ptr != nullptr;
    const bool has_sh_dc = sh_dc_ptr != nullptr;
    const bool has_first_order = sh_first_order_terms > 0 && sh_first_order_ptr != nullptr;
    const bool has_high_order = sh_high_order_terms > 0 && high_order_ptr != nullptr && sh_high_order_ptr != nullptr;
    const bool has_opacity_logits = opacity_logits_ptr != nullptr;
    const bool has_normals = normals_ptr != nullptr;
    const bool has_brush_axes = brush_axes_ptr != nullptr;
    const bool has_stroke_ages = stroke_ages_ptr != nullptr;
    const bool has_palette_ids = palette_ids_ptr != nullptr;
    const bool has_painterly_flags = painterly_flags_ptr != nullptr;

    bool bounds_initialized = false;
    Vector3 min_pos;
    Vector3 max_pos;

    for (int i = 0; i < count; i++) {
        Gaussian g = p_gaussian_data->get_gaussian(i);
        const uint32_t base3 = uint32_t(i) * 3u;
        const uint32_t base4 = uint32_t(i) * 4u;
        const int first_base = i * int(sh_first_order_terms) * 3;
        const int high_base = i * int(sh_high_order_terms) * 3;
        const size_t high_order_base = size_t(i) * size_t(sh_high_order_terms);
        const uint32_t brush_base = uint32_t(i) * 2u;

        // Rotation-aware AABB extent for anisotropic Gaussian scales:
        // extent = abs(R) * sigma, then expand to 3-sigma coverage.
        const Vector3 sigma(Math::abs(g.scale.x), Math::abs(g.scale.y), Math::abs(g.scale.z));
        const Basis rotation_basis(g.rotation);
        const Vector3 axis_x = rotation_basis.get_column(0) * sigma.x;
        const Vector3 axis_y = rotation_basis.get_column(1) * sigma.y;
        const Vector3 axis_z = rotation_basis.get_column(2) * sigma.z;
        Vector3 extent(
                Math::abs(axis_x.x) + Math::abs(axis_y.x) + Math::abs(axis_z.x),
                Math::abs(axis_x.y) + Math::abs(axis_y.y) + Math::abs(axis_z.y),
                Math::abs(axis_x.z) + Math::abs(axis_y.z) + Math::abs(axis_z.z));
        extent *= 3.0f;
        Vector3 local_min = g.position - extent;
        Vector3 local_max = g.position + extent;
        if (!bounds_initialized) {
            min_pos = local_min;
            max_pos = local_max;
            bounds_initialized = true;
        } else {
            min_pos = min_pos.min(local_min);
            max_pos = max_pos.max(local_max);
        }

        if (has_positions) {
            positions_ptr[base3 + 0] = g.position.x;
            positions_ptr[base3 + 1] = g.position.y;
            positions_ptr[base3 + 2] = g.position.z;
        }

        if (has_colors) {
            Color color = g.sh_dc;
            color.a = g.opacity;
            colors_ptr[i] = color;
        }

        if (has_scales) {
            scales_ptr[base3 + 0] = g.scale.x;
            scales_ptr[base3 + 1] = g.scale.y;
            scales_ptr[base3 + 2] = g.scale.z;
        }

        if (has_rotations) {
            rotations_ptr[base4 + 0] = g.rotation.w;
            rotations_ptr[base4 + 1] = g.rotation.x;
            rotations_ptr[base4 + 2] = g.rotation.y;
            rotations_ptr[base4 + 3] = g.rotation.z;
        }

        // SH coefficients
        if (has_sh_dc) {
            sh_dc_ptr[base3 + 0] = g.sh_dc.r;
            sh_dc_ptr[base3 + 1] = g.sh_dc.g;
            sh_dc_ptr[base3 + 2] = g.sh_dc.b;
        }

        if (has_first_order) {
            for (uint32_t term = 0; term < sh_first_order_terms; term++) {
                const Vector3 &coeff = g.sh_1[term];
                const int term_base = first_base + int(term) * 3;
                sh_first_order_ptr[term_base + 0] = coeff.x;
                sh_first_order_ptr[term_base + 1] = coeff.y;
                sh_first_order_ptr[term_base + 2] = coeff.z;
            }
        }

        if (has_high_order) {
            for (uint32_t term = 0; term < sh_high_order_terms; term++) {
                const Vector3 &coeff = high_order_ptr[high_order_base + term];
                const int term_base = high_base + int(term) * 3;
                sh_high_order_ptr[term_base + 0] = coeff.x;
                sh_high_order_ptr[term_base + 1] = coeff.y;
                sh_high_order_ptr[term_base + 2] = coeff.z;
            }
        }

        float clamped_opacity = CLAMP(g.opacity, 0.0001f, 0.9999f);
        if (has_opacity_logits) {
            opacity_logits_ptr[i] = Math::log(clamped_opacity / (1.0f - clamped_opacity));
        }

        if (has_normals) {
            normals_ptr[base3 + 0] = g.normal.x;
            normals_ptr[base3 + 1] = g.normal.y;
            normals_ptr[base3 + 2] = g.normal.z;
        }

        if (has_brush_axes) {
            brush_axes_ptr[brush_base + 0] = g.brush_axes.x;
            brush_axes_ptr[brush_base + 1] = g.brush_axes.y;
        }

        if (has_stroke_ages) {
            stroke_ages_ptr[i] = g.stroke_age;
        }

        if (has_palette_ids) {
            palette_ids_ptr[i] = (int)gaussian_get_palette_id(g.painterly_meta);
        }

        if (has_painterly_flags) {
            painterly_flags_ptr[i] = (int)gaussian_get_brush_override_id(g.painterly_meta);
        }
    }

    import_metadata[StringName("splat_count")] = count;
    import_metadata[StringName("sh_first_order_terms")] = (int)sh_first_order_terms;
    import_metadata[StringName("sh_high_order_terms")] = (int)sh_high_order_terms;
    import_metadata[StringName("sh_degree")] = (int)p_gaussian_data->get_sh_degree();
    import_metadata[StringName("has_normals")] = normals.size() == splat_count * 3;
    import_metadata[StringName("has_palette_ids")] = palette_ids.size() == splat_count;
    import_metadata[StringName("has_painterly_flags")] = painterly_flags.size() == splat_count;
    import_metadata[StringName("has_brush_override_ids")] = painterly_flags.size() == splat_count;
    import_metadata[StringName("has_brush_axes")] = brush_axes.size() == splat_count * 2;
    import_metadata[StringName("has_stroke_age")] = stroke_ages.size() == splat_count;
    import_metadata[StringName("opacity_encoding")] = StringName("logit");
    import_metadata[StringName("gaussian_2d_mode")] = p_gaussian_data->get_2d_mode();
    if (bounds_initialized) {
        import_metadata[StringName("bounds")] = AABB(min_pos, max_pos - min_pos);
        import_metadata[StringName("bounds_dirty")] = false;
    } else {
        _invalidate_bounds_metadata();
    }

    emit_changed();

    return OK;
}
