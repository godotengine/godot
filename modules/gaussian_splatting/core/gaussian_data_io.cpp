/**
 * gaussian_data_io.cpp -- Companion .cpp for gaussian_data.h.
 *
 * Contains the file I/O methods of GaussianData:
 *   - load_from_file
 *   - populate_from_asset
 *   - save_to_file
 *
 * These were extracted from gaussian_data.cpp to keep the main translation
 * unit focused on core data operations while isolating serialisation /
 * deserialisation logic here.
 */

#include "gaussian_data.h"
#include "gaussian_splat_asset.h"
#include "core/io/file_access.h"
#include "core/math/math_funcs.h"
#include "../io/gaussian_data_loader.h"
#include "../logger/gs_logger.h"
#include <cmath>

// ---------------------------------------------------------------------------
// Anonymous-namespace helpers used only by the I/O methods in this TU.
// ---------------------------------------------------------------------------
namespace {

template <typename T, typename Container>
void copy_local_vector(LocalVector<T> &r_target, const Container &p_source) {
    int count = (int)p_source.size();
    r_target.resize(count);
    for (int i = 0; i < count; i++) {
        r_target[i] = p_source[i];
    }
}

} // namespace

namespace {

static constexpr float SAVE_TO_FILE_SH_C0_INV = 1.0f / 0.28209479177387814f;
static constexpr float SAVE_TO_FILE_MIN_SCALE = 1.0e-6f;
static constexpr float SAVE_TO_FILE_MIN_OPACITY = 0.001f;
static constexpr float SAVE_TO_FILE_MAX_OPACITY = 0.999f;
static constexpr float SAVE_TO_FILE_FALLBACK_OPACITY = 0.5f;

struct SaveToFileSnapshot {
    LocalVector<Gaussian> gaussians;
    bool is_2d_mode = false;
};

static float _sanitize_scale_for_serialization(float p_scale, uint32_t &r_invalid_count) {
    if (!Math::is_finite(p_scale) || p_scale <= 0.0f) {
        r_invalid_count++;
        return SAVE_TO_FILE_MIN_SCALE;
    }
    return MAX(p_scale, SAVE_TO_FILE_MIN_SCALE);
}

static float _sanitize_opacity_for_serialization(float p_opacity, uint32_t &r_invalid_count) {
    if (!Math::is_finite(p_opacity)) {
        r_invalid_count++;
        return SAVE_TO_FILE_FALLBACK_OPACITY;
    }

    const float clamped = CLAMP(p_opacity, SAVE_TO_FILE_MIN_OPACITY, SAVE_TO_FILE_MAX_OPACITY);
    if (!Math::is_equal_approx(clamped, p_opacity)) {
        r_invalid_count++;
    }
    return clamped;
}

static bool _validate_asset_buffer_size(const char *p_label, int p_actual_size, int p_expected_size, Error &r_error) {
    if (p_actual_size == p_expected_size) {
        return true;
    }

    GS_LOG_ERROR_DEFAULT(vformat(
            "[GaussianData] populate_from_asset rejected malformed %s buffer: expected=%d actual=%d",
            p_label, p_expected_size, p_actual_size));
    r_error = ERR_INVALID_DATA;
    return false;
}

} // namespace

// ---------------------------------------------------------------------------
// GaussianData -- file I/O methods
// ---------------------------------------------------------------------------

Error GaussianData::load_from_file(const String &p_path) {
    GaussianDataLoadResult load_result;
    Error err = load_gaussian_data_from_file(p_path, load_result);
    if (err != OK) {
        GS_LOG_ERROR_DEFAULT("Failed to load splat file: " + p_path);
        return err;
    }

    if (load_result.used_ply && !load_result.missing_required.is_empty()) {
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

    Ref<::GaussianData> loaded_data = load_result.data;
    if (loaded_data.is_null()) {
        GS_LOG_ERROR_DEFAULT("No data loaded from splat file");
        return ERR_INVALID_DATA;
    }

    // Copy metadata
    LocalVector<Gaussian> loaded_gaussians;
    LocalVector<Vector3> loaded_high_coeffs;
    uint32_t loaded_first_order = 0;
    uint32_t loaded_high_order = 0;
    bool loaded_is_2d_mode = false;
    {
        RWLockRead loaded_lock(loaded_data->data_rwlock);
        loaded_gaussians = loaded_data->gaussians;
        loaded_first_order = loaded_data->sh_first_order_count;
        loaded_high_order = loaded_data->sh_high_order_count;
        loaded_high_coeffs = loaded_data->sh_high_order_coefficients;
        loaded_is_2d_mode = loaded_data->is_2d_mode;
    }

    // Route through the bulk payload setter to ensure full storage invalidation.
    set_gaussian_payload(loaded_gaussians, loaded_high_coeffs, loaded_first_order, loaded_high_order, loaded_is_2d_mode);

    const int count = static_cast<int>(loaded_gaussians.size());

    GS_LOG_INFO_DEFAULT(vformat("[GaussianData] Loaded %d splats from: %s", count, p_path));
    return OK;
}

Error GaussianData::populate_from_asset(const Ref<GaussianSplatAsset> &p_asset) {
    if (p_asset.is_null()) {
        GS_LOG_ERROR_DEFAULT("populate_from_asset called with invalid GaussianSplatAsset reference");
        return ERR_INVALID_PARAMETER;
    }

    const int splat_count = p_asset->get_splat_count();
    if (splat_count <= 0) {
        GS_LOG_ERROR_DEFAULT("GaussianSplatAsset contains no splats");
        return ERR_FILE_CORRUPT;
    }

    const PackedVector3Array positions = p_asset->get_position_vectors();
    const PackedVector3Array scales = p_asset->get_scale_vectors();
    const TypedArray<Quaternion> rotations = p_asset->get_rotation_quaternions();
    const PackedFloat32Array spherical_harmonics = p_asset->get_spherical_harmonics_buffer();
    const PackedFloat32Array opacities = p_asset->get_opacities();
    const PackedInt32Array palette_ids = p_asset->get_palette_ids_buffer();
    const PackedInt32Array brush_override_ids = p_asset->get_brush_override_ids_buffer();
    const PackedVector3Array normals = p_asset->get_normal_vectors();
    const PackedVector2Array brush_axes = p_asset->get_brush_axes_vector2();
    const PackedFloat32Array stroke_ages = p_asset->get_stroke_ages_buffer();

    Error validation_error = OK;
    const bool sizes_valid =
            _validate_asset_buffer_size("positions", positions.size(), splat_count, validation_error) &&
            _validate_asset_buffer_size("scales", scales.size(), splat_count, validation_error) &&
            _validate_asset_buffer_size("rotations", rotations.size(), splat_count, validation_error) &&
            _validate_asset_buffer_size("opacities", opacities.size(), splat_count, validation_error) &&
            _validate_asset_buffer_size("palette_ids", palette_ids.size(), splat_count, validation_error) &&
            _validate_asset_buffer_size("brush_override_ids", brush_override_ids.size(), splat_count, validation_error) &&
            _validate_asset_buffer_size("normals", normals.size(), splat_count, validation_error) &&
            _validate_asset_buffer_size("brush_axes", brush_axes.size(), splat_count, validation_error) &&
            _validate_asset_buffer_size("stroke_ages", stroke_ages.size(), splat_count, validation_error);
    if (!sizes_valid) {
        return validation_error;
    }

    if (spherical_harmonics.is_empty() || spherical_harmonics.size() % splat_count != 0) {
        GS_LOG_ERROR_DEFAULT(vformat(
                "[GaussianData] populate_from_asset rejected malformed spherical harmonics buffer: splats=%d coeffs=%d",
                splat_count, spherical_harmonics.size()));
        return ERR_INVALID_DATA;
    }

    const int floats_per_splat = spherical_harmonics.size() / splat_count;
    if (floats_per_splat < 3 || (floats_per_splat % 3) != 0) {
        GS_LOG_ERROR_DEFAULT(vformat(
                "[GaussianData] populate_from_asset rejected malformed spherical harmonics layout: floats_per_splat=%d",
                floats_per_splat));
        return ERR_INVALID_DATA;
    }

    const uint32_t total_sh_terms = uint32_t(floats_per_splat / 3);
    const uint32_t staged_first_order_count = MIN<uint32_t>(total_sh_terms > 0 ? total_sh_terms - 1 : 0, 3u);
    const uint32_t staged_high_order_count = total_sh_terms > (1u + staged_first_order_count)
            ? total_sh_terms - 1u - staged_first_order_count
            : 0u;

    LocalVector<Gaussian> staged_gaussians;
    staged_gaussians.resize(splat_count);

    LocalVector<Vector3> staged_high_order_coefficients;
    if (staged_high_order_count > 0) {
        staged_high_order_coefficients.resize(size_t(splat_count) * staged_high_order_count);
    }

    const float *sh_read = spherical_harmonics.ptr();
    for (int i = 0; i < splat_count; i++) {
        Gaussian &g = staged_gaussians[i];
        g.position = positions[i];
        g.opacity = opacities[i];
        g.scale = scales[i];
        g.area = 1.0f;
        g.rotation = rotations[i];
        g.sh_dc = Color(
                sh_read[i * floats_per_splat + 0],
                sh_read[i * floats_per_splat + 1],
                sh_read[i * floats_per_splat + 2],
                1.0f);

        for (uint32_t term = 0; term < 3; term++) {
            g.sh_1[term] = Vector3();
            if (term < staged_first_order_count) {
                const int coeff_base = i * floats_per_splat + 3 + int(term) * 3;
                g.sh_1[term] = Vector3(
                        sh_read[coeff_base + 0],
                        sh_read[coeff_base + 1],
                        sh_read[coeff_base + 2]);
            }
        }

        g.normal = normals[i];
        g.stroke_age = stroke_ages[i];
        g._padding = 0.0f;
        g.brush_axes = brush_axes[i];
        g.painterly_meta = gaussian_pack_painterly_meta(
                uint16_t(CLAMP(palette_ids[i], 0, 65535)),
                uint16_t(CLAMP(brush_override_ids[i], 0, 65535)));
        g._padding2[0] = 0.0f;
        g._padding2[1] = 0.0f;
        g._padding2[2] = 0.0f;

        for (uint32_t high_term = 0; high_term < staged_high_order_count; high_term++) {
            const int coeff_base = i * floats_per_splat + 3 + int(staged_first_order_count + high_term) * 3;
            staged_high_order_coefficients[size_t(i) * staged_high_order_count + high_term] = Vector3(
                    sh_read[coeff_base + 0],
                    sh_read[coeff_base + 1],
                    sh_read[coeff_base + 2]);
        }
    }

    bool staged_is_2d_mode = false;
    Dictionary import_metadata = p_asset->get_import_metadata();
    if (import_metadata.has(StringName("gaussian_2d_mode"))) {
        staged_is_2d_mode = (bool)import_metadata[StringName("gaussian_2d_mode")];
    }

    set_gaussian_payload(
            staged_gaussians,
            staged_high_order_coefficients,
            staged_first_order_count,
            staged_high_order_count,
            staged_is_2d_mode);
    return OK;
}

Error GaussianData::save_to_file(const String &p_path) const {
    SaveToFileSnapshot snapshot;
    {
        MutexLock lock(sh_mutex);
        copy_local_vector(snapshot.gaussians, gaussians);
        snapshot.is_2d_mode = is_2d_mode;
    }

    Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::WRITE);
    if (file.is_null()) {
        return ERR_FILE_CANT_WRITE;
    }

    const int count = snapshot.gaussians.size();

    // Write PLY header
    file->store_string("ply\n");
    file->store_string("format binary_little_endian 1.0\n");
    file->store_string(vformat("element vertex %d\n", count));

    // Position properties
    file->store_string("property float x\n");
    file->store_string("property float y\n");
    file->store_string("property float z\n");

    // Normal properties for 2D mode
    if (snapshot.is_2d_mode) {
        file->store_string("property float nx\n");
        file->store_string("property float ny\n");
        file->store_string("property float nz\n");
    }

    // Color properties (SH DC coefficients)
    file->store_string("property float f_dc_0\n");
    file->store_string("property float f_dc_1\n");
    file->store_string("property float f_dc_2\n");

    // Scale properties (log scale)
    file->store_string("property float scale_0\n");
    file->store_string("property float scale_1\n");
    file->store_string("property float scale_2\n");

    // Rotation properties (quaternion)
    file->store_string("property float rot_0\n");
    file->store_string("property float rot_1\n");
    file->store_string("property float rot_2\n");
    file->store_string("property float rot_3\n");

    // Opacity (logit)
    file->store_string("property float opacity\n");

    // Painterly fields
    file->store_string("property ushort palette_id\n");
    file->store_string("property ushort brush_override_id\n");
    file->store_string("property float brush_axis_u\n");
    file->store_string("property float brush_axis_v\n");
    file->store_string("property float stroke_age\n");

    file->store_string("end_header\n");

    uint32_t invalid_scale_components = 0;
    uint32_t invalid_opacity_values = 0;

    // Write binary data
    for (int i = 0; i < count; i++) {
        const Gaussian &g = snapshot.gaussians[i];

        // Position
        file->store_float(g.position.x);
        file->store_float(g.position.y);
        file->store_float(g.position.z);

        // Normal (2D mode)
        if (snapshot.is_2d_mode) {
            file->store_float(g.normal.x);
            file->store_float(g.normal.y);
            file->store_float(g.normal.z);
        }

        // Color (convert from RGB to SH DC coefficients)
        file->store_float(g.sh_dc.r * SAVE_TO_FILE_SH_C0_INV);
        file->store_float(g.sh_dc.g * SAVE_TO_FILE_SH_C0_INV);
        file->store_float(g.sh_dc.b * SAVE_TO_FILE_SH_C0_INV);

        // Scale (convert to log scale)
        const float scale_x = _sanitize_scale_for_serialization(g.scale.x, invalid_scale_components);
        const float scale_y = _sanitize_scale_for_serialization(g.scale.y, invalid_scale_components);
        const float scale_z = _sanitize_scale_for_serialization(g.scale.z, invalid_scale_components);
        file->store_float(log(scale_x));
        file->store_float(log(scale_y));
        file->store_float(log(scale_z));

        // Rotation (quaternion)
        file->store_float(g.rotation.w);
        file->store_float(g.rotation.x);
        file->store_float(g.rotation.y);
        file->store_float(g.rotation.z);

        // Opacity (convert to logit)
        const float clamped_opacity = _sanitize_opacity_for_serialization(g.opacity, invalid_opacity_values);
        file->store_float(log(clamped_opacity / (1.0f - clamped_opacity)));

        // Painterly properties
        uint16_t palette = gaussian_get_palette_id(g.painterly_meta);
        uint8_t palette_bytes[2] = { (uint8_t)(palette & 0xFF), (uint8_t)((palette >> 8) & 0xFF) };
        file->store_buffer(palette_bytes, 2);
        uint16_t brush_override_id = gaussian_get_brush_override_id(g.painterly_meta);
        uint8_t brush_override_bytes[2] = { (uint8_t)(brush_override_id & 0xFF), (uint8_t)((brush_override_id >> 8) & 0xFF) };
        file->store_buffer(brush_override_bytes, 2);
        file->store_float(g.brush_axes.x);
        file->store_float(g.brush_axes.y);
        file->store_float(g.stroke_age);
    }

    if (invalid_scale_components > 0 || invalid_opacity_values > 0) {
        GS_LOG_WARN_DEFAULT(vformat(
                "[GaussianData] save_to_file sanitized invalid source data for %s (scale_components=%d, opacities=%d)",
                p_path, invalid_scale_components, invalid_opacity_values));
    }

    GS_LOG_INFO_DEFAULT(vformat("[GaussianData] Saved %d splats to: %s", count, p_path));
    return OK;
}
