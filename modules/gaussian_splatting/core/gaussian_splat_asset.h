#ifndef GAUSSIAN_SPLAT_ASSET_H
#define GAUSSIAN_SPLAT_ASSET_H

#include "core/io/resource.h"
#include "scene/resources/texture.h"
#include "core/io/resource_loader.h"
#include "core/math/quaternion.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include "core/variant/typed_array.h"

#include <atomic>

class GaussianData;

class GaussianSplatAsset : public Resource {
    GDCLASS(GaussianSplatAsset, Resource);
    RES_BASE_EXTENSION("gaussiansplat");

public:
    enum AssetType {
        ASSET_TYPE_STATIC,  // Immutable, optimized for GPU
        ASSET_TYPE_DYNAMIC  // Editable, supports runtime modifications
    };

    enum CompressionFlags {
        COMPRESSION_NONE = 0,
        COMPRESSION_POSITIONS = 1 << 0,
        COMPRESSION_COLORS = 1 << 1,
        COMPRESSION_SCALES = 1 << 2,
        COMPRESSION_ROTATIONS = 1 << 3
    };

private:
    AssetType asset_type = ASSET_TYPE_STATIC;
    PackedFloat32Array positions;  // x,y,z packed
    PackedColorArray colors;        // RGBA colors
    PackedFloat32Array scales;      // 3D scales
    PackedFloat32Array rotations;   // Quaternions
    PackedFloat32Array sh_dc_coefficients;          // RGB coefficients for SH DC band
    PackedFloat32Array sh_first_order_coefficients; // First-order SH coefficients (variable count)
    PackedFloat32Array sh_high_order_coefficients;  // Higher-order SH coefficients
    PackedFloat32Array opacity_logits;              // Opacity logits per splat
    PackedInt32Array palette_ids;                   // Palette identifiers per splat
    PackedInt32Array painterly_flags;               // Shared storage for painterly flags / brush override IDs per splat
    PackedFloat32Array normals;                     // Optional per-splat normals
    PackedFloat32Array brush_axes;                  // Painterly brush axes
    PackedFloat32Array stroke_ages;                 // Painterly stroke age metadata

    uint32_t splat_count = 0;
    uint32_t sh_first_order_terms = 0;
    uint32_t sh_high_order_terms = 0;
    uint32_t compression_flags = COMPRESSION_NONE;
    String import_quality_preset = "high";
    Dictionary import_metadata;
    Ref<Texture2D> thumbnail;
    bool has_sh_dc_coefficients = false;
    mutable Ref<::GaussianData> gaussian_data_cache;

    static std::atomic<uint32_t> instance_count;

    void _recalculate_sh_component_counts();
    void _ensure_buffer_sizes();
    void _invalidate_gaussian_data_cache();
    void _invalidate_bounds_metadata();

protected:
    static void _bind_methods();

public:
    GaussianSplatAsset();
    ~GaussianSplatAsset();

    // Returns true when the asset has been populated with splat data (splat_count > 0).
    bool is_loaded() const { return splat_count > 0; }

    void set_asset_type(AssetType p_type);
    AssetType get_asset_type() const { return asset_type; }

    void set_splat_count(uint32_t p_count);
    uint32_t get_splat_count() const { return splat_count; }

    // Getters
    PackedFloat32Array get_positions() const;
    PackedVector3Array get_position_vectors() const;
    PackedColorArray get_colors() const;
    PackedFloat32Array get_scales() const;
    PackedVector3Array get_scale_vectors() const;
    PackedFloat32Array get_rotations() const;
    TypedArray<Quaternion> get_rotation_quaternions() const;
    PackedFloat32Array get_sh_dc_coefficients() const;
    PackedFloat32Array get_sh_first_order_coefficients() const;
    PackedFloat32Array get_sh_high_order_coefficients() const;
    PackedFloat32Array get_spherical_harmonics_buffer() const;
    PackedFloat32Array get_opacity_logits() const;
    PackedFloat32Array get_opacities() const;
    PackedInt32Array get_palette_ids() const;
    PackedInt32Array get_palette_ids_buffer() const;
    PackedInt32Array get_painterly_flags() const;
    PackedInt32Array get_painterly_flags_buffer() const;
    PackedInt32Array get_brush_override_ids() const;
    PackedInt32Array get_brush_override_ids_buffer() const;
    PackedFloat32Array get_normals() const;
    PackedVector3Array get_normal_vectors() const;
    PackedFloat32Array get_brush_axes() const;
    PackedVector2Array get_brush_axes_vector2() const;
    PackedFloat32Array get_stroke_ages() const;
    PackedFloat32Array get_stroke_ages_buffer() const;
    uint32_t get_sh_first_order_terms() const { return sh_first_order_terms; }
    uint32_t get_sh_high_order_terms() const { return sh_high_order_terms; }

    // Setters - needed for loaders to populate data
    void set_positions(const PackedFloat32Array &p_positions);
    void set_colors(const PackedColorArray &p_colors);
    void set_scales(const PackedFloat32Array &p_scales);
    void set_rotations(const PackedFloat32Array &p_rotations);
    void set_sh_dc_coefficients(const PackedFloat32Array &p_coefficients);
    void set_sh_first_order_coefficients(const PackedFloat32Array &p_coefficients);
    void set_sh_high_order_coefficients(const PackedFloat32Array &p_coefficients);
    void set_opacity_logits(const PackedFloat32Array &p_opacity_logits);
    void set_palette_ids(const PackedInt32Array &p_palette_ids);
    void set_painterly_flags(const PackedInt32Array &p_flags);
    void set_brush_override_ids(const PackedInt32Array &p_override_ids);
    void set_normals(const PackedFloat32Array &p_normals);
    void set_brush_axes(const PackedFloat32Array &p_brush_axes);
    void set_stroke_ages(const PackedFloat32Array &p_stroke_ages);
    void set_sh_component_terms(uint32_t p_first_order_terms, uint32_t p_high_order_terms);

    static uint32_t get_instance_count() { return instance_count.load(); }

    void set_import_metadata(const Dictionary &p_metadata);
    Dictionary get_import_metadata() const { return import_metadata; }

    void set_import_quality_preset(const String &p_preset);
    String get_import_quality_preset() const { return import_quality_preset; }

    void set_compression_flags(uint32_t p_flags);
    uint32_t get_compression_flags() const { return compression_flags; }

    void set_thumbnail(const Ref<Texture2D> &p_thumbnail);
    Ref<Texture2D> get_thumbnail() const { return thumbnail; }

    void set_source_path(const String &p_path);
    String get_source_path() const;

    Error load_from_file(const String &p_path);
    Ref<::GaussianData> get_gaussian_data() const;
    bool populate_gaussian_data(Ref<::GaussianData> &r_data) const;
    Error populate_from_gaussian_data(const Ref<::GaussianData> &p_gaussian_data);
    Error save_to_file(const String &p_path) const;
};

VARIANT_ENUM_CAST(GaussianSplatAsset::AssetType);
VARIANT_ENUM_CAST(GaussianSplatAsset::CompressionFlags);

#endif // GAUSSIAN_SPLAT_ASSET_H
