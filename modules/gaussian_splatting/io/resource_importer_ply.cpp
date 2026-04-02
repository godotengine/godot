#include "resource_importer_ply.h"

#ifdef TOOLS_ENABLED

#include "ply_loader.h"
#include "gaussian_import_preset.h"
#include "../editor/gaussian_import_settings_dialog.h"
#include "core/io/file_access.h"
#include "core/io/resource_saver.h"
#include "core/math/aabb.h"
#include "core/math/math_funcs.h"
#include "core/os/time.h"
#include "core/string/print_string.h"
#include "core/templates/vector.h"
#include "core/variant/dictionary.h"
#include "scene/resources/texture.h"

#include "../core/gaussian_data.h"
#include "../editor/gaussian_thumbnail_generator.h"
#include "../logger/gs_logger.h"

#include <algorithm>
#include <cfloat>

namespace {

// Use inline StringName literals (SNAME macro) to avoid static initialization order issues
#define OPTION_ASSET_TYPE SNAME("general/asset_type")
#define OPTION_ASSET_TYPE_LEGACY SNAME("asset_type")
#define OPTION_PRESET SNAME("quality/preset")
#define OPTION_MAX_SPLATS SNAME("quality/max_splats")
#define OPTION_DENSITY SNAME("quality/density_multiplier")
#define OPTION_ENABLE_LOD SNAME("quality/enable_lod")
#define OPTION_OPTIMIZE_GPU SNAME("quality/optimize_for_gpu")
#define OPTION_VALIDATE SNAME("validation/validate_required_properties")
#define OPTION_VALIDATE_LEGACY SNAME("validate_required_properties")
#define OPTION_WARN_MISSING SNAME("validation/warn_missing_optional")
#define OPTION_WARN_MISSING_LEGACY SNAME("warn_missing_optional")
#define OPTION_NORMALIZE_OPACITY SNAME("processing/normalize_opacity")
#define OPTION_SORT_OPACITY SNAME("processing/sort_by_opacity")
#define OPTION_QUANTIZE_POSITIONS SNAME("compression/quantize_positions")
#define OPTION_QUANTIZE_COLORS SNAME("compression/quantize_colors")
#define OPTION_QUANTIZE_SCALES SNAME("compression/quantize_scales")
#define OPTION_QUANTIZE_ROTATIONS SNAME("compression/quantize_rotations")
#define OPTION_PACK_OPACITY SNAME("compression/pack_opacity")
#define OPTION_GENERATE_THUMBNAIL SNAME("preview/generate_thumbnail")
#define OPTION_THUMBNAIL_STYLE SNAME("preview/thumbnail_style")
#define OPTION_THUMBNAIL_SIZE SNAME("preview/thumbnail_size")
#define OPTION_INCLUDE_STATS SNAME("metadata/include_statistics")
#define OPTION_INCLUDE_MEMORY SNAME("metadata/include_memory_estimate")
#define OPTION_CUSTOMIZED SNAME("quality/customized")

static bool _get_bool_option(const HashMap<StringName, Variant> &p_options, const StringName &p_name,
        const StringName &p_fallback, bool p_default) {
    if (const Variant *value = p_options.getptr(p_name)) {
        return bool(*value);
    }
    if (!p_fallback.is_empty()) {
        if (const Variant *legacy = p_options.getptr(p_fallback)) {
            return bool(*legacy);
        }
    }
    return p_default;
}

static int _get_int_option(const HashMap<StringName, Variant> &p_options, const StringName &p_name, const StringName &p_fallback,
        int p_default) {
    if (const Variant *value = p_options.getptr(p_name)) {
        return int64_t(*value);
    }
    if (!p_fallback.is_empty()) {
        if (const Variant *legacy = p_options.getptr(p_fallback)) {
            return int64_t(*legacy);
        }
    }
    return p_default;
}

static double _get_double_option(const HashMap<StringName, Variant> &p_options, const StringName &p_name,
        double p_default) {
    if (const Variant *value = p_options.getptr(p_name)) {
        return double(*value);
    }
    return p_default;
}

static String _get_string_option(const HashMap<StringName, Variant> &p_options, const StringName &p_name, const String &p_default) {
    if (const Variant *value = p_options.getptr(p_name)) {
        return String(*value);
    }
    return p_default;
}

static int _compute_final_splat_count(int p_original_count, int p_max_splats, double p_density) {
    int final_count = p_original_count;
    if (p_max_splats > 0) {
        final_count = MIN(final_count, p_max_splats);
    }
    final_count = MIN(final_count, int(Math::round(p_original_count * p_density)));
    final_count = MAX(final_count, 1);
    return final_count;
}

static Gaussian _merge_gaussian_range(const Ref<::GaussianData> &p_data, const int *p_indices,
        int p_start, int p_end, bool p_normalize_opacity) {
    // Density multiplier is a subsampling factor; pick a representative splat
    // instead of averaging to avoid "hole" artifacts from blended positions.
    const int count = MAX(1, p_end - p_start);
    const int sample_index = CLAMP(p_start + (count / 2), p_start, p_end - 1);
    Gaussian g = p_data->get_gaussian(p_indices[sample_index]);
    if (p_normalize_opacity) {
        g.opacity = CLAMP(g.opacity, 0.0f, 1.0f);
    }
    return g;
}

static uint32_t _build_compression_flags(bool p_positions, bool p_colors, bool p_scales, bool p_rotations) {
    uint32_t flags = GaussianSplatAsset::COMPRESSION_NONE;
    if (p_positions) {
        flags |= GaussianSplatAsset::COMPRESSION_POSITIONS;
    }
    if (p_colors) {
        flags |= GaussianSplatAsset::COMPRESSION_COLORS;
    }
    if (p_scales) {
        flags |= GaussianSplatAsset::COMPRESSION_SCALES;
    }
    if (p_rotations) {
        flags |= GaussianSplatAsset::COMPRESSION_ROTATIONS;
    }
    return flags;
}

} // namespace

ResourceImporterPLY::ResourceImporterPLY() {}

String ResourceImporterPLY::get_importer_name() const {
    return "gaussian_splat_ply";
}

String ResourceImporterPLY::get_visible_name() const {
    return "Gaussian Splat PLY";
}

void ResourceImporterPLY::get_recognized_extensions(List<String> *p_extensions) const {
    p_extensions->push_back("ply");
}

String ResourceImporterPLY::get_save_extension() const {
    return "tres";
}

String ResourceImporterPLY::get_resource_type() const {
    return "GaussianSplatAsset";
}

int ResourceImporterPLY::get_preset_count() const {
    return gaussian_get_import_presets().size();
}

String ResourceImporterPLY::get_preset_name(int p_idx) const {
    const Vector<GaussianImportPresetDefinition> &presets = gaussian_get_import_presets();
    if (p_idx < 0 || p_idx >= presets.size()) {
        return String();
    }
    return presets[p_idx].display_name;
}

void ResourceImporterPLY::get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const {
    (void)p_path;
    const Vector<GaussianImportPresetDefinition> &presets = gaussian_get_import_presets();
    int preset_index = CLAMP(p_preset, 0, presets.size() - 1);
    const GaussianImportPresetDefinition &preset = gaussian_get_import_preset_by_index(preset_index);

    r_options->push_back(ImportOption(
            PropertyInfo(Variant::STRING, String(OPTION_PRESET), PROPERTY_HINT_ENUM,
                    "mobile,desktop,high,ultra,development,custom"),
            preset.id));

    r_options->push_back(ImportOption(
            PropertyInfo(Variant::INT, String(OPTION_ASSET_TYPE), PROPERTY_HINT_ENUM, "Static,Dynamic"),
            preset.default_asset_type));

    r_options->push_back(ImportOption(
            PropertyInfo(Variant::INT, String(OPTION_MAX_SPLATS), PROPERTY_HINT_RANGE, "0,5000000,1000"),
            preset.max_splats));

    r_options->push_back(ImportOption(
            PropertyInfo(Variant::FLOAT, String(OPTION_DENSITY), PROPERTY_HINT_RANGE, "0.1,1.0,0.05"),
            preset.density_multiplier));

    r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, String(OPTION_ENABLE_LOD)), preset.enable_lod));

    r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, String(OPTION_OPTIMIZE_GPU)), preset.optimize_for_gpu));

    r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, String(OPTION_VALIDATE)), true));

    r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, String(OPTION_WARN_MISSING)), true));

    r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, String(OPTION_NORMALIZE_OPACITY)), true));

    r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, String(OPTION_SORT_OPACITY)), false));

    r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, String(OPTION_QUANTIZE_POSITIONS)), preset.quantize_positions));

    r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, String(OPTION_QUANTIZE_COLORS)), preset.quantize_colors));

    r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, String(OPTION_QUANTIZE_SCALES)), preset.quantize_scales));

    r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, String(OPTION_QUANTIZE_ROTATIONS)), preset.quantize_rotations));

    r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, String(OPTION_GENERATE_THUMBNAIL)), true));

    r_options->push_back(ImportOption(
            PropertyInfo(Variant::INT, String(OPTION_THUMBNAIL_STYLE), PROPERTY_HINT_ENUM, "Color,Density,Normals,Heatmap"),
            preset.thumbnail_style));

    r_options->push_back(ImportOption(
            PropertyInfo(Variant::INT, String(OPTION_THUMBNAIL_SIZE), PROPERTY_HINT_RANGE, "32,512,16"),
            preset.default_thumbnail_size));

    r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, String(OPTION_INCLUDE_STATS)), preset.include_statistics));

    r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, String(OPTION_INCLUDE_MEMORY)), preset.include_memory_estimate));
}

bool ResourceImporterPLY::get_option_visibility(const String &p_path, const String &p_option,
        const HashMap<StringName, Variant> &p_options) const {
    (void)p_path;
    (void)p_option;
    (void)p_options;
    return true;
}

Error ResourceImporterPLY::import(ResourceUID::ID p_source_id, const String &p_source_file, const String &p_save_path,
        const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants,
        List<String> *r_gen_files, Variant *r_metadata) {
    (void)p_source_id;
    (void)r_platform_variants;

    if (!FileAccess::exists(p_source_file)) {
        GS_LOG_ERROR_DEFAULT("PLY file not found: " + p_source_file);
        return ERR_FILE_NOT_FOUND;
    }

    Ref<PLYLoader> loader;
    loader.instantiate();
    Error err = loader->load_file(p_source_file);
    if (err != OK) {
        GS_LOG_ERROR_DEFAULT(vformat("Failed to load PLY file: %s (error %d)", p_source_file, err));
        return err;
    }

    bool validate_required = _get_bool_option(p_options, OPTION_VALIDATE, OPTION_VALIDATE_LEGACY, true);
    bool warn_missing = _get_bool_option(p_options, OPTION_WARN_MISSING, OPTION_WARN_MISSING_LEGACY, true);

    if (validate_required) {
        err = validate_ply_properties(loader);
        if (err != OK) {
            return err;
        }
    }

    if (warn_missing) {
        log_missing_properties(loader);
    }

    Ref<::GaussianData> gaussian_data = loader->get_gaussian_data();
    if (!gaussian_data.is_valid() || gaussian_data->get_count() == 0) {
        GS_LOG_ERROR_DEFAULT("PLY file contains no Gaussian data: " + p_source_file);
        return ERR_FILE_CORRUPT;
    }

    const int original_count = gaussian_data->get_count();

    String preset_name = _get_string_option(p_options, OPTION_PRESET,
            gaussian_get_import_preset_by_index(gaussian_find_import_preset_index("desktop")).id);
    preset_name = preset_name.to_lower();
    const GaussianImportPresetDefinition &preset = gaussian_get_import_preset_by_name(preset_name);

    int asset_type_value = _get_int_option(p_options, OPTION_ASSET_TYPE, OPTION_ASSET_TYPE_LEGACY, preset.default_asset_type);
    GaussianSplatAsset::AssetType asset_type = asset_type_value == 1 ? GaussianSplatAsset::ASSET_TYPE_DYNAMIC : GaussianSplatAsset::ASSET_TYPE_STATIC;

    int max_splats = _get_int_option(p_options, OPTION_MAX_SPLATS, StringName(), preset.max_splats);
    double density_multiplier = _get_double_option(p_options, OPTION_DENSITY, preset.density_multiplier);
    density_multiplier = CLAMP(density_multiplier, 0.1, 1.0);
    bool enable_lod = _get_bool_option(p_options, OPTION_ENABLE_LOD, StringName(), preset.enable_lod);
    bool optimize_for_gpu = _get_bool_option(p_options, OPTION_OPTIMIZE_GPU, StringName(), preset.optimize_for_gpu);
    bool normalize_opacity = _get_bool_option(p_options, OPTION_NORMALIZE_OPACITY, StringName(), true);
    bool sort_by_opacity = _get_bool_option(p_options, OPTION_SORT_OPACITY, StringName(), false);
    bool quantize_positions = _get_bool_option(p_options, OPTION_QUANTIZE_POSITIONS, StringName(), preset.quantize_positions);
    bool quantize_colors = _get_bool_option(p_options, OPTION_QUANTIZE_COLORS, StringName(), preset.quantize_colors);
    bool quantize_scales = _get_bool_option(p_options, OPTION_QUANTIZE_SCALES, StringName(), preset.quantize_scales);
    bool quantize_rotations = _get_bool_option(p_options, OPTION_QUANTIZE_ROTATIONS, StringName(), preset.quantize_rotations);
    const Variant *legacy_pack_opacity_value = p_options.getptr(OPTION_PACK_OPACITY);
    const bool legacy_pack_opacity_requested = legacy_pack_opacity_value && bool(*legacy_pack_opacity_value);
    if (legacy_pack_opacity_requested) {
        WARN_PRINT_ONCE("[ResourceImporterPLY] compression/pack_opacity is deprecated and ignored.");
    }
    bool generate_thumbnail = _get_bool_option(p_options, OPTION_GENERATE_THUMBNAIL, StringName(), true);
    int thumbnail_style = _get_int_option(p_options, OPTION_THUMBNAIL_STYLE, StringName(), preset.thumbnail_style);
    int thumbnail_size = _get_int_option(p_options, OPTION_THUMBNAIL_SIZE, StringName(), preset.default_thumbnail_size);
    bool include_stats = _get_bool_option(p_options, OPTION_INCLUDE_STATS, StringName(), preset.include_statistics);
    bool include_memory = _get_bool_option(p_options, OPTION_INCLUDE_MEMORY, StringName(), preset.include_memory_estimate);
    bool customized = _get_bool_option(p_options, OPTION_CUSTOMIZED, StringName(), false);

    thumbnail_style = CLAMP(thumbnail_style, 0, 3);
    thumbnail_size = CLAMP(thumbnail_size, 32, 512);

    int final_count = _compute_final_splat_count(original_count, max_splats, density_multiplier);
    const bool merge_density = density_multiplier < 0.999 && final_count < original_count;
    const double merge_stride = merge_density ? double(original_count) / double(final_count) : 1.0;

    Vector<int> indices;
    indices.resize(original_count);
    int *indices_ptr = indices.ptrw();
    for (int i = 0; i < original_count; i++) {
        indices_ptr[i] = i;
    }

    if (sort_by_opacity) {
        std::sort(indices_ptr, indices_ptr + original_count, [&](int a, int b) {
            return gaussian_data->get_gaussian(a).opacity > gaussian_data->get_gaussian(b).opacity;
        });
    }

    PackedFloat32Array positions;
    PackedColorArray colors;
    PackedFloat32Array scales;
    PackedFloat32Array rotations;

    positions.resize(final_count * 3);
    colors.resize(final_count);
    scales.resize(final_count * 3);
    rotations.resize(final_count * 4);

    float *positions_ptr = positions.ptrw();
    Color *colors_ptr = colors.ptrw();
    float *scales_ptr = scales.ptrw();
    float *rotations_ptr = rotations.ptrw();

    for (int i = 0; i < final_count; i++) {
        int start = merge_density ? int(Math::floor(double(i) * merge_stride)) : i;
        int end = merge_density ? int(Math::floor(double(i + 1) * merge_stride)) : i + 1;
        start = CLAMP(start, 0, original_count - 1);
        end = CLAMP(end, start + 1, original_count);
        Gaussian g = merge_density ? _merge_gaussian_range(gaussian_data, indices_ptr, start, end, normalize_opacity)
                                   : gaussian_data->get_gaussian(indices_ptr[start]);
        int pos_base = i * 3;
        positions_ptr[pos_base + 0] = g.position.x;
        positions_ptr[pos_base + 1] = g.position.y;
        positions_ptr[pos_base + 2] = g.position.z;

        Color color = g.sh_dc;
        color.a = normalize_opacity ? CLAMP(g.opacity, 0.0f, 1.0f) : g.opacity;
        colors_ptr[i] = color;

        int scale_base = i * 3;
        scales_ptr[scale_base + 0] = g.scale.x;
        scales_ptr[scale_base + 1] = g.scale.y;
        scales_ptr[scale_base + 2] = g.scale.z;

        int rot_base = i * 4;
        rotations_ptr[rot_base + 0] = g.rotation.w;
        rotations_ptr[rot_base + 1] = g.rotation.x;
        rotations_ptr[rot_base + 2] = g.rotation.y;
        rotations_ptr[rot_base + 3] = g.rotation.z;
    }

    Ref<GaussianSplatAsset> asset;
    asset.instantiate();
    asset->set_asset_type(asset_type);
    asset->set_positions(positions);
    asset->set_colors(colors);
    asset->set_scales(scales);
    asset->set_rotations(rotations);
    asset->set_import_quality_preset(preset_name);

    uint32_t compression_flags = _build_compression_flags(quantize_positions, quantize_colors, quantize_scales, quantize_rotations);
    asset->set_compression_flags(compression_flags);
    asset->set_source_path(p_source_file);

    Ref<Texture2D> thumbnail;
    if (generate_thumbnail) {
        Ref<GaussianThumbnailGenerator> generator;
        generator.instantiate();
        thumbnail = generator->generate_thumbnail(asset, thumbnail_size,
                GaussianThumbnailGenerator::style_from_int(thumbnail_style));
        asset->set_thumbnail(thumbnail);
    } else {
        asset->set_thumbnail(Ref<Texture2D>());
    }

    Dictionary option_dict;
    option_dict[OPTION_PRESET] = preset_name;
    option_dict[OPTION_ASSET_TYPE] = asset_type_value;
    option_dict[OPTION_MAX_SPLATS] = max_splats;
    option_dict[OPTION_DENSITY] = density_multiplier;
    option_dict[OPTION_ENABLE_LOD] = enable_lod;
    option_dict[OPTION_OPTIMIZE_GPU] = optimize_for_gpu;
    option_dict[OPTION_VALIDATE] = validate_required;
    option_dict[OPTION_WARN_MISSING] = warn_missing;
    option_dict[OPTION_NORMALIZE_OPACITY] = normalize_opacity;
    option_dict[OPTION_SORT_OPACITY] = sort_by_opacity;
    option_dict[OPTION_QUANTIZE_POSITIONS] = quantize_positions;
    option_dict[OPTION_QUANTIZE_COLORS] = quantize_colors;
    option_dict[OPTION_QUANTIZE_SCALES] = quantize_scales;
    option_dict[OPTION_QUANTIZE_ROTATIONS] = quantize_rotations;
    option_dict[OPTION_GENERATE_THUMBNAIL] = generate_thumbnail;
    option_dict[OPTION_THUMBNAIL_STYLE] = thumbnail_style;
    option_dict[OPTION_THUMBNAIL_SIZE] = thumbnail_size;
    option_dict[OPTION_INCLUDE_STATS] = include_stats;
    option_dict[OPTION_INCLUDE_MEMORY] = include_memory;
    option_dict[OPTION_CUSTOMIZED] = customized;

    Dictionary import_metadata;
    import_metadata[StringName("source_file")] = p_source_file;
    import_metadata[StringName("import_time")] = Time::get_singleton()->get_datetime_dict_from_system();
    import_metadata[StringName("original_splat_count")] = original_count;
    import_metadata[StringName("splat_count")] = final_count;
    import_metadata[StringName("asset_type")] = asset_type;
    import_metadata[StringName("quality_preset")] = preset_name;
    import_metadata[StringName("preset_display")] = preset.display_name;
    import_metadata[StringName("quality_customized")] = customized;
    import_metadata[StringName("enable_lod")] = enable_lod;
    import_metadata[StringName("optimize_for_gpu")] = optimize_for_gpu;
    import_metadata[StringName("density_multiplier")] = density_multiplier;
    import_metadata[StringName("max_splats")] = max_splats;
    import_metadata[StringName("validate_required")] = validate_required;
    import_metadata[StringName("warn_missing_optional")] = warn_missing;
    import_metadata[StringName("normalize_opacity")] = normalize_opacity;
    import_metadata[StringName("sort_by_opacity")] = sort_by_opacity;
    import_metadata[StringName("compression_flags")] = int(compression_flags);
    import_metadata[StringName("options")] = option_dict;
    import_metadata[StringName("thumbnail_generated")] = generate_thumbnail;
    import_metadata[StringName("thumbnail_style")] = thumbnail_style;
    import_metadata[StringName("thumbnail_size")] = thumbnail_size;
    import_metadata[StringName("thumbnail_style_name")] = GaussianThumbnailGenerator::style_to_display_name(
            GaussianThumbnailGenerator::style_from_int(thumbnail_style));

    if (include_stats) {
        import_metadata[StringName("loader_statistics")] = loader->get_load_statistics();
    }
    if (include_memory) {
        Ref<GaussianThumbnailGenerator> generator;
        generator.instantiate();
        Dictionary memory_stats = generator->compute_memory_statistics(final_count, compression_flags, false);
        import_metadata[StringName("memory_estimate_mb")] = memory_stats.get(StringName("total_mb"), 0.0);
        import_metadata[StringName("memory_breakdown_mb")] = memory_stats;
    }

    AABB bounds = gaussian_data->get_aabb();
    import_metadata[StringName("bounds")] = bounds;

    asset->set_import_metadata(import_metadata);

    String save_path = p_save_path + "." + get_save_extension();
    err = ResourceSaver::save(asset, save_path);
    if (err != OK) {
        GS_LOG_ERROR_DEFAULT("Failed to save GaussianSplatAsset: " + save_path);
        return err;
    }

    if (r_gen_files) {
        r_gen_files->push_back(save_path);
        // Register the .gsplatcache written by PLYLoader::write_cache()
        // so Godot's import system treats it as a generated artifact.
        String cache_path = p_source_file.get_basename() + ".gsplatcache";
        if (FileAccess::exists(cache_path)) {
            r_gen_files->push_back(cache_path);
        }
    }

    if (r_metadata) {
        *r_metadata = import_metadata;
    }

    GS_LOG_STREAMING_INFO(vformat("PLY import successful: %d/%d splats from %s", final_count, original_count, p_source_file));
    return OK;
}

Error ResourceImporterPLY::validate_ply_properties(const Ref<PLYLoader> &p_loader) const {
    if (!p_loader.is_valid()) {
        GS_LOG_ERROR_DEFAULT("PLY validation failed: Loader reference is invalid");
        return ERR_INVALID_PARAMETER;
    }

    if (!p_loader->has_property("x") || !p_loader->has_property("y") || !p_loader->has_property("z")) {
        GS_LOG_ERROR_DEFAULT("PLY validation failed: Missing required position properties (x, y, z)");
        return ERR_FILE_CORRUPT;
    }

    if (!p_loader->has_property("f_dc_0") || !p_loader->has_property("f_dc_1") || !p_loader->has_property("f_dc_2")) {
        GS_LOG_ERROR_DEFAULT("PLY validation failed: Missing required SH DC color properties (f_dc_0, f_dc_1, f_dc_2)");
        return ERR_FILE_CORRUPT;
    }

    if (!p_loader->has_property("scale_0") || !p_loader->has_property("scale_1") || !p_loader->has_property("scale_2")) {
        GS_LOG_ERROR_DEFAULT("PLY validation failed: Missing required scale properties (scale_0, scale_1, scale_2)");
        return ERR_FILE_CORRUPT;
    }

    if (!p_loader->has_property("rot_0") || !p_loader->has_property("rot_1") || !p_loader->has_property("rot_2") ||
            !p_loader->has_property("rot_3")) {
        GS_LOG_ERROR_DEFAULT("PLY validation failed: Missing required rotation properties (rot_0, rot_1, rot_2, rot_3)");
        return ERR_FILE_CORRUPT;
    }

    if (!p_loader->has_property("opacity")) {
        GS_LOG_ERROR_DEFAULT("PLY validation failed: Missing required opacity property");
        return ERR_FILE_CORRUPT;
    }

    Ref<::GaussianData> data = p_loader->get_gaussian_data();
    if (!data.is_valid() || data->get_count() == 0) {
        GS_LOG_ERROR_DEFAULT("PLY validation failed: No Gaussian data found");
        return ERR_FILE_CORRUPT;
    }

    Gaussian first_splat = data->get_gaussian(0);
    if (!Math::is_finite(first_splat.position.x) || !Math::is_finite(first_splat.position.y) ||
            !Math::is_finite(first_splat.position.z)) {
        GS_LOG_ERROR_DEFAULT("PLY validation failed: Invalid position values detected");
        return ERR_FILE_CORRUPT;
    }

    if (first_splat.scale.x <= 0 || first_splat.scale.y <= 0 || first_splat.scale.z <= 0) {
        GS_LOG_ERROR_DEFAULT("PLY validation failed: Invalid scale values detected");
        return ERR_FILE_CORRUPT;
    }

    float quat_length_sq = first_splat.rotation.length_squared();
    if (quat_length_sq < 0.99f || quat_length_sq > 1.01f) {
        GS_LOG_STREAMING_WARN("PLY validation warning: Rotation quaternion may not be normalized");
    }

    if (first_splat.opacity < 0.0f || first_splat.opacity > 1.0f || !Math::is_finite(first_splat.opacity)) {
        GS_LOG_ERROR_DEFAULT("PLY validation failed: Invalid opacity values detected");
        return ERR_FILE_CORRUPT;
    }

    return OK;
}

void ResourceImporterPLY::log_missing_properties(const Ref<PLYLoader> &p_loader) const {
    if (!p_loader.is_valid()) {
        GS_LOG_ERROR_DEFAULT("PLY validation skipped: Loader reference is invalid");
        return;
    }

    Dictionary stats = p_loader->get_load_statistics();
    int splat_count = stats.get("splat_count", 0);
    String format = stats.get("format", "unknown");
    int properties = stats.get("properties", 0);

    GS_LOG_STREAMING_INFO(vformat("PLY Import Info: %d splats, %s format, %d properties", splat_count, format, properties));

    Vector<String> optional_properties = {
        "nx", "ny", "nz",
        "palette_id",
        "brush_override_id",
        "brush_axis_u", "brush_axis_v",
        "stroke_age",
        "f_rest_0", "f_rest_1", "f_rest_2"
    };

    Vector<String> missing_optional;
    Vector<String> found_optional;

    for (const String &prop : optional_properties) {
        if (p_loader->has_property(prop)) {
            found_optional.push_back(prop);
        } else {
            missing_optional.push_back(prop);
        }
    }

    if (!found_optional.is_empty()) {
        GS_LOG_STREAMING_INFO("PLY Import Info: Found optional properties: " + String(", ").join(found_optional));
    }
    if (!missing_optional.is_empty()) {
        GS_LOG_STREAMING_INFO("PLY Import Info: Missing optional properties: " + String(", ").join(missing_optional));
    }

    Vector<String> all_properties = p_loader->get_property_names();
    GS_LOG_STREAMING_INFO("PLY Import Info: All properties: " + String(", ").join(all_properties));

    Ref<::GaussianData> data = p_loader->get_gaussian_data();
    if (data.is_valid() && data->get_count() > 0) {
        AABB bounds = data->get_aabb();
        GS_LOG_STREAMING_INFO(vformat("PLY Import Info: Bounds = %v to %v", bounds.position, bounds.position + bounds.size));
    }
}

bool ResourceImporterPLY::has_advanced_options() const {
    return true;
}

void ResourceImporterPLY::show_advanced_options(const String &p_path) {
    GaussianImportSettingsDialog *dialog = GaussianImportSettingsDialog::get_singleton();
    if (dialog) {
        dialog->open_settings(p_path);
    }
}

#endif // TOOLS_ENABLED
