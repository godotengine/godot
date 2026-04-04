#include "resource_importer_spz.h"

#ifdef TOOLS_ENABLED

#include "spz_loader.h"
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

#define OPTION_ASSET_TYPE SNAME("general/asset_type")
#define OPTION_PRESET SNAME("quality/preset")
#define OPTION_MAX_SPLATS SNAME("quality/max_splats")
#define OPTION_DENSITY SNAME("quality/density_multiplier")
#define OPTION_ENABLE_LOD SNAME("quality/enable_lod")
#define OPTION_OPTIMIZE_GPU SNAME("quality/optimize_for_gpu")
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

static bool _get_bool_option(const HashMap<StringName, Variant> &p_options, const StringName &p_name, bool p_default) {
    if (const Variant *value = p_options.getptr(p_name)) {
        return bool(*value);
    }
    return p_default;
}

static int _get_int_option(const HashMap<StringName, Variant> &p_options, const StringName &p_name, int p_default) {
    if (const Variant *value = p_options.getptr(p_name)) {
        return int64_t(*value);
    }
    return p_default;
}

static double _get_double_option(const HashMap<StringName, Variant> &p_options, const StringName &p_name, double p_default) {
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

ResourceImporterSPZ::ResourceImporterSPZ() {}

String ResourceImporterSPZ::get_importer_name() const {
    return "gaussian_splat_spz";
}

String ResourceImporterSPZ::get_visible_name() const {
    return "Gaussian Splat SPZ";
}

void ResourceImporterSPZ::get_recognized_extensions(List<String> *p_extensions) const {
    p_extensions->push_back("spz");
}

String ResourceImporterSPZ::get_save_extension() const {
    return "tres";
}

String ResourceImporterSPZ::get_resource_type() const {
    return "GaussianSplatAsset";
}

int ResourceImporterSPZ::get_preset_count() const {
    return gaussian_get_import_presets().size();
}

String ResourceImporterSPZ::get_preset_name(int p_idx) const {
    const Vector<GaussianImportPresetDefinition> &presets = gaussian_get_import_presets();
    if (p_idx < 0 || p_idx >= presets.size()) {
        return String();
    }
    return presets[p_idx].display_name;
}

void ResourceImporterSPZ::get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const {
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

bool ResourceImporterSPZ::get_option_visibility(const String &p_path, const String &p_option,
        const HashMap<StringName, Variant> &p_options) const {
    (void)p_path;
    (void)p_option;
    (void)p_options;
    return true;
}

Error ResourceImporterSPZ::import(ResourceUID::ID p_source_id, const String &p_source_file, const String &p_save_path,
        const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants,
        List<String> *r_gen_files, Variant *r_metadata) {
    (void)p_source_id;
    (void)r_platform_variants;

    if (!FileAccess::exists(p_source_file)) {
        GS_LOG_ERROR_DEFAULT("SPZ file not found: " + p_source_file);
        return ERR_FILE_NOT_FOUND;
    }

    // Load SPZ file
    Ref<SPZLoader> loader;
    loader.instantiate();
    Error err = loader->load_file(p_source_file);
    if (err != OK) {
        GS_LOG_ERROR_DEFAULT(vformat("Failed to load SPZ file: %s (error %d)", p_source_file, err));
        return err;
    }

    Ref<::GaussianData> gaussian_data = loader->get_gaussian_data();
    if (!gaussian_data.is_valid() || gaussian_data->get_count() == 0) {
        GS_LOG_ERROR_DEFAULT("SPZ file contains no Gaussian data: " + p_source_file);
        return ERR_FILE_CORRUPT;
    }

    const int original_count = gaussian_data->get_count();

    // Get import options
    String preset_name = _get_string_option(p_options, OPTION_PRESET,
            gaussian_get_import_preset_by_index(gaussian_find_import_preset_index("desktop")).id);
    preset_name = preset_name.to_lower();
    const GaussianImportPresetDefinition &preset = gaussian_get_import_preset_by_name(preset_name);

    int asset_type_value = _get_int_option(p_options, OPTION_ASSET_TYPE, preset.default_asset_type);
    GaussianSplatAsset::AssetType asset_type = asset_type_value == 1 ? GaussianSplatAsset::ASSET_TYPE_DYNAMIC : GaussianSplatAsset::ASSET_TYPE_STATIC;

    int max_splats = _get_int_option(p_options, OPTION_MAX_SPLATS, preset.max_splats);
    double density_multiplier = _get_double_option(p_options, OPTION_DENSITY, preset.density_multiplier);
    density_multiplier = CLAMP(density_multiplier, 0.1, 1.0);
    bool enable_lod = _get_bool_option(p_options, OPTION_ENABLE_LOD, preset.enable_lod);
    bool optimize_for_gpu = _get_bool_option(p_options, OPTION_OPTIMIZE_GPU, preset.optimize_for_gpu);
    bool normalize_opacity = _get_bool_option(p_options, OPTION_NORMALIZE_OPACITY, true);
    bool sort_by_opacity = _get_bool_option(p_options, OPTION_SORT_OPACITY, false);
    bool quantize_positions = _get_bool_option(p_options, OPTION_QUANTIZE_POSITIONS, preset.quantize_positions);
    bool quantize_colors = _get_bool_option(p_options, OPTION_QUANTIZE_COLORS, preset.quantize_colors);
    bool quantize_scales = _get_bool_option(p_options, OPTION_QUANTIZE_SCALES, preset.quantize_scales);
    bool quantize_rotations = _get_bool_option(p_options, OPTION_QUANTIZE_ROTATIONS, preset.quantize_rotations);
    const Variant *legacy_pack_opacity_value = p_options.getptr(OPTION_PACK_OPACITY);
    const bool legacy_pack_opacity_requested = legacy_pack_opacity_value && bool(*legacy_pack_opacity_value);
    if (legacy_pack_opacity_requested) {
        WARN_PRINT_ONCE("[ResourceImporterSPZ] compression/pack_opacity is deprecated and ignored.");
    }
    bool generate_thumbnail = _get_bool_option(p_options, OPTION_GENERATE_THUMBNAIL, true);
    int thumbnail_style = _get_int_option(p_options, OPTION_THUMBNAIL_STYLE, preset.thumbnail_style);
    int thumbnail_size = _get_int_option(p_options, OPTION_THUMBNAIL_SIZE, preset.default_thumbnail_size);
    bool include_stats = _get_bool_option(p_options, OPTION_INCLUDE_STATS, preset.include_statistics);
    bool include_memory = _get_bool_option(p_options, OPTION_INCLUDE_MEMORY, preset.include_memory_estimate);

    thumbnail_style = CLAMP(thumbnail_style, 0, 3);
    thumbnail_size = CLAMP(thumbnail_size, 32, 512);

    int final_count = _compute_final_splat_count(original_count, max_splats, density_multiplier);
    const bool merge_density = density_multiplier < 0.999 && final_count < original_count;
    const double merge_stride = merge_density ? double(original_count) / double(final_count) : 1.0;

    // Create index array for potential sorting
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

    // Extract data arrays
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
        int pos_base = i * 3;
        Gaussian g = merge_density ? _merge_gaussian_range(gaussian_data, indices_ptr, start, end, normalize_opacity)
                                   : gaussian_data->get_gaussian(indices_ptr[start]);

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

    // Create asset
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

    // Generate thumbnail
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

    // Build metadata
    Dictionary import_metadata;
    import_metadata[StringName("source_file")] = p_source_file;
    import_metadata[StringName("source_format")] = "spz";
    import_metadata[StringName("dc_encoding")] = "linear_rgb";
    import_metadata[StringName("import_time")] = Time::get_singleton()->get_datetime_dict_from_system();
    import_metadata[StringName("original_splat_count")] = original_count;
    import_metadata[StringName("splat_count")] = final_count;
    import_metadata[StringName("asset_type")] = asset_type;
    import_metadata[StringName("quality_preset")] = preset_name;
    import_metadata[StringName("preset_display")] = preset.display_name;
    import_metadata[StringName("enable_lod")] = enable_lod;
    import_metadata[StringName("optimize_for_gpu")] = optimize_for_gpu;
    import_metadata[StringName("density_multiplier")] = density_multiplier;
    import_metadata[StringName("max_splats")] = max_splats;
    import_metadata[StringName("normalize_opacity")] = normalize_opacity;
    import_metadata[StringName("sort_by_opacity")] = sort_by_opacity;
    import_metadata[StringName("compression_flags")] = int(compression_flags);
    import_metadata[StringName("thumbnail_generated")] = generate_thumbnail;
    import_metadata[StringName("thumbnail_style")] = thumbnail_style;
    import_metadata[StringName("thumbnail_size")] = thumbnail_size;
    import_metadata[StringName("thumbnail_style_name")] = GaussianThumbnailGenerator::style_to_display_name(
            GaussianThumbnailGenerator::style_from_int(thumbnail_style));

    // SPZ-specific metadata
    SPZLoader::SPZHeader spz_header = loader->get_header();
    import_metadata[StringName("spz_version")] = (int)spz_header.version;
    import_metadata[StringName("spz_sh_degree")] = (int)spz_header.sh_degree;
    import_metadata[StringName("spz_fractional_bits")] = (int)spz_header.fractional_bits;
    import_metadata[StringName("spz_antialiased")] = (spz_header.flags & SPZLoader::SPZ_FLAG_ANTIALIASED) != 0;

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

    // Save asset
    String save_path = p_save_path + "." + get_save_extension();
    err = ResourceSaver::save(asset, save_path);
    if (err != OK) {
        GS_LOG_ERROR_DEFAULT("Failed to save GaussianSplatAsset: " + save_path);
        return err;
    }

    if (r_gen_files) {
        r_gen_files->push_back(save_path);
    }

    if (r_metadata) {
        *r_metadata = import_metadata;
    }

    GS_LOG_STREAMING_INFO(vformat("SPZ import successful: %d/%d splats from %s", final_count, original_count, p_source_file));
    return OK;
}

bool ResourceImporterSPZ::has_advanced_options() const {
    return true;
}

void ResourceImporterSPZ::show_advanced_options(const String &p_path) {
    GaussianImportSettingsDialog *dialog = GaussianImportSettingsDialog::get_singleton();
    if (dialog) {
        dialog->open_settings(p_path);
    }
}

#endif // TOOLS_ENABLED
