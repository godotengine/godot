#ifdef TOOLS_ENABLED

#include "gaussian_editor_services.h"

#include "core/math/aabb.h"
#include "core/string/translation.h"
#include "../core/gaussian_splat_asset.h"
#include "../nodes/gaussian_splat_node_3d.h"
#include "../renderer/gaussian_splat_renderer.h"
#include "gaussian_thumbnail_generator.h"

namespace {

static String _bool_to_text(bool p_value) {
    return p_value ? TTR("Yes") : TTR("No");
}

static String _asset_type_to_string(int p_type) {
    switch (p_type) {
        case GaussianSplatAsset::ASSET_TYPE_DYNAMIC:
            return TTR("Dynamic");
        case GaussianSplatAsset::ASSET_TYPE_STATIC:
        default:
            return TTR("Static");
    }
}

static String _format_compression_flags(uint32_t p_flags) {
    PackedStringArray parts;
    if (p_flags & GaussianSplatAsset::COMPRESSION_POSITIONS) {
        parts.push_back(TTR("Positions"));
    }
    if (p_flags & GaussianSplatAsset::COMPRESSION_COLORS) {
        parts.push_back(TTR("Colors"));
    }
    if (p_flags & GaussianSplatAsset::COMPRESSION_SCALES) {
        parts.push_back(TTR("Scales"));
    }
    if (p_flags & GaussianSplatAsset::COMPRESSION_ROTATIONS) {
        parts.push_back(TTR("Rotations"));
    }
    if (parts.is_empty()) {
        return TTR("None");
    }
    return String(", ").join(parts);
}

} // namespace

namespace GaussianEditorServices {

int64_t dict_get_int(const Dictionary &p_dict, const StringName &p_key, int64_t p_default) {
    if (!p_dict.has(p_key)) {
        return p_default;
    }
    return int64_t(p_dict[p_key]);
}

double dict_get_double(const Dictionary &p_dict, const StringName &p_key, double p_default) {
    if (!p_dict.has(p_key)) {
        return p_default;
    }
    return double(p_dict[p_key]);
}

bool dict_get_bool(const Dictionary &p_dict, const StringName &p_key, bool p_default) {
    if (!p_dict.has(p_key)) {
        return p_default;
    }
    return bool(p_dict[p_key]);
}

String dict_get_string(const Dictionary &p_dict, const StringName &p_key, const String &p_default) {
    if (!p_dict.has(p_key)) {
        return p_default;
    }
    return String(p_dict[p_key]);
}

String format_gaussian_splat_stats(GaussianSplatNode3D *p_node, const Ref<GaussianSplatRenderer> &p_renderer) {
    if (!p_node) {
        return String("No Gaussian splat node selected.");
    }

    Dictionary node_stats = p_node->get_statistics();
    const int64_t visible = dict_get_int(node_stats, "visible_splats", 0);
    const int64_t total = dict_get_int(node_stats, "total_splats", 0);
    const double update_ms = dict_get_double(node_stats, "update_time_ms", 0.0);
    const double gpu_mb = dict_get_double(node_stats, "gpu_memory_mb", 0.0);

    String text = vformat("Visible Splats: %d / %d", int(visible), int(total));
    text += "\nGPU Memory: " + String::num(gpu_mb, 2) + " MB";
    text += "\nUpdate Time: " + String::num(update_ms, 2) + " ms";

    if (p_renderer.is_valid()) {
        Dictionary render_stats = p_renderer->get_render_stats();
        const double sort_ms = dict_get_double(render_stats, "sort_time_ms", 0.0);
        const double render_ms = dict_get_double(render_stats, "render_time_ms", 0.0);
        const int64_t frame_count = dict_get_int(render_stats, "frame_count", 0);

        text += "\nSort Time: " + String::num(sort_ms, 2) + " ms";
        text += "\nRender Time: " + String::num(render_ms, 2) + " ms";
        if (frame_count > 0) {
            text += "\nFrames Rendered: " + itos(frame_count);
        }
        if (render_stats.has(StringName("debug_preview_mode"))) {
            int mode_value = int(render_stats[StringName("debug_preview_mode")]);
            static const char *mode_names[] = { "Off", "Wireframe", "Points", "Depth", "Heatmap", "Runtime Modifications" };
            if (mode_value >= 0 && mode_value < 6) {
                text += "\nPreview Mode: " + String(mode_names[mode_value]);
            }
        }
    }

    static const char *node_mode_names[] = { "Off", "Wireframe", "Points", "Heatmap" };
    int node_mode = p_node->get_debug_draw_mode();
    if (node_mode >= 0 && node_mode < 4) {
        text += "\nNode Preview: " + String(node_mode_names[node_mode]);
    }
    text += "\nLOD Spheres: " + String(p_node->is_showing_lod_spheres() ? "On" : "Off");
    text += "\nOverlay: " + String(p_node->is_showing_performance_overlay() ? "On" : "Off");

    return text;
}

String format_asset_metadata_summary(const Ref<GaussianSplatAsset> &p_asset, const Dictionary &p_metadata, int p_default_thumbnail_size) {
    if (p_asset.is_null()) {
        return TTR("No Gaussian asset selected.");
    }

    Dictionary metadata = p_metadata;
    Dictionary options = metadata.get(StringName("options"), Dictionary());

    String text;

    const int splat_count = p_asset->get_splat_count();
    const int original_count = dict_get_int(metadata, StringName("original_splat_count"), splat_count);
    text += vformat(TTR("Splats: %d"), splat_count);
    if (original_count != splat_count) {
        text += vformat(TTR(" (original %d)"), original_count);
    }
    text += "\n";

    int asset_type_value = dict_get_int(metadata, StringName("asset_type"), int(p_asset->get_asset_type()));
    text += vformat(TTR("Asset Type: %s"), _asset_type_to_string(asset_type_value)) + "\n";

    String source_path = p_asset->get_source_path();
    if (source_path.is_empty()) {
        source_path = dict_get_string(metadata, StringName("source_file"));
    }
    if (!source_path.is_empty()) {
        text += vformat(TTR("Source: %s"), source_path) + "\n";
    }

    String preset_id = dict_get_string(metadata, StringName("quality_preset"), p_asset->get_import_quality_preset());
    String preset_display = dict_get_string(metadata, StringName("preset_display"), preset_id.capitalize());
    bool customized = dict_get_bool(metadata, StringName("quality_customized"), false) ||
            dict_get_bool(options, StringName("quality/customized"), false);
    if (customized && !preset_display.ends_with(TTR(" (modified)"))) {
        preset_display += TTR(" (modified)");
    }
    text += vformat(TTR("Quality: %s (%s)"), preset_display, preset_id) + "\n";

    bool enable_lod = dict_get_bool(metadata, StringName("enable_lod"),
            dict_get_bool(options, StringName("quality/enable_lod"), true));
    text += vformat(TTR("LOD: %s"), _bool_to_text(enable_lod)) + "\n";

    bool optimize_gpu = dict_get_bool(metadata, StringName("optimize_for_gpu"),
            dict_get_bool(options, StringName("quality/optimize_for_gpu"), true));
    text += vformat(TTR("Optimize for GPU: %s"), _bool_to_text(optimize_gpu)) + "\n";

    double density_multiplier = dict_get_double(metadata, StringName("density_multiplier"),
            dict_get_double(options, StringName("quality/density_multiplier"), 1.0));
    text += vformat(TTR("Density Multiplier: %.2f"), density_multiplier) + "\n";

    int max_splats = dict_get_int(metadata, StringName("max_splats"),
            dict_get_int(options, StringName("quality/max_splats"), 0));
    if (max_splats > 0) {
        text += vformat(TTR("Max Splats: %d"), max_splats) + "\n";
    }

    text += vformat(TTR("Compression: %s"), _format_compression_flags(p_asset->get_compression_flags())) + "\n";

    bool pack_opacity = dict_get_bool(metadata, StringName("pack_opacity"),
            dict_get_bool(options, StringName("compression/pack_opacity"), false));
    text += vformat(TTR("Pack Opacity: %s"), _bool_to_text(pack_opacity)) + "\n";

    bool normalize_opacity = dict_get_bool(metadata, StringName("normalize_opacity"),
            dict_get_bool(options, StringName("processing/normalize_opacity"), true));
    text += vformat(TTR("Normalize Opacity: %s"), _bool_to_text(normalize_opacity)) + "\n";

    bool sort_by_opacity = dict_get_bool(metadata, StringName("sort_by_opacity"),
            dict_get_bool(options, StringName("processing/sort_by_opacity"), false));
    text += vformat(TTR("Sort by Opacity: %s"), _bool_to_text(sort_by_opacity)) + "\n";

    double memory_estimate = metadata.get(StringName("memory_estimate_mb"), 0.0);
    if (memory_estimate > 0.0) {
        text += vformat(TTR("Estimated Memory: %.2f MB"), memory_estimate) + "\n";
        Dictionary breakdown = metadata.get(StringName("memory_breakdown_mb"), Dictionary());
        if (!breakdown.is_empty()) {
            double positions_mb = breakdown.get(StringName("positions_mb"), 0.0);
            double colors_mb = breakdown.get(StringName("colors_mb"), 0.0);
            double scales_mb = breakdown.get(StringName("scales_mb"), 0.0);
            double rotations_mb = breakdown.get(StringName("rotations_mb"), 0.0);
            text += vformat(TTR("Memory Breakdown (MB): P %.2f | C %.2f | S %.2f | R %.2f"), positions_mb, colors_mb, scales_mb,
                           rotations_mb) + "\n";
        }
    }

    bool thumbnail_generated = dict_get_bool(metadata, StringName("thumbnail_generated"),
            dict_get_bool(options, StringName("preview/generate_thumbnail"), true));
    if (thumbnail_generated) {
        int thumbnail_size = dict_get_int(metadata, StringName("thumbnail_size"),
                dict_get_int(options, StringName("preview/thumbnail_size"), p_default_thumbnail_size));
        int style_index = dict_get_int(metadata, StringName("thumbnail_style"),
                dict_get_int(options, StringName("preview/thumbnail_style"), 0));
        String style_name = dict_get_string(metadata, StringName("thumbnail_style_name"));
        if (style_name.is_empty()) {
            style_name = GaussianThumbnailGenerator::style_to_display_name(
                    GaussianThumbnailGenerator::style_from_int(style_index));
        }
        text += vformat(TTR("Thumbnail: %s (%d px)"), style_name, thumbnail_size) + "\n";
    } else {
        text += TTR("Thumbnail: Skipped") + "\n";
    }

    if (metadata.has(StringName("bounds"))) {
        AABB bounds = metadata[StringName("bounds")];
        text += vformat(TTR("Bounds: origin %s size %s"), bounds.position, bounds.size) + "\n";
    }

    Dictionary loader_stats = metadata.get(StringName("loader_statistics"), Dictionary());
    if (!loader_stats.is_empty()) {
        if (loader_stats.has(StringName("format"))) {
            text += vformat(TTR("Source Format: %s"), String(loader_stats[StringName("format")]));
            text += "\n";
        }
        if (loader_stats.has(StringName("properties"))) {
            text += vformat(TTR("Properties: %d"), int64_t(loader_stats[StringName("properties")])) + "\n";
        }
        if (loader_stats.has(StringName("splat_count"))) {
            text += vformat(TTR("Loader Splats: %d"), int64_t(loader_stats[StringName("splat_count")])) + "\n";
        }
    }

    if (metadata.has(StringName("import_time"))) {
        text += vformat(TTR("Imported: %s"), String(metadata[StringName("import_time")]));
        text += "\n";
    }

    return text.strip_edges();
}

String describe_error(Error p_error) {
    switch (p_error) {
        case OK:
            return TTR("OK");
        case ERR_FILE_NOT_FOUND:
            return TTR("File not found");
        case ERR_FILE_UNRECOGNIZED:
            return TTR("Unrecognized file format");
        case ERR_FILE_CORRUPT:
            return TTR("Corrupt or truncated file");
        case ERR_CANT_OPEN:
            return TTR("Could not open file");
        case ERR_FILE_CANT_READ:
            return TTR("Could not read file");
        case ERR_PARSE_ERROR:
            return TTR("Parse error");
        default:
            return vformat(TTR("Error %d"), int(p_error));
    }
}

String import_error_recovery_hint(Error p_error, const String &p_extension) {
    if (p_error == ERR_FILE_NOT_FOUND) {
        return TTR("Recovery: verify the file still exists under res:// and refresh the editor filesystem.");
    }
    if (p_error == ERR_FILE_UNRECOGNIZED) {
        if (p_extension == "ply") {
            return TTR("Recovery: export as Gaussian PLY with vertex properties and a valid PLY header.");
        }
        if (p_extension == "spz") {
            return TTR("Recovery: export a valid SPZ v2/v3 file (magic NGSP) from a compatible tool.");
        }
        return TTR("Recovery: verify the file extension and source exporter format.");
    }
    if (p_error == ERR_FILE_CORRUPT) {
        return TTR("Recovery: re-export the dataset and ensure the download/copy completed without truncation.");
    }
    if (p_error == ERR_CANT_OPEN || p_error == ERR_FILE_CANT_READ) {
        return TTR("Recovery: close any process locking the file and check read permissions.");
    }
    return TTR("Recovery: re-export the source dataset and retry import.");
}

String format_import_failure_message(const String &p_source_path, Error p_error, const String &p_extension, const String &p_context) {
    String text = vformat(TTR("Failed to load %s (%s, code %d)."), p_source_path, describe_error(p_error), int(p_error));
    String hint = import_error_recovery_hint(p_error, p_extension.to_lower());
    if (!hint.is_empty()) {
        text += "\n" + hint;
    }
    if (!p_context.is_empty()) {
        text += "\n" + p_context.strip_edges();
    }
    return text;
}

} // namespace GaussianEditorServices

#endif // TOOLS_ENABLED
