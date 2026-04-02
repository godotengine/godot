#include "ply_loader.h"
#include "gaussian_splat_world_io.h"
#include "core/os/os.h"
#include "core/config/project_settings.h"
#include "core/math/math_funcs.h"
#include "io_settings_utils.h"
#include "../core/gaussian_splat_world.h"
#include "../logger/gs_logger.h"

#include <climits>

namespace {

// Bump this when the GaussianSplatWorld binary format or the metadata
// contract changes.  try_load_cache() rejects caches with a different version
// so stale data is never silently reused after a format change.
static constexpr int PLY_CACHE_VERSION = 1;

static constexpr int SH_DC_COMPONENTS = 3;
static constexpr int SH_REST_COMPONENTS = 45;
static constexpr int SH_FIRST_ORDER_COEFFS = 3;
static constexpr int SH_HIGH_ORDER_COEFFS = 12;
static constexpr float SH_C0 = 0.28209479177387814f;

static bool _variant_to_uint64(const Variant &p_value, uint64_t &r_out) {
    switch (p_value.get_type()) {
        case Variant::INT: {
            int64_t v = p_value.operator int64_t();
            if (v < 0) {
                return false;
            }
            r_out = static_cast<uint64_t>(v);
            return true;
        }
        case Variant::FLOAT: {
            double v = p_value.operator double();
            if (v < 0.0) {
                return false;
            }
            r_out = static_cast<uint64_t>(Math::round(v));
            return true;
        }
        default:
            return false;
    }
}

static bool _is_ply_cache_enabled() {
    if (ProjectSettings *ps = ProjectSettings::get_singleton()) {
        return GaussianSplattingIO::get_bool_setting(ps, "rendering/gaussian_splatting/import/use_gsplatworld_cache", true);
    }
    return true;
}

static bool _is_ascii_strict_parse_enabled() {
    if (ProjectSettings *ps = ProjectSettings::get_singleton()) {
        return GaussianSplattingIO::get_bool_setting(ps, "rendering/gaussian_splatting/import/ply_ascii_strict_parse", true);
    }
    return true;
}

} // namespace

PLYLoader::PLYLoader() {
    gaussian_data.instantiate();
}

PLYLoader::~PLYLoader() {
}

void PLYLoader::_bind_methods() {
    ClassDB::bind_method(D_METHOD("load_file", "path"), &PLYLoader::load_file);
    ClassDB::bind_method(D_METHOD("get_gaussian_data"), &PLYLoader::get_gaussian_data);
    ClassDB::bind_method(D_METHOD("get_splat_count"), &PLYLoader::get_splat_count);
    ClassDB::bind_method(D_METHOD("get_load_statistics"), &PLYLoader::get_load_statistics);
    ClassDB::bind_method(D_METHOD("has_property", "name"), &PLYLoader::has_property);
    ClassDB::bind_method(D_METHOD("get_property_names"), &PLYLoader::get_property_names);
}

Error PLYLoader::load_file(const String &p_path) {
    Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::READ);
    if (file.is_null()) {
        GS_LOG_ERROR_DEFAULT("Failed to open PLY file: " + p_path);
        return ERR_FILE_NOT_FOUND;
    }

    last_cache_hit = false;
    last_header_time_us = 0;
    last_parse_time_us = 0;
    last_cache_time_us = 0;

    uint64_t start_time = OS::get_singleton()->get_ticks_usec();
    const int64_t source_size_signed = FileAccess::get_size(p_path);
    const uint64_t source_size = source_size_signed > 0
            ? static_cast<uint64_t>(source_size_signed)
            : file->get_length();
    const uint64_t source_mtime = FileAccess::get_modified_time(p_path);

    // Parse header
    uint64_t header_start = OS::get_singleton()->get_ticks_usec();
    Error err = parse_header(file);
    last_header_time_us = OS::get_singleton()->get_ticks_usec() - header_start;
    if (err != OK) {
        GS_LOG_ERROR_DEFAULT("Failed to parse PLY header");
        return err;
    }

    // Try cache before parsing large payloads
    if (_is_ply_cache_enabled()) {
        uint64_t cache_start = OS::get_singleton()->get_ticks_usec();
        if (try_load_cache(p_path, source_size, source_mtime)) {
            last_cache_time_us = OS::get_singleton()->get_ticks_usec() - cache_start;
            last_cache_hit = true;
            last_load_time_us = OS::get_singleton()->get_ticks_usec() - start_time;
            GS_LOG_STREAMING_INFO(vformat("PLY cache hit: %d splats in %.2f ms",
                    header.vertex_count, last_load_time_us / 1000.0));
            return OK;
        }
        last_cache_time_us = OS::get_singleton()->get_ticks_usec() - cache_start;
    }

    // Parse data based on format
    uint64_t parse_start = OS::get_singleton()->get_ticks_usec();
    if (header.is_binary) {
        err = parse_binary_data(file);
    } else {
        err = parse_ascii_data(file);
    }
    last_parse_time_us = OS::get_singleton()->get_ticks_usec() - parse_start;

    if (err != OK) {
        GS_LOG_ERROR_DEFAULT("Failed to parse PLY data");
        return err;
    }

    if (_is_ply_cache_enabled()) {
        write_cache(p_path, source_size, source_mtime);
    }

    last_load_time_us = OS::get_singleton()->get_ticks_usec() - start_time;
    GS_LOG_STREAMING_INFO(vformat("PLY loaded: %d splats in %.2f ms", header.vertex_count, last_load_time_us / 1000.0));

    return OK;
}

Error PLYLoader::parse_header(Ref<FileAccess> file) {
    header = PLYHeader(); // Reset header

    String line = file->get_line();
    if (line != "ply") {
        return ERR_FILE_UNRECOGNIZED;
    }

    String current_element = ""; // Track current element type being parsed
    int vertex_property_offset = 0;

    while (!file->eof_reached()) {
        line = file->get_line().strip_edges();

        if (line.begins_with("format")) {
            Vector<String> parts = line.split(" ");
            if (parts.size() >= 2) {
                if (parts[1] == "binary_little_endian") {
                    header.is_binary = true;
                    header.is_little_endian = true;
                } else if (parts[1] == "binary_big_endian") {
                    header.is_binary = true;
                    header.is_little_endian = false;
                } else if (parts[1] == "ascii") {
                    header.is_binary = false;
                }
            }
        } else if (line.begins_with("element")) {
            Vector<String> parts = line.split(" ");
            if (parts.size() >= 2) {
                current_element = parts[1]; // Could be "vertex", "face", etc.
                if (current_element == "vertex" && parts.size() >= 3) {
                    header.vertex_count = parts[2].to_int();
                    vertex_property_offset = 0; // Reset offset for vertex properties
                    header.properties.clear();
                }
            }
        } else if (line.begins_with("property") && current_element == "vertex") {
            // Only process properties if we're in the vertex element section
            PLYProperty prop;
            Vector<String> parts = line.split(" ");
            if (parts.size() >= 3) {
                if (parts[1] == "list") {
                    continue; // Unsupported property list
                }

                prop.type = parts[1];
                prop.name = parts[2];

                // Determine size based on type
                if (prop.type == "float" || prop.type == "float32") {
                    prop.size = 4;
                } else if (prop.type == "double" || prop.type == "float64") {
                    prop.size = 8;
                } else if (prop.type == "uchar" || prop.type == "uint8") {
                    prop.size = 1;
                } else if (prop.type == "ushort" || prop.type == "uint16") {
                    prop.size = 2;
                } else if (prop.type == "int" || prop.type == "int32") {
                    prop.size = 4;
                } else if (prop.type == "short" || prop.type == "int16") {
                    prop.size = 2;
                }

                prop.offset = vertex_property_offset;
                vertex_property_offset += prop.size;

                header.properties.push_back(prop);
            }
        } else if (line == "end_header") {
            header.header_size = file->get_position();
            break;
        }
    }

    if (header.vertex_count == 0) {
        return ERR_FILE_CORRUPT;
    }

    return OK;
}

bool PLYLoader::try_load_cache(const String &p_source_path, uint64_t p_source_size, uint64_t p_source_mtime) {
    if (p_source_path.is_empty()) {
        return false;
    }

    // Use .gsplatcache (not .gsplatworld) so Godot's filesystem scanner does
    // not treat PLY caches as standalone importable resources.  User-created
    // .gsplatworld files from the bake script remain importable.
    const String cache_path = p_source_path.get_basename() + ".gsplatcache";
    // Fall back to legacy .gsplatworld caches written before the rename.
    const String legacy_cache_path = p_source_path.get_basename() + ".gsplatworld";
    const bool legacy_fallback = !FileAccess::exists(cache_path) && FileAccess::exists(legacy_cache_path);
    const String effective_cache_path = legacy_fallback ? legacy_cache_path : cache_path;
    if (!FileAccess::exists(effective_cache_path)) {
        return false;
    }

    // Call the format loader directly with CACHE_MODE_IGNORE so we always
    // read from disk.  Going through ResourceLoader::load() with the default
    // CACHE_MODE_REUSE would return stale in-memory data when the cache file
    // has been rewritten during the same editor session.
    ResourceFormatLoaderGaussianSplatWorld format_loader;
    Error load_err = OK;
    Ref<Resource> resource = format_loader.load(effective_cache_path, effective_cache_path,
            &load_err, false, nullptr, ResourceFormatLoader::CACHE_MODE_IGNORE);
    Ref<GaussianSplatWorld> world = resource;
    if (world.is_null()) {
        return false;
    }

    Dictionary cache_metadata = world->get_metadata();
    uint64_t cached_size = 0;
    uint64_t cached_mtime = 0;
    if (!_variant_to_uint64(cache_metadata.get(StringName("cache_source_size"), Variant()), cached_size)) {
        return false;
    }
    if (!_variant_to_uint64(cache_metadata.get(StringName("cache_source_mtime"), Variant()), cached_mtime)) {
        return false;
    }
    if (cached_size != p_source_size || cached_mtime != p_source_mtime) {
        return false;
    }

    const Variant version_var = cache_metadata.get(StringName("cache_version"), Variant());
    if (version_var.get_type() != Variant::INT && version_var.get_type() != Variant::FLOAT) {
        return false;
    }
    if (int(version_var) != PLY_CACHE_VERSION) {
        return false;
    }

    Ref<::GaussianData> cached_data = world->get_gaussian_data();
    if (cached_data.is_null() || cached_data->get_count() <= 0) {
        return false;
    }
    if (header.vertex_count > 0 && cached_data->get_count() != header.vertex_count) {
        return false;
    }

    gaussian_data = cached_data;
    return true;
}

void PLYLoader::write_cache(const String &p_source_path, uint64_t p_source_size, uint64_t p_source_mtime) const {
    if (p_source_path.is_empty()) {
        return;
    }
    if (gaussian_data.is_null() || gaussian_data->get_count() <= 0) {
        return;
    }

    const String cache_path = p_source_path.get_basename() + ".gsplatcache";

    Ref<GaussianSplatWorld> world;
    world.instantiate();
    world->set_gaussian_data(gaussian_data);
    world->set_bounds(gaussian_data->get_aabb());

    Dictionary cache_metadata;
    cache_metadata[StringName("cache_source_path")] = p_source_path;
    cache_metadata[StringName("cache_source_size")] = static_cast<int64_t>(p_source_size);
    cache_metadata[StringName("cache_source_mtime")] = static_cast<int64_t>(p_source_mtime);
    cache_metadata[StringName("cache_version")] = PLY_CACHE_VERSION;
    world->set_metadata(cache_metadata);

    // Write through the format saver directly so .gsplatcache does not need
    // to be a globally recognised ResourceSaver extension.
    ResourceFormatSaverGaussianSplatWorld format_saver;
    Error err = format_saver.save(world, cache_path);
    if (err != OK && GaussianSplattingIO::is_data_log_enabled()) {
        GS_LOG_STREAMING_DEBUG(vformat("Failed to write PLY cache: %s (error %d)", cache_path, err));
    }
}

Error PLYLoader::parse_binary_data(Ref<FileAccess> file) {
    if (header.properties.size() == 0) {
        return ERR_FILE_CORRUPT;
    }

    gaussian_data->resize(header.vertex_count);

    // Calculate total size per vertex
    int vertex_size = 0;
    for (const PLYProperty &prop : header.properties) {
        vertex_size = MAX(vertex_size, prop.offset + prop.size);
    }

    if (vertex_size <= 0) {
        return ERR_FILE_CORRUPT;
    }

    // Precompute frequently used property indices (avoid per-vertex scans)
    const int x_idx = find_property_index("x");
    const int y_idx = find_property_index("y");
    const int z_idx = find_property_index("z");

    const int sx_idx = find_property_index("scale_0");
    const int sy_idx = find_property_index("scale_1");
    const int sz_idx = find_property_index("scale_2");

    const int rw_idx = find_property_index("rot_0");
    const int rx_idx = find_property_index("rot_1");
    const int ry_idx = find_property_index("rot_2");
    const int rz_idx = find_property_index("rot_3");

    const int opacity_idx = find_property_index("opacity");

    const int nx_idx = find_property_index("nx");
    const int ny_idx = find_property_index("ny");
    const int nz_idx = find_property_index("nz");

    // Check if normal properties exist to determine 2D mode
    const bool has_normals = (nx_idx >= 0 && ny_idx >= 0 && nz_idx >= 0);

    if (has_normals) {
        gaussian_data->set_2d_mode(true);
        GS_LOG_STREAMING_INFO("PLY contains normal vectors - enabling 2D mode");
    }

    int palette_idx = find_property_index("palette_id");
    int brush_override_idx = find_property_index("brush_override_id");
    int brush_u_idx = find_property_index("brush_axis_u");
    int brush_v_idx = find_property_index("brush_axis_v");
    int stroke_age_idx = find_property_index("stroke_age");

    int sh_dc_indices[SH_DC_COMPONENTS];
    bool dc_property_exists[SH_DC_COMPONENTS];
    for (int c = 0; c < SH_DC_COMPONENTS; c++) {
        String name = vformat("f_dc_%d", c);
        sh_dc_indices[c] = find_property_index(name);
        dc_property_exists[c] = sh_dc_indices[c] >= 0;
    }

    int sh_rest_indices[SH_REST_COMPONENTS];
    bool rest_property_exists[SH_REST_COMPONENTS];
    bool has_rest_properties = false;
    for (int idx = 0; idx < SH_REST_COMPONENTS; idx++) {
        String name = vformat("f_rest_%d", idx);
        sh_rest_indices[idx] = find_property_index(name);
        rest_property_exists[idx] = sh_rest_indices[idx] >= 0;
        if (rest_property_exists[idx]) {
            has_rest_properties = true;
        }
    }

    Gaussian default_gaussian{};
    default_gaussian.position = Vector3();
    default_gaussian.scale = Vector3(1, 1, 1);
    default_gaussian.rotation = Quaternion();
    default_gaussian.sh_dc = Color(1, 1, 1, 1);
    default_gaussian.opacity = 1.0f;
    default_gaussian.normal = Vector3(0, 0, 1);
    default_gaussian.area = 1.0f;
    default_gaussian.brush_axes = Vector2(1.0f, 1.0f);
    default_gaussian.stroke_age = 0.0f;
    default_gaussian.painterly_meta = gaussian_pack_painterly_meta(0);
    for (int j = 0; j < 3; j++) {
        default_gaussian.sh_1[j] = Vector3();
    }

    auto parse_vertex = [&](int p_index, const uint8_t *p_vertex) {
        Gaussian g = default_gaussian;

        if (x_idx >= 0 && y_idx >= 0 && z_idx >= 0) {
            g.position.x = read_float_property(p_vertex, header.properties[x_idx]);
            g.position.y = read_float_property(p_vertex, header.properties[y_idx]);
            g.position.z = read_float_property(p_vertex, header.properties[z_idx]);
        }

        float sh_dc_values[SH_DC_COMPONENTS] = {};
        bool sh_dc_present[SH_DC_COMPONENTS];
        for (int c = 0; c < SH_DC_COMPONENTS; c++) {
            sh_dc_present[c] = dc_property_exists[c];
            if (sh_dc_present[c]) {
                sh_dc_values[c] = read_float_property(p_vertex, header.properties[sh_dc_indices[c]]);
            }
        }

        float sh_rest_values[SH_REST_COMPONENTS] = {};
        bool sh_rest_present[SH_REST_COMPONENTS];
        if (has_rest_properties) {
            for (int idx = 0; idx < SH_REST_COMPONENTS; idx++) {
                sh_rest_present[idx] = rest_property_exists[idx];
                if (sh_rest_present[idx]) {
                    sh_rest_values[idx] = read_float_property(p_vertex, header.properties[sh_rest_indices[idx]]);
                }
            }
        } else {
            for (int idx = 0; idx < SH_REST_COMPONENTS; idx++) {
                sh_rest_present[idx] = false;
                sh_rest_values[idx] = 0.0f;
            }
        }

        float sh_coeffs[SH_DC_COMPONENTS + SH_REST_COMPONENTS];
        int sh_float_count = assemble_sh_coefficients(g,
                sh_dc_values,
                sh_dc_present,
                sh_rest_values,
                sh_rest_present,
                sh_coeffs);

        if (sx_idx >= 0 && sy_idx >= 0 && sz_idx >= 0) {
            g.scale.x = exp(read_float_property(p_vertex, header.properties[sx_idx]));
            g.scale.y = exp(read_float_property(p_vertex, header.properties[sy_idx]));
            g.scale.z = exp(read_float_property(p_vertex, header.properties[sz_idx]));
        }

        if (rw_idx >= 0 && rx_idx >= 0 && ry_idx >= 0 && rz_idx >= 0) {
            g.rotation.w = read_float_property(p_vertex, header.properties[rw_idx]);
            g.rotation.x = read_float_property(p_vertex, header.properties[rx_idx]);
            g.rotation.y = read_float_property(p_vertex, header.properties[ry_idx]);
            g.rotation.z = read_float_property(p_vertex, header.properties[rz_idx]);
            g.rotation.normalize();
        }

        if (opacity_idx >= 0) {
            float opacity = read_float_property(p_vertex, header.properties[opacity_idx]);
            g.opacity = 1.0f / (1.0f + exp(-opacity));
        }

        if (has_normals) {
            g.normal.x = read_float_property(p_vertex, header.properties[nx_idx]);
            g.normal.y = read_float_property(p_vertex, header.properties[ny_idx]);
            g.normal.z = read_float_property(p_vertex, header.properties[nz_idx]);
            g.normal.normalize();
        }

        if (palette_idx >= 0) {
            uint32_t palette_value = read_uint_property(p_vertex, header.properties[palette_idx]);
            uint16_t palette = (uint16_t)CLAMP(palette_value, 0u, 65535u);
            g.painterly_meta = gaussian_set_palette_id(g.painterly_meta, palette);
        }

        if (brush_override_idx >= 0) {
            uint32_t brush_override_value = read_uint_property(p_vertex, header.properties[brush_override_idx]);
            uint16_t brush_override_id = (uint16_t)CLAMP(brush_override_value, 0u, 65535u);
            g.painterly_meta = gaussian_set_brush_override_id(g.painterly_meta, brush_override_id);
        }

        if (brush_u_idx >= 0 && brush_v_idx >= 0) {
            g.brush_axes.x = read_float_property(p_vertex, header.properties[brush_u_idx]);
            g.brush_axes.y = read_float_property(p_vertex, header.properties[brush_v_idx]);
        }

        if (stroke_age_idx >= 0) {
            g.stroke_age = read_float_property(p_vertex, header.properties[stroke_age_idx]);
        }

        gaussian_data->set_gaussian(p_index, g);
        gaussian_data->set_spherical_harmonics(p_index, sh_coeffs, sh_float_count);

        if (GaussianSplattingIO::is_data_log_enabled() && p_index == 0) {
            GS_LOG_STREAMING_DEBUG(vformat("[PLY] Loaded %d SH floats for gaussian 0", sh_float_count));
            if (sh_float_count >= 6) {
                GS_LOG_STREAMING_DEBUG(vformat("[PLY] First 3 SH coeffs (after DC): [%f, %f, %f]",
                    sh_coeffs[3], sh_coeffs[4], sh_coeffs[5]));
            }
        }
    };

    const uint64_t total_bytes = uint64_t(vertex_size) * uint64_t(header.vertex_count);
    const uint64_t max_bulk_bytes = static_cast<uint64_t>(INT_MAX);
    const uint64_t target_chunk_bytes = 16ull * 1024ull * 1024ull;
    uint64_t max_vertices_per_chunk = target_chunk_bytes / uint64_t(vertex_size);
    if (max_vertices_per_chunk == 0) {
        max_vertices_per_chunk = 1;
    }

    if (total_bytes <= max_bulk_bytes) {
        Vector<uint8_t> bulk_buffer;
        bulk_buffer.resize(static_cast<int>(total_bytes));
        const uint64_t read = file->get_buffer(bulk_buffer.ptrw(), total_bytes);
        if (read != total_bytes) {
            return ERR_FILE_CORRUPT;
        }

        const uint8_t *data = bulk_buffer.ptr();
        for (int i = 0; i < header.vertex_count; i++) {
            const uint8_t *vertex = data + uint64_t(i) * uint64_t(vertex_size);
            parse_vertex(i, vertex);
        }
    } else {
        Vector<uint8_t> chunk_buffer;
        int remaining = header.vertex_count;
        int base_index = 0;
        while (remaining > 0) {
            const uint64_t batch_vertices_u = (max_vertices_per_chunk < static_cast<uint64_t>(remaining))
                    ? max_vertices_per_chunk
                    : static_cast<uint64_t>(remaining);
            const int batch_vertices = static_cast<int>(batch_vertices_u);
            const uint64_t batch_bytes = uint64_t(batch_vertices) * uint64_t(vertex_size);
            chunk_buffer.resize(static_cast<int>(batch_bytes));
            const uint64_t read = file->get_buffer(chunk_buffer.ptrw(), batch_bytes);
            if (read != batch_bytes) {
                return ERR_FILE_CORRUPT;
            }

            const uint8_t *data = chunk_buffer.ptr();
            for (int local = 0; local < batch_vertices; local++) {
                const uint8_t *vertex = data + uint64_t(local) * uint64_t(vertex_size);
                parse_vertex(base_index + local, vertex);
            }

            base_index += batch_vertices;
            remaining -= batch_vertices;
        }
    }

    return OK;
}

Error PLYLoader::parse_ascii_data(Ref<FileAccess> file) {
    gaussian_data->resize(header.vertex_count);
    const bool strict_ascii_parse = _is_ascii_strict_parse_enabled();

    // Check if normal properties exist to determine 2D mode
    bool has_normals = (find_property_index("nx") >= 0 &&
                        find_property_index("ny") >= 0 &&
                        find_property_index("nz") >= 0);

    if (has_normals) {
        gaussian_data->set_2d_mode(true);
        GS_LOG_STREAMING_INFO("PLY contains normal vectors - enabling 2D mode");
    }

    int sh_dc_indices[SH_DC_COMPONENTS];
    bool dc_property_exists[SH_DC_COMPONENTS];
    for (int c = 0; c < SH_DC_COMPONENTS; c++) {
        String name = vformat("f_dc_%d", c);
        sh_dc_indices[c] = find_property_index(name);
        dc_property_exists[c] = sh_dc_indices[c] >= 0;
    }

    int sh_rest_indices[SH_REST_COMPONENTS];
    bool rest_property_exists[SH_REST_COMPONENTS];
    bool has_rest_properties = false;
    for (int idx = 0; idx < SH_REST_COMPONENTS; idx++) {
        String name = vformat("f_rest_%d", idx);
        sh_rest_indices[idx] = find_property_index(name);
        rest_property_exists[idx] = sh_rest_indices[idx] >= 0;
        if (rest_property_exists[idx]) {
            has_rest_properties = true;
        }
    }

    // ASCII parsing - simpler but slower
    for (int i = 0; i < header.vertex_count; i++) {
        if (file->eof_reached()) {
            return ERR_FILE_CORRUPT;
        }

        String line = file->get_line().strip_edges();
        // Split by any whitespace (spaces, tabs) for robust parsing
        Vector<String> values = line.split_spaces();

        if (values.size() < header.properties.size()) {
            String msg = vformat("[PLY ASCII] Malformed row at vertex index %d: expected at least %d fields, got %d.",
                    i, header.properties.size(), values.size());
            if (strict_ascii_parse) {
                GS_LOG_ERROR_DEFAULT(msg + " Treating file as corrupt.");
                return ERR_FILE_CORRUPT;
            }
            WARN_PRINT(msg + " Skipping row because strict parsing is disabled.");
            continue;
        }

        Gaussian g;
        // Set defaults
        g.position = Vector3();
        g.scale = Vector3(1, 1, 1);
        g.rotation = Quaternion();
        g.sh_dc = Color(1, 1, 1, 1);
        g.opacity = 1.0f;
        g.normal = Vector3(0, 0, 1);
        g.area = 1.0f;
        g.brush_axes = Vector2(1.0f, 1.0f);
        g.stroke_age = 0.0f;
        g.painterly_meta = gaussian_pack_painterly_meta(0);
        for (int k = 0; k < 3; k++) {
            g.sh_1[k] = Vector3();
        }

        // Parse based on property order
        for (int j = 0; j < header.properties.size(); j++) {
            const PLYProperty &prop = header.properties[j];
            const String &token = values[j];
            if (strict_ascii_parse && !token.is_valid_float()) {
                GS_LOG_ERROR_DEFAULT(vformat("[PLY ASCII] Invalid numeric token at vertex index %d property '%s' (column %d): '%s'.",
                        i, prop.name, j, token));
                return ERR_FILE_CORRUPT;
            }
            float value = token.to_float();

            if (prop.name == "x") {
                g.position.x = value;
            } else if (prop.name == "y") {
                g.position.y = value;
            } else if (prop.name == "z") {
                g.position.z = value;
            } else if (prop.name == "scale_0") {
                g.scale.x = exp(value);
            } else if (prop.name == "scale_1") {
                g.scale.y = exp(value);
            } else if (prop.name == "scale_2") {
                g.scale.z = exp(value);
            } else if (prop.name == "rot_0") {
                g.rotation.w = value;
            } else if (prop.name == "rot_1") {
                g.rotation.x = value;
            } else if (prop.name == "rot_2") {
                g.rotation.y = value;
            } else if (prop.name == "rot_3") {
                g.rotation.z = value;
            } else if (prop.name == "opacity") {
                g.opacity = 1.0f / (1.0f + exp(-value));
            } else if (prop.name == "palette_id") {
                g.painterly_meta = gaussian_set_palette_id(g.painterly_meta, (uint16_t)CLAMP((int)value, 0, 65535));
            } else if (prop.name == "brush_override_id") {
                g.painterly_meta = gaussian_set_brush_override_id(g.painterly_meta, (uint16_t)CLAMP((int)value, 0, 65535));
            } else if (prop.name == "brush_axis_u") {
                g.brush_axes.x = value;
            } else if (prop.name == "brush_axis_v") {
                g.brush_axes.y = value;
            } else if (prop.name == "stroke_age") {
                g.stroke_age = value;
            } else if (prop.name == "nx") {
                g.normal.x = value;
            } else if (prop.name == "ny") {
                g.normal.y = value;
            } else if (prop.name == "nz") {
                g.normal.z = value;
            }
        }

        float sh_dc_values[SH_DC_COMPONENTS] = {};
        bool sh_dc_present[SH_DC_COMPONENTS];
        for (int c = 0; c < SH_DC_COMPONENTS; c++) {
            sh_dc_present[c] = dc_property_exists[c];
            if (sh_dc_present[c] && sh_dc_indices[c] < values.size()) {
                sh_dc_values[c] = values[sh_dc_indices[c]].to_float();
            }
        }

        float sh_rest_values[SH_REST_COMPONENTS] = {};
        bool sh_rest_present[SH_REST_COMPONENTS];
        if (has_rest_properties) {
            for (int idx = 0; idx < SH_REST_COMPONENTS; idx++) {
                sh_rest_present[idx] = rest_property_exists[idx];
                if (sh_rest_present[idx] && sh_rest_indices[idx] < values.size()) {
                    sh_rest_values[idx] = values[sh_rest_indices[idx]].to_float();
                }
            }
        } else {
            for (int idx = 0; idx < SH_REST_COMPONENTS; idx++) {
                sh_rest_present[idx] = false;
                sh_rest_values[idx] = 0.0f;
            }
        }

        float sh_coeffs[SH_DC_COMPONENTS + SH_REST_COMPONENTS];
        int sh_float_count = assemble_sh_coefficients(g,
                sh_dc_values,
                sh_dc_present,
                sh_rest_values,
                sh_rest_present,
                sh_coeffs);

        g.rotation.normalize();

        // Normalize the normal vector if it was loaded
        if (has_normals) {
            g.normal.normalize();
        }

        gaussian_data->set_gaussian(i, g);
        gaussian_data->set_spherical_harmonics(i, sh_coeffs, sh_float_count);

        // DEBUG: Log first gaussian SH data
        if (GaussianSplattingIO::is_data_log_enabled() && i == 0) {
            GS_LOG_STREAMING_DEBUG(vformat("[PLY ASCII] Loaded %d SH floats for gaussian 0", sh_float_count));
            if (sh_float_count >= 6) {
                GS_LOG_STREAMING_DEBUG(vformat("[PLY ASCII] First 3 SH coeffs (after DC): [%f, %f, %f]",
                    sh_coeffs[3], sh_coeffs[4], sh_coeffs[5]));
            }
        }
    }

    return OK;
}

int PLYLoader::assemble_sh_coefficients(Gaussian &r_gaussian,
        const float *p_dc_values,
        const bool *p_dc_present,
        const float *p_rest_values,
        const bool *p_rest_present,
        float *r_output) const {
    // -----------------------------------------------------------------
    // SH sign convention documentation (ISSUE-038)
    // -----------------------------------------------------------------
    // PLY files from 3DGS (Kerbl et al. 2023) store SH coefficients that
    // were trained with the Condon-Shortley (CS) phase baked into the
    // coefficient values, using real spherical harmonics.  Specifically:
    //
    //   Band 0 (l=0):  Y_0^0 = SH_C0                   (no sign issue)
    //   Band 1 (l=1):  Y_1^{-1}=-C1*y, Y_1^0=C1*z, Y_1^1=-C1*x
    //   Band 2 (l=2):  standard real SH with CS phase
    //   Band 3 (l=3):  standard real SH with CS phase
    //
    // The loader does NOT negate or transform any coefficients; it passes
    // them through as stored in the PLY (channel-major -> coeff-major
    // repack only).  The GPU shader (gaussian_splat_common_inc.glsl,
    // tile_binning.glsl) evaluates the SH basis with the same CS-phase
    // sign convention, so the two are consistent.
    //
    // If importing from a source that omits the CS phase, a sign flip on
    // odd-m coefficients would be needed here.
    // -----------------------------------------------------------------

    // DC term corresponds to spherical harmonics band l=0.
    // PLY stores DC as SH coefficients (scaled by SH_C0), centered around 0.
    // We add 0.5 here to convert to 0-1 color space, matching SPZ format.
    // This allows shaders to use DC directly without format-specific offsets.
    Color sh_dc = r_gaussian.sh_dc;
    if (p_dc_present[0]) {
        sh_dc.r = SH_C0 * p_dc_values[0];
    }
    if (p_dc_present[1]) {
        sh_dc.g = SH_C0 * p_dc_values[1];
    }
    if (p_dc_present[2]) {
        sh_dc.b = SH_C0 * p_dc_values[2];
    }
    sh_dc.a = 1.0f;
    r_gaussian.sh_dc = sh_dc;

    // PLY stores f_rest_0-44 in channel-major order (all R coeffs, then G, then B).
    // Repack into coefficient-major RGB triplets for the renderer.
    const int sh_coeffs_per_channel = SH_REST_COMPONENTS / 3;
    Vector3 sh_band1[SH_FIRST_ORDER_COEFFS];
    for (int i = 0; i < SH_FIRST_ORDER_COEFFS; i++) {
        sh_band1[i] = Vector3();
    }
    uint32_t first_order_count = 0;
    for (int coeff = 0; coeff < SH_FIRST_ORDER_COEFFS; coeff++) {
        Vector3 coeff_value;
        bool has_value = false;
        int base_r = coeff;
        int base_g = coeff + sh_coeffs_per_channel;
        int base_b = coeff + 2 * sh_coeffs_per_channel;
        if (base_r < SH_REST_COMPONENTS && p_rest_present[base_r]) {
            coeff_value.x = p_rest_values[base_r];
            has_value = true;
        }
        if (base_g < SH_REST_COMPONENTS && p_rest_present[base_g]) {
            coeff_value.y = p_rest_values[base_g];
            has_value = true;
        }
        if (base_b < SH_REST_COMPONENTS && p_rest_present[base_b]) {
            coeff_value.z = p_rest_values[base_b];
            has_value = true;
        }
        if (has_value) {
            first_order_count = coeff + 1;
        }
        sh_band1[coeff] = coeff_value;
    }

    // Higher order bands (l=2..3) occupy the remaining coefficients.
    Vector3 sh_high[SH_HIGH_ORDER_COEFFS];
    for (int i = 0; i < SH_HIGH_ORDER_COEFFS; i++) {
        sh_high[i] = Vector3();
    }
    uint32_t high_order_count = 0;
    for (int coeff = 0; coeff < SH_HIGH_ORDER_COEFFS; coeff++) {
        Vector3 coeff_value;
        bool has_value = false;
        int coeff_index = coeff + SH_FIRST_ORDER_COEFFS;
        int base_r = coeff_index;
        int base_g = coeff_index + sh_coeffs_per_channel;
        int base_b = coeff_index + 2 * sh_coeffs_per_channel;
        if (base_r < SH_REST_COMPONENTS && p_rest_present[base_r]) {
            coeff_value.x = p_rest_values[base_r];
            has_value = true;
        }
        if (base_g < SH_REST_COMPONENTS && p_rest_present[base_g]) {
            coeff_value.y = p_rest_values[base_g];
            has_value = true;
        }
        if (base_b < SH_REST_COMPONENTS && p_rest_present[base_b]) {
            coeff_value.z = p_rest_values[base_b];
            has_value = true;
        }
        if (has_value) {
            high_order_count = coeff + 1;
        }
        sh_high[coeff] = coeff_value;
    }

    for (int i = 0; i < SH_FIRST_ORDER_COEFFS; i++) {
        if ((uint32_t)i < first_order_count) {
            r_gaussian.sh_1[i] = sh_band1[i];
        } else {
            r_gaussian.sh_1[i] = Vector3();
        }
    }

    int out_index = 0;
    r_output[out_index++] = sh_dc.r;
    r_output[out_index++] = sh_dc.g;
    r_output[out_index++] = sh_dc.b;

    for (uint32_t i = 0; i < first_order_count; i++) {
        r_output[out_index++] = sh_band1[i].x;
        r_output[out_index++] = sh_band1[i].y;
        r_output[out_index++] = sh_band1[i].z;
    }

    for (uint32_t i = 0; i < high_order_count; i++) {
        r_output[out_index++] = sh_high[i].x;
        r_output[out_index++] = sh_high[i].y;
        r_output[out_index++] = sh_high[i].z;
    }

    return out_index;
}

int PLYLoader::find_property_index(const String &name) const {
    for (int i = 0; i < header.properties.size(); i++) {
        if (header.properties[i].name == name) {
            return i;
        }
    }
    return -1;
}

float PLYLoader::read_float_property(const uint8_t *data, const PLYProperty &prop) const {
    if (prop.type == "float" || prop.type == "float32") {
        float value;
        memcpy(&value, data + prop.offset, sizeof(float));
        if (!header.is_little_endian) {
            // Swap bytes for big endian
            uint32_t *int_val = (uint32_t*)&value;
            *int_val = ((*int_val & 0xFF000000) >> 24) |
                      ((*int_val & 0x00FF0000) >> 8) |
                      ((*int_val & 0x0000FF00) << 8) |
                      ((*int_val & 0x000000FF) << 24);
        }
        return value;
    } else if (prop.type == "double" || prop.type == "float64") {
        double value;
        memcpy(&value, data + prop.offset, sizeof(double));
        if (!header.is_little_endian) {
            // Swap bytes for big endian (64-bit)
            uint64_t *int_val = (uint64_t*)&value;
            *int_val = ((*int_val & 0xFF00000000000000ULL) >> 56) |
                      ((*int_val & 0x00FF000000000000ULL) >> 40) |
                      ((*int_val & 0x0000FF0000000000ULL) >> 24) |
                      ((*int_val & 0x000000FF00000000ULL) >> 8) |
                      ((*int_val & 0x00000000FF000000ULL) << 8) |
                      ((*int_val & 0x0000000000FF0000ULL) << 24) |
                      ((*int_val & 0x000000000000FF00ULL) << 40) |
                      ((*int_val & 0x00000000000000FFULL) << 56);
        }
        return (float)value;
    }
    return 0.0f;
}

uint32_t PLYLoader::read_uint_property(const uint8_t *data, const PLYProperty &prop) const {
    if (prop.type == "uchar" || prop.type == "uint8") {
        return data[prop.offset];
    } else if (prop.type == "ushort" || prop.type == "uint16") {
        uint16_t value;
        memcpy(&value, data + prop.offset, sizeof(uint16_t));
        if (!header.is_little_endian) {
            value = (uint16_t)(((value & 0x00FF) << 8) | ((value & 0xFF00) >> 8));
        }
        return value;
    } else if (prop.type == "uint" || prop.type == "uint32") {
        uint32_t value;
        memcpy(&value, data + prop.offset, sizeof(uint32_t));
        if (!header.is_little_endian) {
            value = ((value & 0xFF000000) >> 24) |
                    ((value & 0x00FF0000) >> 8) |
                    ((value & 0x0000FF00) << 8) |
                    ((value & 0x000000FF) << 24);
        }
        return value;
    } else if (prop.type == "char" || prop.type == "int8") {
        int8_t value = *(int8_t*)(data + prop.offset);
        return (uint32_t)MAX(value, (int8_t)0);
    } else if (prop.type == "short" || prop.type == "int16") {
        int16_t value;
        memcpy(&value, data + prop.offset, sizeof(int16_t));
        if (!header.is_little_endian) {
            value = (int16_t)(((value & 0x00FF) << 8) | ((value & 0xFF00) >> 8));
        }
        return (uint32_t)MAX(value, (int16_t)0);
    } else if (prop.type == "int" || prop.type == "int32") {
        int32_t value;
        memcpy(&value, data + prop.offset, sizeof(int32_t));
        if (!header.is_little_endian) {
            uint32_t *int_val = (uint32_t*)&value;
            *int_val = ((*int_val & 0xFF000000) >> 24) |
                      ((*int_val & 0x00FF0000) >> 8) |
                      ((*int_val & 0x0000FF00) << 8) |
                      ((*int_val & 0x000000FF) << 24);
        }
        return (uint32_t)MAX(value, 0);
    }
    return 0;
}

int PLYLoader::get_splat_count() const {
    return gaussian_data.is_valid() ? gaussian_data->get_count() : 0;
}

Dictionary PLYLoader::get_load_statistics() const {
    Dictionary stats;
    stats["splat_count"] = get_splat_count();
    stats["format"] = header.is_binary ? "binary" : "ascii";
    stats["properties"] = header.properties.size();
    if (last_load_time_us > 0) {
        stats["load_time_ms"] = last_load_time_us / 1000.0;
        stats["header_time_ms"] = last_header_time_us / 1000.0;
        stats["parse_time_ms"] = last_parse_time_us / 1000.0;
        stats["cache_time_ms"] = last_cache_time_us / 1000.0;
        stats["cache_hit"] = last_cache_hit;
    }

    if (gaussian_data.is_valid()) {
        AABB aabb = gaussian_data->get_aabb();
        stats["bounds_min"] = aabb.position;
        stats["bounds_max"] = aabb.position + aabb.size;
    }

    return stats;
}

void PLYLoader::get_property_deficiencies(PackedStringArray &r_missing_required, PackedStringArray &r_missing_optional) const {
    r_missing_required = PackedStringArray();
    r_missing_optional = PackedStringArray();

    static const char *required_props[] = {
        "x", "y", "z",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
        "opacity"
    };

    for (const char *prop : required_props) {
        if (!has_property(prop)) {
            r_missing_required.push_back(prop);
        }
    }

    static const char *optional_props[] = {
        "nx", "ny", "nz",
        "palette_id",
        "brush_override_id",
        "brush_axis_u", "brush_axis_v",
        "stroke_age",
        "f_rest_0", "f_rest_1", "f_rest_2"
    };

    for (const char *prop : optional_props) {
        if (!has_property(prop)) {
            r_missing_optional.push_back(prop);
        }
    }
}

Dictionary PLYLoader::get_property_summary() const {
    Dictionary summary;

    PackedStringArray property_names;
    for (const PLYProperty &prop : header.properties) {
        property_names.push_back(prop.name);
    }

    PackedStringArray missing_required;
    PackedStringArray missing_optional;
    get_property_deficiencies(missing_required, missing_optional);

    summary["property_names"] = property_names;
    summary["property_count"] = property_names.size();
    summary["missing_required"] = missing_required;
    summary["missing_optional"] = missing_optional;
    summary["has_normals"] = has_property("nx") && has_property("ny") && has_property("nz");
    summary["has_palette"] = has_property("palette_id");
    summary["has_brush_override_id"] = has_property("brush_override_id");

    return summary;
}

bool PLYLoader::has_property(const String &p_name) const {
    return find_property_index(p_name) >= 0;
}

Vector<String> PLYLoader::get_property_names() const {
    Vector<String> names;
    for (const PLYProperty &prop : header.properties) {
        names.push_back(prop.name);
    }
    return names;
}
