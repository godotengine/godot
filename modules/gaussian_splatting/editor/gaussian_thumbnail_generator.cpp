#ifdef TOOLS_ENABLED

#include "gaussian_thumbnail_generator.h"

#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/math/color.h"
#include "core/math/math_funcs.h"
#include "core/math/quaternion.h"
#include "core/templates/local_vector.h"
#include "core/variant/dictionary.h"
#include "scene/resources/image_texture.h"

#include <cfloat>
#include <cstring>

const char *GaussianThumbnailGenerator::DISK_CACHE_DIR = "res://.godot/cache/gaussian_thumbnails";

namespace {

static uint64_t _mix_hash64(uint64_t p_hash, uint64_t p_value) {
    p_hash ^= p_value + 0x9e3779b97f4a7c15ULL + (p_hash << 6) + (p_hash >> 2);
    return p_hash;
}

static uint32_t _float_to_bits(float p_value) {
    uint32_t bits = 0;
    memcpy(&bits, &p_value, sizeof(float));
    return bits;
}

static void _mix_float_array_samples(uint64_t &r_hash, const PackedFloat32Array &p_values) {
    const int size = p_values.size();
    r_hash = _mix_hash64(r_hash, uint64_t(size));
    if (size <= 0) {
        return;
    }

    const int candidates[4] = { 0, size / 3, (size * 2) / 3, size - 1 };
    int used[4];
    int used_count = 0;

    for (int i = 0; i < 4; i++) {
        const int index = candidates[i];
        if (index < 0 || index >= size) {
            continue;
        }

        bool duplicate = false;
        for (int j = 0; j < used_count; j++) {
            if (used[j] == index) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) {
            continue;
        }

        used[used_count++] = index;
        r_hash = _mix_hash64(r_hash, uint64_t(index));
        r_hash = _mix_hash64(r_hash, uint64_t(_float_to_bits(p_values[index])));
    }
}

static void _mix_color_array_samples(uint64_t &r_hash, const PackedColorArray &p_values) {
    const int size = p_values.size();
    r_hash = _mix_hash64(r_hash, uint64_t(size));
    if (size <= 0) {
        return;
    }

    const int candidates[4] = { 0, size / 3, (size * 2) / 3, size - 1 };
    int used[4];
    int used_count = 0;

    for (int i = 0; i < 4; i++) {
        const int index = candidates[i];
        if (index < 0 || index >= size) {
            continue;
        }

        bool duplicate = false;
        for (int j = 0; j < used_count; j++) {
            if (used[j] == index) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) {
            continue;
        }

        used[used_count++] = index;
        const Color sample = p_values[index];
        r_hash = _mix_hash64(r_hash, uint64_t(index));
        r_hash = _mix_hash64(r_hash, uint64_t(_float_to_bits(sample.r)));
        r_hash = _mix_hash64(r_hash, uint64_t(_float_to_bits(sample.g)));
        r_hash = _mix_hash64(r_hash, uint64_t(_float_to_bits(sample.b)));
        r_hash = _mix_hash64(r_hash, uint64_t(_float_to_bits(sample.a)));
    }
}

} // namespace

uint64_t GaussianThumbnailGenerator::_compute_asset_fingerprint(const Ref<GaussianSplatAsset> &p_asset) const {
    if (p_asset.is_null()) {
        return 0;
    }

    uint64_t hash = 1469598103934665603ULL;
    hash = _mix_hash64(hash, uint64_t(p_asset->get_splat_count()));
    hash = _mix_hash64(hash, uint64_t(p_asset->get_compression_flags()));
    hash = _mix_hash64(hash, uint64_t(p_asset->get_source_path().hash()));
    hash = _mix_hash64(hash, uint64_t(p_asset->get_path().hash()));

    Dictionary metadata = p_asset->get_import_metadata();
    if (!metadata.is_empty()) {
        hash = _mix_hash64(hash, uint64_t(String(metadata.get(StringName("import_time"), String())).hash()));
        hash = _mix_hash64(hash, uint64_t(String(metadata.get(StringName("source_file"), String())).hash()));
    }

    _mix_float_array_samples(hash, p_asset->get_positions());
    _mix_color_array_samples(hash, p_asset->get_colors());
    _mix_float_array_samples(hash, p_asset->get_scales());
    _mix_float_array_samples(hash, p_asset->get_rotations());
    _mix_float_array_samples(hash, p_asset->get_opacities());
    _mix_float_array_samples(hash, p_asset->get_normals());

    return hash;
}

String GaussianThumbnailGenerator::_build_cache_key(const Ref<GaussianSplatAsset> &p_asset, int p_size, ThumbnailStyle p_style) const {
    if (p_asset.is_null()) {
        return String();
    }

    String asset_id = p_asset->get_source_path();
    if (asset_id.is_empty()) {
        asset_id = p_asset->get_path();
    }
    if (asset_id.is_empty()) {
        asset_id = "<in-memory>";
    }

    const uint64_t fingerprint = _compute_asset_fingerprint(p_asset);
    return asset_id + "|" + String::num_uint64(fingerprint) + "|" + itos(p_size) + "|" + itos(int(p_style));
}

void GaussianThumbnailGenerator::_touch_cache_key(const String &p_key) const {
    for (int i = 0; i < thumbnail_cache_order.size(); i++) {
        if (thumbnail_cache_order[i] == p_key) {
            thumbnail_cache_order.remove_at(i);
            break;
        }
    }
    thumbnail_cache_order.push_back(p_key);
}

void GaussianThumbnailGenerator::_prune_cache_if_needed() const {
    while (thumbnail_cache_order.size() > MAX_CACHE_ENTRIES) {
        const String oldest_key = thumbnail_cache_order[0];
        thumbnail_cache_order.remove_at(0);
        thumbnail_cache.erase(oldest_key);
    }
}

bool GaussianThumbnailGenerator::_ensure_disk_cache_dir() const {
    Ref<DirAccess> dir = DirAccess::open("res://");
    if (dir.is_null()) {
        return false;
    }
    if (dir->dir_exists(DISK_CACHE_DIR)) {
        return true;
    }
    Error err = dir->make_dir_recursive(String(DISK_CACHE_DIR).trim_prefix("res://"));
    return err == OK;
}

String GaussianThumbnailGenerator::_disk_cache_path_for_key(uint64_t p_fingerprint, int p_size, ThumbnailStyle p_style) const {
    return String(DISK_CACHE_DIR) + "/" + String::num_uint64(p_fingerprint, 16) + "_" + itos(p_size) + "_" + itos(int(p_style)) + ".png";
}

Ref<Texture2D> GaussianThumbnailGenerator::_load_from_disk_cache(uint64_t p_fingerprint, int p_size, ThumbnailStyle p_style) const {
    String path = _disk_cache_path_for_key(p_fingerprint, p_size, p_style);
    if (!FileAccess::exists(path)) {
        return Ref<Texture2D>();
    }

    Ref<Image> image;
    image.instantiate();
    Error err = image->load(path);
    if (err != OK || image->is_empty()) {
        return Ref<Texture2D>();
    }

    return ImageTexture::create_from_image(image);
}

void GaussianThumbnailGenerator::_save_to_disk_cache(uint64_t p_fingerprint, int p_size, ThumbnailStyle p_style, const Ref<Texture2D> &p_texture) const {
    if (p_texture.is_null()) {
        return;
    }

    Ref<Image> image = p_texture->get_image();
    if (image.is_null() || image->is_empty()) {
        return;
    }

    if (!_ensure_disk_cache_dir()) {
        return;
    }

    String path = _disk_cache_path_for_key(p_fingerprint, p_size, p_style);
    image->save_png(path);
}

Dictionary GaussianThumbnailGenerator::_project_to_canvas(const Ref<GaussianSplatAsset> &p_asset, int p_size,
        Vector<int> &r_hits, Vector<Color> &r_accum) const {
    Dictionary result;
    r_hits.clear();
    r_accum.clear();

    r_hits.resize(p_size * p_size);
    r_accum.resize(p_size * p_size);

    int *hits_ptr = r_hits.ptrw();
    Color *accum_ptr = r_accum.ptrw();
    for (int i = 0; i < p_size * p_size; i++) {
        hits_ptr[i] = 0;
        accum_ptr[i] = Color();
    }

    PackedFloat32Array positions = p_asset->get_positions();
    PackedColorArray colors = p_asset->get_colors();
    PackedFloat32Array scales = p_asset->get_scales();
    const int splat_count = p_asset->get_splat_count();

    Vector3 min_pos(FLT_MAX, FLT_MAX, FLT_MAX);
    Vector3 max_pos(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    for (int i = 0; i < splat_count; i++) {
        int base = i * 3;
        if (base + 2 >= positions.size()) {
            continue;
        }
        Vector3 pos(positions[base], positions[base + 1], positions[base + 2]);
        min_pos = min_pos.min(pos);
        max_pos = max_pos.max(pos);
    }

    if (min_pos.x >= max_pos.x) {
        max_pos.x = min_pos.x + 1.0f;
    }
    if (min_pos.y >= max_pos.y) {
        max_pos.y = min_pos.y + 1.0f;
    }
    if (min_pos.z >= max_pos.z) {
        max_pos.z = min_pos.z + 1.0f;
    }

    Vector3 extent = max_pos - min_pos;
    extent.x = MAX(extent.x, 0.0001f);
    extent.y = MAX(extent.y, 0.0001f);
    extent.z = MAX(extent.z, 0.0001f);

    const int color_size = colors.size();

    result[StringName("min_pos")] = min_pos;
    result[StringName("max_pos")] = max_pos;
    result[StringName("extent")] = extent;

    for (int i = 0; i < splat_count; i++) {
        int base = i * 3;
        if (base + 2 >= positions.size()) {
            continue;
        }

        Vector3 pos(positions[base], positions[base + 1], positions[base + 2]);
        float u = (pos.x - min_pos.x) / extent.x;
        float v = (pos.z - min_pos.z) / extent.z;

        int px = CLAMP((int)Math::round(u * (p_size - 1)), 0, p_size - 1);
        int py = CLAMP((int)Math::round((1.0f - v) * (p_size - 1)), 0, p_size - 1);
        int idx = py * p_size + px;

        Color c = Color(1.0f, 1.0f, 1.0f, 1.0f);
        if (i < color_size) {
            c = colors[i];
        }
        accum_ptr[idx] += c;
        hits_ptr[idx] += 1;
    }

    if (scales.size() >= splat_count * 3) {
        double avg_scale = 0.0;
        for (int i = 0; i < splat_count; i++) {
            avg_scale += double(scales[i * 3 + 0] + scales[i * 3 + 1] + scales[i * 3 + 2]);
        }
        avg_scale /= MAX(1, splat_count * 3);
        result[StringName("average_scale")] = avg_scale;
    }

    return result;
}

Ref<Texture2D> GaussianThumbnailGenerator::_generate_color_thumbnail(const Ref<GaussianSplatAsset> &p_asset, int p_size) const {
    Vector<int> hits;
    Vector<Color> accum;
    _project_to_canvas(p_asset, p_size, hits, accum);

    Ref<Image> image = Image::create_empty(p_size, p_size, false, Image::FORMAT_RGBA8);
    image->fill(Color(0.08f, 0.08f, 0.08f, 1.0f));

    const int *hits_ptr = hits.ptr();
    const Color *accum_ptr = accum.ptr();

    for (int i = 0; i < hits.size(); i++) {
        int count = hits_ptr[i];
        if (count == 0) {
            continue;
        }
        Color c = accum_ptr[i] / float(count);
        image->set_pixel(i % p_size, i / p_size, c);
    }

    return ImageTexture::create_from_image(image);
}

Ref<Texture2D> GaussianThumbnailGenerator::_generate_density_thumbnail(const Ref<GaussianSplatAsset> &p_asset, int p_size) const {
    Vector<int> hits;
    Vector<Color> accum;
    _project_to_canvas(p_asset, p_size, hits, accum);

    int max_hit = 1;
    const int *hits_ptr = hits.ptr();
    for (int i = 0; i < hits.size(); i++) {
        max_hit = MAX(max_hit, hits_ptr[i]);
    }

    Ref<Image> image = Image::create_empty(p_size, p_size, false, Image::FORMAT_RGBA8);

    for (int i = 0; i < hits.size(); i++) {
        int hit = hits_ptr[i];
        if (hit == 0) {
            continue;
        }
        float intensity = Math::sqrt(float(hit) / float(max_hit));
        Color c(intensity, intensity, intensity, 1.0f);
        image->set_pixel(i % p_size, i / p_size, c);
    }

    return ImageTexture::create_from_image(image);
}

Ref<Texture2D> GaussianThumbnailGenerator::_generate_normals_thumbnail(const Ref<GaussianSplatAsset> &p_asset, int p_size) const {
    Vector<int> hits;
    Vector<Color> accum;
    Dictionary projection = _project_to_canvas(p_asset, p_size, hits, accum);

    Ref<Image> image = Image::create_empty(p_size, p_size, false, Image::FORMAT_RGBA8);
    image->fill(Color(0.3f, 0.3f, 0.4f, 1.0f));

    PackedFloat32Array positions = p_asset->get_positions();
    PackedFloat32Array rotations = p_asset->get_rotations();
    PackedFloat32Array normals_array = p_asset->get_normals();
    const int splat_count = p_asset->get_splat_count();
    const int rot_size = rotations.size();
    const bool has_normals = normals_array.size() >= splat_count * 3;

    Vector3 min_pos = projection.get(StringName("min_pos"), Vector3());
    Vector3 extent = projection.get(StringName("extent"), Vector3(1, 1, 1));
    extent.x = MAX(extent.x, 0.0001f);
    extent.y = MAX(extent.y, 0.0001f);
    extent.z = MAX(extent.z, 0.0001f);

    Vector<Vector3> normal_accum;
    normal_accum.resize(hits.size());
    Vector<int> normal_counts;
    normal_counts.resize(hits.size());

    Vector3 *normal_ptr = normal_accum.ptrw();
    int *count_ptr = normal_counts.ptrw();
    for (int i = 0; i < hits.size(); i++) {
        normal_ptr[i] = Vector3();
        count_ptr[i] = 0;
    }

    for (int s = 0; s < splat_count; s++) {
        int pos_base = s * 3;
        int rot_base = s * 4;
        if (pos_base + 2 >= positions.size() || rot_base + 3 >= rot_size) {
            continue;
        }

        Vector3 pos(positions[pos_base], positions[pos_base + 1], positions[pos_base + 2]);
        float u = (pos.x - min_pos.x) / extent.x;
        float v = (pos.z - min_pos.z) / extent.z;

        int px = CLAMP((int)Math::round(u * (p_size - 1)), 0, p_size - 1);
        int py = CLAMP((int)Math::round((1.0f - v) * (p_size - 1)), 0, p_size - 1);
        int idx = py * p_size + px;

        Vector3 normal;
        if (has_normals) {
            normal = Vector3(normals_array[s * 3 + 0], normals_array[s * 3 + 1], normals_array[s * 3 + 2]);
        } else if (rot_base + 3 < rot_size) {
            Quaternion q(rotations[rot_base + 1], rotations[rot_base + 2], rotations[rot_base + 3], rotations[rot_base + 0]);
            normal = q.xform(Vector3(0, 0, 1));
        } else {
            normal = Vector3(0, 0, 1);
        }
        normal.normalize();

        normal_ptr[idx] += normal;
        count_ptr[idx] += 1;
    }

    for (int i = 0; i < hits.size(); i++) {
        if (count_ptr[i] == 0) {
            continue;
        }
        Vector3 n = normal_ptr[i] / float(count_ptr[i]);
        if (n.length() == 0) {
            continue;
        }
        n.normalize();
        Color c = Color(n.x * 0.5f + 0.5f, n.y * 0.5f + 0.5f, n.z * 0.5f + 0.5f, 1.0f);
        image->set_pixel(i % p_size, i / p_size, c);
    }

    return ImageTexture::create_from_image(image);
}

Ref<Texture2D> GaussianThumbnailGenerator::_generate_heatmap_thumbnail(const Ref<GaussianSplatAsset> &p_asset, int p_size) const {
    Vector<int> hits;
    Vector<Color> accum;
    _project_to_canvas(p_asset, p_size, hits, accum);

    int max_hit = 1;
    const int *hits_ptr = hits.ptr();
    for (int i = 0; i < hits.size(); i++) {
        max_hit = MAX(max_hit, hits_ptr[i]);
    }

    Ref<Image> image = Image::create_empty(p_size, p_size, false, Image::FORMAT_RGBA8);

    for (int i = 0; i < hits.size(); i++) {
        int hit = hits_ptr[i];
        if (hit == 0) {
            continue;
        }
        float t = float(hit) / float(max_hit);
        Color c;
        c.r = CLAMP(2.0f * t, 0.0f, 1.0f);
        c.g = CLAMP(2.0f * (1.0f - Math::abs(t - 0.5f)), 0.0f, 1.0f);
        c.b = CLAMP(2.0f * (1.0f - t), 0.0f, 1.0f);
        c.a = 1.0f;
        image->set_pixel(i % p_size, i / p_size, c);
    }

    return ImageTexture::create_from_image(image);
}

Ref<Texture2D> GaussianThumbnailGenerator::generate_thumbnail(const Ref<GaussianSplatAsset> &p_asset, int p_size,
        ThumbnailStyle p_style) const {
    ERR_FAIL_COND_V(p_size <= 0, Ref<Texture2D>());
    ERR_FAIL_COND_V(p_asset.is_null(), Ref<Texture2D>());
    ERR_FAIL_COND_V(p_asset->get_splat_count() == 0, Ref<Texture2D>());

    const String cache_key = _build_cache_key(p_asset, p_size, p_style);
    const uint64_t fingerprint = _compute_asset_fingerprint(p_asset);

    // Check in-memory cache first.
    if (!cache_key.is_empty()) {
        const Ref<Texture2D> *cached = thumbnail_cache.getptr(cache_key);
        if (cached && cached->is_valid()) {
            cache_hit_count++;
            _touch_cache_key(cache_key);
            return *cached;
        }
    }

    // Check disk cache before generating.
    if (fingerprint != 0) {
        Ref<Texture2D> disk_cached = _load_from_disk_cache(fingerprint, p_size, p_style);
        if (disk_cached.is_valid()) {
            disk_cache_hit_count++;
            if (!cache_key.is_empty()) {
                thumbnail_cache[cache_key] = disk_cached;
                _touch_cache_key(cache_key);
                _prune_cache_if_needed();
            }
            return disk_cached;
        }
        disk_cache_miss_count++;
    }

    cache_miss_count++;
    Ref<Texture2D> generated;

    switch (p_style) {
        case THUMBNAIL_STYLE_COLOR:
            generated = _generate_color_thumbnail(p_asset, p_size);
            break;
        case THUMBNAIL_STYLE_DENSITY:
            generated = _generate_density_thumbnail(p_asset, p_size);
            break;
        case THUMBNAIL_STYLE_NORMALS:
            generated = _generate_normals_thumbnail(p_asset, p_size);
            break;
        case THUMBNAIL_STYLE_HEATMAP:
            generated = _generate_heatmap_thumbnail(p_asset, p_size);
            break;
        default:
            generated = _generate_color_thumbnail(p_asset, p_size);
            break;
    }

    if (generated.is_valid()) {
        if (!cache_key.is_empty()) {
            thumbnail_cache[cache_key] = generated;
            _touch_cache_key(cache_key);
            _prune_cache_if_needed();
        }

        // Persist to disk cache for future editor sessions.
        if (fingerprint != 0) {
            _save_to_disk_cache(fingerprint, p_size, p_style, generated);
        }
    }

    return generated;
}

Dictionary GaussianThumbnailGenerator::compute_memory_statistics(uint32_t p_splat_count, uint32_t p_compression_flags,
        bool p_pack_opacity) const {
    const double float_bytes = 4.0;
    double positions_bytes = double(p_splat_count) * 3.0 * ((p_compression_flags & GaussianSplatAsset::COMPRESSION_POSITIONS) ? 2.0 : float_bytes);
    double colors_bytes = double(p_splat_count) * ((p_compression_flags & GaussianSplatAsset::COMPRESSION_COLORS) ? 4.0 : 4.0 * float_bytes);
    double scales_bytes = double(p_splat_count) * 3.0 * ((p_compression_flags & GaussianSplatAsset::COMPRESSION_SCALES) ? 2.0 : float_bytes);
    double rotations_bytes = double(p_splat_count) * 4.0 * ((p_compression_flags & GaussianSplatAsset::COMPRESSION_ROTATIONS) ? 2.0 : float_bytes);

    if (p_pack_opacity) {
        colors_bytes -= double(p_splat_count) * (float_bytes - 1.0);
    }

    double total_bytes = positions_bytes + colors_bytes + scales_bytes + rotations_bytes;
    double total_mb = total_bytes / (1024.0 * 1024.0);

    Dictionary result;
    result[StringName("positions_mb")] = positions_bytes / (1024.0 * 1024.0);
    result[StringName("colors_mb")] = colors_bytes / (1024.0 * 1024.0);
    result[StringName("scales_mb")] = scales_bytes / (1024.0 * 1024.0);
    result[StringName("rotations_mb")] = rotations_bytes / (1024.0 * 1024.0);
    result[StringName("total_mb")] = total_mb;
    return result;
}

int GaussianThumbnailGenerator::get_cache_entry_count() const {
    return thumbnail_cache.size();
}

Dictionary GaussianThumbnailGenerator::get_cache_statistics() const {
    Dictionary stats;
    const int entries = thumbnail_cache.size();
    const uint64_t total_requests = cache_hit_count + cache_miss_count + disk_cache_hit_count;
    const double hit_ratio = total_requests > 0 ? double(cache_hit_count + disk_cache_hit_count) / double(total_requests) : 0.0;

    stats[StringName("entries")] = entries;
    stats[StringName("hits")] = int64_t(cache_hit_count);
    stats[StringName("misses")] = int64_t(cache_miss_count);
    stats[StringName("hit_ratio")] = hit_ratio;
    stats[StringName("disk_hits")] = int64_t(disk_cache_hit_count);
    stats[StringName("disk_misses")] = int64_t(disk_cache_miss_count);
    return stats;
}

void GaussianThumbnailGenerator::clear_cache() {
    thumbnail_cache.clear();
    thumbnail_cache_order.clear();
    cache_hit_count = 0;
    cache_miss_count = 0;
    disk_cache_hit_count = 0;
    disk_cache_miss_count = 0;
}

void GaussianThumbnailGenerator::clear_disk_cache() {
    Ref<DirAccess> dir = DirAccess::open(DISK_CACHE_DIR);
    if (dir.is_null()) {
        return;
    }

    dir->list_dir_begin();
    String file_name = dir->get_next();
    while (!file_name.is_empty()) {
        if (!dir->current_is_dir() && file_name.ends_with(".png")) {
            dir->remove(file_name);
        }
        file_name = dir->get_next();
    }
    dir->list_dir_end();
}

String GaussianThumbnailGenerator::style_to_display_name(ThumbnailStyle p_style) {
    switch (p_style) {
        case THUMBNAIL_STYLE_COLOR:
            return TTR("Color");
        case THUMBNAIL_STYLE_DENSITY:
            return TTR("Density");
        case THUMBNAIL_STYLE_NORMALS:
            return TTR("Normals");
        case THUMBNAIL_STYLE_HEATMAP:
            return TTR("Heatmap");
        default:
            break;
    }
    return TTR("Color");
}

GaussianThumbnailGenerator::ThumbnailStyle GaussianThumbnailGenerator::style_from_int(int p_value) {
    switch (p_value) {
        case 0:
            return THUMBNAIL_STYLE_COLOR;
        case 1:
            return THUMBNAIL_STYLE_DENSITY;
        case 2:
            return THUMBNAIL_STYLE_NORMALS;
        case 3:
            return THUMBNAIL_STYLE_HEATMAP;
        default:
            break;
    }
    return THUMBNAIL_STYLE_COLOR;
}

void GaussianThumbnailGenerator::_bind_methods() {
    ClassDB::bind_method(D_METHOD("get_cache_entry_count"), &GaussianThumbnailGenerator::get_cache_entry_count);
    ClassDB::bind_method(D_METHOD("get_cache_statistics"), &GaussianThumbnailGenerator::get_cache_statistics);
    ClassDB::bind_method(D_METHOD("clear_cache"), &GaussianThumbnailGenerator::clear_cache);
    ClassDB::bind_method(D_METHOD("clear_disk_cache"), &GaussianThumbnailGenerator::clear_disk_cache);

    BIND_ENUM_CONSTANT(THUMBNAIL_STYLE_COLOR);
    BIND_ENUM_CONSTANT(THUMBNAIL_STYLE_DENSITY);
    BIND_ENUM_CONSTANT(THUMBNAIL_STYLE_NORMALS);
    BIND_ENUM_CONSTANT(THUMBNAIL_STYLE_HEATMAP);
}

#endif // TOOLS_ENABLED
