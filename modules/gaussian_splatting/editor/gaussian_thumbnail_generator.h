#ifndef GAUSSIAN_THUMBNAIL_GENERATOR_H
#define GAUSSIAN_THUMBNAIL_GENERATOR_H

#ifdef TOOLS_ENABLED

#include "core/math/color.h"
#include "core/math/vector3.h"
#include "core/object/ref_counted.h"
#include "core/templates/hash_map.h"
#include "core/templates/vector.h"
#include "core/variant/dictionary.h"
#include "scene/resources/texture.h"

#include "../core/gaussian_splat_asset.h"

class GaussianThumbnailGenerator : public RefCounted {
    GDCLASS(GaussianThumbnailGenerator, RefCounted);

public:
    enum ThumbnailStyle {
        THUMBNAIL_STYLE_COLOR = 0,
        THUMBNAIL_STYLE_DENSITY = 1,
        THUMBNAIL_STYLE_NORMALS = 2,
        THUMBNAIL_STYLE_HEATMAP = 3,
    };

private:
    static constexpr int MAX_CACHE_ENTRIES = 64;
    static const char *DISK_CACHE_DIR;

    mutable HashMap<String, Ref<Texture2D>> thumbnail_cache;
    mutable Vector<String> thumbnail_cache_order;
    mutable uint64_t cache_hit_count = 0;
    mutable uint64_t cache_miss_count = 0;
    mutable uint64_t disk_cache_hit_count = 0;
    mutable uint64_t disk_cache_miss_count = 0;

    String _build_cache_key(const Ref<GaussianSplatAsset> &p_asset, int p_size, ThumbnailStyle p_style) const;
    uint64_t _compute_asset_fingerprint(const Ref<GaussianSplatAsset> &p_asset) const;
    void _touch_cache_key(const String &p_key) const;
    void _prune_cache_if_needed() const;

    // Disk cache helpers.
    String _disk_cache_path_for_key(uint64_t p_fingerprint, int p_size, ThumbnailStyle p_style) const;
    Ref<Texture2D> _load_from_disk_cache(uint64_t p_fingerprint, int p_size, ThumbnailStyle p_style) const;
    void _save_to_disk_cache(uint64_t p_fingerprint, int p_size, ThumbnailStyle p_style, const Ref<Texture2D> &p_texture) const;
    bool _ensure_disk_cache_dir() const;

    Ref<Texture2D> _generate_color_thumbnail(const Ref<GaussianSplatAsset> &p_asset, int p_size) const;
    Ref<Texture2D> _generate_density_thumbnail(const Ref<GaussianSplatAsset> &p_asset, int p_size) const;
    Ref<Texture2D> _generate_normals_thumbnail(const Ref<GaussianSplatAsset> &p_asset, int p_size) const;
    Ref<Texture2D> _generate_heatmap_thumbnail(const Ref<GaussianSplatAsset> &p_asset, int p_size) const;

    Dictionary _project_to_canvas(const Ref<GaussianSplatAsset> &p_asset, int p_size, Vector<int> &r_hits,
            Vector<Color> &r_accum) const;

public:
    Ref<Texture2D> generate_thumbnail(const Ref<GaussianSplatAsset> &p_asset, int p_size, ThumbnailStyle p_style) const;
    Dictionary compute_memory_statistics(uint32_t p_splat_count, uint32_t p_compression_flags, bool p_pack_opacity) const;
    int get_cache_entry_count() const;
    Dictionary get_cache_statistics() const;
    void clear_cache();
    void clear_disk_cache();

    static String style_to_display_name(ThumbnailStyle p_style);
    static ThumbnailStyle style_from_int(int p_value);

protected:
    static void _bind_methods();
};

VARIANT_ENUM_CAST(GaussianThumbnailGenerator::ThumbnailStyle);

#endif // TOOLS_ENABLED

#endif // GAUSSIAN_THUMBNAIL_GENERATOR_H
