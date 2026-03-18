#include "gaussian_splat_world.h"

void GaussianSplatWorld::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_gaussian_data", "data"), &GaussianSplatWorld::set_gaussian_data);
    ClassDB::bind_method(D_METHOD("get_gaussian_data"), &GaussianSplatWorld::get_gaussian_data);
    ClassDB::bind_method(D_METHOD("set_bounds", "bounds"), &GaussianSplatWorld::set_bounds);
    ClassDB::bind_method(D_METHOD("get_bounds"), &GaussianSplatWorld::get_bounds);
    ClassDB::bind_method(D_METHOD("set_metadata", "metadata"), &GaussianSplatWorld::set_metadata);
    ClassDB::bind_method(D_METHOD("get_metadata"), &GaussianSplatWorld::get_metadata);
    ClassDB::bind_method(D_METHOD("get_chunk_count"), &GaussianSplatWorld::get_chunk_count);
    ClassDB::bind_method(D_METHOD("get_chunk_sizes"), &GaussianSplatWorld::get_chunk_sizes);
    ClassDB::bind_method(D_METHOD("get_chunk_aabbs"), &GaussianSplatWorld::get_chunk_aabbs);
    ClassDB::bind_method(D_METHOD("clear"), &GaussianSplatWorld::clear);

    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "gaussian_data", PROPERTY_HINT_RESOURCE_TYPE, "GaussianData"),
            "set_gaussian_data", "get_gaussian_data");
    ADD_PROPERTY(PropertyInfo(Variant::AABB, "bounds"), "set_bounds", "get_bounds");
    ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "metadata"), "set_metadata", "get_metadata");
}

bool GaussianSplatWorld::_get(const StringName &p_name, Variant &r_ret) const {
    if (p_name == StringName("stats/total_splats")) {
        r_ret = gaussian_data.is_valid() ? gaussian_data->get_count() : 0;
        return true;
    }
    if (p_name == StringName("stats/chunk_count")) {
        r_ret = static_chunks.size();
        return true;
    }
    if (p_name == StringName("stats/lod_levels")) {
        int lod_levels = 0;
        if (metadata.has(StringName("lod_levels"))) {
            lod_levels = int(metadata[StringName("lod_levels")]);
        } else if (metadata.has(StringName("lod_level_count"))) {
            lod_levels = int(metadata[StringName("lod_level_count")]);
        }
        r_ret = lod_levels;
        return true;
    }
    if (p_name == StringName("stats/memory_mb")) {
        double bytes = 0.0;
        if (gaussian_data.is_valid()) {
            bytes += gaussian_data->get_memory_usage();
            const uint32_t high_order = gaussian_data->get_sh_high_order_count();
            if (high_order > 0) {
                bytes += double(gaussian_data->get_count()) * double(high_order) * sizeof(Vector3);
            }
        }
        for (int i = 0; i < static_chunks.size(); i++) {
            bytes += double(static_chunks[i].indices.size()) * sizeof(uint32_t);
        }
        r_ret = bytes / (1024.0 * 1024.0);
        return true;
    }
    return false;
}

void GaussianSplatWorld::_get_property_list(List<PropertyInfo> *p_list) const {
    const uint32_t usage = PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY;
    p_list->push_back(PropertyInfo(Variant::INT, "stats/total_splats", PROPERTY_HINT_NONE, "", usage));
    p_list->push_back(PropertyInfo(Variant::INT, "stats/chunk_count", PROPERTY_HINT_NONE, "", usage));
    p_list->push_back(PropertyInfo(Variant::INT, "stats/lod_levels", PROPERTY_HINT_NONE, "", usage));
    p_list->push_back(PropertyInfo(Variant::FLOAT, "stats/memory_mb", PROPERTY_HINT_NONE, "", usage));
}

void GaussianSplatWorld::set_gaussian_data(const Ref<GaussianData> &p_data) {
    gaussian_data = p_data;
    notify_property_list_changed();
    emit_changed();
}

void GaussianSplatWorld::set_bounds(const AABB &p_bounds) {
    bounds = p_bounds;
    emit_changed();
}

void GaussianSplatWorld::set_metadata(const Dictionary &p_metadata) {
    metadata = p_metadata;
    notify_property_list_changed();
    emit_changed();
}

void GaussianSplatWorld::set_static_chunks(const Vector<GaussianSplatRenderer::StaticChunk> &p_chunks) {
    static_chunks = p_chunks;
    notify_property_list_changed();
    emit_changed();
}

int GaussianSplatWorld::get_chunk_count() const {
    return static_chunks.size();
}

PackedInt32Array GaussianSplatWorld::get_chunk_sizes() const {
    PackedInt32Array sizes;
    sizes.resize(static_chunks.size());
    for (int i = 0; i < static_chunks.size(); i++) {
        sizes.set(i, static_cast<int>(static_chunks[i].indices.size()));
    }
    return sizes;
}

Array GaussianSplatWorld::get_chunk_aabbs() const {
    Array aabbs;
    aabbs.resize(static_chunks.size());
    for (int i = 0; i < static_chunks.size(); i++) {
        aabbs.set(i, static_chunks[i].bounds);
    }
    return aabbs;
}

void GaussianSplatWorld::clear() {
    gaussian_data.unref();
    static_chunks.clear();
    bounds = AABB();
    metadata.clear();
}
