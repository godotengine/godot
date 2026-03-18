#include "gaussian_splat_container.h"

#include "core/error/error_macros.h"
#include "core/math/math_funcs.h"
#include "core/templates/hash_set.h"
#include "gaussian_splat_node_3d.h"
#include "gaussian_splat_world_3d.h"
#include "../core/gaussian_splat_merge_utils.h"
#include "../core/gaussian_splat_world.h"
#include "../logger/gs_logger.h"

void GaussianSplatContainer::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_merge_on_ready", "enabled"), &GaussianSplatContainer::set_merge_on_ready);
    ClassDB::bind_method(D_METHOD("is_merge_on_ready"), &GaussianSplatContainer::is_merge_on_ready);
    ClassDB::bind_method(D_METHOD("set_chunk_size", "size"), &GaussianSplatContainer::set_chunk_size);
    ClassDB::bind_method(D_METHOD("get_chunk_size"), &GaussianSplatContainer::get_chunk_size);
    ClassDB::bind_method(D_METHOD("set_hide_children_after_merge", "hide"), &GaussianSplatContainer::set_hide_children_after_merge);
    ClassDB::bind_method(D_METHOD("get_hide_children_after_merge"), &GaussianSplatContainer::get_hide_children_after_merge);
    ClassDB::bind_method(D_METHOD("set_apply_to_target_on_merge", "enabled"), &GaussianSplatContainer::set_apply_to_target_on_merge);
    ClassDB::bind_method(D_METHOD("is_apply_to_target_on_merge"), &GaussianSplatContainer::is_apply_to_target_on_merge);
    ClassDB::bind_method(D_METHOD("set_target_node_path", "path"), &GaussianSplatContainer::set_target_node_path);
    ClassDB::bind_method(D_METHOD("get_target_node_path"), &GaussianSplatContainer::get_target_node_path);
    ClassDB::bind_method(D_METHOD("merge_children"), &GaussianSplatContainer::merge_children);
    ClassDB::bind_method(D_METHOD("clear_merged_data"), &GaussianSplatContainer::clear_merged_data);
    ClassDB::bind_method(D_METHOD("apply_to_renderer", "renderer"), &GaussianSplatContainer::apply_to_renderer);
    ClassDB::bind_method(D_METHOD("apply_to_node", "node"), &GaussianSplatContainer::apply_to_node);
    ClassDB::bind_method(D_METHOD("merge_children_to_node", "node"), &GaussianSplatContainer::merge_children_to_node);
    ClassDB::bind_method(D_METHOD("export_world_resource"), &GaussianSplatContainer::export_world_resource);
    ClassDB::bind_method(D_METHOD("get_merged_data"), &GaussianSplatContainer::get_merged_data);
    ClassDB::bind_method(D_METHOD("get_chunk_count"), &GaussianSplatContainer::get_chunk_count);
    ClassDB::bind_method(D_METHOD("get_chunk_sizes"), &GaussianSplatContainer::get_chunk_sizes);
    ClassDB::bind_method(D_METHOD("get_chunk_aabbs"), &GaussianSplatContainer::get_chunk_aabbs);

    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "merge_on_ready"), "set_merge_on_ready", "is_merge_on_ready");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "chunk_size", PROPERTY_HINT_RANGE, "0.1,1024,0.1"), "set_chunk_size", "get_chunk_size");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "hide_children_after_merge"), "set_hide_children_after_merge", "get_hide_children_after_merge");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "apply_to_target_on_merge"), "set_apply_to_target_on_merge", "is_apply_to_target_on_merge");
    ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target_node_path"), "set_target_node_path", "get_target_node_path");
}

void GaussianSplatContainer::_notification(int p_what) {
    switch (p_what) {
        case NOTIFICATION_READY: {
            if (merge_on_ready) {
                merge_children();
            }
        } break;
    }
}

void GaussianSplatContainer::set_merge_on_ready(bool p_enabled) {
    merge_on_ready = p_enabled;
}

void GaussianSplatContainer::set_chunk_size(float p_size) {
    chunk_size = MAX(0.1f, p_size);
}

void GaussianSplatContainer::set_hide_children_after_merge(bool p_hide) {
    hide_children_after_merge = p_hide;
}

void GaussianSplatContainer::set_apply_to_target_on_merge(bool p_enabled) {
    apply_to_target_on_merge = p_enabled;
}

void GaussianSplatContainer::set_target_node_path(const NodePath &p_path) {
    target_node_path = p_path;
}

void GaussianSplatContainer::merge_children() {
    _merge_children_internal();
    if (hide_children_after_merge) {
        _apply_child_visibility(false);
    }
    if (apply_to_target_on_merge && target_node_path != NodePath()) {
        Node *target = get_node_or_null(target_node_path);
        if (!target) {
            GS_LOG_WARN_DEFAULT("GaussianSplatContainer: target_node_path not found; skipping apply.");
        } else {
            apply_to_node(target);
        }
    }
}

Error GaussianSplatContainer::apply_to_renderer(const Ref<GaussianSplatRenderer> &p_renderer) {
    ERR_FAIL_COND_V(p_renderer.is_null(), ERR_INVALID_PARAMETER);
    if (merged_data.is_null() || merged_data->get_count() == 0) {
        GS_LOG_WARN_DEFAULT("GaussianSplatContainer: no merged data available to apply.");
        return ERR_UNAVAILABLE;
    }

    Error err = p_renderer->set_gaussian_data(merged_data);
    if (err != OK) {
        GS_LOG_WARN_DEFAULT(vformat("GaussianSplatContainer: failed to apply merged data (err=%d).", err));
        return err;
    }
    p_renderer->set_static_chunks(merged_chunks);
    return OK;
}

Error GaussianSplatContainer::apply_to_node(Node *p_node) {
    ERR_FAIL_NULL_V(p_node, ERR_INVALID_PARAMETER);

    if (GaussianSplatWorld3D *world_node = Object::cast_to<GaussianSplatWorld3D>(p_node)) {
        Ref<GaussianSplatWorld> world_resource = export_world_resource();
        if (world_resource.is_null()) {
            return ERR_UNAVAILABLE;
        }
        if (!world_node->get_global_transform().is_equal_approx(Transform3D())) {
            GS_LOG_WARN_DEFAULT("GaussianSplatContainer: target world node transform is not identity; merged data is in world space.");
        }
        world_node->set_world(world_resource);
        world_node->apply_world();
        return OK;
    }

    GaussianSplatNode3D *splat_node = Object::cast_to<GaussianSplatNode3D>(p_node);
    if (!splat_node) {
        GS_LOG_WARN_DEFAULT("GaussianSplatContainer: target node is not a GaussianSplatNode3D or GaussianSplatWorld3D.");
        return ERR_INVALID_PARAMETER;
    }

    if (!splat_node->get_global_transform().is_equal_approx(Transform3D())) {
        GS_LOG_WARN_DEFAULT("GaussianSplatContainer: target node transform is not identity; merged data is in world space.");
    }

    Ref<GaussianSplatAsset> merged_asset;
    merged_asset.instantiate();
    Error err = merged_asset->populate_from_gaussian_data(merged_data);
    if (err != OK) {
        GS_LOG_WARN_DEFAULT(vformat("GaussianSplatContainer: failed to build asset from merged data (err=%d).", err));
        return err;
    }

    splat_node->set_splat_asset(merged_asset);
    return OK;
}

Error GaussianSplatContainer::merge_children_to_node(Node *p_node) {
    merge_children();
    return apply_to_node(p_node);
}

Ref<GaussianSplatWorld> GaussianSplatContainer::export_world_resource() const {
    if (merged_data.is_null() || merged_data->get_count() == 0) {
        GS_LOG_WARN_DEFAULT("GaussianSplatContainer: no merged data available to export.");
        return Ref<GaussianSplatWorld>();
    }

    Ref<GaussianSplatWorld> world_resource;
    world_resource.instantiate();
    world_resource->set_gaussian_data(merged_data);
    world_resource->set_bounds(merged_bounds);
    world_resource->set_static_chunks(merged_chunks);
    return world_resource;
}

void GaussianSplatContainer::clear_merged_data() {
    merged_chunks.clear();
    if (merged_data.is_valid()) {
        merged_data->resize(0);
    }
    merged_bounds = AABB();
    if (hide_children_after_merge) {
        _apply_child_visibility(true);
    }
}

void GaussianSplatContainer::_apply_child_visibility(bool p_visible) {
    const int child_count = get_child_count();
    for (int i = 0; i < child_count; i++) {
        Node *child = get_child(i);
        if (GaussianSplatNode3D *splat_node = Object::cast_to<GaussianSplatNode3D>(child)) {
            splat_node->set_visible(p_visible);
        }
    }
}

void GaussianSplatContainer::_merge_children_internal() {
    Vector<GaussianSplatMergeSource> sources;

    const int child_count = get_child_count();
    for (int i = 0; i < child_count; i++) {
        Node *child = get_child(i);
        GaussianSplatNode3D *splat_node = Object::cast_to<GaussianSplatNode3D>(child);
        if (!splat_node) {
            continue;
        }

        Ref<GaussianSplatAsset> asset = splat_node->get_splat_asset();
        if (asset.is_null()) {
            continue;
        }

        uint32_t splat_count = asset->get_splat_count();
        if (splat_count == 0) {
            continue;
        }

        GaussianSplatMergeSource source;
        source.asset = asset;
        source.transform = splat_node->get_global_transform();
        Dictionary import_metadata = asset->get_import_metadata();
        const bool import_is_2d = import_metadata.has(StringName("gaussian_2d_mode")) && (bool)import_metadata[StringName("gaussian_2d_mode")];
        source.is_2d = import_is_2d;
        sources.push_back(source);
    }

    if (sources.is_empty()) {
        clear_merged_data();
        return;
    }

    GaussianSplatMergeResult result;
    if (!gaussian_splat_merge_sources(sources, chunk_size, result)) {
        clear_merged_data();
        return;
    }

    merged_data = result.data;
    merged_bounds = result.bounds;
    merged_chunks = result.chunks;
}

PackedInt32Array GaussianSplatContainer::get_chunk_sizes() const {
    PackedInt32Array result;
    result.resize(merged_chunks.size());
    for (int i = 0; i < merged_chunks.size(); i++) {
        result.set(i, merged_chunks[i].indices.size());
    }
    return result;
}

Array GaussianSplatContainer::get_chunk_aabbs() const {
    Array result;
    result.resize(merged_chunks.size());
    for (int i = 0; i < merged_chunks.size(); i++) {
        result[i] = merged_chunks[i].bounds;
    }
    return result;
}
