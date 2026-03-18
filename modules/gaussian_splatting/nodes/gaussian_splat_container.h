#ifndef GAUSSIAN_SPLAT_CONTAINER_H
#define GAUSSIAN_SPLAT_CONTAINER_H

#include "scene/3d/node_3d.h"
#include "core/error/error_list.h"
#include "core/variant/typed_array.h"
#include "core/string/node_path.h"
#include "../core/gaussian_data.h"
#include "../renderer/gaussian_splat_renderer.h"

class GaussianSplatNode3D;
class GaussianSplatWorld;
class GaussianSplatWorld3D;

class GaussianSplatContainer : public Node3D {
    GDCLASS(GaussianSplatContainer, Node3D);

private:
    bool merge_on_ready = false;
    float chunk_size = 8.0f;
    bool hide_children_after_merge = false;
    bool apply_to_target_on_merge = false;
    NodePath target_node_path;

    Ref<GaussianData> merged_data;
    Vector<GaussianSplatRenderer::StaticChunk> merged_chunks;
    AABB merged_bounds;

    void _merge_children_internal();
    void _apply_child_visibility(bool p_visible);

protected:
    static void _bind_methods();
    void _notification(int p_what);

public:
    GaussianSplatContainer() = default;

    void set_merge_on_ready(bool p_enabled);
    bool is_merge_on_ready() const { return merge_on_ready; }

    void set_chunk_size(float p_size);
    float get_chunk_size() const { return chunk_size; }

    void set_hide_children_after_merge(bool p_hide);
    bool get_hide_children_after_merge() const { return hide_children_after_merge; }
    void set_apply_to_target_on_merge(bool p_enabled);
    bool is_apply_to_target_on_merge() const { return apply_to_target_on_merge; }
    void set_target_node_path(const NodePath &p_path);
    NodePath get_target_node_path() const { return target_node_path; }

    void merge_children();
    void clear_merged_data();
    Error apply_to_renderer(const Ref<GaussianSplatRenderer> &p_renderer);
    Error apply_to_node(Node *p_node);
    Error merge_children_to_node(Node *p_node);
    Ref<GaussianSplatWorld> export_world_resource() const;

    Ref<GaussianData> get_merged_data() const { return merged_data; }
    int get_chunk_count() const { return merged_chunks.size(); }
    PackedInt32Array get_chunk_sizes() const;
    Array get_chunk_aabbs() const;

    const Vector<GaussianSplatRenderer::StaticChunk> &get_static_chunks() const { return merged_chunks; }
    AABB get_merged_bounds() const { return merged_bounds; }
};

#endif // GAUSSIAN_SPLAT_CONTAINER_H
