#ifndef GAUSSIAN_SPLAT_WORLD_3D_H
#define GAUSSIAN_SPLAT_WORLD_3D_H

#include "scene/3d/node_3d.h"
#include "core/math/aabb.h"

#include "../core/gaussian_splat_world.h"
#include "../renderer/gaussian_splat_renderer.h"

class GaussianSplatWorld3D : public Node3D {
    GDCLASS(GaussianSplatWorld3D, Node3D);

private:
    Ref<GaussianSplatWorld> world;
    Ref<GaussianSplatRenderer> renderer;

    RID render_instance;
    RID gaussian_base;

    bool auto_apply_on_ready = true;
    bool cast_shadow = false;
    bool bounds_dirty = true;
    bool warned_non_identity_transform = false;

    AABB local_aabb;
    AABB world_aabb;

    // Renderer parity settings (matches the core controls on GaussianSplatNode3D).
    bool lod_enabled = true;
    float lod_bias = 1.0f;
    float max_render_distance = 1000.0f;
    int max_splat_count = 1000000;
    bool use_frustum_culling = true;
    bool async_upload_enabled = true;
    float opacity = 1.0f;

    void _ensure_renderer();
    Dictionary _build_desired_renderer_overrides() const;
    void _resubmit_world_submission_if_registered();
    void _apply_world_internal();
    void _register_shared_renderer();
    void _unregister_shared_renderer();
    void _update_bounds();
    void _update_render_instance();
    void _ensure_gaussian_base();
    void _release_gaussian_base();
    void _sync_gaussian_storage();
    void _set_instance_base(const RID &p_base);

    void _notification_process();
protected:
    static void _bind_methods();
    void _notification(int p_what);

public:
    void set_world(const Ref<GaussianSplatWorld> &p_world);
    Ref<GaussianSplatWorld> get_world() const { return world; }

    void set_auto_apply_on_ready(bool p_enabled);
    bool is_auto_apply_on_ready() const { return auto_apply_on_ready; }

    void set_cast_shadow(bool p_enabled);
    bool is_cast_shadow() const { return cast_shadow; }

    void set_lod_enabled(bool p_enabled);
    bool is_lod_enabled() const { return lod_enabled; }

    void set_lod_bias(float p_bias);
    float get_lod_bias() const { return lod_bias; }

    void set_max_render_distance(float p_distance);
    float get_max_render_distance() const { return max_render_distance; }

    void set_max_splat_count(int p_count);
    int get_max_splat_count() const { return max_splat_count; }

    void set_use_frustum_culling(bool p_enabled);
    bool is_frustum_culling_enabled() const { return use_frustum_culling; }

    void set_async_upload_enabled(bool p_enabled);
    bool is_async_upload_enabled() const { return async_upload_enabled; }

    void set_opacity(float p_opacity);
    float get_opacity() const { return opacity; }

    void apply_world();
    void clear_world();

    Ref<GaussianSplatRenderer> get_renderer() const { return renderer; }
};

#endif // GAUSSIAN_SPLAT_WORLD_3D_H
