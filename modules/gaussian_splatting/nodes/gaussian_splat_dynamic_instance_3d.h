/**
 * @file gaussian_splat_dynamic_instance_3d.h
 * @brief Scene node that registers instance-pipeline splats without a full renderer.
 */

#ifndef GAUSSIAN_SPLAT_DYNAMIC_INSTANCE_3D_H
#define GAUSSIAN_SPLAT_DYNAMIC_INSTANCE_3D_H

#include "scene/3d/node_3d.h"
#include "../core/gaussian_data.h"
#include "../core/gaussian_splat_asset.h"

/**
 * @class GaussianSplatDynamicInstance3D
 * @brief Lightweight node that feeds dynamic splats into the streaming renderer.
 *
 * This node does not render by itself. It registers an instance in the scene
 * director so the instance pipeline can render it alongside other splat nodes.
 */
class GaussianSplatDynamicInstance3D : public Node3D {
    GDCLASS(GaussianSplatDynamicInstance3D, Node3D);

private:
    String ply_file_path;
    Ref<GaussianSplatAsset> splat_asset;
    Ref<GaussianSplatAsset> runtime_asset;
    Ref<GaussianData> gaussian_data;
    bool auto_load = true;
    bool registered = false;

    void _register_instance();
    void _unregister_instance();
    bool _can_register_instance() const;
    void _update_transform();
    uint32_t _get_instance_flags() const;
    bool _register_instance_registry();
    void _unregister_instance_registry();
    void _update_instance_transform_registry();
    void _on_asset_changed();
    bool _load_from_file();
    bool _populate_from_asset();

protected:
    static void _bind_methods();
    void _notification(int p_what);

public:
    GaussianSplatDynamicInstance3D();
    ~GaussianSplatDynamicInstance3D() override;

    void set_ply_file_path(const String &p_path);
    String get_ply_file_path() const { return ply_file_path; }

    void set_splat_asset(const Ref<GaussianSplatAsset> &p_asset);
    Ref<GaussianSplatAsset> get_splat_asset() const { return splat_asset; }

    void set_gaussian_data(const Ref<GaussianData> &p_data);
    Ref<GaussianData> get_gaussian_data() const { return gaussian_data; }

    void set_auto_load(bool p_enabled);
    bool is_auto_load_enabled() const { return auto_load; }

    void reload_asset();
    void register_instance();
    void unregister_instance();
    bool is_registered() const { return registered; }
};

#endif // GAUSSIAN_SPLAT_DYNAMIC_INSTANCE_3D_H
