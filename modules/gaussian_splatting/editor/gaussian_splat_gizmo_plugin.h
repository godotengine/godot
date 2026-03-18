#ifndef GAUSSIAN_SPLAT_GIZMO_PLUGIN_H
#define GAUSSIAN_SPLAT_GIZMO_PLUGIN_H

#ifdef TOOLS_ENABLED

#include "editor/scene/3d/node_3d_editor_plugin.h"
#include "scene/resources/material.h"
#include "scene/resources/texture.h"

#include "../nodes/gaussian_splat_node_3d.h"

class GaussianSplatGizmoPlugin : public EditorNode3DGizmoPlugin {
    GDCLASS(GaussianSplatGizmoPlugin, EditorNode3DGizmoPlugin);

private:
    // Gizmo materials
    Ref<StandardMaterial3D> bounds_material;
    Ref<StandardMaterial3D> bounds_material_selected;
    Ref<StandardMaterial3D> lod_sphere_materials[3];
    Ref<StandardMaterial3D> statistics_material;
    Ref<StandardMaterial3D> performance_material_good;
    Ref<StandardMaterial3D> performance_material_warning;
    Ref<StandardMaterial3D> performance_material_critical;
    Ref<StandardMaterial3D> preview_wire_material;
    Ref<StandardMaterial3D> preview_point_material;
    Ref<StandardMaterial3D> preview_heatmap_materials[3];

    // Icons

    void create_materials();

protected:
    static void _bind_methods() {}

public:
    GaussianSplatGizmoPlugin();

    bool has_gizmo(Node3D *p_spatial) override;
    String get_gizmo_name() const override;
    int get_priority() const override;

    void redraw(EditorNode3DGizmo *p_gizmo) override;
    int subgizmos_intersect_ray(const EditorNode3DGizmo *p_gizmo, Camera3D *p_camera, const Vector2 &p_point) const override;
    Vector<int> subgizmos_intersect_frustum(const EditorNode3DGizmo *p_gizmo, const Camera3D *p_camera, const Vector<Plane> &p_frustum) const override;
    Transform3D get_subgizmo_transform(const EditorNode3DGizmo *p_gizmo, int p_id) const override;
    void set_subgizmo_transform(const EditorNode3DGizmo *p_gizmo, int p_id, Transform3D p_transform) override;
    void commit_subgizmos(const EditorNode3DGizmo *p_gizmo, const Vector<int> &p_ids, const Vector<Transform3D> &p_restore, bool p_cancel) override;

    // Custom helpers
    void draw_bounds(EditorNode3DGizmo *p_gizmo, GaussianSplatNode3D *p_node);
    void draw_lod_radius(EditorNode3DGizmo *p_gizmo, GaussianSplatNode3D *p_node);
    void draw_statistics(EditorNode3DGizmo *p_gizmo, GaussianSplatNode3D *p_node);
    void draw_splat_preview(EditorNode3DGizmo *p_gizmo, GaussianSplatNode3D *p_node);
};

#endif // TOOLS_ENABLED

#endif // GAUSSIAN_SPLAT_GIZMO_PLUGIN_H
