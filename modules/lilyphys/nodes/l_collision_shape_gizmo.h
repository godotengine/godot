//
// Created by amara on 25/11/2021.
//

#ifndef LILYPHYS_L_COLLISION_SHAPE_GIZMO_H
#define LILYPHYS_L_COLLISION_SHAPE_GIZMO_H

#include "editor/plugins/spatial_editor_plugin.h"

class LCollisionShapeGizmoPlugin : public EditorSpatialGizmoPlugin {
    GDCLASS(LCollisionShapeGizmoPlugin, EditorSpatialGizmoPlugin);

public:
    bool has_gizmo(Spatial *p_spatial);
    String get_name() const;
    int get_priority() const;
    void redraw(EditorSpatialGizmo *p_gizmo);

    String get_handle_name(const EditorSpatialGizmo *p_gizmo, int p_idx) const;
    Variant get_handle_value(EditorSpatialGizmo *p_gizmo, int p_idx) const;
    void set_handle(EditorSpatialGizmo *p_gizmo, int p_idx, Camera *p_camera, const Point2 &p_point);
    void commit_handle(EditorSpatialGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel = false);

    LCollisionShapeGizmoPlugin();
};


#endif //LILYPHYS_L_COLLISION_SHAPE_GIZMO_H
