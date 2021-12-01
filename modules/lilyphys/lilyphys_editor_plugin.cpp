//
// Created by amara on 26/11/2021.
//

#include "lilyphys_editor_plugin.h"

#include "nodes/l_collision_shape_gizmo.h"

LilyphysEditorPlugin::LilyphysEditorPlugin(EditorNode *p_editor) {
    Ref<LCollisionShapeGizmoPlugin> gizmo_plugin = Ref<LCollisionShapeGizmoPlugin>(memnew(LCollisionShapeGizmoPlugin));
    SpatialEditor::get_singleton()->add_gizmo_plugin(gizmo_plugin);
}