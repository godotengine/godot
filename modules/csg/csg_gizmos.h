#ifndef CSG_GIZMOS_H
#define CSG_GIZMOS_H

#include "csg_shape.h"
#include "editor/editor_plugin.h"
#include "editor/spatial_editor_gizmos.h"

class CSGShapeSpatialGizmo : public EditorSpatialGizmo {

	GDCLASS(CSGShapeSpatialGizmo, EditorSpatialGizmo);

	CSGShape *cs;

public:
	virtual String get_handle_name(int p_idx) const;
	virtual Variant get_handle_value(int p_idx) const;
	virtual void set_handle(int p_idx, Camera *p_camera, const Point2 &p_point);
	virtual void commit_handle(int p_idx, const Variant &p_restore, bool p_cancel = false);
	void redraw();
	CSGShapeSpatialGizmo(CSGShape *p_cs = NULL);
};

class EditorPluginCSG : public EditorPlugin {
	GDCLASS(EditorPluginCSG, EditorPlugin)
public:
	virtual Ref<SpatialEditorGizmo> create_spatial_gizmo(Spatial *p_spatial);
	EditorPluginCSG(EditorNode *p_editor);
};

#endif // CSG_GIZMOS_H
