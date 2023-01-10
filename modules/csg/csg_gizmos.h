/**************************************************************************/
/*  csg_gizmos.h                                                          */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef CSG_GIZMOS_H
#define CSG_GIZMOS_H

#include "csg_shape.h"
#include "editor/editor_plugin.h"
#include "editor/spatial_editor_gizmos.h"

class CSGShapeSpatialGizmoPlugin : public EditorSpatialGizmoPlugin {
	GDCLASS(CSGShapeSpatialGizmoPlugin, EditorSpatialGizmoPlugin);

public:
	bool has_gizmo(Spatial *p_spatial);
	String get_name() const;
	int get_priority() const;
	bool is_selectable_when_hidden() const;
	void redraw(EditorSpatialGizmo *p_gizmo);

	String get_handle_name(const EditorSpatialGizmo *p_gizmo, int p_idx) const;
	Variant get_handle_value(EditorSpatialGizmo *p_gizmo, int p_idx) const;
	void set_handle(EditorSpatialGizmo *p_gizmo, int p_idx, Camera *p_camera, const Point2 &p_point);
	void commit_handle(EditorSpatialGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel);

	CSGShapeSpatialGizmoPlugin();
};

class EditorPluginCSG : public EditorPlugin {
	GDCLASS(EditorPluginCSG, EditorPlugin);

public:
	EditorPluginCSG(EditorNode *p_editor);
};

#endif // CSG_GIZMOS_H
