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

#ifdef TOOLS_ENABLED

#include "../csg_shape.h"

#include "editor/plugins/editor_plugin.h"
#include "editor/plugins/node_3d_editor_gizmos.h"

class Gizmo3DHelper;

class CSGShape3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(CSGShape3DGizmoPlugin, EditorNode3DGizmoPlugin);

	Ref<Gizmo3DHelper> helper;

public:
	virtual bool has_gizmo(Node3D *p_spatial) override;
	virtual String get_gizmo_name() const override;
	virtual int get_priority() const override;
	virtual bool is_selectable_when_hidden() const override;
	virtual void redraw(EditorNode3DGizmo *p_gizmo) override;

	virtual String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const override;
	virtual Variant get_handle_value(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const override;
	void begin_handle_action(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) override;
	virtual void set_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point) override;
	virtual void commit_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel) override;

	CSGShape3DGizmoPlugin();
	~CSGShape3DGizmoPlugin();
};

class EditorPluginCSG : public EditorPlugin {
	GDCLASS(EditorPluginCSG, EditorPlugin);

public:
	EditorPluginCSG();
};

#endif // TOOLS_ENABLED

#endif // CSG_GIZMOS_H
