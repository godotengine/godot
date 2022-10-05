/**************************************************************************/
/*  vr_editor.h                                                           */
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

#ifndef VR_EDITOR_H
#define VR_EDITOR_H

#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor/editor_themes.h"
#include "editor/plugins/node_3d_editor_plugin.h"
#include "scene/main/viewport.h"
#include "vr_editor_avatar.h"
#include "vr_window.h"

class VREditor : public Node3D {
	GDCLASS(VREditor, Node3D);

	enum {
		GIZMO_VR_LAYER = Node3DEditorViewport::GIZMO_BASE_LAYER + 4, // 0-3 are taken by our normal views and while not shown, may still create/update things
	};

private:
	Viewport *xr_viewport = nullptr; // Viewport we render our XR content too

	VRWindow *editor_window = nullptr; // Window in which we show our editor
	EditorNode *editor_node = nullptr; // Our editor instance
	Node3DEditor *spatial_editor = nullptr; // Our 3D editor instance

	VREditorAvatar *avatar = nullptr;

	void _update_layers();

	/* gizmo logic */
	RID move_gizmo_instance[3], move_plane_gizmo_instance[3], rotate_gizmo_instance[4], scale_gizmo_instance[3], scale_plane_gizmo_instance[3], axis_gizmo_instance[3];
	void _init_gizmo_instance();
	void _finish_gizmo_instances();

protected:
	EditorNode *get_editor_node() const { return editor_node; }

	static void _bind_methods();
	void _notification(int p_notification);

public:
	static EditorNode *init_editor(SceneTree *p_scene_tree);

	void update_transform_gizmo_view();

	VREditor(Viewport *p_xr_viewport);
	~VREditor();
};

#endif // VR_EDITOR_H
