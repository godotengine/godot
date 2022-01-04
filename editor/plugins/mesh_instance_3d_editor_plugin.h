/*************************************************************************/
/*  mesh_instance_3d_editor_plugin.h                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef MESH_INSTANCE_EDITOR_PLUGIN_H
#define MESH_INSTANCE_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/gui/spin_box.h"

class MeshInstance3DEditor : public Control {
	GDCLASS(MeshInstance3DEditor, Control);

	enum Menu {
		MENU_OPTION_CREATE_STATIC_TRIMESH_BODY,
		MENU_OPTION_CREATE_TRIMESH_COLLISION_SHAPE,
		MENU_OPTION_CREATE_SINGLE_CONVEX_COLLISION_SHAPE,
		MENU_OPTION_CREATE_SIMPLIFIED_CONVEX_COLLISION_SHAPE,
		MENU_OPTION_CREATE_MULTIPLE_CONVEX_COLLISION_SHAPES,
		MENU_OPTION_CREATE_NAVMESH,
		MENU_OPTION_CREATE_OUTLINE_MESH,
		MENU_OPTION_CREATE_UV2,
		MENU_OPTION_DEBUG_UV1,
		MENU_OPTION_DEBUG_UV2,
	};

	MeshInstance3D *node;

	MenuButton *options;

	ConfirmationDialog *outline_dialog;
	SpinBox *outline_size;

	AcceptDialog *err_dialog;

	AcceptDialog *debug_uv_dialog;
	Control *debug_uv;
	Vector<Vector2> uv_lines;

	void _menu_option(int p_option);
	void _create_outline_mesh();

	void _create_uv_lines(int p_layer);
	friend class MeshInstance3DEditorPlugin;

	void _debug_uv_draw();

protected:
	void _node_removed(Node *p_node);
	static void _bind_methods();

public:
	void edit(MeshInstance3D *p_mesh);
	MeshInstance3DEditor();
};

class MeshInstance3DEditorPlugin : public EditorPlugin {
	GDCLASS(MeshInstance3DEditorPlugin, EditorPlugin);

	MeshInstance3DEditor *mesh_editor;
	EditorNode *editor;

public:
	virtual String get_name() const override { return "MeshInstance3D"; }
	bool has_main_screen() const override { return false; }
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;

	MeshInstance3DEditorPlugin(EditorNode *p_node);
	~MeshInstance3DEditorPlugin();
};

#endif // MESH_EDITOR_PLUGIN_H
