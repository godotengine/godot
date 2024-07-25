/**************************************************************************/
/*  navigation_obstacle_3d_editor_plugin.h                                */
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

#ifndef NAVIGATION_OBSTACLE_3D_EDITOR_PLUGIN_H
#define NAVIGATION_OBSTACLE_3D_EDITOR_PLUGIN_H

#include "editor/plugins/editor_plugin.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/physics/collision_polygon_3d.h"
#include "scene/gui/box_container.h"
#include "scene/resources/immediate_mesh.h"

#include "scene/3d/navigation_obstacle_3d.h"

class CanvasItemEditor;
class MenuButton;

class NavigationObstacle3DEditor : public HBoxContainer {
	GDCLASS(NavigationObstacle3DEditor, HBoxContainer);

	enum Mode {
		MODE_CREATE,
		MODE_EDIT,

	};

	Mode mode;

	Button *button_create = nullptr;
	Button *button_edit = nullptr;

	Ref<StandardMaterial3D> line_material;
	Ref<StandardMaterial3D> handle_material;

	Panel *panel = nullptr;
	NavigationObstacle3D *obstacle_node = nullptr;
	Ref<ImmediateMesh> point_lines_mesh;
	MeshInstance3D *point_lines_meshinstance = nullptr;
	MeshInstance3D *point_handles_meshinstance = nullptr;
	Ref<ArrayMesh> point_handle_mesh;

	MenuButton *options = nullptr;

	int edited_point = 0;
	Vector2 edited_point_pos;
	PackedVector2Array pre_move_edit;
	PackedVector2Array wip;
	bool wip_active;
	bool snap_ignore;

	float prev_depth = 0.0f;

	void _wip_close();
	void _polygon_draw();
	void _menu_option(int p_option);

	PackedVector2Array _get_polygon();
	void _set_polygon(const PackedVector2Array &p_poly);

protected:
	void _notification(int p_what);
	void _node_removed(Node *p_node);
	static void _bind_methods();

public:
	virtual EditorPlugin::AfterGUIInput forward_3d_gui_input(Camera3D *p_camera, const Ref<InputEvent> &p_event);
	void edit(Node *p_node);
	NavigationObstacle3DEditor();
	~NavigationObstacle3DEditor();
};

class NavigationObstacle3DEditorPlugin : public EditorPlugin {
	GDCLASS(NavigationObstacle3DEditorPlugin, EditorPlugin);

	NavigationObstacle3DEditor *obstacle_editor = nullptr;

public:
	virtual EditorPlugin::AfterGUIInput forward_3d_gui_input(Camera3D *p_camera, const Ref<InputEvent> &p_event) override { return obstacle_editor->forward_3d_gui_input(p_camera, p_event); }

	virtual String get_name() const override { return "NavigationObstacle3DEditor"; }
	bool has_main_screen() const override { return false; }
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;

	NavigationObstacle3DEditorPlugin();
	~NavigationObstacle3DEditorPlugin();
};

#endif // NAVIGATION_OBSTACLE_3D_EDITOR_PLUGIN_H
