/**************************************************************************/
/*  path_3d_editor_plugin.h                                               */
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

#ifndef PATH_3D_EDITOR_PLUGIN_H
#define PATH_3D_EDITOR_PLUGIN_H

#include "editor/editor_plugin.h"
#include "editor/plugins/node_3d_editor_gizmos.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/path_3d.h"

class HBoxContainer;
class MenuButton;

class Path3DGizmo : public EditorNode3DGizmo {
	GDCLASS(Path3DGizmo, EditorNode3DGizmo);

	// Map handle id to control point id and handle type.
	enum HandleType {
		HANDLE_TYPE_IN,
		HANDLE_TYPE_OUT,
		HANDLE_TYPE_TILT,
	};

	struct HandleInfo {
		int point_idx; // Index of control point.
		HandleType type; // Type of this handle.
	};

	Path3D *path = nullptr;
	mutable Vector3 original;
	mutable float orig_in_length;
	mutable float orig_out_length;
	mutable float disk_size = 0.8;

	// Cache information of secondary handles.
	Vector<HandleInfo> _secondary_handles_info;

public:
	virtual String get_handle_name(int p_id, bool p_secondary) const override;
	virtual Variant get_handle_value(int p_id, bool p_secondary) const override;
	virtual void set_handle(int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point) override;
	virtual void commit_handle(int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel = false) override;

	virtual void redraw() override;
	Path3DGizmo(Path3D *p_path = nullptr, float p_disk_size = 0.8);
};

class Path3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(Path3DGizmoPlugin, EditorNode3DGizmoPlugin);

	float disk_size = 0.8;

protected:
	Ref<EditorNode3DGizmo> create_gizmo(Node3D *p_spatial) override;

public:
	String get_gizmo_name() const override;
	int get_priority() const override;
	Path3DGizmoPlugin(float p_disk_size);
};

class Path3DEditorPlugin : public EditorPlugin {
	GDCLASS(Path3DEditorPlugin, EditorPlugin);

	HBoxContainer *topmenu_bar = nullptr;
	Button *curve_create = nullptr;
	Button *curve_edit = nullptr;
	Button *curve_edit_curve = nullptr;
	Button *curve_del = nullptr;
	Button *curve_close = nullptr;
	MenuButton *handle_menu = nullptr;

	float disk_size = 0.8;

	enum Mode {
		MODE_CREATE,
		MODE_EDIT,
		MODE_EDIT_CURVE,
		MODE_DELETE,
		ACTION_CLOSE
	};

	Path3D *path = nullptr;

	void _update_theme();

	void _mode_changed(int p_mode);
	void _close_curve();
	void _handle_option_pressed(int p_option);
	bool handle_clicked = false;
	bool mirror_handle_angle;
	bool mirror_handle_length;

	enum HandleOption {
		HANDLE_OPTION_ANGLE,
		HANDLE_OPTION_LENGTH
	};

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	Path3D *get_edited_path() { return path; }

	static Path3DEditorPlugin *singleton;
	virtual EditorPlugin::AfterGUIInput forward_3d_gui_input(Camera3D *p_camera, const Ref<InputEvent> &p_event) override;

	virtual String get_name() const override { return "Path3D"; }
	bool has_main_screen() const override { return false; }
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;

	bool mirror_angle_enabled() { return mirror_handle_angle; }
	bool mirror_length_enabled() { return mirror_handle_length; }
	bool is_handle_clicked() { return handle_clicked; }
	void set_handle_clicked(bool clicked) { handle_clicked = clicked; }

	Path3DEditorPlugin();
	~Path3DEditorPlugin();
};

#endif // PATH_3D_EDITOR_PLUGIN_H
