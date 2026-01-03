/**************************************************************************/
/*  camera_2d_editor_plugin.h                                             */
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

#pragma once

#include "editor/plugins/editor_plugin.h"

class Camera2D;
class Label;
class MenuButton;

class Camera2DEditor : public Control {
	GDCLASS(Camera2DEditor, Control);

	EditorPlugin *plugin = nullptr;

	enum Menu {
		MENU_SNAP_LIMITS_TO_VIEWPORT,
	};

	enum class Drag {
		NONE,
		LEFT,
		TOP,
		RIGHT,
		BOTTOM,
		TOP_LEFT,
		TOP_RIGHT,
		BOTTOM_LEFT,
		BOTTOM_RIGHT,
		CENTER,
	};
	Drag drag_type = Drag::NONE;
	Drag hover_type = Drag::NONE;

	Rect2 drag_revert;
	Vector2 center_drag_point;

	Camera2D *selected_camera = nullptr;

	friend class Camera2DEditorPlugin;
	MenuButton *options = nullptr;

	void _menu_option(int p_option);
	void _snap_limits_to_viewport(Camera2D *p_camera);
	void _update_overlays_if_needed(Camera2D *p_camera);
	void _update_hover(const Vector2 &p_mouse_pos);
	void _editor_tool_changed();

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	void edit(Camera2D *p_camera);

	bool forward_canvas_gui_input(const Ref<InputEvent> &p_event);
	void forward_canvas_draw_over_viewport(Control *p_overlay);

	Camera2DEditor(EditorPlugin *p_plugin);
};

class Camera2DEditorPlugin : public EditorPlugin {
	GDCLASS(Camera2DEditorPlugin, EditorPlugin);

	Camera2DEditor *camera_2d_editor = nullptr;

public:
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;

	virtual bool forward_canvas_gui_input(const Ref<InputEvent> &p_event) override { return camera_2d_editor->forward_canvas_gui_input(p_event); }
	virtual void forward_canvas_draw_over_viewport(Control *p_overlay) override { camera_2d_editor->forward_canvas_draw_over_viewport(p_overlay); }

	Camera2DEditorPlugin();
};
