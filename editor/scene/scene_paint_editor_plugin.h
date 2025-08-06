/**************************************************************************/
/*  scene_paint_editor_plugin.h                                           */
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

#include "core/templates/hash_map.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/plugins/editor_plugin.h"
#include "editor/scene/canvas_item_editor_plugin.h"
#include "scene/2d/node_2d.h"
#include "scene/gui/box_container.h"
#include "scene/gui/check_box.h"
#include "scene/gui/control.h"
#include "scene/gui/item_list.h"
#include "scene/gui/spin_box.h"
#include "scene/gui/tree.h"

class ScenePaintEditor : public Control {
	GDCLASS(ScenePaintEditor, Control);

	friend class ScenePaintEditorPlugin;

	enum {
		PICK_FILE_SYSTEM,
		PICK_SCENE_TREE,
		PICK_CANVAS_ITEM,
	};

	bool pinned = false;
	bool scene_picker = false;
	bool edit_properties = false;
	bool grid = false;
	bool snap_grid = false;
	bool is_tool_selected = false;
	bool is_painting = false;
	bool is_erasing = false;

	Size2i grid_step = Size2i(16, 16);
	Vector2 last_paint_pos;
	Rect2 paint_rect;
	Vector2 mouse_pos;
	Transform2D canvas_xform;

	EditorUndoRedoManager *undo_redo = nullptr;
	Popup *grid_settings_popup = nullptr;
	CanvasItemEditor *canvas = nullptr;
	Control *viewport = nullptr;
	Node2D *node = nullptr;
	Node2D *preview = nullptr;
	Node2D *instance = nullptr;

	HBoxContainer *toolbar = nullptr;

	Button *pin_node_button = nullptr;
	Button *scene_picker_button = nullptr;
	Button *edit_properties_button = nullptr;

	Button *grid_settings_button = nullptr;
	CheckBox *grid_toggle_button = nullptr;
	CheckBox *snap_grid_button = nullptr;
	SpinBox *grid_step_x = nullptr;
	SpinBox *grid_step_y = nullptr;

	Node2D *selected_scene;

	void _edit(Object *p_object);

	void _draw();
	void _draw_grid();
	void _draw_grid_highlight();
	void _update_draw();
	void _update_preview();
	void _update_paint_rect();
	void _update_preview_position();

	void _gui_input_viewport(const Ref<InputEvent> &p_event);
	void _add_node_at_pos();
	void _remove_node_at_pos();

	Vector2 _get_mouse_grid_cell();

	void _pinned_toggled(bool p_pressed);

	void _scene_picker_toggled(bool p_pressed);
	void _update_scene_picker(int p_mode);
	void _file_system_input(const Ref<InputEvent> &p_event);
	void _scene_tree_input(const Ref<InputEvent> &p_event);

	void _edit_properties_toggled(bool p_pressed);
	void _edit_properties();

	void _grid_settings_pressed();
	void _grid_toggled(bool p_toggled);
	void _snap_grid_toggled(bool p_toggled);
	void _grid_step_changed();
	void _update_grid_step();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_grid_step(const Size2i p_size);
	Size2i get_grid_step() const;

	ScenePaintEditor();
	~ScenePaintEditor();
};

class ScenePaintEditorPlugin : public EditorPlugin {
	GDCLASS(ScenePaintEditorPlugin, EditorPlugin);

	ScenePaintEditor *scene_paint_editor = nullptr;

	mutable bool is_node_2d = false;

	void _canvas_item_tool_changed(int p_tool);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual String get_plugin_name() const override { return "ScenePaint"; }
	bool has_main_screen() const override { return false; }
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;

	ScenePaintEditorPlugin();
};
