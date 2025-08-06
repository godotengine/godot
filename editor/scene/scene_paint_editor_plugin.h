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

#include "editor/plugins/editor_plugin.h"
#include "scene/gui/control.h"

class CanvasItemEditor;
class Node2D;
class HBoxContainer;
class BaseButton;
class CheckBox;
class SpinBox;
class ButtonGroup;

class ScenePaintEditor : public Control {
	GDCLASS(ScenePaintEditor, Control);

	friend class ScenePaintEditorPlugin;

	enum PaintMode {
		PAINT_MODE_FREE,
		PAINT_MODE_SNAP_CELL,
		PAINT_MODE_SNAP_GRID,
	};

	enum PickMode {
		PICK_FILE_SYSTEM,
		PICK_SCENE_TREE,
		PICK_CANVAS_ITEM,
	};

	bool pinned = false;
	bool scene_picker = false;
	bool viewport_scene_picker = false;
	bool edit_properties = false;
	bool grid = false;
	bool snap_cell = false;
	bool snap_grid = false;
	bool is_tool_selected = false;
	bool is_painting = false;
	bool is_erasing = false;
	bool is_panning = false;

	Size2i grid_step = Size2i(16, 16);

	EditorUndoRedoManager *undo_redo = nullptr;
	Popup *grid_settings_popup = nullptr;
	CanvasItemEditor *canvas_item_editor = nullptr;
	Control *custom_overlay = nullptr;
	Control *viewport = nullptr;
	HashMap<Node *, Node *> pinned_nodes;
	Node2D *cache_node = nullptr;
	Node2D *node = nullptr;
	Node2D *preview = nullptr;
	Node2D *instance = nullptr;

	HBoxContainer *toolbar = nullptr;

	Button *pin_node_button = nullptr;
	Button *scene_picker_button = nullptr;
	Button *edit_properties_button = nullptr;

	Ref<ButtonGroup> button_group;
	Button *grid_settings_button = nullptr;
	CheckBox *grid_toggle_button = nullptr;
	CheckBox *free_paint_button = nullptr;
	CheckBox *snap_cell_button = nullptr;
	CheckBox *snap_grid_button = nullptr;
	SpinBox *grid_step_x = nullptr;
	SpinBox *grid_step_y = nullptr;

	Node2D *selected_scene = nullptr;

	void _can_handle(bool p_is_node_2d, bool p_edit);

	void _edit(Object *p_object);

	void _update_node(Object *p_object = nullptr);

	void _draw_overlay();
	void _update_draw_overlay();

	void _update_preview();
	void _update_preview_position();

	void _set_input_tool(bool p_painting, bool p_erasing, bool p_force);
	void _gui_input_viewport(const Ref<InputEvent> &p_event);
	void _add_node_at_pos();
	void _remove_node_at_pos();

	Vector2 _get_mouse_grid_cell();

	void _pinned_toggled(bool p_pressed);
	void _scene_changed();

	void _scene_picker_toggled(bool p_pressed);
	void _update_scene_picker(PickMode p_mode);
	void _file_system_input(const Ref<InputEvent> &p_event);
	void _scene_tree_input(const Ref<InputEvent> &p_event);

	bool _is_instance_valid(Node2D *p_node) const;
	bool _is_scene_painted(Node2D *p_node) const;
	Node2D *_get_node_root(Node *p_node) const;

	void _edit_properties_toggled(bool p_pressed);
	void _edit_properties();

	void _grid_settings_pressed();
	void _grid_toggled(bool p_toggled);
	void _paint_mode_changed(PaintMode p_mode);
	void _grid_step_changed();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_picked_scene(Node2D *p_scene);
	void forward_canvas_draw_over_viewport(Control *p_overlay);
	void set_grid_step(const Size2i p_size);

	ScenePaintEditor();
};

class ScenePaintEditorPlugin : public EditorPlugin {
	GDCLASS(ScenePaintEditorPlugin, EditorPlugin);

	ScenePaintEditor *scene_paint_editor = nullptr;

	mutable bool is_node_2d = false;

	void _canvas_item_tool_changed(int p_tool);

protected:
	void _notification(int p_what);

public:
	virtual String get_plugin_name() const override { return "ScenePaint"; }
	bool has_main_screen() const override { return false; }
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;
	virtual void forward_canvas_draw_over_viewport(Control *p_overlay) override;

	ScenePaintEditorPlugin();
};

VARIANT_ENUM_CAST(ScenePaintEditor::PaintMode);
VARIANT_ENUM_CAST(ScenePaintEditor::PickMode);
