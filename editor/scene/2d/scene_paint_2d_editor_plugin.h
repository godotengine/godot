/**************************************************************************/
/*  scene_paint_2d_editor_plugin.h                                        */
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

class Node2D;
class HBoxContainer;
class BaseButton;
class CheckBox;
class SpinBox;
class ButtonGroup;
class ItemList;
class OptionButton;

class ScenePaint2DEditor : public Control {
	GDCLASS(ScenePaint2DEditor, Control);

	friend class ScenePaint2DEditorPlugin;

	enum PaintMode {
		PAINT_MODE_FREE,
		PAINT_MODE_SNAP_CELL,
		PAINT_MODE_SNAP_GRID,
	};

	enum PickMode {
		PICK_FILE_SYSTEM,
		PICK_SCENE_TREE,
		PICK_CANVAS_ITEM,
		PICK_RECENT_LIST,
	};

	enum InputTool {
		INPUT_TOOL_NONE,
		INPUT_TOOL_PAINT,
		INPUT_TOOL_ERASE,
		INPUT_TOOL_PAN,
		INPUT_TOOL_PICK,
		INPUT_TOOL_QUICK_PICK,
	};

	bool pinned = false;
	bool edit_properties = false;
	bool grid = false;
	bool is_tool_selected = false;
	bool allow_overlapping = false;

	PaintMode paint_mode = PAINT_MODE_FREE;
	InputTool input_tool = INPUT_TOOL_NONE;

	Size2i grid_step = Size2i(16, 16);

	int recent_idx = -1;

	Popup *advanced_settings_popup = nullptr;
	Control *custom_overlay = nullptr;
	Control *viewport = nullptr;
	HashMap<Node *, Node *> pinned_nodes;

	Node2D *cache_node = nullptr;
	Node2D *node = nullptr;
	Node2D *instance = nullptr;
	Node2D *instance_container = nullptr;

	HBoxContainer *toolbar = nullptr;

	Button *pin_node_button = nullptr;
	Button *scene_picker_button = nullptr;
	OptionButton *recent_scenes_button = nullptr;
	Button *edit_properties_button = nullptr;

	Ref<ButtonGroup> button_group;
	Button *advanced_settings_button = nullptr;
	CheckBox *grid_toggle_button = nullptr;
	CheckBox *free_paint_button = nullptr;
	CheckBox *snap_cell_button = nullptr;
	CheckBox *snap_grid_button = nullptr;
	CheckBox *allow_overlapping_button = nullptr;

	Node2D *selected_scene = nullptr;

	void _can_handle(bool p_is_node_2d, bool p_edit);

	void _edit(Object *p_object);

	void _update_node(Object *p_object = nullptr);
	bool _is_node_valid();

	void _add_instance(bool p_show = false);
	void _clear_instance(bool p_hide = false);
	void _update_instance();
	bool _is_instance_valid();

	void _draw_overlay();
	void _update_draw_overlay();

	void _gui_input_viewport(const Ref<InputEvent> &p_event);
	void _add_node_at_pos();
	void _remove_node_at_pos();

	Vector2 _get_mouse_grid_cell();

	void _set_pinned(bool p_pinned);
	void _pinned_toggled(bool p_pressed);
	void _scene_changed();

	void _scene_picker_toggled(bool p_pressed);
	void _update_scene_picker(int p_mode);
	void _file_system_input(const Ref<InputEvent> &p_event);
	void _scene_tree_input(const Ref<InputEvent> &p_event);
	void _recent_item_selected(int p_idx);
	void _set_picked_scene(Node2D *p_scene);

	void _add_to_recent_scenes(const String &p_scene);
	void _update_recent_scenes();

	bool _is_selected_scene_valid(Node2D *p_node) const;
	bool _is_scene_painted(Node2D *p_node) const;
	Node2D *_get_node_root(Node *p_node) const;

	void _edit_properties_toggled(bool p_pressed);
	void _edit_properties();

	void _advanced_settings_pressed();
	void _grid_toggled(bool p_toggled);
	void _paint_mode_changed(int p_mode);
	void _grid_step_changed();
	void _allow_overlapping_toggled(bool p_toggled) { allow_overlapping = p_toggled; }

protected:
	void _notification(int p_what);

public:
	void forward_canvas_draw_over_viewport(Control *p_overlay);

	ScenePaint2DEditor();
};

class ScenePaint2DEditorPlugin : public EditorPlugin {
	GDCLASS(ScenePaint2DEditorPlugin, EditorPlugin);

	ScenePaint2DEditor *scene_paint_2d_editor = nullptr;

	mutable bool is_node_2d = false;

	void _canvas_item_tool_changed(int p_tool);

protected:
	void _notification(int p_what);

public:
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;
	virtual void forward_canvas_draw_over_viewport(Control *p_overlay) override;

	ScenePaint2DEditorPlugin();
};
