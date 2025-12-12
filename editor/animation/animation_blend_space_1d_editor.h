/**************************************************************************/
/*  animation_blend_space_1d_editor.h                                     */
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

#include "editor/animation/animation_tree_editor_plugin.h"
#include "editor/plugins/editor_plugin.h"
#include "scene/animation/animation_blend_space_1d.h"
#include "scene/gui/graph_edit.h"

class Button;
class CheckBox;
class LineEdit;
class OptionButton;
class PanelContainer;
class SpinBox;
class Timer;
class VSeparator;

class AnimationNodeBlendSpace1DEditor : public AnimationTreeNodeEditorPlugin {
	GDCLASS(AnimationNodeBlendSpace1DEditor, AnimationTreeNodeEditorPlugin);

	Ref<AnimationNodeBlendSpace1D> blend_space;
	bool read_only = false;

	PanelContainer *panel = nullptr;
	Button *tool_blend = nullptr;
	Button *tool_select = nullptr;
	Button *tool_create = nullptr;
	VSeparator *tool_erase_sep = nullptr;
	Button *tool_erase = nullptr;
	Button *snap = nullptr;
	SpinBox *snap_value = nullptr;

	LineEdit *label_value = nullptr;
	SpinBox *max_value = nullptr;
	SpinBox *min_value = nullptr;

	CheckBox *sync = nullptr;
	OptionButton *interpolation = nullptr;

	HBoxContainer *edit_hb = nullptr;
	SpinBox *edit_value = nullptr;
	Button *open_editor = nullptr;
	VSeparator *open_editor_sep = nullptr;
	SpinBox *index_edit = nullptr;

	int selected_point = -1;

	Control *blend_space_draw = nullptr;

	PanelContainer *error_panel = nullptr;
	Label *error_label = nullptr;

	bool updating = false;

	static AnimationNodeBlendSpace1DEditor *singleton;

	void _blend_space_gui_input(const Ref<InputEvent> &p_event);
	void _blend_space_draw();

	void _update_space();

	void _config_changed(double);
	void _labels_changed(String);
	void _snap_toggled();

	PopupMenu *menu = nullptr;
	PopupMenu *animations_menu = nullptr;
	Vector<String> animations_to_add;
	float add_point_pos = 0.0f;
	Vector<real_t> points;

	bool dragging_blend_position = false;
	bool dragging_selected_attempt = false;
	bool dragging_selected = false;
	Vector2 drag_from;
	Vector2 drag_ofs;

	Vector<Rect2> text_rects;
	int editing_point = -1;
	LineEdit *inline_editor = nullptr;
	float inline_editor_point_x = 0.0f;
	bool index_edit_has_focus = false;
	bool show_indices = false;
	Timer *index_focus_cooldown_timer = nullptr;

	void _add_menu_type(int p_index);
	void _add_animation_type(int p_index);

	void _tool_switch(int p_tool);
	void _update_edited_point_pos();
	void _update_edited_point_name();
	void _update_tool_erase();
	void _erase_selected();
	void _edit_point_pos(double);
	void _edit_point_name(const String &p_name);
	void _edit_point_index(double p_index);
	void _set_selected_point(int p_index);
	void _start_inline_edit(int p_point);
	void _finish_inline_edit();
	void _finish_inline_edit_with_text(const String &p_text);
	void _cancel_inline_edit();
	void _inline_editor_text_changed(const String &p_text);
	void _open_editor();
	void _index_edit_focus_entered();
	void _index_edit_focus_exited();
	void _index_focus_cooldown_timeout();
	void _show_indices_with_cooldown();

	EditorFileDialog *open_file = nullptr;
	Ref<AnimationNode> file_loaded;
	void _file_opened(const String &p_file);

	enum {
		MENU_LOAD_FILE = 1000,
		MENU_PASTE = 1001,
		MENU_LOAD_FILE_CONFIRM = 1002
	};

	StringName get_blend_position_path() const;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	static AnimationNodeBlendSpace1DEditor *get_singleton() { return singleton; }
	void refresh_editor() { _update_space(); }
	virtual bool can_edit(const Ref<AnimationNode> &p_node) override;
	virtual void edit(const Ref<AnimationNode> &p_node) override;
	AnimationNodeBlendSpace1DEditor();
};
