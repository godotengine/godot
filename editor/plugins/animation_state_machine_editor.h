/*************************************************************************/
/*  animation_state_machine_editor.h                                     */
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

#ifndef ANIMATION_STATE_MACHINE_EDITOR_H
#define ANIMATION_STATE_MACHINE_EDITOR_H

#include "editor/editor_plugin.h"
#include "editor/plugins/animation_tree_editor_plugin.h"
#include "editor/property_editor.h"
#include "scene/animation/animation_node_state_machine.h"
#include "scene/gui/button.h"
#include "scene/gui/graph_edit.h"
#include "scene/gui/popup.h"
#include "scene/gui/tree.h"

class EditorFileDialog;

class AnimationNodeStateMachineEditor : public AnimationTreeNodeEditorPlugin {
	GDCLASS(AnimationNodeStateMachineEditor, AnimationTreeNodeEditorPlugin);

	Ref<AnimationNodeStateMachine> state_machine;

	Button *tool_select = nullptr;
	Button *tool_create = nullptr;
	Button *tool_connect = nullptr;
	Button *tool_group = nullptr;
	Button *tool_ungroup = nullptr;
	Popup *name_edit_popup = nullptr;
	LineEdit *name_edit = nullptr;

	HBoxContainer *tool_erase_hb = nullptr;
	Button *tool_erase = nullptr;

	OptionButton *transition_mode = nullptr;
	OptionButton *play_mode = nullptr;

	PanelContainer *panel = nullptr;

	StringName selected_node;
	Set<StringName> selected_nodes;

	HScrollBar *h_scroll = nullptr;
	VScrollBar *v_scroll = nullptr;

	Control *state_machine_draw = nullptr;
	Control *state_machine_play_pos = nullptr;

	PanelContainer *error_panel = nullptr;
	Label *error_label = nullptr;

	bool updating = false;

	UndoRedo *undo_redo = nullptr;

	static AnimationNodeStateMachineEditor *singleton;

	void _state_machine_gui_input(const Ref<InputEvent> &p_event);
	void _connection_draw(const Vector2 &p_from, const Vector2 &p_to, AnimationNodeStateMachineTransition::SwitchMode p_mode, bool p_enabled, bool p_selected, bool p_travel, bool p_auto_advance, bool p_multi_transitions);
	void _state_machine_draw();
	void _state_machine_pos_draw();

	void _update_graph();

	PopupMenu *menu = nullptr;
	PopupMenu *connect_menu = nullptr;
	PopupMenu *state_machine_menu = nullptr;
	PopupMenu *end_menu = nullptr;
	PopupMenu *animations_menu = nullptr;
	Vector<String> animations_to_add;
	Vector<String> nodes_to_connect;

	Vector2 add_node_pos;

	ConfirmationDialog *delete_window;
	Tree *delete_tree;

	bool box_selecting = false;
	Point2 box_selecting_from;
	Point2 box_selecting_to;
	Rect2 box_selecting_rect;
	Set<StringName> previous_selected;

	bool dragging_selected_attempt = false;
	bool dragging_selected = false;
	Vector2 drag_from;
	Vector2 drag_ofs;
	StringName snap_x;
	StringName snap_y;

	bool connecting = false;
	StringName connecting_from;
	Vector2 connecting_to;
	StringName connecting_to_node;

	void _add_menu_type(int p_index);
	void _add_animation_type(int p_index);
	void _connect_to(int p_index);

	void _removed_from_graph();

	struct NodeRect {
		StringName node_name;
		Rect2 node;
		Rect2 play;
		Rect2 name;
		Rect2 edit;
	};

	Vector<NodeRect> node_rects;

	struct TransitionLine {
		StringName from_node;
		StringName to_node;
		Vector2 from;
		Vector2 to;
		AnimationNodeStateMachineTransition::SwitchMode mode;
		StringName advance_condition_name;
		bool advance_condition_state = false;
		bool disabled = false;
		bool auto_advance = false;
		float width = 0;
		bool selected;
		bool travel;
		bool hidden;
		int transition_index;
		Vector<TransitionLine> multi_transitions;
	};

	Vector<TransitionLine> transition_lines;

	struct NodeUR {
		StringName name;
		Ref<AnimationNode> node;
		Vector2 position;
	};

	struct TransitionUR {
		StringName new_from;
		StringName new_to;
		StringName old_from;
		StringName old_to;
		Ref<AnimationNodeStateMachineTransition> transition;
	};

	StringName selected_transition_from;
	StringName selected_transition_to;
	int selected_transition_index;
	TransitionLine selected_multi_transition;
	void _add_transition(const bool p_nested_action = false);

	StringName over_node;
	int over_node_what = -1;

	String prev_name;
	void _name_edited(const String &p_text);
	void _name_edited_focus_out();
	void _open_editor(const String &p_name);
	void _scroll_changed(double);

	void _clip_src_line_to_rect(Vector2 &r_from, const Vector2 &p_to, const Rect2 &p_rect);
	void _clip_dst_line_to_rect(const Vector2 &p_from, Vector2 &r_to, const Rect2 &p_rect);

	void _erase_selected(const bool p_nested_action = false);
	void _update_mode();
	void _open_menu(const Vector2 &p_position);
	void _open_connect_menu(const Vector2 &p_position);
	bool _create_submenu(PopupMenu *p_menu, Ref<AnimationNodeStateMachine> p_nodesm, const StringName &p_name, const StringName &p_path, bool from_root = false, Vector<Ref<AnimationNodeStateMachine>> p_parents = Vector<Ref<AnimationNodeStateMachine>>());
	void _stop_connecting();

	void _group_selected_nodes();
	void _ungroup_selected_nodes();

	void _delete_selected();
	void _delete_all();
	void _delete_tree_draw();

	bool last_active = false;
	StringName last_blend_from_node;
	StringName last_current_node;
	Vector<StringName> last_travel_path;
	float last_play_pos = 0.0f;
	float play_pos = 0.0f;
	float current_length = 0.0f;

	float error_time = 0.0f;
	String error_text;

	EditorFileDialog *open_file = nullptr;
	Ref<AnimationNode> file_loaded;
	void _file_opened(const String &p_file);

	enum {
		MENU_LOAD_FILE = 1000,
		MENU_PASTE = 1001,
		MENU_LOAD_FILE_CONFIRM = 1002
	};

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	static AnimationNodeStateMachineEditor *get_singleton() { return singleton; }
	virtual bool can_edit(const Ref<AnimationNode> &p_node) override;
	virtual void edit(const Ref<AnimationNode> &p_node) override;
	virtual CursorShape get_cursor_shape(const Point2 &p_pos) const override;
	AnimationNodeStateMachineEditor();
};

class EditorAnimationMultiTransitionEdit : public RefCounted {
	GDCLASS(EditorAnimationMultiTransitionEdit, RefCounted);

	struct Transition {
		StringName from;
		StringName to;
		Ref<AnimationNodeStateMachineTransition> transition;
	};

	Vector<Transition> transitions;

protected:
	bool _set(const StringName &p_name, const Variant &p_property);
	bool _get(const StringName &p_name, Variant &r_property) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	void add_transition(const StringName &p_from, const StringName &p_to, Ref<AnimationNodeStateMachineTransition> p_transition);

	EditorAnimationMultiTransitionEdit(){};
};

#endif // ANIMATION_STATE_MACHINE_EDITOR_H
