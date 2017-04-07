/*************************************************************************/
/*  visual_script_editor.h                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef VISUALSCRIPT_EDITOR_H
#define VISUALSCRIPT_EDITOR_H

#include "editor/create_dialog.h"
#include "editor/plugins/script_editor_plugin.h"
#include "editor/property_editor.h"
#include "editor/property_selector.h"
#include "scene/gui/graph_edit.h"
#include "visual_script.h"
class VisualScriptEditorSignalEdit;
class VisualScriptEditorVariableEdit;

#ifdef TOOLS_ENABLED

class VisualScriptEditor : public ScriptEditorBase {
	GDCLASS(VisualScriptEditor, ScriptEditorBase)

	enum {
		TYPE_SEQUENCE = 1000,
		INDEX_BASE_SEQUENCE = 1024

	};

	enum {
		EDIT_DELETE_NODES,
		EDIT_TOGGLE_BREAKPOINT,
		EDIT_FIND_NODE_TYPE,
		EDIT_COPY_NODES,
		EDIT_CUT_NODES,
		EDIT_PASTE_NODES,
	};

	enum PortAction {

		CREATE_CALL,
		CREATE_SET,
		CREATE_GET,
		CREATE_COND,
		CREATE_SEQUENCE,
		CREATE_SWITCH,
		CREATE_ITERATOR,
		CREATE_WHILE,
		CREATE_RETURN,
	};

	MenuButton *edit_menu;

	Ref<VisualScript> script;

	Button *base_type_select;

	HSplitContainer *main_hsplit;
	VSplitContainer *left_vsplit;

	GraphEdit *graph;

	LineEdit *node_filter;
	TextureRect *node_filter_icon;

	VisualScriptEditorSignalEdit *signal_editor;

	AcceptDialog *edit_signal_dialog;
	PropertyEditor *edit_signal_edit;

	PropertySelector *method_select;
	PropertySelector *new_connect_node_select;

	VisualScriptEditorVariableEdit *variable_editor;

	AcceptDialog *edit_variable_dialog;
	PropertyEditor *edit_variable_edit;

	CustomPropertyEditor *default_value_edit;

	UndoRedo *undo_redo;

	Tree *members;
	Tree *nodes;

	Label *hint_text;
	Timer *hint_text_timer;

	Label *select_func_text;

	bool updating_graph;

	void _show_hint(const String &p_hint);
	void _hide_timer();

	CreateDialog *select_base_type;

	struct VirtualInMenu {
		String name;
		Variant::Type ret;
		bool ret_variant;
		Vector<Pair<Variant::Type, String> > args;
	};

	Map<int, VirtualInMenu> virtuals_in_menu;

	PopupMenu *new_function_menu;

	StringName edited_func;

	void _update_graph_connections();
	void _update_graph(int p_only_id = -1);

	bool updating_members;

	void _update_members();

	StringName selected;

	String _validate_name(const String &p_name) const;

	struct Clipboard {

		Map<int, Ref<VisualScriptNode> > nodes;
		Map<int, Vector2> nodes_positions;

		Set<VisualScript::SequenceConnection> sequence_connections;
		Set<VisualScript::DataConnection> data_connections;
	};

	static Clipboard *clipboard;

	PopupMenu *port_action_popup;

	PortAction port_action;
	int port_action_node;
	int port_action_output;
	Vector2 port_action_pos;
	int port_action_new_node;
	void _port_action_menu(int p_option);
	void _selected_connect_node_method_or_setget(const String &p_text);

	int error_line;

	void _node_selected(Node *p_node);
	void _center_on_node(int p_id);

	void _node_filter_changed(const String &p_text);
	void _change_base_type_callback();
	void _change_base_type();
	void _member_selected();
	void _member_edited();
	void _override_pressed(int p_id);

	void _begin_node_move();
	void _end_node_move();
	void _move_node(String func, int p_id, const Vector2 &p_to);

	void _node_moved(Vector2 p_from, Vector2 p_to, int p_id);
	void _remove_node(int p_id);
	void _graph_connected(const String &p_from, int p_from_slot, const String &p_to, int p_to_slot);
	void _graph_disconnected(const String &p_from, int p_from_slot, const String &p_to, int p_to_slot);
	void _graph_connect_to_empty(const String &p_from, int p_from_slot, const Vector2 &p_release_pos);

	void _node_ports_changed(const String &p_func, int p_id);
	void _available_node_doubleclicked();

	void _update_available_nodes();

	void _member_button(Object *p_item, int p_column, int p_button);

	void _expression_text_changed(const String &p_text, int p_id);

	String revert_on_drag;

	void _input(const InputEvent &p_event);
	void _on_nodes_delete();
	void _on_nodes_duplicate();

	Variant get_drag_data_fw(const Point2 &p_point, Control *p_from);
	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);

	int editing_id;
	int editing_input;

	void _default_value_changed();
	void _default_value_edited(Node *p_button, int p_id, int p_input_port);

	void _menu_option(int p_what);

	void _graph_ofs_changed(const Vector2 &p_ofs);
	void _comment_node_resized(const Vector2 &p_new_size, int p_node);

	int selecting_method_id;
	void _selected_method(const String &p_method);

	void _draw_color_over_button(Object *obj, Color p_color);
	void _button_resource_previewed(const String &p_path, const Ref<Texture> &p_preview, Variant p_ud);

	VisualScriptNode::TypeGuess _guess_output_type(int p_port_action_node, int p_port_action_output, Set<int> &visited_nodes);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual void apply_code();
	virtual Ref<Script> get_edited_script() const;
	virtual Vector<String> get_functions();
	virtual void set_edited_script(const Ref<Script> &p_script);
	virtual void reload_text();
	virtual String get_name();
	virtual Ref<Texture> get_icon();
	virtual bool is_unsaved();
	virtual Variant get_edit_state();
	virtual void set_edit_state(const Variant &p_state);
	virtual void goto_line(int p_line, bool p_with_error = false);
	virtual void trim_trailing_whitespace();
	virtual void ensure_focus();
	virtual void tag_saved_version();
	virtual void reload(bool p_soft);
	virtual void get_breakpoints(List<int> *p_breakpoints);
	virtual bool goto_method(const String &p_method);
	virtual void add_callback(const String &p_function, PoolStringArray p_args);
	virtual void update_settings();
	virtual void set_debugger_active(bool p_active);
	virtual void set_tooltip_request_func(String p_method, Object *p_obj);
	virtual Control *get_edit_menu();
	virtual bool can_lose_focus_on_node_selection() { return false; }

	static void register_editor();

	static void free_clipboard();

	VisualScriptEditor();
	~VisualScriptEditor();
};
#endif

#endif // VISUALSCRIPT_EDITOR_H
