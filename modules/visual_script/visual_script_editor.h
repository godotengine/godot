/*************************************************************************/
/*  visual_script_editor.h                                               */
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

#ifndef VISUALSCRIPT_EDITOR_H
#define VISUALSCRIPT_EDITOR_H

#include "editor/create_dialog.h"
#include "editor/plugins/script_editor_plugin.h"
#include "editor/property_editor.h"
#include "scene/gui/graph_edit.h"
#include "visual_script.h"
#include "visual_script_property_selector.h"

class VisualScriptEditorSignalEdit;
class VisualScriptEditorVariableEdit;

#ifdef TOOLS_ENABLED

class VisualScriptEditor : public ScriptEditorBase {
	GDCLASS(VisualScriptEditor, ScriptEditorBase);

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
		EDIT_CREATE_FUNCTION,
		REFRESH_GRAPH
	};

	enum PortAction {

		CREATE_CALL_SET_GET,
		CREATE_ACTION,
	};

	enum MemberAction {
		MEMBER_EDIT,
		MEMBER_REMOVE

	};

	enum MemberType {
		MEMBER_FUNCTION,
		MEMBER_VARIABLE,
		MEMBER_SIGNAL
	};

	VBoxContainer *members_section;
	MenuButton *edit_menu;

	Ref<VisualScript> script;

	Button *base_type_select;

	LineEdit *func_name_box;
	ScrollContainer *func_input_scroll;
	VBoxContainer *func_input_vbox;
	ConfirmationDialog *function_create_dialog;

	GraphEdit *graph;

	VisualScriptEditorSignalEdit *signal_editor;

	AcceptDialog *edit_signal_dialog;
	EditorInspector *edit_signal_edit;

	VisualScriptPropertySelector *method_select;
	VisualScriptPropertySelector *new_connect_node_select;
	VisualScriptPropertySelector *new_virtual_method_select;

	VisualScriptEditorVariableEdit *variable_editor;

	AcceptDialog *edit_variable_dialog;
	EditorInspector *edit_variable_edit;

	CustomPropertyEditor *default_value_edit;

	UndoRedo *undo_redo;

	Tree *members;
	PopupDialog *function_name_edit;
	LineEdit *function_name_box;

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
		Vector<Pair<Variant::Type, String>> args;
	};

	HashMap<StringName, Ref<StyleBox>> node_styles;
	StringName edited_func;
	StringName default_func;

	void _update_graph_connections();
	void _update_graph(int p_only_id = -1);

	bool updating_members;

	void _update_members();
	String _sanitized_variant_text(const StringName &property_name);

	StringName selected;

	String _validate_name(const String &p_name) const;

	struct Clipboard {
		Map<int, Ref<VisualScriptNode>> nodes;
		Map<int, Vector2> nodes_positions;

		Set<VisualScript::SequenceConnection> sequence_connections;
		Set<VisualScript::DataConnection> data_connections;
	};

	static Clipboard *clipboard;

	PopupMenu *member_popup;
	MemberType member_type;
	String member_name;

	PortAction port_action;
	int port_action_node;
	int port_action_output;
	Vector2 port_action_pos;
	int port_action_new_node;

	bool saved_pos_dirty;
	Vector2 saved_position;

	Vector2 mouse_up_position;

	void _port_action_menu(int p_option, const StringName &p_func);

	NodePath drop_path;
	Node *drop_node = nullptr;
	Vector2 drop_position;
	void connect_data(Ref<VisualScriptNode> vnode_old, Ref<VisualScriptNode> vnode, int new_id);

	void _selected_connect_node(const String &p_text, const String &p_category, const bool p_connecting = true);
	void connect_seq(Ref<VisualScriptNode> vnode_old, Ref<VisualScriptNode> vnode_new, int new_id);

	void _cancel_connect_node();
	int _create_new_node_from_name(const String &p_text, const Vector2 &p_point, const StringName &p_func = StringName());
	void _selected_new_virtual_method(const String &p_text, const String &p_category, const bool p_connecting);

	int error_line;

	void _node_selected(Node *p_node);
	void _center_on_node(const StringName &p_func, int p_id);

	void _node_filter_changed(const String &p_text);
	void _change_base_type_callback();
	void _change_base_type();
	void _toggle_tool_script();
	void _member_selected();
	void _member_edited();

	void _begin_node_move();
	void _end_node_move();
	void _move_node(const StringName &p_func, int p_id, const Vector2 &p_to);

	void _get_ends(int p_node, const List<VisualScript::SequenceConnection> &p_seqs, const Set<int> &p_selected, Set<int> &r_end_nodes);

	void _node_moved(Vector2 p_from, Vector2 p_to, int p_id);
	void _remove_node(int p_id);
	void _graph_connected(const String &p_from, int p_from_slot, const String &p_to, int p_to_slot);
	void _graph_disconnected(const String &p_from, int p_from_slot, const String &p_to, int p_to_slot);
	void _graph_connect_to_empty(const String &p_from, int p_from_slot, const Vector2 &p_release_pos);

	void _node_ports_changed(const String &p_func, int p_id);
	void _node_create();

	void _update_available_nodes();

	void _member_button(Object *p_item, int p_column, int p_button);

	void _expression_text_changed(const String &p_text, int p_id);
	void _add_input_port(int p_id);
	void _add_output_port(int p_id);
	void _remove_input_port(int p_id, int p_port);
	void _remove_output_port(int p_id, int p_port);
	void _change_port_type(int p_select, int p_id, int p_port, bool is_input);
	void _update_node_size(int p_id);
	void _port_name_focus_out(const Node *p_name_box, int p_id, int p_port, bool is_input);

	Vector2 _get_pos_in_graph(Vector2 p_point) const;
	Vector2 _get_available_pos(bool p_centered = true, Vector2 p_pos = Vector2()) const;
	StringName _get_function_of_node(int p_id) const;

	void _move_nodes_with_rescan(const StringName &p_func_from, const StringName &p_func_to, int p_id);
	bool node_has_sequence_connections(const StringName &p_func, int p_id);

	void _generic_search(String p_base_type = "", Vector2 pos = Vector2(), bool node_centered = false);

	void _input(const Ref<InputEvent> &p_event);
	void _graph_gui_input(const Ref<InputEvent> &p_event);
	void _members_gui_input(const Ref<InputEvent> &p_event);
	void _fn_name_box_input(const Ref<InputEvent> &p_event);
	void _rename_function(const String &p_name, const String &p_new_name);

	void _create_function_dialog();
	void _create_function();
	void _add_func_input();
	void _remove_func_input(Node *p_node);
	void _deselect_input_names();
	void _add_node_dialog();
	void _node_item_selected();
	void _node_item_unselected();

	void _on_nodes_delete();
	void _on_nodes_duplicate();

	Variant get_drag_data_fw(const Point2 &p_point, Control *p_from);
	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);

	int editing_id;
	int editing_input;

	bool can_swap;
	int data_disconnect_node;
	int data_disconnect_port;

	void _default_value_changed();
	void _default_value_edited(Node *p_button, int p_id, int p_input_port);

	void _menu_option(int p_what);

	void _graph_ofs_changed(const Vector2 &p_ofs);
	void _comment_node_resized(const Vector2 &p_new_size, int p_node);

	int selecting_method_id;
	void _selected_method(const String &p_method, const String &p_type, const bool p_connecting);

	void _draw_color_over_button(Object *obj, Color p_color);
	void _button_resource_previewed(const String &p_path, const Ref<Texture> &p_preview, const Ref<Texture> &p_small_preview, Variant p_ud);

	VisualScriptNode::TypeGuess _guess_output_type(int p_port_action_node, int p_port_action_output, Set<int> &p_visited_nodes);

	void _member_rmb_selected(const Vector2 &p_pos);
	void _member_option(int p_option);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual void add_syntax_highlighter(SyntaxHighlighter *p_highlighter);
	virtual void set_syntax_highlighter(SyntaxHighlighter *p_highlighter);

	virtual void apply_code();
	virtual RES get_edited_resource() const;
	virtual void set_edited_resource(const RES &p_res);
	virtual void enable_editor();
	virtual Vector<String> get_functions();
	virtual void reload_text();
	virtual String get_name();
	virtual Ref<Texture> get_icon();
	virtual bool is_unsaved();
	virtual Variant get_edit_state();
	virtual void set_edit_state(const Variant &p_state);
	virtual void goto_line(int p_line, bool p_with_error = false);
	virtual void set_executing_line(int p_line);
	virtual void clear_executing_line();
	virtual void trim_trailing_whitespace();
	virtual void insert_final_newline();
	virtual void convert_indent_to_spaces();
	virtual void convert_indent_to_tabs();
	virtual void ensure_focus();
	virtual void tag_saved_version();
	virtual void reload(bool p_soft);
	virtual void get_breakpoints(List<int> *p_breakpoints);
	virtual void add_callback(const String &p_function, PoolStringArray p_args);
	virtual void update_settings();
	virtual bool show_members_overview();
	virtual void set_debugger_active(bool p_active);
	virtual void set_tooltip_request_func(String p_method, Object *p_obj);
	virtual Control *get_edit_menu();
	virtual void clear_edit_menu();
	virtual bool can_lose_focus_on_node_selection() { return false; }
	virtual void validate();

	static void register_editor();

	static void free_clipboard();

	VisualScriptEditor();
	~VisualScriptEditor();
};

// Singleton
class _VisualScriptEditor : public Object {
	GDCLASS(_VisualScriptEditor, Object);

	friend class VisualScriptLanguage;

protected:
	static void _bind_methods();
	static _VisualScriptEditor *singleton;

	static Map<String, RefPtr> custom_nodes;
	static Ref<VisualScriptNode> create_node_custom(const String &p_name);

public:
	static _VisualScriptEditor *get_singleton() { return singleton; }

	void add_custom_node(const String &p_name, const String &p_category, const Ref<Script> &p_script);
	void remove_custom_node(const String &p_name, const String &p_category);

	_VisualScriptEditor();
	~_VisualScriptEditor();
};
#endif

#endif // VISUALSCRIPT_EDITOR_H
