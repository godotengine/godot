/*************************************************************************/
/*  visual_shader_editor_plugin.h                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef VISUAL_SHADER_EDITOR_PLUGIN_H
#define VISUAL_SHADER_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "editor/property_editor.h"
#include "scene/gui/button.h"
#include "scene/gui/graph_edit.h"
#include "scene/gui/popup.h"
#include "scene/gui/tree.h"
#include "scene/resources/visual_shader.h"

class VisualShaderNodePlugin : public Reference {

	GDCLASS(VisualShaderNodePlugin, Reference)
protected:
	static void _bind_methods();

public:
	virtual Control *create_editor(const Ref<VisualShaderNode> &p_node);
};

class VisualShaderEditor : public VBoxContainer {

	GDCLASS(VisualShaderEditor, VBoxContainer);

	CustomPropertyEditor *property_editor;
	int editing_node;
	int editing_port;

	Ref<VisualShader> visual_shader;
	GraphEdit *graph;
	ToolButton *add_node;

	OptionButton *edit_type;

	PanelContainer *error_panel;
	Label *error_label;

	UndoRedo *undo_redo;
	Point2 saved_node_pos;
	bool saved_node_pos_dirty;

	ConfirmationDialog *members_dialog;
	MenuButton *tools;

	enum ToolsMenuOptions {
		EXPAND_ALL,
		COLLAPSE_ALL
	};

	Tree *members;
	AcceptDialog *alert;
	LineEdit *node_filter;
	RichTextLabel *node_desc;

	void _tools_menu_option(int p_idx);
	void _show_members_dialog();

	void _update_graph();

	struct AddOption {
		String name;
		String category;
		String sub_category;
		String type;
		String description;
		int sub_func;
		String sub_func_str;
		Ref<Script> script;
		int mode;
		int return_type;

		AddOption(const String &p_name = String(), const String &p_category = String(), const String &p_sub_category = String(), const String &p_type = String(), const String &p_description = String(), int p_sub_func = -1, int p_return_type = -1, int p_mode = -1) {
			name = p_name;
			type = p_type;
			category = p_category;
			sub_category = p_sub_category;
			description = p_description;
			sub_func = p_sub_func;
			return_type = p_return_type;
			mode = p_mode;
		}

		AddOption(const String &p_name, const String &p_category, const String &p_sub_category, const String &p_type, const String &p_description, const String &p_sub_func, int p_return_type = -1, int p_mode = -1) {
			name = p_name;
			type = p_type;
			category = p_category;
			sub_category = p_sub_category;
			description = p_description;
			sub_func_str = p_sub_func;
			return_type = p_return_type;
			mode = p_mode;
		}
	};

	Vector<AddOption> add_options;

	void _draw_color_over_button(Object *obj, Color p_color);

	void _add_node(int p_idx, int p_op_idx = -1);
	void _update_options_menu();

	static VisualShaderEditor *singleton;

	void _node_dragged(const Vector2 &p_from, const Vector2 &p_to, int p_node);
	bool updating;

	void _connection_request(const String &p_from, int p_from_index, const String &p_to, int p_to_index);
	void _disconnection_request(const String &p_from, int p_from_index, const String &p_to, int p_to_index);

	void _scroll_changed(const Vector2 &p_scroll);
	void _node_selected(Object *p_node);

	void _delete_request(int);

	void _removed_from_graph();

	void _node_changed(int p_id);

	void _edit_port_default_input(Object *p_button, int p_node, int p_port);
	void _port_edited();

	void _connection_to_empty(const String &p_from, int p_from_slot, const Vector2 &p_release_position);

	void _line_edit_changed(const String &p_text, Object *line_edit, int p_node_id);
	void _line_edit_focus_out(Object *line_edit, int p_node_id);

	void _duplicate_nodes();

	Vector<Ref<VisualShaderNodePlugin> > plugins;

	void _mode_selected(int p_id);

	void _input_select_item(Ref<VisualShaderNodeInput> input, String name);

	void _preview_select_port(int p_node, int p_port);
	void _input(const Ref<InputEvent> p_event);

	void _member_gui_input(const Ref<InputEvent> p_event);
	void _member_filter_changed(const String &p_text);
	void _member_selected();
	void _member_unselected();
	void _member_create();

	Variant get_drag_data_fw(const Point2 &p_point, Control *p_from);
	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);

	bool _is_available(int p_flags);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void add_plugin(const Ref<VisualShaderNodePlugin> &p_plugin);
	void remove_plugin(const Ref<VisualShaderNodePlugin> &p_plugin);

	static VisualShaderEditor *get_singleton() { return singleton; }

	void add_custom_type(const String &p_name, const String &p_category, const Ref<Script> &p_script);
	void remove_custom_type(const Ref<Script> &p_script);

	virtual Size2 get_minimum_size() const;
	void edit(VisualShader *p_visual_shader);
	VisualShaderEditor();
};

class VisualShaderEditorPlugin : public EditorPlugin {

	GDCLASS(VisualShaderEditorPlugin, EditorPlugin);

	VisualShaderEditor *visual_shader_editor;
	EditorNode *editor;
	Button *button;

public:
	virtual String get_name() const { return "VisualShader"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_object);
	virtual bool handles(Object *p_object) const;
	virtual void make_visible(bool p_visible);

	VisualShaderEditorPlugin(EditorNode *p_node);
	~VisualShaderEditorPlugin();
};

class VisualShaderNodePluginDefault : public VisualShaderNodePlugin {

	GDCLASS(VisualShaderNodePluginDefault, VisualShaderNodePlugin)

public:
	virtual Control *create_editor(const Ref<VisualShaderNode> &p_node);
};

class EditorPropertyShaderMode : public EditorProperty {
	GDCLASS(EditorPropertyShaderMode, EditorProperty)
	OptionButton *options;

	void _option_selected(int p_which);

protected:
	static void _bind_methods();

public:
	void setup(const Vector<String> &p_options);
	virtual void update_property();
	void set_option_button_clip(bool p_enable);
	EditorPropertyShaderMode();
};

class EditorInspectorShaderModePlugin : public EditorInspectorPlugin {
	GDCLASS(EditorInspectorShaderModePlugin, EditorInspectorPlugin)

public:
	virtual bool can_handle(Object *p_object);
	virtual void parse_begin(Object *p_object);
	virtual bool parse_property(Object *p_object, Variant::Type p_type, const String &p_path, PropertyHint p_hint, const String &p_hint_text, int p_usage);
	virtual void parse_end();
};

class VisualShaderNodePortPreview : public Control {
	GDCLASS(VisualShaderNodePortPreview, Control)
	Ref<VisualShader> shader;
	VisualShader::Type type;
	int node;
	int port;
	void _shader_changed(); //must regen
protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual Size2 get_minimum_size() const;
	void setup(const Ref<VisualShader> &p_shader, VisualShader::Type p_type, int p_node, int p_port);
	VisualShaderNodePortPreview();
};

#endif // VISUAL_SHADER_EDITOR_PLUGIN_H
