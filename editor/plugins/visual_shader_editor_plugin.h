/*************************************************************************/
/*  visual_shader_editor_plugin.h                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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
#include "editor/plugins/curve_editor_plugin.h"
#include "editor/property_editor.h"
#include "scene/gui/button.h"
#include "scene/gui/graph_edit.h"
#include "scene/gui/popup.h"
#include "scene/gui/tree.h"
#include "scene/resources/visual_shader.h"

class VisualShaderNodePlugin : public Reference {
	GDCLASS(VisualShaderNodePlugin, Reference);

protected:
	static void _bind_methods();

public:
	virtual Control *create_editor(const Ref<Resource> &p_parent_resource, const Ref<VisualShaderNode> &p_node);
};

class VisualShaderGraphPlugin : public Reference {
	GDCLASS(VisualShaderGraphPlugin, Reference);

private:
	struct InputPort {
		Button *default_input_button = nullptr;
	};

	struct Port {
		TextureButton *preview_button = nullptr;
	};

	struct Link {
		VisualShader::Type type = VisualShader::Type::TYPE_MAX;
		VisualShaderNode *visual_node = nullptr;
		GraphNode *graph_node = nullptr;
		bool preview_visible = false;
		int preview_pos = 0;
		Map<int, InputPort> input_ports;
		Map<int, Port> output_ports;
		VBoxContainer *preview_box = nullptr;
		LineEdit *uniform_name = nullptr;
		OptionButton *const_op = nullptr;
		CodeEdit *expression_edit = nullptr;
		CurveEditor *curve_editor = nullptr;
	};

	Ref<VisualShader> visual_shader;
	Map<int, Link> links;
	List<VisualShader::Connection> connections;
	bool dirty = false;

	Color vector_expanded_color[3];

protected:
	static void _bind_methods();

public:
	void register_shader(VisualShader *p_visual_shader);
	void set_connections(List<VisualShader::Connection> &p_connections);
	void register_link(VisualShader::Type p_type, int p_id, VisualShaderNode *p_visual_node, GraphNode *p_graph_node);
	void register_output_port(int p_id, int p_port, TextureButton *p_button);
	void register_uniform_name(int p_id, LineEdit *p_uniform_name);
	void register_default_input_button(int p_node_id, int p_port_id, Button *p_button);
	void register_constant_option_btn(int p_node_id, OptionButton *p_button);
	void register_expression_edit(int p_node_id, CodeEdit *p_expression_edit);
	void register_curve_editor(int p_node_id, CurveEditor *p_curve_editor);
	void clear_links();
	void set_shader_type(VisualShader::Type p_type);
	bool is_preview_visible(int p_id) const;
	bool is_dirty() const;
	void make_dirty(bool p_enabled);
	void update_node(VisualShader::Type p_type, int p_id);
	void update_node_deferred(VisualShader::Type p_type, int p_node_id);
	void add_node(VisualShader::Type p_type, int p_id);
	void remove_node(VisualShader::Type p_type, int p_id);
	void connect_nodes(VisualShader::Type p_type, int p_from_node, int p_from_port, int p_to_node, int p_to_port);
	void disconnect_nodes(VisualShader::Type p_type, int p_from_node, int p_from_port, int p_to_node, int p_to_port);
	void show_port_preview(VisualShader::Type p_type, int p_node_id, int p_port_id);
	void set_node_position(VisualShader::Type p_type, int p_id, const Vector2 &p_position);
	void set_node_size(VisualShader::Type p_type, int p_id, const Vector2 &p_size);
	void refresh_node_ports(VisualShader::Type p_type, int p_node);
	void set_input_port_default_value(VisualShader::Type p_type, int p_node_id, int p_port_id, Variant p_value);
	void update_uniform_refs();
	void set_uniform_name(VisualShader::Type p_type, int p_node_id, const String &p_name);
	void update_curve(int p_node_id);
	void update_constant(VisualShader::Type p_type, int p_node_id);
	void set_expression(VisualShader::Type p_type, int p_node_id, const String &p_expression);
	int get_constant_index(float p_constant) const;
	void update_node_size(int p_node_id);
	void update_theme();
	VisualShader::Type get_shader_type() const;

	VisualShaderGraphPlugin();
	~VisualShaderGraphPlugin();
};

class VisualShaderEditor : public VBoxContainer {
	GDCLASS(VisualShaderEditor, VBoxContainer);
	friend class VisualShaderGraphPlugin;

	CustomPropertyEditor *property_editor;
	int editing_node;
	int editing_port;

	Ref<VisualShader> visual_shader;
	GraphEdit *graph;
	Button *add_node;
	Button *preview_shader;

	OptionButton *edit_type = nullptr;
	OptionButton *edit_type_standart;
	OptionButton *edit_type_particles;
	OptionButton *edit_type_sky;

	PanelContainer *error_panel;
	Label *error_label;

	bool pending_update_preview;
	bool shader_error;
	Window *preview_window;
	VBoxContainer *preview_vbox;
	CodeEdit *preview_text;
	Ref<CodeHighlighter> syntax_highlighter;
	Label *error_text;

	UndoRedo *undo_redo;
	Point2 saved_node_pos;
	bool saved_node_pos_dirty;

	ConfirmationDialog *members_dialog;
	PopupMenu *popup_menu;
	MenuButton *tools;

	PopupPanel *comment_title_change_popup = nullptr;
	LineEdit *comment_title_change_edit = nullptr;

	PopupPanel *comment_desc_change_popup = nullptr;
	TextEdit *comment_desc_change_edit = nullptr;

	bool preview_first = true;
	bool preview_showed = false;

	enum ShaderModeFlags {
		MODE_FLAGS_SPATIAL_CANVASITEM = 1,
		MODE_FLAGS_SKY = 2,
		MODE_FLAGS_PARTICLES = 4
	};

	int mode = MODE_FLAGS_SPATIAL_CANVASITEM;

	enum TypeFlags {
		TYPE_FLAGS_VERTEX = 1,
		TYPE_FLAGS_FRAGMENT = 2,
		TYPE_FLAGS_LIGHT = 4,
	};

	enum ParticlesTypeFlags {
		TYPE_FLAGS_EMIT = 1,
		TYPE_FLAGS_PROCESS = 2,
		TYPE_FLAGS_END = 4
	};

	enum SkyTypeFlags {
		TYPE_FLAGS_SKY = 1,
	};

	enum ToolsMenuOptions {
		EXPAND_ALL,
		COLLAPSE_ALL
	};

	enum NodeMenuOptions {
		ADD,
		SEPARATOR, // ignore
		COPY,
		PASTE,
		DELETE,
		DUPLICATE,
		SEPARATOR2, // ignore
		CONVERT_CONSTANTS_TO_UNIFORMS,
		CONVERT_UNIFORMS_TO_CONSTANTS,
		SEPARATOR3, // ignore
		SET_COMMENT_TITLE,
		SET_COMMENT_DESCRIPTION,
	};

	Tree *members;
	AcceptDialog *alert;
	LineEdit *node_filter;
	RichTextLabel *node_desc;
	Label *highend_label;

	void _tools_menu_option(int p_idx);
	void _show_members_dialog(bool at_mouse_pos);

	void _update_graph();

	struct AddOption {
		String name;
		String category;
		String type;
		String description;
		int sub_func = 0;
		String sub_func_str;
		Ref<Script> script;
		int mode = 0;
		int return_type = 0;
		int func = 0;
		float value = 0;
		bool highend = false;
		bool is_custom = false;
		int temp_idx = 0;

		AddOption(const String &p_name = String(), const String &p_category = String(), const String &p_sub_category = String(), const String &p_type = String(), const String &p_description = String(), int p_sub_func = -1, int p_return_type = -1, int p_mode = -1, int p_func = -1, float p_value = -1, bool p_highend = false) {
			name = p_name;
			type = p_type;
			category = p_category + "/" + p_sub_category;
			description = p_description;
			sub_func = p_sub_func;
			return_type = p_return_type;
			mode = p_mode;
			func = p_func;
			value = p_value;
			highend = p_highend;
			is_custom = false;
		}

		AddOption(const String &p_name, const String &p_category, const String &p_sub_category, const String &p_type, const String &p_description, const String &p_sub_func, int p_return_type = -1, int p_mode = -1, int p_func = -1, float p_value = -1, bool p_highend = false) {
			name = p_name;
			type = p_type;
			category = p_category + "/" + p_sub_category;
			description = p_description;
			sub_func = 0;
			sub_func_str = p_sub_func;
			return_type = p_return_type;
			mode = p_mode;
			func = p_func;
			value = p_value;
			highend = p_highend;
			is_custom = false;
		}
	};
	struct _OptionComparator {
		_FORCE_INLINE_ bool operator()(const AddOption &a, const AddOption &b) const {
			return a.category.count("/") > b.category.count("/") || (a.category + "/" + a.name).naturalnocasecmp_to(b.category + "/" + b.name) < 0;
		}
	};

	Vector<AddOption> add_options;
	int cubemap_node_option_idx;
	int texture2d_node_option_idx;
	int texture2d_array_node_option_idx;
	int texture3d_node_option_idx;
	int custom_node_option_idx;
	int curve_node_option_idx;
	List<String> keyword_list;

	List<VisualShaderNodeUniformRef> uniform_refs;

	void _draw_color_over_button(Object *obj, Color p_color);

	void _setup_node(VisualShaderNode *p_node, int p_op_idx);
	void _add_node(int p_idx, int p_op_idx = -1, String p_resource_path = "", int p_node_idx = -1);
	void _update_options_menu();
	void _set_mode(int p_which);

	void _show_preview_text();
	void _preview_close_requested();
	void _preview_size_changed();
	void _update_preview();
	String _get_description(int p_idx);

	static VisualShaderEditor *singleton;

	struct DragOp {
		VisualShader::Type type = VisualShader::Type::TYPE_MAX;
		int node = 0;
		Vector2 from;
		Vector2 to;
	};
	List<DragOp> drag_buffer;
	bool drag_dirty = false;
	void _node_dragged(const Vector2 &p_from, const Vector2 &p_to, int p_node);
	void _nodes_dragged();
	bool updating;

	void _connection_request(const String &p_from, int p_from_index, const String &p_to, int p_to_index);
	void _disconnection_request(const String &p_from, int p_from_index, const String &p_to, int p_to_index);

	void _scroll_changed(const Vector2 &p_scroll);
	void _node_selected(Object *p_node);

	void _delete_nodes(int p_type, const List<int> &p_nodes);
	void _delete_node_request(int p_type, int p_node);
	void _delete_nodes_request();

	void _removed_from_graph();

	void _node_changed(int p_id);

	void _edit_port_default_input(Object *p_button, int p_node, int p_port);
	void _port_edited();

	int to_node;
	int to_slot;
	int from_node;
	int from_slot;

	Set<int> selected_constants;
	Set<int> selected_uniforms;
	int selected_comment = -1;

	void _convert_constants_to_uniforms(bool p_vice_versa);
	void _replace_node(VisualShader::Type p_type_id, int p_node_id, const StringName &p_from, const StringName &p_to);
	void _update_constant(VisualShader::Type p_type_id, int p_node_id, Variant p_var, int p_preview_port);
	void _update_uniform(VisualShader::Type p_type_id, int p_node_id, Variant p_var, int p_preview_port);

	void _connection_to_empty(const String &p_from, int p_from_slot, const Vector2 &p_release_position);
	void _connection_from_empty(const String &p_to, int p_to_slot, const Vector2 &p_release_position);

	void _comment_title_popup_show(const Point2 &p_position, int p_node_id);
	void _comment_title_popup_hide();
	void _comment_title_popup_focus_out();
	void _comment_title_text_changed(const String &p_new_text);
	void _comment_title_text_entered(const String &p_new_text);

	void _comment_desc_popup_show(const Point2 &p_position, int p_node_id);
	void _comment_desc_popup_hide();
	void _comment_desc_confirm();
	void _comment_desc_text_changed();

	void _uniform_line_edit_changed(const String &p_text, int p_node_id);
	void _uniform_line_edit_focus_out(Object *line_edit, int p_node_id);

	void _port_name_focus_out(Object *line_edit, int p_node_id, int p_port_id, bool p_output);

	void _dup_copy_nodes(int p_type, List<int> &r_nodes, Set<int> &r_excluded);
	void _dup_update_excluded(int p_type, Set<int> &r_excluded);
	void _dup_paste_nodes(int p_type, int p_pasted_type, List<int> &r_nodes, Set<int> &r_excluded, const Vector2 &p_offset, bool p_select);

	void _duplicate_nodes();

	Vector2 selection_center;
	int copy_type; // shader type
	List<int> copy_nodes_buffer;
	Set<int> copy_nodes_excluded_buffer;

	void _clear_buffer();
	void _copy_nodes();
	void _paste_nodes(bool p_use_custom_position = false, const Vector2 &p_custom_position = Vector2());

	Vector<Ref<VisualShaderNodePlugin>> plugins;
	Ref<VisualShaderGraphPlugin> graph_plugin;

	void _mode_selected(int p_id);

	void _input_select_item(Ref<VisualShaderNodeInput> input, String name);
	void _uniform_select_item(Ref<VisualShaderNodeUniformRef> p_uniform, String p_name);

	void _float_constant_selected(int p_index, int p_node);

	VisualShader::Type get_current_shader_type() const;

	void _add_input_port(int p_node, int p_port, int p_port_type, const String &p_name);
	void _remove_input_port(int p_node, int p_port);
	void _change_input_port_type(int p_type, int p_node, int p_port);
	void _change_input_port_name(const String &p_text, Object *p_line_edit, int p_node, int p_port);

	void _add_output_port(int p_node, int p_port, int p_port_type, const String &p_name);
	void _remove_output_port(int p_node, int p_port);
	void _change_output_port_type(int p_type, int p_node, int p_port);
	void _change_output_port_name(const String &p_text, Object *p_line_edit, int p_node, int p_port);
	void _expand_output_port(int p_node, int p_port, bool p_expand);

	void _expression_focus_out(Object *code_edit, int p_node);

	void _set_node_size(int p_type, int p_node, const Size2 &p_size);
	void _node_resized(const Vector2 &p_new_size, int p_type, int p_node);

	void _preview_select_port(int p_node, int p_port);
	void _graph_gui_input(const Ref<InputEvent> &p_event);

	void _member_filter_changed(const String &p_text);
	void _sbox_input(const Ref<InputEvent> &p_ie);
	void _member_selected();
	void _member_unselected();
	void _member_create();
	void _member_cancel();

	Vector2 menu_point;
	void _node_menu_id_pressed(int p_idx);

	Variant get_drag_data_fw(const Point2 &p_point, Control *p_from);
	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);

	bool _is_available(int p_mode);
	void _update_created_node(GraphNode *node);
	void _update_uniforms(bool p_update_refs);
	void _update_uniform_refs(Set<String> &p_names);

	void _visibility_changed();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void update_custom_nodes();
	void add_plugin(const Ref<VisualShaderNodePlugin> &p_plugin);
	void remove_plugin(const Ref<VisualShaderNodePlugin> &p_plugin);

	static VisualShaderEditor *get_singleton() { return singleton; }
	VisualShaderGraphPlugin *get_graph_plugin() { return graph_plugin.ptr(); }

	void clear_custom_types();
	void add_custom_type(const String &p_name, const Ref<Script> &p_script, const String &p_description, int p_return_icon_type, const String &p_category, bool p_highend);

	virtual Size2 get_minimum_size() const override;
	void edit(VisualShader *p_visual_shader);
	VisualShaderEditor();
};

class VisualShaderEditorPlugin : public EditorPlugin {
	GDCLASS(VisualShaderEditorPlugin, EditorPlugin);

	VisualShaderEditor *visual_shader_editor;
	EditorNode *editor;
	Button *button;

public:
	virtual String get_name() const override { return "VisualShader"; }
	bool has_main_screen() const override { return false; }
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;

	VisualShaderEditorPlugin(EditorNode *p_node);
	~VisualShaderEditorPlugin();
};

class VisualShaderNodePluginDefault : public VisualShaderNodePlugin {
	GDCLASS(VisualShaderNodePluginDefault, VisualShaderNodePlugin);

public:
	virtual Control *create_editor(const Ref<Resource> &p_parent_resource, const Ref<VisualShaderNode> &p_node) override;
};

class EditorPropertyShaderMode : public EditorProperty {
	GDCLASS(EditorPropertyShaderMode, EditorProperty);
	OptionButton *options;

	void _option_selected(int p_which);

protected:
	static void _bind_methods();

public:
	void setup(const Vector<String> &p_options);
	virtual void update_property() override;
	void set_option_button_clip(bool p_enable);
	EditorPropertyShaderMode();
};

class EditorInspectorShaderModePlugin : public EditorInspectorPlugin {
	GDCLASS(EditorInspectorShaderModePlugin, EditorInspectorPlugin);

public:
	virtual bool can_handle(Object *p_object) override;
	virtual void parse_begin(Object *p_object) override;
	virtual bool parse_property(Object *p_object, Variant::Type p_type, const String &p_path, PropertyHint p_hint, const String &p_hint_text, int p_usage, bool p_wide = false) override;
	virtual void parse_end() override;
};

class VisualShaderNodePortPreview : public Control {
	GDCLASS(VisualShaderNodePortPreview, Control);
	Ref<VisualShader> shader;
	VisualShader::Type type = VisualShader::Type::TYPE_MAX;
	int node = 0;
	int port = 0;
	void _shader_changed(); //must regen
protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual Size2 get_minimum_size() const override;
	void setup(const Ref<VisualShader> &p_shader, VisualShader::Type p_type, int p_node, int p_port);
};

class VisualShaderConversionPlugin : public EditorResourceConversionPlugin {
	GDCLASS(VisualShaderConversionPlugin, EditorResourceConversionPlugin);

public:
	virtual String converts_to() const override;
	virtual bool handles(const Ref<Resource> &p_resource) const override;
	virtual Ref<Resource> convert(const Ref<Resource> &p_resource) const override;
};

#endif // VISUAL_SHADER_EDITOR_PLUGIN_H
