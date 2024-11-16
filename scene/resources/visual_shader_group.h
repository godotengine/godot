/**************************************************************************/
/*  visual_shader_group.h                                                 */
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

#ifndef VISUAL_SHADER_GROUP_H
#define VISUAL_SHADER_GROUP_H

#include "scene/gui/dialogs.h"
#include "scene/resources/visual_shader.h"

class VisualShaderGroup : public Resource {
	GDCLASS(VisualShaderGroup, Resource);

	inline static int NODE_ID_GROUP_INPUT = 0;
	inline static int NODE_ID_GROUP_OUTPUT = 1;

public:
	struct Port {
		VisualShaderNode::PortType type = VisualShaderNode::PortType::PORT_TYPE_MAX;
		String name;
	};

private:
	String group_name;

	// TODO: Why does this need to be a HashMap? (copied from Expression node)
	HashMap<int, Port> input_ports;
	HashMap<int, Port> output_ports;

	Ref<ShaderGraph> graph;

	mutable SafeFlag dirty;
	String code;
	String global_code;

protected:
	static void _bind_methods();

	void _queue_update();
	void _update_group();

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	Ref<ShaderGraph> get_graph() const;
	String get_code();
	String get_global_code();

	void set_group_name(const String &p_name);
	String get_group_name() const;

	// TODO: Make private?
	String _validate_port_name(const String &p_port_name, int p_port_id, bool p_output) const;
	String _validate_group_name(const String &p_name) const;

	void add_input_port(int p_id, VisualShaderNode::PortType p_type, const String &p_name);
	void set_input_port_name(int p_id, const String &p_name);
	void set_input_port_type(int p_id, VisualShaderNode::PortType p_type);
	Port get_input_port(int p_id) const;
	// TODO: Maybe replace this method with get_input_port_count(...)
	Vector<Port> get_input_ports() const;
	void remove_input_port(int p_id);

	void add_output_port(int p_id, VisualShaderNode::PortType p_type, const String &p_name);
	void set_output_port_name(int p_id, const String &p_name);
	void set_output_port_type(int p_id, VisualShaderNode::PortType p_type);
	Port get_output_port(int p_id) const;
	// TODO: Maybe replace this method with get_output_port_count(...)
	Vector<Port> get_output_ports() const;
	void remove_output_port(int p_id);

	void add_node(const Ref<VisualShaderNode> &p_node, const Vector2 &p_position, int p_id);
	void set_node_position(int p_id, const Vector2 &p_position);
	Vector2 get_node_position(int p_id) const;
	Ref<VisualShaderNode> get_node(int p_id) const;

	Vector<int> get_node_ids() const;
	int get_valid_node_id() const;
	int find_node_id(const Ref<VisualShaderNode> &p_node) const;
	void remove_node(int p_id);
	void replace_node(int p_id, const StringName &p_new_class);

	// TODO: Rename this method and evaluate whether it is necessary.
	bool are_nodes_connected(int p_from_node, int p_from_port, int p_to_node, int p_to_port) const;

	bool is_nodes_connected_relatively(int p_node, int p_target) const;
	bool can_connect_nodes(int p_from_node, int p_from_port, int p_to_node, int p_to_port) const;
	Error connect_nodes(int p_from_node, int p_from_port, int p_to_node, int p_to_port);
	void disconnect_nodes(int p_from_node, int p_from_port, int p_to_node, int p_to_port);
	void connect_nodes_forced(int p_from_node, int p_from_port, int p_to_node, int p_to_port);
	bool is_port_types_compatible(int p_a, int p_b) const;

	void attach_node_to_frame(int p_node, int p_frame);
	void detach_node_from_frame(int p_node);

	String get_reroute_parameter_name(int p_reroute_node) const;

	// TODO: Maybe change this method to use a return type.
	void get_node_connections(List<ShaderGraph::Connection> *r_connections) const;
	// TODO: Implement?
	String generate_preview_shader(int p_node, int p_port, Vector<ShaderGraph::DefaultTextureParam> &r_default_tex_params) const;

	// TODO: Implement?
	String validate_parameter_name(const String &p_name, const Ref<VisualShaderNodeParameter> &p_parameter) const;

	VisualShaderGroup();
};

class VisualShaderNodeGroup : public VisualShaderNode {
	GDCLASS(VisualShaderNodeGroup, VisualShaderNode);

	Ref<VisualShaderGroup> group;

	// For validation.
	ShaderGraph::Type shader_type = ShaderGraph::Type::TYPE_MAX; // TYPE_MAX when used in a VisualShaderGroup itself.
	Shader::Mode shader_mode = Shader::Mode::MODE_MAX; // MODE_MAX when used in a VisualShaderGroup itself.

	void _emit_changed();

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const override;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual bool is_show_prop_names() const override;
	virtual Vector<StringName> get_editable_properties() const override;
	virtual bool is_use_prop_slots() const override;

	void set_group(const Ref<VisualShaderGroup> &p_group);
	Ref<VisualShaderGroup> get_group() const;

	void set_shader_type(ShaderGraph::Type p_type);
	void set_shader_mode(Shader::Mode p_mode);

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;
	// virtual String generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const override;
	String generate_group_function(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const;
	virtual bool is_output_port_expandable(int p_port) const override;

	virtual Category get_category() const override { return CATEGORY_SPECIAL; }

	VisualShaderNodeGroup();
};

class VisualShaderNodeGroupInput : public VisualShaderNode {
	GDCLASS(VisualShaderNodeGroupInput, VisualShaderNode);

	// TODO: Possibly dangerous, but it is necessary for now since we don't have a proper weak reference.
	VisualShaderGroup *group = nullptr;

	// struct Port {
	// 	PortType type = PortType::PORT_TYPE_MAX;
	// 	const char *name;
	// 	const char *string;
	// };

	// static const Port preview_ports[];

public:
	void set_group(VisualShaderGroup *p_group);
	VisualShaderGroup *get_group() const;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;
	virtual bool is_output_port_expandable(int p_port) const override;

	virtual String get_caption() const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	void set_input_name(String p_name);
	String get_input_name() const;
	String get_input_real_name() const;

	int get_input_index_count() const;
	PortType get_input_index_type(int p_index) const;
	String get_input_index_name(int p_index) const;

	PortType get_input_type_by_name(String p_name) const;

	virtual Vector<StringName> get_editable_properties() const override;

	virtual Category get_category() const override { return CATEGORY_INPUT; }

	VisualShaderNodeGroupInput();
};

class VisualShaderNodeGroupOutput : public VisualShaderNode {
	GDCLASS(VisualShaderNodeGroupOutput, VisualShaderNode);

	// TODO: Possibly dangerous, but it is necessary for now since we don't have a proper weak reference.
	VisualShaderGroup *group = nullptr;

	// struct Port {
	// 	PortType type = PortType::PORT_TYPE_MAX;
	// 	const char *name;
	// 	const char *string;
	// };

	// static const Port ports[];

public:
	void set_group(VisualShaderGroup *p_group);
	VisualShaderGroup *get_group() const;

	virtual int get_input_port_count() const override;
	virtual PortType get_input_port_type(int p_port) const override;
	virtual String get_input_port_name(int p_port) const override;
	Variant get_input_port_default_value(int p_port) const;

	virtual int get_output_port_count() const override;
	virtual PortType get_output_port_type(int p_port) const override;
	virtual String get_output_port_name(int p_port) const override;

	virtual bool is_port_separator(int p_index) const override;

	virtual String get_caption() const override;

	virtual String generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview = false) const override;

	virtual Category get_category() const override { return CATEGORY_OUTPUT; }

	VisualShaderNodeGroupOutput();
};

class Button;
class ItemList;
class OptionButton;
class LineEdit;

class VisualShaderGroupPortsDialog : public AcceptDialog {
	GDCLASS(VisualShaderGroupPortsDialog, AcceptDialog);

	VisualShaderGroup *group = nullptr;
	bool edit_inputs = false; // Determines whether the dialog is used for input or output ports.

	Button *add_port_btn = nullptr;
	Button *remove_port_btn = nullptr;

	ItemList *port_item_list = nullptr;

	LineEdit *name_edit = nullptr;
	OptionButton *port_type_optbtn = nullptr;

	void _add_port();
	void _update_editor_for_port(int p_idx);
	void _remove_port();

	void _on_port_item_selected(int p_idx);
	void _on_port_name_changed(const String &p_name);
	void _on_port_type_changed(int p_idx);
	void _on_dialog_about_to_popup();

	// TODO: Update graph on exit.
public:
	void set_dialog_mode(bool p_edit_inputs);
	void set_group(VisualShaderGroup *p_group);

	VisualShaderGroupPortsDialog();
};

#endif // VISUAL_SHADER_GROUP_H
