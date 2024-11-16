/**************************************************************************/
/*  visual_shader_group.cpp                                               */
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

#include "visual_shader_group.h"

#include "core/error/error_macros.h"
#include "core/object/callable_method_pointer.h"
#include "core/string/ustring.h"
#include "core/templates/hash_set.h"
#include "core/templates/vmap.h"
#include "editor/plugins/visual_shader_editor_plugin.h"
#include "scene/gui/box_container.h"
#include "scene/gui/graph_node.h"
#include "scene/gui/item_list.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/option_button.h"
#include "visual_shader_particle_nodes.h"

String VisualShaderGroup::_validate_port_name(const String &p_port_name, int p_port_id, bool p_output) const {
	String port_name = p_port_name;

	if (port_name.is_empty()) {
		return String();
	}

	while (port_name.length() && !is_ascii_alphabet_char(port_name[0])) {
		port_name = port_name.substr(1, port_name.length() - 1);
	}

	if (!port_name.is_empty()) {
		String valid_name;

		for (int i = 0; i < port_name.length(); i++) {
			if (is_ascii_identifier_char(port_name[i])) {
				valid_name += String::chr(port_name[i]);
			} else if (port_name[i] == ' ') {
				valid_name += "_";
			}
		}

		port_name = valid_name;
	} else {
		return String();
	}

	List<String> input_names;
	List<String> output_names;

	for (int i = 0; i < get_input_ports().size(); i++) {
		if (!p_output && i == p_port_id) {
			continue;
		}
		if (port_name == get_input_port(i).name) {
			return String();
		}
	}
	for (int i = 0; i < get_output_ports().size(); i++) {
		if (p_output && i == p_port_id) {
			continue;
		}
		if (port_name == get_output_port(i).name) {
			return String();
		}
	}

	return port_name;
}

String VisualShaderGroup::_validate_group_name(const String &p_name) const {
	String valid_name;

	for (int i = 0; i < p_name.length(); i++) {
		if (is_ascii_identifier_char(p_name[i])) {
			valid_name += String::chr(p_name[i]);
		} else if (p_name[i] == ' ') {
			valid_name += "_";
		}
	}

	return valid_name;
}

void VisualShaderGroup::_bind_methods() {
	// TODO: Bind setters/getters for input/output ports.

	ClassDB::bind_method(D_METHOD("set_group_name", "name"), &VisualShaderGroup::set_group_name);
	ClassDB::bind_method(D_METHOD("get_group_name"), &VisualShaderGroup::get_group_name);

	ClassDB::bind_method(D_METHOD("add_node", "node", "position", "id"), &VisualShaderGroup::add_node);
	ClassDB::bind_method(D_METHOD("get_node", "id"), &VisualShaderGroup::get_node);

	ClassDB::bind_method(D_METHOD("set_node_position", "id", "position"), &VisualShaderGroup::set_node_position);
	ClassDB::bind_method(D_METHOD("get_node_position", "id"), &VisualShaderGroup::get_node_position);

	ClassDB::bind_method(D_METHOD("get_node_list"), &VisualShaderGroup::get_node_ids);
	ClassDB::bind_method(D_METHOD("get_valid_node_id"), &VisualShaderGroup::get_valid_node_id);

	ClassDB::bind_method(D_METHOD("remove_node", "id"), &VisualShaderGroup::remove_node);
	ClassDB::bind_method(D_METHOD("replace_node", "id", "new_class"), &VisualShaderGroup::replace_node);

	ClassDB::bind_method(D_METHOD("is_node_connection", "from_node", "from_port", "to_node", "to_port"), &VisualShaderGroup::are_nodes_connected);
	ClassDB::bind_method(D_METHOD("can_connect_nodes", "from_node", "from_port", "to_node", "to_port"), &VisualShaderGroup::can_connect_nodes);

	ClassDB::bind_method(D_METHOD("connect_nodes", "from_node", "from_port", "to_node", "to_port"), &VisualShaderGroup::connect_nodes);
	ClassDB::bind_method(D_METHOD("disconnect_nodes", "from_node", "from_port", "to_node", "to_port"), &VisualShaderGroup::disconnect_nodes);
	ClassDB::bind_method(D_METHOD("connect_nodes_forced", "from_node", "from_port", "to_node", "to_port"), &VisualShaderGroup::connect_nodes_forced);

	// TODO: Re-add this method.
	// ClassDB::bind_method(D_METHOD("get_node_connections", "type"), &VisualShaderGroup::get_node_connections);

	ClassDB::bind_method(D_METHOD("attach_node_to_frame", "id", "frame"), &VisualShaderGroup::attach_node_to_frame);
	ClassDB::bind_method(D_METHOD("detach_node_from_frame", "id"), &VisualShaderGroup::detach_node_from_frame);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "group_name"), "set_group_name", "get_group_name");
	ADD_PROPERTY_DEFAULT("group_name", "Node group");
}

void VisualShaderGroup::_queue_update() {
	if (dirty.is_set()) {
		return;
	}

	dirty.set();
	callable_mp(this, &VisualShaderGroup::_update_group).call_deferred();
}

void VisualShaderGroup::_update_group() {
	if (!dirty.is_set()) {
		return;
	}

	dirty.clear();

	// TODO: Update group.

	StringBuilder global_code_builder;
	StringBuilder global_code_per_node_builder;
	HashMap<ShaderGraph::Type, StringBuilder> global_code_per_func_builder;
	StringBuilder code_builder;
	Vector<ShaderGraph::DefaultTextureParam> default_tex_params;
	// static const char *shader_mode_str[Shader::MODE_MAX] = { "spatial", "canvas_item", "particles", "sky", "fog" };

	HashSet<StringName> classes;
	HashMap<int, int> insertion_pos;

	String global_expressions;
	HashSet<String> used_parameter_names;
	List<VisualShaderNodeParameter *> parameters;
	List<int> emitters;
	HashMap<int, List<int>> varying_setters;

	// Preprocess nodes.
	int index = 0;
	for (const KeyValue<int, ShaderGraph::Node> &E : graph->nodes) {
		Ref<VisualShaderNodeGlobalExpression> global_expression = E.value.node;
		if (global_expression.is_valid()) {
			String expr = "";
			expr += "// " + global_expression->get_caption() + ":" + itos(index++) + "\n";
			expr += global_expression->generate_global(Shader::MODE_MAX, VisualShader::TYPE_MAX, -1);
			expr = expr.replace("\n", "\n	");
			expr += "\n";
			global_expressions += expr;
		}
		Ref<VisualShaderNodeParameterRef> parameter_ref = E.value.node;
		if (parameter_ref.is_valid()) {
			used_parameter_names.insert(parameter_ref->get_parameter_name());
		}
		Ref<VisualShaderNodeParameter> parameter = E.value.node;
		if (parameter.is_valid()) {
			parameters.push_back(parameter.ptr());
		}
		Ref<VisualShaderNodeParticleEmit> emit_particle = E.value.node;
		if (emit_particle.is_valid()) {
			emitters.push_back(E.key);
		}
	}

	// TODO: Forbid parameters.
	// int idx = 0
	// for (List<VisualShaderNodeParameter *>::Iterator itr = parameters.begin(); itr != parameters.end(); ++itr, ++idx) {
	// 	VisualShaderNodeParameter *parameter = *itr;
	// 	if (used_parameter_names.has(parameter->get_parameter_name())) {
	// 		global_code += parameter->generate_global(get_mode(), Type(idx), -1);
	// 		const_cast<VisualShaderNodeParameter *>(parameter)->set_global_code_generated(true);
	// 	} else {
	// 		const_cast<VisualShaderNodeParameter *>(parameter)->set_global_code_generated(false);
	// 	}
	// }
	HashMap<int, String> code_map;
	HashSet<int> empty_funcs;
	VMap<ShaderGraph::ConnectionKey, const List<ShaderGraph::Connection>::Element *> input_connections;
	VMap<ShaderGraph::ConnectionKey, const List<ShaderGraph::Connection>::Element *> output_connections;

	StringBuilder group_code;
	HashSet<int> processed;

	for (const List<ShaderGraph::Connection>::Element *E = graph->connections.front(); E; E = E->next()) {
		ShaderGraph::ConnectionKey from_key;
		from_key.node = E->get().from_node;
		from_key.port = E->get().from_port;

		output_connections.insert(from_key, E);

		ShaderGraph::ConnectionKey to_key;
		to_key.node = E->get().to_node;
		to_key.port = E->get().to_port;

		input_connections.insert(to_key, E);
	}

	Error err = graph->_write_node(&global_code_builder, &global_code_per_node_builder, &global_code_per_func_builder, group_code, default_tex_params, input_connections, output_connections, NODE_ID_GROUP_OUTPUT, processed, false, classes);
	ERR_FAIL_COND(err != OK);

	// TODO: Figure out why this needs to be separately.
	for (int &E : emitters) {
		err = graph->_write_node(&global_code_builder, &global_code_per_node_builder, &global_code_per_func_builder, group_code, default_tex_params, input_connections, output_connections, E, processed, false, classes);
		ERR_FAIL_COND(err != OK);
	}

	// TODO: Use concept of previous code to determine whether to fire the changed signal?

	code_builder += "// Group content: " + group_name + "\n";
	code_builder += group_code;

	global_code_builder.append(global_code_per_node_builder);
	global_code_builder.append(global_expressions);
	global_code = global_code_builder.as_string();
	// TODO: Insert global code per func
	code = code_builder.as_string();
}

bool VisualShaderGroup::_set(const StringName &p_name, const Variant &p_value) {
	if (p_name == "input_ports") {
		input_ports.clear();
		const Array &ports = p_value;
		for (int i = 0; i < ports.size(); i++) {
			const Dictionary &port = ports[i];
			Port p;
			p.type = (VisualShaderNode::PortType)(int)port["type"];
			p.name = port["name"];
			input_ports[port["id"]] = p;
		}
		emit_changed();
		return true;
	} else if (p_name == "output_ports") {
		output_ports.clear();
		const Array &ports = p_value;
		for (int i = 0; i < ports.size(); i++) {
			const Dictionary &port = ports[i];
			Port p;
			p.type = (VisualShaderNode::PortType)(int)port["type"];
			p.name = port["name"];
			output_ports[port["id"]] = p;
		}
		emit_changed();
		return true;
	}
	return graph->_set(p_name, p_value);
}

bool VisualShaderGroup::_get(const StringName &p_name, Variant &r_ret) const {
	if (p_name == "input_ports") {
		Array ports;
		for (const KeyValue<int, Port> &E : input_ports) {
			Dictionary port;
			port["id"] = E.key;
			port["type"] = E.value.type;
			port["name"] = E.value.name;
			ports.push_back(port);
		}
		r_ret = ports;
		return true;
	} else if (p_name == "output_ports") {
		Array ports;
		for (const KeyValue<int, Port> &E : output_ports) {
			Dictionary port;
			port["id"] = E.key;
			port["type"] = E.value.type;
			port["name"] = E.value.name;
			ports.push_back(port);
		}
		r_ret = ports;
		return true;
	}
	return graph->_get(p_name, r_ret);
}

void VisualShaderGroup::_get_property_list(List<PropertyInfo> *p_list) const {
	graph->_get_property_list(p_list);
	// TODO: Should these properties be added with their own getters/setters?
	p_list->push_back(PropertyInfo(Variant::ARRAY, "input_ports", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
	p_list->push_back(PropertyInfo(Variant::ARRAY, "output_ports", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
}

Ref<ShaderGraph> VisualShaderGroup::get_graph() const {
	return graph;
}

String VisualShaderGroup::get_code() {
	if (dirty.is_set()) {
		_update_group();
	}
	return code;
}

String VisualShaderGroup::get_global_code() {
	if (dirty.is_set()) {
		_update_group();
	}
	return global_code;
}

void VisualShaderGroup::set_group_name(const String &p_name) {
	const String valid_name = _validate_group_name(p_name);

	if (group_name == p_name || valid_name.is_empty()) {
		return;
	}

	group_name = p_name; // Don't use valid_name here, since we want to keep the original name.
	emit_changed();
}

String VisualShaderGroup::get_group_name() const {
	return group_name;
}

void VisualShaderGroup::add_input_port(int p_id, VisualShaderNode::PortType p_type, const String &p_name) {
	const String valid_name = _validate_port_name(p_name, p_id, false);

	if (valid_name.is_empty()) {
		return;
	}

	input_ports[p_id] = Port{ p_type, valid_name };
	emit_changed();
}

void VisualShaderGroup::set_input_port_name(int p_id, const String &p_name) {
	ERR_FAIL_COND(!input_ports.has(p_id));

	const String valid_name = _validate_port_name(p_name, p_id, false);

	if (valid_name.is_empty()) {
		return;
	}

	input_ports[p_id].name = valid_name;
	emit_changed();
}

void VisualShaderGroup::set_input_port_type(int p_id, VisualShaderNode::PortType p_type) {
	ERR_FAIL_COND(!input_ports.has(p_id));

	input_ports[p_id].type = p_type;
	emit_changed();
}

VisualShaderGroup::Port VisualShaderGroup::get_input_port(int p_id) const {
	return input_ports[p_id];
}

Vector<VisualShaderGroup::Port> VisualShaderGroup::get_input_ports() const {
	Vector<Port> ports;
	for (const KeyValue<int, Port> &E : input_ports) {
		ports.push_back(E.value);
	}
	return ports;
}

void VisualShaderGroup::remove_input_port(int p_id) {
	input_ports.erase(p_id);
	emit_changed();
}

void VisualShaderGroup::add_output_port(int p_id, VisualShaderNode::PortType p_type, const String &p_name) {
	const String valid_name = _validate_port_name(p_name, p_id, true);

	if (valid_name.is_empty()) {
		return;
	}

	output_ports[p_id] = Port{ p_type, valid_name };
	emit_changed();
}

void VisualShaderGroup::set_output_port_name(int p_id, const String &p_name) {
	ERR_FAIL_COND(!output_ports.has(p_id));

	const String valid_name = _validate_port_name(p_name, p_id, true);

	if (valid_name.is_empty()) {
		return;
	}

	output_ports[p_id].name = valid_name;
	emit_changed();
}

void VisualShaderGroup::set_output_port_type(int p_id, VisualShaderNode::PortType p_type) {
	ERR_FAIL_COND(!output_ports.has(p_id));

	output_ports[p_id].type = p_type;
	emit_changed();
}

VisualShaderGroup::Port VisualShaderGroup::get_output_port(int p_id) const {
	return output_ports[p_id];
}

Vector<VisualShaderGroup::Port> VisualShaderGroup::get_output_ports() const {
	Vector<Port> ports;
	for (const KeyValue<int, Port> &E : output_ports) {
		ports.push_back(E.value);
	}
	return ports;
}

void VisualShaderGroup::remove_output_port(int p_id) {
	output_ports.erase(p_id);
	emit_changed();
}

void VisualShaderGroup::add_node(const Ref<VisualShaderNode> &p_node, const Vector2 &p_position, int p_id) {
	graph->add_node(p_node, p_position, p_id);
}

void VisualShaderGroup::set_node_position(int p_id, const Vector2 &p_position) {
	graph->set_node_position(p_id, p_position);
}

Vector2 VisualShaderGroup::get_node_position(int p_id) const {
	return graph->get_node_position(p_id);
}

Ref<VisualShaderNode> VisualShaderGroup::get_node(int p_id) const {
	return graph->get_node(p_id);
}

Vector<int> VisualShaderGroup::get_node_ids() const {
	return graph->get_node_ids();
}

int VisualShaderGroup::get_valid_node_id() const {
	return graph->get_valid_node_id();
}

int VisualShaderGroup::find_node_id(const Ref<VisualShaderNode> &p_node) const {
	return graph->find_node_id(p_node);
}

void VisualShaderGroup::remove_node(int p_id) {
	graph->remove_node(p_id);
}

void VisualShaderGroup::replace_node(int p_id, const StringName &p_new_class) {
	graph->replace_node(p_id, p_new_class);
}

bool VisualShaderGroup::are_nodes_connected(int p_from_node, int p_from_port, int p_to_node, int p_to_port) const {
	return graph->are_nodes_connected(p_from_node, p_from_port, p_to_node, p_to_port);
}

bool VisualShaderGroup::is_nodes_connected_relatively(int p_node, int p_target) const {
	return graph->is_nodes_connected_relatively(p_node, p_target);
}

bool VisualShaderGroup::can_connect_nodes(int p_from_node, int p_from_port, int p_to_node, int p_to_port) const {
	return graph->can_connect_nodes(p_from_node, p_from_port, p_to_node, p_to_port);
}

Error VisualShaderGroup::connect_nodes(int p_from_node, int p_from_port, int p_to_node, int p_to_port) {
	return graph->connect_nodes(p_from_node, p_from_port, p_to_node, p_to_port);
}

void VisualShaderGroup::disconnect_nodes(int p_from_node, int p_from_port, int p_to_node, int p_to_port) {
	graph->disconnect_nodes(p_from_node, p_from_port, p_to_node, p_to_port);
}

void VisualShaderGroup::connect_nodes_forced(int p_from_node, int p_from_port, int p_to_node, int p_to_port) {
	graph->connect_nodes_forced(p_from_node, p_from_port, p_to_node, p_to_port);
}

bool VisualShaderGroup::is_port_types_compatible(int p_a, int p_b) const {
	return graph->is_port_types_compatible(p_a, p_b);
}

void VisualShaderGroup::attach_node_to_frame(int p_node, int p_frame) {
	graph->attach_node_to_frame(p_node, p_frame);
}

void VisualShaderGroup::detach_node_from_frame(int p_node) {
	graph->detach_node_from_frame(p_node);
}

String VisualShaderGroup::get_reroute_parameter_name(int p_reroute_node) const {
	return graph->get_reroute_parameter_name(p_reroute_node);
}

void VisualShaderGroup::get_node_connections(List<ShaderGraph::Connection> *r_connections) const {
	graph->get_node_connections(r_connections);
}

VisualShaderGroup::VisualShaderGroup() {
	dirty.set();

	graph.instantiate();
	graph->connect("graph_changed", callable_mp(this, &VisualShaderGroup::_queue_update));

	Ref<VisualShaderNodeGroupInput> input_node;
	input_node.instantiate();
	input_node->set_group(this);
	graph->nodes[NODE_ID_GROUP_INPUT].node = input_node;
	graph->nodes[NODE_ID_GROUP_INPUT].position = Vector2(0, 150);

	Ref<VisualShaderNodeGroupOutput> output_node;
	output_node.instantiate();
	output_node->set_group(this);
	graph->nodes[NODE_ID_GROUP_OUTPUT].node = output_node;
	graph->nodes[NODE_ID_GROUP_OUTPUT].position = Vector2(400, 150);

	group_name = TTR("Node group");
}

////////////// Group

void VisualShaderNodeGroup::_emit_changed() {
	emit_changed();
}

void VisualShaderNodeGroup::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_group", "group"), &VisualShaderNodeGroup::set_group);
	ClassDB::bind_method(D_METHOD("get_group"), &VisualShaderNodeGroup::get_group);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "group", PROPERTY_HINT_RESOURCE_TYPE, "VisualShaderGroup"), "set_group", "get_group");
}

String VisualShaderNodeGroup::get_caption() const {
	if (group.is_null()) {
		return TTR("Node group");
	}
	return group->get_group_name();
}

int VisualShaderNodeGroup::get_input_port_count() const {
	if (group.is_null()) {
		return 0;
	}
	return group->get_input_ports().size();
}

VisualShaderNode::PortType VisualShaderNodeGroup::get_input_port_type(int p_port) const {
	if (group.is_null()) {
		return PortType();
	}
	return group->get_input_port(p_port).type;
}

String VisualShaderNodeGroup::get_input_port_name(int p_port) const {
	if (group.is_null()) {
		return String();
	}
	return group->get_input_port(p_port).name;
}

int VisualShaderNodeGroup::get_output_port_count() const {
	if (group.is_null()) {
		return 0;
	}
	return group->get_output_ports().size();
}

VisualShaderNode::PortType VisualShaderNodeGroup::get_output_port_type(int p_port) const {
	if (group.is_null()) {
		return PortType();
	}
	return group->get_output_port(p_port).type;
}

String VisualShaderNodeGroup::get_output_port_name(int p_port) const {
	if (group.is_null()) {
		return String();
	}
	return group->get_output_port(p_port).name;
}

bool VisualShaderNodeGroup::is_show_prop_names() const {
	return true;
}

Vector<StringName> VisualShaderNodeGroup::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("group");
	return props;
}

bool VisualShaderNodeGroup::is_use_prop_slots() const {
	return true;
}

void VisualShaderNodeGroup::set_group(const Ref<VisualShaderGroup> &p_group) {
	if (group == p_group) {
		return;
	}
	group = p_group;
	emit_changed();
}

Ref<VisualShaderGroup> VisualShaderNodeGroup::get_group() const {
	return group;
}

void VisualShaderNodeGroup::set_shader_type(ShaderGraph::Type p_type) {
	shader_type = p_type;
}

void VisualShaderNodeGroup::set_shader_mode(Shader::Mode p_mode) {
	shader_mode = p_mode;
}

String VisualShaderNodeGroup::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	if (group.is_null()) {
		return String();
	}
	// TODO:Validate name and append unique id.

	// Generate the code for the group.
	String code = String("/* Group: ") + group->get_group_name() + " */\n";

	const String valid_group_name = group->_validate_group_name(group->get_group_name());
	ERR_FAIL_COND_V(valid_group_name.is_empty(), "");
	code += "group_" + valid_group_name + "(";

	const Vector<VisualShaderGroup::Port> input_ports = group->get_input_ports();
	int param_idx = 0;
	for (int i = 0; i < input_ports.size(); i++) {
		if (i > 0) {
			code += ",";
		}
		code += p_input_vars[i];
		param_idx++;
	}

	const Vector<VisualShaderGroup::Port> output_ports = group->get_output_ports();
	for (int i = 0; i < output_ports.size(); i++) {
		if (param_idx > 0) {
			code += ",";
		}
		code += p_output_vars[i];
		param_idx++;
	}

	code += ");\n";

	return code;
}

String VisualShaderNodeGroup::generate_group_function(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	if (group.is_null()) {
		return String();
	}

	// Generate a global function for the group.
	String code = String("/* Group: ") + group->get_group_name() + " */\n";

	code += group->get_global_code();

	// TODO: Don't use the type and id for the function name.
	const String valid_group_name = group->_validate_group_name(group->get_group_name());
	ERR_FAIL_COND_V(valid_group_name.is_empty(), "");

	code += "void group_" + valid_group_name + "(";

	// Add all inputs/outputs as function parameters.
	const Vector<VisualShaderGroup::Port> input_ports = group->get_input_ports();
	for (int i = 0; i < input_ports.size(); i++) {
		if (i == 0) {
			code += "in ";
		} else {
			code += ", in ";
		}
		code += VisualShaderNode::port_type_to_shader_string(input_ports[i].type) + " ";
		code += input_ports[i].name;
	}

	const Vector<VisualShaderGroup::Port> output_ports = group->get_output_ports();
	for (int i = 0; i < output_ports.size(); i++) {
		code += ", out ";
		code += VisualShaderNode::port_type_to_shader_string(output_ports[i].type) + " ";
		code += output_ports[i].name;
	}

	code += ") {\n";

	// Add the code for the group.
	code += group->get_code();

	code += "}\n";
	return code;
}

bool VisualShaderNodeGroup::is_output_port_expandable(int p_port) const {
	// TODO: Implement.
	return false;
}

VisualShaderNodeGroup::VisualShaderNodeGroup() {
	simple_decl = false;
}

void VisualShaderNodeGroupInput::set_group(VisualShaderGroup *p_group) {
	group = p_group;
}

VisualShaderGroup *VisualShaderNodeGroupInput::get_group() const {
	return group;
}

int VisualShaderNodeGroupInput::get_input_port_count() const {
	return 0;
}

VisualShaderNode::PortType VisualShaderNodeGroupInput::get_input_port_type(int p_port) const {
	return PortType();
}

String VisualShaderNodeGroupInput::get_input_port_name(int p_port) const {
	return String();
}

int VisualShaderNodeGroupInput::get_output_port_count() const {
	if (!group) {
		return 0;
	}
	return group->get_input_ports().size();
}

VisualShaderNode::PortType VisualShaderNodeGroupInput::get_output_port_type(int p_port) const {
	if (!group) {
		return PortType();
	}
	return group->get_input_port(p_port).type;
}

String VisualShaderNodeGroupInput::get_output_port_name(int p_port) const {
	if (!group) {
		return String();
	}
	return group->get_input_port(p_port).name;
}

bool VisualShaderNodeGroupInput::is_output_port_expandable(int p_port) const {
	// TODO: Implement.
	return false;
}

String VisualShaderNodeGroupInput::get_caption() const {
	return "Group Input";
}

String VisualShaderNodeGroupInput::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	ERR_FAIL_NULL_V(group, "");

	String code;
	for (int i = 0; i < group->get_input_ports().size(); i++) {
		code += p_output_vars[i] + " = " + group->get_input_port(i).name + ";\n";
	}
	return code;
}

Vector<StringName> VisualShaderNodeGroupInput::get_editable_properties() const {
	return Vector<StringName>();
}

VisualShaderNodeGroupInput::VisualShaderNodeGroupInput() {
}

void VisualShaderNodeGroupOutput::set_group(VisualShaderGroup *p_group) {
	group = p_group;
}

VisualShaderGroup *VisualShaderNodeGroupOutput::get_group() const {
	return group;
}

int VisualShaderNodeGroupOutput::get_input_port_count() const {
	if (!group) {
		return 0;
	}
	return group->get_output_ports().size();
}

VisualShaderNode::PortType VisualShaderNodeGroupOutput::get_input_port_type(int p_port) const {
	if (!group) {
		return PortType();
	}
	return group->get_output_port(p_port).type;
}

String VisualShaderNodeGroupOutput::get_input_port_name(int p_port) const {
	if (!group) {
		return String();
	}
	return group->get_output_port(p_port).name;
}

Variant VisualShaderNodeGroupOutput::get_input_port_default_value(int p_port) const {
	// TODO: Implement.
	return Variant();
}

int VisualShaderNodeGroupOutput::get_output_port_count() const {
	return 0;
}

VisualShaderNode::PortType VisualShaderNodeGroupOutput::get_output_port_type(int p_port) const {
	return PortType();
}

String VisualShaderNodeGroupOutput::get_output_port_name(int p_port) const {
	return String();
}

bool VisualShaderNodeGroupOutput::is_port_separator(int p_index) const {
	// TODO: Remove this?
	return false;
}

String VisualShaderNodeGroupOutput::get_caption() const {
	return "Group Output";
}

String VisualShaderNodeGroupOutput::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	ERR_FAIL_NULL_V(group, String());

	String code;
	for (int i = 0; i < group->get_output_ports().size(); i++) {
		if (p_input_vars[i].is_empty()) {
			continue;
		};
		code += group->get_output_port(i).name + " = " + p_input_vars[i] + ";\n";
	}
	return code;
}

VisualShaderNodeGroupOutput::VisualShaderNodeGroupOutput() {
}

void VisualShaderGroupPortsDialog::_add_port() {
	ERR_FAIL_NULL(group);

	// Add a new port to the group.
	const VisualShaderNode::PortType port_type = VisualShaderNode::PORT_TYPE_SCALAR;
	String port_name = edit_inputs ? "new_in_port" : "new_out_port";

	// Find a new valid name for the port.
	int port_idx = 2;
	String port_name_numerated = port_name;
	while (group->_validate_port_name(port_name_numerated, -1, !edit_inputs).is_empty()) {
		port_name_numerated = port_name + itos(port_idx);
	}
	port_name = port_name_numerated;

	if (edit_inputs) {
		group->add_input_port(group->get_input_ports().size(), port_type, port_name);
	} else {
		group->add_output_port(group->get_output_ports().size(), port_type, port_name);
	}

	// Update the item list.
	const Ref<Texture2D> port_icon = get_theme_icon(SNAME("port"), SNAME("GraphNode"));
	port_item_list->add_item(port_name, port_icon);

	const Vector<Color> port_colors = VisualShaderGraphPlugin::get_connection_type_colors();

	port_item_list->set_item_icon_modulate(port_item_list->get_item_count() - 1, port_colors[port_type]);

	// Select the new port.
	port_item_list->select(port_item_list->get_item_count() - 1);
	_update_editor_for_port(port_item_list->get_item_count() - 1);
}

void VisualShaderGroupPortsDialog::_update_editor_for_port(int p_idx) {
	ERR_FAIL_NULL(group);

	if (p_idx < 0 || p_idx >= port_item_list->get_item_count()) {
		name_edit->set_visible(false);
		port_type_optbtn->set_visible(false);
		return;
	}

	name_edit->set_visible(true);
	port_type_optbtn->set_visible(true);

	// Update the controls in the editor area of the dialog.
	ERR_FAIL_INDEX(p_idx, edit_inputs ? group->get_input_ports().size() : group->get_output_ports().size());
	const VisualShaderGroup::Port port = edit_inputs ? group->get_input_port(p_idx) : group->get_output_port(p_idx);

	name_edit->set_text(port.name);
	port_type_optbtn->select(port.type);
}

void VisualShaderGroupPortsDialog::_remove_port() {
	ERR_FAIL_NULL(group);
	ERR_FAIL_COND(port_item_list->get_selected_items().size() != 1);

	const int selected_idx = port_item_list->get_selected_items()[0];

	// Remove the port from the group.
	if (edit_inputs) {
		group->remove_input_port(selected_idx);
	} else {
		group->remove_output_port(selected_idx);
	}

	// Update the item list.
	port_item_list->remove_item(selected_idx);

	// Select the next port.
	if (selected_idx < port_item_list->get_item_count()) {
		port_item_list->select(selected_idx);
		_update_editor_for_port(selected_idx);
	}

	_update_editor_for_port(-1);
}

void VisualShaderGroupPortsDialog::_on_port_item_selected(int p_index) {
	_update_editor_for_port(p_index);
}

void VisualShaderGroupPortsDialog::_on_port_name_changed(const String &p_name) {
	ERR_FAIL_NULL(group);
	ERR_FAIL_COND(port_item_list->get_selected_items().size() != 1);

	// Update the port name in the group.
	const int port_idx = port_item_list->get_selected_items()[0];
	if (edit_inputs) {
		group->set_input_port_name(port_idx, p_name);
	} else {
		group->set_output_port_name(port_idx, p_name);
	}

	// Update the item list.
	port_item_list->set_item_text(port_idx, p_name);
}

void VisualShaderGroupPortsDialog::_on_port_type_changed(int p_idx) {
	ERR_FAIL_NULL(group);
	ERR_FAIL_COND(port_item_list->get_selected_items().size() != 1);

	// Update the port type in the group.
	const int port_idx = port_item_list->get_selected_items()[0];
	if (edit_inputs) {
		group->set_input_port_type(port_idx, VisualShaderNode::PortType(p_idx));
	} else {
		group->set_output_port_type(port_idx, VisualShaderNode::PortType(p_idx));
	}

	// Update the port color.
	const Vector<Color> port_colors = VisualShaderGraphPlugin::get_connection_type_colors();
	port_item_list->set_item_icon_modulate(port_idx, port_colors[p_idx]);
}

void VisualShaderGroupPortsDialog::_on_dialog_about_to_popup() {
	ERR_FAIL_NULL(group);
	if (port_item_list->get_item_count() == 0) {
		_update_editor_for_port(-1);
		return;
	}

	if (port_item_list->get_selected_items().size() != 1) {
		port_item_list->select(0);
	}
	_update_editor_for_port(port_item_list->get_selected_items()[0]);
}

void VisualShaderGroupPortsDialog::set_dialog_mode(bool p_edit_inputs) {
	if (edit_inputs == p_edit_inputs) {
		return;
	}
	edit_inputs = p_edit_inputs;
}

void VisualShaderGroupPortsDialog::set_group(VisualShaderGroup *p_group) {
	ERR_FAIL_NULL(p_group);
	group = p_group;

	// Update the item list.
	port_item_list->clear();
	const Vector<VisualShaderGroup::Port> ports = edit_inputs ? group->get_input_ports() : group->get_output_ports();
	const Vector<Color> port_colors = VisualShaderGraphPlugin::get_connection_type_colors();

	Ref<Texture2D> port_icon = get_theme_icon(SNAME("port"), SNAME("GraphNode"));
	for (int i = 0; i < ports.size(); i++) {
		port_item_list->add_item(ports[i].name, port_icon);
		port_item_list->set_item_icon_modulate(i, port_colors[ports[i].type]);
	}
}

VisualShaderGroupPortsDialog::VisualShaderGroupPortsDialog() {
	connect(SNAME("about_to_popup"), callable_mp(this, &VisualShaderGroupPortsDialog::_on_dialog_about_to_popup));
	set_title("Edit group ports");

	VBoxContainer *vbc = memnew(VBoxContainer);
	add_child(vbc);

	HBoxContainer *hbc = memnew(HBoxContainer);
	vbc->add_child(hbc);

	add_port_btn = memnew(Button);
	add_port_btn->set_text("Add port");
	add_port_btn->connect(SceneStringName(pressed), callable_mp(this, &VisualShaderGroupPortsDialog::_add_port));
	hbc->add_child(add_port_btn);

	remove_port_btn = memnew(Button);
	remove_port_btn->set_text("Remove port");
	remove_port_btn->connect(SceneStringName(pressed), callable_mp(this, &VisualShaderGroupPortsDialog::_remove_port));
	hbc->add_child(remove_port_btn);

	port_item_list = memnew(ItemList);
	port_item_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	port_item_list->connect(SceneStringName(item_selected), callable_mp(this, &VisualShaderGroupPortsDialog::_on_port_item_selected));
	vbc->add_child(port_item_list);

	name_edit = memnew(LineEdit);
	name_edit->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	name_edit->connect(SceneStringName(text_changed), callable_mp(this, &VisualShaderGroupPortsDialog::_on_port_name_changed));
	vbc->add_child(name_edit);

	port_type_optbtn = memnew(OptionButton);
	// TODO: Refactor this to use a global get_port_types() function.
	port_type_optbtn->add_item(TTR("Float"));
	port_type_optbtn->add_item(TTR("Int"));
	port_type_optbtn->add_item(TTR("UInt"));
	port_type_optbtn->add_item(TTR("Vector2"));
	port_type_optbtn->add_item(TTR("Vector3"));
	port_type_optbtn->add_item(TTR("Vector4"));
	port_type_optbtn->add_item(TTR("Boolean"));
	port_type_optbtn->add_item(TTR("Transform"));
	port_type_optbtn->add_item(TTR("Sampler"));
	port_type_optbtn->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	port_type_optbtn->connect(SceneStringName(item_selected), callable_mp(this, &VisualShaderGroupPortsDialog::_on_port_type_changed));
	vbc->add_child(port_type_optbtn);
}
