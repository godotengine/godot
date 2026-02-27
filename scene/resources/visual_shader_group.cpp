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

#include "core/object/class_db.h"
#include "scene/resources/visual_shader.h"
#include "scene/resources/visual_shader_particle_nodes.h"

String VisualShaderGroup::_validate_port_name(const String &p_port_name, int p_port_id, bool p_output) const {
	String port_name = p_port_name;

	if (port_name.is_empty()) {
		return String();
	}

	while (port_name.length() && !is_ascii_alphabet_char(port_name[0])) {
		port_name = port_name.substr(1);
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

	// Deduplication.
	int attempt = 1;
	while (true) {
		bool exists = false;
		for (int i = 0; i < get_input_port_count(); i++) {
			if (!p_output && i == p_port_id) {
				continue;
			}
			if (port_name == get_input_port_name(i)) {
				exists = true;
				break;
			}
		}
		if (!exists) {
			for (int i = 0; i < get_output_port_count(); i++) {
				if (p_output && i == p_port_id) {
					continue;
				}
				if (port_name == get_output_port_name(i)) {
					exists = true;
					break;
				}
			}
		}

		if (exists) {
			// Strip trailing digits, append an incremented number and try again.
			attempt++;
			while (port_name.length() && is_digit(port_name[port_name.length() - 1])) {
				port_name = port_name.substr(0, port_name.length() - 1);
			}
			ERR_FAIL_COND_V(port_name.is_empty(), String());
			port_name += itos(attempt);
		} else {
			break;
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
	ClassDB::bind_method(D_METHOD("set_group_name", "name"), &VisualShaderGroup::set_group_name);
	ClassDB::bind_method(D_METHOD("get_group_name"), &VisualShaderGroup::get_group_name);

	ClassDB::bind_method(D_METHOD("insert_input_port", "id", "type", "name"), &VisualShaderGroup::insert_input_port);
	ClassDB::bind_method(D_METHOD("remove_input_port", "id"), &VisualShaderGroup::remove_input_port);
	ClassDB::bind_method(D_METHOD("move_input_port", "from", "to"), &VisualShaderGroup::move_input_port);
	ClassDB::bind_method(D_METHOD("set_input_port_name", "id", "name"), &VisualShaderGroup::set_input_port_name);
	ClassDB::bind_method(D_METHOD("set_input_port_type", "id", "type"), &VisualShaderGroup::set_input_port_type);
	ClassDB::bind_method(D_METHOD("get_input_port_count"), &VisualShaderGroup::get_input_port_count);
	ClassDB::bind_method(D_METHOD("get_input_port_name", "id"), &VisualShaderGroup::get_input_port_name);
	ClassDB::bind_method(D_METHOD("get_input_port_type", "id"), &VisualShaderGroup::get_input_port_type);

	ClassDB::bind_method(D_METHOD("insert_output_port", "id", "type", "name"), &VisualShaderGroup::insert_output_port);
	ClassDB::bind_method(D_METHOD("remove_output_port", "id"), &VisualShaderGroup::remove_output_port);
	ClassDB::bind_method(D_METHOD("move_output_port", "from", "to"), &VisualShaderGroup::move_output_port);
	ClassDB::bind_method(D_METHOD("set_output_port_name", "id", "name"), &VisualShaderGroup::set_output_port_name);
	ClassDB::bind_method(D_METHOD("set_output_port_type", "id", "type"), &VisualShaderGroup::set_output_port_type);
	ClassDB::bind_method(D_METHOD("get_output_port_count"), &VisualShaderGroup::get_output_port_count);
	ClassDB::bind_method(D_METHOD("get_output_port_name", "id"), &VisualShaderGroup::get_output_port_name);
	ClassDB::bind_method(D_METHOD("get_output_port_type", "id"), &VisualShaderGroup::get_output_port_type);

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

	ClassDB::bind_method(D_METHOD("get_node_connections"), &VisualShaderGroup::_get_node_connections);

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

	StringBuilder global_code_builder;
	StringBuilder global_code_per_node_builder;
	HashMap<ShaderGraph::Type, StringBuilder> global_code_per_func_builder;
	StringBuilder code_builder;
	default_tex_params.clear();

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

	HashMap<int, String> code_map;
	HashSet<int> empty_funcs;
	HashMap<ShaderGraph::ConnectionKey, const List<ShaderGraph::Connection>::Element *> input_connections;
	HashMap<ShaderGraph::ConnectionKey, const List<ShaderGraph::Connection>::Element *> output_connections;

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

	// Write code starting from each the first output node that is active.
	// Note: Right now that is just the first one.
	for (const KeyValue<int, ShaderGraph::Node> &E : graph->nodes) {
		Ref<VisualShaderNodeGroupOutput> group_output = E.value.node;
		if (group_output.is_valid()) {
			graph->_write_node(&global_code_builder, &global_code_per_node_builder, &global_code_per_func_builder, group_code, default_tex_params, input_connections, output_connections, E.key, processed, false, classes);
			break;
		}
	}

	for (int &E : emitters) {
		graph->_write_node(&global_code_builder, &global_code_per_node_builder, &global_code_per_func_builder, group_code, default_tex_params, input_connections, output_connections, E, processed, false, classes);
	}

	code_builder += "// Group content: " + group_name + "\n";
	code_builder += group_code;

	global_code_builder.append(global_code_per_node_builder);
	global_code_builder.append(global_expressions);
	global_code = global_code_builder.as_string();
	code = code_builder.as_string();

	emit_changed();
}

bool VisualShaderGroup::_set(const StringName &p_name, const Variant &p_value) {
	if (p_name == "input_ports") {
		input_ports.clear();
		const Array &ports = p_value;
		for (const Dictionary port : ports) {
			Port p;
			p.type = (VisualShaderNode::PortType)(int)port["type"];
			p.name = port["name"];
			input_ports.push_back(p);
		}
		emit_changed();
		return true;
	} else if (p_name == "output_ports") {
		output_ports.clear();
		const Array &ports = p_value;
		for (const Dictionary port : ports) {
			Port p;
			p.type = (VisualShaderNode::PortType)(int)port["type"];
			p.name = port["name"];
			output_ports.push_back(p);
		}
		emit_changed();
		return true;
	}

	bool result = graph->_set(p_name, p_value);

	// Fix up group pointers for deserialized group input/output nodes.
	const String prop_name_str = p_name;
	if (result && prop_name_str.begins_with("nodes/")) {
		const String index = prop_name_str.get_slicec('/', 1);
		const String node_info = prop_name_str.get_slicec('/', 2);
		if (node_info == "node") {
			const int id = index.to_int();
			Ref<VisualShaderNodeGroupInput> input = graph->get_node_unchecked(id);
			if (input.is_valid()) {
				input->set_group(this);
			}
			Ref<VisualShaderNodeGroupOutput> output = graph->get_node_unchecked(id);
			if (output.is_valid()) {
				output->set_group(this);
			}
		}
	}

	return result;
}

bool VisualShaderGroup::_get(const StringName &p_name, Variant &r_ret) const {
	if (p_name == "input_ports") {
		Array ports;
		for (int i = 0; i < input_ports.size(); i++) {
			Dictionary port;
			port["id"] = i;
			port["type"] = input_ports[i].type;
			port["name"] = input_ports[i].name;
			ports.push_back(port);
		}
		r_ret = ports;
		return true;
	} else if (p_name == "output_ports") {
		Array ports;
		for (int i = 0; i < output_ports.size(); i++) {
			Dictionary port;
			port["id"] = i;
			port["type"] = output_ports[i].type;
			port["name"] = output_ports[i].name;
			ports.push_back(port);
		}
		r_ret = ports;
		return true;
	}
	return graph->_get(p_name, r_ret);
}

void VisualShaderGroup::_get_property_list(List<PropertyInfo> *p_list) const {
	// Input/output ports must be added first so that group input/output nodes
	// already know their ports before connections are loaded.
	p_list->push_back(PropertyInfo(Variant::ARRAY, "input_ports", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
	p_list->push_back(PropertyInfo(Variant::ARRAY, "output_ports", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
	graph->_get_property_list(p_list);
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

Vector<ShaderGraph::DefaultTextureParam> VisualShaderGroup::get_default_texture_params() {
	if (dirty.is_set()) {
		_update_group();
	}
	return default_tex_params;
}

String VisualShaderGroup::get_unique_func_name() const {
	const String valid_name = _validate_group_name(group_name);
	// Use the scene unique ID as suffix as it's stable and available for both external and embedded subresources.
	// Stability is not strictly necessary right now, because the shader is regenerated at runtime (and not saved), but maybe this is changed in the future.
	const String suid = get_scene_unique_id();
	const String suffix = suid.is_empty() ? uitos(get_instance_id()) : suid; // Fallback to unstable instance ID.
	if (valid_name.is_empty()) {
		return "group_" + suffix;
	}
	return "group_" + valid_name + "_" + suffix;
}

void VisualShaderGroup::set_group_name(const String &p_name) {
	if (group_name == p_name) {
		return;
	}

	group_name = p_name;
	emit_changed();
}

String VisualShaderGroup::get_group_name() const {
	return group_name;
}

String VisualShaderGroup::insert_input_port(int p_id, VisualShaderNode::PortType p_type, const String &p_name) {
	ERR_FAIL_INDEX_V(p_id, input_ports.size() + 1, String());

	const String valid_name = _validate_port_name(p_name, p_id, false);
	if (valid_name.is_empty()) {
		return String();
	}

	// Shift connections above upward.
	if (p_id < input_ports.size()) {
		List<ShaderGraph::Connection> conns;
		get_node_connections(&conns);
		for (const ShaderGraph::Connection &c : conns) {
			Ref<VisualShaderNodeGroupInput> group_input = graph->get_node(c.from_node);
			if (group_input.is_valid() && c.from_port >= p_id) {
				disconnect_nodes(c.from_node, c.from_port, c.to_node, c.to_port);
				connect_nodes_forced(c.from_node, c.from_port + 1, c.to_node, c.to_port);
			}
		}
	}

	input_ports.insert(p_id, Port{ p_type, valid_name });
	_queue_update();
	emit_changed();
	return valid_name;
}

void VisualShaderGroup::set_input_port_name(int p_id, const String &p_name) {
	ERR_FAIL_INDEX(p_id, input_ports.size());

	const String valid_name = _validate_port_name(p_name, p_id, false);
	if (valid_name.is_empty()) {
		return;
	}

	input_ports.write[p_id].name = valid_name;
	_queue_update();
	emit_changed();
}

void VisualShaderGroup::set_input_port_type(int p_id, VisualShaderNode::PortType p_type) {
	ERR_FAIL_INDEX(p_id, input_ports.size());

	if (input_ports[p_id].type == p_type) {
		return;
	}

	input_ports.write[p_id].type = p_type;
	_queue_update();
	emit_changed();
}

int VisualShaderGroup::get_input_port_count() const {
	return input_ports.size();
}

String VisualShaderGroup::get_input_port_name(int p_id) const {
	ERR_FAIL_INDEX_V(p_id, input_ports.size(), String());
	return input_ports[p_id].name;
}

VisualShaderNode::PortType VisualShaderGroup::get_input_port_type(int p_id) const {
	ERR_FAIL_INDEX_V(p_id, input_ports.size(), VisualShaderNode::PORT_TYPE_SCALAR);
	return input_ports[p_id].type;
}

void VisualShaderGroup::remove_input_port(int p_id) {
	ERR_FAIL_INDEX(p_id, input_ports.size());

	// Shift connections above downward.
	List<ShaderGraph::Connection> conns;
	get_node_connections(&conns);
	for (const ShaderGraph::Connection &c : conns) {
		const Ref<VisualShaderNodeGroupInput> group_input = graph->get_node(c.from_node);
		if (group_input.is_valid()) {
			if (c.from_port == p_id) {
				disconnect_nodes(c.from_node, c.from_port, c.to_node, c.to_port);
			} else if (c.from_port > p_id) {
				disconnect_nodes(c.from_node, c.from_port, c.to_node, c.to_port);
				connect_nodes_forced(c.from_node, c.from_port - 1, c.to_node, c.to_port);
			}
		}
	}

	input_ports.remove_at(p_id);
	_queue_update();
	emit_changed();
}

void VisualShaderGroup::move_input_port(int p_from, int p_to) {
	ERR_FAIL_INDEX(p_from, input_ports.size());
	ERR_FAIL_INDEX(p_to, input_ports.size());
	if (p_from == p_to) {
		return;
	}

	// Update connections that reference moved port indices.
	List<ShaderGraph::Connection> conns;
	get_node_connections(&conns);
	for (const ShaderGraph::Connection &c : conns) {
		const Ref<VisualShaderNodeGroupInput> group_input = graph->get_node(c.from_node);
		if (group_input.is_valid()) {
			int new_port = c.from_port;
			if (c.from_port == p_from) {
				new_port = p_to;
			} else if (p_from < p_to) {
				// Shifting ports in (p_from, p_to] down by one.
				if (c.from_port > p_from && c.from_port <= p_to) {
					new_port = c.from_port - 1;
				}
			} else {
				// Shifting ports in [p_to, p_from) up by one.
				if (c.from_port >= p_to && c.from_port < p_from) {
					new_port = c.from_port + 1;
				}
			}
			if (new_port != c.from_port) {
				disconnect_nodes(c.from_node, c.from_port, c.to_node, c.to_port);
				connect_nodes_forced(c.from_node, new_port, c.to_node, c.to_port);
			}
		}
	}

	const Port port = input_ports[p_from];
	input_ports.remove_at(p_from);
	input_ports.insert(p_to, port);
	_queue_update();
	emit_changed();
}

String VisualShaderGroup::insert_output_port(int p_id, VisualShaderNode::PortType p_type, const String &p_name) {
	ERR_FAIL_INDEX_V(p_id, output_ports.size() + 1, String());

	const String valid_name = _validate_port_name(p_name, p_id, true);
	if (valid_name.is_empty()) {
		return String();
	}

	// Shift connections above upward.
	if (p_id < output_ports.size()) {
		List<ShaderGraph::Connection> conns;
		get_node_connections(&conns);
		for (const ShaderGraph::Connection &c : conns) {
			Ref<VisualShaderNodeGroupOutput> group_output = graph->get_node(c.to_node);
			if (group_output.is_valid() && c.to_port >= p_id) {
				disconnect_nodes(c.from_node, c.from_port, c.to_node, c.to_port);
				connect_nodes_forced(c.from_node, c.from_port, c.to_node, c.to_port + 1);
			}
		}
	}

	output_ports.insert(p_id, Port{ p_type, valid_name });
	_queue_update();
	emit_changed();
	return valid_name;
}

void VisualShaderGroup::set_output_port_name(int p_id, const String &p_name) {
	ERR_FAIL_INDEX(p_id, output_ports.size());

	const String valid_name = _validate_port_name(p_name, p_id, true);
	if (valid_name.is_empty()) {
		return;
	}

	output_ports.write[p_id].name = valid_name;
	_queue_update();
	emit_changed();
}

void VisualShaderGroup::set_output_port_type(int p_id, VisualShaderNode::PortType p_type) {
	ERR_FAIL_INDEX(p_id, output_ports.size());

	if (output_ports[p_id].type == p_type) {
		return;
	}

	output_ports.write[p_id].type = p_type;
	_queue_update();
	emit_changed();
}

int VisualShaderGroup::get_output_port_count() const {
	return output_ports.size();
}

String VisualShaderGroup::get_output_port_name(int p_id) const {
	ERR_FAIL_INDEX_V(p_id, output_ports.size(), String());
	return output_ports[p_id].name;
}

VisualShaderNode::PortType VisualShaderGroup::get_output_port_type(int p_id) const {
	ERR_FAIL_INDEX_V(p_id, output_ports.size(), VisualShaderNode::PORT_TYPE_SCALAR);
	return output_ports[p_id].type;
}

void VisualShaderGroup::remove_output_port(int p_id) {
	ERR_FAIL_INDEX(p_id, output_ports.size());

	// Shift connections above downward.
	List<ShaderGraph::Connection> conns;
	get_node_connections(&conns);
	for (const ShaderGraph::Connection &c : conns) {
		const Ref<VisualShaderNodeGroupOutput> group_output = graph->get_node(c.to_node);
		if (group_output.is_valid()) {
			if (c.to_port == p_id) {
				disconnect_nodes(c.from_node, c.from_port, c.to_node, c.to_port);
			} else if (c.to_port > p_id) {
				disconnect_nodes(c.from_node, c.from_port, c.to_node, c.to_port);
				connect_nodes_forced(c.from_node, c.from_port, c.to_node, c.to_port - 1);
			}
		}
	}

	output_ports.remove_at(p_id);
	_queue_update();
	emit_changed();
}

void VisualShaderGroup::move_output_port(int p_from, int p_to) {
	ERR_FAIL_INDEX(p_from, output_ports.size());
	ERR_FAIL_INDEX(p_to, output_ports.size());
	if (p_from == p_to) {
		return;
	}

	// Update connections that reference moved port indices.
	List<ShaderGraph::Connection> conns;
	get_node_connections(&conns);
	for (const ShaderGraph::Connection &c : conns) {
		const Ref<VisualShaderNodeGroupOutput> group_output = graph->get_node(c.to_node);

		if (group_output.is_valid()) {
			int new_port = c.to_port;
			if (c.to_port == p_from) {
				new_port = p_to;
			} else if (p_from < p_to) {
				// Shifting ports in (p_from, p_to] down by one.
				if (c.to_port > p_from && c.to_port <= p_to) {
					new_port = c.to_port - 1;
				}
			} else {
				// Shifting ports in [p_to, p_from) up by one.
				if (c.to_port >= p_to && c.to_port < p_from) {
					new_port = c.to_port + 1;
				}
			}
			if (new_port != c.to_port) {
				disconnect_nodes(c.from_node, c.from_port, c.to_node, c.to_port);
				connect_nodes_forced(c.from_node, c.from_port, c.to_node, new_port);
			}
		}
	}

	Port port = output_ports[p_from];
	output_ports.remove_at(p_from);
	output_ports.insert(p_to, port);
	_queue_update();
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

bool VisualShaderGroup::is_node_reachable(int p_from, int p_target) const {
	return graph->is_node_reachable(p_from, p_target);
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

TypedArray<Dictionary> VisualShaderGroup::_get_node_connections() const {
	TypedArray<Dictionary> ret;
	for (const ShaderGraph::Connection &E : graph->connections) {
		Dictionary d;
		d["from_node"] = E.from_node;
		d["from_port"] = E.from_port;
		d["to_node"] = E.to_node;
		d["to_port"] = E.to_port;
		ret.push_back(d);
	}
	return ret;
}

void VisualShaderGroup::get_node_connections(List<ShaderGraph::Connection> *r_connections) const {
	graph->get_node_connections(r_connections);
}

String VisualShaderGroup::generate_preview_shader(int p_node, int p_port, Vector<ShaderGraph::DefaultTextureParam> &r_default_tex_params) const {
	return graph->generate_preview_shader(p_node, p_port, r_default_tex_params);
}

void VisualShaderGroup::create_default_nodes_if_empty() {
	if (!graph->nodes.is_empty()) {
		return;
	}

	Ref<VisualShaderNodeGroupInput> input_node;
	input_node.instantiate();
	input_node->set_group(this);
	graph->add_node(input_node, Vector2(0, 150), graph->get_valid_node_id());

	Ref<VisualShaderNodeGroupOutput> output_node;
	output_node.instantiate();
	output_node->set_group(this);
	graph->add_node(output_node, Vector2(400, 150), graph->get_valid_node_id());
}

VisualShaderGroup::VisualShaderGroup() {
	graph.instantiate(0); // No reserved node IDs; input/output are regular nodes.
	graph->connect("graph_changed", callable_mp(this, &VisualShaderGroup::_queue_update));

	group_name = RTR("Node group");
}

////////////// Group

void VisualShaderNodeGroup::_emit_changed() {
	for (int i = 0; i < get_input_port_count(); i++) {
		const PortType type = get_input_port_type(i);
		const Variant default_value_variant = VisualShaderNode::get_port_type_default_value_variant(type);
		if (!default_input_values.has(i) ||
				default_value_variant.get_type() != default_input_values[i].get_type()) {
			set_input_port_default_value(i, default_value_variant);
		}
	}

	emit_changed();
}

bool VisualShaderNodeGroup::_has_incompatible_nodes(Shader::Mode p_mode, VisualShader::Type p_type) const {
	if (group.is_null()) {
		return false;
	}

	Ref<ShaderGraph> graph = group->get_graph();
	if (graph.is_null()) {
		return false;
	}

	for (const int id : graph->get_node_ids()) {
		Ref<VisualShaderNode> node = graph->get_node(id);
		if (node.is_null()) {
			continue;
		}

		if (!node->is_available(p_mode, p_type)) {
			return true;
		}
	}

	return false;
}

void VisualShaderNodeGroup::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_group", "group"), &VisualShaderNodeGroup::set_group);
	ClassDB::bind_method(D_METHOD("get_group"), &VisualShaderNodeGroup::get_group);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "group", PROPERTY_HINT_RESOURCE_TYPE, VisualShaderGroup::get_class_static()), "set_group", "get_group");
}

String VisualShaderNodeGroup::get_caption() const {
	if (group.is_null()) {
		return RTR("Node group");
	}
	return group->get_group_name();
}

int VisualShaderNodeGroup::get_input_port_count() const {
	if (group.is_null()) {
		return 0;
	}
	return group->get_input_port_count();
}

VisualShaderNode::PortType VisualShaderNodeGroup::get_input_port_type(int p_port) const {
	if (group.is_null() || p_port < 0 || p_port >= group->get_input_port_count()) {
		return PortType();
	}
	return group->get_input_port_type(p_port);
}

String VisualShaderNodeGroup::get_input_port_name(int p_port) const {
	if (group.is_null() || p_port < 0 || p_port >= group->get_input_port_count()) {
		return String();
	}
	return group->get_input_port_name(p_port);
}

int VisualShaderNodeGroup::get_output_port_count() const {
	if (group.is_null()) {
		return 0;
	}
	return group->get_output_port_count();
}

VisualShaderNode::PortType VisualShaderNodeGroup::get_output_port_type(int p_port) const {
	if (group.is_null() || p_port < 0 || p_port >= group->get_output_port_count()) {
		return PortType();
	}
	return group->get_output_port_type(p_port);
}

String VisualShaderNodeGroup::get_output_port_name(int p_port) const {
	if (group.is_null() || p_port < 0 || p_port >= group->get_output_port_count()) {
		return String();
	}
	return group->get_output_port_name(p_port);
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

	if (group.is_valid()) {
		group->disconnect_changed(callable_mp(this, &VisualShaderNodeGroup::_emit_changed));
	}
	group = p_group;
	if (group.is_valid()) {
		group->create_default_nodes_if_empty();
		group->connect_changed(callable_mp(this, &VisualShaderNodeGroup::_emit_changed));
	}
	emit_changed();
}

Ref<VisualShaderGroup> VisualShaderNodeGroup::get_group() const {
	return group;
}

String VisualShaderNodeGroup::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	if (group.is_null()) {
		return String();
	}

	// If the group contains nodes incompatible with the current shader context,
	// output default values instead of calling the group function.
	if (_has_incompatible_nodes(p_mode, p_type)) {
		String code = String("/* Group: ") + group->get_group_name() + " */\n";
		for (int i = 0; i < group->get_output_port_count(); i++) {
			if (!p_output_vars[i].is_empty()) {
				code += "\t" + p_output_vars[i] + " = " + get_port_type_default_value_shader_string(group->get_output_port_type(i)) + ";\n";
			}
		}
		return code;
	}

	// Generate the code for the group.
	String code = String("/* Group: ") + group->get_group_name() + " */\n";

	const String func_name = group->get_unique_func_name();
	code += func_name + "(";

	int param_idx = 0;
	for (int i = 0; i < group->get_input_port_count(); i++) {
		if (i > 0) {
			code += ",";
		}
		if (group->get_input_port_type(i) == PORT_TYPE_SAMPLER && p_input_vars[i].is_empty()) {
			// Use the default sampler uniform when nothing is connected.
			code += make_unique_id(p_type, p_id, "sampler_" + itos(i));
		} else {
			code += p_input_vars[i];
		}
		param_idx++;
	}

	for (int i = 0; i < group->get_output_port_count(); i++) {
		if (param_idx > 0) {
			code += ",";
		}
		code += p_output_vars[i];
		param_idx++;
	}

	code += ");\n";

	return code;
}

Vector<ShaderGraph::DefaultTextureParam> VisualShaderNodeGroup::get_default_texture_parameters(VisualShader::Type p_type, int p_id) const {
	if (group.is_null()) {
		return Vector<ShaderGraph::DefaultTextureParam>();
	}
	return group->get_default_texture_params();
}

String VisualShaderNodeGroup::generate_group_function(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	if (group.is_null()) {
		return String();
	}

	// Don't generate the function if the group contains incompatible nodes.
	if (_has_incompatible_nodes(p_mode, p_type)) {
		return String();
	}

	// Generate a global function for the group.
	String code = String("/* Group: ") + group->get_group_name() + " */\n";

	code += group->get_global_code();

	const String func_name = group->get_unique_func_name();

	code += "void " + func_name + "(";

	// Add all inputs/outputs as function parameters prefixed with "p_" (to prevent redefining builtins like UV, etc.).
	for (int i = 0; i < group->get_input_port_count(); i++) {
		if (i > 0) {
			code += ", ";
		}
		code += "in ";
		code += VisualShaderNode::get_port_type_shader_string(group->get_input_port_type(i)) + " ";
		code += "p_" + group->get_input_port_name(i);
	}

	for (int i = 0; i < group->get_output_port_count(); i++) {
		if (i > 0 || group->get_input_port_count() > 0) {
			code += ", ";
		}
		code += "out ";
		code += VisualShaderNode::get_port_type_shader_string(group->get_output_port_type(i)) + " ";
		code += "p_" + group->get_output_port_name(i);
	}

	code += ") {\n";

	// Add the code for the group.
	code += group->get_code();

	code += "}\n";
	return code;
}

bool VisualShaderNodeGroup::is_output_port_expandable(int p_port) const {
	return false;
}

String VisualShaderNodeGroup::get_warning(Shader::Mode p_mode, VisualShader::Type p_type) const {
	if (group.is_null()) {
		return RTR("No group resource assigned.");
	}

	Vector<String> warnings;

	// Check for missing output node.
	Ref<ShaderGraph> sgraph = group->get_graph();
	if (sgraph.is_valid()) {
		int output_node_count = 0;
		Vector<int> node_ids = sgraph->get_node_ids();
		for (int id : node_ids) {
			Ref<VisualShaderNodeGroupOutput> output_node = sgraph->get_node(id);
			if (output_node.is_valid()) {
				output_node_count++;
			}
		}
		if (output_node_count == 0) {
			warnings.push_back(RTR("Group is missing an output node."));
		} else if (output_node_count > 1) {
			warnings.push_back(RTR("Group has multiple output nodes. Only one is allowed."));
		}

		// Check for parameters and varyings.
		// Note: For now they are just forbidden due to the additional complexity of supporting them.
		for (const int id : node_ids) {
			Ref<VisualShaderNodeParameter> param_node = sgraph->get_node(id);
			if (param_node.is_valid()) {
				warnings.push_back(RTR("Parameters are not supported inside groups."));
				break;
			}
		}
		for (const int id : node_ids) {
			Ref<VisualShaderNodeVarying> varying_node = sgraph->get_node(id);
			if (varying_node.is_valid()) {
				warnings.push_back(RTR("Varyings are not supported inside groups."));
				break;
			}
		}

		// Check for nodes incompatible with the current shader mode/type.
		if (_has_incompatible_nodes(p_mode, p_type)) {
			warnings.push_back(RTR("Group contains nodes incompatible with the current shader. Outputs will use default values."));
		}
	}

	String warning_str;
	for (const String &warning : warnings) {
		warning_str += warning + "\n";
	}
	return warning_str.trim_suffix("\n");
}

bool VisualShaderNodeGroup::is_available(Shader::Mode p_mode, VisualShader::Type p_type) const {
	return !_has_incompatible_nodes(p_mode, p_type);
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
	return group->get_input_port_count();
}

VisualShaderNode::PortType VisualShaderNodeGroupInput::get_output_port_type(int p_port) const {
	if (!group) {
		return PortType();
	}
	return group->get_input_port_type(p_port);
}

String VisualShaderNodeGroupInput::get_output_port_name(int p_port) const {
	if (!group) {
		return String();
	}
	return group->get_input_port_name(p_port);
}

bool VisualShaderNodeGroupInput::is_output_port_expandable(int p_port) const {
	return false;
}

String VisualShaderNodeGroupInput::get_caption() const {
	return "Group Input";
}

String VisualShaderNodeGroupInput::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	ERR_FAIL_NULL_V(group, "");

	String code;
	for (int i = 0; i < group->get_input_port_count(); i++) {
		if (group->get_input_port_type(i) == PORT_TYPE_SAMPLER) {
			continue;
		}
		if (p_for_preview) {
			// When generating preview shaders, we use default values instead of the group input port names
			// since the port names are not valid identifiers in the preview context.
			// TODO: Maybe we find a way to use the inputs of the VisualShaderNodeGroup that was used to open the node group? (or maybe this is not a good idea?)
			code += p_output_vars[i] + " = " + VisualShaderNode::get_port_type_default_value_shader_string(group->get_input_port_type(i)) + ";\n";
		} else {
			code += p_output_vars[i] + " = " + "p_" + group->get_input_port_name(i) + ";\n";
		}
	}
	return code;
}

Vector<StringName> VisualShaderNodeGroupInput::get_editable_properties() const {
	return Vector<StringName>();
}

VisualShaderNodeGroupInput::VisualShaderNodeGroupInput() {
}

void VisualShaderNodeGroupOutput::_group_changed() {
	// Set default values if they don't exist or if the type doesn't match the port type anymore.
	for (int i = 0; i < get_input_port_count(); i++) {
		const PortType type = get_input_port_type(i);
		const Variant default_value_variant = VisualShaderNode::get_port_type_default_value_variant(type);
		if (!default_input_values.has(i) ||
				default_value_variant.get_type() != default_input_values[i].get_type()) {
			set_input_port_default_value(i, default_value_variant);
		}
	}
}

void VisualShaderNodeGroupOutput::set_group(VisualShaderGroup *p_group) {
	group = p_group;

	if (group) {
		group->connect_changed(callable_mp(this, &VisualShaderNodeGroupOutput::_group_changed));
	}
	emit_changed();
}

VisualShaderGroup *VisualShaderNodeGroupOutput::get_group() const {
	return group;
}

int VisualShaderNodeGroupOutput::get_input_port_count() const {
	if (!group) {
		return 0;
	}
	return group->get_output_port_count();
}

VisualShaderNode::PortType VisualShaderNodeGroupOutput::get_input_port_type(int p_port) const {
	if (!group) {
		return PortType();
	}
	return group->get_output_port_type(p_port);
}

String VisualShaderNodeGroupOutput::get_input_port_name(int p_port) const {
	if (!group) {
		return String();
	}
	return group->get_output_port_name(p_port);
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

String VisualShaderNodeGroupOutput::get_caption() const {
	return "Group Output";
}

String VisualShaderNodeGroupOutput::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	ERR_FAIL_NULL_V(group, String());

	String code;
	for (int i = 0; i < group->get_output_port_count(); i++) {
		if (p_input_vars[i].is_empty()) {
			continue;
		}
		code += "p_" + group->get_output_port_name(i) + " = " + p_input_vars[i] + ";\n";
	}
	return code;
}

VisualShaderNodeGroupOutput::VisualShaderNodeGroupOutput() {
}
