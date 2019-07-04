/*************************************************************************/
/*  visual_shader.cpp                                                    */
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

#include "visual_shader.h"

#include "core/vmap.h"
#include "servers/visual/shader_types.h"

void VisualShaderNode::set_output_port_for_preview(int p_index) {

	port_preview = p_index;
}

int VisualShaderNode::get_output_port_for_preview() const {

	return port_preview;
}

void VisualShaderNode::set_input_port_default_value(int p_port, const Variant &p_value) {
	default_input_values[p_port] = p_value;
	emit_changed();
}

Variant VisualShaderNode::get_input_port_default_value(int p_port) const {
	if (default_input_values.has(p_port)) {
		return default_input_values[p_port];
	}

	return Variant();
}

bool VisualShaderNode::is_port_separator(int p_index) const {
	return false;
}

Vector<VisualShader::DefaultTextureParam> VisualShaderNode::get_default_texture_parameters(VisualShader::Type p_type, int p_id) const {
	return Vector<VisualShader::DefaultTextureParam>();
}
String VisualShaderNode::generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	return String();
}

Vector<StringName> VisualShaderNode::get_editable_properties() const {
	return Vector<StringName>();
}

Array VisualShaderNode::_get_default_input_values() const {

	Array ret;
	for (Map<int, Variant>::Element *E = default_input_values.front(); E; E = E->next()) {
		ret.push_back(E->key());
		ret.push_back(E->get());
	}
	return ret;
}
void VisualShaderNode::_set_default_input_values(const Array &p_values) {

	if (p_values.size() % 2 == 0) {
		for (int i = 0; i < p_values.size(); i += 2) {
			default_input_values[p_values[i + 0]] = p_values[i + 1];
		}
	}

	emit_changed();
}

String VisualShaderNode::get_warning(Shader::Mode p_mode, VisualShader::Type p_type) const {
	return String();
}

void VisualShaderNode::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_output_port_for_preview", "port"), &VisualShaderNode::set_output_port_for_preview);
	ClassDB::bind_method(D_METHOD("get_output_port_for_preview"), &VisualShaderNode::get_output_port_for_preview);

	ClassDB::bind_method(D_METHOD("set_input_port_default_value", "port", "value"), &VisualShaderNode::set_input_port_default_value);
	ClassDB::bind_method(D_METHOD("get_input_port_default_value", "port"), &VisualShaderNode::get_input_port_default_value);

	ClassDB::bind_method(D_METHOD("_set_default_input_values", "values"), &VisualShaderNode::_set_default_input_values);
	ClassDB::bind_method(D_METHOD("_get_default_input_values"), &VisualShaderNode::_get_default_input_values);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "output_port_for_preview"), "set_output_port_for_preview", "get_output_port_for_preview");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "default_input_values", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "_set_default_input_values", "_get_default_input_values");
	ADD_SIGNAL(MethodInfo("editor_refresh_request"));
}

VisualShaderNode::VisualShaderNode() {
	port_preview = -1;
}

/////////////////////////////////////////////////////////

bool VisualShader::is_valid_type_index(int p_type) const {

	return p_type < get_function_count();
}

void VisualShader::add_node(Type p_type, const Ref<VisualShaderNode> &p_node, const Vector2 &p_position, int p_id) {

	ERR_FAIL_COND(p_node.is_null());
	ERR_FAIL_COND(p_id < 2);
	ERR_FAIL_COND(!is_valid_type_index(p_type));
	Graph *g = get_graph_set(p_type);
	ERR_FAIL_COND(g->nodes.has(p_id));
	Node n;
	n.node = p_node;
	n.position = p_position;

	Ref<VisualShaderNodeUniform> uniform = n.node;
	if (uniform.is_valid()) {
		String valid_name = validate_uniform_name(uniform->get_uniform_name(), uniform);
		uniform->set_uniform_name(valid_name);
	}

	Ref<VisualShaderNodeInput> input = n.node;
	if (input.is_valid()) {
		if (p_type >= TYPE_MAX) {
			input->shader_mode = Shader::Mode(-1);
		} else {
			input->shader_mode = shader_mode;
		}
		input->shader_name = g->name;
		input->shader_type = p_type;
		input->connect("input_type_changed", this, "_input_type_changed", varray(p_type, p_id));
	}

	n.node->connect("changed", this, "_queue_update");

	g->nodes[p_id] = n;

	_queue_update();
}

void VisualShader::clear_custom_funcs() {

	for (int i = TYPE_MAX; i < get_function_count(); i++) {
		graph[i]->inputs.clear();
	}

	VisualShaderNodeInput::clear_custom_funcs();
	VisualShaderNodeOutput::clear_custom_funcs();
}

void VisualShader::set_node_position(Type p_type, int p_id, const Vector2 &p_position) {

	ERR_FAIL_COND(!is_valid_type_index(p_type));
	Graph *g = get_graph_set(p_type);
	ERR_FAIL_COND(!g->nodes.has(p_id));
	g->nodes[p_id].position = p_position;
}

Vector2 VisualShader::get_node_position(Type p_type, int p_id) const {

	ERR_FAIL_COND_V(!is_valid_type_index(p_type), Vector2());
	const Graph *g = get_graph(p_type);
	ERR_FAIL_COND_V(!g->nodes.has(p_id), Vector2());
	return g->nodes[p_id].position;
}

Ref<VisualShaderNode> VisualShader::get_node(Type p_type, int p_id) const {

	ERR_FAIL_COND_V(!is_valid_type_index(p_type), Ref<VisualShaderNode>());
	const Graph *g = get_graph(p_type);
	ERR_FAIL_COND_V(!g->nodes.has(p_id), Ref<VisualShaderNode>());
	return g->nodes[p_id].node;
}

const VisualShader::Graph *VisualShader::get_graph(Type p_type) const {
	return graph[p_type];
}

VisualShader::Graph *VisualShader::get_graph_set(Type p_type) {
	return graph[p_type];
}

Vector<int> VisualShader::get_node_list(Type p_type) const {

	ERR_FAIL_COND_V(!is_valid_type_index(p_type), Vector<int>());

	const Graph *g = get_graph(p_type);

	Vector<int> ret;
	for (Map<int, Node>::Element *E = g->nodes.front(); E; E = E->next()) {
		ret.push_back(E->key());
	}

	return ret;
}

int VisualShader::get_valid_node_id(Type p_type) const {

	ERR_FAIL_COND_V(!is_valid_type_index(p_type), NODE_ID_INVALID);
	const Graph *g = get_graph(p_type);
	return g->nodes.size() ? MAX(2, g->nodes.back()->key() + 1) : 2;
}

int VisualShader::find_node_id(Type p_type, const Ref<VisualShaderNode> &p_node) const {

	ERR_FAIL_COND_V(!is_valid_type_index(p_type), NODE_ID_INVALID);
	for (const Map<int, Node>::Element *E = graph[p_type]->nodes.front(); E; E = E->next()) {
		if (E->get().node == p_node)
			return E->key();
	}
	return NODE_ID_INVALID;
}

void VisualShader::remove_node(Type p_type, int p_id) {

	ERR_FAIL_COND(!is_valid_type_index(p_type));
	ERR_FAIL_COND(p_id < 2);
	Graph *g = get_graph_set(p_type);
	ERR_FAIL_COND(!g->nodes.has(p_id));

	Ref<VisualShaderNodeInput> input = g->nodes[p_id].node;
	if (input.is_valid()) {
		input->disconnect("input_type_changed", this, "_input_type_changed");
	}

	g->nodes[p_id].node->disconnect("changed", this, "_queue_update");

	g->nodes.erase(p_id);

	for (List<Connection>::Element *E = g->connections.front(); E;) {
		List<Connection>::Element *N = E->next();
		if (E->get().from_node == p_id || E->get().to_node == p_id) {
			g->connections.erase(E);
		}
		E = N;
	}

	_queue_update();
}

bool VisualShader::is_node_connection(Type p_type, int p_from_node, int p_from_port, int p_to_node, int p_to_port) const {

	ERR_FAIL_COND_V(!is_valid_type_index(p_type), false);
	const Graph *g = get_graph(p_type);

	for (const List<Connection>::Element *E = g->connections.front(); E; E = E->next()) {

		if (E->get().from_node == p_from_node && E->get().from_port == p_from_port && E->get().to_node == p_to_node && E->get().to_port == p_to_port) {
			return true;
		}
	}

	return false;
}

bool VisualShader::can_connect_nodes(Type p_type, int p_from_node, int p_from_port, int p_to_node, int p_to_port) const {

	ERR_FAIL_COND_V(!is_valid_type_index(p_type), false);
	const Graph *g = get_graph(p_type);

	if (!g->nodes.has(p_from_node))
		return false;

	if (p_from_node == p_to_node)
		return false;

	if (p_from_port < 0 || p_from_port >= g->nodes[p_from_node].node->get_output_port_count())
		return false;

	if (!g->nodes.has(p_to_node))
		return false;

	if (p_to_port < 0 || p_to_port >= g->nodes[p_to_node].node->get_input_port_count())
		return false;

	VisualShaderNode::PortType from_port_type = g->nodes[p_from_node].node->get_output_port_type(p_from_port);
	VisualShaderNode::PortType to_port_type = g->nodes[p_to_node].node->get_input_port_type(p_to_port);

	if (!is_port_types_compatible(from_port_type, to_port_type)) {
		return false;
	}

	for (const List<Connection>::Element *E = g->connections.front(); E; E = E->next()) {

		if (E->get().from_node == p_from_node && E->get().from_port == p_from_port && E->get().to_node == p_to_node && E->get().to_port == p_to_port) {
			return false;
		}
	}

	return true;
}

bool VisualShader::is_port_types_compatible(int p_a, int p_b) const {

	return MAX(0, p_a - 2) == (MAX(0, p_b - 2));
}

void VisualShader::connect_nodes_forced(Type p_type, int p_from_node, int p_from_port, int p_to_node, int p_to_port) {

	ERR_FAIL_COND(!is_valid_type_index(p_type));
	Graph *g = get_graph_set(p_type);
	Connection c;
	c.from_node = p_from_node;
	c.from_port = p_from_port;
	c.to_node = p_to_node;
	c.to_port = p_to_port;
	g->connections.push_back(c);
	_queue_update();
}

Error VisualShader::connect_nodes(Type p_type, int p_from_node, int p_from_port, int p_to_node, int p_to_port) {

	ERR_FAIL_COND_V(!is_valid_type_index(p_type), ERR_CANT_CONNECT);
	Graph *g = get_graph_set(p_type);

	ERR_FAIL_COND_V(!g->nodes.has(p_from_node), ERR_INVALID_PARAMETER);
	ERR_FAIL_INDEX_V(p_from_port, g->nodes[p_from_node].node->get_output_port_count(), ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(!g->nodes.has(p_to_node), ERR_INVALID_PARAMETER);
	ERR_FAIL_INDEX_V(p_to_port, g->nodes[p_to_node].node->get_input_port_count(), ERR_INVALID_PARAMETER);

	VisualShaderNode::PortType from_port_type = g->nodes[p_from_node].node->get_output_port_type(p_from_port);
	VisualShaderNode::PortType to_port_type = g->nodes[p_to_node].node->get_input_port_type(p_to_port);

	if (!is_port_types_compatible(from_port_type, to_port_type)) {
		ERR_EXPLAIN("Incompatible port types (scalar/vec/bool) with transform");
		ERR_FAIL_V(ERR_INVALID_PARAMETER);
		return ERR_INVALID_PARAMETER;
	}

	for (List<Connection>::Element *E = g->connections.front(); E; E = E->next()) {

		if (E->get().from_node == p_from_node && E->get().from_port == p_from_port && E->get().to_node == p_to_node && E->get().to_port == p_to_port) {
			ERR_FAIL_V(ERR_ALREADY_EXISTS);
		}
	}

	Connection c;
	c.from_node = p_from_node;
	c.from_port = p_from_port;
	c.to_node = p_to_node;
	c.to_port = p_to_port;
	g->connections.push_back(c);

	_queue_update();
	return OK;
}

void VisualShader::disconnect_nodes(Type p_type, int p_from_node, int p_from_port, int p_to_node, int p_to_port) {

	ERR_FAIL_COND(!is_valid_type_index(p_type));
	Graph *g = get_graph_set(p_type);

	for (List<Connection>::Element *E = g->connections.front(); E; E = E->next()) {

		if (E->get().from_node == p_from_node && E->get().from_port == p_from_port && E->get().to_node == p_to_node && E->get().to_port == p_to_port) {
			g->connections.erase(E);
			_queue_update();
			return;
		}
	}
}

Array VisualShader::_get_node_connections(Type p_type) const {

	ERR_FAIL_COND_V(!is_valid_type_index(p_type), Array());
	const Graph *g = get_graph(p_type);

	Array ret;
	for (const List<Connection>::Element *E = g->connections.front(); E; E = E->next()) {
		Dictionary d;
		d["from_node"] = E->get().from_node;
		d["from_port"] = E->get().from_port;
		d["to_node"] = E->get().to_node;
		d["to_port"] = E->get().to_port;
		ret.push_back(d);
	}

	return ret;
}

int VisualShader::get_function_count() const {

	return graph.size();
}

String VisualShader::get_function_name(int p_id) const {

	ERR_FAIL_COND_V(!is_valid_type_index(p_id), "");
	return graph[p_id]->name;
}

String VisualShader::get_function_name_by_index(int p_id) const {

	for (int i = TYPE_MAX; i < get_function_count(); i++) {
		if (graph[i]->index == p_id) {
			return graph[i]->name;
		}
	}
	return "";
}

int VisualShader::get_function_index_by_name(const String &p_name) const {
	for (int func_id = TYPE_MAX; func_id < get_function_count(); func_id++) {
		if (graph[func_id]->name == p_name) {
			return func_id;
		}
	}
	return -1;
}

int VisualShader::get_function_input_port_count(int p_func_id) const {

	ERR_FAIL_INDEX_V(p_func_id, get_function_count(), 0);
	return graph[p_func_id]->inputs.size();
}

String VisualShader::get_function_input_port_name(int p_func_id, int p_port_id) const {

	ERR_FAIL_INDEX_V(p_func_id, get_function_count(), String(""));
	ERR_FAIL_INDEX_V(p_port_id, graph[p_func_id]->inputs.size(), String(""));
	return graph[p_func_id]->inputs[p_port_id].name;
}

int VisualShader::get_function_input_port_type(int p_func_id, int p_port_id) const {

	ERR_FAIL_INDEX_V(p_func_id, get_function_count(), 0);
	ERR_FAIL_INDEX_V(p_port_id, graph[p_func_id]->inputs.size(), 0);
	return graph[p_func_id]->inputs[p_port_id].type;
}

int VisualShader::get_function_output_port_type(int p_func_id) const {

	ERR_FAIL_INDEX_V(p_func_id, get_function_count(), 0);
	return graph[p_func_id]->output_type;
}

bool VisualShader::has_function_name(const String &p_name) const {

	if (p_name == "vertex" || p_name == "fragment" || p_name == "light")
		return true;

	for (int i = 0; i < get_function_count(); i++) {
		if (graph[i]->name.to_lower() == p_name) {
			return true;
		}
	}
	return false;
}

void VisualShader::add_function(const String &p_name) {
	Graph *g = memnew(Graph);
	g->name = p_name;
	g->index = graph.size() - TYPE_MAX;
	graph.push_back(g);
}

void VisualShader::rename_function(const String &p_name, const String &p_new_name) {
	int idx = 0;
	while (idx < graph.size()) {
		if (graph[idx]->name == p_name) {
			graph[idx]->name = p_new_name;
			break;
		}
		idx++;
	}
	VisualShaderNodeInput::rename_ports_func(p_name, p_new_name);
	VisualShaderNodeOutput::rename_ports_func(p_name, p_new_name);
	_queue_update();
}

void VisualShader::move_func_up(int p_func_id) {
	if (p_func_id - 1 >= TYPE_MAX) {
		graph[p_func_id]->index -= 1;
		graph[p_func_id - 1]->index += 1;
	}
}

void VisualShader::move_func_down(int p_func_id) {
	if (p_func_id + 1 < get_function_count()) {
		graph[p_func_id]->index += 1;
		graph[p_func_id + 1]->index -= 1;
	}
}

void VisualShader::set_function_index(const String &p_func_name, int p_index) {
	Graph *g = NULL;
	for (int idx = 0; idx < graph.size(); idx++) {
		if (graph[idx]->name == p_func_name) {
			g = graph[idx];
			break;
		}
	}
	g->index = p_index;
}

int VisualShader::get_function_index(int p_func_id) const {

	return graph[p_func_id]->index;
}

void VisualShader::set_function_output_type(const String &p_func_name, int p_type) {

	Graph *g = NULL;
	for (int idx = 0; idx < graph.size(); idx++) {
		if (graph[idx]->name == p_func_name) {
			g = graph[idx];
			break;
		}
	}

	Ref<VisualShaderNodeOutput> output;
	output.instance();
	output->shader_name = g->name;
	output->shader_mode = Shader::MODE_MAX;
	output->add_port(g->name, "result", VisualShaderNode::PortType(p_type));

	g->output_type = p_type;
	g->nodes[NODE_ID_OUTPUT].node = output;
	g->nodes[NODE_ID_OUTPUT].position = Vector2(400, 150);
}

void VisualShader::remove_function(const String &p_name) {

	String name = p_name.to_lower();
	ERR_FAIL_COND(name == "vertex" || name == "fragment" || name == "light");
	int id = -1;

	for (int i = TYPE_MAX; i < get_function_count(); i++) {
		Graph *g = graph[i];
		if (g->name.to_lower() == name) {
			id = i;
			VisualShaderNodeOutput::remove_func(name);
			VisualShaderNodeInput::remove_func(name);
			graph.erase(g);
			memdelete(g);
			break;
		}
	}

	if (id >= TYPE_MAX) {

		// remove ports from calls
		for (int func_id = 0; func_id < get_function_count(); func_id++) {
			Type p_type = Type(func_id);

			List<VisualShader::Connection> conns;
			get_node_connections(p_type, &conns);

			for (List<VisualShader::Connection>::Element *E = conns.front(); E; E = E->next()) {

				int from_node = E->get().from_node;
				int from_port = E->get().from_port;
				int to_node = E->get().to_node;
				int to_port = E->get().to_port;

				Ref<VisualShaderNodeCall> in_call = get_node(p_type, to_node);
				if (in_call.is_valid()) {
					if (in_call->get_function_name() == name) {
						disconnect_nodes(p_type, from_node, from_port, to_node, to_port);
					}
				}
				Ref<VisualShaderNodeCall> out_call = get_node(p_type, from_node);
				if (out_call.is_valid()) {
					if (out_call->get_function_name() == name) {
						disconnect_nodes(p_type, from_node, from_port, to_node, to_port);
					}
				}
			}

			Vector<int> nodes = get_node_list(p_type);

			// correct all calls
			for (int n_i = 0; n_i < nodes.size(); n_i++) {

				Ref<VisualShaderNodeCall> call = get_node(p_type, nodes[n_i]);
				if (call.is_valid()) {

					if (call->get_function_name() == name) {

						call->set_function_id(0);
						call->set_function_name("");

						int port_count = call->get_input_port_count();
						for (int port_id = 0; port_id < port_count; port_id++) {
							call->remove_input_port(port_id);
						}
						if (call->has_output_port(0)) {
							call->remove_output_port(0);
						}
					} else {
						if (call->get_function_id() > id) {
							call->set_function_id(call->get_function_id() - 1);
						}
					}
				}
			}
		}

		_queue_update();
	}
}

void VisualShader::add_custom_input(const String &p_func_name, const String &p_name, int p_type) {

	for (int i = TYPE_MAX; i < get_function_count(); i++) {
		Graph *g = graph[i];
		if (g->name == p_func_name) {
			g->inputs.push_back({ p_type, p_name, "p_" + p_name });
			VisualShaderNodeInput::add_custom_port(p_func_name, p_name, p_type);
			break;
		}
	}
}

void VisualShader::commit_function(int p_func_id) {

	ERR_FAIL_INDEX(p_func_id, get_function_count());
	Graph *g = graph[p_func_id];
	String inputs;
	for (int i = 0; i < g->inputs.size(); i++) {
		switch (g->inputs[i].type) {
			case 0: // Scalar
				inputs += "float ";
				break;
			case 1: // Vector
				inputs += "vec3 ";
				break;
			case 2: // Bool
				inputs += "bool ";
				break;
			case 3: // Transform
				inputs += "mat4 ";
				break;
			default:
				break;
		}
		inputs += g->inputs[i].string;
		if (i < g->inputs.size() - 1) {
			inputs += ", ";
		}
	}
	g->inputs_str = inputs;
	_queue_update();
}

void VisualShader::get_node_connections(Type p_type, List<Connection> *r_connections) const {

	ERR_FAIL_COND(!is_valid_type_index(p_type));

	const Graph *g = get_graph(p_type);

	for (const List<Connection>::Element *E = g->connections.front(); E; E = E->next()) {
		r_connections->push_back(E->get());
	}
}

void VisualShader::set_mode(Mode p_mode) {

	if (shader_mode == p_mode) {
		return;
	}

	//erase input/output connections
	modes.clear();
	flags.clear();
	shader_mode = p_mode;
	for (int i = 0; i < get_function_count(); i++) {

		for (Map<int, Node>::Element *E = get_graph(Type(i))->nodes.front(); E; E = E->next()) {

			Ref<VisualShaderNodeInput> input = E->get().node;
			if (input.is_valid()) {
				input->shader_mode = shader_mode;
				//input->input_index = 0;
			}
		}

		Ref<VisualShaderNodeOutput> output = get_graph(Type(i))->nodes[NODE_ID_OUTPUT].node;
		output->shader_mode = shader_mode;

		// clear connections since they are no longer valid
		for (List<Connection>::Element *E = get_graph_set(Type(i))->connections.front(); E;) {

			bool keep = true;

			List<Connection>::Element *N = E->next();

			int from = E->get().from_node;
			int to = E->get().to_node;

			if (!get_graph(Type(i))->nodes.has(from)) {
				keep = false;
			} else {
				Ref<VisualShaderNode> from_node = get_graph(Type(i))->nodes[from].node;
				if (from_node->is_class("VisualShaderNodeOutput") || from_node->is_class("VisualShaderNodeInput")) {
					keep = false;
				}
			}

			if (!get_graph(Type(i))->nodes.has(to)) {
				keep = false;
			} else {
				Ref<VisualShaderNode> to_node = get_graph(Type(i))->nodes[to].node;
				if (to_node->is_class("VisualShaderNodeOutput") || to_node->is_class("VisualShaderNodeInput")) {
					keep = false;
				}
			}

			if (!keep) {
				get_graph_set(Type(i))->connections.erase(E);
			}
			E = N;
		}
	}

	_queue_update();
	_change_notify();
}

void VisualShader::set_graph_offset(const Vector2 &p_offset) {

	graph_offset = p_offset;
}

Vector2 VisualShader::get_graph_offset() const {

	return graph_offset;
}

Shader::Mode VisualShader::get_mode() const {

	return shader_mode;
}

bool VisualShader::is_text_shader() const {

	return false;
}

String VisualShader::generate_preview_shader(Type p_type, int p_node, int p_port, Vector<DefaultTextureParam> &default_tex_params) const {

	Ref<VisualShaderNode> node = get_node(p_type, p_node);
	ERR_FAIL_COND_V(!node.is_valid(), String());
	ERR_FAIL_COND_V(p_port < 0 || p_port >= node->get_output_port_count(), String());
	ERR_FAIL_COND_V(node->get_output_port_type(p_port) == VisualShaderNode::PORT_TYPE_TRANSFORM, String());

	StringBuilder global_code;
	StringBuilder code;

	global_code += String() + "shader_type canvas_item;\n";

	// need to paste all custom functions for correct preview
	for (int i = TYPE_MAX; i < get_function_count(); i++) {

		VMap<ConnectionKey, const List<Connection>::Element *> input_connections;
		VMap<ConnectionKey, const List<Connection>::Element *> output_connections;

		for (const List<Connection>::Element *E = graph[i]->connections.front(); E; E = E->next()) {
			ConnectionKey from_key;
			from_key.node = E->get().from_node;
			from_key.port = E->get().from_port;

			output_connections.insert(from_key, E);

			ConnectionKey to_key;
			to_key.node = E->get().to_node;
			to_key.port = E->get().to_port;

			input_connections.insert(to_key, E);
		}

		code += "\n";

		Ref<VisualShaderNodeOutput> output = graph[i]->nodes[NODE_ID_OUTPUT].node;
		switch (output->get_input_port_type(0)) {
			case VisualShaderNode::PORT_TYPE_SCALAR:
				code += "float";
				break;
			case VisualShaderNode::PORT_TYPE_VECTOR:
				code += "vec3";
				break;
			case VisualShaderNode::PORT_TYPE_BOOLEAN:
				code += "bool";
				break;
			case VisualShaderNode::PORT_TYPE_TRANSFORM:
				code += "mat4";
				break;
			default:
				break;
		}

		code += " " + graph[i]->name + "(";
		code += graph[i]->inputs_str;
		code += ") {\n";

		Set<int> processed;
		_write_node(Type(i), global_code, code, default_tex_params, input_connections, output_connections, NODE_ID_OUTPUT, processed, true);
		code += "}\n";
	}

	//make it faster to go around through shader
	VMap<ConnectionKey, const List<Connection>::Element *> input_connections;
	VMap<ConnectionKey, const List<Connection>::Element *> output_connections;

	for (const List<Connection>::Element *E = get_graph(p_type)->connections.front(); E; E = E->next()) {
		ConnectionKey from_key;
		from_key.node = E->get().from_node;
		from_key.port = E->get().from_port;

		output_connections.insert(from_key, E);

		ConnectionKey to_key;
		to_key.node = E->get().to_node;
		to_key.port = E->get().to_port;

		input_connections.insert(to_key, E);
	}

	code += "\nvoid fragment() {\n";

	Set<int> processed;
	Error err = _write_node(p_type, global_code, code, default_tex_params, input_connections, output_connections, p_node, processed, true);
	ERR_FAIL_COND_V(err != OK, String());

	if (node->get_output_port_type(p_port) == VisualShaderNode::PORT_TYPE_SCALAR) {
		code += "\tCOLOR.rgb = vec3( n_out" + itos(p_node) + "p" + itos(p_port) + " );\n";
	} else if (node->get_output_port_type(p_port) == VisualShaderNode::PORT_TYPE_BOOLEAN) {
		code += "\tCOLOR.rgb = vec3( n_out" + itos(p_node) + "p" + itos(p_port) + " ? 1.0 : 0.0 );\n";
	} else {
		code += "\tCOLOR.rgb = n_out" + itos(p_node) + "p" + itos(p_port) + ";\n";
	}
	code += "}\n";

	//set code secretly
	global_code += "\n\n";
	String final_code = global_code;
	final_code += code;
	return final_code;
}

#define IS_INITIAL_CHAR(m_d) (((m_d) >= 'a' && (m_d) <= 'z') || ((m_d) >= 'A' && (m_d) <= 'Z'))

#define IS_SYMBOL_CHAR(m_d) (((m_d) >= 'a' && (m_d) <= 'z') || ((m_d) >= 'A' && (m_d) <= 'Z') || ((m_d) >= '0' && (m_d) <= '9') || (m_d) == '_')

String VisualShader::validate_port_name(const String &p_name, const List<String> &p_input_ports, const List<String> &p_output_ports) const {

	String name = p_name;

	while (name.length() && !IS_INITIAL_CHAR(name[0])) {
		name = name.substr(1, name.length() - 1);
	}

	if (name != String()) {

		String valid_name;

		for (int i = 0; i < name.length(); i++) {
			if (IS_SYMBOL_CHAR(name[i])) {
				valid_name += String::chr(name[i]);
			} else if (name[i] == ' ') {
				valid_name += "_";
			}
		}

		name = valid_name;
	}

	String valid_name = name;
	bool is_equal = false;

	for (int i = 0; i < p_input_ports.size(); i++) {
		if (name == p_input_ports[i]) {
			is_equal = true;
			break;
		}
	}

	if (!is_equal) {
		for (int i = 0; i < p_output_ports.size(); i++) {
			if (name == p_output_ports[i]) {
				is_equal = true;
				break;
			}
		}
	}

	if (is_equal) {
		name = "";
	}

	return name;
}

String VisualShader::validate_uniform_name(const String &p_name, const Ref<VisualShaderNodeUniform> &p_uniform) const {

	String name = p_name; //validate name first
	while (name.length() && !IS_INITIAL_CHAR(name[0])) {
		name = name.substr(1, name.length() - 1);
	}
	if (name != String()) {

		String valid_name;

		for (int i = 0; i < name.length(); i++) {
			if (IS_SYMBOL_CHAR(name[i])) {
				valid_name += String::chr(name[i]);
			} else if (name[i] == ' ') {
				valid_name += "_";
			}
		}

		name = valid_name;
	}

	if (name == String()) {
		name = p_uniform->get_caption();
	}

	int attempt = 1;

	while (true) {

		bool exists = false;
		for (int i = 0; i < get_function_count(); i++) {
			for (const Map<int, Node>::Element *E = get_graph(Type(i))->nodes.front(); E; E = E->next()) {
				Ref<VisualShaderNodeUniform> node = E->get().node;
				if (node == p_uniform) { //do not test on self
					continue;
				}
				if (node.is_valid() && node->get_uniform_name() == name) {
					exists = true;
					break;
				}
			}
			if (exists) {
				break;
			}
		}

		if (exists) {
			//remove numbers, put new and try again
			attempt++;
			while (name.length() && name[name.length() - 1] >= '0' && name[name.length() - 1] <= '9') {
				name = name.substr(0, name.length() - 1);
			}
			ERR_FAIL_COND_V(name == String(), String());
			name += itos(attempt);
		} else {
			break;
		}
	}

	return name;
}

VisualShader::RenderModeEnums VisualShader::render_mode_enums[] = {
	{ Shader::MODE_SPATIAL, "blend" },
	{ Shader::MODE_SPATIAL, "depth_draw" },
	{ Shader::MODE_SPATIAL, "cull" },
	{ Shader::MODE_SPATIAL, "diffuse" },
	{ Shader::MODE_SPATIAL, "specular" },
	{ Shader::MODE_CANVAS_ITEM, "blend" },
	{ Shader::MODE_CANVAS_ITEM, NULL }
};

bool VisualShader::_set(const StringName &p_name, const Variant &p_value) {

	String name = p_name;
	if (name == "mode") {
		set_mode(Shader::Mode(int(p_value)));
		return true;
	} else if (name.begins_with("flags/")) {
		StringName flag = name.get_slicec('/', 1);
		bool enable = p_value;
		if (enable) {
			flags.insert(flag);
		} else {
			flags.erase(flag);
		}
		_queue_update();
		return true;
	} else if (name.begins_with("modes/")) {
		String mode = name.get_slicec('/', 1);
		int value = p_value;
		if (value == 0) {
			modes.erase(mode); //means it's default anyway, so don't store it
		} else {
			modes[mode] = value;
		}
		_queue_update();
		return true;
	} else if (name.begins_with("funcs/")) {
		String func = name.get_slicec('/', 1);
		int func_id = func.to_int() + TYPE_MAX;

		static String func_name = "";
		String index = name.get_slicec('/', 2);
		if (index == "name") {
			add_function(p_value);
			func_name = p_value;
			return true;
		} else if (index == "index") {
			set_function_index(func_name, p_value);
			return true;
		} else if (index == "output_port_type") {
			set_function_output_type(func_name, p_value);
			return true;
		}
		String what = name.get_slicec('/', 3);
		static String port_name = "";
		if (what == "input_port_name") {
			port_name = p_value;
			return true;
		} else if (what == "input_port_type") {
			add_custom_input(func_name, port_name, p_value);
			commit_function(func_id);
			return true;
		}
	} else if (name.begins_with("nodes/")) {
		String typestr = name.get_slicec('/', 1);
		Type type = TYPE_VERTEX;
		for (int i = 0; i < get_function_count(); i++) {
			if (typestr == graph[i]->name) {
				type = Type(i);
				break;
			}
		}

		String index = name.get_slicec('/', 2);
		if (index == "connections") {

			Vector<int> conns = p_value;
			if (conns.size() % 4 == 0) {
				for (int i = 0; i < conns.size(); i += 4) {
					connect_nodes_forced(type, conns[i + 0], conns[i + 1], conns[i + 2], conns[i + 3]);
				}
			}
			return true;
		}

		int id = index.to_int();
		String what = name.get_slicec('/', 3);

		if (what == "node") {
			add_node(type, p_value, Vector2(), id);
			return true;
		} else if (what == "position") {
			set_node_position(type, id, p_value);
			return true;
		} else if (what == "size") {
			((VisualShaderNodeGroupBase *)get_node(type, id).ptr())->set_size(p_value);
			return true;
		} else if (what == "input_ports") {
			((VisualShaderNodeGroupBase *)get_node(type, id).ptr())->set_inputs(p_value);
			return true;
		} else if (what == "output_ports") {
			((VisualShaderNodeGroupBase *)get_node(type, id).ptr())->set_outputs(p_value);
			return true;
		} else if (what == "expression") {
			((VisualShaderNodeExpression *)get_node(type, id).ptr())->set_expression(p_value);
			return true;
		}
	}
	return false;
}

bool VisualShader::_get(const StringName &p_name, Variant &r_ret) const {

	String name = p_name;
	if (name == "mode") {
		r_ret = get_mode();
		return true;
	} else if (name.begins_with("flags/")) {
		StringName flag = name.get_slicec('/', 1);
		r_ret = flags.has(flag);
		return true;
	} else if (name.begins_with("modes/")) {
		String mode = name.get_slicec('/', 1);
		if (modes.has(mode)) {
			r_ret = modes[mode];
		} else {
			r_ret = 0;
		}
		return true;
	} else if (name.begins_with("funcs/")) {
		String func = name.get_slicec('/', 1);
		int func_id = func.to_int() + TYPE_MAX;

		String index = name.get_slicec('/', 2);

		if (index == "name") {
			r_ret = get_function_name(func_id);
			return true;
		} else if (index == "index") {
			r_ret = get_function_index(func_id);
			return true;
		} else if (index == "output_port_type") {
			r_ret = get_function_output_port_type(func_id);
			return true;
		} else if (index == "input_port_count") {
			r_ret = get_function_input_port_count(func_id);
			return true;
		}
		int id = index.to_int();
		String what = name.get_slicec('/', 3);
		if (what == "input_port_name") {
			r_ret = get_function_input_port_name(func_id, id);
			return true;
		} else if (what == "input_port_type") {
			r_ret = get_function_input_port_type(func_id, id);
			return true;
		}
	} else if (name.begins_with("nodes/")) {
		String typestr = name.get_slicec('/', 1);
		Type type = TYPE_VERTEX;
		for (int i = 0; i < get_function_count(); i++) {
			if (typestr == graph[i]->name) {
				type = Type(i);
				break;
			}
		}

		String index = name.get_slicec('/', 2);
		if (index == "connections") {

			Vector<int> conns;
			for (const List<Connection>::Element *E = graph[type]->connections.front(); E; E = E->next()) {
				conns.push_back(E->get().from_node);
				conns.push_back(E->get().from_port);
				conns.push_back(E->get().to_node);
				conns.push_back(E->get().to_port);
			}

			r_ret = conns;
			return true;
		}

		int id = index.to_int();
		String what = name.get_slicec('/', 3);

		if (what == "node") {
			r_ret = get_node(type, id);
			return true;
		} else if (what == "position") {
			r_ret = get_node_position(type, id);
			return true;
		} else if (what == "size") {
			r_ret = ((VisualShaderNodeGroupBase *)get_node(type, id).ptr())->get_size();
			return true;
		} else if (what == "input_ports") {
			r_ret = ((VisualShaderNodeGroupBase *)get_node(type, id).ptr())->get_inputs();
			return true;
		} else if (what == "output_ports") {
			r_ret = ((VisualShaderNodeGroupBase *)get_node(type, id).ptr())->get_outputs();
			return true;
		} else if (what == "expression") {
			r_ret = ((VisualShaderNodeExpression *)get_node(type, id).ptr())->get_expression();
			return true;
		}
	}
	return false;
}

void VisualShader::_get_property_list(List<PropertyInfo> *p_list) const {

	//mode
	p_list->push_back(PropertyInfo(Variant::INT, "mode", PROPERTY_HINT_ENUM, "Spatial,CanvasItem,Particles"));
	//render modes

	Map<String, String> blend_mode_enums;
	Set<String> toggles;

	for (int i = 0; i < ShaderTypes::get_singleton()->get_modes(VisualServer::ShaderMode(shader_mode)).size(); i++) {

		String mode = ShaderTypes::get_singleton()->get_modes(VisualServer::ShaderMode(shader_mode))[i];
		int idx = 0;
		bool in_enum = false;
		while (render_mode_enums[idx].string) {
			if (mode.begins_with(render_mode_enums[idx].string)) {
				String begin = render_mode_enums[idx].string;
				String option = mode.replace_first(begin + "_", "");
				if (!blend_mode_enums.has(begin)) {
					blend_mode_enums[begin] = option;
				} else {
					blend_mode_enums[begin] += "," + option;
				}
				in_enum = true;
				break;
			}
			idx++;
		}

		if (!in_enum) {
			toggles.insert(mode);
		}
	}

	for (Map<String, String>::Element *E = blend_mode_enums.front(); E; E = E->next()) {

		p_list->push_back(PropertyInfo(Variant::INT, "modes/" + E->key(), PROPERTY_HINT_ENUM, E->get()));
	}

	for (Set<String>::Element *E = toggles.front(); E; E = E->next()) {
		p_list->push_back(PropertyInfo(Variant::BOOL, "flags/" + E->get()));
	}

	for (int i = TYPE_MAX; i < get_function_count(); i++) {

		String prop_name = "funcs/";
		prop_name += itos(i - TYPE_MAX);
		p_list->push_back(PropertyInfo(Variant::STRING, prop_name + "/name", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
		p_list->push_back(PropertyInfo(Variant::INT, prop_name + "/index", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
		p_list->push_back(PropertyInfo(Variant::INT, prop_name + "/output_port_type", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));

		if (!get_graph(Type(i))->inputs.empty())
			p_list->push_back(PropertyInfo(Variant::INT, prop_name + "/input_port_count", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));

		for (int j = 0; j < get_graph(Type(i))->inputs.size(); j++) {
			p_list->push_back(PropertyInfo(Variant::STRING, prop_name + "/" + itos(j) + "/input_port_name", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
			p_list->push_back(PropertyInfo(Variant::STRING, prop_name + "/" + itos(j) + "/input_port_type", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
		}
	}

	for (int i = 0; i < get_function_count(); i++) {

		const String &func_name = get_graph(Type(i))->name;

		for (Map<int, Node>::Element *E = get_graph(Type(i))->nodes.front(); E; E = E->next()) {

			String prop_name = "nodes/";
			prop_name += func_name;
			prop_name += "/" + itos(E->key());

			if (E->key() != NODE_ID_OUTPUT) {

				p_list->push_back(PropertyInfo(Variant::OBJECT, prop_name + "/node", PROPERTY_HINT_RESOURCE_TYPE, "VisualShaderNode", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_DO_NOT_SHARE_ON_DUPLICATE));
			}
			p_list->push_back(PropertyInfo(Variant::VECTOR2, prop_name + "/position", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));

			if (Object::cast_to<VisualShaderNodeGroupBase>(E->get().node.ptr()) != NULL) {
				p_list->push_back(PropertyInfo(Variant::VECTOR2, prop_name + "/size", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
				p_list->push_back(PropertyInfo(Variant::STRING, prop_name + "/input_ports", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
				p_list->push_back(PropertyInfo(Variant::STRING, prop_name + "/output_ports", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
			}
			if (Object::cast_to<VisualShaderNodeExpression>(E->get().node.ptr()) != NULL) {
				p_list->push_back(PropertyInfo(Variant::STRING, prop_name + "/expression", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
			}
		}
		p_list->push_back(PropertyInfo(Variant::POOL_INT_ARRAY, "nodes/" + func_name + "/connections", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
	}
}

Error VisualShader::_write_node(Type type, StringBuilder &global_code, StringBuilder &code, Vector<VisualShader::DefaultTextureParam> &def_tex_params, const VMap<ConnectionKey, const List<Connection>::Element *> &input_connections, const VMap<ConnectionKey, const List<Connection>::Element *> &output_connections, int node, Set<int> &processed, bool for_preview) const {

	const Ref<VisualShaderNode> vsnode = graph[type]->nodes[node].node;

	//check inputs recursively first
	int input_count = vsnode->get_input_port_count();
	for (int i = 0; i < input_count; i++) {
		ConnectionKey ck;
		ck.node = node;
		ck.port = i;

		if (input_connections.has(ck)) {
			int from_node = input_connections[ck]->get().from_node;
			if (processed.has(from_node)) {
				continue;
			}

			Error err = _write_node(type, global_code, code, def_tex_params, input_connections, output_connections, from_node, processed, for_preview);
			if (err)
				return err;
		}
	}

	// then this node

	code += "// " + vsnode->get_caption() + ":" + itos(node) + "\n";
	Vector<String> input_vars;

	input_vars.resize(vsnode->get_input_port_count());
	String *inputs = input_vars.ptrw();

	for (int i = 0; i < input_count; i++) {
		ConnectionKey ck;
		ck.node = node;
		ck.port = i;

		if (input_connections.has(ck)) {
			//connected to something, use that output
			int from_node = input_connections[ck]->get().from_node;
			int from_port = input_connections[ck]->get().from_port;

			VisualShaderNode::PortType in_type = vsnode->get_input_port_type(i);
			VisualShaderNode::PortType out_type = graph[type]->nodes[from_node].node->get_output_port_type(from_port);

			String src_var = "n_out" + itos(from_node) + "p" + itos(from_port);

			if (in_type == out_type) {
				inputs[i] = src_var;
			} else if (in_type == VisualShaderNode::PORT_TYPE_SCALAR && out_type == VisualShaderNode::PORT_TYPE_VECTOR) {
				inputs[i] = "dot(" + src_var + ",vec3(0.333333,0.333333,0.333333))";
			} else if (in_type == VisualShaderNode::PORT_TYPE_VECTOR && out_type == VisualShaderNode::PORT_TYPE_SCALAR) {
				inputs[i] = "vec3(" + src_var + ")";
			} else if (in_type == VisualShaderNode::PORT_TYPE_BOOLEAN && out_type == VisualShaderNode::PORT_TYPE_VECTOR) {
				inputs[i] = "all(bvec3(" + src_var + "))";
			} else if (in_type == VisualShaderNode::PORT_TYPE_BOOLEAN && out_type == VisualShaderNode::PORT_TYPE_SCALAR) {
				inputs[i] = src_var + ">0.0?true:false";
			} else if (in_type == VisualShaderNode::PORT_TYPE_SCALAR && out_type == VisualShaderNode::PORT_TYPE_BOOLEAN) {
				inputs[i] = src_var + "?1.0:0.0";
			} else if (in_type == VisualShaderNode::PORT_TYPE_VECTOR && out_type == VisualShaderNode::PORT_TYPE_BOOLEAN) {
				inputs[i] = "vec3(" + src_var + "?1.0:0.0)";
			}
		} else {

			Variant defval = vsnode->get_input_port_default_value(i);
			if (defval.get_type() == Variant::REAL || defval.get_type() == Variant::INT) {
				float val = defval;
				inputs[i] = "n_in" + itos(node) + "p" + itos(i);
				code += "\tfloat " + inputs[i] + " = " + vformat("%.5f", val) + ";\n";
			} else if (defval.get_type() == Variant::BOOL) {
				bool val = defval;
				inputs[i] = "n_in" + itos(node) + "p" + itos(i);
				code += "\nbool " + inputs[i] + " = " + (val ? "true" : "false") + ";\n";
			} else if (defval.get_type() == Variant::VECTOR3) {
				Vector3 val = defval;
				inputs[i] = "n_in" + itos(node) + "p" + itos(i);
				code += "\tvec3 " + inputs[i] + " = " + vformat("vec3(%.5f,%.5f,%.5f);\n", val.x, val.y, val.z);
			} else if (defval.get_type() == Variant::TRANSFORM) {
				Transform val = defval;
				val.basis.transpose();
				inputs[i] = "n_in" + itos(node) + "p" + itos(i);
				Array values;
				for (int j = 0; j < 3; j++) {
					values.push_back(val.basis[j].x);
					values.push_back(val.basis[j].y);
					values.push_back(val.basis[j].z);
				}
				values.push_back(val.origin.x);
				values.push_back(val.origin.y);
				values.push_back(val.origin.z);
				bool err = false;
				code += "\tmat4 " + inputs[i] + " = " + String("mat4( vec4(%.5f,%.5f,%.5f,0.0),vec4(%.5f,%.5f,%.5f,0.0),vec4(%.5f,%.5f,%.5f,0.0),vec4(%.5f,%.5f,%.5f,1.0) );\n").sprintf(values, &err);
			} else {
				//will go empty, node is expected to know what it is doing at this point and handle it
			}
		}
	}

	int output_count = vsnode->get_output_port_count();
	Vector<String> output_vars;
	output_vars.resize(vsnode->get_output_port_count());
	String *outputs = output_vars.ptrw();

	for (int i = 0; i < output_count; i++) {

		outputs[i] = "n_out" + itos(node) + "p" + itos(i);
		switch (vsnode->get_output_port_type(i)) {
			case VisualShaderNode::PORT_TYPE_SCALAR: code += String() + "\tfloat " + outputs[i] + ";\n"; break;
			case VisualShaderNode::PORT_TYPE_VECTOR: code += String() + "\tvec3 " + outputs[i] + ";\n"; break;
			case VisualShaderNode::PORT_TYPE_BOOLEAN: code += String() + "\tbool " + outputs[i] + ";\n"; break;
			case VisualShaderNode::PORT_TYPE_TRANSFORM: code += String() + "\tmat4 " + outputs[i] + ";\n"; break;
			default: {
			}
		}
	}

	Vector<VisualShader::DefaultTextureParam> params = vsnode->get_default_texture_parameters(type, node);
	for (int i = 0; i < params.size(); i++) {
		def_tex_params.push_back(params[i]);
	}

	Ref<VisualShaderNodeInput> input = vsnode;
	bool skip_global = input.is_valid() && for_preview;

	if (!skip_global) {
		global_code += vsnode->generate_global(get_mode(), type, node);
	}

	//handle normally
	code += vsnode->generate_code(get_mode(), type, node, inputs, outputs, for_preview);

	code += "\n"; //
	processed.insert(node);

	return OK;
}

void VisualShader::_update_shader() const {
	if (!dirty)
		return;

	dirty = false;

	StringBuilder global_code;
	StringBuilder code;
	Vector<VisualShader::DefaultTextureParam> default_tex_params;
	static const char *shader_mode_str[Shader::MODE_MAX] = { "spatial", "canvas_item", "particles" };

	global_code += String() + "shader_type " + shader_mode_str[shader_mode] + ";\n";

	String render_mode;

	{
		//fill render mode enums
		int idx = 0;
		while (render_mode_enums[idx].string) {

			if (shader_mode == render_mode_enums[idx].mode) {

				if (modes.has(render_mode_enums[idx].string)) {

					int which = modes[render_mode_enums[idx].string];
					int count = 0;
					for (int i = 0; i < ShaderTypes::get_singleton()->get_modes(VisualServer::ShaderMode(shader_mode)).size(); i++) {
						String mode = ShaderTypes::get_singleton()->get_modes(VisualServer::ShaderMode(shader_mode))[i];
						if (mode.begins_with(render_mode_enums[idx].string)) {
							if (count == which) {
								if (render_mode != String()) {
									render_mode += ", ";
								}
								render_mode += mode;
								break;
							}
							count++;
						}
					}
				}
			}
			idx++;
		}

		//fill render mode flags
		for (int i = 0; i < ShaderTypes::get_singleton()->get_modes(VisualServer::ShaderMode(shader_mode)).size(); i++) {

			String mode = ShaderTypes::get_singleton()->get_modes(VisualServer::ShaderMode(shader_mode))[i];
			if (flags.has(mode)) {
				if (render_mode != String()) {
					render_mode += ", ";
				}
				render_mode += mode;
			}
		}
	}

	if (render_mode != String()) {

		global_code += "render_mode " + render_mode + ";\n\n";
	}

	List<Graph *> graphs;

	//custom functions should be written first
	for (int i = TYPE_MAX; i < get_function_count(); i++) {

		//make it faster to go around through shader
		VMap<ConnectionKey, const List<Connection>::Element *> input_connections;
		VMap<ConnectionKey, const List<Connection>::Element *> output_connections;

		for (const List<Connection>::Element *E = graph[i]->connections.front(); E; E = E->next()) {
			ConnectionKey from_key;
			from_key.node = E->get().from_node;
			from_key.port = E->get().from_port;

			output_connections.insert(from_key, E);

			ConnectionKey to_key;
			to_key.node = E->get().to_node;
			to_key.port = E->get().to_port;

			input_connections.insert(to_key, E);
		}

		code += "\n";

		Ref<VisualShaderNodeOutput> output = graph[i]->nodes[NODE_ID_OUTPUT].node;
		switch (output->get_input_port_type(0)) {
			case VisualShaderNode::PORT_TYPE_SCALAR:
				code += "float";
				break;
			case VisualShaderNode::PORT_TYPE_VECTOR:
				code += "vec3";
				break;
			case VisualShaderNode::PORT_TYPE_BOOLEAN:
				code += "bool";
				break;
			case VisualShaderNode::PORT_TYPE_TRANSFORM:
				code += "mat4";
				break;
			default:
				break;
		}

		code += " " + graph[i]->name + "(";
		code += graph[i]->inputs_str;
		code += ") {\n";

		Set<int> processed;
		Error err = _write_node(Type(i), global_code, code, default_tex_params, input_connections, output_connections, NODE_ID_OUTPUT, processed, false);
		ERR_FAIL_COND(err != OK);

		code += "}\n";
	}

	for (int i = 0; i < TYPE_MAX; i++) {
		//make it faster to go around through shader
		VMap<ConnectionKey, const List<Connection>::Element *> input_connections;
		VMap<ConnectionKey, const List<Connection>::Element *> output_connections;

		for (const List<Connection>::Element *E = graph[i]->connections.front(); E; E = E->next()) {
			ConnectionKey from_key;
			from_key.node = E->get().from_node;
			from_key.port = E->get().from_port;

			output_connections.insert(from_key, E);

			ConnectionKey to_key;
			to_key.node = E->get().to_node;
			to_key.port = E->get().to_port;

			input_connections.insert(to_key, E);
		}

		code += "\nvoid " + graph[i]->name + "() {\n";

		Set<int> processed;
		Error err = _write_node(Type(i), global_code, code, default_tex_params, input_connections, output_connections, NODE_ID_OUTPUT, processed, false);
		ERR_FAIL_COND(err != OK);

		code += "}\n";
	}

	//set code secretly
	global_code += "\n\n";
	String final_code = global_code;
	final_code += code;
	const_cast<VisualShader *>(this)->set_code(final_code);
	for (int i = 0; i < default_tex_params.size(); i++) {
		const_cast<VisualShader *>(this)->set_default_texture_param(default_tex_params[i].name, default_tex_params[i].param);
	}
}

void VisualShader::_queue_update() {
	if (dirty) {
		return;
	}

	dirty = true;
	call_deferred("_update_shader");
}

void VisualShader::_input_type_changed(Type p_type, int p_id) {
	//erase connections using this input, as type changed
	Graph *g = get_graph_set(p_type);

	for (List<Connection>::Element *E = g->connections.front(); E;) {
		List<Connection>::Element *N = E->next();
		if (E->get().from_node == p_id) {
			g->connections.erase(E);
		}
		E = N;
	}
}

void VisualShader::rebuild() {
	dirty = true;
	_update_shader();
}

void VisualShader::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_mode", "mode"), &VisualShader::set_mode);

	ClassDB::bind_method(D_METHOD("add_node", "type", "node", "position", "id"), &VisualShader::add_node);
	ClassDB::bind_method(D_METHOD("get_node", "type", "id"), &VisualShader::get_node);

	ClassDB::bind_method(D_METHOD("set_node_position", "type", "id", "position"), &VisualShader::set_node_position);
	ClassDB::bind_method(D_METHOD("get_node_position", "type", "id"), &VisualShader::get_node_position);

	ClassDB::bind_method(D_METHOD("get_node_list", "type"), &VisualShader::get_node_list);
	ClassDB::bind_method(D_METHOD("get_valid_node_id", "type"), &VisualShader::get_valid_node_id);

	ClassDB::bind_method(D_METHOD("remove_node", "type", "id"), &VisualShader::remove_node);
	ClassDB::bind_method(D_METHOD("rebuild"), &VisualShader::rebuild);

	ClassDB::bind_method(D_METHOD("is_node_connection", "type", "from_node", "from_port", "to_node", "to_port"), &VisualShader::is_node_connection);
	ClassDB::bind_method(D_METHOD("can_connect_nodes", "type", "from_node", "from_port", "to_node", "to_port"), &VisualShader::is_node_connection);

	ClassDB::bind_method(D_METHOD("connect_nodes", "type", "from_node", "from_port", "to_node", "to_port"), &VisualShader::connect_nodes);
	ClassDB::bind_method(D_METHOD("disconnect_nodes", "type", "from_node", "from_port", "to_node", "to_port"), &VisualShader::disconnect_nodes);
	ClassDB::bind_method(D_METHOD("connect_nodes_forced", "type", "from_node", "from_port", "to_node", "to_port"), &VisualShader::connect_nodes_forced);

	ClassDB::bind_method(D_METHOD("get_node_connections", "type"), &VisualShader::_get_node_connections);

	ClassDB::bind_method(D_METHOD("set_graph_offset", "offset"), &VisualShader::set_graph_offset);
	ClassDB::bind_method(D_METHOD("get_graph_offset"), &VisualShader::get_graph_offset);

	ClassDB::bind_method(D_METHOD("_queue_update"), &VisualShader::_queue_update);
	ClassDB::bind_method(D_METHOD("_update_shader"), &VisualShader::_update_shader);

	ClassDB::bind_method(D_METHOD("_input_type_changed"), &VisualShader::_input_type_changed);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "graph_offset", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_graph_offset", "get_graph_offset");

	BIND_ENUM_CONSTANT(TYPE_VERTEX);
	BIND_ENUM_CONSTANT(TYPE_FRAGMENT);
	BIND_ENUM_CONSTANT(TYPE_LIGHT);
	BIND_ENUM_CONSTANT(TYPE_MAX);

	BIND_CONSTANT(NODE_ID_INVALID);
	BIND_CONSTANT(NODE_ID_OUTPUT);
}

VisualShader::VisualShader() {
	shader_mode = Shader::MODE_SPATIAL;

	for (int i = 0; i < TYPE_MAX; i++) {

		Graph *g = memnew(Graph);
		g->index = -1;
		switch (i) {
			case 0:
				g->name = "vertex";
				break;
			case 1:
				g->name = "fragment";
				break;
			case 2:
				g->name = "light";
				break;
		}
		graph.push_back(g);

		Ref<VisualShaderNodeOutput> output;
		output.instance();
		output->shader_name = g->name;
		output->shader_mode = shader_mode;
		graph[i]->nodes[NODE_ID_OUTPUT].node = output;
		graph[i]->nodes[NODE_ID_OUTPUT].position = Vector2(400, 150);
	}

	dirty = true;
}

///////////////////////////////////////////////////////////

Map<String, List<VisualShaderNodeInput::Port> > VisualShaderNodeInput::ports;

const VisualShaderNodeInput::Port VisualShaderNodeInput::temp_ports[] = {
	// Spatial, Vertex
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "vertex", "VERTEX" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "normal", "NORMAL" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "tangent", "TANGENT" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "binormal", "BINORMAL" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "uv", "vec3(UV,0.0)" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "uv2", "vec3(UV2,0.0)" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "color", "COLOR.rgb" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "alpha", "COLOR.a" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "point_size", "POINT_SIZE" },

	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_TRANSFORM, "world", "WORLD_MATRIX" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_TRANSFORM, "modelview", "MODELVIEW_MATRIX" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_TRANSFORM, "camera", "CAMERA_MATRIX" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_TRANSFORM, "inv_camera", "INV_CAMERA_MATRIX" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_TRANSFORM, "projection", "PROJECTION_MATRIX" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_TRANSFORM, "inv_projection", "INV_PROJECTION_MATRIX" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "time", "TIME" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "viewport_size", "vec3(VIEWPORT_SIZE, 0)" },

	// Spatial, Fragment
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "fragcoord", "FRAGCOORD.xyz" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "vertex", "VERTEX" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "normal", "NORMAL" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "tangent", "TANGENT" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "binormal", "BINORMAL" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "view", "VIEW" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "uv", "vec3(UV,0.0)" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "uv2", "vec3(UV2,0.0)" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "color", "COLOR.rgb" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "alpha", "COLOR.a" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "point_coord", "vec3(POINT_COORD,0.0)" },

	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "screen_uv", "vec3(SCREEN_UV,0.0)" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "side", "float(FRONT_FACING ? 1.0 : 0.0)" },

	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_TRANSFORM, "world", "WORLD_MATRIX" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_TRANSFORM, "inv_camera", "INV_CAMERA_MATRIX" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_TRANSFORM, "camera", "CAMERA_MATRIX" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_TRANSFORM, "projection", "PROJECTION_MATRIX" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_TRANSFORM, "inv_projection", "INV_PROJECTION_MATRIX" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "time", "TIME" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "viewport_size", "vec3(VIEWPORT_SIZE, 0.0)" },

	// Spatial, Light
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "fragcoord", "FRAGCOORD.xyz" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "normal", "NORMAL" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "view", "VIEW" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "light", "LIGHT" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "light_color", "LIGHT_COLOR" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "attenuation", "ATTENUATION" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "albedo", "ALBEDO" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "transmission", "TRANSMISSION" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "diffuse", "DIFFUSE_LIGHT" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "specular", "SPECULAR_LIGHT" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_SCALAR, "roughness", "ROUGHNESS" },

	{ Shader::MODE_SPATIAL, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_TRANSFORM, "world", "WORLD_MATRIX" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_TRANSFORM, "inv_camera", "INV_CAMERA_MATRIX" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_TRANSFORM, "camera", "CAMERA_MATRIX" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_TRANSFORM, "projection", "PROJECTION_MATRIX" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_TRANSFORM, "inv_projection", "INV_PROJECTION_MATRIX" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_SCALAR, "time", "TIME" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "viewport_size", "vec3(VIEWPORT_SIZE, 0.0)" },
	// Canvas Item, Vertex
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "vertex", "vec3(VERTEX,0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "uv", "vec3(UV,0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "color", "COLOR.rgb" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "alpha", "COLOR.a" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "point_size", "POINT_SIZE" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "texture_pixel_size", "vec3(TEXTURE_PIXEL_SIZE, 1.0)" },

	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_TRANSFORM, "world", "WORLD_MATRIX" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_TRANSFORM, "projection", "PROJECTION_MATRIX" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_TRANSFORM, "extra", "EXTRA_MATRIX" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "time", "TIME" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "light_pass", "float(AT_LIGHT_PASS ? 1.0 : 0.0)" },
	// Canvas Item, Fragment
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "fragcoord", "FRAGCOORD.xyz" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "uv", "vec3(UV,0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "color", "COLOR.rgb" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "alpha", "COLOR.a" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "screen_uv", "vec3(SCREEN_UV,0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "texture_pixel_size", "vec3(TEXTURE_PIXEL_SIZE, 1.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "screen_pixel_size", "vec3(SCREEN_PIXEL_SIZE, 1.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "point_coord", "vec3(POINT_COORD,0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "time", "TIME" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "light_pass", "float(AT_LIGHT_PASS ? 1.0 : 0.0)" },
	// Canvas Item, Light
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "fragcoord", "FRAGCOORD.xyz" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "uv", "vec3(UV,0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "normal", "NORMAL" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "color", "COLOR.rgb" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_SCALAR, "alpha", "COLOR.a" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "light_vec", "vec3(LIGHT_VEC,0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_SCALAR, "light_height", "LIGHT_HEIGHT" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "light_color", "LIGHT_COLOR.rgb" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "light_alpha", "LIGHT_COLOR.a" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "light_uv", "vec3(LIGHT_UV,0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "shadow_color", "SHADOW_COLOR.rgb" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "screen_uv", "vec3(SCREEN_UV,0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "texture_pixel_size", "vec3(TEXTURE_PIXEL_SIZE, 1.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "point_coord", "vec3(POINT_COORD,0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_SCALAR, "time", "TIME" },

	// Particles, Vertex
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "color", "COLOR.rgb" },
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "alpha", "COLOR.a" },
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "velocity", "VELOCITY" },
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "restart", "float(RESTART ? 1.0 : 0.0)" },
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "active", "float(ACTIVE ? 1.0 : 0.0)" },
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "custom", "CUSTOM.rgb" },
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "custom_alpha", "CUSTOM.a" },
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_TRANSFORM, "transform", "TRANSFORM" },

	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "delta", "DELTA" },
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "lifetime", "LIFETIME" },
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "index", "float(INDEX)" },

	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_TRANSFORM, "emission_transform", "EMISSION_TRANSFORM" },
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "time", "TIME" },
	{ Shader::MODE_MAX, VisualShader::TYPE_MAX, VisualShaderNode::PORT_TYPE_TRANSFORM, "", "" },
};

const VisualShaderNodeInput::Port VisualShaderNodeInput::preview_ports[] = {

	// Spatial, Fragment
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "normal", "vec3(0.0,0.0,1.0)" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "tangent", "vec3(0.0,1.0,0.0)" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "binormal", "vec3(1.0,0.0,0.0)" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "uv", "vec3(UV,0.0)" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "uv2", "vec3(UV,0.0)" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "color", "vec3(1.0)" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "alpha", "1.0" },

	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "screen_uv", "vec3(SCREEN_UV,0.0)" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "side", "1.0" },

	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "time", "TIME" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "viewport_size", "vec3(1.0,1.0, 0.0)" },

	// Spatial, Light
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "normal", "vec3(0.0,0.0,1.0)" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_SCALAR, "time", "TIME" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "viewport_size", "vec3(1.0, 1.0, 0.0)" },
	// Canvas Item, Vertex
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "vertex", "vec3(VERTEX,0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "uv", "vec3(UV,0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "color", "vec3(1.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "alpha", "1.0" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "time", "TIME" },
	// Canvas Item, Fragment
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "uv", "vec3(UV,0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "color", "vec3(1.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "alpha", "1.0" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "screen_uv", "vec3(SCREEN_UV,0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "time", "TIME" },
	// Canvas Item, Light
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "uv", "vec3(UV,0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "normal", "vec3(0.0,0.0,1.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "color", "vec3(1.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_SCALAR, "alpha", "1.0" },

	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "screen_uv", "vec3(SCREEN_UV,0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_SCALAR, "time", "TIME" },
	// Particles, Vertex
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "color", "vec3(1.0)" },
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "alpha", "1.0" },
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "velocity", "vec3(0.0,0.0,1.0)" },
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "time", "TIME" },
	{ Shader::MODE_MAX, VisualShader::TYPE_MAX, VisualShaderNode::PORT_TYPE_TRANSFORM, "", "" },
};

void VisualShaderNodeInput::_init_default_ports() { // static

	static bool is_initialized = false;
	if (!is_initialized) {
		int idx = 0;
		while (temp_ports[idx].mode != Shader::MODE_MAX) {
			VisualShaderNodeInput::Port port;
			port.mode = temp_ports[idx].mode;
			port.name = temp_ports[idx].name;
			port.shader_type = temp_ports[idx].shader_type;
			port.string = temp_ports[idx].string;
			port.type = temp_ports[idx].type;
			String shader_name;
			switch (temp_ports[idx].shader_type) {
				case VisualShader::TYPE_VERTEX:
					shader_name = "vertex";
					break;
				case VisualShader::TYPE_FRAGMENT:
					shader_name = "fragment";
					break;
				case VisualShader::TYPE_LIGHT:
					shader_name = "light";
					break;
				default:
					break;
			}
			ports[shader_name].push_back(port);
			idx++;
		}
		is_initialized = true;
	}
}

void VisualShaderNodeInput::rename_ports_func(const String &p_func_name, const String &p_new_func_name) { // static

	for (int port_id = 0; port_id < ports[p_func_name].size(); port_id++) {
		ports[p_new_func_name].push_back(ports[p_func_name][port_id]);
	}
	ports[p_func_name].clear();
	ports.erase(p_func_name);
}

int VisualShaderNodeInput::get_input_port_count() const {

	return 0;
}

VisualShaderNodeInput::PortType VisualShaderNodeInput::get_input_port_type(int p_port) const {

	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeInput::get_input_port_name(int p_port) const {

	return "";
}

bool VisualShaderNodeInput::is_custom_func(const String &p_func_name) {
	return p_func_name != "vertex" && p_func_name != "fragment" && p_func_name != "light";
}

void VisualShaderNodeInput::clear_custom_funcs() { // static

	for (Map<String, List<Port> >::Element *E = ports.front(); E; E = E->next()) {
		if (is_custom_func(E->key())) {
			ports.erase(E->key());
		}
	}
}

int VisualShaderNodeInput::get_func_port_count(const String &p_func_name) { // static

	ERR_FAIL_COND_V(!ports.has(p_func_name), 0);
	return ports[p_func_name].size();
}

void VisualShaderNodeInput::remove_func(const String &p_func_name) { // static
	if (ports.has(p_func_name)) {
		ports.erase(p_func_name);
	}
}

void VisualShaderNodeInput::add_custom_port(const String &p_func_name, const String &p_name, int p_type) { // static

	Port port;
	port.name = p_name;
	port.mode = Shader::Mode(-1);
	port.type = (VisualShaderNode::PortType)p_type;
	port.string = "p_" + p_name;
	ports[p_func_name].push_back(port);
}

void VisualShaderNodeInput::set_custom_port_name(const String &p_func_name, int p_port_id, const String &p_name) { // static

	ERR_FAIL_COND(!ports.has(p_func_name));
	ERR_FAIL_INDEX(p_port_id, ports[p_func_name].size())
	ports[p_func_name][p_port_id].name = p_name;
}

void VisualShaderNodeInput::set_custom_port_type(const String &p_func_name, int p_port_id, int p_type) { // static

	ERR_FAIL_COND(!ports.has(p_func_name));
	ERR_FAIL_INDEX(p_port_id, ports[p_func_name].size())
	ports[p_func_name][p_port_id].type = (VisualShaderNode::PortType)p_type;
}

String VisualShaderNodeInput::get_func_port_name(const String &p_func_name, int p_id) { // static

	ERR_FAIL_COND_V(!ports.has(p_func_name), "");
	return ports[p_func_name][p_id].name;
}

int VisualShaderNodeInput::get_func_port_type(const String &p_func_name, int p_id) { // static

	ERR_FAIL_COND_V(!ports.has(p_func_name), 0);
	return ports[p_func_name][p_id].type;
}

int VisualShaderNodeInput::get_output_port_count() const {

	return 1;
}

VisualShaderNodeInput::PortType VisualShaderNodeInput::get_output_port_type(int p_port) const {

	return get_input_type_by_name(input_name);
}

String VisualShaderNodeInput::get_output_port_name(int p_port) const {

	return "";
}

String VisualShaderNodeInput::get_caption() const {

	return TTR("Input");
}

String VisualShaderNodeInput::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {

	String code;

	if (p_for_preview) {
		int idx = 0;

		while (preview_ports[idx].mode != Shader::MODE_MAX) {
			if (preview_ports[idx].mode == shader_mode && preview_ports[idx].shader_type == shader_type && preview_ports[idx].name == input_name) {
				code = "\t" + p_output_vars[0] + " = " + preview_ports[idx].string + ";\n";
				break;
			}
			idx++;
		}

		if (code == String()) {

			for (Map<String, List<Port> >::Element *E = ports.front(); E; E = E->next()) {
				if (is_custom_func(E->key())) {
					for (List<Port>::Element *P = E->get().front(); P; P = P->next()) {
						if (P->get().name == input_name) {
							code = "\t" + p_output_vars[0] + " = " + P->get().string + ";\n";
							break;
						}
					}
				}
			}

			if (code == String()) {

				switch (get_output_port_type(0)) {
					case PORT_TYPE_SCALAR: {
						code = "\t" + p_output_vars[0] + " = 0.0;\n";
					} break; //default (none found) is scalar
					case PORT_TYPE_VECTOR: {
						code = "\t" + p_output_vars[0] + " = vec3(0.0);\n";
					} break; //default (none found) is scalar
					case PORT_TYPE_TRANSFORM: {
						code = "\t" + p_output_vars[0] + " = mat4( vec4(1.0,0.0,0.0,0.0), vec4(0.0,1.0,0.0,0.0), vec4(0.0,0.0,1.0,0.0), vec4(0.0,0.0,0.0,1.0) );\n";
					} break; //default (none found) is scalar
					case PORT_TYPE_BOOLEAN: {
						code = "\t" + p_output_vars[0] + " = false;\n";
					} break;
					default:
						break;
				}
			}
		}

		return code;

	} else {

		for (int port_id = 0; port_id < ports[shader_name].size(); port_id++) {
			Port port = ports[shader_name][port_id];
			if ((port.mode == shader_mode || shader_type >= VisualShader::TYPE_MAX) && port.name == input_name) {
				code = "\t" + p_output_vars[0] + " = " + port.string + ";\n";
				break;
			}
		}
		if (code == String()) {
			code = "\t" + p_output_vars[0] + " = 0.0;\n"; //default (none found) is scalar
		}

		return code;
	}
}

void VisualShaderNodeInput::set_input_name(String p_name) {

	PortType prev_type = get_input_type_by_name(input_name);
	input_name = p_name;
	emit_changed();
	if (get_input_type_by_name(input_name) != prev_type) {
		emit_signal("input_type_changed");
	}
}

String VisualShaderNodeInput::get_input_name() const {

	return input_name;
}

VisualShaderNodeInput::PortType VisualShaderNodeInput::get_input_type_by_name(String p_name) const {

	for (int port_id = 0; port_id < ports[shader_name].size(); port_id++) {
		Port port = ports[shader_name][port_id];
		if (port.mode == shader_mode) {
			if (port.name == p_name) {
				return port.type;
			}
		}
	}

	return PORT_TYPE_SCALAR;
}

int VisualShaderNodeInput::get_input_index_count() const {

	int count = 0;
	for (List<Port>::Element *E = ports[shader_name].front(); E; E = E->next()) {
		if (E->get().mode == shader_mode) {
			count++;
		}
	}
	return count;
}

VisualShaderNodeInput::PortType VisualShaderNodeInput::get_input_index_type(int p_index) const {

	List<Port> filtered_ports;
	for (List<Port>::Element *E = ports[shader_name].front(); E; E = E->next()) {
		if (E->get().mode == shader_mode) {
			filtered_ports.push_back(E->get());
		}
	}
	return filtered_ports[p_index].type;
}

String VisualShaderNodeInput::get_input_index_name(int p_index) const {

	List<Port> filtered_ports;
	for (List<Port>::Element *E = ports[shader_name].front(); E; E = E->next()) {
		if (E->get().mode == shader_mode) {
			filtered_ports.push_back(E->get());
		}
	}
	return filtered_ports[p_index].name;
}

void VisualShaderNodeInput::_validate_property(PropertyInfo &property) const {

	if (property.name == "input_name") {
		String port_list;

		for (int port_id = 0; port_id < ports[shader_name].size(); port_id++) {
			Port port = ports[shader_name][port_id];
			if (port.mode == shader_mode) {
				if (port_list != String()) {
					port_list += ",";
				}
				port_list += port.name;
			}
		}

		if (port_list == "") {
			port_list = TTR("None");
		}
		property.hint_string = port_list;
	}
}

Vector<StringName> VisualShaderNodeInput::get_editable_properties() const {

	Vector<StringName> props;
	props.push_back("input_name");
	return props;
}

void VisualShaderNodeInput::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_input_name", "name"), &VisualShaderNodeInput::set_input_name);
	ClassDB::bind_method(D_METHOD("get_input_name"), &VisualShaderNodeInput::get_input_name);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "input_name", PROPERTY_HINT_ENUM, ""), "set_input_name", "get_input_name");
	ADD_SIGNAL(MethodInfo("input_type_changed"));
}

VisualShaderNodeInput::VisualShaderNodeInput() {
	_init_default_ports();

	input_name = "[None]";
	// changed when set
	shader_type = VisualShader::TYPE_MAX;
	shader_mode = Shader::MODE_MAX;
	shader_name = "";
}

////////////////////////////////////////////

Map<String, List<VisualShaderNodeOutput::Port> > VisualShaderNodeOutput::ports;

const VisualShaderNodeOutput::Port VisualShaderNodeOutput::temp_ports[] = {
	// Spatial, Vertex
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "vertex", "VERTEX" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "normal", "NORMAL" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "tangent", "TANGENT" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "binormal", "BINORMAL" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "uv", "UV:xy" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "uv2", "UV2:xy" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "color", "COLOR.rgb" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "alpha", "COLOR.a" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "roughness", "ROUGHNESS" },
	// Spatial, Fragment

	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "albedo", "ALBEDO" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "alpha", "ALPHA" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "metallic", "METALLIC" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "roughness", "ROUGHNESS" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "specular", "SPECULAR" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "emission", "EMISSION" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "ao", "AO" },

	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "normal", "NORMAL" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "normalmap", "NORMALMAP" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "normalmap_depth", "NORMALMAP_DEPTH" },

	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "rim", "RIM" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "rim_tint", "RIM_TINT" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "clearcoat", "CLEARCOAT" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "clearcoat_gloss", "CLEARCOAT_GLOSS" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "anisotropy", "ANISOTROPY" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "anisotropy_flow", "ANISOTROPY_FLOW:xy" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "subsurf_scatter", "SSS_STRENGTH" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "transmission", "TRANSMISSION" },

	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "alpha_scissor", "ALPHA_SCISSOR" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "ao_light_affect", "AO_LIGHT_AFFECT" },

	// Spatial, Light
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "diffuse", "DIFFUSE_LIGHT" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "specular", "SPECULAR_LIGHT" },
	// Canvas Item, Vertex
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "vertex", "VERTEX:xy" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "uv", "UV:xy" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "color", "COLOR.rgb" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "alpha", "COLOR.a" },
	// Canvas Item, Fragment
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "color", "COLOR.rgb" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "alpha", "COLOR.a" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "normal", "NORMAL" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "normalmap", "NORMALMAP" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "normalmap_depth", "NORMALMAP_DEPTH" },
	// Canvas Item, Light
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "light", "LIGHT.rgb" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_SCALAR, "light_alpha", "LIGHT.rgb" },
	// Particles, Vertex
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "color", "COLOR.rgb" },
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "alpha", "COLOR.a" },
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "velocity", "VELOCITY" },
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "custom", "CUSTOM.rgb" },
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "custom_alpha", "CUSTOM.a" },
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_TRANSFORM, "transform", "TRANSFORM" },
	{ Shader::MODE_MAX, VisualShader::TYPE_MAX, VisualShaderNode::PORT_TYPE_TRANSFORM, "", "" },
};

void VisualShaderNodeOutput::_init_default_ports() { // static
	static bool is_initialized = false;
	if (!is_initialized) {
		int idx = 0;
		while (temp_ports[idx].mode != Shader::MODE_MAX) {
			VisualShaderNodeOutput::Port port;
			port.mode = temp_ports[idx].mode;
			port.name = temp_ports[idx].name;
			port.shader_type = temp_ports[idx].shader_type;
			port.string = temp_ports[idx].string;
			port.type = temp_ports[idx].type;
			String shader_name;
			switch (temp_ports[idx].shader_type) {
				case VisualShader::TYPE_VERTEX:
					shader_name = "vertex";
					break;
				case VisualShader::TYPE_FRAGMENT:
					shader_name = "fragment";
					break;
				case VisualShader::TYPE_LIGHT:
					shader_name = "light";
					break;
				default:
					break;
			}
			ports[shader_name].push_back(port);
			idx++;
		}
		is_initialized = true;
	}
}

void VisualShaderNodeOutput::rename_ports_func(const String &p_func_name, const String &p_new_func_name) { // static

	for (int port_id = 0; port_id < ports[p_func_name].size(); port_id++) {
		ports[p_new_func_name].push_back(ports[p_func_name][port_id]);
	}
	ports[p_func_name].clear();
	ports.erase(p_func_name);
}

int VisualShaderNodeOutput::get_input_port_count() const {
	if (is_custom_func(shader_name)) {
		return 1;
	}

	int count = 0;
	for (int port_id = 0; port_id < ports[shader_name].size(); port_id++) {
		Port port = ports[shader_name][port_id];
		if (port.mode == shader_mode) {
			count++;
		}
	}
	return count;
}

VisualShaderNodeOutput::PortType VisualShaderNodeOutput::get_input_port_type(int p_port) const {

	int count = 0;

	for (int port_id = 0; port_id < ports[shader_name].size(); port_id++) {
		Port port = ports[shader_name][port_id];
		if (port.mode == shader_mode) {
			if (count == p_port) {
				return port.type;
			}
			count++;
		}
	}

	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeOutput::get_input_port_name(int p_port) const {

	for (int port_id = 0; port_id < ports[shader_name].size(); port_id++) {
		Port port = ports[shader_name][port_id];
		if (port.mode == shader_mode) {
			if (port_id == p_port) {
				return String(port.name).capitalize();
			}
		}
	}

	return String();
}

Variant VisualShaderNodeOutput::get_input_port_default_value(int p_port) const {
	return Variant();
}

int VisualShaderNodeOutput::get_output_port_count() const {

	return 0;
}
VisualShaderNodeOutput::PortType VisualShaderNodeOutput::get_output_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}
String VisualShaderNodeOutput::get_output_port_name(int p_port) const {
	return String();
}

void VisualShaderNodeOutput::add_port(const String &p_func_name, const String &p_name, PortType p_type) {
	Port port;
	port.mode = Shader::MODE_MAX;
	port.name = p_name;
	port.type = p_type;
	port.string = "RESULT";
	if (ports[p_func_name].empty()) {
		ports[p_func_name].push_back(port);
	} else {
		ports[p_func_name][0] = port;
	}
}

bool VisualShaderNodeOutput::is_custom_func(const String &p_func_name) {
	return p_func_name != "vertex" && p_func_name != "fragment" && p_func_name != "light";
}

void VisualShaderNodeOutput::clear_custom_funcs() { // static

	for (Map<String, List<Port> >::Element *E = ports.front(); E; E = E->next()) {
		if (is_custom_func(E->key())) {
			ports.erase(E->key());
		}
	}
}

void VisualShaderNodeOutput::remove_func(const String &p_func_name) { // static
	if (ports.has(p_func_name)) {
		ports.erase(p_func_name);
	}
}

bool VisualShaderNodeOutput::is_port_separator(int p_index) const {

	if (shader_mode == Shader::MODE_SPATIAL && shader_name == "fragment") {
		String name = get_input_port_name(p_index);
		return (name == "Normal" || name == "Rim" || name == "Alpha Scissor");
	}
	return false;
}

String VisualShaderNodeOutput::get_caption() const {
	return TTR("Output");
}

String VisualShaderNodeOutput::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {

	String code;

	int count = 0;

	if (is_custom_func(shader_name)) {

		String s;
		if (p_input_vars[0] == String("")) {
			switch (get_input_port_type(0)) {
				case 0: // Scalar
					s = "0.0";
					break;
				case 1: // Vector
					s = "vec3(0.0, 0.0, 0.0)";
					break;
				case 2: // Boolean
					s = "false";
					break;
				case 3: // Transform
					s = "mat4(1.0)";
					break;
				default:
					break;
			}
		} else {
			s = p_input_vars[0];
		}
		code += "\treturn " + s + ";\n";

	} else {

		for (int port_id = 0; port_id < ports[shader_name].size(); port_id++) {
			Port port = ports[shader_name][port_id];
			if (port.mode == shader_mode) {
				if (p_input_vars[count] != String()) {
					String s = port.string;
					if (s.find(":") != -1) {
						code += "\t" + s.get_slicec(':', 0) + " = " + p_input_vars[count] + "." + s.get_slicec(':', 1) + ";\n";
					} else {
						code += "\t" + s + " = " + p_input_vars[count] + ";\n";
					}
				}
				count++;
			}
		}
	}

	return code;
}

VisualShaderNodeOutput::VisualShaderNodeOutput() {
	_init_default_ports();
	shader_name = "";
}

///////////////////////////

void VisualShaderNodeUniform::set_uniform_name(const String &p_name) {
	uniform_name = p_name;
	emit_signal("name_changed");
	emit_changed();
}

String VisualShaderNodeUniform::get_uniform_name() const {
	return uniform_name;
}

void VisualShaderNodeUniform::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_uniform_name", "name"), &VisualShaderNodeUniform::set_uniform_name);
	ClassDB::bind_method(D_METHOD("get_uniform_name"), &VisualShaderNodeUniform::get_uniform_name);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "uniform_name"), "set_uniform_name", "get_uniform_name");
}

VisualShaderNodeUniform::VisualShaderNodeUniform() {
}

////////////// GroupBase

String VisualShaderNodeGroupBase::get_caption() const {
	return "Group";
}

void VisualShaderNodeGroupBase::set_size(const Vector2 &p_size) {
	size = p_size;
}

Vector2 VisualShaderNodeGroupBase::get_size() const {
	return size;
}

void VisualShaderNodeGroupBase::set_inputs(const String &p_inputs) {

	if (inputs == p_inputs)
		return;

	clear_input_ports();

	inputs = p_inputs;

	Vector<String> input_strings = inputs.split(";", false);

	int input_port_count = input_strings.size();

	for (int i = 0; i < input_port_count; i++) {

		Vector<String> arr = input_strings[i].split(",");
		ERR_FAIL_COND(arr.size() != 3);

		int port_idx = arr[0].to_int();
		int port_type = arr[1].to_int();
		String port_name = arr[2];

		Port port;
		port.type = (PortType)port_type;
		port.name = port_name;
		input_ports[port_idx] = port;
	}
}

String VisualShaderNodeGroupBase::get_inputs() const {
	return inputs;
}

void VisualShaderNodeGroupBase::set_outputs(const String &p_outputs) {

	if (outputs == p_outputs)
		return;

	clear_output_ports();

	outputs = p_outputs;

	Vector<String> output_strings = outputs.split(";", false);

	int output_port_count = output_strings.size();

	for (int i = 0; i < output_port_count; i++) {

		Vector<String> arr = output_strings[i].split(",");
		ERR_FAIL_COND(arr.size() != 3);

		int port_idx = arr[0].to_int();
		int port_type = arr[1].to_int();
		String port_name = arr[2];

		Port port;
		port.type = (PortType)port_type;
		port.name = port_name;
		output_ports[port_idx] = port;
	}
}

String VisualShaderNodeGroupBase::get_outputs() const {
	return outputs;
}

void VisualShaderNodeGroupBase::set_editable(bool p_enabled) {
	editable = p_enabled;
}

bool VisualShaderNodeGroupBase::is_editable() const {
	return editable;
}

void VisualShaderNodeGroupBase::set_resizable(bool p_enabled) {
	resizable = p_enabled;
}

bool VisualShaderNodeGroupBase::is_resizable() const {
	return resizable;
}

bool VisualShaderNodeGroupBase::is_valid_port_name(const String &p_name) const {
	if (!p_name.is_valid_identifier()) {
		return false;
	}
	for (int i = 0; i < get_input_port_count(); i++) {
		if (get_input_port_name(i) == p_name) {
			return false;
		}
	}
	for (int i = 0; i < get_output_port_count(); i++) {
		if (get_output_port_name(i) == p_name) {
			return false;
		}
	}
	return true;
}

void VisualShaderNodeGroupBase::add_input_port(int p_id, int p_type, const String &p_name) {

	String str = itos(p_id) + "," + itos(p_type) + "," + p_name + ";";
	Vector<String> inputs_strings = inputs.split(";", false);
	int index = 0;
	if (p_id < inputs_strings.size()) {
		for (int i = 0; i < inputs_strings.size(); i++) {
			if (i == p_id) {
				inputs = inputs.insert(index, str);
				break;
			}
			index += inputs_strings[i].size();
		}
	} else {
		inputs += str;
	}

	inputs_strings = inputs.split(";", false);
	index = 0;

	for (int i = 0; i < inputs_strings.size(); i++) {
		int count = 0;
		for (int j = 0; j < inputs_strings[i].size(); j++) {
			if (inputs_strings[i][j] == ',') {
				break;
			}
			count++;
		}

		inputs.erase(index, count);
		inputs = inputs.insert(index, itos(i));
		index += inputs_strings[i].size();
	}

	_apply_port_changes();
}

void VisualShaderNodeGroupBase::remove_input_port(int p_id) {

	ERR_FAIL_COND(!has_input_port(p_id));

	Vector<String> inputs_strings = inputs.split(";", false);
	int count = 0;
	int index = 0;
	for (int i = 0; i < inputs_strings.size(); i++) {
		Vector<String> arr = inputs_strings[i].split(",");
		if (arr[0].to_int() == p_id) {
			count = inputs_strings[i].size();
			break;
		}
		index += inputs_strings[i].size();
	}
	inputs.erase(index, count);

	inputs_strings = inputs.split(";", false);
	for (int i = p_id; i < inputs_strings.size(); i++) {
		inputs = inputs.replace_first(inputs_strings[i].split(",")[0], itos(i));
	}

	_apply_port_changes();
}

int VisualShaderNodeGroupBase::get_input_port_count() const {
	return input_ports.size();
}

bool VisualShaderNodeGroupBase::has_input_port(int p_id) const {
	return input_ports.has(p_id);
}

void VisualShaderNodeGroupBase::add_output_port(int p_id, int p_type, const String &p_name) {

	String str = itos(p_id) + "," + itos(p_type) + "," + p_name + ";";
	Vector<String> outputs_strings = outputs.split(";", false);
	int index = 0;
	if (p_id < outputs_strings.size()) {
		for (int i = 0; i < outputs_strings.size(); i++) {
			if (i == p_id) {
				outputs = outputs.insert(index, str);
				break;
			}
			index += outputs_strings[i].size();
		}
	} else {
		outputs += str;
	}

	outputs_strings = outputs.split(";", false);
	index = 0;

	for (int i = 0; i < outputs_strings.size(); i++) {
		int count = 0;
		for (int j = 0; j < outputs_strings[i].size(); j++) {
			if (outputs_strings[i][j] == ',') {
				break;
			}
			count++;
		}

		outputs.erase(index, count);
		outputs = outputs.insert(index, itos(i));
		index += outputs_strings[i].size();
	}

	_apply_port_changes();
}

void VisualShaderNodeGroupBase::remove_output_port(int p_id) {

	ERR_FAIL_COND(!has_output_port(p_id));

	Vector<String> outputs_strings = outputs.split(";", false);
	int count = 0;
	int index = 0;
	for (int i = 0; i < outputs_strings.size(); i++) {
		Vector<String> arr = outputs_strings[i].split(",");
		if (arr[0].to_int() == p_id) {
			count = outputs_strings[i].size();
			break;
		}
		index += outputs_strings[i].size();
	}
	outputs.erase(index, count);

	outputs_strings = outputs.split(";", false);
	for (int i = p_id; i < outputs_strings.size(); i++) {
		outputs = outputs.replace_first(outputs_strings[i].split(",")[0], itos(i));
	}

	_apply_port_changes();
}

int VisualShaderNodeGroupBase::get_output_port_count() const {
	return output_ports.size();
}

bool VisualShaderNodeGroupBase::has_output_port(int p_id) const {
	return output_ports.has(p_id);
}

void VisualShaderNodeGroupBase::clear_input_ports() {
	input_ports.clear();
}

void VisualShaderNodeGroupBase::clear_output_ports() {
	output_ports.clear();
}

void VisualShaderNodeGroupBase::set_input_port_type(int p_id, int p_type) {

	ERR_FAIL_COND(!has_input_port(p_id));
	ERR_FAIL_COND(p_type < 0 || p_type > PORT_TYPE_TRANSFORM);

	if (input_ports[p_id].type == p_type)
		return;

	Vector<String> inputs_strings = inputs.split(";", false);
	int count = 0;
	int index = 0;
	for (int i = 0; i < inputs_strings.size(); i++) {
		Vector<String> arr = inputs_strings[i].split(",");
		if (arr[0].to_int() == p_id) {
			index += arr[0].size();
			count = arr[1].size() - 1;
			break;
		}
		index += inputs_strings[i].size();
	}

	inputs.erase(index, count);

	inputs = inputs.insert(index, itos(p_type));

	_apply_port_changes();
}

VisualShaderNodeGroupBase::PortType VisualShaderNodeGroupBase::get_input_port_type(int p_id) const {
	ERR_FAIL_COND_V(!input_ports.has(p_id), (PortType)0);
	return input_ports[p_id].type;
}

void VisualShaderNodeGroupBase::set_input_port_name(int p_id, const String &p_name) {

	ERR_FAIL_COND(!has_input_port(p_id));
	ERR_FAIL_COND(!is_valid_port_name(p_name));

	if (input_ports[p_id].name == p_name)
		return;

	Vector<String> inputs_strings = inputs.split(";", false);
	int count = 0;
	int index = 0;
	for (int i = 0; i < inputs_strings.size(); i++) {
		Vector<String> arr = inputs_strings[i].split(",");
		if (arr[0].to_int() == p_id) {
			index += arr[0].size() + arr[1].size();
			count = arr[2].size() - 1;
			break;
		}
		index += inputs_strings[i].size();
	}

	inputs.erase(index, count);

	inputs = inputs.insert(index, p_name);

	_apply_port_changes();
}

String VisualShaderNodeGroupBase::get_input_port_name(int p_id) const {
	ERR_FAIL_COND_V(!input_ports.has(p_id), "");
	return input_ports[p_id].name;
}

void VisualShaderNodeGroupBase::set_output_port_type(int p_id, int p_type) {

	ERR_FAIL_COND(!has_output_port(p_id));
	ERR_FAIL_COND(p_type < 0 || p_type > PORT_TYPE_TRANSFORM);

	if (output_ports[p_id].type == p_type)
		return;

	Vector<String> output_strings = outputs.split(";", false);
	int count = 0;
	int index = 0;
	for (int i = 0; i < output_strings.size(); i++) {
		Vector<String> arr = output_strings[i].split(",");
		if (arr[0].to_int() == p_id) {
			index += arr[0].size();
			count = arr[1].size() - 1;
			break;
		}
		index += output_strings[i].size();
	}

	outputs.erase(index, count);

	outputs = outputs.insert(index, itos(p_type));

	_apply_port_changes();
}

VisualShaderNodeGroupBase::PortType VisualShaderNodeGroupBase::get_output_port_type(int p_id) const {
	ERR_FAIL_COND_V(!output_ports.has(p_id), (PortType)0);
	return output_ports[p_id].type;
}

void VisualShaderNodeGroupBase::set_output_port_name(int p_id, const String &p_name) {

	ERR_FAIL_COND(!has_output_port(p_id));
	ERR_FAIL_COND(!is_valid_port_name(p_name));

	if (output_ports[p_id].name == p_name)
		return;

	Vector<String> output_strings = outputs.split(";", false);
	int count = 0;
	int index = 0;
	for (int i = 0; i < output_strings.size(); i++) {
		Vector<String> arr = output_strings[i].split(",");
		if (arr[0].to_int() == p_id) {
			index += arr[0].size() + arr[1].size();
			count = arr[2].size() - 1;
			break;
		}
		index += output_strings[i].size();
	}

	outputs.erase(index, count);

	outputs = outputs.insert(index, p_name);

	_apply_port_changes();
}

String VisualShaderNodeGroupBase::get_output_port_name(int p_id) const {
	ERR_FAIL_COND_V(!output_ports.has(p_id), "");
	return output_ports[p_id].name;
}

int VisualShaderNodeGroupBase::get_free_input_port_id() const {
	return input_ports.size();
}

int VisualShaderNodeGroupBase::get_free_output_port_id() const {
	return output_ports.size();
}

void VisualShaderNodeGroupBase::set_control(Control *p_control, int p_index) {
	controls[p_index] = p_control;
}

Control *VisualShaderNodeGroupBase::get_control(int p_index) {
	ERR_FAIL_COND_V(!controls.has(p_index), NULL);
	return controls[p_index];
}

void VisualShaderNodeGroupBase::_apply_port_changes() {

	Vector<String> inputs_strings = inputs.split(";", false);
	Vector<String> outputs_strings = outputs.split(";", false);

	clear_input_ports();
	clear_output_ports();

	for (int i = 0; i < inputs_strings.size(); i++) {
		Vector<String> arr = inputs_strings[i].split(",");
		Port port;
		port.type = (PortType)arr[1].to_int();
		port.name = arr[2];
		input_ports[i] = port;
	}
	for (int i = 0; i < outputs_strings.size(); i++) {
		Vector<String> arr = outputs_strings[i].split(",");
		Port port;
		port.type = (PortType)arr[1].to_int();
		port.name = arr[2];
		output_ports[i] = port;
	}
}

void VisualShaderNodeGroupBase::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_size", "size"), &VisualShaderNodeGroupBase::set_size);
	ClassDB::bind_method(D_METHOD("get_size"), &VisualShaderNodeGroupBase::get_size);

	ClassDB::bind_method(D_METHOD("set_editable", "enabled"), &VisualShaderNodeGroupBase::set_editable);
	ClassDB::bind_method(D_METHOD("is_editable"), &VisualShaderNodeGroupBase::is_editable);

	ClassDB::bind_method(D_METHOD("set_resizable", "inputs"), &VisualShaderNodeGroupBase::set_resizable);
	ClassDB::bind_method(D_METHOD("is_resizable"), &VisualShaderNodeGroupBase::is_resizable);

	ClassDB::bind_method(D_METHOD("set_inputs", "inputs"), &VisualShaderNodeGroupBase::set_inputs);
	ClassDB::bind_method(D_METHOD("get_inputs"), &VisualShaderNodeGroupBase::get_inputs);

	ClassDB::bind_method(D_METHOD("set_outputs", "outputs"), &VisualShaderNodeGroupBase::set_outputs);
	ClassDB::bind_method(D_METHOD("get_outputs"), &VisualShaderNodeGroupBase::get_outputs);

	ClassDB::bind_method(D_METHOD("is_valid_port_name", "name"), &VisualShaderNodeGroupBase::is_valid_port_name);

	ClassDB::bind_method(D_METHOD("add_input_port", "id", "type", "name"), &VisualShaderNodeGroupBase::add_input_port);
	ClassDB::bind_method(D_METHOD("remove_input_port", "id"), &VisualShaderNodeGroupBase::remove_input_port);
	ClassDB::bind_method(D_METHOD("get_input_port_count"), &VisualShaderNodeGroupBase::get_input_port_count);
	ClassDB::bind_method(D_METHOD("has_input_port", "id"), &VisualShaderNodeGroupBase::has_input_port);
	ClassDB::bind_method(D_METHOD("clear_input_ports"), &VisualShaderNodeGroupBase::clear_input_ports);

	ClassDB::bind_method(D_METHOD("add_output_port", "id", "type", "name"), &VisualShaderNodeGroupBase::add_output_port);
	ClassDB::bind_method(D_METHOD("remove_output_port", "id"), &VisualShaderNodeGroupBase::remove_output_port);
	ClassDB::bind_method(D_METHOD("get_output_port_count"), &VisualShaderNodeGroupBase::get_output_port_count);
	ClassDB::bind_method(D_METHOD("has_output_port", "id"), &VisualShaderNodeGroupBase::has_output_port);
	ClassDB::bind_method(D_METHOD("clear_output_ports"), &VisualShaderNodeGroupBase::clear_output_ports);

	ClassDB::bind_method(D_METHOD("set_input_port_name"), &VisualShaderNodeGroupBase::set_input_port_name);
	ClassDB::bind_method(D_METHOD("set_input_port_type"), &VisualShaderNodeGroupBase::set_input_port_type);
	ClassDB::bind_method(D_METHOD("set_output_port_name"), &VisualShaderNodeGroupBase::set_output_port_name);
	ClassDB::bind_method(D_METHOD("set_output_port_type"), &VisualShaderNodeGroupBase::set_output_port_type);

	ClassDB::bind_method(D_METHOD("get_free_input_port_id"), &VisualShaderNodeGroupBase::get_free_input_port_id);
	ClassDB::bind_method(D_METHOD("get_free_output_port_id"), &VisualShaderNodeGroupBase::get_free_output_port_id);

	ClassDB::bind_method(D_METHOD("set_control", "control", "index"), &VisualShaderNodeGroupBase::set_control);
	ClassDB::bind_method(D_METHOD("get_control", "index"), &VisualShaderNodeGroupBase::get_control);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "editable"), "set_editable", "is_editable");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "resizable"), "set_resizable", "is_resizable");
}

String VisualShaderNodeGroupBase::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "";
}

VisualShaderNodeGroupBase::VisualShaderNodeGroupBase() {
	editable = false;
	resizable = true;
	size = Size2(0, 0);
	inputs = "";
	outputs = "";
}

////////////// Expression

String VisualShaderNodeExpression::get_caption() const {
	return "Expression";
}

void VisualShaderNodeExpression::set_expression(const String &p_expression) {
	expression = p_expression;
}

void VisualShaderNodeExpression::build() {
	emit_changed();
}

String VisualShaderNodeExpression::get_expression() const {
	return expression;
}

String VisualShaderNodeExpression::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {

	String _expression = expression;

	_expression = _expression.insert(0, "\n");
	_expression = _expression.replace("\n", "\n\t\t");

	static Vector<String> pre_symbols;
	if (pre_symbols.empty()) {
		pre_symbols.push_back("\t");
		pre_symbols.push_back("{");
		pre_symbols.push_back("[");
		pre_symbols.push_back("(");
		pre_symbols.push_back(" ");
		pre_symbols.push_back("-");
		pre_symbols.push_back("*");
		pre_symbols.push_back("/");
		pre_symbols.push_back("+");
		pre_symbols.push_back("=");
		pre_symbols.push_back("&");
		pre_symbols.push_back("|");
		pre_symbols.push_back("!");
	}

	static Vector<String> post_symbols;
	if (post_symbols.empty()) {
		post_symbols.push_back("\0");
		post_symbols.push_back("\t");
		post_symbols.push_back("\n");
		post_symbols.push_back(";");
		post_symbols.push_back("}");
		post_symbols.push_back("]");
		post_symbols.push_back(")");
		post_symbols.push_back(" ");
		post_symbols.push_back(".");
		post_symbols.push_back("-");
		post_symbols.push_back("*");
		post_symbols.push_back("/");
		post_symbols.push_back("+");
		post_symbols.push_back("=");
		post_symbols.push_back("&");
		post_symbols.push_back("|");
		post_symbols.push_back("!");
	}

	for (int i = 0; i < get_input_port_count(); i++) {
		for (int j = 0; j < pre_symbols.size(); j++) {
			for (int k = 0; k < post_symbols.size(); k++) {
				_expression = _expression.replace(pre_symbols[j] + get_input_port_name(i) + post_symbols[k], pre_symbols[j] + p_input_vars[i] + post_symbols[k]);
			}
		}
	}
	for (int i = 0; i < get_output_port_count(); i++) {
		for (int j = 0; j < pre_symbols.size(); j++) {
			for (int k = 0; k < post_symbols.size(); k++) {
				_expression = _expression.replace(pre_symbols[j] + get_output_port_name(i) + post_symbols[k], pre_symbols[j] + p_output_vars[i] + post_symbols[k]);
			}
		}
	}

	String output_initializer;

	for (int i = 0; i < get_output_port_count(); i++) {
		int port_type = get_output_port_type(i);
		String tk = "";
		switch (port_type) {
			case PORT_TYPE_SCALAR:
				tk = "0.0";
				break;
			case PORT_TYPE_VECTOR:
				tk = "vec3(0.0, 0.0, 0.0)";
				break;
			case PORT_TYPE_BOOLEAN:
				tk = "false";
				break;
			case PORT_TYPE_TRANSFORM:
				tk = "mat4(1.0)";
				break;
			default:
				continue;
		}
		output_initializer += "\t" + p_output_vars[i] + "=" + tk + ";\n";
	}

	String code;
	code += output_initializer;
	code += "\t{";
	code += _expression;
	code += "\n\t}";

	return code;
}

void VisualShaderNodeExpression::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_expression", "expression"), &VisualShaderNodeExpression::set_expression);
	ClassDB::bind_method(D_METHOD("get_expression"), &VisualShaderNodeExpression::get_expression);

	ClassDB::bind_method(D_METHOD("build"), &VisualShaderNodeExpression::build);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "expression"), "set_expression", "get_expression");
}

VisualShaderNodeExpression::VisualShaderNodeExpression() {

	editable = true;
	expression = "";
}

////////////// Call

String VisualShaderNodeCall::get_caption() const {
	return "Call";
}

void VisualShaderNodeCall::set_function_id(int p_function_id) {
	function_id = p_function_id;
}

int VisualShaderNodeCall::get_function_id() const {
	return function_id;
}

void VisualShaderNodeCall::set_function_name(const String &p_function_name) {
	function_name = p_function_name;
}

String VisualShaderNodeCall::get_function_name() const {
	return function_name;
}

String VisualShaderNodeCall::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {

	String code;

	code += "\t" + p_output_vars[0] + " = " + function_name + "(";

	for (int i = 0; i < get_input_port_count(); i++) {

		String s = p_input_vars[i];
		if (s == String("")) {
			switch (get_input_port_type(i)) {
				case 0:
					s = "0.0";
					break;
				case 1:
					s = "vec3(0.0, 0.0, 0.0)";
					break;
				case 2:
					s = "false";
					break;
				case 3:
					s = "mat4(1.0)";
					break;
				default:
					break;
			}
		}
		code += s;
		if (i != get_input_port_count() - 1)
			code += ", ";
	}
	code += ");\n";

	return code;
}

void VisualShaderNodeCall::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_function_id", "function_name"), &VisualShaderNodeCall::set_function_id);
	ClassDB::bind_method(D_METHOD("get_function_id"), &VisualShaderNodeCall::get_function_id);

	ClassDB::bind_method(D_METHOD("set_function_name", "function_name"), &VisualShaderNodeCall::set_function_name);
	ClassDB::bind_method(D_METHOD("get_function_name"), &VisualShaderNodeCall::get_function_name);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "function_id"), "set_function_id", "get_function_id");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "function_name"), "set_function_name", "get_function_name");
}

VisualShaderNodeCall::VisualShaderNodeCall() {
	function_name = "";
	function_id = -1;
	resizable = false;
}
