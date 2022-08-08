/*************************************************************************/
/*  visual_shader.cpp                                                    */
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

#include "visual_shader.h"

#include "core/vmap.h"
#include "servers/visual/shader_types.h"

bool VisualShaderNode::is_simple_decl() const {
	return simple_decl;
}

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

bool VisualShaderNode::is_output_port_connected(int p_port) const {
	if (connected_output_ports.has(p_port)) {
		return connected_output_ports[p_port] > 0;
	}
	return false;
}

void VisualShaderNode::set_output_port_connected(int p_port, bool p_connected) {
	if (p_connected) {
		connected_output_ports[p_port]++;
	} else {
		connected_output_ports[p_port]--;
	}
}

bool VisualShaderNode::is_input_port_connected(int p_port) const {
	if (connected_input_ports.has(p_port)) {
		return connected_input_ports[p_port];
	}
	return false;
}

void VisualShaderNode::set_input_port_connected(int p_port, bool p_connected) {
	connected_input_ports[p_port] = p_connected;
}

bool VisualShaderNode::is_generate_input_var(int p_port) const {
	return true;
}

bool VisualShaderNode::is_code_generated() const {
	return true;
}

bool VisualShaderNode::is_show_prop_names() const {
	return false;
}

Vector<VisualShader::DefaultTextureParam> VisualShaderNode::get_default_texture_parameters(VisualShader::Type p_type, int p_id) const {
	return Vector<VisualShader::DefaultTextureParam>();
}
String VisualShaderNode::generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	return String();
}

String VisualShaderNode::generate_global_per_node(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	return String();
}

String VisualShaderNode::generate_global_per_func(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	return String();
}

Vector<StringName> VisualShaderNode::get_editable_properties() const {
	return Vector<StringName>();
}

Array VisualShaderNode::get_default_input_values() const {
	Array ret;
	for (Map<int, Variant>::Element *E = default_input_values.front(); E; E = E->next()) {
		ret.push_back(E->key());
		ret.push_back(E->get());
	}
	return ret;
}
void VisualShaderNode::set_default_input_values(const Array &p_values) {
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

String VisualShaderNode::get_input_port_default_hint(int p_port) const {
	return "";
}

void VisualShaderNode::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_output_port_for_preview", "port"), &VisualShaderNode::set_output_port_for_preview);
	ClassDB::bind_method(D_METHOD("get_output_port_for_preview"), &VisualShaderNode::get_output_port_for_preview);

	ClassDB::bind_method(D_METHOD("set_input_port_default_value", "port", "value"), &VisualShaderNode::set_input_port_default_value);
	ClassDB::bind_method(D_METHOD("get_input_port_default_value", "port"), &VisualShaderNode::get_input_port_default_value);

	ClassDB::bind_method(D_METHOD("set_default_input_values", "values"), &VisualShaderNode::set_default_input_values);
	ClassDB::bind_method(D_METHOD("get_default_input_values"), &VisualShaderNode::get_default_input_values);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "output_port_for_preview"), "set_output_port_for_preview", "get_output_port_for_preview");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "default_input_values", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL), "set_default_input_values", "get_default_input_values");
	ADD_SIGNAL(MethodInfo("editor_refresh_request"));

	BIND_ENUM_CONSTANT(PORT_TYPE_SCALAR);
	BIND_ENUM_CONSTANT(PORT_TYPE_VECTOR);
	BIND_ENUM_CONSTANT(PORT_TYPE_BOOLEAN);
	BIND_ENUM_CONSTANT(PORT_TYPE_TRANSFORM);
	BIND_ENUM_CONSTANT(PORT_TYPE_SAMPLER);
	BIND_ENUM_CONSTANT(PORT_TYPE_MAX);
}

VisualShaderNode::VisualShaderNode() {
	port_preview = -1;
	simple_decl = true;
}

/////////////////////////////////////////////////////////

void VisualShaderNodeCustom::update_ports() {
	ERR_FAIL_COND(!get_script_instance());

	input_ports.clear();
	if (get_script_instance()->has_method("_get_input_port_count")) {
		int input_port_count = (int)get_script_instance()->call("_get_input_port_count");
		bool has_name = get_script_instance()->has_method("_get_input_port_name");
		bool has_type = get_script_instance()->has_method("_get_input_port_type");
		for (int i = 0; i < input_port_count; i++) {
			Port port;
			if (has_name) {
				port.name = (String)get_script_instance()->call("_get_input_port_name", i);
			} else {
				port.name = "in" + itos(i);
			}
			if (has_type) {
				port.type = (int)get_script_instance()->call("_get_input_port_type", i);
			} else {
				port.type = (int)PortType::PORT_TYPE_SCALAR;
			}
			input_ports.push_back(port);
		}
	}
	output_ports.clear();
	if (get_script_instance()->has_method("_get_output_port_count")) {
		int output_port_count = (int)get_script_instance()->call("_get_output_port_count");
		bool has_name = get_script_instance()->has_method("_get_output_port_name");
		bool has_type = get_script_instance()->has_method("_get_output_port_type");
		for (int i = 0; i < output_port_count; i++) {
			Port port;
			if (has_name) {
				port.name = (String)get_script_instance()->call("_get_output_port_name", i);
			} else {
				port.name = "out" + itos(i);
			}
			if (has_type) {
				port.type = (int)get_script_instance()->call("_get_output_port_type", i);
			} else {
				port.type = (int)PortType::PORT_TYPE_SCALAR;
			}
			output_ports.push_back(port);
		}
	}
}

String VisualShaderNodeCustom::get_caption() const {
	ERR_FAIL_COND_V(!get_script_instance(), "");
	if (get_script_instance()->has_method("_get_name")) {
		return (String)get_script_instance()->call("_get_name");
	}
	return "Unnamed";
}

int VisualShaderNodeCustom::get_input_port_count() const {
	return input_ports.size();
}

VisualShaderNodeCustom::PortType VisualShaderNodeCustom::get_input_port_type(int p_port) const {
	ERR_FAIL_INDEX_V(p_port, input_ports.size(), PORT_TYPE_SCALAR);
	return (PortType)input_ports[p_port].type;
}

String VisualShaderNodeCustom::get_input_port_name(int p_port) const {
	ERR_FAIL_INDEX_V(p_port, input_ports.size(), "");
	return input_ports[p_port].name;
}

int VisualShaderNodeCustom::get_output_port_count() const {
	return output_ports.size();
}

VisualShaderNodeCustom::PortType VisualShaderNodeCustom::get_output_port_type(int p_port) const {
	ERR_FAIL_INDEX_V(p_port, output_ports.size(), PORT_TYPE_SCALAR);
	return (PortType)output_ports[p_port].type;
}

String VisualShaderNodeCustom::get_output_port_name(int p_port) const {
	ERR_FAIL_INDEX_V(p_port, output_ports.size(), "");
	return output_ports[p_port].name;
}

String VisualShaderNodeCustom::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	ERR_FAIL_COND_V(!get_script_instance(), "");
	ERR_FAIL_COND_V(!get_script_instance()->has_method("_get_code"), "");
	Array input_vars;
	for (int i = 0; i < get_input_port_count(); i++) {
		input_vars.push_back(p_input_vars[i]);
	}
	Array output_vars;
	for (int i = 0; i < get_output_port_count(); i++) {
		output_vars.push_back(p_output_vars[i]);
	}
	String code = "\t{\n";
	String _code = (String)get_script_instance()->call("_get_code", input_vars, output_vars, (int)p_mode, (int)p_type);
	bool nend = _code.ends_with("\n");
	_code = _code.insert(0, "\t\t");
	_code = _code.replace("\n", "\n\t\t");
	code += _code;
	if (!nend) {
		code += "\n\t}";
	} else {
		code.remove(code.size() - 1);
		code += "}";
	}
	code += "\n";
	return code;
}

String VisualShaderNodeCustom::generate_global_per_node(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	ERR_FAIL_COND_V(!get_script_instance(), "");
	if (get_script_instance()->has_method("_get_global_code")) {
		String code = "// " + get_caption() + "\n";
		code += (String)get_script_instance()->call("_get_global_code", (int)p_mode);
		code += "\n";
		return code;
	}
	return "";
}

void VisualShaderNodeCustom::set_input_port_default_value(int p_port, const Variant &p_value) {
	if (!is_initialized) {
		VisualShaderNode::set_input_port_default_value(p_port, p_value);
	}
}

void VisualShaderNodeCustom::set_default_input_values(const Array &p_values) {
	if (!is_initialized) {
		VisualShaderNode::set_default_input_values(p_values);
	}
}

void VisualShaderNodeCustom::_set_input_port_default_value(int p_port, const Variant &p_value) {
	VisualShaderNode::set_input_port_default_value(p_port, p_value);
}

bool VisualShaderNodeCustom::_is_initialized() {
	return is_initialized;
}

void VisualShaderNodeCustom::_set_initialized(bool p_enabled) {
	is_initialized = p_enabled;
}

void VisualShaderNodeCustom::_bind_methods() {
	BIND_VMETHOD(MethodInfo(Variant::STRING, "_get_name"));
	BIND_VMETHOD(MethodInfo(Variant::STRING, "_get_description"));
	BIND_VMETHOD(MethodInfo(Variant::STRING, "_get_category"));
	BIND_VMETHOD(MethodInfo(Variant::STRING, "_get_subcategory"));
	BIND_VMETHOD(MethodInfo(Variant::INT, "_get_return_icon_type"));
	BIND_VMETHOD(MethodInfo(Variant::INT, "_get_input_port_count"));
	BIND_VMETHOD(MethodInfo(Variant::INT, "_get_input_port_type", PropertyInfo(Variant::INT, "port")));
	BIND_VMETHOD(MethodInfo(Variant::STRING, "_get_input_port_name", PropertyInfo(Variant::INT, "port")));
	BIND_VMETHOD(MethodInfo(Variant::INT, "_get_output_port_count"));
	BIND_VMETHOD(MethodInfo(Variant::INT, "_get_output_port_type", PropertyInfo(Variant::INT, "port")));
	BIND_VMETHOD(MethodInfo(Variant::STRING, "_get_output_port_name", PropertyInfo(Variant::INT, "port")));
	BIND_VMETHOD(MethodInfo(Variant::STRING, "_get_code", PropertyInfo(Variant::ARRAY, "input_vars"), PropertyInfo(Variant::ARRAY, "output_vars"), PropertyInfo(Variant::INT, "mode"), PropertyInfo(Variant::INT, "type")));
	BIND_VMETHOD(MethodInfo(Variant::STRING, "_get_global_code", PropertyInfo(Variant::INT, "mode")));

	ClassDB::bind_method(D_METHOD("_set_initialized", "enabled"), &VisualShaderNodeCustom::_set_initialized);
	ClassDB::bind_method(D_METHOD("_is_initialized"), &VisualShaderNodeCustom::_is_initialized);
	ClassDB::bind_method(D_METHOD("_set_input_port_default_value", "port", "value"), &VisualShaderNodeCustom::_set_input_port_default_value);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "initialized", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL), "_set_initialized", "_is_initialized");
}

VisualShaderNodeCustom::VisualShaderNodeCustom() {
	simple_decl = false;
}

/////////////////////////////////////////////////////////

void VisualShader::add_node(Type p_type, const Ref<VisualShaderNode> &p_node, const Vector2 &p_position, int p_id) {
	ERR_FAIL_COND(p_node.is_null());
	ERR_FAIL_COND(p_id < 2);
	ERR_FAIL_INDEX(p_type, TYPE_MAX);
	Graph *g = &graph[p_type];
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
		input->shader_mode = shader_mode;
		input->shader_type = p_type;
		input->connect("input_type_changed", this, "_input_type_changed", varray(p_type, p_id));
	}

	n.node->connect("changed", this, "_queue_update");

	Ref<VisualShaderNodeCustom> custom = n.node;
	if (custom.is_valid()) {
		custom->update_ports();
	}

	g->nodes[p_id] = n;

	_queue_update();
}

void VisualShader::set_node_position(Type p_type, int p_id, const Vector2 &p_position) {
	ERR_FAIL_INDEX(p_type, TYPE_MAX);
	Graph *g = &graph[p_type];
	ERR_FAIL_COND(!g->nodes.has(p_id));
	g->nodes[p_id].position = p_position;
}

Vector2 VisualShader::get_node_position(Type p_type, int p_id) const {
	ERR_FAIL_INDEX_V(p_type, TYPE_MAX, Vector2());
	const Graph *g = &graph[p_type];
	ERR_FAIL_COND_V(!g->nodes.has(p_id), Vector2());
	return g->nodes[p_id].position;
}

Ref<VisualShaderNode> VisualShader::get_node(Type p_type, int p_id) const {
	ERR_FAIL_INDEX_V(p_type, TYPE_MAX, Ref<VisualShaderNode>());
	const Graph *g = &graph[p_type];
	ERR_FAIL_COND_V(!g->nodes.has(p_id), Ref<VisualShaderNode>());
	return g->nodes[p_id].node;
}

Vector<int> VisualShader::get_node_list(Type p_type) const {
	ERR_FAIL_INDEX_V(p_type, TYPE_MAX, Vector<int>());
	const Graph *g = &graph[p_type];

	Vector<int> ret;
	for (Map<int, Node>::Element *E = g->nodes.front(); E; E = E->next()) {
		ret.push_back(E->key());
	}

	return ret;
}
int VisualShader::get_valid_node_id(Type p_type) const {
	ERR_FAIL_INDEX_V(p_type, TYPE_MAX, NODE_ID_INVALID);
	const Graph *g = &graph[p_type];
	return g->nodes.size() ? MAX(2, g->nodes.back()->key() + 1) : 2;
}

int VisualShader::find_node_id(Type p_type, const Ref<VisualShaderNode> &p_node) const {
	for (const Map<int, Node>::Element *E = graph[p_type].nodes.front(); E; E = E->next()) {
		if (E->get().node == p_node) {
			return E->key();
		}
	}

	return NODE_ID_INVALID;
}

void VisualShader::remove_node(Type p_type, int p_id) {
	ERR_FAIL_INDEX(p_type, TYPE_MAX);
	ERR_FAIL_COND(p_id < 2);
	Graph *g = &graph[p_type];
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
			if (E->get().from_node == p_id) {
				g->nodes[E->get().to_node].prev_connected_nodes.erase(p_id);
				g->nodes[E->get().to_node].node->set_input_port_connected(E->get().to_port, false);
			}
		}
		E = N;
	}

	_queue_update();
}

bool VisualShader::is_node_connection(Type p_type, int p_from_node, int p_from_port, int p_to_node, int p_to_port) const {
	ERR_FAIL_INDEX_V(p_type, TYPE_MAX, false);
	const Graph *g = &graph[p_type];

	for (const List<Connection>::Element *E = g->connections.front(); E; E = E->next()) {
		if (E->get().from_node == p_from_node && E->get().from_port == p_from_port && E->get().to_node == p_to_node && E->get().to_port == p_to_port) {
			return true;
		}
	}

	return false;
}

bool VisualShader::is_nodes_connected_relatively(const Graph *p_graph, int p_node, int p_target) const {
	bool result = false;

	const VisualShader::Node &node = p_graph->nodes[p_node];

	for (const List<int>::Element *E = node.prev_connected_nodes.front(); E; E = E->next()) {
		if (E->get() == p_target) {
			return true;
		}

		result = is_nodes_connected_relatively(p_graph, E->get(), p_target);
		if (result) {
			break;
		}
	}
	return result;
}

bool VisualShader::can_connect_nodes(Type p_type, int p_from_node, int p_from_port, int p_to_node, int p_to_port) const {
	ERR_FAIL_INDEX_V(p_type, TYPE_MAX, false);
	const Graph *g = &graph[p_type];

	if (!g->nodes.has(p_from_node)) {
		return false;
	}

	if (p_from_node == p_to_node) {
		return false;
	}

	if (p_from_port < 0 || p_from_port >= g->nodes[p_from_node].node->get_output_port_count()) {
		return false;
	}

	if (!g->nodes.has(p_to_node)) {
		return false;
	}

	if (p_to_port < 0 || p_to_port >= g->nodes[p_to_node].node->get_input_port_count()) {
		return false;
	}

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

	if (is_nodes_connected_relatively(g, p_from_node, p_to_node)) {
		return false;
	}

	return true;
}

bool VisualShader::is_port_types_compatible(int p_a, int p_b) const {
	return MAX(0, p_a - 2) == (MAX(0, p_b - 2));
}

void VisualShader::connect_nodes_forced(Type p_type, int p_from_node, int p_from_port, int p_to_node, int p_to_port) {
	ERR_FAIL_INDEX(p_type, TYPE_MAX);
	Graph *g = &graph[p_type];

	ERR_FAIL_COND(!g->nodes.has(p_from_node));
	ERR_FAIL_INDEX(p_from_port, g->nodes[p_from_node].node->get_output_port_count());
	ERR_FAIL_COND(!g->nodes.has(p_to_node));
	ERR_FAIL_INDEX(p_to_port, g->nodes[p_to_node].node->get_input_port_count());

	Connection c;
	c.from_node = p_from_node;
	c.from_port = p_from_port;
	c.to_node = p_to_node;
	c.to_port = p_to_port;
	g->connections.push_back(c);
	g->nodes[p_to_node].prev_connected_nodes.push_back(p_from_node);
	g->nodes[p_from_node].node->set_output_port_connected(p_from_port, true);
	g->nodes[p_to_node].node->set_input_port_connected(p_to_port, true);

	_queue_update();
}

Error VisualShader::connect_nodes(Type p_type, int p_from_node, int p_from_port, int p_to_node, int p_to_port) {
	ERR_FAIL_INDEX_V(p_type, TYPE_MAX, ERR_CANT_CONNECT);
	Graph *g = &graph[p_type];

	ERR_FAIL_COND_V(!g->nodes.has(p_from_node), ERR_INVALID_PARAMETER);
	ERR_FAIL_INDEX_V(p_from_port, g->nodes[p_from_node].node->get_output_port_count(), ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(!g->nodes.has(p_to_node), ERR_INVALID_PARAMETER);
	ERR_FAIL_INDEX_V(p_to_port, g->nodes[p_to_node].node->get_input_port_count(), ERR_INVALID_PARAMETER);

	VisualShaderNode::PortType from_port_type = g->nodes[p_from_node].node->get_output_port_type(p_from_port);
	VisualShaderNode::PortType to_port_type = g->nodes[p_to_node].node->get_input_port_type(p_to_port);

	ERR_FAIL_COND_V_MSG(!is_port_types_compatible(from_port_type, to_port_type), ERR_INVALID_PARAMETER, "Incompatible port types (scalar/vec/bool) with transform.");

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
	g->nodes[p_to_node].prev_connected_nodes.push_back(p_from_node);
	g->nodes[p_from_node].node->set_output_port_connected(p_from_port, true);
	g->nodes[p_to_node].node->set_input_port_connected(p_to_port, true);

	_queue_update();
	return OK;
}

void VisualShader::disconnect_nodes(Type p_type, int p_from_node, int p_from_port, int p_to_node, int p_to_port) {
	ERR_FAIL_INDEX(p_type, TYPE_MAX);
	Graph *g = &graph[p_type];

	for (List<Connection>::Element *E = g->connections.front(); E; E = E->next()) {
		if (E->get().from_node == p_from_node && E->get().from_port == p_from_port && E->get().to_node == p_to_node && E->get().to_port == p_to_port) {
			g->connections.erase(E);
			g->nodes[p_to_node].prev_connected_nodes.erase(p_from_node);
			g->nodes[p_from_node].node->set_output_port_connected(p_from_port, false);
			g->nodes[p_to_node].node->set_input_port_connected(p_to_port, false);
			_queue_update();
			return;
		}
	}
}

Array VisualShader::_get_node_connections(Type p_type) const {
	ERR_FAIL_INDEX_V(p_type, TYPE_MAX, Array());
	const Graph *g = &graph[p_type];

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

void VisualShader::get_node_connections(Type p_type, List<Connection> *r_connections) const {
	ERR_FAIL_INDEX(p_type, TYPE_MAX);
	const Graph *g = &graph[p_type];

	for (const List<Connection>::Element *E = g->connections.front(); E; E = E->next()) {
		r_connections->push_back(E->get());
	}
}

void VisualShader::set_mode(Mode p_mode) {
	ERR_FAIL_INDEX_MSG(p_mode, Mode::MODE_MAX, vformat("Invalid shader mode: %d.", p_mode));

	if (shader_mode == p_mode) {
		return;
	}

	//erase input/output connections
	modes.clear();
	flags.clear();
	shader_mode = p_mode;
	for (int i = 0; i < TYPE_MAX; i++) {
		for (Map<int, Node>::Element *E = graph[i].nodes.front(); E; E = E->next()) {
			Ref<VisualShaderNodeInput> input = E->get().node;
			if (input.is_valid()) {
				input->shader_mode = shader_mode;
				//input->input_index = 0;
			}
		}

		Ref<VisualShaderNodeOutput> output = graph[i].nodes[NODE_ID_OUTPUT].node;
		output->shader_mode = shader_mode;

		// clear connections since they are no longer valid
		for (List<Connection>::Element *E = graph[i].connections.front(); E;) {
			bool keep = true;

			List<Connection>::Element *N = E->next();

			int from = E->get().from_node;
			int to = E->get().to_node;

			if (!graph[i].nodes.has(from)) {
				keep = false;
			} else {
				Ref<VisualShaderNode> from_node = graph[i].nodes[from].node;
				if (from_node->is_class("VisualShaderNodeOutput") || from_node->is_class("VisualShaderNodeInput")) {
					keep = false;
				}
			}

			if (!graph[i].nodes.has(to)) {
				keep = false;
			} else {
				Ref<VisualShaderNode> to_node = graph[i].nodes[to].node;
				if (to_node->is_class("VisualShaderNodeOutput") || to_node->is_class("VisualShaderNodeInput")) {
					keep = false;
				}
			}

			if (!keep) {
				graph[i].connections.erase(E);
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
	StringBuilder global_code_per_node;
	Map<Type, StringBuilder> global_code_per_func;
	StringBuilder code;
	Set<StringName> classes;

	global_code += String() + "shader_type canvas_item;\n";

	String global_expressions;
	for (int i = 0, index = 0; i < TYPE_MAX; i++) {
		for (Map<int, Node>::Element *E = graph[i].nodes.front(); E; E = E->next()) {
			Ref<VisualShaderNodeGlobalExpression> global_expression = Object::cast_to<VisualShaderNodeGlobalExpression>(E->get().node.ptr());
			if (global_expression.is_valid()) {
				String expr = "";
				expr += "// " + global_expression->get_caption() + ":" + itos(index++) + "\n";
				expr += global_expression->generate_global(get_mode(), Type(i), -1);
				expr = expr.replace("\n", "\n\t");
				expr += "\n";
				global_expressions += expr;
			}
		}
	}

	global_code += "\n";
	global_code += global_expressions;

	//make it faster to go around through shader
	VMap<ConnectionKey, const List<Connection>::Element *> input_connections;
	VMap<ConnectionKey, const List<Connection>::Element *> output_connections;

	for (const List<Connection>::Element *E = graph[p_type].connections.front(); E; E = E->next()) {
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
	Error err = _write_node(p_type, global_code, global_code_per_node, global_code_per_func, code, default_tex_params, input_connections, output_connections, p_node, processed, true, classes);
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
	final_code += global_code_per_node;
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
		for (int i = 0; i < TYPE_MAX; i++) {
			for (const Map<int, Node>::Element *E = graph[i].nodes.front(); E; E = E->next()) {
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
	{ Shader::MODE_SPATIAL, PNAME("blend") },
	{ Shader::MODE_SPATIAL, PNAME("depth_draw") },
	{ Shader::MODE_SPATIAL, PNAME("cull") },
	{ Shader::MODE_SPATIAL, PNAME("diffuse") },
	{ Shader::MODE_SPATIAL, PNAME("specular") },
	{ Shader::MODE_SPATIAL, PNAME("async") },
	{ Shader::MODE_CANVAS_ITEM, PNAME("blend") },
	{ Shader::MODE_CANVAS_ITEM, nullptr }
};

static const char *type_string[VisualShader::TYPE_MAX] = {
	"vertex",
	"fragment",
	"light"
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
	} else if (name.begins_with("nodes/")) {
		String typestr = name.get_slicec('/', 1);
		Type type = TYPE_VERTEX;
		for (int i = 0; i < TYPE_MAX; i++) {
			if (typestr == type_string[i]) {
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
	} else if (name.begins_with("nodes/")) {
		String typestr = name.get_slicec('/', 1);
		Type type = TYPE_VERTEX;
		for (int i = 0; i < TYPE_MAX; i++) {
			if (typestr == type_string[i]) {
				type = Type(i);
				break;
			}
		}

		String index = name.get_slicec('/', 2);
		if (index == "connections") {
			Vector<int> conns;
			for (const List<Connection>::Element *E = graph[type].connections.front(); E; E = E->next()) {
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
	p_list->push_back(PropertyInfo(Variant::INT, PNAME("mode"), PROPERTY_HINT_ENUM, "Spatial,CanvasItem,Particles"));
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
		p_list->push_back(PropertyInfo(Variant::INT, vformat("%s/%s", PNAME("modes"), E->key()), PROPERTY_HINT_ENUM, E->get()));
	}

	for (Set<String>::Element *E = toggles.front(); E; E = E->next()) {
		p_list->push_back(PropertyInfo(Variant::BOOL, vformat("%s/%s", PNAME("flags"), E->get())));
	}

	for (int i = 0; i < TYPE_MAX; i++) {
		for (Map<int, Node>::Element *E = graph[i].nodes.front(); E; E = E->next()) {
			String prop_name = "nodes/";
			prop_name += type_string[i];
			prop_name += "/" + itos(E->key());

			if (E->key() != NODE_ID_OUTPUT) {
				p_list->push_back(PropertyInfo(Variant::OBJECT, prop_name + "/node", PROPERTY_HINT_RESOURCE_TYPE, "VisualShaderNode", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_DO_NOT_SHARE_ON_DUPLICATE));
			}
			p_list->push_back(PropertyInfo(Variant::VECTOR2, prop_name + "/position", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));

			if (Object::cast_to<VisualShaderNodeGroupBase>(E->get().node.ptr()) != nullptr) {
				p_list->push_back(PropertyInfo(Variant::VECTOR2, prop_name + "/size", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
				p_list->push_back(PropertyInfo(Variant::STRING, prop_name + "/input_ports", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
				p_list->push_back(PropertyInfo(Variant::STRING, prop_name + "/output_ports", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
			}
			if (Object::cast_to<VisualShaderNodeExpression>(E->get().node.ptr()) != nullptr) {
				p_list->push_back(PropertyInfo(Variant::STRING, prop_name + "/expression", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
			}
		}
		p_list->push_back(PropertyInfo(Variant::POOL_INT_ARRAY, "nodes/" + String(type_string[i]) + "/connections", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
	}
}

Error VisualShader::_write_node(Type type, StringBuilder &global_code, StringBuilder &global_code_per_node, Map<Type, StringBuilder> &global_code_per_func, StringBuilder &code, Vector<VisualShader::DefaultTextureParam> &def_tex_params, const VMap<ConnectionKey, const List<Connection>::Element *> &input_connections, const VMap<ConnectionKey, const List<Connection>::Element *> &output_connections, int node, Set<int> &processed, bool for_preview, Set<StringName> &r_classes) const {
	const Ref<VisualShaderNode> vsnode = graph[type].nodes[node].node;

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

			Error err = _write_node(type, global_code, global_code_per_node, global_code_per_func, code, def_tex_params, input_connections, output_connections, from_node, processed, for_preview, r_classes);
			if (err) {
				return err;
			}
		}
	}

	// then this node

	Vector<VisualShader::DefaultTextureParam> params = vsnode->get_default_texture_parameters(type, node);
	for (int i = 0; i < params.size(); i++) {
		def_tex_params.push_back(params[i]);
	}

	Ref<VisualShaderNodeInput> input = vsnode;
	bool skip_global = input.is_valid() && for_preview;

	if (!skip_global) {
		Ref<VisualShaderNodeUniform> uniform = vsnode;
		if (!uniform.is_valid() || !uniform->is_global_code_generated()) {
			global_code += vsnode->generate_global(get_mode(), type, node);
		}
		String class_name = vsnode->get_class_name();
		if (class_name == "VisualShaderNodeCustom") {
			class_name = vsnode->get_script_instance()->get_script()->get_path();
		}
		if (!r_classes.has(class_name)) {
			global_code_per_node += vsnode->generate_global_per_node(get_mode(), type, node);
			for (int i = 0; i < TYPE_MAX; i++) {
				global_code_per_func[Type(i)] += vsnode->generate_global_per_func(get_mode(), Type(i), node);
			}
			r_classes.insert(class_name);
		}
	}

	if (!vsnode->is_code_generated()) { // just generate globals and ignore locals
		processed.insert(node);
		return OK;
	}

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
			VisualShaderNode::PortType out_type = graph[type].nodes[from_node].node->get_output_port_type(from_port);

			String src_var = "n_out" + itos(from_node) + "p" + itos(from_port);

			if (in_type == VisualShaderNode::PORT_TYPE_SAMPLER && out_type == VisualShaderNode::PORT_TYPE_SAMPLER) {
				VisualShaderNode *ptr = const_cast<VisualShaderNode *>(graph[type].nodes[from_node].node.ptr());
				if (ptr->has_method("get_input_real_name")) {
					inputs[i] = ptr->call("get_input_real_name");
				} else if (ptr->has_method("get_uniform_name")) {
					inputs[i] = ptr->call("get_uniform_name");
				} else {
					inputs[i] = "";
				}
			} else if (in_type == out_type) {
				inputs[i] = src_var;
			} else if (in_type == VisualShaderNode::PORT_TYPE_SCALAR && out_type == VisualShaderNode::PORT_TYPE_VECTOR) {
				inputs[i] = "dot(" + src_var + ", vec3(0.333333, 0.333333, 0.333333))";
			} else if (in_type == VisualShaderNode::PORT_TYPE_VECTOR && out_type == VisualShaderNode::PORT_TYPE_SCALAR) {
				inputs[i] = "vec3(" + src_var + ")";
			} else if (in_type == VisualShaderNode::PORT_TYPE_BOOLEAN && out_type == VisualShaderNode::PORT_TYPE_VECTOR) {
				inputs[i] = "all(bvec3(" + src_var + "))";
			} else if (in_type == VisualShaderNode::PORT_TYPE_BOOLEAN && out_type == VisualShaderNode::PORT_TYPE_SCALAR) {
				inputs[i] = src_var + " > 0.0 ? true : false";
			} else if (in_type == VisualShaderNode::PORT_TYPE_SCALAR && out_type == VisualShaderNode::PORT_TYPE_BOOLEAN) {
				inputs[i] = "(" + src_var + " ? 1.0 : 0.0)";
			} else if (in_type == VisualShaderNode::PORT_TYPE_VECTOR && out_type == VisualShaderNode::PORT_TYPE_BOOLEAN) {
				inputs[i] = "vec3(" + src_var + " ? 1.0 : 0.0)";
			}
		} else {
			if (!vsnode->is_generate_input_var(i)) {
				continue;
			}

			Variant defval = vsnode->get_input_port_default_value(i);
			if (defval.get_type() == Variant::REAL || defval.get_type() == Variant::INT) {
				float val = defval;
				inputs[i] = "n_in" + itos(node) + "p" + itos(i);
				code += "\tfloat " + inputs[i] + " = " + vformat("%.5f", val) + ";\n";
			} else if (defval.get_type() == Variant::BOOL) {
				bool val = defval;
				inputs[i] = "n_in" + itos(node) + "p" + itos(i);
				code += "\tbool " + inputs[i] + " = " + (val ? "true" : "false") + ";\n";
			} else if (defval.get_type() == Variant::VECTOR3) {
				Vector3 val = defval;
				inputs[i] = "n_in" + itos(node) + "p" + itos(i);
				code += "\tvec3 " + inputs[i] + " = " + vformat("vec3(%.5f, %.5f, %.5f);\n", val.x, val.y, val.z);
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
				code += "\tmat4 " + inputs[i] + " = " + String("mat4(vec4(%.5f, %.5f, %.5f, 0.0), vec4(%.5f, %.5f, %.5f, 0.0), vec4(%.5f, %.5f, %.5f, 0.0), vec4(%.5f, %.5f, %.5f, 1.0));\n").sprintf(values, &err);
			} else {
				//will go empty, node is expected to know what it is doing at this point and handle it
			}
		}
	}

	int output_count = vsnode->get_output_port_count();
	Vector<String> output_vars;
	output_vars.resize(vsnode->get_output_port_count());
	String *outputs = output_vars.ptrw();

	if (vsnode->is_simple_decl()) { // less code to generate for some simple_decl nodes
		for (int i = 0; i < output_count; i++) {
			String var_name = "n_out" + itos(node) + "p" + itos(i);
			switch (vsnode->get_output_port_type(i)) {
				case VisualShaderNode::PORT_TYPE_SCALAR:
					outputs[i] = "float " + var_name;
					break;
				case VisualShaderNode::PORT_TYPE_VECTOR:
					outputs[i] = "vec3 " + var_name;
					break;
				case VisualShaderNode::PORT_TYPE_BOOLEAN:
					outputs[i] = "bool " + var_name;
					break;
				case VisualShaderNode::PORT_TYPE_TRANSFORM:
					outputs[i] = "mat4 " + var_name;
					break;
				default: {
				}
			}
		}

	} else {
		for (int i = 0; i < output_count; i++) {
			outputs[i] = "n_out" + itos(node) + "p" + itos(i);
			switch (vsnode->get_output_port_type(i)) {
				case VisualShaderNode::PORT_TYPE_SCALAR:
					code += String() + "\tfloat " + outputs[i] + ";\n";
					break;
				case VisualShaderNode::PORT_TYPE_VECTOR:
					code += String() + "\tvec3 " + outputs[i] + ";\n";
					break;
				case VisualShaderNode::PORT_TYPE_BOOLEAN:
					code += String() + "\tbool " + outputs[i] + ";\n";
					break;
				case VisualShaderNode::PORT_TYPE_TRANSFORM:
					code += String() + "\tmat4 " + outputs[i] + ";\n";
					break;
				default: {
				}
			}
		}
	}

	code += vsnode->generate_code(get_mode(), type, node, inputs, outputs, for_preview);

	code += "\n"; //
	processed.insert(node);

	return OK;
}

void VisualShader::_update_shader() const {
	if (!dirty.is_set()) {
		return;
	}

	dirty.clear();

	StringBuilder global_code;
	StringBuilder global_code_per_node;
	Map<Type, StringBuilder> global_code_per_func;
	StringBuilder code;
	Vector<VisualShader::DefaultTextureParam> default_tex_params;
	Set<StringName> classes;
	List<int> insertion_pos;
	static const char *shader_mode_str[Shader::MODE_MAX] = { "spatial", "canvas_item", "particles" };

	global_code += String() + "shader_type " + shader_mode_str[shader_mode] + ";\n";

	String render_mode;

	{
		//fill render mode enums
		int idx = 0;
		bool specular = false;
		while (render_mode_enums[idx].string) {
			if (shader_mode == render_mode_enums[idx].mode) {
				if (shader_mode == Shader::MODE_SPATIAL) {
					if (String(render_mode_enums[idx].string) == "specular") {
						specular = true;
					}
				}
				if (modes.has(render_mode_enums[idx].string) || specular) {
					int which = 0;
					if (modes.has(render_mode_enums[idx].string)) {
						which = modes[render_mode_enums[idx].string];
					}
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

	static const char *func_name[TYPE_MAX] = { "vertex", "fragment", "light" };

	String global_expressions;
	Set<String> used_uniform_names;
	List<VisualShaderNodeUniform *> uniforms;

	for (int i = 0, index = 0; i < TYPE_MAX; i++) {
		for (Map<int, Node>::Element *E = graph[i].nodes.front(); E; E = E->next()) {
			Ref<VisualShaderNodeGlobalExpression> global_expression = Object::cast_to<VisualShaderNodeGlobalExpression>(E->get().node.ptr());
			if (global_expression.is_valid()) {
				String expr = "";
				expr += "// " + global_expression->get_caption() + ":" + itos(index++) + "\n";
				expr += global_expression->generate_global(get_mode(), Type(i), -1);
				expr = expr.replace("\n", "\n\t");
				expr += "\n";
				global_expressions += expr;
			}
			Ref<VisualShaderNodeUniformRef> uniform_ref = Object::cast_to<VisualShaderNodeUniformRef>(E->get().node.ptr());
			if (uniform_ref.is_valid()) {
				used_uniform_names.insert(uniform_ref->get_uniform_name());
			}
			Ref<VisualShaderNodeUniform> uniform = Object::cast_to<VisualShaderNodeUniform>(E->get().node.ptr());
			if (uniform.is_valid()) {
				uniforms.push_back(uniform.ptr());
			}
		}
	}

	for (int i = 0; i < uniforms.size(); i++) {
		VisualShaderNodeUniform *uniform = uniforms[i];
		if (used_uniform_names.has(uniform->get_uniform_name())) {
			global_code += uniform->generate_global(get_mode(), Type(i), -1);
			const_cast<VisualShaderNodeUniform *>(uniform)->set_global_code_generated(true);
		} else {
			const_cast<VisualShaderNodeUniform *>(uniform)->set_global_code_generated(false);
		}
	}

	for (int i = 0; i < TYPE_MAX; i++) {
		//make it faster to go around through shader
		VMap<ConnectionKey, const List<Connection>::Element *> input_connections;
		VMap<ConnectionKey, const List<Connection>::Element *> output_connections;

		for (const List<Connection>::Element *E = graph[i].connections.front(); E; E = E->next()) {
			ConnectionKey from_key;
			from_key.node = E->get().from_node;
			from_key.port = E->get().from_port;

			output_connections.insert(from_key, E);

			ConnectionKey to_key;
			to_key.node = E->get().to_node;
			to_key.port = E->get().to_port;

			input_connections.insert(to_key, E);
		}

		code += "\nvoid " + String(func_name[i]) + "() {\n";

		Set<int> processed;
		Error err = _write_node(Type(i), global_code, global_code_per_node, global_code_per_func, code, default_tex_params, input_connections, output_connections, NODE_ID_OUTPUT, processed, false, classes);
		ERR_FAIL_COND(err != OK);
		insertion_pos.push_back(code.get_string_length());

		code += "}\n";
	}

	//set code secretly
	global_code += "\n\n";
	String final_code = global_code;
	final_code += global_code_per_node;
	final_code += global_expressions;
	String tcode = code;
	for (int i = 0; i < TYPE_MAX; i++) {
		tcode = tcode.insert(insertion_pos[i], global_code_per_func[Type(i)]);
	}
	final_code += tcode;

	const_cast<VisualShader *>(this)->set_code(final_code);
	for (int i = 0; i < default_tex_params.size(); i++) {
		const_cast<VisualShader *>(this)->set_default_texture_param(default_tex_params[i].name, default_tex_params[i].param);
	}
	if (previous_code != final_code) {
		const_cast<VisualShader *>(this)->emit_signal("changed");
	}
	previous_code = final_code;
}

void VisualShader::_queue_update() {
	if (dirty.is_set()) {
		return;
	}

	dirty.set();
	call_deferred("_update_shader");
}

void VisualShader::_input_type_changed(Type p_type, int p_id) {
	ERR_FAIL_INDEX(p_type, TYPE_MAX);
	//erase connections using this input, as type changed
	Graph *g = &graph[p_type];

	for (List<Connection>::Element *E = g->connections.front(); E;) {
		List<Connection>::Element *N = E->next();
		if (E->get().from_node == p_id) {
			g->connections.erase(E);
			g->nodes[E->get().to_node].prev_connected_nodes.erase(p_id);
		}
		E = N;
	}
}

void VisualShader::rebuild() {
	dirty.set();
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

	ADD_PROPERTY_DEFAULT("code", ""); // Inherited from Shader, prevents showing default code as override in docs.

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
		Ref<VisualShaderNodeOutput> output;
		output.instance();
		output->shader_type = Type(i);
		output->shader_mode = shader_mode;
		graph[i].nodes[NODE_ID_OUTPUT].node = output;
		graph[i].nodes[NODE_ID_OUTPUT].position = Vector2(400, 150);
	}

	dirty.set();
}

///////////////////////////////////////////////////////////

const VisualShaderNodeInput::Port VisualShaderNodeInput::ports[] = {
	// Spatial, Vertex
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "vertex", "VERTEX" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "normal", "NORMAL" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "tangent", "TANGENT" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "binormal", "BINORMAL" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "uv", "vec3(UV, 0.0)" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "uv2", "vec3(UV2, 0.0)" },
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
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_BOOLEAN, "output_is_srgb", "OUTPUT_IS_SRGB" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "node_position_world", "NODE_POSITION_WORLD" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "camera_position_world", "CAMERA_POSITION_WORLD" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "camera_direction_world", "CAMERA_DIRECTION_WORLD" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "node_position_view", "NODE_POSITION_VIEW" },

	// Spatial, Fragment
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "fragcoord", "FRAGCOORD.xyz" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "vertex", "VERTEX" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "normal", "NORMAL" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "tangent", "TANGENT" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "binormal", "BINORMAL" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "view", "VIEW" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "uv", "vec3(UV, 0.0)" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "uv2", "vec3(UV2, 0.0)" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "color", "COLOR.rgb" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "alpha", "COLOR.a" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "point_coord", "vec3(POINT_COORD, 0.0)" },

	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "screen_uv", "vec3(SCREEN_UV, 0.0)" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "side", "float(FRONT_FACING ? 1.0 : 0.0)" },

	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_TRANSFORM, "world", "WORLD_MATRIX" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_TRANSFORM, "inv_camera", "INV_CAMERA_MATRIX" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_TRANSFORM, "camera", "CAMERA_MATRIX" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_TRANSFORM, "projection", "PROJECTION_MATRIX" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_TRANSFORM, "inv_projection", "INV_PROJECTION_MATRIX" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "time", "TIME" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "viewport_size", "vec3(VIEWPORT_SIZE, 0.0)" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_BOOLEAN, "output_is_srgb", "OUTPUT_IS_SRGB" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_BOOLEAN, "front_facing", "FRONT_FACING" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SAMPLER, "screen_texture", "SCREEN_TEXTURE" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SAMPLER, "depth_texture", "DEPTH_TEXTURE" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "node_position_world", "NODE_POSITION_WORLD" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "camera_position_world", "CAMERA_POSITION_WORLD" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "camera_direction_world", "CAMERA_DIRECTION_WORLD" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "node_position_view", "NODE_POSITION_VIEW" },

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
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_SCALAR, "metallic", "METALLIC" },

	{ Shader::MODE_SPATIAL, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_TRANSFORM, "world", "WORLD_MATRIX" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_TRANSFORM, "inv_camera", "INV_CAMERA_MATRIX" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_TRANSFORM, "camera", "CAMERA_MATRIX" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_TRANSFORM, "projection", "PROJECTION_MATRIX" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_TRANSFORM, "inv_projection", "INV_PROJECTION_MATRIX" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_SCALAR, "time", "TIME" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "viewport_size", "vec3(VIEWPORT_SIZE, 0.0)" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_BOOLEAN, "output_is_srgb", "OUTPUT_IS_SRGB" },
	// Canvas Item, Vertex
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "vertex", "vec3(VERTEX, 0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "uv", "vec3(UV,0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "color", "COLOR.rgb" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "alpha", "COLOR.a" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "modulate_color", "MODULATE.rgb" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "modulate_alpha", "MODULATE.a" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "point_size", "POINT_SIZE" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "texture_pixel_size", "vec3(TEXTURE_PIXEL_SIZE, 1.0)" },

	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_TRANSFORM, "world", "WORLD_MATRIX" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_TRANSFORM, "projection", "PROJECTION_MATRIX" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_TRANSFORM, "extra", "EXTRA_MATRIX" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "time", "TIME" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "light_pass", "float(AT_LIGHT_PASS ? 1.0 : 0.0)" },
	// Canvas Item, Fragment
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "fragcoord", "FRAGCOORD.xyz" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "uv", "vec3(UV, 0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "color", "COLOR.rgb" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "alpha", "COLOR.a" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "modulate_color", "MODULATE.rgb" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "modulate_alpha", "MODULATE.a" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "screen_uv", "vec3(SCREEN_UV, 0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "texture_pixel_size", "vec3(TEXTURE_PIXEL_SIZE, 1.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "screen_pixel_size", "vec3(SCREEN_PIXEL_SIZE, 1.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "point_coord", "vec3(POINT_COORD, 0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "time", "TIME" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "light_pass", "float(AT_LIGHT_PASS ? 1.0 : 0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SAMPLER, "texture", "TEXTURE" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SAMPLER, "normal_texture", "NORMAL_TEXTURE" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SAMPLER, "screen_texture", "SCREEN_TEXTURE" },
	// Canvas Item, Light
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "fragcoord", "FRAGCOORD.xyz" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "uv", "vec3(UV, 0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "normal", "NORMAL" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "color", "COLOR.rgb" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_SCALAR, "alpha", "COLOR.a" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "modulate_color", "MODULATE.rgb" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_SCALAR, "modulate_alpha", "MODULATE.a" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "light_vec", "vec3(LIGHT_VEC, 0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_SCALAR, "light_height", "LIGHT_HEIGHT" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "light_color", "LIGHT_COLOR.rgb" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_SCALAR, "light_alpha", "LIGHT_COLOR.a" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "light_uv", "vec3(LIGHT_UV, 0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "shadow_color", "SHADOW_COLOR.rgb" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_SCALAR, "shadow_alpha", "SHADOW_COLOR.a" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "shadow_vec", "vec3(SHADOW_VEC, 0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "screen_uv", "vec3(SCREEN_UV, 0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "texture_pixel_size", "vec3(TEXTURE_PIXEL_SIZE, 1.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "point_coord", "vec3(POINT_COORD, 0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_SCALAR, "time", "TIME" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_SAMPLER, "texture", "TEXTURE" },

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
	{ Shader::MODE_MAX, VisualShader::TYPE_MAX, VisualShaderNode::PORT_TYPE_TRANSFORM, nullptr, nullptr },
};

const VisualShaderNodeInput::Port VisualShaderNodeInput::preview_ports[] = {

	// Spatial, Fragment
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "normal", "vec3(0.0, 0.0, 1.0)" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "tangent", "vec3(0.0, 1.0, 0.0)" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "binormal", "vec3(1.0, 0.0, 0.0)" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "uv", "vec3(UV, 0.0)" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "uv2", "vec3(UV, 0.0)" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "color", "vec3(1.0)" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "alpha", "1.0" },

	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "screen_uv", "vec3(SCREEN_UV, 0.0)" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "side", "1.0" },

	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "time", "TIME" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "viewport_size", "vec3(1.0,1.0, 0.0)" },

	// Spatial, Light
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "normal", "vec3(0.0, 0.0, 1.0)" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_SCALAR, "time", "TIME" },
	{ Shader::MODE_SPATIAL, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "viewport_size", "vec3(1.0, 1.0, 0.0)" },
	// Canvas Item, Vertex
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "vertex", "vec3(VERTEX, 0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "uv", "vec3(UV, 0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "color", "vec3(1.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "alpha", "1.0" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "time", "TIME" },
	// Canvas Item, Fragment
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "uv", "vec3(UV, 0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "color", "vec3(1.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "alpha", "1.0" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_VECTOR, "screen_uv", "vec3(SCREEN_UV, 0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_FRAGMENT, VisualShaderNode::PORT_TYPE_SCALAR, "time", "TIME" },
	// Canvas Item, Light
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "uv", "vec3(UV, 0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "normal", "vec3(0.0, 0.0, 1.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "color", "vec3(1.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_SCALAR, "alpha", "1.0" },

	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_VECTOR, "screen_uv", "vec3(SCREEN_UV, 0.0)" },
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_SCALAR, "time", "TIME" },
	// Particles, Vertex
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "color", "vec3(1.0)" },
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "alpha", "1.0" },
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "velocity", "vec3(0.0, 0.0, 1.0)" },
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "time", "TIME" },
	{ Shader::MODE_MAX, VisualShader::TYPE_MAX, VisualShaderNode::PORT_TYPE_TRANSFORM, nullptr, nullptr },
};

int VisualShaderNodeInput::get_input_port_count() const {
	return 0;
}
VisualShaderNodeInput::PortType VisualShaderNodeInput::get_input_port_type(int p_port) const {
	return PORT_TYPE_SCALAR;
}
String VisualShaderNodeInput::get_input_port_name(int p_port) const {
	return "";
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
	return "Input";
}

String VisualShaderNodeInput::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	if (get_output_port_type(0) == PORT_TYPE_SAMPLER) {
		return "";
	}

	if (p_for_preview) {
		int idx = 0;

		String code;

		while (preview_ports[idx].mode != Shader::MODE_MAX) {
			if (preview_ports[idx].mode == shader_mode && preview_ports[idx].shader_type == shader_type && preview_ports[idx].name == input_name) {
				code = "\t" + p_output_vars[0] + " = " + preview_ports[idx].string + ";\n";
				break;
			}
			idx++;
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

		return code;

	} else {
		int idx = 0;

		String code;

		while (ports[idx].mode != Shader::MODE_MAX) {
			if (ports[idx].mode == shader_mode && ports[idx].shader_type == shader_type && ports[idx].name == input_name) {
				code = "\t" + p_output_vars[0] + " = " + ports[idx].string + ";\n";
				break;
			}
			idx++;
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

String VisualShaderNodeInput::get_input_real_name() const {
	int idx = 0;

	while (ports[idx].mode != Shader::MODE_MAX) {
		if (ports[idx].mode == shader_mode && ports[idx].shader_type == shader_type && ports[idx].name == input_name) {
			return String(ports[idx].string);
		}
		idx++;
	}

	return "";
}

VisualShaderNodeInput::PortType VisualShaderNodeInput::get_input_type_by_name(String p_name) const {
	int idx = 0;

	while (ports[idx].mode != Shader::MODE_MAX) {
		if (ports[idx].mode == shader_mode && ports[idx].shader_type == shader_type && ports[idx].name == p_name) {
			return ports[idx].type;
		}
		idx++;
	}

	return PORT_TYPE_SCALAR;
}

int VisualShaderNodeInput::get_input_index_count() const {
	int idx = 0;
	int count = 0;

	while (ports[idx].mode != Shader::MODE_MAX) {
		if (ports[idx].mode == shader_mode && ports[idx].shader_type == shader_type) {
			count++;
		}
		idx++;
	}

	return count;
}

VisualShaderNodeInput::PortType VisualShaderNodeInput::get_input_index_type(int p_index) const {
	int idx = 0;
	int count = 0;

	while (ports[idx].mode != Shader::MODE_MAX) {
		if (ports[idx].mode == shader_mode && ports[idx].shader_type == shader_type) {
			if (count == p_index) {
				return ports[idx].type;
			}
			count++;
		}
		idx++;
	}

	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeInput::get_input_index_name(int p_index) const {
	int idx = 0;
	int count = 0;

	while (ports[idx].mode != Shader::MODE_MAX) {
		if (ports[idx].mode == shader_mode && ports[idx].shader_type == shader_type) {
			if (count == p_index) {
				return ports[idx].name;
			}
			count++;
		}
		idx++;
	}

	return "";
}

void VisualShaderNodeInput::_validate_property(PropertyInfo &property) const {
	if (property.name == "input_name") {
		String port_list;

		int idx = 0;

		while (ports[idx].mode != Shader::MODE_MAX) {
			if (ports[idx].mode == shader_mode && ports[idx].shader_type == shader_type) {
				if (port_list != String()) {
					port_list += ",";
				}
				port_list += ports[idx].name;
			}
			idx++;
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
	ClassDB::bind_method(D_METHOD("get_input_real_name"), &VisualShaderNodeInput::get_input_real_name);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "input_name", PROPERTY_HINT_ENUM, ""), "set_input_name", "get_input_name");
	ADD_SIGNAL(MethodInfo("input_type_changed"));
}
VisualShaderNodeInput::VisualShaderNodeInput() {
	input_name = "[None]";
	// changed when set
	shader_type = VisualShader::TYPE_MAX;
	shader_mode = Shader::MODE_MAX;
}

////////////// UniformRef

List<VisualShaderNodeUniformRef::Uniform> uniforms;

void VisualShaderNodeUniformRef::add_uniform(const String &p_name, UniformType p_type) {
	uniforms.push_back({ p_name, p_type });
}

void VisualShaderNodeUniformRef::clear_uniforms() {
	uniforms.clear();
}

String VisualShaderNodeUniformRef::get_caption() const {
	return "UniformRef";
}

int VisualShaderNodeUniformRef::get_input_port_count() const {
	return 0;
}

VisualShaderNodeUniformRef::PortType VisualShaderNodeUniformRef::get_input_port_type(int p_port) const {
	return PortType::PORT_TYPE_SCALAR;
}

String VisualShaderNodeUniformRef::get_input_port_name(int p_port) const {
	return "";
}

int VisualShaderNodeUniformRef::get_output_port_count() const {
	switch (uniform_type) {
		case UniformType::UNIFORM_TYPE_SCALAR:
			return 1;
		case UniformType::UNIFORM_TYPE_BOOLEAN:
			return 1;
		case UniformType::UNIFORM_TYPE_VECTOR:
			return 1;
		case UniformType::UNIFORM_TYPE_TRANSFORM:
			return 1;
		case UniformType::UNIFORM_TYPE_COLOR:
			return 2;
		case UniformType::UNIFORM_TYPE_SAMPLER:
			return 1;
		default:
			break;
	}
	return 0;
}

VisualShaderNodeUniformRef::PortType VisualShaderNodeUniformRef::get_output_port_type(int p_port) const {
	switch (uniform_type) {
		case UniformType::UNIFORM_TYPE_SCALAR:
			return PortType::PORT_TYPE_SCALAR;
		case UniformType::UNIFORM_TYPE_BOOLEAN:
			return PortType::PORT_TYPE_BOOLEAN;
		case UniformType::UNIFORM_TYPE_VECTOR:
			return PortType::PORT_TYPE_VECTOR;
		case UniformType::UNIFORM_TYPE_TRANSFORM:
			return PortType::PORT_TYPE_TRANSFORM;
		case UniformType::UNIFORM_TYPE_COLOR:
			if (p_port == 0) {
				return PortType::PORT_TYPE_VECTOR;
			} else if (p_port == 1) {
				return PORT_TYPE_SCALAR;
			}
			break;
		case UniformType::UNIFORM_TYPE_SAMPLER:
			return PortType::PORT_TYPE_SAMPLER;
		default:
			break;
	}
	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeUniformRef::get_output_port_name(int p_port) const {
	switch (uniform_type) {
		case UniformType::UNIFORM_TYPE_SCALAR:
			return "";
		case UniformType::UNIFORM_TYPE_BOOLEAN:
			return "";
		case UniformType::UNIFORM_TYPE_VECTOR:
			return "";
		case UniformType::UNIFORM_TYPE_TRANSFORM:
			return "";
		case UniformType::UNIFORM_TYPE_COLOR:
			if (p_port == 0) {
				return "rgb";
			} else if (p_port == 1) {
				return "alpha";
			}
			break;
		case UniformType::UNIFORM_TYPE_SAMPLER:
			return "";
			break;
		default:
			break;
	}
	return "";
}

void VisualShaderNodeUniformRef::set_uniform_name(const String &p_name) {
	uniform_name = p_name;
	if (p_name != "[None]") {
		uniform_type = get_uniform_type_by_name(p_name);
	} else {
		uniform_type = UniformType::UNIFORM_TYPE_SCALAR;
	}
	emit_changed();
}

String VisualShaderNodeUniformRef::get_uniform_name() const {
	return uniform_name;
}

int VisualShaderNodeUniformRef::get_uniforms_count() const {
	return uniforms.size();
}

String VisualShaderNodeUniformRef::get_uniform_name_by_index(int p_idx) const {
	if (p_idx >= 0 && p_idx < uniforms.size()) {
		return uniforms[p_idx].name;
	}
	return "";
}

VisualShaderNodeUniformRef::UniformType VisualShaderNodeUniformRef::get_uniform_type_by_name(const String &p_name) const {
	for (int i = 0; i < uniforms.size(); i++) {
		if (uniforms[i].name == p_name) {
			return uniforms[i].type;
		}
	}
	return UniformType::UNIFORM_TYPE_SCALAR;
}

VisualShaderNodeUniformRef::UniformType VisualShaderNodeUniformRef::get_uniform_type_by_index(int p_idx) const {
	if (p_idx >= 0 && p_idx < uniforms.size()) {
		return uniforms[p_idx].type;
	}
	return UniformType::UNIFORM_TYPE_SCALAR;
}

String VisualShaderNodeUniformRef::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	switch (uniform_type) {
		case UniformType::UNIFORM_TYPE_SCALAR:
			if (uniform_name == "[None]") {
				return "\t" + p_output_vars[0] + " = 0.0f;\n";
			}
			return "\t" + p_output_vars[0] + " = " + get_uniform_name() + ";\n";
		case UniformType::UNIFORM_TYPE_BOOLEAN:
			return "\t" + p_output_vars[0] + " = " + get_uniform_name() + ";\n";
		case UniformType::UNIFORM_TYPE_VECTOR:
			return "\t" + p_output_vars[0] + " = " + get_uniform_name() + ";\n";
		case UniformType::UNIFORM_TYPE_TRANSFORM:
			return "\t" + p_output_vars[0] + " = " + get_uniform_name() + ";\n";
		case UniformType::UNIFORM_TYPE_COLOR: {
			String code = "\t" + p_output_vars[0] + " = " + get_uniform_name() + ".rgb;\n";
			code += "\t" + p_output_vars[1] + " = " + get_uniform_name() + ".a;\n";
			return code;
		} break;
		case UniformType::UNIFORM_TYPE_SAMPLER:
			break;
		default:
			break;
	}
	return "";
}

void VisualShaderNodeUniformRef::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_uniform_name", "name"), &VisualShaderNodeUniformRef::set_uniform_name);
	ClassDB::bind_method(D_METHOD("get_uniform_name"), &VisualShaderNodeUniformRef::get_uniform_name);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "uniform_name", PROPERTY_HINT_ENUM, ""), "set_uniform_name", "get_uniform_name");
}

Vector<StringName> VisualShaderNodeUniformRef::get_editable_properties() const {
	Vector<StringName> props;
	props.push_back("uniform_name");
	return props;
}

VisualShaderNodeUniformRef::VisualShaderNodeUniformRef() {
	uniform_name = "[None]";
	uniform_type = UniformType::UNIFORM_TYPE_SCALAR;
}

////////////////////////////////////////////

const VisualShaderNodeOutput::Port VisualShaderNodeOutput::ports[] = {
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
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_SCALAR, "light_alpha", "LIGHT.a" },
	// Particles, Vertex
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "color", "COLOR.rgb" },
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "alpha", "COLOR.a" },
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "velocity", "VELOCITY" },
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "custom", "CUSTOM.rgb" },
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "custom_alpha", "CUSTOM.a" },
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_TRANSFORM, "transform", "TRANSFORM" },
	{ Shader::MODE_MAX, VisualShader::TYPE_MAX, VisualShaderNode::PORT_TYPE_TRANSFORM, nullptr, nullptr },
};

int VisualShaderNodeOutput::get_input_port_count() const {
	int idx = 0;
	int count = 0;

	while (ports[idx].mode != Shader::MODE_MAX) {
		if (ports[idx].mode == shader_mode && ports[idx].shader_type == shader_type) {
			count++;
		}
		idx++;
	}

	return count;
}

VisualShaderNodeOutput::PortType VisualShaderNodeOutput::get_input_port_type(int p_port) const {
	int idx = 0;
	int count = 0;

	while (ports[idx].mode != Shader::MODE_MAX) {
		if (ports[idx].mode == shader_mode && ports[idx].shader_type == shader_type) {
			if (count == p_port) {
				return ports[idx].type;
			}
			count++;
		}
		idx++;
	}

	return PORT_TYPE_SCALAR;
}

String VisualShaderNodeOutput::get_input_port_name(int p_port) const {
	int idx = 0;
	int count = 0;

	while (ports[idx].mode != Shader::MODE_MAX) {
		if (ports[idx].mode == shader_mode && ports[idx].shader_type == shader_type) {
			if (count == p_port) {
				return String(ports[idx].name).capitalize();
			}
			count++;
		}
		idx++;
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

bool VisualShaderNodeOutput::is_port_separator(int p_index) const {
	if (shader_mode == Shader::MODE_SPATIAL && shader_type == VisualShader::TYPE_FRAGMENT) {
		String name = get_input_port_name(p_index);
		return (name == "Normal" || name == "Rim" || name == "Alpha Scissor");
	}
	return false;
}

String VisualShaderNodeOutput::get_caption() const {
	return "Output";
}

String VisualShaderNodeOutput::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	int idx = 0;
	int count = 0;

	String code;
	while (ports[idx].mode != Shader::MODE_MAX) {
		if (ports[idx].mode == shader_mode && ports[idx].shader_type == shader_type) {
			if (p_input_vars[count] != String()) {
				String s = ports[idx].string;
				if (s.find(":") != -1) {
					code += "\t" + s.get_slicec(':', 0) + " = " + p_input_vars[count] + "." + s.get_slicec(':', 1) + ";\n";
				} else {
					code += "\t" + s + " = " + p_input_vars[count] + ";\n";
				}
			}
			count++;
		}
		idx++;
	}

	return code;
}

VisualShaderNodeOutput::VisualShaderNodeOutput() {
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

void VisualShaderNodeUniform::set_global_code_generated(bool p_enabled) {
	global_code_generated = p_enabled;
}

bool VisualShaderNodeUniform::is_global_code_generated() const {
	return global_code_generated;
}

void VisualShaderNodeUniform::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_uniform_name", "name"), &VisualShaderNodeUniform::set_uniform_name);
	ClassDB::bind_method(D_METHOD("get_uniform_name"), &VisualShaderNodeUniform::get_uniform_name);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "uniform_name"), "set_uniform_name", "get_uniform_name");
}

bool VisualShaderNodeUniform::is_show_prop_names() const {
	return true;
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
	if (inputs == p_inputs) {
		return;
	}

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
	if (outputs == p_outputs) {
		return;
	}

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
	ERR_FAIL_COND(p_type < 0 || p_type >= PORT_TYPE_MAX);

	if (input_ports[p_id].type == p_type) {
		return;
	}

	Vector<String> inputs_strings = inputs.split(";", false);
	int count = 0;
	int index = 0;
	for (int i = 0; i < inputs_strings.size(); i++) {
		Vector<String> arr = inputs_strings[i].split(",");
		ERR_FAIL_COND(arr.size() != 3);

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

	if (input_ports[p_id].name == p_name) {
		return;
	}

	Vector<String> inputs_strings = inputs.split(";", false);
	int count = 0;
	int index = 0;
	for (int i = 0; i < inputs_strings.size(); i++) {
		Vector<String> arr = inputs_strings[i].split(",");
		ERR_FAIL_COND(arr.size() != 3);

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
	ERR_FAIL_COND(p_type < 0 || p_type >= PORT_TYPE_MAX);

	if (output_ports[p_id].type == p_type) {
		return;
	}

	Vector<String> output_strings = outputs.split(";", false);
	int count = 0;
	int index = 0;
	for (int i = 0; i < output_strings.size(); i++) {
		Vector<String> arr = output_strings[i].split(",");
		ERR_FAIL_COND(arr.size() != 3);

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

	if (output_ports[p_id].name == p_name) {
		return;
	}

	Vector<String> output_strings = outputs.split(";", false);
	int count = 0;
	int index = 0;
	for (int i = 0; i < output_strings.size(); i++) {
		Vector<String> arr = output_strings[i].split(",");
		ERR_FAIL_COND(arr.size() != 3);

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
	ERR_FAIL_COND_V(!controls.has(p_index), nullptr);
	return controls[p_index];
}

void VisualShaderNodeGroupBase::_apply_port_changes() {
	Vector<String> inputs_strings = inputs.split(";", false);
	Vector<String> outputs_strings = outputs.split(";", false);

	clear_input_ports();
	clear_output_ports();

	for (int i = 0; i < inputs_strings.size(); i++) {
		Vector<String> arr = inputs_strings[i].split(",");
		ERR_FAIL_COND(arr.size() != 3);

		Port port;
		port.type = (PortType)arr[1].to_int();
		port.name = arr[2];
		input_ports[i] = port;
	}
	for (int i = 0; i < outputs_strings.size(); i++) {
		Vector<String> arr = outputs_strings[i].split(",");
		ERR_FAIL_COND(arr.size() != 3);

		Port port;
		port.type = (PortType)arr[1].to_int();
		port.name = arr[2];
		output_ports[i] = port;
	}
}

void VisualShaderNodeGroupBase::set_editable(bool p_enabled) {
	editable = p_enabled;
}

bool VisualShaderNodeGroupBase::is_editable() const {
	return editable;
}

void VisualShaderNodeGroupBase::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_size", "size"), &VisualShaderNodeGroupBase::set_size);
	ClassDB::bind_method(D_METHOD("get_size"), &VisualShaderNodeGroupBase::get_size);

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

	ClassDB::bind_method(D_METHOD("set_input_port_name", "id", "name"), &VisualShaderNodeGroupBase::set_input_port_name);
	ClassDB::bind_method(D_METHOD("set_input_port_type", "id", "type"), &VisualShaderNodeGroupBase::set_input_port_type);
	ClassDB::bind_method(D_METHOD("set_output_port_name", "id", "name"), &VisualShaderNodeGroupBase::set_output_port_name);
	ClassDB::bind_method(D_METHOD("set_output_port_type", "id", "type"), &VisualShaderNodeGroupBase::set_output_port_type);

	ClassDB::bind_method(D_METHOD("get_free_input_port_id"), &VisualShaderNodeGroupBase::get_free_input_port_id);
	ClassDB::bind_method(D_METHOD("get_free_output_port_id"), &VisualShaderNodeGroupBase::get_free_output_port_id);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "size"), "set_size", "get_size");
}

String VisualShaderNodeGroupBase::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {
	return "";
}

VisualShaderNodeGroupBase::VisualShaderNodeGroupBase() {
	size = Size2(0, 0);
	inputs = "";
	outputs = "";
	editable = false;
	simple_decl = false;
}

////////////// Expression

String VisualShaderNodeExpression::get_caption() const {
	return "Expression";
}

void VisualShaderNodeExpression::set_expression(const String &p_expression) {
	expression = p_expression;
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
		pre_symbols.push_back(",");
		pre_symbols.push_back(";");
		pre_symbols.push_back("{");
		pre_symbols.push_back("[");
		pre_symbols.push_back("]");
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
		post_symbols.push_back("\t");
		post_symbols.push_back("\n");
		post_symbols.push_back(",");
		post_symbols.push_back(";");
		post_symbols.push_back("}");
		post_symbols.push_back("[");
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
		output_initializer += "\t" + p_output_vars[i] + " = " + tk + ";\n";
	}

	String code;
	code += output_initializer;
	code += "\t{";
	code += _expression;
	code += "\n\t}\n";

	return code;
}

void VisualShaderNodeExpression::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_expression", "expression"), &VisualShaderNodeExpression::set_expression);
	ClassDB::bind_method(D_METHOD("get_expression"), &VisualShaderNodeExpression::get_expression);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "expression"), "set_expression", "get_expression");
}

VisualShaderNodeExpression::VisualShaderNodeExpression() {
	expression = "";
	set_editable(true);
}

////////////// Global Expression

String VisualShaderNodeGlobalExpression::get_caption() const {
	return "GlobalExpression";
}

String VisualShaderNodeGlobalExpression::generate_global(Shader::Mode p_mode, VisualShader::Type p_type, int p_id) const {
	return expression;
}

VisualShaderNodeGlobalExpression::VisualShaderNodeGlobalExpression() {
	set_editable(false);
}
