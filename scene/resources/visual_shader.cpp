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
		if (E->get().node == p_node)
			return E->key();
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

bool VisualShader::can_connect_nodes(Type p_type, int p_from_node, int p_from_port, int p_to_node, int p_to_port) const {

	ERR_FAIL_INDEX_V(p_type, TYPE_MAX, false);
	const Graph *g = &graph[p_type];

	if (!g->nodes.has(p_from_node))
		return false;

	if (p_from_port < 0 || p_from_port >= g->nodes[p_from_node].node->get_output_port_count())
		return false;

	if (!g->nodes.has(p_to_node))
		return false;

	if (p_to_port < 0 || p_to_port >= g->nodes[p_to_node].node->get_input_port_count())
		return false;

	VisualShaderNode::PortType from_port_type = g->nodes[p_from_node].node->get_output_port_type(p_from_port);
	VisualShaderNode::PortType to_port_type = g->nodes[p_to_node].node->get_input_port_type(p_to_port);

	if (MAX(0, from_port_type - 2) != (MAX(0, to_port_type - 2))) {
		return false;
	}

	for (const List<Connection>::Element *E = g->connections.front(); E; E = E->next()) {

		if (E->get().from_node == p_from_node && E->get().from_port == p_from_port && E->get().to_node == p_to_node && E->get().to_port == p_to_port) {
			return false;
		}
	}

	return true;
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

	if (MAX(0, from_port_type - 2) != (MAX(0, to_port_type - 2))) {
		ERR_EXPLAIN("Incompatible port types (scalar/vec/bool with transform");
		ERR_FAIL_V(ERR_INVALID_PARAMETER)
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
	ERR_FAIL_INDEX(p_type, TYPE_MAX);
	Graph *g = &graph[p_type];

	for (List<Connection>::Element *E = g->connections.front(); E; E = E->next()) {

		if (E->get().from_node == p_from_node && E->get().from_port == p_from_port && E->get().to_node == p_to_node && E->get().to_port == p_to_port) {
			g->connections.erase(E);
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
	StringBuilder code;

	global_code += String() + "shader_type canvas_item;\n";

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
	{ Shader::MODE_SPATIAL, "blend" },
	{ Shader::MODE_SPATIAL, "depth_draw" },
	{ Shader::MODE_SPATIAL, "cull" },
	{ Shader::MODE_SPATIAL, "diffuse" },
	{ Shader::MODE_SPATIAL, "specular" },
	{ Shader::MODE_CANVAS_ITEM, "blend" },
	{ Shader::MODE_CANVAS_ITEM, NULL }
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
					connect_nodes(type, conns[i + 0], conns[i + 1], conns[i + 2], conns[i + 3]);
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

	for (int i = 0; i < TYPE_MAX; i++) {
		for (Map<int, Node>::Element *E = graph[i].nodes.front(); E; E = E->next()) {

			String prop_name = "nodes/";
			prop_name += type_string[i];
			prop_name += "/" + itos(E->key());

			if (E->key() != NODE_ID_OUTPUT) {

				p_list->push_back(PropertyInfo(Variant::OBJECT, prop_name + "/node", PROPERTY_HINT_RESOURCE_TYPE, "VisualShaderNode", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_DO_NOT_SHARE_ON_DUPLICATE));
			}
			p_list->push_back(PropertyInfo(Variant::VECTOR2, prop_name + "/position", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
		}
		p_list->push_back(PropertyInfo(Variant::POOL_INT_ARRAY, "nodes/" + String(type_string[i]) + "/connections", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
	}
}

Error VisualShader::_write_node(Type type, StringBuilder &global_code, StringBuilder &code, Vector<VisualShader::DefaultTextureParam> &def_tex_params, const VMap<ConnectionKey, const List<Connection>::Element *> &input_connections, const VMap<ConnectionKey, const List<Connection>::Element *> &output_connections, int node, Set<int> &processed, bool for_preview) const {

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
			VisualShaderNode::PortType out_type = graph[type].nodes[from_node].node->get_output_port_type(from_port);

			String src_var = "n_out" + itos(from_node) + "p" + itos(from_port);

			if (in_type == out_type) {
				inputs[i] = src_var;
			} else if (in_type == VisualShaderNode::PORT_TYPE_SCALAR && out_type == VisualShaderNode::PORT_TYPE_VECTOR) {
				inputs[i] = "dot(" + src_var + ",vec3(0.333333,0.333333,0.333333))";
			} else if (in_type == VisualShaderNode::PORT_TYPE_VECTOR && out_type == VisualShaderNode::PORT_TYPE_SCALAR) {
				inputs[i] = "vec3(" + src_var + ")";
			} else if (in_type == VisualShaderNode::PORT_TYPE_BOOLEAN && out_type == VisualShaderNode::PORT_TYPE_VECTOR) {
				inputs[i] = "all(" + src_var + ")";
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

	static const char *func_name[TYPE_MAX] = { "vertex", "fragment", "light" };

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
	Graph *g = &graph[p_type];

	for (List<Connection>::Element *E = g->connections.front(); E;) {
		List<Connection>::Element *N = E->next();
		if (E->get().from_node == p_id) {
			g->connections.erase(E);
		}
		E = N;
	}
}

void VisualShader::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_mode", "mode"), &VisualShader::set_mode);

	ClassDB::bind_method(D_METHOD("add_node", "type", "node", "position", "id"), &VisualShader::add_node);
	ClassDB::bind_method(D_METHOD("set_node_position", "type", "id", "position"), &VisualShader::set_node_position);

	ClassDB::bind_method(D_METHOD("get_node", "type", "id"), &VisualShader::get_node);
	ClassDB::bind_method(D_METHOD("get_node_position", "type", "id"), &VisualShader::get_node_position);

	ClassDB::bind_method(D_METHOD("get_node_list", "type"), &VisualShader::get_node_list);
	ClassDB::bind_method(D_METHOD("get_valid_node_id", "type"), &VisualShader::get_valid_node_id);

	ClassDB::bind_method(D_METHOD("remove_node", "type", "id"), &VisualShader::remove_node);

	ClassDB::bind_method(D_METHOD("is_node_connection", "type", "from_node", "from_port", "to_node", "to_port"), &VisualShader::is_node_connection);
	ClassDB::bind_method(D_METHOD("can_connect_nodes", "type", "from_node", "from_port", "to_node", "to_port"), &VisualShader::is_node_connection);

	ClassDB::bind_method(D_METHOD("connect_nodes", "type", "from_node", "from_port", "to_node", "to_port"), &VisualShader::connect_nodes);
	ClassDB::bind_method(D_METHOD("disconnect_nodes", "type", "from_node", "from_port", "to_node", "to_port"), &VisualShader::disconnect_nodes);

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
		Ref<VisualShaderNodeOutput> output;
		output.instance();
		output->shader_type = Type(i);
		output->shader_mode = shader_mode;
		graph[i].nodes[NODE_ID_OUTPUT].node = output;
		graph[i].nodes[NODE_ID_OUTPUT].position = Vector2(400, 150);
	}

	dirty = true;
}

///////////////////////////////////////////////////////////

const VisualShaderNodeInput::Port VisualShaderNodeInput::ports[] = {
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
	{ Shader::MODE_MAX, VisualShader::TYPE_MAX, VisualShaderNode::PORT_TYPE_TRANSFORM, NULL, NULL },
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
	{ Shader::MODE_MAX, VisualShader::TYPE_MAX, VisualShaderNode::PORT_TYPE_TRANSFORM, NULL, NULL },
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
	return TTR("Input");
}

String VisualShaderNodeInput::generate_code(Shader::Mode p_mode, VisualShader::Type p_type, int p_id, const String *p_input_vars, const String *p_output_vars, bool p_for_preview) const {

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

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "input_name", PROPERTY_HINT_ENUM, ""), "set_input_name", "get_input_name");
	ADD_SIGNAL(MethodInfo("input_type_changed"));
}
VisualShaderNodeInput::VisualShaderNodeInput() {
	input_name = "[None]";
	// changed when set
	shader_type = VisualShader::TYPE_MAX;
	shader_mode = Shader::MODE_MAX;
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
	{ Shader::MODE_CANVAS_ITEM, VisualShader::TYPE_LIGHT, VisualShaderNode::PORT_TYPE_SCALAR, "light_alpha", "LIGHT.rgb" },
	// Particles, Vertex
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "color", "COLOR.rgb" },
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "alpha", "COLOR.a" },
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "velocity", "VELOCITY" },
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_VECTOR, "custom", "CUSTOM.rgb" },
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_SCALAR, "custom_alpha", "CUSTOM.a" },
	{ Shader::MODE_PARTICLES, VisualShader::TYPE_VERTEX, VisualShaderNode::PORT_TYPE_TRANSFORM, "transform", "TRANSFORM" },
	{ Shader::MODE_MAX, VisualShader::TYPE_MAX, VisualShaderNode::PORT_TYPE_TRANSFORM, NULL, NULL },
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
	return TTR("Output");
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

void VisualShaderNodeUniform::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_uniform_name", "name"), &VisualShaderNodeUniform::set_uniform_name);
	ClassDB::bind_method(D_METHOD("get_uniform_name"), &VisualShaderNodeUniform::get_uniform_name);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "uniform_name"), "set_uniform_name", "get_uniform_name");
}

VisualShaderNodeUniform::VisualShaderNodeUniform() {
}
