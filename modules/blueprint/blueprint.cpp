/**************************************************************************/
/*  blueprint.cpp                                                         */
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

#include "blueprint.h"

#include "core/object/class_db.h"

static Vector<BlueprintNodeDef> node_defs;

static void _init_node_defs() {
	if (!node_defs.is_empty()) {
		return;
	}

	{
		BlueprintNodeDef def;
		def.type = "event_ready";
		def.category = "event";
		def.title = "On Ready";
		def.is_event = true;
		def.outputs.push_back({ "", true, "" });
		node_defs.push_back(def);
	}
	{
		BlueprintNodeDef def;
		def.type = "event_process";
		def.category = "event";
		def.title = "On Process";
		def.is_event = true;
		def.outputs.push_back({ "", true, "" });
		def.outputs.push_back({ "delta", false, "" });
		node_defs.push_back(def);
	}
	{
		BlueprintNodeDef def;
		def.type = "event_input";
		def.category = "event";
		def.title = "On Input Action";
		def.is_event = true;
		def.config_params.push_back("action");
		def.outputs.push_back({ "pressed", true, "" });
		def.outputs.push_back({ "released", true, "" });
		def.default_params["action"] = "ui_accept";
		node_defs.push_back(def);
	}
	{
		BlueprintNodeDef def;
		def.type = "event_signal";
		def.category = "event";
		def.title = "On Signal";
		def.is_event = true;
		def.config_params.push_back("node");
		def.config_params.push_back("signal");
		def.outputs.push_back({ "", true, "" });
		def.default_params["node"] = "..";
		def.default_params["signal"] = "";
		node_defs.push_back(def);
	}
	{
		BlueprintNodeDef def;
		def.type = "print";
		def.category = "action";
		def.title = "Print";
		def.inputs.push_back({ "", true, "" });
		def.inputs.push_back({ "text", false, "text" });
		def.outputs.push_back({ "", true, "" });
		def.default_params["text"] = "Hello";
		node_defs.push_back(def);
	}
	{
		BlueprintNodeDef def;
		def.type = "call_method";
		def.category = "action";
		def.title = "Call Method";
		def.config_params.push_back("node");
		def.config_params.push_back("method");
		def.inputs.push_back({ "", true, "" });
		def.inputs.push_back({ "arg1", false, "arg1" });
		def.inputs.push_back({ "arg2", false, "arg2" });
		def.outputs.push_back({ "", true, "" });
		def.outputs.push_back({ "result", false, "" });
		def.default_params["node"] = "..";
		def.default_params["method"] = "";
		def.default_params["arg1"] = "";
		def.default_params["arg2"] = "";
		node_defs.push_back(def);
	}
	{
		BlueprintNodeDef def;
		def.type = "set_property";
		def.category = "action";
		def.title = "Set Property";
		def.config_params.push_back("node");
		def.config_params.push_back("property");
		def.inputs.push_back({ "", true, "" });
		def.inputs.push_back({ "value", false, "value" });
		def.outputs.push_back({ "", true, "" });
		def.default_params["node"] = "..";
		def.default_params["property"] = "";
		def.default_params["value"] = "";
		node_defs.push_back(def);
	}
	{
		BlueprintNodeDef def;
		def.type = "branch";
		def.category = "flow";
		def.title = "Branch";
		def.inputs.push_back({ "", true, "" });
		def.inputs.push_back({ "condition", false, "condition" });
		def.outputs.push_back({ "true", true, "" });
		def.outputs.push_back({ "false", true, "" });
		def.default_params["condition"] = "false";
		node_defs.push_back(def);
	}
	{
		BlueprintNodeDef def;
		def.type = "get_property";
		def.category = "data";
		def.title = "Get Property";
		def.config_params.push_back("node");
		def.config_params.push_back("property");
		def.outputs.push_back({ "value", false, "" });
		def.default_params["node"] = "..";
		def.default_params["property"] = "";
		node_defs.push_back(def);
	}
	{
		BlueprintNodeDef def;
		def.type = "constant";
		def.category = "data";
		def.title = "Constant";
		def.outputs.push_back({ "value", false, "value" });
		def.default_params["value"] = "0";
		node_defs.push_back(def);
	}
	{
		BlueprintNodeDef def;
		def.type = "add";
		def.category = "data";
		def.title = "Add (a + b)";
		def.inputs.push_back({ "a", false, "a" });
		def.inputs.push_back({ "b", false, "b" });
		def.outputs.push_back({ "result", false, "" });
		def.default_params["a"] = "0";
		def.default_params["b"] = "0";
		node_defs.push_back(def);
	}
}

const Vector<BlueprintNodeDef> &blueprint_get_node_defs() {
	_init_node_defs();
	return node_defs;
}

const BlueprintNodeDef *blueprint_get_node_def(const String &p_type) {
	_init_node_defs();
	for (const BlueprintNodeDef &def : node_defs) {
		if (def.type == p_type) {
			return &def;
		}
	}
	return nullptr;
}

void Blueprint::set_nodes(const Array &p_nodes) {
	nodes = p_nodes;
	emit_changed();
}

Array Blueprint::get_nodes() const {
	return nodes;
}

void Blueprint::set_connections(const Array &p_connections) {
	connections = p_connections;
	emit_changed();
}

Array Blueprint::get_connections() const {
	return connections;
}

void Blueprint::set_next_id(int p_next_id) {
	next_id = p_next_id;
}

int Blueprint::get_next_id() const {
	return next_id;
}

int Blueprint::add_node(const String &p_type, const Vector2 &p_position) {
	const BlueprintNodeDef *def = blueprint_get_node_def(p_type);
	ERR_FAIL_NULL_V_MSG(def, -1, vformat("Unknown blueprint node type: %s.", p_type));

	Dictionary node;
	node["id"] = next_id++;
	node["type"] = p_type;
	node["position"] = p_position;
	node["params"] = def->default_params.duplicate();
	nodes.push_back(node);
	emit_changed();
	return node["id"];
}

void Blueprint::remove_node(int p_id) {
	for (int i = nodes.size() - 1; i >= 0; i--) {
		Dictionary node = nodes[i];
		if (int(node["id"]) == p_id) {
			nodes.remove_at(i);
		}
	}
	for (int i = connections.size() - 1; i >= 0; i--) {
		Dictionary conn = connections[i];
		if (int(conn["from_node"]) == p_id || int(conn["to_node"]) == p_id) {
			connections.remove_at(i);
		}
	}
	emit_changed();
}

Dictionary Blueprint::get_node_data(int p_id) const {
	for (int i = 0; i < nodes.size(); i++) {
		Dictionary node = nodes[i];
		if (int(node["id"]) == p_id) {
			return node;
		}
	}
	return Dictionary();
}

void Blueprint::set_node_position(int p_id, const Vector2 &p_position) {
	for (int i = 0; i < nodes.size(); i++) {
		Dictionary node = nodes[i];
		if (int(node["id"]) == p_id) {
			node["position"] = p_position;
			emit_changed();
			return;
		}
	}
}

void Blueprint::set_node_param(int p_id, const String &p_key, const Variant &p_value) {
	for (int i = 0; i < nodes.size(); i++) {
		Dictionary node = nodes[i];
		if (int(node["id"]) == p_id) {
			Dictionary params = node["params"];
			params[p_key] = p_value;
			node["params"] = params;
			emit_changed();
			return;
		}
	}
}

void Blueprint::add_connection(int p_from_node, int p_from_port, int p_to_node, int p_to_port) {
	Dictionary conn;
	conn["from_node"] = p_from_node;
	conn["from_port"] = p_from_port;
	conn["to_node"] = p_to_node;
	conn["to_port"] = p_to_port;
	connections.push_back(conn);
	emit_changed();
}

void Blueprint::remove_connection(int p_from_node, int p_from_port, int p_to_node, int p_to_port) {
	for (int i = connections.size() - 1; i >= 0; i--) {
		Dictionary conn = connections[i];
		if (int(conn["from_node"]) == p_from_node && int(conn["from_port"]) == p_from_port &&
				int(conn["to_node"]) == p_to_node && int(conn["to_port"]) == p_to_port) {
			connections.remove_at(i);
		}
	}
	emit_changed();
}

PackedStringArray Blueprint::get_node_type_list() {
	PackedStringArray list;
	for (const BlueprintNodeDef &def : blueprint_get_node_defs()) {
		list.push_back(def.type);
	}
	return list;
}

void Blueprint::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_nodes", "nodes"), &Blueprint::set_nodes);
	ClassDB::bind_method(D_METHOD("get_nodes"), &Blueprint::get_nodes);
	ClassDB::bind_method(D_METHOD("set_connections", "connections"), &Blueprint::set_connections);
	ClassDB::bind_method(D_METHOD("get_connections"), &Blueprint::get_connections);
	ClassDB::bind_method(D_METHOD("set_next_id", "next_id"), &Blueprint::set_next_id);
	ClassDB::bind_method(D_METHOD("get_next_id"), &Blueprint::get_next_id);

	ClassDB::bind_method(D_METHOD("add_node", "type", "position"), &Blueprint::add_node);
	ClassDB::bind_method(D_METHOD("remove_node", "id"), &Blueprint::remove_node);
	ClassDB::bind_method(D_METHOD("get_node_data", "id"), &Blueprint::get_node_data);
	ClassDB::bind_method(D_METHOD("set_node_position", "id", "position"), &Blueprint::set_node_position);
	ClassDB::bind_method(D_METHOD("set_node_param", "id", "key", "value"), &Blueprint::set_node_param);
	ClassDB::bind_method(D_METHOD("add_connection", "from_node", "from_port", "to_node", "to_port"), &Blueprint::add_connection);
	ClassDB::bind_method(D_METHOD("remove_connection", "from_node", "from_port", "to_node", "to_port"), &Blueprint::remove_connection);

	ClassDB::bind_static_method("Blueprint", D_METHOD("get_node_type_list"), &Blueprint::get_node_type_list);

	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "nodes", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_nodes", "get_nodes");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "connections", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_connections", "get_connections");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "next_id", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_next_id", "get_next_id");
}
