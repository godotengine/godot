/*************************************************************************/
/*  visual_script.cpp                                                    */
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

#include "visual_script.h"

#include "core/config/project_settings.h"
#include "core/core_string_names.h"
#include "core/os/os.h"
#include "scene/main/node.h"
#include "visual_script_nodes.h"

// Used by editor, this is not really saved.
void VisualScriptNode::set_breakpoint(bool p_breakpoint) {
	breakpoint = p_breakpoint;
}

bool VisualScriptNode::is_breakpoint() const {
	return breakpoint;
}

void VisualScriptNode::ports_changed_notify() {
	emit_signal(SNAME("ports_changed"));
}

void VisualScriptNode::set_default_input_value(int p_port, const Variant &p_value) {
	ERR_FAIL_INDEX(p_port, default_input_values.size());

	default_input_values[p_port] = p_value;

#ifdef TOOLS_ENABLED
	if (script_used.is_valid()) {
		script_used->set_edited(true);
	}
#endif
}

Variant VisualScriptNode::get_default_input_value(int p_port) const {
	ERR_FAIL_INDEX_V(p_port, default_input_values.size(), Variant());
	return default_input_values[p_port];
}

void VisualScriptNode::_set_default_input_values(Array p_values) {
	default_input_values = p_values;
}

void VisualScriptNode::validate_input_default_values() {
	default_input_values.resize(MAX(default_input_values.size(), get_input_value_port_count())); //let it grow as big as possible, we don't want to lose values on resize

	// Actually validate on save.
	for (int i = 0; i < get_input_value_port_count(); i++) {
		Variant::Type expected = get_input_value_port_info(i).type;

		if (expected == Variant::NIL || expected == default_input_values[i].get_type()) {
			continue;
		} else {
			// Not the same, reconvert.
			Callable::CallError ce;
			Variant existing = default_input_values[i];
			const Variant *existingp = &existing;
			Variant::construct(expected, default_input_values[i], &existingp, 1, ce);
			if (ce.error != Callable::CallError::CALL_OK) {
				//could not convert? force..
				Variant::construct(expected, default_input_values[i], nullptr, 0, ce);
			}
		}
	}
}

Array VisualScriptNode::_get_default_input_values() const {
	// Validate on save, since on load there is little info about this.
	Array values = default_input_values;
	values.resize(get_input_value_port_count());

	return values;
}

String VisualScriptNode::get_text() const {
	return "";
}

void VisualScriptNode::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_visual_script"), &VisualScriptNode::get_visual_script);
	ClassDB::bind_method(D_METHOD("set_default_input_value", "port_idx", "value"), &VisualScriptNode::set_default_input_value);
	ClassDB::bind_method(D_METHOD("get_default_input_value", "port_idx"), &VisualScriptNode::get_default_input_value);
	ClassDB::bind_method(D_METHOD("ports_changed_notify"), &VisualScriptNode::ports_changed_notify);
	ClassDB::bind_method(D_METHOD("_set_default_input_values", "values"), &VisualScriptNode::_set_default_input_values);
	ClassDB::bind_method(D_METHOD("_get_default_input_values"), &VisualScriptNode::_get_default_input_values);

	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "_default_input_values", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL), "_set_default_input_values", "_get_default_input_values");
	ADD_SIGNAL(MethodInfo("ports_changed"));
}

VisualScriptNode::TypeGuess VisualScriptNode::guess_output_type(TypeGuess *p_inputs, int p_output) const {
	ERR_FAIL_COND_V(get_output_value_port_count() <= p_output, TypeGuess());

	PropertyInfo pinfo = get_output_value_port_info(p_output);

	TypeGuess tg;

	tg.type = pinfo.type;
	if (pinfo.hint == PROPERTY_HINT_RESOURCE_TYPE) {
		tg.gdclass = pinfo.hint_string;
	}

	return tg;
}

Ref<VisualScript> VisualScriptNode::get_visual_script() const {
	return script_used;
}

VisualScriptNode::VisualScriptNode() {
}

////////////////

/////////////////////

VisualScriptNodeInstance::VisualScriptNodeInstance() {
}

VisualScriptNodeInstance::~VisualScriptNodeInstance() {
	if (sequence_outputs) {
		memdelete_arr(sequence_outputs);
	}

	if (input_ports) {
		memdelete_arr(input_ports);
	}

	if (output_ports) {
		memdelete_arr(output_ports);
	}
}

void VisualScript::add_function(const StringName &p_name, int p_func_node_id) {
	ERR_FAIL_COND(instances.size());
	ERR_FAIL_COND(!String(p_name).is_valid_identifier());
	ERR_FAIL_COND(functions.has(p_name));
	ERR_FAIL_COND(variables.has(p_name));
	ERR_FAIL_COND(custom_signals.has(p_name));

	functions[p_name] = Function();
	functions[p_name].func_id = p_func_node_id;
}

bool VisualScript::has_function(const StringName &p_name) const {
	return functions.has(p_name);
}

void VisualScript::remove_function(const StringName &p_name) {
	ERR_FAIL_COND(instances.size());
	ERR_FAIL_COND(!functions.has(p_name));

	// Let the editor handle the node removal.
	functions.erase(p_name);
}

void VisualScript::rename_function(const StringName &p_name, const StringName &p_new_name) {
	ERR_FAIL_COND(instances.size());
	ERR_FAIL_COND(!functions.has(p_name));
	if (p_new_name == p_name) {
		return;
	}

	ERR_FAIL_COND(!String(p_new_name).is_valid_identifier());

	ERR_FAIL_COND(functions.has(p_new_name));
	ERR_FAIL_COND(variables.has(p_new_name));
	ERR_FAIL_COND(custom_signals.has(p_new_name));

	functions[p_new_name] = functions[p_name];
	functions.erase(p_name);
}

void VisualScript::set_scroll(const Vector2 &p_scroll) {
	scroll = p_scroll;
}

Vector2 VisualScript::get_scroll() const {
	return scroll;
}

void VisualScript::get_function_list(List<StringName> *r_functions) const {
	functions.get_key_list(r_functions);
	// r_functions->sort_custom<StringName::AlphCompare>(); // Don't force sorting.
}

int VisualScript::get_function_node_id(const StringName &p_name) const {
	ERR_FAIL_COND_V(!functions.has(p_name), -1);

	return functions[p_name].func_id;
}

void VisualScript::_node_ports_changed(int p_id) {
	Ref<VisualScriptNode> vsn = nodes[p_id].node;

	vsn->validate_input_default_values();

	// Must revalidate all the functions.

	{
		List<SequenceConnection> to_remove;

		for (Set<SequenceConnection>::Element *E = sequence_connections.front(); E; E = E->next()) {
			if (E->get().from_node == p_id && E->get().from_output >= vsn->get_output_sequence_port_count()) {
				to_remove.push_back(E->get());
			}
			if (E->get().to_node == p_id && !vsn->has_input_sequence_port()) {
				to_remove.push_back(E->get());
			}
		}

		while (to_remove.size()) {
			sequence_connections.erase(to_remove.front()->get());
			to_remove.pop_front();
		}
	}

	{
		List<DataConnection> to_remove;

		for (Set<DataConnection>::Element *E = data_connections.front(); E; E = E->next()) {
			if (E->get().from_node == p_id && E->get().from_port >= vsn->get_output_value_port_count()) {
				to_remove.push_back(E->get());
			}
			if (E->get().to_node == p_id && E->get().to_port >= vsn->get_input_value_port_count()) {
				to_remove.push_back(E->get());
			}
		}

		while (to_remove.size()) {
			data_connections.erase(to_remove.front()->get());
			to_remove.pop_front();
		}
	}

#ifdef TOOLS_ENABLED
	set_edited(true); // Something changed, let's set as edited.
	emit_signal(SNAME("node_ports_changed"), p_id);
#endif
}

void VisualScript::add_node(int p_id, const Ref<VisualScriptNode> &p_node, const Point2 &p_pos) {
	ERR_FAIL_COND(instances.size());
	ERR_FAIL_COND(nodes.has(p_id)); // ID can exist only one in script.
	ERR_FAIL_COND(p_node.is_null());

	NodeData nd;
	nd.node = p_node;
	nd.pos = p_pos;

	Ref<VisualScriptNode> vsn = p_node;
	vsn->connect("ports_changed", callable_mp(this, &VisualScript::_node_ports_changed), varray(p_id));
	vsn->script_used = Ref<VisualScript>(this);
	vsn->validate_input_default_values(); // Validate when fully loaded.

	nodes[p_id] = nd;
}

void VisualScript::remove_node(int p_id) {
	ERR_FAIL_COND(instances.size());
	ERR_FAIL_COND(!nodes.has(p_id));
	{
		List<SequenceConnection> to_remove;

		for (Set<SequenceConnection>::Element *E = sequence_connections.front(); E; E = E->next()) {
			if (E->get().from_node == p_id || E->get().to_node == p_id) {
				to_remove.push_back(E->get());
			}
		}

		while (to_remove.size()) {
			sequence_connections.erase(to_remove.front()->get());
			to_remove.pop_front();
		}
	}

	{
		List<DataConnection> to_remove;

		for (Set<DataConnection>::Element *E = data_connections.front(); E; E = E->next()) {
			if (E->get().from_node == p_id || E->get().to_node == p_id) {
				to_remove.push_back(E->get());
			}
		}

		while (to_remove.size()) {
			data_connections.erase(to_remove.front()->get());
			to_remove.pop_front();
		}
	}

	nodes[p_id].node->disconnect("ports_changed", callable_mp(this, &VisualScript::_node_ports_changed));
	nodes[p_id].node->script_used.unref();

	nodes.erase(p_id);
}

bool VisualScript::has_node(int p_id) const {
	return nodes.has(p_id);
}

Ref<VisualScriptNode> VisualScript::get_node(int p_id) const {
	ERR_FAIL_COND_V(!nodes.has(p_id), Ref<VisualScriptNode>());

	return nodes[p_id].node;
}

void VisualScript::set_node_position(int p_id, const Point2 &p_pos) {
	ERR_FAIL_COND(instances.size());
	ERR_FAIL_COND(!nodes.has(p_id));
	nodes[p_id].pos = p_pos;
}

Point2 VisualScript::get_node_position(int p_id) const {
	ERR_FAIL_COND_V(!nodes.has(p_id), Point2());
	return nodes[p_id].pos;
}

void VisualScript::get_node_list(List<int> *r_nodes) const {
	nodes.get_key_list(r_nodes);
}

void VisualScript::sequence_connect(int p_from_node, int p_from_output, int p_to_node) {
	ERR_FAIL_COND(instances.size());

	SequenceConnection sc;
	sc.from_node = p_from_node;
	sc.from_output = p_from_output;
	sc.to_node = p_to_node;
	ERR_FAIL_COND(sequence_connections.has(sc));

	sequence_connections.insert(sc);
}

void VisualScript::sequence_disconnect(int p_from_node, int p_from_output, int p_to_node) {
	SequenceConnection sc;
	sc.from_node = p_from_node;
	sc.from_output = p_from_output;
	sc.to_node = p_to_node;
	ERR_FAIL_COND(!sequence_connections.has(sc));

	sequence_connections.erase(sc);
}

bool VisualScript::has_sequence_connection(int p_from_node, int p_from_output, int p_to_node) const {
	SequenceConnection sc;
	sc.from_node = p_from_node;
	sc.from_output = p_from_output;
	sc.to_node = p_to_node;

	return sequence_connections.has(sc);
}

void VisualScript::get_sequence_connection_list(List<SequenceConnection> *r_connection) const {
	for (const Set<SequenceConnection>::Element *E = sequence_connections.front(); E; E = E->next()) {
		r_connection->push_back(E->get());
	}
}

void VisualScript::data_connect(int p_from_node, int p_from_port, int p_to_node, int p_to_port) {
	ERR_FAIL_COND(instances.size());

	DataConnection dc;
	dc.from_node = p_from_node;
	dc.from_port = p_from_port;
	dc.to_node = p_to_node;
	dc.to_port = p_to_port;

	ERR_FAIL_COND(data_connections.has(dc));

	data_connections.insert(dc);
}

void VisualScript::data_disconnect(int p_from_node, int p_from_port, int p_to_node, int p_to_port) {
	DataConnection dc;
	dc.from_node = p_from_node;
	dc.from_port = p_from_port;
	dc.to_node = p_to_node;
	dc.to_port = p_to_port;

	ERR_FAIL_COND(!data_connections.has(dc));

	data_connections.erase(dc);
}

bool VisualScript::has_data_connection(int p_from_node, int p_from_port, int p_to_node, int p_to_port) const {
	DataConnection dc;
	dc.from_node = p_from_node;
	dc.from_port = p_from_port;
	dc.to_node = p_to_node;
	dc.to_port = p_to_port;

	return data_connections.has(dc);
}

bool VisualScript::is_input_value_port_connected(int p_node, int p_port) const {
	for (const Set<DataConnection>::Element *E = data_connections.front(); E; E = E->next()) {
		if (E->get().to_node == p_node && E->get().to_port == p_port) {
			return true;
		}
	}
	return false;
}

bool VisualScript::get_input_value_port_connection_source(int p_node, int p_port, int *r_node, int *r_port) const {
	for (const Set<DataConnection>::Element *E = data_connections.front(); E; E = E->next()) {
		if (E->get().to_node == p_node && E->get().to_port == p_port) {
			*r_node = E->get().from_node;
			*r_port = E->get().from_port;
			return true;
		}
	}
	return false;
}

void VisualScript::get_data_connection_list(List<DataConnection> *r_connection) const {
	for (const Set<DataConnection>::Element *E = data_connections.front(); E; E = E->next()) {
		r_connection->push_back(E->get());
	}
}

void VisualScript::set_tool_enabled(bool p_enabled) {
	is_tool_script = p_enabled;
}

void VisualScript::add_variable(const StringName &p_name, const Variant &p_default_value, bool p_export) {
	ERR_FAIL_COND(instances.size());
	ERR_FAIL_COND(!String(p_name).is_valid_identifier());
	ERR_FAIL_COND(variables.has(p_name));

	Variable v;
	v.default_value = p_default_value;
	v.info.type = p_default_value.get_type();
	v.info.name = p_name;
	v.info.hint = PROPERTY_HINT_NONE;
	v._export = p_export;

	variables[p_name] = v;

#ifdef TOOLS_ENABLED
	_update_placeholders();
#endif
}

bool VisualScript::has_variable(const StringName &p_name) const {
	return variables.has(p_name);
}

void VisualScript::remove_variable(const StringName &p_name) {
	ERR_FAIL_COND(!variables.has(p_name));
	variables.erase(p_name);

#ifdef TOOLS_ENABLED
	_update_placeholders();
#endif
}

void VisualScript::set_variable_default_value(const StringName &p_name, const Variant &p_value) {
	ERR_FAIL_COND(!variables.has(p_name));

	variables[p_name].default_value = p_value;

#ifdef TOOLS_ENABLED
	_update_placeholders();
#endif
}

Variant VisualScript::get_variable_default_value(const StringName &p_name) const {
	ERR_FAIL_COND_V(!variables.has(p_name), Variant());
	return variables[p_name].default_value;
}

void VisualScript::set_variable_info(const StringName &p_name, const PropertyInfo &p_info) {
	ERR_FAIL_COND(instances.size());
	ERR_FAIL_COND(!variables.has(p_name));
	variables[p_name].info = p_info;
	variables[p_name].info.name = p_name;

#ifdef TOOLS_ENABLED
	_update_placeholders();
#endif
}

PropertyInfo VisualScript::get_variable_info(const StringName &p_name) const {
	ERR_FAIL_COND_V(!variables.has(p_name), PropertyInfo());
	return variables[p_name].info;
}

void VisualScript::set_variable_export(const StringName &p_name, bool p_export) {
	ERR_FAIL_COND(!variables.has(p_name));

	variables[p_name]._export = p_export;
#ifdef TOOLS_ENABLED
	_update_placeholders();
#endif
}

bool VisualScript::get_variable_export(const StringName &p_name) const {
	ERR_FAIL_COND_V(!variables.has(p_name), false);
	return variables[p_name]._export;
}

void VisualScript::_set_variable_info(const StringName &p_name, const Dictionary &p_info) {
	PropertyInfo pinfo;
	if (p_info.has("type")) {
		pinfo.type = Variant::Type(int(p_info["type"]));
	}
	if (p_info.has("name")) {
		pinfo.name = p_info["name"];
	}
	if (p_info.has("hint")) {
		pinfo.hint = PropertyHint(int(p_info["hint"]));
	}
	if (p_info.has("hint_string")) {
		pinfo.hint_string = p_info["hint_string"];
	}
	if (p_info.has("usage")) {
		pinfo.usage = p_info["usage"];
	}

	set_variable_info(p_name, pinfo);
}

Dictionary VisualScript::_get_variable_info(const StringName &p_name) const {
	PropertyInfo pinfo = get_variable_info(p_name);
	Dictionary d;
	d["type"] = pinfo.type;
	d["name"] = pinfo.name;
	d["hint"] = pinfo.hint;
	d["hint_string"] = pinfo.hint_string;
	d["usage"] = pinfo.usage;

	return d;
}

void VisualScript::get_variable_list(List<StringName> *r_variables) const {
	variables.get_key_list(r_variables);
	// r_variables->sort_custom<StringName::AlphCompare>(); // Don't force it.
}

void VisualScript::set_instance_base_type(const StringName &p_type) {
	ERR_FAIL_COND(instances.size());
	base_type = p_type;
}

void VisualScript::rename_variable(const StringName &p_name, const StringName &p_new_name) {
	ERR_FAIL_COND(instances.size());
	ERR_FAIL_COND(!variables.has(p_name));
	if (p_new_name == p_name) {
		return;
	}

	ERR_FAIL_COND(!String(p_new_name).is_valid_identifier());

	ERR_FAIL_COND(functions.has(p_new_name));
	ERR_FAIL_COND(variables.has(p_new_name));
	ERR_FAIL_COND(custom_signals.has(p_new_name));

	variables[p_new_name] = variables[p_name];
	variables.erase(p_name);
	List<int> ids;
	get_node_list(&ids);
	for (int &E : ids) {
		Ref<VisualScriptVariableGet> nodeget = get_node(E);
		if (nodeget.is_valid()) {
			if (nodeget->get_variable() == p_name) {
				nodeget->set_variable(p_new_name);
			}
		} else {
			Ref<VisualScriptVariableSet> nodeset = get_node(E);
			if (nodeset.is_valid()) {
				if (nodeset->get_variable() == p_name) {
					nodeset->set_variable(p_new_name);
				}
			}
		}
	}
}

void VisualScript::add_custom_signal(const StringName &p_name) {
	ERR_FAIL_COND(instances.size());
	ERR_FAIL_COND(!String(p_name).is_valid_identifier());
	ERR_FAIL_COND(custom_signals.has(p_name));

	custom_signals[p_name] = Vector<Argument>();
}

bool VisualScript::has_custom_signal(const StringName &p_name) const {
	return custom_signals.has(p_name);
}

void VisualScript::custom_signal_add_argument(const StringName &p_func, Variant::Type p_type, const String &p_name, int p_index) {
	ERR_FAIL_COND(instances.size());
	ERR_FAIL_COND(!custom_signals.has(p_func));
	Argument arg;
	arg.type = p_type;
	arg.name = p_name;
	if (p_index < 0) {
		custom_signals[p_func].push_back(arg);
	} else {
		custom_signals[p_func].insert(0, arg);
	}
}

void VisualScript::custom_signal_set_argument_type(const StringName &p_func, int p_argidx, Variant::Type p_type) {
	ERR_FAIL_COND(instances.size());
	ERR_FAIL_COND(!custom_signals.has(p_func));
	ERR_FAIL_INDEX(p_argidx, custom_signals[p_func].size());
	custom_signals[p_func].write[p_argidx].type = p_type;
}

Variant::Type VisualScript::custom_signal_get_argument_type(const StringName &p_func, int p_argidx) const {
	ERR_FAIL_COND_V(!custom_signals.has(p_func), Variant::NIL);
	ERR_FAIL_INDEX_V(p_argidx, custom_signals[p_func].size(), Variant::NIL);
	return custom_signals[p_func][p_argidx].type;
}

void VisualScript::custom_signal_set_argument_name(const StringName &p_func, int p_argidx, const String &p_name) {
	ERR_FAIL_COND(instances.size());
	ERR_FAIL_COND(!custom_signals.has(p_func));
	ERR_FAIL_INDEX(p_argidx, custom_signals[p_func].size());
	custom_signals[p_func].write[p_argidx].name = p_name;
}

String VisualScript::custom_signal_get_argument_name(const StringName &p_func, int p_argidx) const {
	ERR_FAIL_COND_V(!custom_signals.has(p_func), String());
	ERR_FAIL_INDEX_V(p_argidx, custom_signals[p_func].size(), String());
	return custom_signals[p_func][p_argidx].name;
}

void VisualScript::custom_signal_remove_argument(const StringName &p_func, int p_argidx) {
	ERR_FAIL_COND(instances.size());
	ERR_FAIL_COND(!custom_signals.has(p_func));
	ERR_FAIL_INDEX(p_argidx, custom_signals[p_func].size());
	custom_signals[p_func].remove_at(p_argidx);
}

int VisualScript::custom_signal_get_argument_count(const StringName &p_func) const {
	ERR_FAIL_COND_V(!custom_signals.has(p_func), 0);
	return custom_signals[p_func].size();
}

void VisualScript::custom_signal_swap_argument(const StringName &p_func, int p_argidx, int p_with_argidx) {
	ERR_FAIL_COND(instances.size());
	ERR_FAIL_COND(!custom_signals.has(p_func));
	ERR_FAIL_INDEX(p_argidx, custom_signals[p_func].size());
	ERR_FAIL_INDEX(p_with_argidx, custom_signals[p_func].size());

	SWAP(custom_signals[p_func].write[p_argidx], custom_signals[p_func].write[p_with_argidx]);
}

void VisualScript::remove_custom_signal(const StringName &p_name) {
	ERR_FAIL_COND(instances.size());
	ERR_FAIL_COND(!custom_signals.has(p_name));
	custom_signals.erase(p_name);
}

void VisualScript::rename_custom_signal(const StringName &p_name, const StringName &p_new_name) {
	ERR_FAIL_COND(instances.size());
	ERR_FAIL_COND(!custom_signals.has(p_name));
	if (p_new_name == p_name) {
		return;
	}

	ERR_FAIL_COND(!String(p_new_name).is_valid_identifier());

	ERR_FAIL_COND(functions.has(p_new_name));
	ERR_FAIL_COND(variables.has(p_new_name));
	ERR_FAIL_COND(custom_signals.has(p_new_name));

	custom_signals[p_new_name] = custom_signals[p_name];
	custom_signals.erase(p_name);
}

void VisualScript::get_custom_signal_list(List<StringName> *r_custom_signals) const {
	for (const KeyValue<StringName, Vector<Argument>> &E : custom_signals) {
		r_custom_signals->push_back(E.key);
	}

	r_custom_signals->sort_custom<StringName::AlphCompare>();
}

int VisualScript::get_available_id() const {
	// This is infinitely increasing,
	// so one might want to implement a better solution,
	// if the there is a case for huge number of nodes to be added to visual script.
	List<int> nds;
	nodes.get_key_list(&nds);
	int max = -1;
	for (const int &E : nds) {
		if (E > max) {
			max = E;
		}
	}
	return (max + 1);
}

/////////////////////////////////

bool VisualScript::can_instantiate() const {
	return true; // ScriptServer::is_scripting_enabled();
}

StringName VisualScript::get_instance_base_type() const {
	return base_type;
}

Ref<Script> VisualScript::get_base_script() const {
	return Ref<Script>(); // No inheritance in visual script.
}

#ifdef TOOLS_ENABLED
void VisualScript::_placeholder_erased(PlaceHolderScriptInstance *p_placeholder) {
	placeholders.erase(p_placeholder);
}

void VisualScript::_update_placeholders() {
	if (placeholders.size() == 0) {
		return; // No bother if no placeholders.
	}
	List<PropertyInfo> pinfo;
	Map<StringName, Variant> values;

	List<StringName> keys;
	variables.get_key_list(&keys);

	for (const StringName &E : keys) {
		if (!variables[E]._export) {
			continue;
		}

		PropertyInfo p = variables[E].info;
		p.name = String(E);
		pinfo.push_back(p);
		values[p.name] = variables[E].default_value;
	}

	for (Set<PlaceHolderScriptInstance *>::Element *E = placeholders.front(); E; E = E->next()) {
		E->get()->update(pinfo, values);
	}
}

#endif

ScriptInstance *VisualScript::instance_create(Object *p_this) {
#ifdef TOOLS_ENABLED

	if (!ScriptServer::is_scripting_enabled() && !is_tool_script) {
		PlaceHolderScriptInstance *sins = memnew(PlaceHolderScriptInstance(VisualScriptLanguage::singleton, Ref<Script>((Script *)this), p_this));
		placeholders.insert(sins);

		List<PropertyInfo> pinfo;
		Map<StringName, Variant> values;

		List<StringName> keys;
		variables.get_key_list(&keys);

		for (const StringName &E : keys) {
			if (!variables[E]._export) {
				continue;
			}

			PropertyInfo p = variables[E].info;
			p.name = String(E);
			pinfo.push_back(p);
			values[p.name] = variables[E].default_value;
		}
		sins->update(pinfo, values);

		return sins;
	}
#endif

	VisualScriptInstance *instance = memnew(VisualScriptInstance);
	instance->create(Ref<VisualScript>(this), p_this);

	{
		MutexLock lock(VisualScriptLanguage::singleton->lock);

		instances[p_this] = instance;
	}

	return instance;
}

bool VisualScript::instance_has(const Object *p_this) const {
	return instances.has((Object *)p_this);
}

bool VisualScript::has_source_code() const {
	return false;
}

String VisualScript::get_source_code() const {
	return String();
}

void VisualScript::set_source_code(const String &p_code) {
}

Error VisualScript::reload(bool p_keep_state) {
	return OK;
}

bool VisualScript::is_tool() const {
	return is_tool_script;
}

bool VisualScript::is_valid() const {
	return true; // Always valid.
}

ScriptLanguage *VisualScript::get_language() const {
	return VisualScriptLanguage::singleton;
}

bool VisualScript::has_script_signal(const StringName &p_signal) const {
	return custom_signals.has(p_signal);
}

void VisualScript::get_script_signal_list(List<MethodInfo> *r_signals) const {
	for (const KeyValue<StringName, Vector<Argument>> &E : custom_signals) {
		MethodInfo mi;
		mi.name = E.key;
		for (int i = 0; i < E.value.size(); i++) {
			PropertyInfo arg;
			arg.type = E.value[i].type;
			arg.name = E.value[i].name;
			mi.arguments.push_back(arg);
		}

		r_signals->push_back(mi);
	}
}

bool VisualScript::get_property_default_value(const StringName &p_property, Variant &r_value) const {
	if (!variables.has(p_property)) {
		return false;
	}

	r_value = variables[p_property].default_value;
	return true;
}

void VisualScript::get_script_method_list(List<MethodInfo> *p_list) const {
	List<StringName> funcs;
	functions.get_key_list(&funcs);

	for (const StringName &E : funcs) {
		MethodInfo mi;
		mi.name = E;
		if (functions[E].func_id >= 0) {
			Ref<VisualScriptFunction> func = nodes[functions[E].func_id].node;
			if (func.is_valid()) {
				for (int i = 0; i < func->get_argument_count(); i++) {
					PropertyInfo arg;
					arg.name = func->get_argument_name(i);
					arg.type = func->get_argument_type(i);
					mi.arguments.push_back(arg);
				}

				p_list->push_back(mi);
			}
		}
	}
}

bool VisualScript::has_method(const StringName &p_method) const {
	return functions.has(p_method);
}

MethodInfo VisualScript::get_method_info(const StringName &p_method) const {
	const Function funct = functions[p_method];
	if (funct.func_id == -1) {
		return MethodInfo();
	}

	MethodInfo mi;
	mi.name = p_method;
	if (funct.func_id >= 0) {
		Ref<VisualScriptFunction> func = nodes[funct.func_id].node;
		if (func.is_valid()) {
			for (int i = 0; i < func->get_argument_count(); i++) {
				PropertyInfo arg;
				arg.name = func->get_argument_name(i);
				arg.type = func->get_argument_type(i);
				mi.arguments.push_back(arg);
			}

			if (!func->is_sequenced()) {
				mi.flags |= METHOD_FLAG_CONST;
			}
		}
	}

	return mi;
}

void VisualScript::get_script_property_list(List<PropertyInfo> *p_list) const {
	List<StringName> vars;
	get_variable_list(&vars);

	for (const StringName &E : vars) {
		//if (!variables[E]._export)
		//	continue;
		PropertyInfo pi = variables[E].info;
		pi.usage |= PROPERTY_USAGE_SCRIPT_VARIABLE;
		p_list->push_back(pi);
	}
}

int VisualScript::get_member_line(const StringName &p_member) const {
	return functions[p_member].func_id; // will be -1 if not found
}

#ifdef TOOLS_ENABLED
bool VisualScript::are_subnodes_edited() const {
	List<int> keys;
	nodes.get_key_list(&keys);
	for (const int &F : keys) {
		if (nodes[F].node->is_edited()) {
			return true;
		}
	}
	return false;
}
#endif

const Vector<Multiplayer::RPCConfig> VisualScript::get_rpc_methods() const {
	return rpc_functions;
}

void VisualScript::_set_data(const Dictionary &p_data) {
	Dictionary d = p_data;
	if (d.has("base_type")) {
		base_type = d["base_type"];
	}

	variables.clear();
	Array vars = d["variables"];
	for (int i = 0; i < vars.size(); i++) {
		Dictionary v = vars[i];
		StringName name = v["name"];
		add_variable(name);
		_set_variable_info(name, v);
		set_variable_default_value(name, v["default_value"]);
		set_variable_export(name, v.has("export") && bool(v["export"]));
	}

	custom_signals.clear();
	Array sigs = d["signals"];
	for (int i = 0; i < sigs.size(); i++) {
		Dictionary cs = sigs[i];
		add_custom_signal(cs["name"]);

		Array args = cs["arguments"];
		for (int j = 0; j < args.size(); j += 2) {
			custom_signal_add_argument(cs["name"], Variant::Type(int(args[j + 1])), args[j]);
		}
	}

	Array funcs = d["functions"];
	functions.clear();

	for (int i = 0; i < funcs.size(); i++) {
		Dictionary func = funcs[i];
		add_function(func["name"], func["function_id"]);
	}
	{
		Array nodes = d["nodes"];
		for (int i = 0; i < nodes.size(); i += 3) {
			add_node(nodes[i], nodes[i + 2], nodes[i + 1]);
		}

		Array sequence_connections = d["sequence_connections"];
		for (int j = 0; j < sequence_connections.size(); j += 3) {
			sequence_connect(sequence_connections[j + 0], sequence_connections[j + 1], sequence_connections[j + 2]);
		}

		Array data_connections = d["data_connections"];
		for (int j = 0; j < data_connections.size(); j += 4) {
			data_connect(data_connections[j + 0], data_connections[j + 1], data_connections[j + 2], data_connections[j + 3]);
		}
	}
	is_tool_script = d["is_tool_script"];
	scroll = d["scroll"];

	// Takes all the rpc methods.
	rpc_functions.clear();
	List<StringName> fns;
	functions.get_key_list(&fns);
	for (const StringName &E : fns) {
		if (functions[E].func_id >= 0 && nodes.has(functions[E].func_id)) {
			Ref<VisualScriptFunction> vsf = nodes[functions[E].func_id].node;
			if (vsf.is_valid()) {
				if (vsf->get_rpc_mode() != Multiplayer::RPC_MODE_DISABLED) {
					Multiplayer::RPCConfig nd;
					nd.name = E;
					nd.rpc_mode = vsf->get_rpc_mode();
					nd.transfer_mode = Multiplayer::TRANSFER_MODE_RELIABLE; // TODO
					if (rpc_functions.find(nd) == -1) {
						rpc_functions.push_back(nd);
					}
				}
			}
		}
	}

	// Sort so we are 100% that they are always the same.
	rpc_functions.sort_custom<Multiplayer::SortRPCConfig>();
}

Dictionary VisualScript::_get_data() const {
	Dictionary d;
	d["base_type"] = base_type;

	Array vars;
	List<StringName> var_names;
	variables.get_key_list(&var_names);
	for (const StringName &E : var_names) {
		Dictionary var = _get_variable_info(E);
		var["name"] = E; // Make sure it's the right one.
		var["default_value"] = variables[E].default_value;
		var["export"] = variables[E]._export;
		vars.push_back(var);
	}
	d["variables"] = vars;

	Array sigs;
	for (const KeyValue<StringName, Vector<Argument>> &E : custom_signals) {
		Dictionary cs;
		cs["name"] = E.key;
		Array args;
		for (int i = 0; i < E.value.size(); i++) {
			args.push_back(E.value[i].name);
			args.push_back(E.value[i].type);
		}
		cs["arguments"] = args;

		sigs.push_back(cs);
	}

	d["signals"] = sigs;

	Array funcs;
	List<StringName> func_names;
	functions.get_key_list(&func_names);
	for (const StringName &E : func_names) {
		Dictionary func;
		func["name"] = E;
		func["function_id"] = functions[E].func_id;
		funcs.push_back(func);
	}
	d["functions"] = funcs;

	Array nds;
	List<int> node_ids;
	nodes.get_key_list(&node_ids);
	for (const int &F : node_ids) {
		nds.push_back(F);
		nds.push_back(nodes[F].pos);
		nds.push_back(nodes[F].node);
	}
	d["nodes"] = nds;

	Array seqconns;
	for (const Set<SequenceConnection>::Element *F = sequence_connections.front(); F; F = F->next()) {
		seqconns.push_back(F->get().from_node);
		seqconns.push_back(F->get().from_output);
		seqconns.push_back(F->get().to_node);
	}
	d["sequence_connections"] = seqconns;

	Array dataconns;
	for (const Set<DataConnection>::Element *F = data_connections.front(); F; F = F->next()) {
		dataconns.push_back(F->get().from_node);
		dataconns.push_back(F->get().from_port);
		dataconns.push_back(F->get().to_node);
		dataconns.push_back(F->get().to_port);
	}
	d["data_connections"] = dataconns;

	d["is_tool_script"] = is_tool_script;
	d["scroll"] = scroll;

	return d;
}

void VisualScript::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_function", "name", "func_node_id"), &VisualScript::add_function);
	ClassDB::bind_method(D_METHOD("has_function", "name"), &VisualScript::has_function);
	ClassDB::bind_method(D_METHOD("remove_function", "name"), &VisualScript::remove_function);
	ClassDB::bind_method(D_METHOD("rename_function", "name", "new_name"), &VisualScript::rename_function);
	ClassDB::bind_method(D_METHOD("set_scroll", "ofs"), &VisualScript::set_scroll);
	ClassDB::bind_method(D_METHOD("get_scroll"), &VisualScript::get_scroll);

	ClassDB::bind_method(D_METHOD("add_node", "id", "node", "position"), &VisualScript::add_node, DEFVAL(Point2()));
	ClassDB::bind_method(D_METHOD("remove_node", "id"), &VisualScript::remove_node);
	ClassDB::bind_method(D_METHOD("get_function_node_id", "name"), &VisualScript::get_function_node_id);

	ClassDB::bind_method(D_METHOD("get_node", "id"), &VisualScript::get_node);
	ClassDB::bind_method(D_METHOD("has_node", "id"), &VisualScript::has_node);
	ClassDB::bind_method(D_METHOD("set_node_position", "id", "position"), &VisualScript::set_node_position);
	ClassDB::bind_method(D_METHOD("get_node_position", "id"), &VisualScript::get_node_position);

	ClassDB::bind_method(D_METHOD("sequence_connect", "from_node", "from_output", "to_node"), &VisualScript::sequence_connect);
	ClassDB::bind_method(D_METHOD("sequence_disconnect", "from_node", "from_output", "to_node"), &VisualScript::sequence_disconnect);
	ClassDB::bind_method(D_METHOD("has_sequence_connection", "from_node", "from_output", "to_node"), &VisualScript::has_sequence_connection);

	ClassDB::bind_method(D_METHOD("data_connect", "from_node", "from_port", "to_node", "to_port"), &VisualScript::data_connect);
	ClassDB::bind_method(D_METHOD("data_disconnect", "from_node", "from_port", "to_node", "to_port"), &VisualScript::data_disconnect);
	ClassDB::bind_method(D_METHOD("has_data_connection", "from_node", "from_port", "to_node", "to_port"), &VisualScript::has_data_connection);

	ClassDB::bind_method(D_METHOD("add_variable", "name", "default_value", "export"), &VisualScript::add_variable, DEFVAL(Variant()), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("has_variable", "name"), &VisualScript::has_variable);
	ClassDB::bind_method(D_METHOD("remove_variable", "name"), &VisualScript::remove_variable);
	ClassDB::bind_method(D_METHOD("set_variable_default_value", "name", "value"), &VisualScript::set_variable_default_value);
	ClassDB::bind_method(D_METHOD("get_variable_default_value", "name"), &VisualScript::get_variable_default_value);
	ClassDB::bind_method(D_METHOD("set_variable_info", "name", "value"), &VisualScript::_set_variable_info);
	ClassDB::bind_method(D_METHOD("get_variable_info", "name"), &VisualScript::_get_variable_info);
	ClassDB::bind_method(D_METHOD("set_variable_export", "name", "enable"), &VisualScript::set_variable_export);
	ClassDB::bind_method(D_METHOD("get_variable_export", "name"), &VisualScript::get_variable_export);
	ClassDB::bind_method(D_METHOD("rename_variable", "name", "new_name"), &VisualScript::rename_variable);

	ClassDB::bind_method(D_METHOD("add_custom_signal", "name"), &VisualScript::add_custom_signal);
	ClassDB::bind_method(D_METHOD("has_custom_signal", "name"), &VisualScript::has_custom_signal);
	ClassDB::bind_method(D_METHOD("custom_signal_add_argument", "name", "type", "argname", "index"), &VisualScript::custom_signal_add_argument, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("custom_signal_set_argument_type", "name", "argidx", "type"), &VisualScript::custom_signal_set_argument_type);
	ClassDB::bind_method(D_METHOD("custom_signal_get_argument_type", "name", "argidx"), &VisualScript::custom_signal_get_argument_type);
	ClassDB::bind_method(D_METHOD("custom_signal_set_argument_name", "name", "argidx", "argname"), &VisualScript::custom_signal_set_argument_name);
	ClassDB::bind_method(D_METHOD("custom_signal_get_argument_name", "name", "argidx"), &VisualScript::custom_signal_get_argument_name);
	ClassDB::bind_method(D_METHOD("custom_signal_remove_argument", "name", "argidx"), &VisualScript::custom_signal_remove_argument);
	ClassDB::bind_method(D_METHOD("custom_signal_get_argument_count", "name"), &VisualScript::custom_signal_get_argument_count);
	ClassDB::bind_method(D_METHOD("custom_signal_swap_argument", "name", "argidx", "withidx"), &VisualScript::custom_signal_swap_argument);
	ClassDB::bind_method(D_METHOD("remove_custom_signal", "name"), &VisualScript::remove_custom_signal);
	ClassDB::bind_method(D_METHOD("rename_custom_signal", "name", "new_name"), &VisualScript::rename_custom_signal);

	//ClassDB::bind_method(D_METHOD("set_variable_info","name","info"),&VScript::set_variable_info);
	//ClassDB::bind_method(D_METHOD("get_variable_info","name"),&VScript::set_variable_info);

	ClassDB::bind_method(D_METHOD("set_instance_base_type", "type"), &VisualScript::set_instance_base_type);

	ClassDB::bind_method(D_METHOD("_set_data", "data"), &VisualScript::_set_data);
	ClassDB::bind_method(D_METHOD("_get_data"), &VisualScript::_get_data);

	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL), "_set_data", "_get_data");

	ADD_SIGNAL(MethodInfo("node_ports_changed", PropertyInfo(Variant::INT, "id")));
}

VisualScript::VisualScript() {
	base_type = "Object";
	is_tool_script = false;
}

bool VisualScript::inherits_script(const Ref<Script> &p_script) const {
	return this == p_script.ptr(); // There is no inheritance in visual scripts, so this is enough.
}

Set<int> VisualScript::get_output_sequence_ports_connected(int from_node) {
	List<VisualScript::SequenceConnection> *sc = memnew(List<VisualScript::SequenceConnection>);
	get_sequence_connection_list(sc);
	Set<int> connected;
	for (List<VisualScript::SequenceConnection>::Element *E = sc->front(); E; E = E->next()) {
		if (E->get().from_node == from_node) {
			connected.insert(E->get().from_output);
		}
	}
	memdelete(sc);
	return connected;
}

VisualScript::~VisualScript() {
	// Remove all nodes and stuff that hold data refs.
	List<int> nds;
	nodes.get_key_list(&nds);
	for (const int &E : nds) {
		remove_node(E);
	}
}

////////////////////////////////////////////

bool VisualScriptInstance::set(const StringName &p_name, const Variant &p_value) {
	Map<StringName, Variant>::Element *E = variables.find(p_name);
	if (!E) {
		return false;
	}

	E->get() = p_value;

	return true;
}

bool VisualScriptInstance::get(const StringName &p_name, Variant &r_ret) const {
	const Map<StringName, Variant>::Element *E = variables.find(p_name);
	if (!E) {
		return false;
	}

	r_ret = E->get();
	return true;
}

void VisualScriptInstance::get_property_list(List<PropertyInfo> *p_properties) const {
	List<StringName> vars;
	script->variables.get_key_list(&vars);
	for (const StringName &E : vars) {
		if (!script->variables[E]._export) {
			continue;
		}
		PropertyInfo p = script->variables[E].info;
		p.name = String(E);
		p.usage |= PROPERTY_USAGE_SCRIPT_VARIABLE;
		p_properties->push_back(p);
	}
}

Variant::Type VisualScriptInstance::get_property_type(const StringName &p_name, bool *r_is_valid) const {
	if (!script->variables.has(p_name)) {
		if (r_is_valid) {
			*r_is_valid = false;
		}
		ERR_FAIL_V(Variant::NIL);
	}

	if (r_is_valid) {
		*r_is_valid = true;
	}

	return script->variables[p_name].info.type;
}

void VisualScriptInstance::get_method_list(List<MethodInfo> *p_list) const {
	List<StringName> fns;
	script->functions.get_key_list(&fns);
	for (const StringName &E : fns) {
		MethodInfo mi;
		mi.name = E;
		if (script->functions[E].func_id >= 0 && script->nodes.has(script->functions[E].func_id)) {
			Ref<VisualScriptFunction> vsf = script->nodes[script->functions[E].func_id].node;
			if (vsf.is_valid()) {
				for (int i = 0; i < vsf->get_argument_count(); i++) {
					PropertyInfo arg;
					arg.name = vsf->get_argument_name(i);
					arg.type = vsf->get_argument_type(i);

					mi.arguments.push_back(arg);
				}

				if (!vsf->is_sequenced()) { // Assumed constant if not sequenced.
					mi.flags |= METHOD_FLAG_CONST;
				}
			}
		}
		p_list->push_back(mi);
	}
}

bool VisualScriptInstance::has_method(const StringName &p_method) const {
	return script->functions.has(p_method);
}

//#define VSDEBUG(m_text) print_line(m_text)
#define VSDEBUG(m_text)

void VisualScriptInstance::_dependency_step(VisualScriptNodeInstance *node, int p_pass, int *pass_stack, const Variant **input_args, Variant **output_args, Variant *variant_stack, Callable::CallError &r_error, String &error_str, VisualScriptNodeInstance **r_error_node) {
	ERR_FAIL_COND(node->pass_idx == -1);

	if (pass_stack[node->pass_idx] == p_pass) {
		return;
	}

	pass_stack[node->pass_idx] = p_pass;

	if (!node->dependencies.is_empty()) {
		int dc = node->dependencies.size();
		VisualScriptNodeInstance **deps = node->dependencies.ptrw();

		for (int i = 0; i < dc; i++) {
			_dependency_step(deps[i], p_pass, pass_stack, input_args, output_args, variant_stack, r_error, error_str, r_error_node);
			if (r_error.error != Callable::CallError::CALL_OK) {
				return;
			}
		}
	}

	for (int i = 0; i < node->input_port_count; i++) {
		int index = node->input_ports[i] & VisualScriptNodeInstance::INPUT_MASK;

		if (node->input_ports[i] & VisualScriptNodeInstance::INPUT_DEFAULT_VALUE_BIT) {
			// Is a default value (unassigned input port).
			input_args[i] = &default_values[index];
		} else {
			// Regular temporary in stack.
			input_args[i] = &variant_stack[index];
		}
	}
	for (int i = 0; i < node->output_port_count; i++) {
		output_args[i] = &variant_stack[node->output_ports[i]];
	}

	Variant *working_mem = node->working_mem_idx >= 0 ? &variant_stack[node->working_mem_idx] : (Variant *)nullptr;

	node->step(input_args, output_args, VisualScriptNodeInstance::START_MODE_BEGIN_SEQUENCE, working_mem, r_error, error_str);
	// Ignore return.
	if (r_error.error != Callable::CallError::CALL_OK) {
		*r_error_node = node;
	}
}

Variant VisualScriptInstance::_call_internal(const StringName &p_method, void *p_stack, int p_stack_size, VisualScriptNodeInstance *p_node, int p_flow_stack_pos, int p_pass, bool p_resuming_yield, Callable::CallError &r_error) {
	Map<StringName, Function>::Element *F = functions.find(p_method);
	ERR_FAIL_COND_V(!F, Variant());
	Function *f = &F->get();

	// This call goes separate, so it can be yielded and suspended.
	Variant *variant_stack = (Variant *)p_stack;
	bool *sequence_bits = (bool *)(variant_stack + f->max_stack);
	const Variant **input_args = (const Variant **)(sequence_bits + f->node_count);
	Variant **output_args = (Variant **)(input_args + max_input_args);
	int flow_max = f->flow_stack_size;
	int *flow_stack = flow_max ? (int *)(output_args + max_output_args) : (int *)nullptr;
	int *pass_stack = flow_stack ? (int *)(flow_stack + flow_max) : (int *)nullptr;

	String error_str;

	VisualScriptNodeInstance *node = p_node;
	bool error = false;
	int current_node_id = f->node;
	Variant return_value;
	Variant *working_mem = nullptr;

	int flow_stack_pos = p_flow_stack_pos;

#ifdef DEBUG_ENABLED
	if (EngineDebugger::is_active()) {
		VisualScriptLanguage::singleton->enter_function(this, &p_method, variant_stack, &working_mem, &current_node_id);
	}
#endif

	while (true) {
		p_pass++; // Increment pass.
		current_node_id = node->get_id();

		VSDEBUG("==========AT NODE: " + itos(current_node_id) + " base: " + node->get_base_node()->get_class_name());
		VSDEBUG("AT STACK POS: " + itos(flow_stack_pos));

		// Setup working mem.
		working_mem = node->working_mem_idx >= 0 ? &variant_stack[node->working_mem_idx] : (Variant *)nullptr;

		VSDEBUG("WORKING MEM: " + itos(node->working_mem_idx));

		if (current_node_id == f->node) {
			// If function node, set up function arguments from beginning of stack.

			for (int i = 0; i < f->argument_count; i++) {
				input_args[i] = &variant_stack[i];
			}
		} else {
			// Run dependencies first.

			if (!node->dependencies.is_empty()) {
				int dc = node->dependencies.size();
				VisualScriptNodeInstance **deps = node->dependencies.ptrw();

				for (int i = 0; i < dc; i++) {
					_dependency_step(deps[i], p_pass, pass_stack, input_args, output_args, variant_stack, r_error, error_str, &node);
					if (r_error.error != Callable::CallError::CALL_OK) {
						error = true;
						current_node_id = node->id;
						break;
					}
				}
			}

			if (!error) {
				// Setup input pointers normally.
				VSDEBUG("INPUT PORTS: " + itos(node->input_port_count));

				for (int i = 0; i < node->input_port_count; i++) {
					int index = node->input_ports[i] & VisualScriptNodeInstance::INPUT_MASK;

					if (node->input_ports[i] & VisualScriptNodeInstance::INPUT_DEFAULT_VALUE_BIT) {
						// Is a default value (unassigned input port).
						input_args[i] = &default_values[index];
						VSDEBUG("\tPORT " + itos(i) + " DEFAULT VAL");
					} else {
						// Regular temporary in stack.
						input_args[i] = &variant_stack[index];
						VSDEBUG("PORT " + itos(i) + " AT STACK " + itos(index));
					}
				}
			}
		}

		if (error) {
			break;
		}

		// Setup output pointers.

		VSDEBUG("OUTPUT PORTS: " + itos(node->output_port_count));
		for (int i = 0; i < node->output_port_count; i++) {
			output_args[i] = &variant_stack[node->output_ports[i]];
			VSDEBUG("PORT " + itos(i) + " AT STACK " + itos(node->output_ports[i]));
		}

		// Do step.

		VisualScriptNodeInstance::StartMode start_mode;
		{
			if (p_resuming_yield) {
				start_mode = VisualScriptNodeInstance::START_MODE_RESUME_YIELD;
				p_resuming_yield = false; // Should resume only the first time.
			} else if (flow_stack && (flow_stack[flow_stack_pos] & VisualScriptNodeInstance::FLOW_STACK_PUSHED_BIT)) {
				// If there is a push bit, it means we are continuing a sequence.
				start_mode = VisualScriptNodeInstance::START_MODE_CONTINUE_SEQUENCE;
			} else {
				start_mode = VisualScriptNodeInstance::START_MODE_BEGIN_SEQUENCE;
			}
		}

		VSDEBUG("STEP - STARTSEQ: " + itos(start_mode));

		int ret = node->step(input_args, output_args, start_mode, working_mem, r_error, error_str);

		if (r_error.error != Callable::CallError::CALL_OK) {
			// Use error from step.
			error = true;
			break;
		}

		if (ret & VisualScriptNodeInstance::STEP_YIELD_BIT) {
			// Yielded!
			if (node->get_working_memory_size() == 0) {
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
				error_str = RTR("A node yielded without working memory, please read the docs on how to yield properly!");
				error = true;
				break;

			} else {
				Ref<VisualScriptFunctionState> state = *working_mem;
				if (!state.is_valid()) {
					r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
					error_str = RTR("Node yielded, but did not return a function state in the first working memory.");
					error = true;
					break;
				}

				// Step 1, capture all state.
				state->instance_id = get_owner_ptr()->get_instance_id();
				state->script_id = get_script()->get_instance_id();
				state->instance = this;
				state->function = p_method;
				state->working_mem_index = node->working_mem_idx;
				state->variant_stack_size = f->max_stack;
				state->node = node;
				state->flow_stack_pos = flow_stack_pos;
				state->stack.resize(p_stack_size);
				state->pass = p_pass;
				memcpy(state->stack.ptrw(), p_stack, p_stack_size);
				// Step 2, run away, return directly.
				r_error.error = Callable::CallError::CALL_OK;

#ifdef DEBUG_ENABLED
				// Will re-enter later, so exiting.
				if (EngineDebugger::is_active()) {
					VisualScriptLanguage::singleton->exit_function();
				}
#endif

				return state;
			}
		}

#ifdef DEBUG_ENABLED
		if (EngineDebugger::is_active()) {
			// line
			bool do_break = false;

			if (EngineDebugger::get_script_debugger()->get_lines_left() > 0) {
				if (EngineDebugger::get_script_debugger()->get_depth() <= 0) {
					EngineDebugger::get_script_debugger()->set_lines_left(EngineDebugger::get_script_debugger()->get_lines_left() - 1);
				}
				if (EngineDebugger::get_script_debugger()->get_lines_left() <= 0) {
					do_break = true;
				}
			}

			if (EngineDebugger::get_script_debugger()->is_breakpoint(current_node_id, source)) {
				do_break = true;
			}

			if (do_break) {
				VisualScriptLanguage::singleton->debug_break("Breakpoint", true);
			}

			EngineDebugger::get_singleton()->line_poll();
		}
#endif
		int output = ret & VisualScriptNodeInstance::STEP_MASK;

		VSDEBUG("STEP RETURN: " + itos(ret));

		if (ret & VisualScriptNodeInstance::STEP_EXIT_FUNCTION_BIT) {
			if (node->get_working_memory_size() == 0) {
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
				error_str = RTR("Return value must be assigned to first element of node working memory! Fix your node please.");
				error = true;
			} else {
				// Assign from working memory, first element.
				return_value = *working_mem;
			}

			VSDEBUG("EXITING FUNCTION - VALUE " + String(return_value));
			break; // Exit function requested, bye
		}

		VisualScriptNodeInstance *next = nullptr; // Next node.

		if ((ret == output || ret & VisualScriptNodeInstance::STEP_FLAG_PUSH_STACK_BIT) && node->sequence_output_count) {
			// If no exit bit was set, and has sequence outputs, guess next node.
			if (output >= node->sequence_output_count) {
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
				error_str = RTR("Node returned an invalid sequence output: ") + itos(output);
				error = true;
				break;
			}

			next = node->sequence_outputs[output];
			VSDEBUG("GOT NEXT NODE - " + (next ? itos(next->get_id()) : "Null"));
		}

		if (flow_stack) {
			// Update flow stack pos (may have changed).
			flow_stack[flow_stack_pos] = current_node_id;

			// Add stack push bit if requested.
			if (ret & VisualScriptNodeInstance::STEP_FLAG_PUSH_STACK_BIT) {
				flow_stack[flow_stack_pos] |= VisualScriptNodeInstance::FLOW_STACK_PUSHED_BIT;
				sequence_bits[node->sequence_index] = true; // Remember sequence bit.
				VSDEBUG("NEXT SEQ - FLAG BIT");
			} else {
				sequence_bits[node->sequence_index] = false; // Forget sequence bit.
				VSDEBUG("NEXT SEQ - NORMAL");
			}

			if (ret & VisualScriptNodeInstance::STEP_FLAG_GO_BACK_BIT) {
				// Go back request.

				if (flow_stack_pos > 0) {
					flow_stack_pos--;
					node = instances[flow_stack[flow_stack_pos] & VisualScriptNodeInstance::FLOW_STACK_MASK];
					VSDEBUG("NEXT IS GO BACK");
				} else {
					VSDEBUG("NEXT IS GO BACK, BUT NO NEXT SO EXIT");
					break; // Simply exit without value or error.
				}
			} else if (next) {
				if (sequence_bits[next->sequence_index]) {
					// What happened here is that we are entering a node that is in the middle of doing a sequence (pushed stack) from the front
					// because each node has a working memory, we can't really do a sub-sequence
					// as a result, the sequence will be restarted and the stack will roll back to find where this node
					// started the sequence.

					bool found = false;

					for (int i = flow_stack_pos; i >= 0; i--) {
						if ((flow_stack[i] & VisualScriptNodeInstance::FLOW_STACK_MASK) == next->get_id()) {
							flow_stack_pos = i; // Roll back and remove bit.
							flow_stack[i] = next->get_id();
							sequence_bits[next->sequence_index] = false;
							found = true;
						}
					}

					if (!found) {
						r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
						error_str = RTR("Found sequence bit but not the node in the stack, report bug!");
						error = true;
						break;
					}

					node = next;
					VSDEBUG("RE-ENTERED A LOOP, RETURNED STACK POS TO - " + itos(flow_stack_pos));

				} else {
					// Check for stack overflow.
					if (flow_stack_pos + 1 >= flow_max) {
						r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
						error_str = RTR("Stack overflow with stack depth: ") + itos(output);
						error = true;
						break;
					}

					node = next;

					flow_stack_pos++;
					flow_stack[flow_stack_pos] = node->get_id();

					VSDEBUG("INCREASE FLOW STACK");
				}

			} else {
				// No next node, try to go back in stack to pushed bit.

				bool found = false;

				for (int i = flow_stack_pos; i >= 0; i--) {
					VSDEBUG("FS " + itos(i) + " - " + itos(flow_stack[i]));
					if (flow_stack[i] & VisualScriptNodeInstance::FLOW_STACK_PUSHED_BIT) {
						node = instances[flow_stack[i] & VisualScriptNodeInstance::FLOW_STACK_MASK];
						flow_stack_pos = i;
						found = true;
						break;
					}
				}

				if (!found) {
					VSDEBUG("NO NEXT NODE, NO GO BACK, EXITING");
					break; // Done, couldn't find a push stack bit.
				}

				VSDEBUG("NO NEXT NODE, GO BACK TO: " + itos(flow_stack_pos));
			}
		} else {
			node = next; // Stackless mode, simply assign next node.
		}
	}

	if (error) {
		// Error
		// Function, file, line, error, explanation.
		String err_file = script->get_path();
		String err_func = p_method;
		int err_line = current_node_id; // Not a line but it works as one.

		if (node && (r_error.error != Callable::CallError::CALL_ERROR_INVALID_METHOD || error_str == String())) {
			if (error_str != String()) {
				error_str += " ";
			}

			if (r_error.error == Callable::CallError::CALL_ERROR_INVALID_ARGUMENT) {
				int errorarg = r_error.argument;
				error_str += "Cannot convert argument " + itos(errorarg + 1) + " to " + Variant::get_type_name(Variant::Type(r_error.expected)) + ".";
			} else if (r_error.error == Callable::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS) {
				error_str += "Expected " + itos(r_error.argument) + " arguments.";
			} else if (r_error.error == Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS) {
				error_str += "Expected " + itos(r_error.argument) + " arguments.";
			} else if (r_error.error == Callable::CallError::CALL_ERROR_INVALID_METHOD) {
				error_str += "Invalid Call.";
			} else if (r_error.error == Callable::CallError::CALL_ERROR_INSTANCE_IS_NULL) {
				error_str += "Base Instance is null";
			}
		}

		//if (!GDScriptLanguage::get_singleton()->debug_break(err_text,false)) {
		// debugger break did not happen

		if (!VisualScriptLanguage::singleton->debug_break(error_str, false)) {
			_err_print_error(err_func.utf8().get_data(), err_file.utf8().get_data(), err_line, error_str.utf8().get_data(), false, ERR_HANDLER_SCRIPT);
		}

		//}
	} else {
		//return_value=
	}

#ifdef DEBUG_ENABLED
	if (EngineDebugger::is_active()) {
		VisualScriptLanguage::singleton->exit_function();
	}
#endif

	// Clean up variant stack.
	for (int i = 0; i < f->max_stack; i++) {
		variant_stack[i].~Variant();
	}

	return return_value;
}

Variant VisualScriptInstance::call(const StringName &p_method, const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	r_error.error = Callable::CallError::CALL_OK; //ok by default

	Map<StringName, Function>::Element *F = functions.find(p_method);
	if (!F) {
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
		return Variant();
	}

	VSDEBUG("CALLING: " + String(p_method));

	Function *f = &F->get();

	int total_stack_size = 0;

	total_stack_size += f->max_stack * sizeof(Variant); //variants
	total_stack_size += f->node_count * sizeof(bool);
	total_stack_size += (max_input_args + max_output_args) * sizeof(Variant *); //arguments
	total_stack_size += f->flow_stack_size * sizeof(int); //flow
	total_stack_size += f->pass_stack_size * sizeof(int);

	VSDEBUG("STACK SIZE: " + itos(total_stack_size));
	VSDEBUG("STACK VARIANTS: : " + itos(f->max_stack));
	VSDEBUG("SEQBITS: : " + itos(f->node_count));
	VSDEBUG("MAX INPUT: " + itos(max_input_args));
	VSDEBUG("MAX OUTPUT: " + itos(max_output_args));
	VSDEBUG("FLOW STACK SIZE: " + itos(f->flow_stack_size));
	VSDEBUG("PASS STACK SIZE: " + itos(f->pass_stack_size));

	void *stack = alloca(total_stack_size);

	Variant *variant_stack = (Variant *)stack;
	bool *sequence_bits = (bool *)(variant_stack + f->max_stack);
	const Variant **input_args = (const Variant **)(sequence_bits + f->node_count);
	Variant **output_args = (Variant **)(input_args + max_input_args);
	int flow_max = f->flow_stack_size;
	int *flow_stack = flow_max ? (int *)(output_args + max_output_args) : (int *)nullptr;
	int *pass_stack = flow_stack ? (int *)(flow_stack + flow_max) : (int *)nullptr;

	for (int i = 0; i < f->node_count; i++) {
		sequence_bits[i] = false; // All starts as false.
	}

	memset(pass_stack, 0, f->pass_stack_size * sizeof(int));

	Map<int, VisualScriptNodeInstance *>::Element *E = instances.find(f->node);
	if (!E) {
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;

		ERR_FAIL_V_MSG(Variant(), "No VisualScriptFunction node in function.");
	}

	VisualScriptNodeInstance *node = E->get();

	if (flow_stack) {
		flow_stack[0] = node->get_id();
	}

	VSDEBUG("ARGUMENTS: " + itos(f->argument_count) = " RECEIVED: " + itos(p_argcount));

	if (p_argcount < f->argument_count) {
		r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.argument = node->get_input_port_count();

		return Variant();
	}

	if (p_argcount > f->argument_count) {
		r_error.error = Callable::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS;
		r_error.argument = node->get_input_port_count();

		return Variant();
	}

	// Allocate variant stack.
	for (int i = 0; i < f->max_stack; i++) {
		memnew_placement(&variant_stack[i], Variant);
	}

	// Allocate function arguments (must be copied for yield to work properly).
	for (int i = 0; i < p_argcount; i++) {
		variant_stack[i] = *p_args[i];
	}

	return _call_internal(p_method, stack, total_stack_size, node, 0, 0, false, r_error);
}

void VisualScriptInstance::notification(int p_notification) {
	// Do nothing as this is called using virtual.

	Variant what = p_notification;
	const Variant *whatp = &what;
	Callable::CallError ce;
	call(VisualScriptLanguage::singleton->notification, &whatp, 1, ce); // Do as call.
}

String VisualScriptInstance::to_string(bool *r_valid) {
	if (has_method(CoreStringNames::get_singleton()->_to_string)) {
		Callable::CallError ce;
		Variant ret = call(CoreStringNames::get_singleton()->_to_string, nullptr, 0, ce);
		if (ce.error == Callable::CallError::CALL_OK) {
			if (ret.get_type() != Variant::STRING) {
				if (r_valid) {
					*r_valid = false;
				}
				ERR_FAIL_V_MSG(String(), "Wrong type for " + CoreStringNames::get_singleton()->_to_string + ", must be a String.");
			}
			if (r_valid) {
				*r_valid = true;
			}
			return ret.operator String();
		}
	}
	if (r_valid) {
		*r_valid = false;
	}
	return String();
}

Ref<Script> VisualScriptInstance::get_script() const {
	return script;
}

const Vector<Multiplayer::RPCConfig> VisualScriptInstance::get_rpc_methods() const {
	return script->get_rpc_methods();
}

void VisualScriptInstance::create(const Ref<VisualScript> &p_script, Object *p_owner) {
	script = p_script;
	owner = p_owner;
	source = p_script->get_path();

	max_input_args = 0;
	max_output_args = 0;

	// Setup variables.
	{
		List<StringName> keys;
		script->variables.get_key_list(&keys);
		for (const StringName &E : keys) {
			variables[E] = script->variables[E].default_value;
		}
	}

	// Setup functions from sequence trees.
	{
		List<StringName> keys;
		script->functions.get_key_list(&keys);
		for (const StringName &E : keys) {
			const VisualScript::Function vsfn = p_script->functions[E];
			Function function;
			function.node = vsfn.func_id;
			function.max_stack = 0;
			function.flow_stack_size = 0;
			function.pass_stack_size = 0;
			function.node_count = 0;

			Map<StringName, int> local_var_indices;

			if (function.node < 0) {
				VisualScriptLanguage::singleton->debug_break_parse(get_script()->get_path(), 0, "No start node in function: " + String(E));
				ERR_CONTINUE(function.node < 0);
			}

			{
				Ref<VisualScriptFunction> func_node = script->get_node(vsfn.func_id);

				if (func_node.is_null()) {
					VisualScriptLanguage::singleton->debug_break_parse(get_script()->get_path(), 0, "No VisualScriptFunction typed start node in function: " + String(E));
				}

				ERR_CONTINUE(!func_node.is_valid());

				function.argument_count = func_node->get_argument_count();
				function.max_stack += function.argument_count;
				function.flow_stack_size = func_node->is_stack_less() ? 0 : func_node->get_stack_size();
				max_input_args = MAX(max_input_args, function.argument_count);
			}
			// Function nodes graphs.
			Set<VisualScript::SequenceConnection> seqconns;
			Set<VisualScript::DataConnection> dataconns;
			Set<int> node_ids;
			node_ids.insert(function.node);
			{
				List<int> nd_queue;
				nd_queue.push_back(function.node);
				while (!nd_queue.is_empty()) {
					for (const Set<VisualScript::SequenceConnection>::Element *F = script->sequence_connections.front(); F; F = F->next()) {
						if (nd_queue.front()->get() == F->get().from_node && !node_ids.has(F->get().to_node)) {
							nd_queue.push_back(F->get().to_node);
							node_ids.insert(F->get().to_node);
						}
						if (nd_queue.front()->get() == F->get().from_node && !seqconns.has(F->get())) {
							seqconns.insert(F->get());
						}
					}
					nd_queue.pop_front();
				}
				HashMap<int, HashMap<int, Pair<int, int>>> dc_lut; // :: to -> to_port -> (from, from_port)
				for (const Set<VisualScript::DataConnection>::Element *F = script->data_connections.front(); F; F = F->next()) {
					dc_lut[F->get().to_node][F->get().to_port] = Pair<int, int>(F->get().from_node, F->get().from_port);
				}
				for (const Set<int>::Element *F = node_ids.front(); F; F = F->next()) {
					nd_queue.push_back(F->get());
				}
				List<int> dc_keys;
				while (!nd_queue.is_empty()) {
					int ky = nd_queue.front()->get();
					dc_lut[ky].get_key_list(&dc_keys);
					for (const int &F : dc_keys) {
						VisualScript::DataConnection dc;
						dc.from_node = dc_lut[ky][F].first;
						dc.from_port = dc_lut[ky][F].second;
						dc.to_node = ky;
						dc.to_port = F;
						dataconns.insert(dc);
						nd_queue.push_back(dc.from_node);
						node_ids.insert(dc.from_node);
					}
					dc_keys.clear(); // Necessary as get_key_list does a push_back not a set.
					nd_queue.pop_front();
				}
			}

			//Multiple passes are required to set up this complex thing..
			//First create the nodes.
			for (const Set<int>::Element *F = node_ids.front(); F; F = F->next()) {
				Ref<VisualScriptNode> node = script->nodes[F->get()].node;

				VisualScriptNodeInstance *instance = node->instantiate(this); // Create instance.
				ERR_FAIL_COND(!instance);

				instance->base = node.ptr();

				instance->id = F->get();
				instance->input_port_count = node->get_input_value_port_count();
				instance->input_ports = nullptr;
				instance->output_port_count = node->get_output_value_port_count();
				instance->output_ports = nullptr;
				instance->sequence_output_count = node->get_output_sequence_port_count();
				instance->sequence_index = function.node_count++;
				instance->sequence_outputs = nullptr;
				instance->pass_idx = -1;

				if (instance->input_port_count) {
					instance->input_ports = memnew_arr(int, instance->input_port_count);
					for (int i = 0; i < instance->input_port_count; i++) {
						instance->input_ports[i] = -1; // If not assigned, will become default value.
					}
				}

				if (instance->output_port_count) {
					instance->output_ports = memnew_arr(int, instance->output_port_count);
					for (int i = 0; i < instance->output_port_count; i++) {
						instance->output_ports[i] = -1; // If not assigned, will output to trash.
					}
				}

				if (instance->sequence_output_count) {
					instance->sequence_outputs = memnew_arr(VisualScriptNodeInstance *, instance->sequence_output_count);
					for (int i = 0; i < instance->sequence_output_count; i++) {
						instance->sequence_outputs[i] = nullptr; // If it remains null, flow ends here.
					}
				}

				if (Object::cast_to<VisualScriptLocalVar>(node.ptr()) || Object::cast_to<VisualScriptLocalVarSet>(*node)) {
					// Working memory is shared only for this node, for the same variables.
					Ref<VisualScriptLocalVar> vslv = node;

					StringName var_name;

					if (Object::cast_to<VisualScriptLocalVar>(*node)) {
						var_name = String(Object::cast_to<VisualScriptLocalVar>(*node)->get_var_name()).strip_edges();
					} else {
						var_name = String(Object::cast_to<VisualScriptLocalVarSet>(*node)->get_var_name()).strip_edges();
					}

					if (!local_var_indices.has(var_name)) {
						local_var_indices[var_name] = function.max_stack;
						function.max_stack++;
					}

					instance->working_mem_idx = local_var_indices[var_name];

				} else if (instance->get_working_memory_size()) {
					instance->working_mem_idx = function.max_stack;
					function.max_stack += instance->get_working_memory_size();
				} else {
					instance->working_mem_idx = -1; //no working mem
				}

				max_input_args = MAX(max_input_args, instance->input_port_count);
				max_output_args = MAX(max_output_args, instance->output_port_count);

				instances[F->get()] = instance;
			}

			function.trash_pos = function.max_stack++; // create pos for trash

			// Second pass, do data connections.
			for (const Set<VisualScript::DataConnection>::Element *F = dataconns.front(); F; F = F->next()) {
				VisualScript::DataConnection dc = F->get();
				ERR_CONTINUE(!instances.has(dc.from_node));
				VisualScriptNodeInstance *from = instances[dc.from_node];
				ERR_CONTINUE(!instances.has(dc.to_node));
				VisualScriptNodeInstance *to = instances[dc.to_node];
				ERR_CONTINUE(dc.from_port >= from->output_port_count);
				ERR_CONTINUE(dc.to_port >= to->input_port_count);

				if (from->output_ports[dc.from_port] == -1) {
					int stack_pos = function.max_stack++;
					from->output_ports[dc.from_port] = stack_pos;
				}

				if (from->get_sequence_output_count() == 0 && to->dependencies.find(from) == -1) {
					// If the node we are reading from has no output sequence, we must call step() before reading from it.
					if (from->pass_idx == -1) {
						from->pass_idx = function.pass_stack_size;
						function.pass_stack_size++;
					}
					to->dependencies.push_back(from);
				}

				to->input_ports[dc.to_port] = from->output_ports[dc.from_port]; // Read from wherever the stack is.
			}

			// Third pass, do sequence connections.
			for (const Set<VisualScript::SequenceConnection>::Element *F = seqconns.front(); F; F = F->next()) {
				VisualScript::SequenceConnection sc = F->get();
				ERR_CONTINUE(!instances.has(sc.from_node));
				VisualScriptNodeInstance *from = instances[sc.from_node];
				ERR_CONTINUE(!instances.has(sc.to_node));
				VisualScriptNodeInstance *to = instances[sc.to_node];
				ERR_CONTINUE(sc.from_output >= from->sequence_output_count);

				from->sequence_outputs[sc.from_output] = to;
			}

			//fourth pass:
			// 1) unassigned input ports to default values
			// 2) connect unassigned output ports to trash
			for (const Set<int>::Element *F = node_ids.front(); F; F = F->next()) {
				ERR_CONTINUE(!instances.has(F->get()));

				Ref<VisualScriptNode> node = script->nodes[F->get()].node;
				VisualScriptNodeInstance *instance = instances[F->get()];

				// Connect to default values.
				for (int i = 0; i < instance->input_port_count; i++) {
					if (instance->input_ports[i] == -1) {
						// Unassigned, connect to default val.
						instance->input_ports[i] = default_values.size() | VisualScriptNodeInstance::INPUT_DEFAULT_VALUE_BIT;
						default_values.push_back(node->get_default_input_value(i));
					}
				}

				// Connect to trash.
				for (int i = 0; i < instance->output_port_count; i++) {
					if (instance->output_ports[i] == -1) {
						instance->output_ports[i] = function.trash_pos; //trash is same for all
					}
				}
			}

			functions[E] = function;
		}
	}
}

ScriptLanguage *VisualScriptInstance::get_language() {
	return VisualScriptLanguage::singleton;
}

VisualScriptInstance::VisualScriptInstance() {
}

VisualScriptInstance::~VisualScriptInstance() {
	{
		MutexLock lock(VisualScriptLanguage::singleton->lock);

		script->instances.erase(owner);
	}

	for (const KeyValue<int, VisualScriptNodeInstance *> &E : instances) {
		memdelete(E.value);
	}
}

/////////////////////////////////////////////

/////////////////////

Variant VisualScriptFunctionState::_signal_callback(const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	ERR_FAIL_COND_V(function == StringName(), Variant());

#ifdef DEBUG_ENABLED

	ERR_FAIL_COND_V_MSG(instance_id.is_valid() && !ObjectDB::get_instance(instance_id), Variant(), "Resumed after yield, but class instance is gone.");
	ERR_FAIL_COND_V_MSG(script_id.is_valid() && !ObjectDB::get_instance(script_id), Variant(), "Resumed after yield, but script is gone.");

#endif

	r_error.error = Callable::CallError::CALL_OK;

	Array args;

	if (p_argcount == 0) {
		r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.argument = 1;
		return Variant();
	} else if (p_argcount == 1) {
		//noooneee, reserved for me, me and only me.
	} else {
		for (int i = 0; i < p_argcount - 1; i++) {
			args.push_back(*p_args[i]);
		}
	}

	Ref<VisualScriptFunctionState> self = *p_args[p_argcount - 1]; //hi, I'm myself, needed this to remain alive.

	if (self.is_null()) {
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = p_argcount - 1;
		r_error.expected = Variant::OBJECT;
		return Variant();
	}

	r_error.error = Callable::CallError::CALL_OK;

	Variant *working_mem = ((Variant *)stack.ptr()) + working_mem_index;

	*working_mem = args; // Arguments go to working mem.

	Variant ret = instance->_call_internal(function, stack.ptrw(), stack.size(), node, flow_stack_pos, pass, true, r_error);
	function = StringName(); //invalidate
	return ret;
}

void VisualScriptFunctionState::connect_to_signal(Object *p_obj, const String &p_signal, Array p_binds) {
	ERR_FAIL_NULL(p_obj);
	Vector<Variant> binds;
	for (int i = 0; i < p_binds.size(); i++) {
		binds.push_back(p_binds[i]);
	}
	binds.push_back(Ref<VisualScriptFunctionState>(this)); //add myself on the back to avoid dying from unreferencing
	p_obj->connect(p_signal, Callable(this, "_signal_callback"), binds, CONNECT_ONESHOT);
}

bool VisualScriptFunctionState::is_valid() const {
	return function != StringName();
}

Variant VisualScriptFunctionState::resume(Array p_args) {
	ERR_FAIL_COND_V(function == StringName(), Variant());
#ifdef DEBUG_ENABLED

	ERR_FAIL_COND_V_MSG(instance_id.is_valid() && !ObjectDB::get_instance(instance_id), Variant(), "Resumed after yield, but class instance is gone.");
	ERR_FAIL_COND_V_MSG(script_id.is_valid() && !ObjectDB::get_instance(script_id), Variant(), "Resumed after yield, but script is gone.");

#endif

	Callable::CallError r_error;
	r_error.error = Callable::CallError::CALL_OK;

	Variant *working_mem = ((Variant *)stack.ptr()) + working_mem_index;

	*working_mem = p_args; // Arguments go to working mem.

	Variant ret = instance->_call_internal(function, stack.ptrw(), stack.size(), node, flow_stack_pos, pass, true, r_error);
	function = StringName(); //invalidate
	return ret;
}

void VisualScriptFunctionState::_bind_methods() {
	ClassDB::bind_method(D_METHOD("connect_to_signal", "obj", "signals", "args"), &VisualScriptFunctionState::connect_to_signal);
	ClassDB::bind_method(D_METHOD("resume", "args"), &VisualScriptFunctionState::resume, DEFVAL(Array()));
	ClassDB::bind_method(D_METHOD("is_valid"), &VisualScriptFunctionState::is_valid);
	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "_signal_callback", &VisualScriptFunctionState::_signal_callback, MethodInfo("_signal_callback"));
}

VisualScriptFunctionState::VisualScriptFunctionState() {
}

VisualScriptFunctionState::~VisualScriptFunctionState() {
	if (function != StringName()) {
		Variant *s = ((Variant *)stack.ptr());
		for (int i = 0; i < variant_stack_size; i++) {
			s[i].~Variant();
		}
	}
}

///////////////////////////////////////////////

String VisualScriptLanguage::get_name() const {
	return "VisualScript";
}

/* LANGUAGE FUNCTIONS */
void VisualScriptLanguage::init() {
}

String VisualScriptLanguage::get_type() const {
	return "VisualScript";
}

String VisualScriptLanguage::get_extension() const {
	return "vs";
}

Error VisualScriptLanguage::execute_file(const String &p_path) {
	return OK;
}

void VisualScriptLanguage::finish() {
}

/* EDITOR FUNCTIONS */
void VisualScriptLanguage::get_reserved_words(List<String> *p_words) const {
}

bool VisualScriptLanguage::is_control_flow_keyword(String p_keyword) const {
	return false;
}

void VisualScriptLanguage::get_comment_delimiters(List<String> *p_delimiters) const {
}

void VisualScriptLanguage::get_string_delimiters(List<String> *p_delimiters) const {
}

Ref<Script> VisualScriptLanguage::get_template(const String &p_class_name, const String &p_base_class_name) const {
	Ref<VisualScript> script;
	script.instantiate();
	script->set_instance_base_type(p_base_class_name);
	return script;
}

bool VisualScriptLanguage::is_using_templates() {
	return true;
}

void VisualScriptLanguage::make_template(const String &p_class_name, const String &p_base_class_name, Ref<Script> &p_script) {
	Ref<VisualScript> script = p_script;
	script->set_instance_base_type(p_base_class_name);
}

bool VisualScriptLanguage::validate(const String &p_script, const String &p_path, List<String> *r_functions, List<ScriptLanguage::ScriptError> *r_errors, List<ScriptLanguage::Warning> *r_warnings, Set<int> *r_safe_lines) const {
	return false;
}

Script *VisualScriptLanguage::create_script() const {
	return memnew(VisualScript);
}

bool VisualScriptLanguage::has_named_classes() const {
	return false;
}

bool VisualScriptLanguage::supports_builtin_mode() const {
	return true;
}

int VisualScriptLanguage::find_function(const String &p_function, const String &p_code) const {
	return -1;
}

String VisualScriptLanguage::make_function(const String &p_class, const String &p_name, const PackedStringArray &p_args) const {
	return String();
}

void VisualScriptLanguage::auto_indent_code(String &p_code, int p_from_line, int p_to_line) const {
}

void VisualScriptLanguage::add_global_constant(const StringName &p_variable, const Variant &p_value) {
}

/* DEBUGGER FUNCTIONS */

bool VisualScriptLanguage::debug_break_parse(const String &p_file, int p_node, const String &p_error) {
	// Break because of parse error.

	if (EngineDebugger::is_active() && Thread::get_caller_id() == Thread::get_main_id()) {
		_debug_parse_err_node = p_node;
		_debug_parse_err_file = p_file;
		_debug_error = p_error;
		EngineDebugger::get_script_debugger()->debug(this, false, true);
		return true;
	} else {
		return false;
	}
}

bool VisualScriptLanguage::debug_break(const String &p_error, bool p_allow_continue) {
	if (EngineDebugger::is_active() && Thread::get_caller_id() == Thread::get_main_id()) {
		_debug_parse_err_node = -1;
		_debug_parse_err_file = "";
		_debug_error = p_error;
		EngineDebugger::get_script_debugger()->debug(this, p_allow_continue, true);
		return true;
	} else {
		return false;
	}
}

String VisualScriptLanguage::debug_get_error() const {
	return _debug_error;
}

int VisualScriptLanguage::debug_get_stack_level_count() const {
	if (_debug_parse_err_node >= 0) {
		return 1;
	}

	return _debug_call_stack_pos;
}

int VisualScriptLanguage::debug_get_stack_level_line(int p_level) const {
	if (_debug_parse_err_node >= 0) {
		return _debug_parse_err_node;
	}

	ERR_FAIL_INDEX_V(p_level, _debug_call_stack_pos, -1);

	int l = _debug_call_stack_pos - p_level - 1;

	return *(_call_stack[l].current_id);
}

String VisualScriptLanguage::debug_get_stack_level_function(int p_level) const {
	if (_debug_parse_err_node >= 0) {
		return "";
	}

	ERR_FAIL_INDEX_V(p_level, _debug_call_stack_pos, "");
	int l = _debug_call_stack_pos - p_level - 1;
	return *_call_stack[l].function;
}

String VisualScriptLanguage::debug_get_stack_level_source(int p_level) const {
	if (_debug_parse_err_node >= 0) {
		return _debug_parse_err_file;
	}

	ERR_FAIL_INDEX_V(p_level, _debug_call_stack_pos, "");
	int l = _debug_call_stack_pos - p_level - 1;
	return _call_stack[l].instance->get_script_ptr()->get_path();
}

void VisualScriptLanguage::debug_get_stack_level_locals(int p_level, List<String> *p_locals, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {
	if (_debug_parse_err_node >= 0) {
		return;
	}

	ERR_FAIL_INDEX(p_level, _debug_call_stack_pos);

	int l = _debug_call_stack_pos - p_level - 1;
	const StringName *f = _call_stack[l].function;

	ERR_FAIL_COND(!_call_stack[l].instance->functions.has(*f));

	VisualScriptNodeInstance *node = _call_stack[l].instance->instances[*_call_stack[l].current_id];
	ERR_FAIL_COND(!node);

	p_locals->push_back("node_name");
	p_values->push_back(node->get_base_node()->get_text());

	for (int i = 0; i < node->input_port_count; i++) {
		String name = node->get_base_node()->get_input_value_port_info(i).name;
		if (name == String()) {
			name = "in_" + itos(i);
		}

		p_locals->push_back("input/" + name);

		//value is trickier

		int in_from = node->input_ports[i];
		int in_value = in_from & VisualScriptNodeInstance::INPUT_MASK;

		if (in_from & VisualScriptNodeInstance::INPUT_DEFAULT_VALUE_BIT) {
			p_values->push_back(_call_stack[l].instance->default_values[in_value]);
		} else {
			p_values->push_back(_call_stack[l].stack[in_value]);
		}
	}

	for (int i = 0; i < node->output_port_count; i++) {
		String name = node->get_base_node()->get_output_value_port_info(i).name;
		if (name == String()) {
			name = "out_" + itos(i);
		}

		p_locals->push_back("output/" + name);

		//value is trickier

		int in_from = node->output_ports[i];
		p_values->push_back(_call_stack[l].stack[in_from]);
	}

	for (int i = 0; i < node->get_working_memory_size(); i++) {
		p_locals->push_back("working_mem/mem_" + itos(i));
		p_values->push_back((*_call_stack[l].work_mem)[i]);
	}
}

void VisualScriptLanguage::debug_get_stack_level_members(int p_level, List<String> *p_members, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {
	if (_debug_parse_err_node >= 0) {
		return;
	}

	ERR_FAIL_INDEX(p_level, _debug_call_stack_pos);
	int l = _debug_call_stack_pos - p_level - 1;

	Ref<VisualScript> vs = _call_stack[l].instance->get_script();
	if (vs.is_null()) {
		return;
	}

	List<StringName> vars;
	vs->get_variable_list(&vars);
	for (const StringName &E : vars) {
		Variant v;
		if (_call_stack[l].instance->get_variable(E, &v)) {
			p_members->push_back("variables/" + E);
			p_values->push_back(v);
		}
	}
}

void VisualScriptLanguage::debug_get_globals(List<String> *p_locals, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {
	// No globals are really reachable in gdscript.
}

String VisualScriptLanguage::debug_parse_stack_level_expression(int p_level, const String &p_expression, int p_max_subitems, int p_max_depth) {
	return "";
}

void VisualScriptLanguage::reload_all_scripts() {
}

void VisualScriptLanguage::reload_tool_script(const Ref<Script> &p_script, bool p_soft_reload) {
}

/* LOADER FUNCTIONS */

void VisualScriptLanguage::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("vs");
}

void VisualScriptLanguage::get_public_functions(List<MethodInfo> *p_functions) const {
}

void VisualScriptLanguage::get_public_constants(List<Pair<String, Variant>> *p_constants) const {
}

void VisualScriptLanguage::profiling_start() {
}

void VisualScriptLanguage::profiling_stop() {
}

int VisualScriptLanguage::profiling_get_accumulated_data(ProfilingInfo *p_info_arr, int p_info_max) {
	return 0;
}

int VisualScriptLanguage::profiling_get_frame_data(ProfilingInfo *p_info_arr, int p_info_max) {
	return 0;
}

VisualScriptLanguage *VisualScriptLanguage::singleton = nullptr;

void VisualScriptLanguage::add_register_func(const String &p_name, VisualScriptNodeRegisterFunc p_func) {
	ERR_FAIL_COND(register_funcs.has(p_name));
	register_funcs[p_name] = p_func;
}

void VisualScriptLanguage::remove_register_func(const String &p_name) {
	ERR_FAIL_COND(!register_funcs.has(p_name));
	register_funcs.erase(p_name);
}

Ref<VisualScriptNode> VisualScriptLanguage::create_node_from_name(const String &p_name) {
	ERR_FAIL_COND_V(!register_funcs.has(p_name), Ref<VisualScriptNode>());

	return register_funcs[p_name](p_name);
}

void VisualScriptLanguage::get_registered_node_names(List<String> *r_names) {
	for (const KeyValue<String, VisualScriptNodeRegisterFunc> &E : register_funcs) {
		r_names->push_back(E.key);
	}
}

VisualScriptLanguage::VisualScriptLanguage() {
	singleton = this;

	int dmcs = GLOBAL_DEF("debug/settings/visual_script/max_call_stack", 1024);
	ProjectSettings::get_singleton()->set_custom_property_info("debug/settings/visual_script/max_call_stack", PropertyInfo(Variant::INT, "debug/settings/visual_script/max_call_stack", PROPERTY_HINT_RANGE, "1024,4096,1,or_greater")); //minimum is 1024

	if (EngineDebugger::is_active()) {
		// Debugging enabled!
		_debug_max_call_stack = dmcs;
		_call_stack = memnew_arr(CallLevel, _debug_max_call_stack + 1);

	} else {
		_debug_max_call_stack = 0;
		_call_stack = nullptr;
	}
}

VisualScriptLanguage::~VisualScriptLanguage() {
	if (_call_stack) {
		memdelete_arr(_call_stack);
	}
	singleton = nullptr;
}
