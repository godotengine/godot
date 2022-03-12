/*************************************************************************/
/*  visual_script_nodes.cpp                                              */
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

#include "visual_script_nodes.h"

#include "core/engine.h"
#include "core/global_constants.h"
#include "core/os/input.h"
#include "core/os/os.h"
#include "core/project_settings.h"
#include "scene/main/node.h"
#include "scene/main/scene_tree.h"

//////////////////////////////////////////
////////////////FUNCTION//////////////////
//////////////////////////////////////////

bool VisualScriptFunction::_set(const StringName &p_name, const Variant &p_value) {
	if (p_name == "argument_count") {
		int new_argc = p_value;
		int argc = arguments.size();
		if (argc == new_argc) {
			return true;
		}

		arguments.resize(new_argc);

		for (int i = argc; i < new_argc; i++) {
			arguments.write[i].name = "arg" + itos(i + 1);
			arguments.write[i].type = Variant::NIL;
		}
		ports_changed_notify();
		_change_notify();
		return true;
	}
	if (String(p_name).begins_with("argument_")) {
		int idx = String(p_name).get_slicec('_', 1).get_slicec('/', 0).to_int() - 1;
		ERR_FAIL_INDEX_V(idx, arguments.size(), false);
		String what = String(p_name).get_slice("/", 1);
		if (what == "type") {
			Variant::Type new_type = Variant::Type(int(p_value));
			arguments.write[idx].type = new_type;
			ports_changed_notify();

			return true;
		}

		if (what == "name") {
			arguments.write[idx].name = p_value;
			ports_changed_notify();
			return true;
		}
	}

	if (p_name == "stack/stackless") {
		set_stack_less(p_value);
		return true;
	}

	if (p_name == "stack/size") {
		stack_size = p_value;
		return true;
	}

	if (p_name == "rpc/mode") {
		rpc_mode = MultiplayerAPI::RPCMode(int(p_value));
		return true;
	}

	if (p_name == "sequenced/sequenced") {
		sequenced = p_value;
		ports_changed_notify();
		return true;
	}

	return false;
}

bool VisualScriptFunction::_get(const StringName &p_name, Variant &r_ret) const {
	if (p_name == "argument_count") {
		r_ret = arguments.size();
		return true;
	}
	if (String(p_name).begins_with("argument_")) {
		int idx = String(p_name).get_slicec('_', 1).get_slicec('/', 0).to_int() - 1;
		ERR_FAIL_INDEX_V(idx, arguments.size(), false);
		String what = String(p_name).get_slice("/", 1);
		if (what == "type") {
			r_ret = arguments[idx].type;
			return true;
		}
		if (what == "name") {
			r_ret = arguments[idx].name;
			return true;
		}
	}

	if (p_name == "stack/stackless") {
		r_ret = stack_less;
		return true;
	}

	if (p_name == "stack/size") {
		r_ret = stack_size;
		return true;
	}

	if (p_name == "rpc/mode") {
		r_ret = rpc_mode;
		return true;
	}

	if (p_name == "sequenced/sequenced") {
		r_ret = sequenced;
		return true;
	}

	return false;
}
void VisualScriptFunction::_get_property_list(List<PropertyInfo> *p_list) const {
	p_list->push_back(PropertyInfo(Variant::INT, "argument_count", PROPERTY_HINT_RANGE, "0,256"));
	String argt = "Any";
	for (int i = 1; i < Variant::VARIANT_MAX; i++) {
		argt += "," + Variant::get_type_name(Variant::Type(i));
	}

	for (int i = 0; i < arguments.size(); i++) {
		p_list->push_back(PropertyInfo(Variant::INT, "argument_" + itos(i + 1) + "/type", PROPERTY_HINT_ENUM, argt));
		p_list->push_back(PropertyInfo(Variant::STRING, "argument_" + itos(i + 1) + "/name"));
	}

	p_list->push_back(PropertyInfo(Variant::BOOL, "sequenced/sequenced"));

	if (!stack_less) {
		p_list->push_back(PropertyInfo(Variant::INT, "stack/size", PROPERTY_HINT_RANGE, "1,100000"));
	}
	p_list->push_back(PropertyInfo(Variant::BOOL, "stack/stackless"));
	p_list->push_back(PropertyInfo(Variant::INT, "rpc/mode", PROPERTY_HINT_ENUM, "Disabled,Remote,Master,Puppet,Remote Sync,Master Sync,Puppet Sync"));
}

int VisualScriptFunction::get_output_sequence_port_count() const {
	return 1;
}

bool VisualScriptFunction::has_input_sequence_port() const {
	return false;
}

int VisualScriptFunction::get_input_value_port_count() const {
	return 0;
}
int VisualScriptFunction::get_output_value_port_count() const {
	return arguments.size();
}

String VisualScriptFunction::get_output_sequence_port_text(int p_port) const {
	return String();
}

PropertyInfo VisualScriptFunction::get_input_value_port_info(int p_idx) const {
	ERR_FAIL_V(PropertyInfo());
}
PropertyInfo VisualScriptFunction::get_output_value_port_info(int p_idx) const {
	// Need to check it without ERR_FAIL_COND, to prevent warnings from appearing on node creation via dragging.
	if (p_idx < 0 || p_idx >= arguments.size()) {
		return PropertyInfo();
	}
	PropertyInfo out;
	out.type = arguments[p_idx].type;
	out.name = arguments[p_idx].name;
	out.hint = arguments[p_idx].hint;
	out.hint_string = arguments[p_idx].hint_string;
	return out;
}

String VisualScriptFunction::get_caption() const {
	return RTR("Function");
}

String VisualScriptFunction::get_text() const {
	return get_name(); //use name as function name I guess
}

void VisualScriptFunction::add_argument(Variant::Type p_type, const String &p_name, int p_index, const PropertyHint p_hint, const String &p_hint_string) {
	Argument arg;
	arg.name = p_name;
	arg.type = p_type;
	arg.hint = p_hint;
	arg.hint_string = p_hint_string;
	if (p_index >= 0) {
		arguments.insert(p_index, arg);
	} else {
		arguments.push_back(arg);
	}

	ports_changed_notify();
}
void VisualScriptFunction::set_argument_type(int p_argidx, Variant::Type p_type) {
	ERR_FAIL_INDEX(p_argidx, arguments.size());

	arguments.write[p_argidx].type = p_type;
	ports_changed_notify();
}
Variant::Type VisualScriptFunction::get_argument_type(int p_argidx) const {
	ERR_FAIL_INDEX_V(p_argidx, arguments.size(), Variant::NIL);
	return arguments[p_argidx].type;
}
void VisualScriptFunction::set_argument_name(int p_argidx, const String &p_name) {
	ERR_FAIL_INDEX(p_argidx, arguments.size());

	arguments.write[p_argidx].name = p_name;
	ports_changed_notify();
}
String VisualScriptFunction::get_argument_name(int p_argidx) const {
	ERR_FAIL_INDEX_V(p_argidx, arguments.size(), String());
	return arguments[p_argidx].name;
}
void VisualScriptFunction::remove_argument(int p_argidx) {
	ERR_FAIL_INDEX(p_argidx, arguments.size());

	arguments.remove(p_argidx);
	ports_changed_notify();
}

int VisualScriptFunction::get_argument_count() const {
	return arguments.size();
}

void VisualScriptFunction::set_rpc_mode(MultiplayerAPI::RPCMode p_mode) {
	rpc_mode = p_mode;
}

MultiplayerAPI::RPCMode VisualScriptFunction::get_rpc_mode() const {
	return rpc_mode;
}

class VisualScriptNodeInstanceFunction : public VisualScriptNodeInstance {
public:
	VisualScriptFunction *node;
	VisualScriptInstance *instance;

	//virtual int get_working_memory_size() const { return 0; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {
		int ac = node->get_argument_count();

		for (int i = 0; i < ac; i++) {
#ifdef DEBUG_ENABLED
			Variant::Type expected = node->get_argument_type(i);
			if (expected != Variant::NIL) {
				if (!Variant::can_convert_strict(p_inputs[i]->get_type(), expected)) {
					r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
					r_error.expected = expected;
					r_error.argument = i;
					return 0;
				}
			}
#endif

			*p_outputs[i] = *p_inputs[i];
		}

		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptFunction::instance(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceFunction *instance = memnew(VisualScriptNodeInstanceFunction);
	instance->node = this;
	instance->instance = p_instance;
	return instance;
}

VisualScriptFunction::VisualScriptFunction() {
	stack_size = 256;
	stack_less = false;
	sequenced = true;
	rpc_mode = MultiplayerAPI::RPC_MODE_DISABLED;
}

void VisualScriptFunction::set_stack_less(bool p_enable) {
	stack_less = p_enable;
	_change_notify();
}

bool VisualScriptFunction::is_stack_less() const {
	return stack_less;
}

void VisualScriptFunction::set_sequenced(bool p_enable) {
	sequenced = p_enable;
}

bool VisualScriptFunction::is_sequenced() const {
	return sequenced;
}

void VisualScriptFunction::set_stack_size(int p_size) {
	ERR_FAIL_COND(p_size < 1 || p_size > 100000);
	stack_size = p_size;
}

int VisualScriptFunction::get_stack_size() const {
	return stack_size;
}

//////////////////////////////////////////
/////////////////LISTS////////////////////
//////////////////////////////////////////

int VisualScriptLists::get_output_sequence_port_count() const {
	if (sequenced) {
		return 1;
	}
	return 0;
}
bool VisualScriptLists::has_input_sequence_port() const {
	return sequenced;
}

String VisualScriptLists::get_output_sequence_port_text(int p_port) const {
	return "";
}

int VisualScriptLists::get_input_value_port_count() const {
	return inputports.size();
}
int VisualScriptLists::get_output_value_port_count() const {
	return outputports.size();
}

PropertyInfo VisualScriptLists::get_input_value_port_info(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, inputports.size(), PropertyInfo());

	PropertyInfo pi;
	pi.name = inputports[p_idx].name;
	pi.type = inputports[p_idx].type;
	return pi;
}
PropertyInfo VisualScriptLists::get_output_value_port_info(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, outputports.size(), PropertyInfo());

	PropertyInfo pi;
	pi.name = outputports[p_idx].name;
	pi.type = outputports[p_idx].type;
	return pi;
}

bool VisualScriptLists::is_input_port_editable() const {
	return ((flags & INPUT_EDITABLE) == INPUT_EDITABLE);
}
bool VisualScriptLists::is_input_port_name_editable() const {
	return ((flags & INPUT_NAME_EDITABLE) == INPUT_NAME_EDITABLE);
}
bool VisualScriptLists::is_input_port_type_editable() const {
	return ((flags & INPUT_TYPE_EDITABLE) == INPUT_TYPE_EDITABLE);
}

bool VisualScriptLists::is_output_port_editable() const {
	return ((flags & OUTPUT_EDITABLE) == OUTPUT_EDITABLE);
}
bool VisualScriptLists::is_output_port_name_editable() const {
	return ((flags & INPUT_NAME_EDITABLE) == INPUT_NAME_EDITABLE);
}
bool VisualScriptLists::is_output_port_type_editable() const {
	return ((flags & INPUT_TYPE_EDITABLE) == INPUT_TYPE_EDITABLE);
}

// for the inspector
bool VisualScriptLists::_set(const StringName &p_name, const Variant &p_value) {
	if (p_name == "input_count" && is_input_port_editable()) {
		int new_argc = p_value;
		int argc = inputports.size();
		if (argc == new_argc) {
			return true;
		}

		inputports.resize(new_argc);

		for (int i = argc; i < new_argc; i++) {
			inputports.write[i].name = "arg" + itos(i + 1);
			inputports.write[i].type = Variant::NIL;
		}
		ports_changed_notify();
		_change_notify();
		return true;
	}
	if (String(p_name).begins_with("input_") && is_input_port_editable()) {
		int idx = String(p_name).get_slicec('_', 1).get_slicec('/', 0).to_int() - 1;
		ERR_FAIL_INDEX_V(idx, inputports.size(), false);
		String what = String(p_name).get_slice("/", 1);
		if (what == "type") {
			Variant::Type new_type = Variant::Type(int(p_value));
			inputports.write[idx].type = new_type;
			ports_changed_notify();

			return true;
		}

		if (what == "name") {
			inputports.write[idx].name = p_value;
			ports_changed_notify();
			return true;
		}
	}

	if (p_name == "output_count" && is_output_port_editable()) {
		int new_argc = p_value;
		int argc = outputports.size();
		if (argc == new_argc) {
			return true;
		}

		outputports.resize(new_argc);

		for (int i = argc; i < new_argc; i++) {
			outputports.write[i].name = "arg" + itos(i + 1);
			outputports.write[i].type = Variant::NIL;
		}
		ports_changed_notify();
		_change_notify();
		return true;
	}
	if (String(p_name).begins_with("output_") && is_output_port_editable()) {
		int idx = String(p_name).get_slicec('_', 1).get_slicec('/', 0).to_int() - 1;
		ERR_FAIL_INDEX_V(idx, outputports.size(), false);
		String what = String(p_name).get_slice("/", 1);
		if (what == "type") {
			Variant::Type new_type = Variant::Type(int(p_value));
			outputports.write[idx].type = new_type;
			ports_changed_notify();

			return true;
		}

		if (what == "name") {
			outputports.write[idx].name = p_value;
			ports_changed_notify();
			return true;
		}
	}

	if (p_name == "sequenced/sequenced") {
		sequenced = p_value;
		ports_changed_notify();
		return true;
	}

	return false;
}
bool VisualScriptLists::_get(const StringName &p_name, Variant &r_ret) const {
	if (p_name == "input_count" && is_input_port_editable()) {
		r_ret = inputports.size();
		return true;
	}
	if (String(p_name).begins_with("input_") && is_input_port_editable()) {
		int idx = String(p_name).get_slicec('_', 1).get_slicec('/', 0).to_int() - 1;
		ERR_FAIL_INDEX_V(idx, inputports.size(), false);
		String what = String(p_name).get_slice("/", 1);
		if (what == "type") {
			r_ret = inputports[idx].type;
			return true;
		}
		if (what == "name") {
			r_ret = inputports[idx].name;
			return true;
		}
	}

	if (p_name == "output_count" && is_output_port_editable()) {
		r_ret = outputports.size();
		return true;
	}
	if (String(p_name).begins_with("output_") && is_output_port_editable()) {
		int idx = String(p_name).get_slicec('_', 1).get_slicec('/', 0).to_int() - 1;
		ERR_FAIL_INDEX_V(idx, outputports.size(), false);
		String what = String(p_name).get_slice("/", 1);
		if (what == "type") {
			r_ret = outputports[idx].type;
			return true;
		}
		if (what == "name") {
			r_ret = outputports[idx].name;
			return true;
		}
	}

	if (p_name == "sequenced/sequenced") {
		r_ret = sequenced;
		return true;
	}

	return false;
}
void VisualScriptLists::_get_property_list(List<PropertyInfo> *p_list) const {
	if (is_input_port_editable()) {
		p_list->push_back(PropertyInfo(Variant::INT, "input_count", PROPERTY_HINT_RANGE, "0,256"));
		String argt = "Any";
		for (int i = 1; i < Variant::VARIANT_MAX; i++) {
			argt += "," + Variant::get_type_name(Variant::Type(i));
		}

		for (int i = 0; i < inputports.size(); i++) {
			p_list->push_back(PropertyInfo(Variant::INT, "input_" + itos(i + 1) + "/type", PROPERTY_HINT_ENUM, argt));
			p_list->push_back(PropertyInfo(Variant::STRING, "input_" + itos(i + 1) + "/name"));
		}
	}

	if (is_output_port_editable()) {
		p_list->push_back(PropertyInfo(Variant::INT, "output_count", PROPERTY_HINT_RANGE, "0,256"));
		String argt = "Any";
		for (int i = 1; i < Variant::VARIANT_MAX; i++) {
			argt += "," + Variant::get_type_name(Variant::Type(i));
		}

		for (int i = 0; i < outputports.size(); i++) {
			p_list->push_back(PropertyInfo(Variant::INT, "output_" + itos(i + 1) + "/type", PROPERTY_HINT_ENUM, argt));
			p_list->push_back(PropertyInfo(Variant::STRING, "output_" + itos(i + 1) + "/name"));
		}
	}
	p_list->push_back(PropertyInfo(Variant::BOOL, "sequenced/sequenced"));
}

// input data port interaction
void VisualScriptLists::add_input_data_port(Variant::Type p_type, const String &p_name, int p_index) {
	if (!is_input_port_editable()) {
		return;
	}

	Port inp;
	inp.name = p_name;
	inp.type = p_type;
	if (p_index >= 0) {
		inputports.insert(p_index, inp);
	} else {
		inputports.push_back(inp);
	}

	ports_changed_notify();
	_change_notify();
}
void VisualScriptLists::set_input_data_port_type(int p_idx, Variant::Type p_type) {
	if (!is_input_port_type_editable()) {
		return;
	}

	ERR_FAIL_INDEX(p_idx, inputports.size());

	inputports.write[p_idx].type = p_type;
	ports_changed_notify();
	_change_notify();
}
void VisualScriptLists::set_input_data_port_name(int p_idx, const String &p_name) {
	if (!is_input_port_name_editable()) {
		return;
	}

	ERR_FAIL_INDEX(p_idx, inputports.size());

	inputports.write[p_idx].name = p_name;
	ports_changed_notify();
	_change_notify();
}
void VisualScriptLists::remove_input_data_port(int p_argidx) {
	if (!is_input_port_editable()) {
		return;
	}

	ERR_FAIL_INDEX(p_argidx, inputports.size());

	inputports.remove(p_argidx);

	ports_changed_notify();
	_change_notify();
}

// output data port interaction
void VisualScriptLists::add_output_data_port(Variant::Type p_type, const String &p_name, int p_index) {
	if (!is_output_port_editable()) {
		return;
	}

	Port out;
	out.name = p_name;
	out.type = p_type;
	if (p_index >= 0) {
		outputports.insert(p_index, out);
	} else {
		outputports.push_back(out);
	}

	ports_changed_notify();
	_change_notify();
}
void VisualScriptLists::set_output_data_port_type(int p_idx, Variant::Type p_type) {
	if (!is_output_port_type_editable()) {
		return;
	}

	ERR_FAIL_INDEX(p_idx, outputports.size());

	outputports.write[p_idx].type = p_type;
	ports_changed_notify();
	_change_notify();
}
void VisualScriptLists::set_output_data_port_name(int p_idx, const String &p_name) {
	if (!is_output_port_name_editable()) {
		return;
	}

	ERR_FAIL_INDEX(p_idx, outputports.size());

	outputports.write[p_idx].name = p_name;
	ports_changed_notify();
	_change_notify();
}
void VisualScriptLists::remove_output_data_port(int p_argidx) {
	if (!is_output_port_editable()) {
		return;
	}

	ERR_FAIL_INDEX(p_argidx, outputports.size());

	outputports.remove(p_argidx);

	ports_changed_notify();
	_change_notify();
}

// sequences
void VisualScriptLists::set_sequenced(bool p_enable) {
	if (sequenced == p_enable) {
		return;
	}
	sequenced = p_enable;
	ports_changed_notify();
}
bool VisualScriptLists::is_sequenced() const {
	return sequenced;
}

VisualScriptLists::VisualScriptLists() {
	// initialize
	sequenced = false;
	flags = 0;
}

void VisualScriptLists::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_input_data_port", "type", "name", "index"), &VisualScriptLists::add_input_data_port);
	ClassDB::bind_method(D_METHOD("set_input_data_port_name", "index", "name"), &VisualScriptLists::set_input_data_port_name);
	ClassDB::bind_method(D_METHOD("set_input_data_port_type", "index", "type"), &VisualScriptLists::set_input_data_port_type);
	ClassDB::bind_method(D_METHOD("remove_input_data_port", "index"), &VisualScriptLists::remove_input_data_port);

	ClassDB::bind_method(D_METHOD("add_output_data_port", "type", "name", "index"), &VisualScriptLists::add_output_data_port);
	ClassDB::bind_method(D_METHOD("set_output_data_port_name", "index", "name"), &VisualScriptLists::set_output_data_port_name);
	ClassDB::bind_method(D_METHOD("set_output_data_port_type", "index", "type"), &VisualScriptLists::set_output_data_port_type);
	ClassDB::bind_method(D_METHOD("remove_output_data_port", "index"), &VisualScriptLists::remove_output_data_port);
}

//////////////////////////////////////////
//////////////COMPOSEARRAY////////////////
//////////////////////////////////////////

int VisualScriptComposeArray::get_output_sequence_port_count() const {
	if (sequenced) {
		return 1;
	}
	return 0;
}
bool VisualScriptComposeArray::has_input_sequence_port() const {
	return sequenced;
}

String VisualScriptComposeArray::get_output_sequence_port_text(int p_port) const {
	return "";
}

int VisualScriptComposeArray::get_input_value_port_count() const {
	return inputports.size();
}
int VisualScriptComposeArray::get_output_value_port_count() const {
	return 1;
}

PropertyInfo VisualScriptComposeArray::get_input_value_port_info(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, inputports.size(), PropertyInfo());

	PropertyInfo pi;
	pi.name = inputports[p_idx].name;
	pi.type = inputports[p_idx].type;
	return pi;
}
PropertyInfo VisualScriptComposeArray::get_output_value_port_info(int p_idx) const {
	PropertyInfo pi;
	pi.name = "out";
	pi.type = Variant::ARRAY;
	return pi;
}

String VisualScriptComposeArray::get_caption() const {
	return RTR("Compose Array");
}
String VisualScriptComposeArray::get_text() const {
	return "";
}

class VisualScriptComposeArrayNode : public VisualScriptNodeInstance {
public:
	int input_count = 0;
	virtual int get_working_memory_size() const { return 0; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {
		if (input_count > 0) {
			Array arr;
			for (int i = 0; i < input_count; i++) {
				arr.push_back((*p_inputs[i]));
			}
			Variant va = Variant(arr);

			*p_outputs[0] = va;
		}

		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptComposeArray::instance(VisualScriptInstance *p_instance) {
	VisualScriptComposeArrayNode *instance = memnew(VisualScriptComposeArrayNode);
	instance->input_count = inputports.size();
	return instance;
}

VisualScriptComposeArray::VisualScriptComposeArray() {
	// initialize stuff here
	sequenced = false;
	flags = INPUT_EDITABLE;
}

//////////////////////////////////////////
////////////////OPERATOR//////////////////
//////////////////////////////////////////

int VisualScriptOperator::get_output_sequence_port_count() const {
	return 0;
}

bool VisualScriptOperator::has_input_sequence_port() const {
	return false;
}

int VisualScriptOperator::get_input_value_port_count() const {
	return (op == Variant::OP_BIT_NEGATE || op == Variant::OP_NOT || op == Variant::OP_NEGATE || op == Variant::OP_POSITIVE) ? 1 : 2;
}
int VisualScriptOperator::get_output_value_port_count() const {
	return 1;
}

String VisualScriptOperator::get_output_sequence_port_text(int p_port) const {
	return String();
}

PropertyInfo VisualScriptOperator::get_input_value_port_info(int p_idx) const {
	static const Variant::Type port_types[Variant::OP_MAX][2] = {
		{ Variant::NIL, Variant::NIL }, //OP_EQUAL,
		{ Variant::NIL, Variant::NIL }, //OP_NOT_EQUAL,
		{ Variant::NIL, Variant::NIL }, //OP_LESS,
		{ Variant::NIL, Variant::NIL }, //OP_LESS_EQUAL,
		{ Variant::NIL, Variant::NIL }, //OP_GREATER,
		{ Variant::NIL, Variant::NIL }, //OP_GREATER_EQUAL,
		//mathematic
		{ Variant::NIL, Variant::NIL }, //OP_ADD,
		{ Variant::NIL, Variant::NIL }, //OP_SUBTRACT,
		{ Variant::NIL, Variant::NIL }, //OP_MULTIPLY,
		{ Variant::NIL, Variant::NIL }, //OP_DIVIDE,
		{ Variant::NIL, Variant::NIL }, //OP_NEGATE,
		{ Variant::NIL, Variant::NIL }, //OP_POSITIVE,
		{ Variant::INT, Variant::INT }, //OP_MODULE,
		{ Variant::STRING, Variant::STRING }, //OP_STRING_CONCAT,
		//bitwise
		{ Variant::INT, Variant::INT }, //OP_SHIFT_LEFT,
		{ Variant::INT, Variant::INT }, //OP_SHIFT_RIGHT,
		{ Variant::INT, Variant::INT }, //OP_BIT_AND,
		{ Variant::INT, Variant::INT }, //OP_BIT_OR,
		{ Variant::INT, Variant::INT }, //OP_BIT_XOR,
		{ Variant::INT, Variant::INT }, //OP_BIT_NEGATE,
		//logic
		{ Variant::BOOL, Variant::BOOL }, //OP_AND,
		{ Variant::BOOL, Variant::BOOL }, //OP_OR,
		{ Variant::BOOL, Variant::BOOL }, //OP_XOR,
		{ Variant::BOOL, Variant::BOOL }, //OP_NOT,
		//containment
		{ Variant::NIL, Variant::NIL } //OP_IN,
	};

	ERR_FAIL_INDEX_V(p_idx, 2, PropertyInfo());

	PropertyInfo pinfo;
	pinfo.name = p_idx == 0 ? "A" : "B";
	pinfo.type = port_types[op][p_idx];
	if (pinfo.type == Variant::NIL) {
		pinfo.type = typed;
	}
	return pinfo;
}
PropertyInfo VisualScriptOperator::get_output_value_port_info(int p_idx) const {
	static const Variant::Type port_types[Variant::OP_MAX] = {
		//comparison
		Variant::BOOL, //OP_EQUAL,
		Variant::BOOL, //OP_NOT_EQUAL,
		Variant::BOOL, //OP_LESS,
		Variant::BOOL, //OP_LESS_EQUAL,
		Variant::BOOL, //OP_GREATER,
		Variant::BOOL, //OP_GREATER_EQUAL,
		//mathematic
		Variant::NIL, //OP_ADD,
		Variant::NIL, //OP_SUBTRACT,
		Variant::NIL, //OP_MULTIPLY,
		Variant::NIL, //OP_DIVIDE,
		Variant::NIL, //OP_NEGATE,
		Variant::NIL, //OP_POSITIVE,
		Variant::INT, //OP_MODULE,
		Variant::STRING, //OP_STRING_CONCAT,
		//bitwise
		Variant::INT, //OP_SHIFT_LEFT,
		Variant::INT, //OP_SHIFT_RIGHT,
		Variant::INT, //OP_BIT_AND,
		Variant::INT, //OP_BIT_OR,
		Variant::INT, //OP_BIT_XOR,
		Variant::INT, //OP_BIT_NEGATE,
		//logic
		Variant::BOOL, //OP_AND,
		Variant::BOOL, //OP_OR,
		Variant::BOOL, //OP_XOR,
		Variant::BOOL, //OP_NOT,
		//containment
		Variant::BOOL //OP_IN,
	};

	PropertyInfo pinfo;
	pinfo.name = "";
	pinfo.type = port_types[op];
	if (pinfo.type == Variant::NIL) {
		pinfo.type = typed;
	}
	return pinfo;
}

static const char *op_names[] = {
	//comparison
	"Are Equal", //OP_EQUAL,
	"Are Not Equal", //OP_NOT_EQUAL,
	"Less Than", //OP_LESS,
	"Less Than or Equal", //OP_LESS_EQUAL,
	"Greater Than", //OP_GREATER,
	"Greater Than or Equal", //OP_GREATER_EQUAL,
	//mathematic
	"Add", //OP_ADD,
	"Subtract", //OP_SUBTRACT,
	"Multiply", //OP_MULTIPLY,
	"Divide", //OP_DIVIDE,
	"Negate", //OP_NEGATE,
	"Positive", //OP_POSITIVE,
	"Remainder", //OP_MODULE,
	"Concatenate", //OP_STRING_CONCAT,
	//bitwise
	"Bit Shift Left", //OP_SHIFT_LEFT,
	"Bit Shift Right", //OP_SHIFT_RIGHT,
	"Bit And", //OP_BIT_AND,
	"Bit Or", //OP_BIT_OR,
	"Bit Xor", //OP_BIT_XOR,
	"Bit Negate", //OP_BIT_NEGATE,
	//logic
	"And", //OP_AND,
	"Or", //OP_OR,
	"Xor", //OP_XOR,
	"Not", //OP_NOT,
	//containment
	"In", //OP_IN,
};

String VisualScriptOperator::get_caption() const {
	static const wchar_t *op_names[] = {
		//comparison
		L"A = B", //OP_EQUAL,
		L"A \u2260 B", //OP_NOT_EQUAL,
		L"A < B", //OP_LESS,
		L"A \u2264 B", //OP_LESS_EQUAL,
		L"A > B", //OP_GREATER,
		L"A \u2265 B", //OP_GREATER_EQUAL,
		//mathematic
		L"A + B", //OP_ADD,
		L"A - B", //OP_SUBTRACT,
		L"A \u00D7 B", //OP_MULTIPLY,
		L"A \u00F7 B", //OP_DIVIDE,
		L"\u00AC A", //OP_NEGATE,
		L"+ A", //OP_POSITIVE,
		L"A mod B", //OP_MODULE,
		L"A .. B", //OP_STRING_CONCAT,
		//bitwise
		L"A << B", //OP_SHIFT_LEFT,
		L"A >> B", //OP_SHIFT_RIGHT,
		L"A & B", //OP_BIT_AND,
		L"A | B", //OP_BIT_OR,
		L"A ^ B", //OP_BIT_XOR,
		L"~A", //OP_BIT_NEGATE,
		//logic
		L"A and B", //OP_AND,
		L"A or B", //OP_OR,
		L"A xor B", //OP_XOR,
		L"not A", //OP_NOT,
		L"A in B", //OP_IN,

	};
	return op_names[op];
}

void VisualScriptOperator::set_operator(Variant::Operator p_op) {
	if (op == p_op) {
		return;
	}
	op = p_op;
	ports_changed_notify();
}

Variant::Operator VisualScriptOperator::get_operator() const {
	return op;
}

void VisualScriptOperator::set_typed(Variant::Type p_op) {
	if (typed == p_op) {
		return;
	}

	typed = p_op;
	ports_changed_notify();
}

Variant::Type VisualScriptOperator::get_typed() const {
	return typed;
}

void VisualScriptOperator::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_operator", "op"), &VisualScriptOperator::set_operator);
	ClassDB::bind_method(D_METHOD("get_operator"), &VisualScriptOperator::get_operator);

	ClassDB::bind_method(D_METHOD("set_typed", "type"), &VisualScriptOperator::set_typed);
	ClassDB::bind_method(D_METHOD("get_typed"), &VisualScriptOperator::get_typed);

	String types;
	for (int i = 0; i < Variant::OP_MAX; i++) {
		if (i > 0) {
			types += ",";
		}
		types += op_names[i];
	}

	String argt = "Any";
	for (int i = 1; i < Variant::VARIANT_MAX; i++) {
		argt += "," + Variant::get_type_name(Variant::Type(i));
	}

	ADD_PROPERTY(PropertyInfo(Variant::INT, "operator", PROPERTY_HINT_ENUM, types), "set_operator", "get_operator");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "type", PROPERTY_HINT_ENUM, argt), "set_typed", "get_typed");
}

class VisualScriptNodeInstanceOperator : public VisualScriptNodeInstance {
public:
	bool unary;
	Variant::Operator op;

	//virtual int get_working_memory_size() const { return 0; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {
		bool valid;
		if (unary) {
			Variant::evaluate(op, *p_inputs[0], Variant(), *p_outputs[0], valid);
		} else {
			Variant::evaluate(op, *p_inputs[0], *p_inputs[1], *p_outputs[0], valid);
		}

		if (!valid) {
			r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
			if (p_outputs[0]->get_type() == Variant::STRING) {
				r_error_str = *p_outputs[0];
			} else {
				if (unary) {
					r_error_str = String(op_names[op]) + RTR(": Invalid argument of type: ") + Variant::get_type_name(p_inputs[0]->get_type());
				} else {
					r_error_str = String(op_names[op]) + RTR(": Invalid arguments: ") + "A: " + Variant::get_type_name(p_inputs[0]->get_type()) + "  B: " + Variant::get_type_name(p_inputs[1]->get_type());
				}
			}
		}

		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptOperator::instance(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceOperator *instance = memnew(VisualScriptNodeInstanceOperator);
	instance->unary = get_input_value_port_count() == 1;
	instance->op = op;
	return instance;
}

VisualScriptOperator::VisualScriptOperator() {
	op = Variant::OP_ADD;
	typed = Variant::NIL;
}

template <Variant::Operator OP>
static Ref<VisualScriptNode> create_op_node(const String &p_name) {
	Ref<VisualScriptOperator> node;
	node.instance();
	node->set_operator(OP);
	return node;
}

//////////////////////////////////////////
////////////////OPERATOR//////////////////
//////////////////////////////////////////

int VisualScriptSelect::get_output_sequence_port_count() const {
	return 0;
}

bool VisualScriptSelect::has_input_sequence_port() const {
	return false;
}

int VisualScriptSelect::get_input_value_port_count() const {
	return 3;
}
int VisualScriptSelect::get_output_value_port_count() const {
	return 1;
}

String VisualScriptSelect::get_output_sequence_port_text(int p_port) const {
	return String();
}

PropertyInfo VisualScriptSelect::get_input_value_port_info(int p_idx) const {
	if (p_idx == 0) {
		return PropertyInfo(Variant::BOOL, "cond");
	} else if (p_idx == 1) {
		return PropertyInfo(typed, "a");
	} else {
		return PropertyInfo(typed, "b");
	}
}
PropertyInfo VisualScriptSelect::get_output_value_port_info(int p_idx) const {
	return PropertyInfo(typed, "out");
}

String VisualScriptSelect::get_caption() const {
	return RTR("Select");
}

String VisualScriptSelect::get_text() const {
	return RTR("a if cond, else b");
}

void VisualScriptSelect::set_typed(Variant::Type p_op) {
	if (typed == p_op) {
		return;
	}

	typed = p_op;
	ports_changed_notify();
}

Variant::Type VisualScriptSelect::get_typed() const {
	return typed;
}

void VisualScriptSelect::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_typed", "type"), &VisualScriptSelect::set_typed);
	ClassDB::bind_method(D_METHOD("get_typed"), &VisualScriptSelect::get_typed);

	String argt = "Any";
	for (int i = 1; i < Variant::VARIANT_MAX; i++) {
		argt += "," + Variant::get_type_name(Variant::Type(i));
	}

	ADD_PROPERTY(PropertyInfo(Variant::INT, "type", PROPERTY_HINT_ENUM, argt), "set_typed", "get_typed");
}

class VisualScriptNodeInstanceSelect : public VisualScriptNodeInstance {
public:
	//virtual int get_working_memory_size() const { return 0; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {
		bool cond = *p_inputs[0];
		if (cond) {
			*p_outputs[0] = *p_inputs[1];
		} else {
			*p_outputs[0] = *p_inputs[2];
		}

		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptSelect::instance(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceSelect *instance = memnew(VisualScriptNodeInstanceSelect);
	return instance;
}

VisualScriptSelect::VisualScriptSelect() {
	typed = Variant::NIL;
}

//////////////////////////////////////////
////////////////VARIABLE GET//////////////////
//////////////////////////////////////////

int VisualScriptVariableGet::get_output_sequence_port_count() const {
	return 0;
}

bool VisualScriptVariableGet::has_input_sequence_port() const {
	return false;
}

int VisualScriptVariableGet::get_input_value_port_count() const {
	return 0;
}
int VisualScriptVariableGet::get_output_value_port_count() const {
	return 1;
}

String VisualScriptVariableGet::get_output_sequence_port_text(int p_port) const {
	return String();
}

PropertyInfo VisualScriptVariableGet::get_input_value_port_info(int p_idx) const {
	return PropertyInfo();
}

PropertyInfo VisualScriptVariableGet::get_output_value_port_info(int p_idx) const {
	PropertyInfo pinfo;
	pinfo.name = "value";
	if (get_visual_script().is_valid() && get_visual_script()->has_variable(variable)) {
		PropertyInfo vinfo = get_visual_script()->get_variable_info(variable);
		pinfo.type = vinfo.type;
		pinfo.hint = vinfo.hint;
		pinfo.hint_string = vinfo.hint_string;
	}
	return pinfo;
}

String VisualScriptVariableGet::get_caption() const {
	return vformat(RTR("Get %s"), variable);
}
void VisualScriptVariableGet::set_variable(StringName p_variable) {
	if (variable == p_variable) {
		return;
	}
	variable = p_variable;
	ports_changed_notify();
}

StringName VisualScriptVariableGet::get_variable() const {
	return variable;
}

void VisualScriptVariableGet::_validate_property(PropertyInfo &property) const {
	if (property.name == "var_name" && get_visual_script().is_valid()) {
		Ref<VisualScript> vs = get_visual_script();
		List<StringName> vars;
		vs->get_variable_list(&vars);

		String vhint;
		for (List<StringName>::Element *E = vars.front(); E; E = E->next()) {
			if (vhint != String()) {
				vhint += ",";
			}

			vhint += E->get().operator String();
		}

		property.hint = PROPERTY_HINT_ENUM;
		property.hint_string = vhint;
	}
}

void VisualScriptVariableGet::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_variable", "name"), &VisualScriptVariableGet::set_variable);
	ClassDB::bind_method(D_METHOD("get_variable"), &VisualScriptVariableGet::get_variable);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "var_name"), "set_variable", "get_variable");
}

class VisualScriptNodeInstanceVariableGet : public VisualScriptNodeInstance {
public:
	VisualScriptVariableGet *node;
	VisualScriptInstance *instance;
	StringName variable;

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {
		if (!instance->get_variable(variable, p_outputs[0])) {
			r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
			r_error_str = RTR("VariableGet not found in script: ") + "'" + String(variable) + "'";
			return 0;
		}
		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptVariableGet::instance(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceVariableGet *instance = memnew(VisualScriptNodeInstanceVariableGet);
	instance->node = this;
	instance->instance = p_instance;
	instance->variable = variable;
	return instance;
}
VisualScriptVariableGet::VisualScriptVariableGet() {
}

//////////////////////////////////////////
////////////////VARIABLE SET//////////////////
//////////////////////////////////////////

int VisualScriptVariableSet::get_output_sequence_port_count() const {
	return 1;
}

bool VisualScriptVariableSet::has_input_sequence_port() const {
	return true;
}

int VisualScriptVariableSet::get_input_value_port_count() const {
	return 1;
}
int VisualScriptVariableSet::get_output_value_port_count() const {
	return 0;
}

String VisualScriptVariableSet::get_output_sequence_port_text(int p_port) const {
	return String();
}

PropertyInfo VisualScriptVariableSet::get_input_value_port_info(int p_idx) const {
	PropertyInfo pinfo;
	pinfo.name = "set";
	if (get_visual_script().is_valid() && get_visual_script()->has_variable(variable)) {
		PropertyInfo vinfo = get_visual_script()->get_variable_info(variable);
		pinfo.type = vinfo.type;
		pinfo.hint = vinfo.hint;
		pinfo.hint_string = vinfo.hint_string;
	}
	return pinfo;
}

PropertyInfo VisualScriptVariableSet::get_output_value_port_info(int p_idx) const {
	return PropertyInfo();
}

String VisualScriptVariableSet::get_caption() const {
	return vformat(RTR("Set %s"), variable);
}

void VisualScriptVariableSet::set_variable(StringName p_variable) {
	if (variable == p_variable) {
		return;
	}
	variable = p_variable;
	ports_changed_notify();
}

StringName VisualScriptVariableSet::get_variable() const {
	return variable;
}

void VisualScriptVariableSet::_validate_property(PropertyInfo &property) const {
	if (property.name == "var_name" && get_visual_script().is_valid()) {
		Ref<VisualScript> vs = get_visual_script();
		List<StringName> vars;
		vs->get_variable_list(&vars);

		String vhint;
		for (List<StringName>::Element *E = vars.front(); E; E = E->next()) {
			if (vhint != String()) {
				vhint += ",";
			}

			vhint += E->get().operator String();
		}

		property.hint = PROPERTY_HINT_ENUM;
		property.hint_string = vhint;
	}
}

void VisualScriptVariableSet::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_variable", "name"), &VisualScriptVariableSet::set_variable);
	ClassDB::bind_method(D_METHOD("get_variable"), &VisualScriptVariableSet::get_variable);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "var_name"), "set_variable", "get_variable");
}

class VisualScriptNodeInstanceVariableSet : public VisualScriptNodeInstance {
public:
	VisualScriptVariableSet *node;
	VisualScriptInstance *instance;
	StringName variable;

	//virtual int get_working_memory_size() const { return 0; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {
		if (!instance->set_variable(variable, *p_inputs[0])) {
			r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
			r_error_str = RTR("VariableSet not found in script: ") + "'" + String(variable) + "'";
		}

		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptVariableSet::instance(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceVariableSet *instance = memnew(VisualScriptNodeInstanceVariableSet);
	instance->node = this;
	instance->instance = p_instance;
	instance->variable = variable;
	return instance;
}
VisualScriptVariableSet::VisualScriptVariableSet() {
}

//////////////////////////////////////////
////////////////CONSTANT//////////////////
//////////////////////////////////////////

int VisualScriptConstant::get_output_sequence_port_count() const {
	return 0;
}

bool VisualScriptConstant::has_input_sequence_port() const {
	return false;
}

int VisualScriptConstant::get_input_value_port_count() const {
	return 0;
}
int VisualScriptConstant::get_output_value_port_count() const {
	return 1;
}

String VisualScriptConstant::get_output_sequence_port_text(int p_port) const {
	return String();
}

PropertyInfo VisualScriptConstant::get_input_value_port_info(int p_idx) const {
	return PropertyInfo();
}

PropertyInfo VisualScriptConstant::get_output_value_port_info(int p_idx) const {
	PropertyInfo pinfo;
	pinfo.name = String(value);
	pinfo.type = type;
	return pinfo;
}

String VisualScriptConstant::get_caption() const {
	return RTR("Constant");
}

void VisualScriptConstant::set_constant_type(Variant::Type p_type) {
	if (type == p_type) {
		return;
	}

	type = p_type;
	Variant::CallError ce;
	value = Variant::construct(type, nullptr, 0, ce);
	ports_changed_notify();
	_change_notify();
}

Variant::Type VisualScriptConstant::get_constant_type() const {
	return type;
}

void VisualScriptConstant::set_constant_value(Variant p_value) {
	if (value == p_value) {
		return;
	}

	value = p_value;
	ports_changed_notify();
}
Variant VisualScriptConstant::get_constant_value() const {
	return value;
}

void VisualScriptConstant::_validate_property(PropertyInfo &property) const {
	if (property.name == "value") {
		property.type = type;
		if (type == Variant::NIL) {
			property.usage = 0; //do not save if nil
		}
	}
}

void VisualScriptConstant::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_constant_type", "type"), &VisualScriptConstant::set_constant_type);
	ClassDB::bind_method(D_METHOD("get_constant_type"), &VisualScriptConstant::get_constant_type);

	ClassDB::bind_method(D_METHOD("set_constant_value", "value"), &VisualScriptConstant::set_constant_value);
	ClassDB::bind_method(D_METHOD("get_constant_value"), &VisualScriptConstant::get_constant_value);

	String argt = "Null";
	for (int i = 1; i < Variant::VARIANT_MAX; i++) {
		argt += "," + Variant::get_type_name(Variant::Type(i));
	}

	ADD_PROPERTY(PropertyInfo(Variant::INT, "type", PROPERTY_HINT_ENUM, argt), "set_constant_type", "get_constant_type");
	ADD_PROPERTY(PropertyInfo(Variant::NIL, "value", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NIL_IS_VARIANT | PROPERTY_USAGE_DEFAULT), "set_constant_value", "get_constant_value");
}

class VisualScriptNodeInstanceConstant : public VisualScriptNodeInstance {
public:
	Variant constant;
	//virtual int get_working_memory_size() const { return 0; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {
		*p_outputs[0] = constant;
		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptConstant::instance(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceConstant *instance = memnew(VisualScriptNodeInstanceConstant);
	instance->constant = value;
	return instance;
}

VisualScriptConstant::VisualScriptConstant() {
	type = Variant::NIL;
}

//////////////////////////////////////////
////////////////PRELOAD//////////////////
//////////////////////////////////////////

int VisualScriptPreload::get_output_sequence_port_count() const {
	return 0;
}

bool VisualScriptPreload::has_input_sequence_port() const {
	return false;
}

int VisualScriptPreload::get_input_value_port_count() const {
	return 0;
}
int VisualScriptPreload::get_output_value_port_count() const {
	return 1;
}

String VisualScriptPreload::get_output_sequence_port_text(int p_port) const {
	return String();
}

PropertyInfo VisualScriptPreload::get_input_value_port_info(int p_idx) const {
	return PropertyInfo();
}

PropertyInfo VisualScriptPreload::get_output_value_port_info(int p_idx) const {
	PropertyInfo pinfo;
	pinfo.type = Variant::OBJECT;
	if (preload.is_valid()) {
		pinfo.hint = PROPERTY_HINT_RESOURCE_TYPE;
		pinfo.hint_string = preload->get_class();
		if (preload->get_path().is_resource_file()) {
			pinfo.name = preload->get_path();
		} else if (preload->get_name() != String()) {
			pinfo.name = preload->get_name();
		} else {
			pinfo.name = preload->get_class();
		}
	} else {
		pinfo.name = "<empty>";
	}

	return pinfo;
}

String VisualScriptPreload::get_caption() const {
	return RTR("Preload");
}

void VisualScriptPreload::set_preload(const Ref<Resource> &p_preload) {
	if (preload == p_preload) {
		return;
	}

	preload = p_preload;
	ports_changed_notify();
}

Ref<Resource> VisualScriptPreload::get_preload() const {
	return preload;
}

void VisualScriptPreload::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_preload", "resource"), &VisualScriptPreload::set_preload);
	ClassDB::bind_method(D_METHOD("get_preload"), &VisualScriptPreload::get_preload);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "resource", PROPERTY_HINT_RESOURCE_TYPE, "Resource"), "set_preload", "get_preload");
}

class VisualScriptNodeInstancePreload : public VisualScriptNodeInstance {
public:
	Ref<Resource> preload;
	//virtual int get_working_memory_size() const { return 0; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {
		*p_outputs[0] = preload;
		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptPreload::instance(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstancePreload *instance = memnew(VisualScriptNodeInstancePreload);
	instance->preload = preload;
	return instance;
}

VisualScriptPreload::VisualScriptPreload() {
}

//////////////////////////////////////////
////////////////INDEX////////////////////
//////////////////////////////////////////

int VisualScriptIndexGet::get_output_sequence_port_count() const {
	return 0;
}

bool VisualScriptIndexGet::has_input_sequence_port() const {
	return false;
}

int VisualScriptIndexGet::get_input_value_port_count() const {
	return 2;
}
int VisualScriptIndexGet::get_output_value_port_count() const {
	return 1;
}

String VisualScriptIndexGet::get_output_sequence_port_text(int p_port) const {
	return String();
}

PropertyInfo VisualScriptIndexGet::get_input_value_port_info(int p_idx) const {
	if (p_idx == 0) {
		return PropertyInfo(Variant::NIL, "base");
	} else {
		return PropertyInfo(Variant::NIL, "index");
	}
}

PropertyInfo VisualScriptIndexGet::get_output_value_port_info(int p_idx) const {
	return PropertyInfo();
}

String VisualScriptIndexGet::get_caption() const {
	return RTR("Get Index");
}

class VisualScriptNodeInstanceIndexGet : public VisualScriptNodeInstance {
public:
	//virtual int get_working_memory_size() const { return 0; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {
		bool valid;
		*p_outputs[0] = p_inputs[0]->get(*p_inputs[1], &valid);

		if (!valid) {
			r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
			r_error_str = "Invalid get: " + p_inputs[0]->get_construct_string();
		}
		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptIndexGet::instance(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceIndexGet *instance = memnew(VisualScriptNodeInstanceIndexGet);
	return instance;
}
VisualScriptIndexGet::VisualScriptIndexGet() {
}

//////////////////////////////////////////
////////////////INDEXSET//////////////////
//////////////////////////////////////////

int VisualScriptIndexSet::get_output_sequence_port_count() const {
	return 1;
}

bool VisualScriptIndexSet::has_input_sequence_port() const {
	return true;
}

int VisualScriptIndexSet::get_input_value_port_count() const {
	return 3;
}
int VisualScriptIndexSet::get_output_value_port_count() const {
	return 0;
}

String VisualScriptIndexSet::get_output_sequence_port_text(int p_port) const {
	return String();
}

PropertyInfo VisualScriptIndexSet::get_input_value_port_info(int p_idx) const {
	if (p_idx == 0) {
		return PropertyInfo(Variant::NIL, "base");
	} else if (p_idx == 1) {
		return PropertyInfo(Variant::NIL, "index");

	} else {
		return PropertyInfo(Variant::NIL, "value");
	}
}

PropertyInfo VisualScriptIndexSet::get_output_value_port_info(int p_idx) const {
	return PropertyInfo();
}

String VisualScriptIndexSet::get_caption() const {
	return RTR("Set Index");
}

class VisualScriptNodeInstanceIndexSet : public VisualScriptNodeInstance {
public:
	//virtual int get_working_memory_size() const { return 0; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {
		bool valid;
		((Variant *)p_inputs[0])->set(*p_inputs[1], *p_inputs[2], &valid);

		if (!valid) {
			r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
			r_error_str = "Invalid set: " + p_inputs[1]->get_construct_string();
		}
		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptIndexSet::instance(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceIndexSet *instance = memnew(VisualScriptNodeInstanceIndexSet);
	return instance;
}
VisualScriptIndexSet::VisualScriptIndexSet() {
}

//////////////////////////////////////////
////////////////GLOBALCONSTANT///////////
//////////////////////////////////////////

int VisualScriptGlobalConstant::get_output_sequence_port_count() const {
	return 0;
}

bool VisualScriptGlobalConstant::has_input_sequence_port() const {
	return false;
}

int VisualScriptGlobalConstant::get_input_value_port_count() const {
	return 0;
}
int VisualScriptGlobalConstant::get_output_value_port_count() const {
	return 1;
}

String VisualScriptGlobalConstant::get_output_sequence_port_text(int p_port) const {
	return String();
}

PropertyInfo VisualScriptGlobalConstant::get_input_value_port_info(int p_idx) const {
	return PropertyInfo();
}

PropertyInfo VisualScriptGlobalConstant::get_output_value_port_info(int p_idx) const {
	String name = GlobalConstants::get_global_constant_name(index);
	return PropertyInfo(Variant::INT, name);
}

String VisualScriptGlobalConstant::get_caption() const {
	return RTR("Global Constant");
}

void VisualScriptGlobalConstant::set_global_constant(int p_which) {
	index = p_which;
	_change_notify();
	ports_changed_notify();
}

int VisualScriptGlobalConstant::get_global_constant() {
	return index;
}

class VisualScriptNodeInstanceGlobalConstant : public VisualScriptNodeInstance {
public:
	int index;
	//virtual int get_working_memory_size() const { return 0; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {
		*p_outputs[0] = GlobalConstants::get_global_constant_value(index);
		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptGlobalConstant::instance(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceGlobalConstant *instance = memnew(VisualScriptNodeInstanceGlobalConstant);
	instance->index = index;
	return instance;
}

void VisualScriptGlobalConstant::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_global_constant", "index"), &VisualScriptGlobalConstant::set_global_constant);
	ClassDB::bind_method(D_METHOD("get_global_constant"), &VisualScriptGlobalConstant::get_global_constant);

	String cc;

	for (int i = 0; i < GlobalConstants::get_global_constant_count(); i++) {
		if (i > 0) {
			cc += ",";
		}
		cc += GlobalConstants::get_global_constant_name(i);
	}
	ADD_PROPERTY(PropertyInfo(Variant::INT, "constant", PROPERTY_HINT_ENUM, cc), "set_global_constant", "get_global_constant");
}

VisualScriptGlobalConstant::VisualScriptGlobalConstant() {
	index = 0;
}

//////////////////////////////////////////
////////////////CLASSCONSTANT///////////
//////////////////////////////////////////

int VisualScriptClassConstant::get_output_sequence_port_count() const {
	return 0;
}

bool VisualScriptClassConstant::has_input_sequence_port() const {
	return false;
}

int VisualScriptClassConstant::get_input_value_port_count() const {
	return 0;
}
int VisualScriptClassConstant::get_output_value_port_count() const {
	return 1;
}

String VisualScriptClassConstant::get_output_sequence_port_text(int p_port) const {
	return String();
}

PropertyInfo VisualScriptClassConstant::get_input_value_port_info(int p_idx) const {
	return PropertyInfo();
}

PropertyInfo VisualScriptClassConstant::get_output_value_port_info(int p_idx) const {
	if (name == "") {
		return PropertyInfo(Variant::INT, String(base_type));
	} else {
		return PropertyInfo(Variant::INT, String(base_type) + "." + String(name));
	}
}

String VisualScriptClassConstant::get_caption() const {
	return RTR("Class Constant");
}

void VisualScriptClassConstant::set_class_constant(const StringName &p_which) {
	name = p_which;
	_change_notify();
	ports_changed_notify();
}

StringName VisualScriptClassConstant::get_class_constant() {
	return name;
}

void VisualScriptClassConstant::set_base_type(const StringName &p_which) {
	base_type = p_which;
	List<String> constants;
	ClassDB::get_integer_constant_list(base_type, &constants, true);
	if (constants.size() > 0) {
		bool found_name = false;
		for (List<String>::Element *E = constants.front(); E; E = E->next()) {
			if (E->get() == name) {
				found_name = true;
				break;
			}
		}
		if (!found_name) {
			name = constants[0];
		}
	} else {
		name = "";
	}
	_change_notify();
	ports_changed_notify();
}

StringName VisualScriptClassConstant::get_base_type() {
	return base_type;
}

class VisualScriptNodeInstanceClassConstant : public VisualScriptNodeInstance {
public:
	int value;
	bool valid;
	//virtual int get_working_memory_size() const { return 0; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {
		if (!valid) {
			r_error_str = "Invalid constant name, pick a valid class constant.";
			r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
		}

		*p_outputs[0] = value;
		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptClassConstant::instance(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceClassConstant *instance = memnew(VisualScriptNodeInstanceClassConstant);
	instance->value = ClassDB::get_integer_constant(base_type, name, &instance->valid);
	return instance;
}

void VisualScriptClassConstant::_validate_property(PropertyInfo &property) const {
	if (property.name == "constant") {
		List<String> constants;
		ClassDB::get_integer_constant_list(base_type, &constants, true);

		property.hint_string = "";
		for (List<String>::Element *E = constants.front(); E; E = E->next()) {
			if (property.hint_string != String()) {
				property.hint_string += ",";
			}
			property.hint_string += E->get();
		}
	}
}

void VisualScriptClassConstant::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_class_constant", "name"), &VisualScriptClassConstant::set_class_constant);
	ClassDB::bind_method(D_METHOD("get_class_constant"), &VisualScriptClassConstant::get_class_constant);

	ClassDB::bind_method(D_METHOD("set_base_type", "name"), &VisualScriptClassConstant::set_base_type);
	ClassDB::bind_method(D_METHOD("get_base_type"), &VisualScriptClassConstant::get_base_type);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "base_type", PROPERTY_HINT_TYPE_STRING, "Object"), "set_base_type", "get_base_type");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "constant", PROPERTY_HINT_ENUM, ""), "set_class_constant", "get_class_constant");
}

VisualScriptClassConstant::VisualScriptClassConstant() {
	base_type = "Object";
}

//////////////////////////////////////////
////////////////BASICTYPECONSTANT///////////
//////////////////////////////////////////

int VisualScriptBasicTypeConstant::get_output_sequence_port_count() const {
	return 0;
}

bool VisualScriptBasicTypeConstant::has_input_sequence_port() const {
	return false;
}

int VisualScriptBasicTypeConstant::get_input_value_port_count() const {
	return 0;
}
int VisualScriptBasicTypeConstant::get_output_value_port_count() const {
	return 1;
}

String VisualScriptBasicTypeConstant::get_output_sequence_port_text(int p_port) const {
	return String();
}

PropertyInfo VisualScriptBasicTypeConstant::get_input_value_port_info(int p_idx) const {
	return PropertyInfo();
}

PropertyInfo VisualScriptBasicTypeConstant::get_output_value_port_info(int p_idx) const {
	return PropertyInfo(type, "value");
}

String VisualScriptBasicTypeConstant::get_caption() const {
	return RTR("Basic Constant");
}

String VisualScriptBasicTypeConstant::get_text() const {
	if (name == "") {
		return Variant::get_type_name(type);
	} else {
		return Variant::get_type_name(type) + "." + String(name);
	}
}

void VisualScriptBasicTypeConstant::set_basic_type_constant(const StringName &p_which) {
	name = p_which;
	_change_notify();
	ports_changed_notify();
}

StringName VisualScriptBasicTypeConstant::get_basic_type_constant() const {
	return name;
}

void VisualScriptBasicTypeConstant::set_basic_type(Variant::Type p_which) {
	ERR_FAIL_INDEX(p_which, Variant::VARIANT_MAX);
	type = p_which;

	List<StringName> constants;
	Variant::get_constants_for_type(type, &constants);
	if (constants.size() > 0) {
		bool found_name = false;
		for (List<StringName>::Element *E = constants.front(); E; E = E->next()) {
			if (E->get() == name) {
				found_name = true;
				break;
			}
		}
		if (!found_name) {
			name = constants[0];
		}
	} else {
		name = "";
	}
	_change_notify();
	ports_changed_notify();
}

Variant::Type VisualScriptBasicTypeConstant::get_basic_type() const {
	return type;
}

class VisualScriptNodeInstanceBasicTypeConstant : public VisualScriptNodeInstance {
public:
	Variant value;
	bool valid;
	//virtual int get_working_memory_size() const { return 0; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {
		if (!valid) {
			r_error_str = "Invalid constant name, pick a valid basic type constant.";
			r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
		}

		*p_outputs[0] = value;
		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptBasicTypeConstant::instance(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceBasicTypeConstant *instance = memnew(VisualScriptNodeInstanceBasicTypeConstant);
	instance->value = Variant::get_constant_value(type, name, &instance->valid);
	return instance;
}

void VisualScriptBasicTypeConstant::_validate_property(PropertyInfo &property) const {
	if (property.name == "constant") {
		List<StringName> constants;
		Variant::get_constants_for_type(type, &constants);

		if (constants.size() == 0) {
			property.usage = 0;
			return;
		}
		property.hint_string = "";
		for (List<StringName>::Element *E = constants.front(); E; E = E->next()) {
			if (property.hint_string != String()) {
				property.hint_string += ",";
			}
			property.hint_string += String(E->get());
		}
	}
}

void VisualScriptBasicTypeConstant::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_basic_type", "name"), &VisualScriptBasicTypeConstant::set_basic_type);
	ClassDB::bind_method(D_METHOD("get_basic_type"), &VisualScriptBasicTypeConstant::get_basic_type);

	ClassDB::bind_method(D_METHOD("set_basic_type_constant", "name"), &VisualScriptBasicTypeConstant::set_basic_type_constant);
	ClassDB::bind_method(D_METHOD("get_basic_type_constant"), &VisualScriptBasicTypeConstant::get_basic_type_constant);

	String argt = "Null";
	for (int i = 1; i < Variant::VARIANT_MAX; i++) {
		argt += "," + Variant::get_type_name(Variant::Type(i));
	}

	ADD_PROPERTY(PropertyInfo(Variant::INT, "basic_type", PROPERTY_HINT_ENUM, argt), "set_basic_type", "get_basic_type");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "constant", PROPERTY_HINT_ENUM, ""), "set_basic_type_constant", "get_basic_type_constant");
}

VisualScriptBasicTypeConstant::VisualScriptBasicTypeConstant() {
	type = Variant::NIL;
}

//////////////////////////////////////////
////////////////MATHCONSTANT///////////
//////////////////////////////////////////

const char *VisualScriptMathConstant::const_name[MATH_CONSTANT_MAX] = {
	"One",
	"PI",
	"PI/2",
	"TAU",
	"E",
	"Sqrt2",
	"INF",
	"NAN"
};

double VisualScriptMathConstant::const_value[MATH_CONSTANT_MAX] = {
	1.0,
	Math_PI,
	Math_PI * 0.5,
	Math_TAU,
	2.71828182845904523536,
	Math::sqrt(2.0),
	Math_INF,
	Math_NAN
};

int VisualScriptMathConstant::get_output_sequence_port_count() const {
	return 0;
}

bool VisualScriptMathConstant::has_input_sequence_port() const {
	return false;
}

int VisualScriptMathConstant::get_input_value_port_count() const {
	return 0;
}
int VisualScriptMathConstant::get_output_value_port_count() const {
	return 1;
}

String VisualScriptMathConstant::get_output_sequence_port_text(int p_port) const {
	return String();
}

PropertyInfo VisualScriptMathConstant::get_input_value_port_info(int p_idx) const {
	return PropertyInfo();
}

PropertyInfo VisualScriptMathConstant::get_output_value_port_info(int p_idx) const {
	return PropertyInfo(Variant::REAL, const_name[constant]);
}

String VisualScriptMathConstant::get_caption() const {
	return RTR("Math Constant");
}

void VisualScriptMathConstant::set_math_constant(MathConstant p_which) {
	constant = p_which;
	_change_notify();
	ports_changed_notify();
}

VisualScriptMathConstant::MathConstant VisualScriptMathConstant::get_math_constant() {
	return constant;
}

class VisualScriptNodeInstanceMathConstant : public VisualScriptNodeInstance {
public:
	float value;
	//virtual int get_working_memory_size() const { return 0; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {
		*p_outputs[0] = value;
		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptMathConstant::instance(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceMathConstant *instance = memnew(VisualScriptNodeInstanceMathConstant);
	instance->value = const_value[constant];
	return instance;
}

void VisualScriptMathConstant::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_math_constant", "which"), &VisualScriptMathConstant::set_math_constant);
	ClassDB::bind_method(D_METHOD("get_math_constant"), &VisualScriptMathConstant::get_math_constant);

	String cc;

	for (int i = 0; i < MATH_CONSTANT_MAX; i++) {
		if (i > 0) {
			cc += ",";
		}
		cc += const_name[i];
	}
	ADD_PROPERTY(PropertyInfo(Variant::INT, "constant", PROPERTY_HINT_ENUM, cc), "set_math_constant", "get_math_constant");

	BIND_ENUM_CONSTANT(MATH_CONSTANT_ONE);
	BIND_ENUM_CONSTANT(MATH_CONSTANT_PI);
	BIND_ENUM_CONSTANT(MATH_CONSTANT_HALF_PI);
	BIND_ENUM_CONSTANT(MATH_CONSTANT_TAU);
	BIND_ENUM_CONSTANT(MATH_CONSTANT_E);
	BIND_ENUM_CONSTANT(MATH_CONSTANT_SQRT2);
	BIND_ENUM_CONSTANT(MATH_CONSTANT_INF);
	BIND_ENUM_CONSTANT(MATH_CONSTANT_NAN);
	BIND_ENUM_CONSTANT(MATH_CONSTANT_MAX);
}

VisualScriptMathConstant::VisualScriptMathConstant() {
	constant = MATH_CONSTANT_ONE;
}

//////////////////////////////////////////
////////////////ENGINESINGLETON///////////
//////////////////////////////////////////

int VisualScriptEngineSingleton::get_output_sequence_port_count() const {
	return 0;
}

bool VisualScriptEngineSingleton::has_input_sequence_port() const {
	return false;
}

int VisualScriptEngineSingleton::get_input_value_port_count() const {
	return 0;
}
int VisualScriptEngineSingleton::get_output_value_port_count() const {
	return 1;
}

String VisualScriptEngineSingleton::get_output_sequence_port_text(int p_port) const {
	return String();
}

PropertyInfo VisualScriptEngineSingleton::get_input_value_port_info(int p_idx) const {
	return PropertyInfo();
}

PropertyInfo VisualScriptEngineSingleton::get_output_value_port_info(int p_idx) const {
	return PropertyInfo(Variant::OBJECT, singleton);
}

String VisualScriptEngineSingleton::get_caption() const {
	return RTR("Get Engine Singleton");
}

void VisualScriptEngineSingleton::set_singleton(const String &p_string) {
	singleton = p_string;

	_change_notify();
	ports_changed_notify();
}

String VisualScriptEngineSingleton::get_singleton() {
	return singleton;
}

class VisualScriptNodeInstanceEngineSingleton : public VisualScriptNodeInstance {
public:
	Object *singleton;

	//virtual int get_working_memory_size() const { return 0; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {
		*p_outputs[0] = singleton;
		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptEngineSingleton::instance(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceEngineSingleton *instance = memnew(VisualScriptNodeInstanceEngineSingleton);
	instance->singleton = Engine::get_singleton()->get_singleton_object(singleton);
	return instance;
}

VisualScriptEngineSingleton::TypeGuess VisualScriptEngineSingleton::guess_output_type(TypeGuess *p_inputs, int p_output) const {
	Object *obj = Engine::get_singleton()->get_singleton_object(singleton);
	TypeGuess tg;
	tg.type = Variant::OBJECT;
	if (obj) {
		tg.gdclass = obj->get_class();
		tg.script = obj->get_script();
	}

	return tg;
}

void VisualScriptEngineSingleton::_validate_property(PropertyInfo &property) const {
	String cc;

	List<Engine::Singleton> singletons;

	Engine::get_singleton()->get_singletons(&singletons);

	for (List<Engine::Singleton>::Element *E = singletons.front(); E; E = E->next()) {
		if (E->get().name == "VS" || E->get().name == "PS" || E->get().name == "PS2D" || E->get().name == "AS" || E->get().name == "TS" || E->get().name == "SS" || E->get().name == "SS2D") {
			continue; //skip these, too simple named
		}

		if (cc != String()) {
			cc += ",";
		}
		cc += E->get().name;
	}

	property.hint = PROPERTY_HINT_ENUM;
	property.hint_string = cc;
}

void VisualScriptEngineSingleton::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_singleton", "name"), &VisualScriptEngineSingleton::set_singleton);
	ClassDB::bind_method(D_METHOD("get_singleton"), &VisualScriptEngineSingleton::get_singleton);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "constant"), "set_singleton", "get_singleton");
}

VisualScriptEngineSingleton::VisualScriptEngineSingleton() {
	singleton = String();
}

//////////////////////////////////////////
////////////////GETNODE///////////
//////////////////////////////////////////

int VisualScriptSceneNode::get_output_sequence_port_count() const {
	return 0;
}

bool VisualScriptSceneNode::has_input_sequence_port() const {
	return false;
}

int VisualScriptSceneNode::get_input_value_port_count() const {
	return 0;
}
int VisualScriptSceneNode::get_output_value_port_count() const {
	return 1;
}

String VisualScriptSceneNode::get_output_sequence_port_text(int p_port) const {
	return String();
}

PropertyInfo VisualScriptSceneNode::get_input_value_port_info(int p_idx) const {
	return PropertyInfo();
}

PropertyInfo VisualScriptSceneNode::get_output_value_port_info(int p_idx) const {
	return PropertyInfo(Variant::OBJECT, path.simplified());
}

String VisualScriptSceneNode::get_caption() const {
	return RTR("Get Scene Node");
}

void VisualScriptSceneNode::set_node_path(const NodePath &p_path) {
	path = p_path;
	_change_notify();
	ports_changed_notify();
}

NodePath VisualScriptSceneNode::get_node_path() {
	return path;
}

class VisualScriptNodeInstanceSceneNode : public VisualScriptNodeInstance {
public:
	VisualScriptSceneNode *node;
	VisualScriptInstance *instance;
	NodePath path;

	//virtual int get_working_memory_size() const { return 0; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {
		Node *node = Object::cast_to<Node>(instance->get_owner_ptr());
		if (!node) {
			r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
			r_error_str = "Base object is not a Node!";
			return 0;
		}

		Node *another = node->get_node(path);
		if (!another) {
			r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
			r_error_str = "Path does not lead Node!";
			return 0;
		}

		*p_outputs[0] = another;

		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptSceneNode::instance(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceSceneNode *instance = memnew(VisualScriptNodeInstanceSceneNode);
	instance->node = this;
	instance->instance = p_instance;
	instance->path = path;
	return instance;
}

#ifdef TOOLS_ENABLED

static Node *_find_script_node(Node *p_edited_scene, Node *p_current_node, const Ref<Script> &script) {
	if (p_edited_scene != p_current_node && p_current_node->get_owner() != p_edited_scene) {
		return nullptr;
	}

	Ref<Script> scr = p_current_node->get_script();

	if (scr.is_valid() && scr == script) {
		return p_current_node;
	}

	for (int i = 0; i < p_current_node->get_child_count(); i++) {
		Node *n = _find_script_node(p_edited_scene, p_current_node->get_child(i), script);
		if (n) {
			return n;
		}
	}

	return nullptr;
}

#endif

VisualScriptSceneNode::TypeGuess VisualScriptSceneNode::guess_output_type(TypeGuess *p_inputs, int p_output) const {
	VisualScriptSceneNode::TypeGuess tg;
	tg.type = Variant::OBJECT;
	tg.gdclass = "Node";

#ifdef TOOLS_ENABLED
	Ref<Script> script = get_visual_script();
	if (!script.is_valid()) {
		return tg;
	}

	MainLoop *main_loop = OS::get_singleton()->get_main_loop();
	SceneTree *scene_tree = Object::cast_to<SceneTree>(main_loop);

	if (!scene_tree) {
		return tg;
	}

	Node *edited_scene = scene_tree->get_edited_scene_root();

	if (!edited_scene) {
		return tg;
	}

	Node *script_node = _find_script_node(edited_scene, edited_scene, script);

	if (!script_node) {
		return tg;
	}

	Node *another = script_node->get_node(path);

	if (another) {
		tg.gdclass = another->get_class();
		tg.script = another->get_script();
	}
#endif
	return tg;
}

void VisualScriptSceneNode::_validate_property(PropertyInfo &property) const {
#ifdef TOOLS_ENABLED
	if (property.name == "node_path") {
		Ref<Script> script = get_visual_script();
		if (!script.is_valid()) {
			return;
		}

		MainLoop *main_loop = OS::get_singleton()->get_main_loop();
		SceneTree *scene_tree = Object::cast_to<SceneTree>(main_loop);

		if (!scene_tree) {
			return;
		}

		Node *edited_scene = scene_tree->get_edited_scene_root();

		if (!edited_scene) {
			return;
		}

		Node *script_node = _find_script_node(edited_scene, edited_scene, script);

		if (!script_node) {
			return;
		}

		property.hint_string = script_node->get_path();
	}
#endif
}

void VisualScriptSceneNode::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_node_path", "path"), &VisualScriptSceneNode::set_node_path);
	ClassDB::bind_method(D_METHOD("get_node_path"), &VisualScriptSceneNode::get_node_path);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "node_path", PROPERTY_HINT_NODE_PATH_TO_EDITED_NODE), "set_node_path", "get_node_path");
}

VisualScriptSceneNode::VisualScriptSceneNode() {
	path = String(".");
}

//////////////////////////////////////////
////////////////SceneTree///////////
//////////////////////////////////////////

int VisualScriptSceneTree::get_output_sequence_port_count() const {
	return 0;
}

bool VisualScriptSceneTree::has_input_sequence_port() const {
	return false;
}

int VisualScriptSceneTree::get_input_value_port_count() const {
	return 0;
}
int VisualScriptSceneTree::get_output_value_port_count() const {
	return 1;
}

String VisualScriptSceneTree::get_output_sequence_port_text(int p_port) const {
	return String();
}

PropertyInfo VisualScriptSceneTree::get_input_value_port_info(int p_idx) const {
	return PropertyInfo();
}

PropertyInfo VisualScriptSceneTree::get_output_value_port_info(int p_idx) const {
	return PropertyInfo(Variant::OBJECT, "Scene Tree", PROPERTY_HINT_TYPE_STRING, "SceneTree");
}

String VisualScriptSceneTree::get_caption() const {
	return RTR("Get Scene Tree");
}

class VisualScriptNodeInstanceSceneTree : public VisualScriptNodeInstance {
public:
	VisualScriptSceneTree *node;
	VisualScriptInstance *instance;

	//virtual int get_working_memory_size() const { return 0; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {
		Node *node = Object::cast_to<Node>(instance->get_owner_ptr());
		if (!node) {
			r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
			r_error_str = "Base object is not a Node!";
			return 0;
		}

		SceneTree *tree = node->get_tree();
		if (!tree) {
			r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
			r_error_str = "Attempt to get SceneTree while node is not in the active tree.";
			return 0;
		}

		*p_outputs[0] = tree;

		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptSceneTree::instance(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceSceneTree *instance = memnew(VisualScriptNodeInstanceSceneTree);
	instance->node = this;
	instance->instance = p_instance;
	return instance;
}

VisualScriptSceneTree::TypeGuess VisualScriptSceneTree::guess_output_type(TypeGuess *p_inputs, int p_output) const {
	TypeGuess tg;
	tg.type = Variant::OBJECT;
	tg.gdclass = "SceneTree";
	return tg;
}

void VisualScriptSceneTree::_validate_property(PropertyInfo &property) const {
}

void VisualScriptSceneTree::_bind_methods() {
}

VisualScriptSceneTree::VisualScriptSceneTree() {
}

//////////////////////////////////////////
////////////////RESPATH///////////
//////////////////////////////////////////

int VisualScriptResourcePath::get_output_sequence_port_count() const {
	return 0;
}

bool VisualScriptResourcePath::has_input_sequence_port() const {
	return false;
}

int VisualScriptResourcePath::get_input_value_port_count() const {
	return 0;
}
int VisualScriptResourcePath::get_output_value_port_count() const {
	return 1;
}

String VisualScriptResourcePath::get_output_sequence_port_text(int p_port) const {
	return String();
}

PropertyInfo VisualScriptResourcePath::get_input_value_port_info(int p_idx) const {
	return PropertyInfo();
}

PropertyInfo VisualScriptResourcePath::get_output_value_port_info(int p_idx) const {
	return PropertyInfo(Variant::STRING, path);
}

String VisualScriptResourcePath::get_caption() const {
	return RTR("Resource Path");
}

void VisualScriptResourcePath::set_resource_path(const String &p_path) {
	path = p_path;
	_change_notify();
	ports_changed_notify();
}

String VisualScriptResourcePath::get_resource_path() {
	return path;
}

class VisualScriptNodeInstanceResourcePath : public VisualScriptNodeInstance {
public:
	String path;

	//virtual int get_working_memory_size() const { return 0; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {
		*p_outputs[0] = path;
		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptResourcePath::instance(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceResourcePath *instance = memnew(VisualScriptNodeInstanceResourcePath);
	instance->path = path;
	return instance;
}

void VisualScriptResourcePath::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_resource_path", "path"), &VisualScriptResourcePath::set_resource_path);
	ClassDB::bind_method(D_METHOD("get_resource_path"), &VisualScriptResourcePath::get_resource_path);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "path", PROPERTY_HINT_FILE), "set_resource_path", "get_resource_path");
}

VisualScriptResourcePath::VisualScriptResourcePath() {
	path = "";
}

//////////////////////////////////////////
////////////////SELF///////////
//////////////////////////////////////////

int VisualScriptSelf::get_output_sequence_port_count() const {
	return 0;
}

bool VisualScriptSelf::has_input_sequence_port() const {
	return false;
}

int VisualScriptSelf::get_input_value_port_count() const {
	return 0;
}
int VisualScriptSelf::get_output_value_port_count() const {
	return 1;
}

String VisualScriptSelf::get_output_sequence_port_text(int p_port) const {
	return String();
}

PropertyInfo VisualScriptSelf::get_input_value_port_info(int p_idx) const {
	return PropertyInfo();
}

PropertyInfo VisualScriptSelf::get_output_value_port_info(int p_idx) const {
	String type_name;
	if (get_visual_script().is_valid()) {
		type_name = get_visual_script()->get_instance_base_type();
	} else {
		type_name = "instance";
	}

	return PropertyInfo(Variant::OBJECT, type_name);
}

String VisualScriptSelf::get_caption() const {
	return RTR("Get Self");
}

class VisualScriptNodeInstanceSelf : public VisualScriptNodeInstance {
public:
	VisualScriptInstance *instance;

	//virtual int get_working_memory_size() const { return 0; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {
		*p_outputs[0] = instance->get_owner_ptr();
		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptSelf::instance(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceSelf *instance = memnew(VisualScriptNodeInstanceSelf);
	instance->instance = p_instance;
	return instance;
}

VisualScriptSelf::TypeGuess VisualScriptSelf::guess_output_type(TypeGuess *p_inputs, int p_output) const {
	VisualScriptSceneNode::TypeGuess tg;
	tg.type = Variant::OBJECT;
	tg.gdclass = "Object";

	Ref<Script> script = get_visual_script();
	if (!script.is_valid()) {
		return tg;
	}

	tg.gdclass = script->get_instance_base_type();
	tg.script = script;

	return tg;
}

void VisualScriptSelf::_bind_methods() {
}

VisualScriptSelf::VisualScriptSelf() {
}

//////////////////////////////////////////
////////////////CUSTOM (SCRIPTED)///////////
//////////////////////////////////////////

int VisualScriptCustomNode::get_output_sequence_port_count() const {
	if (get_script_instance() && get_script_instance()->has_method("_get_output_sequence_port_count")) {
		return get_script_instance()->call("_get_output_sequence_port_count");
	}
	return 0;
}

bool VisualScriptCustomNode::has_input_sequence_port() const {
	if (get_script_instance() && get_script_instance()->has_method("_has_input_sequence_port")) {
		return get_script_instance()->call("_has_input_sequence_port");
	}
	return false;
}

int VisualScriptCustomNode::get_input_value_port_count() const {
	if (get_script_instance() && get_script_instance()->has_method("_get_input_value_port_count")) {
		return get_script_instance()->call("_get_input_value_port_count");
	}
	return 0;
}
int VisualScriptCustomNode::get_output_value_port_count() const {
	if (get_script_instance() && get_script_instance()->has_method("_get_output_value_port_count")) {
		return get_script_instance()->call("_get_output_value_port_count");
	}
	return 0;
}

String VisualScriptCustomNode::get_output_sequence_port_text(int p_port) const {
	if (get_script_instance() && get_script_instance()->has_method("_get_output_sequence_port_text")) {
		return get_script_instance()->call("_get_output_sequence_port_text", p_port);
	}

	return String();
}

PropertyInfo VisualScriptCustomNode::get_input_value_port_info(int p_idx) const {
	PropertyInfo info;
	if (get_script_instance() && get_script_instance()->has_method("_get_input_value_port_type")) {
		info.type = Variant::Type(int(get_script_instance()->call("_get_input_value_port_type", p_idx)));
	}
	if (get_script_instance() && get_script_instance()->has_method("_get_input_value_port_name")) {
		info.name = get_script_instance()->call("_get_input_value_port_name", p_idx);
	}
	if (get_script_instance() && get_script_instance()->has_method("_get_input_value_port_hint")) {
		info.hint = PropertyHint(int(get_script_instance()->call("_get_input_value_port_hint", p_idx)));
	}
	if (get_script_instance() && get_script_instance()->has_method("_get_input_value_port_hint_string")) {
		info.hint_string = get_script_instance()->call("_get_input_value_port_hint_string", p_idx);
	}
	return info;
}

PropertyInfo VisualScriptCustomNode::get_output_value_port_info(int p_idx) const {
	PropertyInfo info;
	if (get_script_instance() && get_script_instance()->has_method("_get_output_value_port_type")) {
		info.type = Variant::Type(int(get_script_instance()->call("_get_output_value_port_type", p_idx)));
	}
	if (get_script_instance() && get_script_instance()->has_method("_get_output_value_port_name")) {
		info.name = get_script_instance()->call("_get_output_value_port_name", p_idx);
	}
	if (get_script_instance() && get_script_instance()->has_method("_get_output_value_port_hint")) {
		info.hint = PropertyHint(int(get_script_instance()->call("_get_output_value_port_hint", p_idx)));
	}
	if (get_script_instance() && get_script_instance()->has_method("_get_output_value_port_hint_string")) {
		info.hint_string = get_script_instance()->call("_get_output_value_port_hint_string", p_idx);
	}
	return info;
}

VisualScriptCustomNode::TypeGuess VisualScriptCustomNode::guess_output_type(TypeGuess *p_inputs, int p_output) const {
	TypeGuess tg;
	PropertyInfo pi = VisualScriptCustomNode::get_output_value_port_info(p_output);
	tg.type = pi.type;
	if (pi.type == Variant::OBJECT) {
		if (pi.hint == PROPERTY_HINT_RESOURCE_TYPE) {
			if (pi.hint_string.is_resource_file()) {
				tg.script = ResourceLoader::load(pi.hint_string);
			} else if (ClassDB::class_exists(pi.hint_string)) {
				tg.gdclass = pi.hint_string;
			}
		}
	}
	return tg;
}

String VisualScriptCustomNode::get_caption() const {
	if (get_script_instance() && get_script_instance()->has_method("_get_caption")) {
		return get_script_instance()->call("_get_caption");
	}
	return RTR("CustomNode");
}

String VisualScriptCustomNode::get_text() const {
	if (get_script_instance() && get_script_instance()->has_method("_get_text")) {
		return get_script_instance()->call("_get_text");
	}
	return "";
}

String VisualScriptCustomNode::get_category() const {
	if (get_script_instance() && get_script_instance()->has_method("_get_category")) {
		return get_script_instance()->call("_get_category");
	}
	return "Custom";
}

class VisualScriptNodeInstanceCustomNode : public VisualScriptNodeInstance {
public:
	VisualScriptInstance *instance;
	VisualScriptCustomNode *node;
	int in_count;
	int out_count;
	int work_mem_size;

	virtual int get_working_memory_size() const { return work_mem_size; }
	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {
		if (node->get_script_instance()) {
#ifdef DEBUG_ENABLED
			if (!node->get_script_instance()->has_method(VisualScriptLanguage::singleton->_step)) {
				r_error_str = RTR("Custom node has no _step() method, can't process graph.");
				r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
				return 0;
			}
#endif
			Array in_values;
			Array out_values;
			Array work_mem;

			in_values.resize(in_count);

			for (int i = 0; i < in_count; i++) {
				in_values[i] = *p_inputs[i];
			}

			out_values.resize(out_count);

			work_mem.resize(work_mem_size);

			for (int i = 0; i < work_mem_size; i++) {
				work_mem[i] = p_working_mem[i];
			}

			int ret_out;

			Variant ret = node->get_script_instance()->call(VisualScriptLanguage::singleton->_step, in_values, out_values, p_start_mode, work_mem);
			if (ret.get_type() == Variant::STRING) {
				r_error_str = ret;
				r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
				return 0;
			} else if (ret.is_num()) {
				ret_out = ret;
			} else {
				r_error_str = RTR("Invalid return value from _step(), must be integer (seq out), or string (error).");
				r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
				return 0;
			}

			for (int i = 0; i < out_count; i++) {
				if (i < out_values.size()) {
					*p_outputs[i] = out_values[i];
				}
			}

			for (int i = 0; i < work_mem_size; i++) {
				if (i < work_mem.size()) {
					p_working_mem[i] = work_mem[i];
				}
			}

			return ret_out;
		}

		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptCustomNode::instance(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceCustomNode *instance = memnew(VisualScriptNodeInstanceCustomNode);
	instance->instance = p_instance;
	instance->node = this;
	instance->in_count = get_input_value_port_count();
	instance->out_count = get_output_value_port_count();

	if (get_script_instance() && get_script_instance()->has_method("_get_working_memory_size")) {
		instance->work_mem_size = get_script_instance()->call("_get_working_memory_size");
	} else {
		instance->work_mem_size = 0;
	}

	return instance;
}

void VisualScriptCustomNode::_script_changed() {
	call_deferred("ports_changed_notify");
}

void VisualScriptCustomNode::_bind_methods() {
	BIND_VMETHOD(MethodInfo(Variant::INT, "_get_output_sequence_port_count"));
	BIND_VMETHOD(MethodInfo(Variant::BOOL, "_has_input_sequence_port"));

	BIND_VMETHOD(MethodInfo(Variant::STRING, "_get_output_sequence_port_text", PropertyInfo(Variant::INT, "idx")));
	BIND_VMETHOD(MethodInfo(Variant::INT, "_get_input_value_port_count"));
	BIND_VMETHOD(MethodInfo(Variant::INT, "_get_output_value_port_count"));

	BIND_VMETHOD(MethodInfo(Variant::INT, "_get_input_value_port_type", PropertyInfo(Variant::INT, "idx")));
	BIND_VMETHOD(MethodInfo(Variant::STRING, "_get_input_value_port_name", PropertyInfo(Variant::INT, "idx")));
	BIND_VMETHOD(MethodInfo(Variant::INT, "_get_input_value_port_hint", PropertyInfo(Variant::INT, "idx")));
	BIND_VMETHOD(MethodInfo(Variant::STRING, "_get_input_value_port_hint_string", PropertyInfo(Variant::INT, "idx")));

	BIND_VMETHOD(MethodInfo(Variant::INT, "_get_output_value_port_type", PropertyInfo(Variant::INT, "idx")));
	BIND_VMETHOD(MethodInfo(Variant::STRING, "_get_output_value_port_name", PropertyInfo(Variant::INT, "idx")));
	BIND_VMETHOD(MethodInfo(Variant::INT, "_get_output_value_port_hint", PropertyInfo(Variant::INT, "idx")));
	BIND_VMETHOD(MethodInfo(Variant::STRING, "_get_output_value_port_hint_string", PropertyInfo(Variant::INT, "idx")));

	BIND_VMETHOD(MethodInfo(Variant::STRING, "_get_caption"));
	BIND_VMETHOD(MethodInfo(Variant::STRING, "_get_text"));
	BIND_VMETHOD(MethodInfo(Variant::STRING, "_get_category"));

	BIND_VMETHOD(MethodInfo(Variant::INT, "_get_working_memory_size"));

	MethodInfo stepmi(Variant::NIL, "_step", PropertyInfo(Variant::ARRAY, "inputs"), PropertyInfo(Variant::ARRAY, "outputs"), PropertyInfo(Variant::INT, "start_mode"), PropertyInfo(Variant::ARRAY, "working_mem"));
	stepmi.return_val.usage |= PROPERTY_USAGE_NIL_IS_VARIANT;
	BIND_VMETHOD(stepmi);

	ClassDB::bind_method(D_METHOD("_script_changed"), &VisualScriptCustomNode::_script_changed);

	BIND_ENUM_CONSTANT(START_MODE_BEGIN_SEQUENCE);
	BIND_ENUM_CONSTANT(START_MODE_CONTINUE_SEQUENCE);
	BIND_ENUM_CONSTANT(START_MODE_RESUME_YIELD);

	BIND_CONSTANT(STEP_PUSH_STACK_BIT);
	BIND_CONSTANT(STEP_GO_BACK_BIT);
	BIND_CONSTANT(STEP_NO_ADVANCE_BIT);
	BIND_CONSTANT(STEP_EXIT_FUNCTION_BIT);
	BIND_CONSTANT(STEP_YIELD_BIT);
}

VisualScriptCustomNode::VisualScriptCustomNode() {
	connect("script_changed", this, "_script_changed");
}

//////////////////////////////////////////
////////////////SUBCALL///////////
//////////////////////////////////////////

int VisualScriptSubCall::get_output_sequence_port_count() const {
	return 1;
}

bool VisualScriptSubCall::has_input_sequence_port() const {
	return true;
}

int VisualScriptSubCall::get_input_value_port_count() const {
	Ref<Script> script = get_script();

	if (script.is_valid() && script->has_method(VisualScriptLanguage::singleton->_subcall)) {
		MethodInfo mi = script->get_method_info(VisualScriptLanguage::singleton->_subcall);
		return mi.arguments.size();
	}

	return 0;
}
int VisualScriptSubCall::get_output_value_port_count() const {
	return 1;
}

String VisualScriptSubCall::get_output_sequence_port_text(int p_port) const {
	return String();
}

PropertyInfo VisualScriptSubCall::get_input_value_port_info(int p_idx) const {
	Ref<Script> script = get_script();
	if (script.is_valid() && script->has_method(VisualScriptLanguage::singleton->_subcall)) {
		MethodInfo mi = script->get_method_info(VisualScriptLanguage::singleton->_subcall);
		return mi.arguments[p_idx];
	}

	return PropertyInfo();
}

PropertyInfo VisualScriptSubCall::get_output_value_port_info(int p_idx) const {
	Ref<Script> script = get_script();
	if (script.is_valid() && script->has_method(VisualScriptLanguage::singleton->_subcall)) {
		MethodInfo mi = script->get_method_info(VisualScriptLanguage::singleton->_subcall);
		return mi.return_val;
	}
	return PropertyInfo();
}

String VisualScriptSubCall::get_caption() const {
	return RTR("SubCall");
}

String VisualScriptSubCall::get_text() const {
	Ref<Script> script = get_script();
	if (script.is_valid()) {
		if (script->get_name() != String()) {
			return script->get_name();
		}
		if (script->get_path().is_resource_file()) {
			return script->get_path().get_file();
		}
		return script->get_class();
	}
	return "";
}

String VisualScriptSubCall::get_category() const {
	return "custom";
}

class VisualScriptNodeInstanceSubCall : public VisualScriptNodeInstance {
public:
	VisualScriptInstance *instance;
	VisualScriptSubCall *subcall;
	int input_args;
	bool valid;

	//virtual int get_working_memory_size() const { return 0; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {
		if (!valid) {
			r_error_str = "Node requires a script with a _subcall(<args>) method to work.";
			r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
			return 0;
		}
		*p_outputs[0] = subcall->call(VisualScriptLanguage::singleton->_subcall, p_inputs, input_args, r_error);
		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptSubCall::instance(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceSubCall *instance = memnew(VisualScriptNodeInstanceSubCall);
	instance->instance = p_instance;
	Ref<Script> script = get_script();
	if (script.is_valid() && script->has_method(VisualScriptLanguage::singleton->_subcall)) {
		instance->valid = true;
		instance->input_args = get_input_value_port_count();
	} else {
		instance->valid = false;
	}
	return instance;
}

void VisualScriptSubCall::_bind_methods() {
	MethodInfo scmi(Variant::NIL, "_subcall", PropertyInfo(Variant::NIL, "arguments"));
	scmi.return_val.usage |= PROPERTY_USAGE_NIL_IS_VARIANT;
	BIND_VMETHOD(scmi);
}

VisualScriptSubCall::VisualScriptSubCall() {
}

//////////////////////////////////////////
////////////////Comment///////////
//////////////////////////////////////////

int VisualScriptComment::get_output_sequence_port_count() const {
	return 0;
}

bool VisualScriptComment::has_input_sequence_port() const {
	return false;
}

int VisualScriptComment::get_input_value_port_count() const {
	return 0;
}
int VisualScriptComment::get_output_value_port_count() const {
	return 0;
}

String VisualScriptComment::get_output_sequence_port_text(int p_port) const {
	return String();
}

PropertyInfo VisualScriptComment::get_input_value_port_info(int p_idx) const {
	return PropertyInfo();
}

PropertyInfo VisualScriptComment::get_output_value_port_info(int p_idx) const {
	return PropertyInfo();
}

String VisualScriptComment::get_caption() const {
	return title;
}

String VisualScriptComment::get_text() const {
	return description;
}

void VisualScriptComment::set_title(const String &p_title) {
	if (title == p_title) {
		return;
	}
	title = p_title;
	ports_changed_notify();
}

String VisualScriptComment::get_title() const {
	return title;
}

void VisualScriptComment::set_description(const String &p_description) {
	if (description == p_description) {
		return;
	}
	description = p_description;
	ports_changed_notify();
}
String VisualScriptComment::get_description() const {
	return description;
}

void VisualScriptComment::set_size(const Size2 &p_size) {
	if (size == p_size) {
		return;
	}
	size = p_size;
	ports_changed_notify();
}
Size2 VisualScriptComment::get_size() const {
	return size;
}

String VisualScriptComment::get_category() const {
	return "data";
}

class VisualScriptNodeInstanceComment : public VisualScriptNodeInstance {
public:
	VisualScriptInstance *instance;

	//virtual int get_working_memory_size() const { return 0; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {
		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptComment::instance(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceComment *instance = memnew(VisualScriptNodeInstanceComment);
	instance->instance = p_instance;
	return instance;
}

void VisualScriptComment::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_title", "title"), &VisualScriptComment::set_title);
	ClassDB::bind_method(D_METHOD("get_title"), &VisualScriptComment::get_title);

	ClassDB::bind_method(D_METHOD("set_description", "description"), &VisualScriptComment::set_description);
	ClassDB::bind_method(D_METHOD("get_description"), &VisualScriptComment::get_description);

	ClassDB::bind_method(D_METHOD("set_size", "size"), &VisualScriptComment::set_size);
	ClassDB::bind_method(D_METHOD("get_size"), &VisualScriptComment::get_size);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "title"), "set_title", "get_title");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "description", PROPERTY_HINT_MULTILINE_TEXT), "set_description", "get_description");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "size"), "set_size", "get_size");
}

VisualScriptComment::VisualScriptComment() {
	title = "Comment";
	size = Size2(150, 150);
}

//////////////////////////////////////////
////////////////Constructor///////////
//////////////////////////////////////////

int VisualScriptConstructor::get_output_sequence_port_count() const {
	return 0;
}

bool VisualScriptConstructor::has_input_sequence_port() const {
	return false;
}

int VisualScriptConstructor::get_input_value_port_count() const {
	return constructor.arguments.size();
}
int VisualScriptConstructor::get_output_value_port_count() const {
	return 1;
}

String VisualScriptConstructor::get_output_sequence_port_text(int p_port) const {
	return "";
}

PropertyInfo VisualScriptConstructor::get_input_value_port_info(int p_idx) const {
	return constructor.arguments[p_idx];
}

PropertyInfo VisualScriptConstructor::get_output_value_port_info(int p_idx) const {
	return PropertyInfo(type, "value");
}

String VisualScriptConstructor::get_caption() const {
	return vformat(RTR("Construct %s"), Variant::get_type_name(type));
}

String VisualScriptConstructor::get_category() const {
	return "functions";
}

void VisualScriptConstructor::set_constructor_type(Variant::Type p_type) {
	if (type == p_type) {
		return;
	}

	type = p_type;
	ports_changed_notify();
}

Variant::Type VisualScriptConstructor::get_constructor_type() const {
	return type;
}

void VisualScriptConstructor::set_constructor(const Dictionary &p_info) {
	constructor = MethodInfo::from_dict(p_info);
	ports_changed_notify();
}

Dictionary VisualScriptConstructor::get_constructor() const {
	return constructor;
}

class VisualScriptNodeInstanceConstructor : public VisualScriptNodeInstance {
public:
	VisualScriptInstance *instance;
	Variant::Type type;
	int argcount;

	//virtual int get_working_memory_size() const { return 0; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {
		Variant::CallError ce;
		*p_outputs[0] = Variant::construct(type, p_inputs, argcount, ce);
		if (ce.error != Variant::CallError::CALL_OK) {
			r_error_str = "Invalid arguments for constructor";
		}

		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptConstructor::instance(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceConstructor *instance = memnew(VisualScriptNodeInstanceConstructor);
	instance->instance = p_instance;
	instance->type = type;
	instance->argcount = constructor.arguments.size();
	return instance;
}

void VisualScriptConstructor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_constructor_type", "type"), &VisualScriptConstructor::set_constructor_type);
	ClassDB::bind_method(D_METHOD("get_constructor_type"), &VisualScriptConstructor::get_constructor_type);

	ClassDB::bind_method(D_METHOD("set_constructor", "constructor"), &VisualScriptConstructor::set_constructor);
	ClassDB::bind_method(D_METHOD("get_constructor"), &VisualScriptConstructor::get_constructor);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "type", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL), "set_constructor_type", "get_constructor_type");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "constructor", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL), "set_constructor", "get_constructor");
}

VisualScriptConstructor::VisualScriptConstructor() {
	type = Variant::NIL;
}

static Map<String, Pair<Variant::Type, MethodInfo>> constructor_map;

static Ref<VisualScriptNode> create_constructor_node(const String &p_name) {
	ERR_FAIL_COND_V(!constructor_map.has(p_name), Ref<VisualScriptNode>());

	Ref<VisualScriptConstructor> vsc;
	vsc.instance();
	vsc->set_constructor_type(constructor_map[p_name].first);
	vsc->set_constructor(constructor_map[p_name].second);

	return vsc;
}

//////////////////////////////////////////
////////////////LocalVar///////////
//////////////////////////////////////////

int VisualScriptLocalVar::get_output_sequence_port_count() const {
	return 0;
}

bool VisualScriptLocalVar::has_input_sequence_port() const {
	return false;
}

int VisualScriptLocalVar::get_input_value_port_count() const {
	return 0;
}
int VisualScriptLocalVar::get_output_value_port_count() const {
	return 1;
}

String VisualScriptLocalVar::get_output_sequence_port_text(int p_port) const {
	return "";
}

PropertyInfo VisualScriptLocalVar::get_input_value_port_info(int p_idx) const {
	return PropertyInfo();
}
PropertyInfo VisualScriptLocalVar::get_output_value_port_info(int p_idx) const {
	return PropertyInfo(type, name);
}

String VisualScriptLocalVar::get_caption() const {
	return RTR("Get Local Var");
}

String VisualScriptLocalVar::get_category() const {
	return "data";
}

void VisualScriptLocalVar::set_var_name(const StringName &p_name) {
	if (name == p_name) {
		return;
	}

	name = p_name;
	ports_changed_notify();
}

StringName VisualScriptLocalVar::get_var_name() const {
	return name;
}

void VisualScriptLocalVar::set_var_type(Variant::Type p_type) {
	type = p_type;
	ports_changed_notify();
}

Variant::Type VisualScriptLocalVar::get_var_type() const {
	return type;
}

class VisualScriptNodeInstanceLocalVar : public VisualScriptNodeInstance {
public:
	VisualScriptInstance *instance;
	StringName name;

	virtual int get_working_memory_size() const { return 1; }
	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {
		*p_outputs[0] = *p_working_mem;
		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptLocalVar::instance(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceLocalVar *instance = memnew(VisualScriptNodeInstanceLocalVar);
	instance->instance = p_instance;
	instance->name = name;

	return instance;
}

void VisualScriptLocalVar::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_var_name", "name"), &VisualScriptLocalVar::set_var_name);
	ClassDB::bind_method(D_METHOD("get_var_name"), &VisualScriptLocalVar::get_var_name);

	ClassDB::bind_method(D_METHOD("set_var_type", "type"), &VisualScriptLocalVar::set_var_type);
	ClassDB::bind_method(D_METHOD("get_var_type"), &VisualScriptLocalVar::get_var_type);

	String argt = "Any";
	for (int i = 1; i < Variant::VARIANT_MAX; i++) {
		argt += "," + Variant::get_type_name(Variant::Type(i));
	}

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "var_name"), "set_var_name", "get_var_name");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "type", PROPERTY_HINT_ENUM, argt), "set_var_type", "get_var_type");
}

VisualScriptLocalVar::VisualScriptLocalVar() {
	name = "new_local";
	type = Variant::NIL;
}

//////////////////////////////////////////
////////////////LocalVar///////////
//////////////////////////////////////////

int VisualScriptLocalVarSet::get_output_sequence_port_count() const {
	return 1;
}

bool VisualScriptLocalVarSet::has_input_sequence_port() const {
	return true;
}

int VisualScriptLocalVarSet::get_input_value_port_count() const {
	return 1;
}
int VisualScriptLocalVarSet::get_output_value_port_count() const {
	return 1;
}

String VisualScriptLocalVarSet::get_output_sequence_port_text(int p_port) const {
	return "";
}

PropertyInfo VisualScriptLocalVarSet::get_input_value_port_info(int p_idx) const {
	return PropertyInfo(type, "set");
}
PropertyInfo VisualScriptLocalVarSet::get_output_value_port_info(int p_idx) const {
	return PropertyInfo(type, "get");
}

String VisualScriptLocalVarSet::get_caption() const {
	return RTR("Set Local Var");
}

String VisualScriptLocalVarSet::get_text() const {
	return name;
}

String VisualScriptLocalVarSet::get_category() const {
	return "data";
}

void VisualScriptLocalVarSet::set_var_name(const StringName &p_name) {
	if (name == p_name) {
		return;
	}

	name = p_name;
	ports_changed_notify();
}

StringName VisualScriptLocalVarSet::get_var_name() const {
	return name;
}

void VisualScriptLocalVarSet::set_var_type(Variant::Type p_type) {
	type = p_type;
	ports_changed_notify();
}

Variant::Type VisualScriptLocalVarSet::get_var_type() const {
	return type;
}

class VisualScriptNodeInstanceLocalVarSet : public VisualScriptNodeInstance {
public:
	VisualScriptInstance *instance;
	StringName name;

	virtual int get_working_memory_size() const { return 1; }
	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {
		*p_working_mem = *p_inputs[0];
		*p_outputs[0] = *p_working_mem;
		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptLocalVarSet::instance(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceLocalVarSet *instance = memnew(VisualScriptNodeInstanceLocalVarSet);
	instance->instance = p_instance;
	instance->name = name;

	return instance;
}

void VisualScriptLocalVarSet::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_var_name", "name"), &VisualScriptLocalVarSet::set_var_name);
	ClassDB::bind_method(D_METHOD("get_var_name"), &VisualScriptLocalVarSet::get_var_name);

	ClassDB::bind_method(D_METHOD("set_var_type", "type"), &VisualScriptLocalVarSet::set_var_type);
	ClassDB::bind_method(D_METHOD("get_var_type"), &VisualScriptLocalVarSet::get_var_type);

	String argt = "Any";
	for (int i = 1; i < Variant::VARIANT_MAX; i++) {
		argt += "," + Variant::get_type_name(Variant::Type(i));
	}

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "var_name"), "set_var_name", "get_var_name");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "type", PROPERTY_HINT_ENUM, argt), "set_var_type", "get_var_type");
}

VisualScriptLocalVarSet::VisualScriptLocalVarSet() {
	name = "new_local";
	type = Variant::NIL;
}

//////////////////////////////////////////
////////////////LocalVar///////////
//////////////////////////////////////////

int VisualScriptInputAction::get_output_sequence_port_count() const {
	return 0;
}

bool VisualScriptInputAction::has_input_sequence_port() const {
	return false;
}

int VisualScriptInputAction::get_input_value_port_count() const {
	return 0;
}
int VisualScriptInputAction::get_output_value_port_count() const {
	return 1;
}

String VisualScriptInputAction::get_output_sequence_port_text(int p_port) const {
	return "";
}

PropertyInfo VisualScriptInputAction::get_input_value_port_info(int p_idx) const {
	return PropertyInfo();
}
PropertyInfo VisualScriptInputAction::get_output_value_port_info(int p_idx) const {
	String mstr;
	switch (mode) {
		case MODE_PRESSED: {
			mstr = "pressed";
		} break;
		case MODE_RELEASED: {
			mstr = "not pressed";
		} break;
		case MODE_JUST_PRESSED: {
			mstr = "just pressed";
		} break;
		case MODE_JUST_RELEASED: {
			mstr = "just released";
		} break;
	}

	return PropertyInfo(Variant::BOOL, mstr);
}

String VisualScriptInputAction::get_caption() const {
	return vformat(RTR("Action %s"), name);
}

String VisualScriptInputAction::get_category() const {
	return "data";
}

void VisualScriptInputAction::set_action_name(const StringName &p_name) {
	if (name == p_name) {
		return;
	}

	name = p_name;
	ports_changed_notify();
}

StringName VisualScriptInputAction::get_action_name() const {
	return name;
}

void VisualScriptInputAction::set_action_mode(Mode p_mode) {
	if (mode == p_mode) {
		return;
	}

	mode = p_mode;
	ports_changed_notify();
}
VisualScriptInputAction::Mode VisualScriptInputAction::get_action_mode() const {
	return mode;
}

class VisualScriptNodeInstanceInputAction : public VisualScriptNodeInstance {
public:
	VisualScriptInstance *instance;
	StringName action;
	VisualScriptInputAction::Mode mode;

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {
		switch (mode) {
			case VisualScriptInputAction::MODE_PRESSED: {
				*p_outputs[0] = Input::get_singleton()->is_action_pressed(action);
			} break;
			case VisualScriptInputAction::MODE_RELEASED: {
				*p_outputs[0] = !Input::get_singleton()->is_action_pressed(action);
			} break;
			case VisualScriptInputAction::MODE_JUST_PRESSED: {
				*p_outputs[0] = Input::get_singleton()->is_action_just_pressed(action);
			} break;
			case VisualScriptInputAction::MODE_JUST_RELEASED: {
				*p_outputs[0] = Input::get_singleton()->is_action_just_released(action);
			} break;
		}

		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptInputAction::instance(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceInputAction *instance = memnew(VisualScriptNodeInstanceInputAction);
	instance->instance = p_instance;
	instance->action = name;
	instance->mode = mode;

	return instance;
}

void VisualScriptInputAction::_validate_property(PropertyInfo &property) const {
	if (property.name == "action") {
		property.hint = PROPERTY_HINT_ENUM;
		String actions;

		List<PropertyInfo> pinfo;
		ProjectSettings::get_singleton()->get_property_list(&pinfo);
		Vector<String> al;

		for (List<PropertyInfo>::Element *E = pinfo.front(); E; E = E->next()) {
			const PropertyInfo &pi = E->get();

			if (!pi.name.begins_with("input/")) {
				continue;
			}

			String name = pi.name.substr(pi.name.find("/") + 1, pi.name.length());

			al.push_back(name);
		}

		al.sort();

		for (int i = 0; i < al.size(); i++) {
			if (actions != String()) {
				actions += ",";
			}
			actions += al[i];
		}

		property.hint_string = actions;
	}
}

void VisualScriptInputAction::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_action_name", "name"), &VisualScriptInputAction::set_action_name);
	ClassDB::bind_method(D_METHOD("get_action_name"), &VisualScriptInputAction::get_action_name);

	ClassDB::bind_method(D_METHOD("set_action_mode", "mode"), &VisualScriptInputAction::set_action_mode);
	ClassDB::bind_method(D_METHOD("get_action_mode"), &VisualScriptInputAction::get_action_mode);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "action"), "set_action_name", "get_action_name");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "mode", PROPERTY_HINT_ENUM, "Pressed,Released,JustPressed,JustReleased"), "set_action_mode", "get_action_mode");

	BIND_ENUM_CONSTANT(MODE_PRESSED);
	BIND_ENUM_CONSTANT(MODE_RELEASED);
	BIND_ENUM_CONSTANT(MODE_JUST_PRESSED);
	BIND_ENUM_CONSTANT(MODE_JUST_RELEASED);
}

VisualScriptInputAction::VisualScriptInputAction() {
	name = "";
	mode = MODE_PRESSED;
}

//////////////////////////////////////////
////////////////Constructor///////////
//////////////////////////////////////////

int VisualScriptDeconstruct::get_output_sequence_port_count() const {
	return 0;
}

bool VisualScriptDeconstruct::has_input_sequence_port() const {
	return false;
}

int VisualScriptDeconstruct::get_input_value_port_count() const {
	return 1;
}
int VisualScriptDeconstruct::get_output_value_port_count() const {
	return elements.size();
}

String VisualScriptDeconstruct::get_output_sequence_port_text(int p_port) const {
	return "";
}

PropertyInfo VisualScriptDeconstruct::get_input_value_port_info(int p_idx) const {
	return PropertyInfo(type, "value");
}

PropertyInfo VisualScriptDeconstruct::get_output_value_port_info(int p_idx) const {
	return PropertyInfo(elements[p_idx].type, elements[p_idx].name);
}

String VisualScriptDeconstruct::get_caption() const {
	return vformat(RTR("Deconstruct %s"), Variant::get_type_name(type));
}

String VisualScriptDeconstruct::get_category() const {
	return "functions";
}

void VisualScriptDeconstruct::_update_elements() {
	elements.clear();
	Variant v;
	Variant::CallError ce;
	v = Variant::construct(type, nullptr, 0, ce);

	List<PropertyInfo> pinfo;
	v.get_property_list(&pinfo);

	for (List<PropertyInfo>::Element *E = pinfo.front(); E; E = E->next()) {
		Element e;
		e.name = E->get().name;
		e.type = E->get().type;
		elements.push_back(e);
	}
}

void VisualScriptDeconstruct::set_deconstruct_type(Variant::Type p_type) {
	if (type == p_type) {
		return;
	}

	type = p_type;
	_update_elements();
	ports_changed_notify();
	_change_notify(); //to make input appear/disappear
}

Variant::Type VisualScriptDeconstruct::get_deconstruct_type() const {
	return type;
}

void VisualScriptDeconstruct::_set_elem_cache(const Array &p_elements) {
	ERR_FAIL_COND(p_elements.size() % 2 == 1);
	elements.resize(p_elements.size() / 2);
	for (int i = 0; i < elements.size(); i++) {
		elements.write[i].name = p_elements[i * 2 + 0];
		elements.write[i].type = Variant::Type(int(p_elements[i * 2 + 1]));
	}
}

Array VisualScriptDeconstruct::_get_elem_cache() const {
	Array ret;
	for (int i = 0; i < elements.size(); i++) {
		ret.push_back(elements[i].name);
		ret.push_back(elements[i].type);
	}
	return ret;
}

class VisualScriptNodeInstanceDeconstruct : public VisualScriptNodeInstance {
public:
	VisualScriptInstance *instance;
	Vector<StringName> outputs;

	//virtual int get_working_memory_size() const { return 0; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {
		Variant in = *p_inputs[0];

		for (int i = 0; i < outputs.size(); i++) {
			bool valid;
			*p_outputs[i] = in.get(outputs[i], &valid);
			if (!valid) {
				r_error_str = "Can't obtain element '" + String(outputs[i]) + "' from " + Variant::get_type_name(in.get_type());
				r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
				return 0;
			}
		}

		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptDeconstruct::instance(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceDeconstruct *instance = memnew(VisualScriptNodeInstanceDeconstruct);
	instance->instance = p_instance;
	instance->outputs.resize(elements.size());
	for (int i = 0; i < elements.size(); i++) {
		instance->outputs.write[i] = elements[i].name;
	}

	return instance;
}

void VisualScriptDeconstruct::_validate_property(PropertyInfo &property) const {
}

void VisualScriptDeconstruct::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_deconstruct_type", "type"), &VisualScriptDeconstruct::set_deconstruct_type);
	ClassDB::bind_method(D_METHOD("get_deconstruct_type"), &VisualScriptDeconstruct::get_deconstruct_type);

	ClassDB::bind_method(D_METHOD("_set_elem_cache", "_cache"), &VisualScriptDeconstruct::_set_elem_cache);
	ClassDB::bind_method(D_METHOD("_get_elem_cache"), &VisualScriptDeconstruct::_get_elem_cache);

	String argt = "Any";
	for (int i = 1; i < Variant::VARIANT_MAX; i++) {
		argt += "," + Variant::get_type_name(Variant::Type(i));
	}

	ADD_PROPERTY(PropertyInfo(Variant::INT, "type", PROPERTY_HINT_ENUM, argt), "set_deconstruct_type", "get_deconstruct_type");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "elem_cache", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL), "_set_elem_cache", "_get_elem_cache");
}

VisualScriptDeconstruct::VisualScriptDeconstruct() {
	type = Variant::NIL;
}

template <Variant::Type T>
static Ref<VisualScriptNode> create_node_deconst_typed(const String &p_name) {
	Ref<VisualScriptDeconstruct> node;
	node.instance();
	node->set_deconstruct_type(T);
	return node;
}

void register_visual_script_nodes() {
	VisualScriptLanguage::singleton->add_register_func("data/set_variable", create_node_generic<VisualScriptVariableSet>);
	VisualScriptLanguage::singleton->add_register_func("data/get_variable", create_node_generic<VisualScriptVariableGet>);
	VisualScriptLanguage::singleton->add_register_func("data/engine_singleton", create_node_generic<VisualScriptEngineSingleton>);
	VisualScriptLanguage::singleton->add_register_func("data/scene_node", create_node_generic<VisualScriptSceneNode>);
	VisualScriptLanguage::singleton->add_register_func("data/scene_tree", create_node_generic<VisualScriptSceneTree>);
	VisualScriptLanguage::singleton->add_register_func("data/resource_path", create_node_generic<VisualScriptResourcePath>);
	VisualScriptLanguage::singleton->add_register_func("data/self", create_node_generic<VisualScriptSelf>);
	VisualScriptLanguage::singleton->add_register_func("data/comment", create_node_generic<VisualScriptComment>);
	VisualScriptLanguage::singleton->add_register_func("data/get_local_variable", create_node_generic<VisualScriptLocalVar>);
	VisualScriptLanguage::singleton->add_register_func("data/set_local_variable", create_node_generic<VisualScriptLocalVarSet>);
	VisualScriptLanguage::singleton->add_register_func("data/preload", create_node_generic<VisualScriptPreload>);
	VisualScriptLanguage::singleton->add_register_func("data/action", create_node_generic<VisualScriptInputAction>);

	VisualScriptLanguage::singleton->add_register_func("constants/constant", create_node_generic<VisualScriptConstant>);
	VisualScriptLanguage::singleton->add_register_func("constants/math_constant", create_node_generic<VisualScriptMathConstant>);
	VisualScriptLanguage::singleton->add_register_func("constants/class_constant", create_node_generic<VisualScriptClassConstant>);
	VisualScriptLanguage::singleton->add_register_func("constants/global_constant", create_node_generic<VisualScriptGlobalConstant>);
	VisualScriptLanguage::singleton->add_register_func("constants/basic_type_constant", create_node_generic<VisualScriptBasicTypeConstant>);

	VisualScriptLanguage::singleton->add_register_func("custom/custom_node", create_node_generic<VisualScriptCustomNode>);
	VisualScriptLanguage::singleton->add_register_func("custom/sub_call", create_node_generic<VisualScriptSubCall>);

	VisualScriptLanguage::singleton->add_register_func("index/get_index", create_node_generic<VisualScriptIndexGet>);
	VisualScriptLanguage::singleton->add_register_func("index/set_index", create_node_generic<VisualScriptIndexSet>);

	VisualScriptLanguage::singleton->add_register_func("operators/compare/equal", create_op_node<Variant::OP_EQUAL>);
	VisualScriptLanguage::singleton->add_register_func("operators/compare/not_equal", create_op_node<Variant::OP_NOT_EQUAL>);
	VisualScriptLanguage::singleton->add_register_func("operators/compare/less", create_op_node<Variant::OP_LESS>);
	VisualScriptLanguage::singleton->add_register_func("operators/compare/less_equal", create_op_node<Variant::OP_LESS_EQUAL>);
	VisualScriptLanguage::singleton->add_register_func("operators/compare/greater", create_op_node<Variant::OP_GREATER>);
	VisualScriptLanguage::singleton->add_register_func("operators/compare/greater_equal", create_op_node<Variant::OP_GREATER_EQUAL>);
	//mathematic
	VisualScriptLanguage::singleton->add_register_func("operators/math/add", create_op_node<Variant::OP_ADD>);
	VisualScriptLanguage::singleton->add_register_func("operators/math/subtract", create_op_node<Variant::OP_SUBTRACT>);
	VisualScriptLanguage::singleton->add_register_func("operators/math/multiply", create_op_node<Variant::OP_MULTIPLY>);
	VisualScriptLanguage::singleton->add_register_func("operators/math/divide", create_op_node<Variant::OP_DIVIDE>);
	VisualScriptLanguage::singleton->add_register_func("operators/math/negate", create_op_node<Variant::OP_NEGATE>);
	VisualScriptLanguage::singleton->add_register_func("operators/math/positive", create_op_node<Variant::OP_POSITIVE>);
	VisualScriptLanguage::singleton->add_register_func("operators/math/remainder", create_op_node<Variant::OP_MODULE>);
	VisualScriptLanguage::singleton->add_register_func("operators/math/string_concat", create_op_node<Variant::OP_STRING_CONCAT>);
	//bitwise
	VisualScriptLanguage::singleton->add_register_func("operators/bitwise/shift_left", create_op_node<Variant::OP_SHIFT_LEFT>);
	VisualScriptLanguage::singleton->add_register_func("operators/bitwise/shift_right", create_op_node<Variant::OP_SHIFT_RIGHT>);
	VisualScriptLanguage::singleton->add_register_func("operators/bitwise/bit_and", create_op_node<Variant::OP_BIT_AND>);
	VisualScriptLanguage::singleton->add_register_func("operators/bitwise/bit_or", create_op_node<Variant::OP_BIT_OR>);
	VisualScriptLanguage::singleton->add_register_func("operators/bitwise/bit_xor", create_op_node<Variant::OP_BIT_XOR>);
	VisualScriptLanguage::singleton->add_register_func("operators/bitwise/bit_negate", create_op_node<Variant::OP_BIT_NEGATE>);
	//logic
	VisualScriptLanguage::singleton->add_register_func("operators/logic/and", create_op_node<Variant::OP_AND>);
	VisualScriptLanguage::singleton->add_register_func("operators/logic/or", create_op_node<Variant::OP_OR>);
	VisualScriptLanguage::singleton->add_register_func("operators/logic/xor", create_op_node<Variant::OP_XOR>);
	VisualScriptLanguage::singleton->add_register_func("operators/logic/not", create_op_node<Variant::OP_NOT>);
	VisualScriptLanguage::singleton->add_register_func("operators/logic/in", create_op_node<Variant::OP_IN>);
	VisualScriptLanguage::singleton->add_register_func("operators/logic/select", create_node_generic<VisualScriptSelect>);

	VisualScriptLanguage::singleton->add_register_func("functions/deconstruct/" + Variant::get_type_name(Variant::Type::VECTOR2), create_node_deconst_typed<Variant::Type::VECTOR2>);
	VisualScriptLanguage::singleton->add_register_func("functions/deconstruct/" + Variant::get_type_name(Variant::Type::VECTOR3), create_node_deconst_typed<Variant::Type::VECTOR3>);
	VisualScriptLanguage::singleton->add_register_func("functions/deconstruct/" + Variant::get_type_name(Variant::Type::COLOR), create_node_deconst_typed<Variant::Type::COLOR>);
	VisualScriptLanguage::singleton->add_register_func("functions/deconstruct/" + Variant::get_type_name(Variant::Type::RECT2), create_node_deconst_typed<Variant::Type::RECT2>);
	VisualScriptLanguage::singleton->add_register_func("functions/deconstruct/" + Variant::get_type_name(Variant::Type::TRANSFORM2D), create_node_deconst_typed<Variant::Type::TRANSFORM2D>);
	VisualScriptLanguage::singleton->add_register_func("functions/deconstruct/" + Variant::get_type_name(Variant::Type::PLANE), create_node_deconst_typed<Variant::Type::PLANE>);
	VisualScriptLanguage::singleton->add_register_func("functions/deconstruct/" + Variant::get_type_name(Variant::Type::QUAT), create_node_deconst_typed<Variant::Type::QUAT>);
	VisualScriptLanguage::singleton->add_register_func("functions/deconstruct/" + Variant::get_type_name(Variant::Type::AABB), create_node_deconst_typed<Variant::Type::AABB>);
	VisualScriptLanguage::singleton->add_register_func("functions/deconstruct/" + Variant::get_type_name(Variant::Type::BASIS), create_node_deconst_typed<Variant::Type::BASIS>);
	VisualScriptLanguage::singleton->add_register_func("functions/deconstruct/" + Variant::get_type_name(Variant::Type::TRANSFORM), create_node_deconst_typed<Variant::Type::TRANSFORM>);
	VisualScriptLanguage::singleton->add_register_func("functions/compose_array", create_node_generic<VisualScriptComposeArray>);

	for (int i = 1; i < Variant::VARIANT_MAX; i++) {
		List<MethodInfo> constructors;
		Variant::get_constructor_list(Variant::Type(i), &constructors);

		for (List<MethodInfo>::Element *E = constructors.front(); E; E = E->next()) {
			if (E->get().arguments.size() > 0) {
				String name = "functions/constructors/" + Variant::get_type_name(Variant::Type(i)) + "(";
				for (int j = 0; j < E->get().arguments.size(); j++) {
					if (j > 0) {
						name += ", ";
					}
					if (E->get().arguments.size() == 1) {
						name += Variant::get_type_name(E->get().arguments[j].type);
					} else {
						name += E->get().arguments[j].name;
					}
				}
				name += ")";
				VisualScriptLanguage::singleton->add_register_func(name, create_constructor_node);
				Pair<Variant::Type, MethodInfo> pair;
				pair.first = Variant::Type(i);
				pair.second = E->get();
				constructor_map[name] = pair;
			}
		}
	}
}

void unregister_visual_script_nodes() {
	constructor_map.clear();
}
