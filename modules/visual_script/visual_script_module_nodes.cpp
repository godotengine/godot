/*************************************************************************/
/*  visual_script_module_nodes.cpp                                       */
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

#include "visual_script_module_nodes.h"

void VisualScriptModuleNode::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_module", "module_name"), &VisualScriptModuleNode::set_module);
	ClassDB::bind_method(D_METHOD("get_module_name"), &VisualScriptModuleNode::get_module_name);

	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "module_name"), "set_module", "get_module_name");
}

int VisualScriptModuleNode::get_output_sequence_port_count() const {
	return 1; // until it's figured out how to have multiple sequence inputs this can't be more than 1
}
bool VisualScriptModuleNode::has_input_sequence_port() const {
	return true;
}
String VisualScriptModuleNode::get_output_sequence_port_text(int p_port) const {
	return "";
}

int VisualScriptModuleNode::update_module_data() {
	ERR_FAIL_COND_V(module_name == "" || get_container().is_null(), 0);

	Ref<VisualScript> vs = get_container(); // Modules can't work in other Modules (yet!)
	ERR_FAIL_COND_V(!vs.is_valid(), 0);

	if (!vs->has_module(module_name)) {
		// reset
		input_ports.clear();
		output_ports.clear();
		return -1;
	}

	Ref<VisualScriptModule> module = vs->get_module(module_name);
	ERR_FAIL_COND_V(!module.is_valid(), 0);

	Ref<VisualScriptNode> exit_node = module->get_node(1);
	Ref<VisualScriptNode> entry_node = module->get_node(0);

	int ret = -1;

	// If `exit_node` and `entry_node` are same to current ones then dont do anything
	if (exit_node.is_valid()) {
		int cnt = exit_node->get_input_value_port_count();
		if (cnt == output_ports.size()) {
			PropertyInfo pi;
			for (int i = 0; i < cnt; i++) {
				pi = exit_node->get_input_value_port_info(i);
				if (output_ports[i].name != pi.name && output_ports[i].type != pi.type) {
					output_ports.write[i].name = pi.name;
					output_ports.write[i].type = pi.type;
					ret = 1;
				}
			}
		} else {
			// update everything
			output_ports.resize(cnt);
			PropertyInfo pi;
			for (int i = 0; i < cnt; i++) {
				pi = exit_node->get_input_value_port_info(i);
				output_ports.write[i].name = pi.name;
				output_ports.write[i].type = pi.type;
			}
			ret = 1;
		}
	} else {
		ERR_FAIL_V_MSG(0, "Module doesn't have a valid Exit Node.");
	}

	if (entry_node.is_valid()) {
		int cnt = entry_node->get_output_value_port_count();
		if (cnt == input_ports.size()) {
			PropertyInfo pi;
			for (int i = 0; i < cnt; i++) {
				pi = entry_node->get_output_value_port_info(i);
				if (input_ports[i].name != pi.name && input_ports[i].type != pi.type) {
					input_ports.write[i].name = pi.name;
					input_ports.write[i].type = pi.type;
					ret = 1;
				}
			}
		} else {
			// update everything
			input_ports.resize(cnt);
			PropertyInfo pi;
			for (int i = 0; i < cnt; i++) {
				pi = entry_node->get_output_value_port_info(i);
				input_ports.write[i].name = pi.name;
				input_ports.write[i].type = pi.type;
			}
			ret = 1;
		}
	} else {
		ERR_FAIL_V_MSG(0, "Module doesn't have a valid Entry Node.");
	}

	return ret;
}

int VisualScriptModuleNode::get_input_value_port_count() const {
	int updating_success = const_cast<VisualScriptModuleNode *>(this)->update_module_data();
	ERR_FAIL_COND_V(!updating_success, 0);

	if (updating_success == 1) {
		const_cast<VisualScriptModuleNode *>(this)->validate_input_default_values(); // just make sure this is called if there is an update
	}

	return input_ports.size();
}

int VisualScriptModuleNode::get_output_value_port_count() const {
	int updating_success = const_cast<VisualScriptModuleNode *>(this)->update_module_data();
	ERR_FAIL_COND_V(!updating_success, 0);

	return output_ports.size();
}

PropertyInfo VisualScriptModuleNode::get_input_value_port_info(int p_idx) const {
	int updating_success = const_cast<VisualScriptModuleNode *>(this)->update_module_data();
	ERR_FAIL_COND_V(!updating_success, PropertyInfo());

	if (updating_success == 1) {
		const_cast<VisualScriptModuleNode *>(this)->validate_input_default_values(); // just make sure this is called if there is an update
	}

	ERR_FAIL_INDEX_V(p_idx, input_ports.size(), PropertyInfo());

	PropertyInfo pi;
	pi.name = input_ports[p_idx].name;
	pi.type = input_ports[p_idx].type;
	return pi;
}

PropertyInfo VisualScriptModuleNode::get_output_value_port_info(int p_idx) const {
	int updating_success = const_cast<VisualScriptModuleNode *>(this)->update_module_data();
	ERR_FAIL_COND_V(!updating_success, PropertyInfo());

	ERR_FAIL_INDEX_V(p_idx, output_ports.size(), PropertyInfo());

	PropertyInfo pi;
	pi.name = output_ports[p_idx].name;
	pi.type = output_ports[p_idx].type;
	return pi;
}

String VisualScriptModuleNode::get_caption() const {
	return "Module Node";
}
String VisualScriptModuleNode::get_text() const {
	return "";
}

void VisualScriptModuleNode::set_module(const String &p_name) {
	module_name = p_name;
	ports_changed_notify();
	notify_property_list_changed();
}

String VisualScriptModuleNode::get_module_name() const {
	return module_name;
}

class VisualScriptModuleNodeInstance : public VisualScriptNodeInstance {
public:
	VisualScriptModuleNode *node;
	VisualScriptInstance *instance;

	//virtual int get_working_memory_size() const { return 0; }
	//virtual bool is_output_port_unsequenced(int p_idx) const { return false; }
	//virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return true; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Callable::CallError &r_error, String &r_error_str) {
		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptModuleNode::instantiate(VisualScriptInstance *p_instance) {
	VisualScriptModuleNodeInstance *instance = memnew(VisualScriptModuleNodeInstance);
	instance->node = this;
	instance->instance = p_instance;
	return instance;
}

VisualScriptModuleNode::VisualScriptModuleNode() {
	module_name = "";
}

VisualScriptModuleNode::~VisualScriptModuleNode() {}

// VisualScriptNodeInstance *instance(VisualScriptInstance *p_instance);

void register_visual_script_module_nodes() {
	VisualScriptLanguage::singleton->add_register_func("modules/module_node", create_node_generic<VisualScriptModuleNode>);
}

int VisualScriptModuleEntryNode::get_output_value_port_count() const {
	return outputports.size();
}

PropertyInfo VisualScriptModuleEntryNode::get_output_value_port_info(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, outputports.size(), PropertyInfo());

	PropertyInfo pi;
	pi.name = outputports[p_idx].name;
	pi.type = outputports[p_idx].type;
	return pi;
}

class VisualScriptModuleEntryNodeInstance : public VisualScriptNodeInstance {
public:
	//VisualScriptModuleNode *node;
	VisualScriptInstance *instance;
	VisualScriptModuleEntryNode *node;
	//virtual int get_working_memory_size() const { return 0; }
	//virtual bool is_output_port_unsequenced(int p_idx) const { return false; }
	//virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return true; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Callable::CallError &r_error, String &r_error_str) {
		int ac = node->get_output_value_port_count();

		for (int i = 0; i < ac; i++) {
#ifdef DEBUG_ENABLED
			Variant::Type expected = node->get_output_value_port_info(i).type;
			if (expected != Variant::NIL) {
				if (!Variant::can_convert_strict(p_inputs[i]->get_type(), expected)) {
					r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
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

VisualScriptNodeInstance *VisualScriptModuleEntryNode::instantiate(VisualScriptInstance *p_instance) {
	VisualScriptModuleEntryNodeInstance *instance = memnew(VisualScriptModuleEntryNodeInstance);
	instance->node = this;
	instance->instance = p_instance;
	return instance;
}

VisualScriptModuleEntryNode::VisualScriptModuleEntryNode() {
	stack_less = false;
	stack_size = 256;
}

VisualScriptModuleEntryNode::~VisualScriptModuleEntryNode() {}

void VisualScriptModuleEntryNode::set_stack_less(bool p_enable) {
	stack_less = p_enable;
	notify_property_list_changed();
}

bool VisualScriptModuleEntryNode::is_stack_less() const {
	return stack_less;
}

void VisualScriptModuleEntryNode::set_sequenced(bool p_enable) {
	sequenced = p_enable;
}

bool VisualScriptModuleEntryNode::is_sequenced() const {
	return sequenced;
}

void VisualScriptModuleEntryNode::set_stack_size(int p_size) {
	ERR_FAIL_COND(p_size < 1 || p_size > 100000);
	stack_size = p_size;
}

int VisualScriptModuleEntryNode::get_stack_size() const {
	return stack_size;
}

int VisualScriptModuleExitNode::get_input_value_port_count() const {
	return inputports.size();
}

PropertyInfo VisualScriptModuleExitNode::get_input_value_port_info(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, inputports.size(), PropertyInfo());

	PropertyInfo pi;
	pi.name = inputports[p_idx].name;
	pi.type = inputports[p_idx].type;
	return pi;
}

class VisualScriptModuleExitNodeInstance : public VisualScriptNodeInstance {
public:
	VisualScriptModuleExitNode *node;
	VisualScriptInstance *instance;
	int output_count;

	virtual int get_working_memory_size() const { return 1; }
	//virtual bool is_output_port_unsequenced(int p_idx) const { return false; }
	//virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return true; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Callable::CallError &r_error, String &r_error_str) {
		if (output_count == 1) {
			*p_working_mem = *p_inputs[0];
			return STEP_EXIT_FUNCTION_BIT;
		} else if (output_count > 1) {
			Array arr;
			for (int i = 0; i < output_count; i++) {
				arr.append(*p_inputs[i]);
			}
			*p_working_mem = arr;
			return STEP_EXIT_FUNCTION_BIT;
		} else {
			*p_working_mem = Variant();
			return 0;
		}
	}
};

VisualScriptNodeInstance *VisualScriptModuleExitNode::instantiate(VisualScriptInstance *p_instance) {
	VisualScriptModuleExitNodeInstance *instance = memnew(VisualScriptModuleExitNodeInstance);
	instance->node = this;
	instance->instance = p_instance;
	instance->output_count = get_input_value_port_count();
	return instance;
}

VisualScriptModuleExitNode::VisualScriptModuleExitNode() {}

VisualScriptModuleExitNode::~VisualScriptModuleExitNode() {}
