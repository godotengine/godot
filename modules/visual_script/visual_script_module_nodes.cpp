/*************************************************************************/
/*  visual_script_module_nodes.cpp                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

	// TODO: Try to figure out how to do something to show inspector values based on visual script context....
	// Ref<VisualScript> vs = get_container();
	// String modules = "None";
	// if (vs.is_valid()) {
	// 	List<StringName> mods;
	// 	vs->get_module_list(&mods);
	// 	for (const List<StringName>::Element *E = mods.front(); E; E = E->next()) {
	// 		modules += ", " + E->get();
	// 	}
	// }

	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "module_name"), "set_module", "get_module_name");
}

int VisualScriptModuleNode::get_output_sequence_port_count() const {
	return 1; // don't need much more
}
bool VisualScriptModuleNode::has_input_sequence_port() const {
	return true; // change if required
}
String VisualScriptModuleNode::get_output_sequence_port_text(int p_port) const {
	return ""; // not needed
}
int VisualScriptModuleNode::get_input_value_port_count() const {
	if (module_name == "" || get_container().is_null()) {
		return 0;
	}
	Ref<VisualScript> vs = get_container();
	ERR_FAIL_COND_V(!vs.is_valid(), 0);

	if (!vs->has_module(module_name)) {
		return 0;
	}

	Ref<VisualScriptModule> module = vs->get_module(module_name);
	ERR_FAIL_COND_V(!module.is_valid(), 0);

	Ref<VisualScriptNode> entry_node = module->get_node(0);
	ERR_FAIL_COND_V(!entry_node.is_valid(), 0);

	return entry_node->get_output_value_port_count();
}

int VisualScriptModuleNode::get_output_value_port_count() const {
	if (module_name == "" || get_container().is_null()) {
		return 0;
	}
	Ref<VisualScript> vs = get_container();
	ERR_FAIL_COND_V(!vs.is_valid(), 0);

	Ref<VisualScriptModule> module = vs->get_module(module_name);
	ERR_FAIL_COND_V(!module.is_valid(), 0);

	Ref<VisualScriptNode> exit_node = module->get_node(1);
	ERR_FAIL_COND_V(!exit_node.is_valid(), 0);

	return exit_node->get_input_value_port_count();
}

PropertyInfo VisualScriptModuleNode::get_input_value_port_info(int p_idx) const {
	if (module_name == "" || get_container().is_null()) {
		return PropertyInfo();
	}
	Ref<VisualScript> vs = get_container();
	ERR_FAIL_COND_V(!vs.is_valid(), PropertyInfo());

	if (!vs->has_module(module_name)) {
		return PropertyInfo();
	}

	Ref<VisualScriptModule> module = vs->get_module(module_name);
	ERR_FAIL_COND_V(!module.is_valid(), PropertyInfo());

	Ref<VisualScriptNode> entry_node = module->get_node(0);
	ERR_FAIL_COND_V(!entry_node.is_valid(), PropertyInfo());

	ERR_FAIL_INDEX_V(p_idx, entry_node->get_output_value_port_count(), PropertyInfo());
	return entry_node->get_output_value_port_info(p_idx);
}
PropertyInfo VisualScriptModuleNode::get_output_value_port_info(int p_idx) const {
	if (module_name == "" || get_container().is_null()) {
		return PropertyInfo();
	}
	Ref<VisualScript> vs = get_container();
	ERR_FAIL_COND_V(!vs.is_valid(), PropertyInfo());

	Ref<VisualScriptModule> module = vs->get_module(module_name);
	ERR_FAIL_COND_V(!module.is_valid(), PropertyInfo());

	Ref<VisualScriptNode> exit_node = module->get_node(1);
	ERR_FAIL_COND_V(!exit_node.is_valid(), PropertyInfo());

	ERR_FAIL_INDEX_V(p_idx, exit_node->get_input_value_port_count(), PropertyInfo());
	return exit_node->get_input_value_port_info(p_idx);
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
	_change_notify();
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
		// not sure if I want to add the module call here
		// TODO: call module execution function
		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptModuleNode::instance(VisualScriptInstance *p_instance) {
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

VisualScriptNodeInstance *VisualScriptModuleEntryNode::instance(VisualScriptInstance *p_instance) {
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
	_change_notify();
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
	//VisualScriptModuleNode *node;
	VisualScriptInstance *instance;
	bool with_value;

	virtual int get_working_memory_size() const { return 1; }
	//virtual bool is_output_port_unsequenced(int p_idx) const { return false; }
	//virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return true; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Callable::CallError &r_error, String &r_error_str) {
		if (with_value) {
			*p_working_mem = *p_inputs[0];
			return STEP_EXIT_FUNCTION_BIT;
		} else {
			*p_working_mem = Variant();
			return 0;
		}
	}
};

VisualScriptNodeInstance *VisualScriptModuleExitNode::instance(VisualScriptInstance *p_instance) {
	VisualScriptModuleExitNodeInstance *instance = memnew(VisualScriptModuleExitNodeInstance);
	//instance->node = this;
	instance->instance = p_instance;
	instance->with_value = get_input_value_port_count() > 0 ? true : false;
	return instance;
}

VisualScriptModuleExitNode::VisualScriptModuleExitNode() {}

VisualScriptModuleExitNode::~VisualScriptModuleExitNode() {}
