/*************************************************************************/
/*  visual_script_flow_control.cpp                                       */
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

#include "visual_script_flow_control.h"

#include "core/config/project_settings.h"
#include "core/io/resource_loader.h"
#include "core/os/keyboard.h"

//////////////////////////////////////////
////////////////RETURN////////////////////
//////////////////////////////////////////

int VisualScriptReturn::get_output_sequence_port_count() const {
	return 0;
}

bool VisualScriptReturn::has_input_sequence_port() const {
	return true;
}

int VisualScriptReturn::get_input_value_port_count() const {
	return with_value ? 1 : 0;
}

int VisualScriptReturn::get_output_value_port_count() const {
	return 0;
}

String VisualScriptReturn::get_output_sequence_port_text(int p_port) const {
	return String();
}

PropertyInfo VisualScriptReturn::get_input_value_port_info(int p_idx) const {
	PropertyInfo pinfo;
	pinfo.name = "result";
	pinfo.type = type;
	return pinfo;
}

PropertyInfo VisualScriptReturn::get_output_value_port_info(int p_idx) const {
	return PropertyInfo();
}

String VisualScriptReturn::get_caption() const {
	return "Return";
}

String VisualScriptReturn::get_text() const {
	return get_name();
}

void VisualScriptReturn::set_return_type(Variant::Type p_type) {
	if (type == p_type) {
		return;
	}
	type = p_type;
	ports_changed_notify();
}

Variant::Type VisualScriptReturn::get_return_type() const {
	return type;
}

void VisualScriptReturn::set_enable_return_value(bool p_enable) {
	if (with_value == p_enable) {
		return;
	}

	with_value = p_enable;
	ports_changed_notify();
}

bool VisualScriptReturn::is_return_value_enabled() const {
	return with_value;
}

void VisualScriptReturn::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_return_type", "type"), &VisualScriptReturn::set_return_type);
	ClassDB::bind_method(D_METHOD("get_return_type"), &VisualScriptReturn::get_return_type);
	ClassDB::bind_method(D_METHOD("set_enable_return_value", "enable"), &VisualScriptReturn::set_enable_return_value);
	ClassDB::bind_method(D_METHOD("is_return_value_enabled"), &VisualScriptReturn::is_return_value_enabled);

	String argt = "Any";
	for (int i = 1; i < Variant::VARIANT_MAX; i++) {
		argt += "," + Variant::get_type_name(Variant::Type(i));
	}

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "return_enabled"), "set_enable_return_value", "is_return_value_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "return_type", PROPERTY_HINT_ENUM, argt), "set_return_type", "get_return_type");
}

class VisualScriptNodeInstanceReturn : public VisualScriptNodeInstance {
public:
	VisualScriptReturn *node;
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

VisualScriptNodeInstance *VisualScriptReturn::instantiate(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceReturn *instance = memnew(VisualScriptNodeInstanceReturn);
	instance->node = this;
	instance->instance = p_instance;
	instance->with_value = with_value;
	return instance;
}

VisualScriptReturn::VisualScriptReturn() {
	with_value = false;
	type = Variant::NIL;
}

template <bool with_value>
static Ref<VisualScriptNode> create_return_node(const String &p_name) {
	Ref<VisualScriptReturn> node;
	node.instantiate();
	node->set_enable_return_value(with_value);
	return node;
}

//////////////////////////////////////////
////////////////CONDITION/////////////////
//////////////////////////////////////////

int VisualScriptCondition::get_output_sequence_port_count() const {
	return 3;
}

bool VisualScriptCondition::has_input_sequence_port() const {
	return true;
}

int VisualScriptCondition::get_input_value_port_count() const {
	return 1;
}

int VisualScriptCondition::get_output_value_port_count() const {
	return 0;
}

String VisualScriptCondition::get_output_sequence_port_text(int p_port) const {
	if (p_port == 0) {
		return "true";
	} else if (p_port == 1) {
		return "false";
	} else {
		return "done";
	}
}

PropertyInfo VisualScriptCondition::get_input_value_port_info(int p_idx) const {
	PropertyInfo pinfo;
	pinfo.name = "cond";
	pinfo.type = Variant::BOOL;
	return pinfo;
}

PropertyInfo VisualScriptCondition::get_output_value_port_info(int p_idx) const {
	return PropertyInfo();
}

String VisualScriptCondition::get_caption() const {
	return "Condition";
}

String VisualScriptCondition::get_text() const {
	return "if (cond) is:  ";
}

void VisualScriptCondition::_bind_methods() {
}

class VisualScriptNodeInstanceCondition : public VisualScriptNodeInstance {
public:
	VisualScriptCondition *node;
	VisualScriptInstance *instance;

	//virtual int get_working_memory_size() const { return 1; }
	//virtual bool is_output_port_unsequenced(int p_idx) const { return false; }
	//virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return true; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Callable::CallError &r_error, String &r_error_str) {
		if (p_start_mode == START_MODE_CONTINUE_SEQUENCE) {
			return 2;
		} else if (p_inputs[0]->operator bool()) {
			return 0 | STEP_FLAG_PUSH_STACK_BIT;
		} else {
			return 1 | STEP_FLAG_PUSH_STACK_BIT;
		}
	}
};

VisualScriptNodeInstance *VisualScriptCondition::instantiate(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceCondition *instance = memnew(VisualScriptNodeInstanceCondition);
	instance->node = this;
	instance->instance = p_instance;
	return instance;
}

VisualScriptCondition::VisualScriptCondition() {
}

//////////////////////////////////////////
////////////////WHILE/////////////////
//////////////////////////////////////////

int VisualScriptWhile::get_output_sequence_port_count() const {
	return 2;
}

bool VisualScriptWhile::has_input_sequence_port() const {
	return true;
}

int VisualScriptWhile::get_input_value_port_count() const {
	return 1;
}

int VisualScriptWhile::get_output_value_port_count() const {
	return 0;
}

String VisualScriptWhile::get_output_sequence_port_text(int p_port) const {
	if (p_port == 0) {
		return "repeat";
	} else {
		return "exit";
	}
}

PropertyInfo VisualScriptWhile::get_input_value_port_info(int p_idx) const {
	PropertyInfo pinfo;
	pinfo.name = "cond";
	pinfo.type = Variant::BOOL;
	return pinfo;
}

PropertyInfo VisualScriptWhile::get_output_value_port_info(int p_idx) const {
	return PropertyInfo();
}

String VisualScriptWhile::get_caption() const {
	return "While";
}

String VisualScriptWhile::get_text() const {
	return "while (cond): ";
}

void VisualScriptWhile::_bind_methods() {
}

class VisualScriptNodeInstanceWhile : public VisualScriptNodeInstance {
public:
	VisualScriptWhile *node;
	VisualScriptInstance *instance;

	//virtual int get_working_memory_size() const { return 1; }
	//virtual bool is_output_port_unsequenced(int p_idx) const { return false; }
	//virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return true; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Callable::CallError &r_error, String &r_error_str) {
		bool keep_going = p_inputs[0]->operator bool();

		if (keep_going) {
			return 0 | STEP_FLAG_PUSH_STACK_BIT;
		} else {
			return 1;
		}
	}
};

VisualScriptNodeInstance *VisualScriptWhile::instantiate(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceWhile *instance = memnew(VisualScriptNodeInstanceWhile);
	instance->node = this;
	instance->instance = p_instance;
	return instance;
}

VisualScriptWhile::VisualScriptWhile() {
}

//////////////////////////////////////////
////////////////ITERATOR/////////////////
//////////////////////////////////////////

int VisualScriptIterator::get_output_sequence_port_count() const {
	return 2;
}

bool VisualScriptIterator::has_input_sequence_port() const {
	return true;
}

int VisualScriptIterator::get_input_value_port_count() const {
	return 1;
}

int VisualScriptIterator::get_output_value_port_count() const {
	return 1;
}

String VisualScriptIterator::get_output_sequence_port_text(int p_port) const {
	if (p_port == 0) {
		return "each";
	} else {
		return "exit";
	}
}

PropertyInfo VisualScriptIterator::get_input_value_port_info(int p_idx) const {
	PropertyInfo pinfo;
	pinfo.name = "input";
	pinfo.type = Variant::NIL;
	return pinfo;
}

PropertyInfo VisualScriptIterator::get_output_value_port_info(int p_idx) const {
	PropertyInfo pinfo;
	pinfo.name = "elem";
	pinfo.type = Variant::NIL;
	return pinfo;
}

String VisualScriptIterator::get_caption() const {
	return "Iterator";
}

String VisualScriptIterator::get_text() const {
	return "for (elem) in (input): ";
}

void VisualScriptIterator::_bind_methods() {
}

class VisualScriptNodeInstanceIterator : public VisualScriptNodeInstance {
public:
	VisualScriptIterator *node;
	VisualScriptInstance *instance;

	virtual int get_working_memory_size() const { return 2; }
	//virtual bool is_output_port_unsequenced(int p_idx) const { return false; }
	//virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return true; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Callable::CallError &r_error, String &r_error_str) {
		if (p_start_mode == START_MODE_BEGIN_SEQUENCE) {
			p_working_mem[0] = *p_inputs[0];
			bool valid;
			bool can_iter = p_inputs[0]->iter_init(p_working_mem[1], valid);

			if (!valid) {
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
				r_error_str = RTR("Input type not iterable: ") + Variant::get_type_name(p_inputs[0]->get_type());
				return 0;
			}

			if (!can_iter) {
				return 1; //nothing to iterate
			}

			*p_outputs[0] = p_working_mem[0].iter_get(p_working_mem[1], valid);

			if (!valid) {
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
				r_error_str = RTR("Iterator became invalid");
				return 0;
			}

		} else { //continue sequence

			bool valid;
			bool can_iter = p_working_mem[0].iter_next(p_working_mem[1], valid);

			if (!valid) {
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
				r_error_str = RTR("Iterator became invalid: ") + Variant::get_type_name(p_inputs[0]->get_type());
				return 0;
			}

			if (!can_iter) {
				return 1; //nothing to iterate
			}

			*p_outputs[0] = p_working_mem[0].iter_get(p_working_mem[1], valid);

			if (!valid) {
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
				r_error_str = RTR("Iterator became invalid");
				return 0;
			}
		}

		return 0 | STEP_FLAG_PUSH_STACK_BIT; //go around
	}
};

VisualScriptNodeInstance *VisualScriptIterator::instantiate(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceIterator *instance = memnew(VisualScriptNodeInstanceIterator);
	instance->node = this;
	instance->instance = p_instance;
	return instance;
}

VisualScriptIterator::VisualScriptIterator() {
}

//////////////////////////////////////////
////////////////SEQUENCE/////////////////
//////////////////////////////////////////

int VisualScriptSequence::get_output_sequence_port_count() const {
	return steps;
}

bool VisualScriptSequence::has_input_sequence_port() const {
	return true;
}

int VisualScriptSequence::get_input_value_port_count() const {
	return 0;
}

int VisualScriptSequence::get_output_value_port_count() const {
	return 1;
}

String VisualScriptSequence::get_output_sequence_port_text(int p_port) const {
	return itos(p_port + 1);
}

PropertyInfo VisualScriptSequence::get_input_value_port_info(int p_idx) const {
	return PropertyInfo();
}

PropertyInfo VisualScriptSequence::get_output_value_port_info(int p_idx) const {
	return PropertyInfo(Variant::INT, "current");
}

String VisualScriptSequence::get_caption() const {
	return "Sequence";
}

String VisualScriptSequence::get_text() const {
	return "in order: ";
}

void VisualScriptSequence::set_steps(int p_steps) {
	ERR_FAIL_COND(p_steps < 1);
	if (steps == p_steps) {
		return;
	}

	steps = p_steps;
	ports_changed_notify();
}

int VisualScriptSequence::get_steps() const {
	return steps;
}

void VisualScriptSequence::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_steps", "steps"), &VisualScriptSequence::set_steps);
	ClassDB::bind_method(D_METHOD("get_steps"), &VisualScriptSequence::get_steps);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "steps", PROPERTY_HINT_RANGE, "1,64,1"), "set_steps", "get_steps");
}

class VisualScriptNodeInstanceSequence : public VisualScriptNodeInstance {
public:
	VisualScriptSequence *node;
	VisualScriptInstance *instance;
	int steps;

	virtual int get_working_memory_size() const { return 1; }
	//virtual bool is_output_port_unsequenced(int p_idx) const { return false; }
	//virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return true; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Callable::CallError &r_error, String &r_error_str) {
		if (p_start_mode == START_MODE_BEGIN_SEQUENCE) {
			p_working_mem[0] = 0;
		}

		int step = p_working_mem[0];

		*p_outputs[0] = step;

		if (step + 1 == steps) {
			return step;
		} else {
			p_working_mem[0] = step + 1;
			return step | STEP_FLAG_PUSH_STACK_BIT;
		}
	}
};

VisualScriptNodeInstance *VisualScriptSequence::instantiate(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceSequence *instance = memnew(VisualScriptNodeInstanceSequence);
	instance->node = this;
	instance->instance = p_instance;
	instance->steps = steps;
	return instance;
}

VisualScriptSequence::VisualScriptSequence() {
	steps = 1;
}

//////////////////////////////////////////
////////////////EVENT TYPE FILTER///////////
//////////////////////////////////////////

int VisualScriptSwitch::get_output_sequence_port_count() const {
	return case_values.size() + 1;
}

bool VisualScriptSwitch::has_input_sequence_port() const {
	return true;
}

int VisualScriptSwitch::get_input_value_port_count() const {
	return case_values.size() + 1;
}

int VisualScriptSwitch::get_output_value_port_count() const {
	return 0;
}

String VisualScriptSwitch::get_output_sequence_port_text(int p_port) const {
	if (p_port == case_values.size()) {
		return "done";
	}

	return String();
}

PropertyInfo VisualScriptSwitch::get_input_value_port_info(int p_idx) const {
	if (p_idx < case_values.size()) {
		return PropertyInfo(case_values[p_idx].type, " =");
	} else {
		return PropertyInfo(Variant::NIL, "input");
	}
}

PropertyInfo VisualScriptSwitch::get_output_value_port_info(int p_idx) const {
	return PropertyInfo();
}

String VisualScriptSwitch::get_caption() const {
	return "Switch";
}

String VisualScriptSwitch::get_text() const {
	return "'input' is:";
}

class VisualScriptNodeInstanceSwitch : public VisualScriptNodeInstance {
public:
	VisualScriptInstance *instance;
	int case_count;

	//virtual int get_working_memory_size() const { return 0; }
	//virtual bool is_output_port_unsequenced(int p_idx) const { return false; }
	//virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return false; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Callable::CallError &r_error, String &r_error_str) {
		if (p_start_mode == START_MODE_CONTINUE_SEQUENCE) {
			return case_count; //exit
		}

		for (int i = 0; i < case_count; i++) {
			if (*p_inputs[i] == *p_inputs[case_count]) {
				return i | STEP_FLAG_PUSH_STACK_BIT;
			}
		}

		return case_count;
	}
};

VisualScriptNodeInstance *VisualScriptSwitch::instantiate(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceSwitch *instance = memnew(VisualScriptNodeInstanceSwitch);
	instance->instance = p_instance;
	instance->case_count = case_values.size();
	return instance;
}

bool VisualScriptSwitch::_set(const StringName &p_name, const Variant &p_value) {
	if (String(p_name) == "case_count") {
		case_values.resize(p_value);
		notify_property_list_changed();
		ports_changed_notify();
		return true;
	}

	if (String(p_name).begins_with("case/")) {
		int idx = String(p_name).get_slice("/", 1).to_int();
		ERR_FAIL_INDEX_V(idx, case_values.size(), false);

		case_values.write[idx].type = Variant::Type(int(p_value));
		notify_property_list_changed();
		ports_changed_notify();

		return true;
	}

	return false;
}

bool VisualScriptSwitch::_get(const StringName &p_name, Variant &r_ret) const {
	if (String(p_name) == "case_count") {
		r_ret = case_values.size();
		return true;
	}

	if (String(p_name).begins_with("case/")) {
		int idx = String(p_name).get_slice("/", 1).to_int();
		ERR_FAIL_INDEX_V(idx, case_values.size(), false);

		r_ret = case_values[idx].type;
		return true;
	}

	return false;
}

void VisualScriptSwitch::_get_property_list(List<PropertyInfo> *p_list) const {
	p_list->push_back(PropertyInfo(Variant::INT, "case_count", PROPERTY_HINT_RANGE, "0,128"));

	String argt = "Any";
	for (int i = 1; i < Variant::VARIANT_MAX; i++) {
		argt += "," + Variant::get_type_name(Variant::Type(i));
	}

	for (int i = 0; i < case_values.size(); i++) {
		p_list->push_back(PropertyInfo(Variant::INT, "case/" + itos(i), PROPERTY_HINT_ENUM, argt));
	}
}

void VisualScriptSwitch::reset_state() {
	case_values.clear();
}

void VisualScriptSwitch::_bind_methods() {
}

VisualScriptSwitch::VisualScriptSwitch() {
}

//////////////////////////////////////////
////////////////TYPE CAST///////////
//////////////////////////////////////////

int VisualScriptTypeCast::get_output_sequence_port_count() const {
	return 2;
}

bool VisualScriptTypeCast::has_input_sequence_port() const {
	return true;
}

int VisualScriptTypeCast::get_input_value_port_count() const {
	return 1;
}

int VisualScriptTypeCast::get_output_value_port_count() const {
	return 1;
}

String VisualScriptTypeCast::get_output_sequence_port_text(int p_port) const {
	return p_port == 0 ? "yes" : "no";
}

PropertyInfo VisualScriptTypeCast::get_input_value_port_info(int p_idx) const {
	return PropertyInfo(Variant::OBJECT, "instance");
}

PropertyInfo VisualScriptTypeCast::get_output_value_port_info(int p_idx) const {
	return PropertyInfo(Variant::OBJECT, "", PROPERTY_HINT_TYPE_STRING, get_base_type());
}

String VisualScriptTypeCast::get_caption() const {
	return "Type Cast";
}

String VisualScriptTypeCast::get_text() const {
	if (!script.is_empty()) {
		return "Is " + script.get_file() + "?";
	} else {
		return "Is " + base_type + "?";
	}
}

void VisualScriptTypeCast::set_base_type(const StringName &p_type) {
	if (base_type == p_type) {
		return;
	}

	base_type = p_type;
	notify_property_list_changed();
	ports_changed_notify();
}

StringName VisualScriptTypeCast::get_base_type() const {
	return base_type;
}

void VisualScriptTypeCast::set_base_script(const String &p_path) {
	if (script == p_path) {
		return;
	}

	script = p_path;
	notify_property_list_changed();
	ports_changed_notify();
}

String VisualScriptTypeCast::get_base_script() const {
	return script;
}

VisualScriptTypeCast::TypeGuess VisualScriptTypeCast::guess_output_type(TypeGuess *p_inputs, int p_output) const {
	TypeGuess tg;
	tg.type = Variant::OBJECT;
	if (!script.is_empty()) {
		tg.script = ResourceLoader::load(script);
	}
	//if (!tg.script.is_valid()) {
	//	tg.gdclass = base_type;
	//}

	return tg;
}

class VisualScriptNodeInstanceTypeCast : public VisualScriptNodeInstance {
public:
	VisualScriptInstance *instance;
	StringName base_type;
	String script;

	//virtual int get_working_memory_size() const { return 0; }
	//virtual bool is_output_port_unsequenced(int p_idx) const { return false; }
	//virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return false; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Callable::CallError &r_error, String &r_error_str) {
		Object *obj = *p_inputs[0];

		*p_outputs[0] = Variant();

		if (!obj) {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
			r_error_str = "Instance is null";
			return 0;
		}

		if (!script.is_empty()) {
			Ref<Script> obj_script = obj->get_script();
			if (!obj_script.is_valid()) {
				return 1; //well, definitely not the script because object we got has no script.
			}

			if (!ResourceCache::has(script)) {
				//if the script is not in use by anyone, we can safely assume whatever we got is not casting to it.
				return 1;
			}
			Ref<Script> cast_script = Ref<Resource>(ResourceCache::get(script));
			if (!cast_script.is_valid()) {
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
				r_error_str = "Script path is not a script: " + script;
				return 1;
			}

			while (obj_script.is_valid()) {
				if (cast_script == obj_script) {
					*p_outputs[0] = *p_inputs[0]; //copy
					return 0; // it is the script, yey
				}

				obj_script = obj_script->get_base_script();
			}

			return 1; //not found sorry
		}

		if (ClassDB::is_parent_class(obj->get_class_name(), base_type)) {
			*p_outputs[0] = *p_inputs[0]; //copy
			return 0;
		} else {
			return 1;
		}
	}
};

VisualScriptNodeInstance *VisualScriptTypeCast::instantiate(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceTypeCast *instance = memnew(VisualScriptNodeInstanceTypeCast);
	instance->instance = p_instance;
	instance->base_type = base_type;
	instance->script = script;
	return instance;
}

void VisualScriptTypeCast::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_base_type", "type"), &VisualScriptTypeCast::set_base_type);
	ClassDB::bind_method(D_METHOD("get_base_type"), &VisualScriptTypeCast::get_base_type);

	ClassDB::bind_method(D_METHOD("set_base_script", "path"), &VisualScriptTypeCast::set_base_script);
	ClassDB::bind_method(D_METHOD("get_base_script"), &VisualScriptTypeCast::get_base_script);

	List<String> script_extensions;
	for (int i = 0; i > ScriptServer::get_language_count(); i++) {
		ScriptServer::get_language(i)->get_recognized_extensions(&script_extensions);
	}

	String script_ext_hint;
	for (const String &E : script_extensions) {
		if (!script_ext_hint.is_empty()) {
			script_ext_hint += ",";
		}
		script_ext_hint += "*." + E;
	}

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "base_type", PROPERTY_HINT_TYPE_STRING, "Object"), "set_base_type", "get_base_type");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "base_script", PROPERTY_HINT_FILE, script_ext_hint), "set_base_script", "get_base_script");
}

VisualScriptTypeCast::VisualScriptTypeCast() {
	base_type = "Object";
}

void register_visual_script_flow_control_nodes() {
	VisualScriptLanguage::singleton->add_register_func("flow_control/return", create_return_node<false>);
	VisualScriptLanguage::singleton->add_register_func("flow_control/return_with_value", create_return_node<true>);
	VisualScriptLanguage::singleton->add_register_func("flow_control/condition", create_node_generic<VisualScriptCondition>);
	VisualScriptLanguage::singleton->add_register_func("flow_control/while", create_node_generic<VisualScriptWhile>);
	VisualScriptLanguage::singleton->add_register_func("flow_control/iterator", create_node_generic<VisualScriptIterator>);
	VisualScriptLanguage::singleton->add_register_func("flow_control/sequence", create_node_generic<VisualScriptSequence>);
	VisualScriptLanguage::singleton->add_register_func("flow_control/switch", create_node_generic<VisualScriptSwitch>);
	//VisualScriptLanguage::singleton->add_register_func("flow_control/input", create_node_generic<VisualScriptInputFilter>);
	VisualScriptLanguage::singleton->add_register_func("flow_control/type_cast", create_node_generic<VisualScriptTypeCast>);
}
