/*************************************************************************/
/*  visual_script_flow_control.cpp                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "global_config.h"
#include "os/keyboard.h"

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

	if (type == p_type)
		return;
	type = p_type;
	ports_changed_notify();
}

Variant::Type VisualScriptReturn::get_return_type() const {

	return type;
}

void VisualScriptReturn::set_enable_return_value(bool p_enable) {
	if (with_value == p_enable)
		return;

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

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "return_value/enabled"), "set_enable_return_value", "is_return_value_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "return_value/type", PROPERTY_HINT_ENUM, argt), "set_return_type", "get_return_type");
}

class VisualScriptNodeInstanceReturn : public VisualScriptNodeInstance {
public:
	VisualScriptReturn *node;
	VisualScriptInstance *instance;
	bool with_value;

	virtual int get_working_memory_size() const { return 1; }
	//virtual bool is_output_port_unsequenced(int p_idx) const { return false; }
	//virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return true; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {

		if (with_value) {
			*p_working_mem = *p_inputs[0];
		} else {
			*p_working_mem = Variant();
		}

		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptReturn::instance(VisualScriptInstance *p_instance) {

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
	node.instance();
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

	if (p_port == 0)
		return "true";
	else if (p_port == 1)
		return "false";
	else
		return "done";
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

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {

		if (p_start_mode == START_MODE_CONTINUE_SEQUENCE)
			return 2;
		else if (p_inputs[0]->operator bool())
			return 0 | STEP_FLAG_PUSH_STACK_BIT;
		else
			return 1 | STEP_FLAG_PUSH_STACK_BIT;
	}
};

VisualScriptNodeInstance *VisualScriptCondition::instance(VisualScriptInstance *p_instance) {

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

	if (p_port == 0)
		return "repeat";
	else
		return "exit";
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

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {

		bool keep_going = p_inputs[0]->operator bool();

		if (keep_going)
			return 0 | STEP_FLAG_PUSH_STACK_BIT;
		else
			return 1;
	}
};

VisualScriptNodeInstance *VisualScriptWhile::instance(VisualScriptInstance *p_instance) {

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

	if (p_port == 0)
		return "each";
	else
		return "exit";
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

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {

		if (p_start_mode == START_MODE_BEGIN_SEQUENCE) {
			p_working_mem[0] = *p_inputs[0];
			bool valid;
			bool can_iter = p_inputs[0]->iter_init(p_working_mem[1], valid);

			if (!valid) {
				r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
				r_error_str = RTR("Input type not iterable: ") + Variant::get_type_name(p_inputs[0]->get_type());
				return 0;
			}

			if (!can_iter)
				return 1; //nothing to iterate

			*p_outputs[0] = p_working_mem[0].iter_get(p_working_mem[1], valid);

			if (!valid) {
				r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
				r_error_str = RTR("Iterator became invalid");
				return 0;
			}

		} else { //continue sequence

			bool valid;
			bool can_iter = p_working_mem[0].iter_next(p_working_mem[1], valid);

			if (!valid) {
				r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
				r_error_str = RTR("Iterator became invalid: ") + Variant::get_type_name(p_inputs[0]->get_type());
				return 0;
			}

			if (!can_iter)
				return 1; //nothing to iterate

			*p_outputs[0] = p_working_mem[0].iter_get(p_working_mem[1], valid);

			if (!valid) {
				r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
				r_error_str = RTR("Iterator became invalid");
				return 0;
			}
		}

		return 0 | STEP_FLAG_PUSH_STACK_BIT; //go around
	}
};

VisualScriptNodeInstance *VisualScriptIterator::instance(VisualScriptInstance *p_instance) {

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
	if (steps == p_steps)
		return;

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

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {

		if (p_start_mode == START_MODE_BEGIN_SEQUENCE) {

			p_working_mem[0] = 0;
		}

		int step = p_working_mem[0];

		*p_outputs[0] = step;

		if (step + 1 == steps)
			return step;
		else {
			p_working_mem[0] = step + 1;
			return step | STEP_FLAG_PUSH_STACK_BIT;
		}
	}
};

VisualScriptNodeInstance *VisualScriptSequence::instance(VisualScriptInstance *p_instance) {

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

	if (p_port == case_values.size())
		return "done";

	return String();
}

PropertyInfo VisualScriptSwitch::get_input_value_port_info(int p_idx) const {

	if (p_idx < case_values.size()) {
		return PropertyInfo(case_values[p_idx].type, " =");
	} else
		return PropertyInfo(Variant::NIL, "input");
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

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {

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

VisualScriptNodeInstance *VisualScriptSwitch::instance(VisualScriptInstance *p_instance) {

	VisualScriptNodeInstanceSwitch *instance = memnew(VisualScriptNodeInstanceSwitch);
	instance->instance = p_instance;
	instance->case_count = case_values.size();
	return instance;
}

bool VisualScriptSwitch::_set(const StringName &p_name, const Variant &p_value) {

	if (String(p_name) == "case_count") {
		case_values.resize(p_value);
		_change_notify();
		ports_changed_notify();
		return true;
	}

	if (String(p_name).begins_with("case/")) {

		int idx = String(p_name).get_slice("/", 1).to_int();
		ERR_FAIL_INDEX_V(idx, case_values.size(), false);

		case_values[idx].type = Variant::Type(int(p_value));
		_change_notify();
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

void VisualScriptSwitch::_bind_methods() {
}

VisualScriptSwitch::VisualScriptSwitch() {
}

//////////////////////////////////////////
////////////////EVENT ACTION FILTER///////////
//////////////////////////////////////////

int VisualScriptInputFilter::get_output_sequence_port_count() const {

	return filters.size();
}

bool VisualScriptInputFilter::has_input_sequence_port() const {

	return true;
}

int VisualScriptInputFilter::get_input_value_port_count() const {

	return 1;
}
int VisualScriptInputFilter::get_output_value_port_count() const {

	return 1;
}

String VisualScriptInputFilter::get_output_sequence_port_text(int p_port) const {

	String text;

	switch (filters[p_port].type) {
		case InputEvent::NONE: {
			text = "None";
		} break;
		case InputEvent::KEY: {

			InputEventKey k = filters[p_port].key;

			if (k.scancode == 0 && k.unicode == 0) {
				text = "No Key";
			} else {
				if (k.scancode != 0) {
					text = "KeyCode: " + keycode_get_string(k.scancode);
				} else if (k.unicode != 0) {
					text = "Uniode: " + String::chr(k.unicode);
				}

				if (k.pressed)
					text += ", Pressed";
				else
					text += ", Released";

				if (k.echo)
					text += ", Echo";
				if (k.mod.alt)
					text = "Alt+" + text;
				if (k.mod.shift)
					text = "Shift+" + text;
				if (k.mod.control)
					text = "Ctrl+" + text;
				if (k.mod.meta)
					text = "Meta+" + text;
			}

		} break;
		case InputEvent::MOUSE_MOTION: {
			InputEventMouseMotion mm = filters[p_port].mouse_motion;
			text = "Mouse Motion";

			String b = "Left,Right,Middle,WheelUp,WheelDown,WheelLeft,WheelRight";

			for (int i = 0; i < 7; i++) {
				if (mm.button_mask & (1 << i)) {
					text = b.get_slice(",", i) + "+" + text;
				}
			}
			if (mm.mod.alt)
				text = "Alt+" + text;
			if (mm.mod.shift)
				text = "Shift+" + text;
			if (mm.mod.control)
				text = "Ctrl+" + text;
			if (mm.mod.meta)
				text = "Meta+" + text;
		} break;
		case InputEvent::MOUSE_BUTTON: {

			InputEventMouseButton mb = filters[p_port].mouse_button;

			String b = "Any,Left,Right,Middle,WheelUp,WheelDown,WheelLeft,WheelRight";

			text = b.get_slice(",", mb.button_index) + " Mouse Button";

			if (mb.pressed)
				text += ", Pressed";
			else
				text += ", Released";

			if (mb.doubleclick)
				text += ", DblClick";
			if (mb.mod.alt)
				text = "Alt+" + text;
			if (mb.mod.shift)
				text = "Shift+" + text;
			if (mb.mod.control)
				text = "Ctrl+" + text;
			if (mb.mod.meta)
				text = "Meta+" + text;

		} break;
		case InputEvent::JOYPAD_MOTION: {

			InputEventJoypadMotion jm = filters[p_port].joy_motion;

			text = "JoyMotion Axis " + itos(jm.axis >> 1);
			if (jm.axis & 1)
				text += " > " + rtos(jm.axis_value);
			else
				text += " < " + rtos(-jm.axis_value);

		} break;
		case InputEvent::JOYPAD_BUTTON: {
			InputEventJoypadButton jb = filters[p_port].joy_button;

			text = "JoyButton " + itos(jb.button_index);
			if (jb.pressed)
				text += ", Pressed";
			else
				text += ", Released";
		} break;
		case InputEvent::SCREEN_TOUCH: {
			InputEventScreenTouch sd = filters[p_port].screen_touch;

			text = "Touch Finger " + itos(sd.index);
			if (sd.pressed)
				text += ", Pressed";
			else
				text += ", Released";
		} break;
		case InputEvent::SCREEN_DRAG: {
			InputEventScreenDrag sd = filters[p_port].screen_drag;
			text = "Drag Finger " + itos(sd.index);
		} break;
		case InputEvent::ACTION: {

			List<PropertyInfo> pinfo;
			GlobalConfig::get_singleton()->get_property_list(&pinfo);
			int index = 1;

			text = "No Action";
			for (List<PropertyInfo>::Element *E = pinfo.front(); E; E = E->next()) {
				const PropertyInfo &pi = E->get();

				if (!pi.name.begins_with("input/"))
					continue;

				if (filters[p_port].action.action == index) {
					text = "Action " + pi.name.substr(pi.name.find("/") + 1, pi.name.length());
					break;
				}
				index++;
			}

			if (filters[p_port].action.pressed)
				text += ", Pressed";
			else
				text += ", Released";

		} break;
	}

	return text + " - " + itos(p_port);
}

PropertyInfo VisualScriptInputFilter::get_input_value_port_info(int p_idx) const {

	return PropertyInfo(Variant::INPUT_EVENT, "event");
}

PropertyInfo VisualScriptInputFilter::get_output_value_port_info(int p_idx) const {

	return PropertyInfo(Variant::INPUT_EVENT, "");
}

String VisualScriptInputFilter::get_caption() const {

	return "InputFilter";
}

String VisualScriptInputFilter::get_text() const {

	return "";
}

bool VisualScriptInputFilter::_set(const StringName &p_name, const Variant &p_value) {

	if (p_name == "filter_count") {
		filters.resize(p_value);
		_change_notify();
		ports_changed_notify();
		return true;
	}

	if (String(p_name).begins_with("filter_")) {

		int idx = String(p_name).replace_first("filters_", "").get_slice("/", 0).to_int();

		ERR_FAIL_INDEX_V(idx, filters.size(), false);

		String what = String(p_name).get_slice("/", 1);

		if (what == "type") {
			filters[idx] = InputEvent();
			filters[idx].type = InputEvent::Type(int(p_value));
			if (filters[idx].type == InputEvent::JOYPAD_MOTION) {
				filters[idx].joy_motion.axis_value = 0.5; //for treshold
			} else if (filters[idx].type == InputEvent::KEY) {
				filters[idx].key.pressed = true; //put these as true to make it more user friendly
			} else if (filters[idx].type == InputEvent::MOUSE_BUTTON) {
				filters[idx].mouse_button.pressed = true;
			} else if (filters[idx].type == InputEvent::JOYPAD_BUTTON) {
				filters[idx].joy_button.pressed = true;
			} else if (filters[idx].type == InputEvent::SCREEN_TOUCH) {
				filters[idx].screen_touch.pressed = true;
			} else if (filters[idx].type == InputEvent::ACTION) {
				filters[idx].action.pressed = true;
			}
			_change_notify();
			ports_changed_notify();

			return true;
		}
		if (what == "device") {
			filters[idx].device = p_value;
			ports_changed_notify();
			return true;
		}

		switch (filters[idx].type) {

			case InputEvent::KEY: {

				if (what == "scancode") {
					String sc = p_value;
					if (sc == String()) {
						filters[idx].key.scancode = 0;
					} else {
						filters[idx].key.scancode = find_keycode(p_value);
					}

				} else if (what == "unicode") {

					String uc = p_value;

					if (uc == String()) {
						filters[idx].key.unicode = 0;
					} else {
						filters[idx].key.unicode = uc[0];
					}

				} else if (what == "pressed") {

					filters[idx].key.pressed = p_value;
				} else if (what == "echo") {

					filters[idx].key.echo = p_value;

				} else if (what == "mod_alt") {
					filters[idx].key.mod.alt = p_value;

				} else if (what == "mod_shift") {
					filters[idx].key.mod.shift = p_value;

				} else if (what == "mod_ctrl") {
					filters[idx].key.mod.control = p_value;

				} else if (what == "mod_meta") {
					filters[idx].key.mod.meta = p_value;
				} else {
					return false;
				}
				ports_changed_notify();

				return true;
			} break;
			case InputEvent::MOUSE_MOTION: {

				if (what == "button_mask") {
					filters[idx].mouse_motion.button_mask = p_value;

				} else if (what == "mod_alt") {
					filters[idx].mouse_motion.mod.alt = p_value;

				} else if (what == "mod_shift") {
					filters[idx].mouse_motion.mod.shift = p_value;

				} else if (what == "mod_ctrl") {
					filters[idx].mouse_motion.mod.control = p_value;

				} else if (what == "mod_meta") {
					filters[idx].mouse_motion.mod.meta = p_value;
				} else {
					return false;
				}

				ports_changed_notify();
				return true;

			} break;
			case InputEvent::MOUSE_BUTTON: {

				if (what == "button_index") {
					filters[idx].mouse_button.button_index = p_value;
				} else if (what == "pressed") {
					filters[idx].mouse_button.pressed = p_value;
				} else if (what == "doubleclicked") {
					filters[idx].mouse_button.doubleclick = p_value;

				} else if (what == "mod_alt") {
					filters[idx].mouse_button.mod.alt = p_value;

				} else if (what == "mod_shift") {
					filters[idx].mouse_button.mod.shift = p_value;

				} else if (what == "mod_ctrl") {
					filters[idx].mouse_button.mod.control = p_value;

				} else if (what == "mod_meta") {
					filters[idx].mouse_button.mod.meta = p_value;
				} else {
					return false;
				}
				ports_changed_notify();
				return true;

			} break;
			case InputEvent::JOYPAD_MOTION: {

				if (what == "axis") {
					filters[idx].joy_motion.axis = int(p_value) << 1 | filters[idx].joy_motion.axis;
				} else if (what == "mode") {
					filters[idx].joy_motion.axis |= int(p_value);
				} else if (what == "treshold") {
					filters[idx].joy_motion.axis_value = p_value;
				} else {
					return false;
				}
				ports_changed_notify();
				return true;

			} break;
			case InputEvent::JOYPAD_BUTTON: {

				if (what == "button_index") {
					filters[idx].joy_button.button_index = p_value;
				} else if (what == "pressed") {
					filters[idx].joy_button.pressed = p_value;
				} else {
					return false;
				}
				ports_changed_notify();
				return true;

			} break;
			case InputEvent::SCREEN_TOUCH: {

				if (what == "finger_index") {
					filters[idx].screen_touch.index = p_value;
				} else if (what == "pressed") {
					filters[idx].screen_touch.pressed = p_value;
				} else {
					return false;
				}
				ports_changed_notify();
				return true;
			} break;
			case InputEvent::SCREEN_DRAG: {
				if (what == "finger_index") {
					filters[idx].screen_drag.index = p_value;
				} else {
					return false;
				}
				ports_changed_notify();
				return true;
			} break;
			case InputEvent::ACTION: {

				if (what == "action_name") {

					List<PropertyInfo> pinfo;
					GlobalConfig::get_singleton()->get_property_list(&pinfo);
					int index = 1;

					for (List<PropertyInfo>::Element *E = pinfo.front(); E; E = E->next()) {
						const PropertyInfo &pi = E->get();

						if (!pi.name.begins_with("input/"))
							continue;

						String name = pi.name.substr(pi.name.find("/") + 1, pi.name.length());
						if (name == String(p_value)) {

							filters[idx].action.action = index;
							ports_changed_notify();
							return true;
						}

						index++;
					}

					filters[idx].action.action = 0;
					ports_changed_notify();

					return false;

				} else if (what == "pressed") {

					filters[idx].action.pressed = p_value;
					ports_changed_notify();
					return true;
				}

			} break;
		}
	}
	return false;
}

bool VisualScriptInputFilter::_get(const StringName &p_name, Variant &r_ret) const {

	if (p_name == "filter_count") {
		r_ret = filters.size();
		return true;
	}

	if (String(p_name).begins_with("filter_")) {

		int idx = String(p_name).replace_first("filters_", "").get_slice("/", 0).to_int();

		ERR_FAIL_INDEX_V(idx, filters.size(), false);

		String what = String(p_name).get_slice("/", 1);

		if (what == "type") {
			r_ret = filters[idx].type;
			return true;
		}
		if (what == "device") {
			r_ret = filters[idx].device;
			return true;
		}

		switch (filters[idx].type) {

			case InputEvent::KEY: {

				if (what == "scancode") {
					if (filters[idx].key.scancode == 0)
						r_ret = String();
					else {

						r_ret = keycode_get_string(filters[idx].key.scancode);
					}

				} else if (what == "unicode") {

					if (filters[idx].key.unicode == 0) {
						r_ret = String();
					} else {
						CharType str[2] = { (CharType)filters[idx].key.unicode, 0 };
						r_ret = String(str);
					}

				} else if (what == "pressed") {

					r_ret = filters[idx].key.pressed;
				} else if (what == "echo") {

					r_ret = filters[idx].key.echo;

				} else if (what == "mod_alt") {
					r_ret = filters[idx].key.mod.alt;

				} else if (what == "mod_shift") {
					r_ret = filters[idx].key.mod.shift;

				} else if (what == "mod_ctrl") {
					r_ret = filters[idx].key.mod.control;

				} else if (what == "mod_meta") {
					r_ret = filters[idx].key.mod.meta;
				} else {
					return false;
				}

				return true;
			} break;
			case InputEvent::MOUSE_MOTION: {

				if (what == "button_mask") {
					r_ret = filters[idx].mouse_motion.button_mask;

				} else if (what == "mod_alt") {
					r_ret = filters[idx].mouse_motion.mod.alt;

				} else if (what == "mod_shift") {
					r_ret = filters[idx].mouse_motion.mod.shift;

				} else if (what == "mod_ctrl") {
					r_ret = filters[idx].mouse_motion.mod.control;

				} else if (what == "mod_meta") {
					r_ret = filters[idx].mouse_motion.mod.meta;
				} else {
					return false;
				}

				return true;

			} break;
			case InputEvent::MOUSE_BUTTON: {

				if (what == "button_index") {
					r_ret = filters[idx].mouse_button.button_index;
				} else if (what == "pressed") {
					r_ret = filters[idx].mouse_button.pressed;
				} else if (what == "doubleclicked") {
					r_ret = filters[idx].mouse_button.doubleclick;

				} else if (what == "mod_alt") {
					r_ret = filters[idx].mouse_button.mod.alt;

				} else if (what == "mod_shift") {
					r_ret = filters[idx].mouse_button.mod.shift;

				} else if (what == "mod_ctrl") {
					r_ret = filters[idx].mouse_button.mod.control;

				} else if (what == "mod_meta") {
					r_ret = filters[idx].mouse_button.mod.meta;
				} else {
					return false;
				}
				return true;

			} break;
			case InputEvent::JOYPAD_MOTION: {

				if (what == "axis_index") {
					r_ret = filters[idx].joy_motion.axis >> 1;
				} else if (what == "mode") {
					r_ret = filters[idx].joy_motion.axis & 1;
				} else if (what == "treshold") {
					r_ret = filters[idx].joy_motion.axis_value;
				} else {
					return false;
				}
				return true;

			} break;
			case InputEvent::JOYPAD_BUTTON: {

				if (what == "button_index") {
					r_ret = filters[idx].joy_button.button_index;
				} else if (what == "pressed") {
					r_ret = filters[idx].joy_button.pressed;
				} else {
					return false;
				}
				return true;

			} break;
			case InputEvent::SCREEN_TOUCH: {

				if (what == "finger_index") {
					r_ret = filters[idx].screen_touch.index;
				} else if (what == "pressed") {
					r_ret = filters[idx].screen_touch.pressed;
				} else {
					return false;
				}
				return true;
			} break;
			case InputEvent::SCREEN_DRAG: {
				if (what == "finger_index") {
					r_ret = filters[idx].screen_drag.index;
				} else {
					return false;
				}
				return true;
			} break;
			case InputEvent::ACTION: {

				if (what == "action_name") {

					List<PropertyInfo> pinfo;
					GlobalConfig::get_singleton()->get_property_list(&pinfo);
					int index = 1;

					for (List<PropertyInfo>::Element *E = pinfo.front(); E; E = E->next()) {
						const PropertyInfo &pi = E->get();

						if (!pi.name.begins_with("input/"))
							continue;

						if (filters[idx].action.action == index) {
							r_ret = pi.name.substr(pi.name.find("/") + 1, pi.name.length());
							return true;
						}
						index++;
					}

					r_ret = "None"; //no index
					return false;

				} else if (what == "pressed") {

					r_ret = filters[idx].action.pressed;
					return true;
				}

			} break;
		}
	}
	return false;
}

static const char *event_type_names[InputEvent::TYPE_MAX] = {
	"None",
	"Key",
	"MouseMotion",
	"MouseButton",
	"JoypadMotion",
	"JoypadButton",
	"ScreenTouch",
	"ScreenDrag",
	"Action"
};

void VisualScriptInputFilter::_get_property_list(List<PropertyInfo> *p_list) const {

	p_list->push_back(PropertyInfo(Variant::INT, "filter_count", PROPERTY_HINT_RANGE, "0,64"));

	String et;
	for (int i = 0; i < InputEvent::TYPE_MAX; i++) {
		if (i > 0)
			et += ",";

		et += event_type_names[i];
	}

	String kc;
	String actions;

	for (int i = 0; i < filters.size(); i++) {

		String base = "filter_" + itos(i) + "/";
		p_list->push_back(PropertyInfo(Variant::INT, base + "type", PROPERTY_HINT_ENUM, et));
		p_list->push_back(PropertyInfo(Variant::INT, base + "device"));
		switch (filters[i].type) {

			case InputEvent::NONE: {

			} break;
			case InputEvent::KEY: {
				if (kc == String()) {
					int kcc = keycode_get_count();
					kc = "None";
					for (int i = 0; i < kcc; i++) {
						kc += ",";
						kc += String(keycode_get_name_by_index(i));
					}
				}
				p_list->push_back(PropertyInfo(Variant::STRING, base + "scancode", PROPERTY_HINT_ENUM, kc));
				p_list->push_back(PropertyInfo(Variant::STRING, base + "unicode"));
				p_list->push_back(PropertyInfo(Variant::BOOL, base + "pressed"));
				p_list->push_back(PropertyInfo(Variant::BOOL, base + "echo"));
				p_list->push_back(PropertyInfo(Variant::BOOL, base + "mod_alt"));
				p_list->push_back(PropertyInfo(Variant::BOOL, base + "mod_shift"));
				p_list->push_back(PropertyInfo(Variant::BOOL, base + "mod_ctrl"));
				p_list->push_back(PropertyInfo(Variant::BOOL, base + "mod_meta"));

			} break;
			case InputEvent::MOUSE_MOTION: {
				p_list->push_back(PropertyInfo(Variant::INT, base + "button_mask", PROPERTY_HINT_FLAGS, "Left,Right,Middle,WheelUp,WheelDown,WheelLeft,WheelRight"));
				p_list->push_back(PropertyInfo(Variant::BOOL, base + "mod_alt"));
				p_list->push_back(PropertyInfo(Variant::BOOL, base + "mod_shift"));
				p_list->push_back(PropertyInfo(Variant::BOOL, base + "mod_ctrl"));
				p_list->push_back(PropertyInfo(Variant::BOOL, base + "mod_meta"));

			} break;
			case InputEvent::MOUSE_BUTTON: {
				p_list->push_back(PropertyInfo(Variant::INT, base + "button_index", PROPERTY_HINT_ENUM, "Any,Left,Right,Middle,WheelUp,WheelDown,WheelLeft,WheelRight"));
				p_list->push_back(PropertyInfo(Variant::BOOL, base + "pressed"));
				p_list->push_back(PropertyInfo(Variant::BOOL, base + "doubleclicked"));
				p_list->push_back(PropertyInfo(Variant::BOOL, base + "mod_alt"));
				p_list->push_back(PropertyInfo(Variant::BOOL, base + "mod_shift"));
				p_list->push_back(PropertyInfo(Variant::BOOL, base + "mod_ctrl"));
				p_list->push_back(PropertyInfo(Variant::BOOL, base + "mod_meta"));

			} break;
			case InputEvent::JOYPAD_MOTION: {

				p_list->push_back(PropertyInfo(Variant::INT, base + "axis_index"));
				p_list->push_back(PropertyInfo(Variant::INT, base + "mode", PROPERTY_HINT_ENUM, "Min,Max"));
				p_list->push_back(PropertyInfo(Variant::REAL, base + "treshold", PROPERTY_HINT_RANGE, "0,1,0.01"));
			} break;
			case InputEvent::JOYPAD_BUTTON: {
				p_list->push_back(PropertyInfo(Variant::INT, base + "button_index"));
				p_list->push_back(PropertyInfo(Variant::BOOL, base + "pressed"));

			} break;
			case InputEvent::SCREEN_TOUCH: {
				p_list->push_back(PropertyInfo(Variant::INT, base + "finger_index"));
				p_list->push_back(PropertyInfo(Variant::BOOL, base + "pressed"));

			} break;
			case InputEvent::SCREEN_DRAG: {
				p_list->push_back(PropertyInfo(Variant::INT, base + "finger_index"));
			} break;
			case InputEvent::ACTION: {

				if (actions == String()) {

					actions = "None";

					List<PropertyInfo> pinfo;
					GlobalConfig::get_singleton()->get_property_list(&pinfo);
					Vector<String> al;

					for (List<PropertyInfo>::Element *E = pinfo.front(); E; E = E->next()) {
						const PropertyInfo &pi = E->get();

						if (!pi.name.begins_with("input/"))
							continue;

						String name = pi.name.substr(pi.name.find("/") + 1, pi.name.length());

						al.push_back(name);
					}

					for (int i = 0; i < al.size(); i++) {
						actions += ",";
						actions += al[i];
					}
				}

				p_list->push_back(PropertyInfo(Variant::STRING, base + "action_name", PROPERTY_HINT_ENUM, actions));
				p_list->push_back(PropertyInfo(Variant::BOOL, base + "pressed"));

			} break;
		}
	}
}

class VisualScriptNodeInstanceInputFilter : public VisualScriptNodeInstance {
public:
	VisualScriptInstance *instance;
	Vector<InputEvent> filters;

	//virtual int get_working_memory_size() const { return 0; }
	//virtual bool is_output_port_unsequenced(int p_idx) const { return false; }
	//virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return false; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {

		if (p_inputs[0]->get_type() != Variant::INPUT_EVENT) {
			r_error_str = "Input value not of type event";
			r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
			return 0;
		}

		InputEvent event = *p_inputs[0];

		for (int i = 0; i < filters.size(); i++) {

			const InputEvent &ie = filters[i];
			if (ie.type != event.type)
				continue;

			bool match = false;

			switch (ie.type) {

				case InputEvent::NONE: {

					match = true;
				} break;
				case InputEvent::KEY: {

					InputEventKey k = ie.key;
					InputEventKey k2 = event.key;

					if (k.scancode == 0 && k.unicode == 0 && k2.scancode == 0 && k2.unicode == 0) {
						match = true;

					} else {

						if ((k.scancode != 0 && k.scancode == k2.scancode) || (k.unicode != 0 && k.unicode == k2.unicode)) {
							//key valid

							if (
									k.pressed == k2.pressed &&
									k.echo == k2.echo &&
									k.mod == k2.mod) {
								match = true;
							}
						}
					}

				} break;
				case InputEvent::MOUSE_MOTION: {
					InputEventMouseMotion mm = ie.mouse_motion;
					InputEventMouseMotion mm2 = event.mouse_motion;

					if (mm.button_mask == mm2.button_mask &&
							mm.mod == mm2.mod) {
						match = true;
					}

				} break;
				case InputEvent::MOUSE_BUTTON: {

					InputEventMouseButton mb = ie.mouse_button;
					InputEventMouseButton mb2 = event.mouse_button;

					if (mb.button_index == mb2.button_index &&
							mb.pressed == mb2.pressed &&
							mb.doubleclick == mb2.doubleclick &&
							mb.mod == mb2.mod) {
						match = true;
					}

				} break;
				case InputEvent::JOYPAD_MOTION: {

					InputEventJoypadMotion jm = ie.joy_motion;
					InputEventJoypadMotion jm2 = event.joy_motion;

					int axis = jm.axis >> 1;

					if (axis == jm2.axis) {

						if (jm.axis & 1) {
							//greater
							if (jm2.axis_value > jm.axis_value) {
								match = true;
							}
						} else {
							//less
							if (jm2.axis_value < -jm.axis_value) {
								match = true;
							}
						}
					}

				} break;
				case InputEvent::JOYPAD_BUTTON: {
					InputEventJoypadButton jb = ie.joy_button;
					InputEventJoypadButton jb2 = event.joy_button;

					if (jb.button_index == jb2.button_index &&
							jb.pressed == jb2.pressed) {
						match = true;
					}
				} break;
				case InputEvent::SCREEN_TOUCH: {
					InputEventScreenTouch st = ie.screen_touch;
					InputEventScreenTouch st2 = event.screen_touch;

					if (st.index == st2.index &&
							st.pressed == st2.pressed) {
						match = true;
					}

				} break;
				case InputEvent::SCREEN_DRAG: {
					InputEventScreenDrag sd = ie.screen_drag;
					InputEventScreenDrag sd2 = event.screen_drag;

					if (sd.index == sd2.index) {
						match = true;
					}
				} break;
				case InputEvent::ACTION: {

					InputEventAction ia = ie.action;
					InputEventAction ia2 = event.action;

					if (ia.action == ia2.action &&
							ia.pressed == ia2.pressed) {
						match = true;
					}
				} break;
			}

			*p_outputs[0] = event;

			if (match)
				return i; //go through match output
		}

		return STEP_NO_ADVANCE_BIT; //none found, don't advance
	}
};

VisualScriptNodeInstance *VisualScriptInputFilter::instance(VisualScriptInstance *p_instance) {

	VisualScriptNodeInstanceInputFilter *instance = memnew(VisualScriptNodeInstanceInputFilter);
	instance->instance = p_instance;
	instance->filters = filters;
	return instance;
}

VisualScriptInputFilter::VisualScriptInputFilter() {
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

	return PropertyInfo(Variant::OBJECT, "");
}

String VisualScriptTypeCast::get_caption() const {

	return "TypeCast";
}

String VisualScriptTypeCast::get_text() const {

	if (script != String())
		return "Is " + script.get_file() + "?";
	else
		return "Is " + base_type + "?";
}

void VisualScriptTypeCast::set_base_type(const StringName &p_type) {

	if (base_type == p_type)
		return;

	base_type = p_type;
	_change_notify();
	ports_changed_notify();
}

StringName VisualScriptTypeCast::get_base_type() const {

	return base_type;
}

void VisualScriptTypeCast::set_base_script(const String &p_path) {

	if (script == p_path)
		return;

	script = p_path;
	_change_notify();
	ports_changed_notify();
}
String VisualScriptTypeCast::get_base_script() const {

	return script;
}

class VisualScriptNodeInstanceTypeCast : public VisualScriptNodeInstance {
public:
	VisualScriptInstance *instance;
	StringName base_type;
	String script;

	//virtual int get_working_memory_size() const { return 0; }
	//virtual bool is_output_port_unsequenced(int p_idx) const { return false; }
	//virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return false; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {

		Object *obj = *p_inputs[0];

		*p_outputs[0] = Variant();

		if (!obj) {
			r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
			r_error_str = "Instance is null";
			return 0;
		}

		if (script != String()) {

			Ref<Script> obj_script = obj->get_script();
			if (!obj_script.is_valid()) {
				return 1; //well, definitely not the script because object we got has no script.
			}

			if (!ResourceCache::has(script)) {
				//if the script is not in use by anyone, we can safely assume whathever we got is not casting to it.
				return 1;
			}
			Ref<Script> cast_script = Ref<Resource>(ResourceCache::get(script));
			if (!cast_script.is_valid()) {
				r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
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
		} else
			return 1;
	}
};

VisualScriptNodeInstance *VisualScriptTypeCast::instance(VisualScriptInstance *p_instance) {

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
	for (List<String>::Element *E = script_extensions.front(); E; E = E->next()) {
		if (script_ext_hint != String())
			script_ext_hint += ",";
		script_ext_hint += "*." + E->get();
	}

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "function/base_type", PROPERTY_HINT_TYPE_STRING, "Object"), "set_base_type", "get_base_type");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "property/base_script", PROPERTY_HINT_FILE, script_ext_hint), "set_base_script", "get_base_script");
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
	VisualScriptLanguage::singleton->add_register_func("flow_control/input_filter", create_node_generic<VisualScriptInputFilter>);
	VisualScriptLanguage::singleton->add_register_func("flow_control/type_cast", create_node_generic<VisualScriptTypeCast>);
}
