/**************************************************************************/
/*  debugger_marshalls.cpp                                                */
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

#include "debugger_marshalls.h"

#include "core/io/marshalls.h"

#define CHECK_SIZE(arr, expected, what) ERR_FAIL_COND_V_MSG((uint32_t)arr.size() < (uint32_t)(expected), false, String("Malformed ") + what + " message from script debugger, message too short. Expected size: " + itos(expected) + ", actual size: " + itos(arr.size()))
#define CHECK_END(arr, expected, what) ERR_FAIL_COND_V_MSG((uint32_t)arr.size() > (uint32_t)expected, false, String("Malformed ") + what + " message from script debugger, message too long. Expected size: " + itos(expected) + ", actual size: " + itos(arr.size()))

Array DebuggerMarshalls::ScriptStackDump::serialize() {
	Array arr;
	arr.push_back(frames.size() * 3);
	for (const ScriptLanguage::StackInfo &frame : frames) {
		arr.push_back(frame.file);
		arr.push_back(frame.line);
		arr.push_back(frame.func);
	}
	return arr;
}

bool DebuggerMarshalls::ScriptStackDump::deserialize(const Array &p_arr) {
	CHECK_SIZE(p_arr, 1, "ScriptStackDump");
	uint32_t size = p_arr[0];
	CHECK_SIZE(p_arr, size, "ScriptStackDump");
	int idx = 1;
	for (uint32_t i = 0; i < size / 3; i++) {
		ScriptLanguage::StackInfo sf;
		sf.file = p_arr[idx];
		sf.line = p_arr[idx + 1];
		sf.func = p_arr[idx + 2];
		frames.push_back(sf);
		idx += 3;
	}
	CHECK_END(p_arr, idx, "ScriptStackDump");
	return true;
}

Array DebuggerMarshalls::ScriptStackVariable::serialize(int max_size) {
	Array arr;
	arr.push_back(name);
	arr.push_back(type);
	arr.push_back(value.get_type());

	Variant var = value;
	if (value.get_type() == Variant::OBJECT && value.get_validated_object() == nullptr) {
		var = Variant();
	}

	int len = 0;
	Error err = encode_variant(var, nullptr, len, false);
	if (err != OK) {
		ERR_PRINT("Failed to encode variant.");
	}

	if (len > max_size) {
		arr.push_back(Variant());
	} else {
		arr.push_back(var);
	}
	return arr;
}

bool DebuggerMarshalls::ScriptStackVariable::deserialize(const Array &p_arr) {
	CHECK_SIZE(p_arr, 4, "ScriptStackVariable");
	name = p_arr[0];
	type = p_arr[1];
	var_type = p_arr[2];
	value = p_arr[3];
	CHECK_END(p_arr, 4, "ScriptStackVariable");
	return true;
}

Array DebuggerMarshalls::OutputError::serialize() {
	Array arr;
	arr.push_back(hr);
	arr.push_back(min);
	arr.push_back(sec);
	arr.push_back(msec);
	arr.push_back(source_file);
	arr.push_back(source_func);
	arr.push_back(source_line);
	arr.push_back(error);
	arr.push_back(error_descr);
	arr.push_back(warning);
	unsigned int size = callstack.size();
	const ScriptLanguage::StackInfo *r = callstack.ptr();
	arr.push_back(size * 3);
	for (int i = 0; i < callstack.size(); i++) {
		arr.push_back(r[i].file);
		arr.push_back(r[i].func);
		arr.push_back(r[i].line);
	}
	return arr;
}

bool DebuggerMarshalls::OutputError::deserialize(const Array &p_arr) {
	CHECK_SIZE(p_arr, 11, "OutputError");
	hr = p_arr[0];
	min = p_arr[1];
	sec = p_arr[2];
	msec = p_arr[3];
	source_file = p_arr[4];
	source_func = p_arr[5];
	source_line = p_arr[6];
	error = p_arr[7];
	error_descr = p_arr[8];
	warning = p_arr[9];
	unsigned int stack_size = p_arr[10];
	CHECK_SIZE(p_arr, stack_size, "OutputError");
	int idx = 11;
	callstack.resize(stack_size / 3);
	ScriptLanguage::StackInfo *w = callstack.ptrw();
	for (unsigned int i = 0; i < stack_size / 3; i++) {
		w[i].file = p_arr[idx];
		w[i].func = p_arr[idx + 1];
		w[i].line = p_arr[idx + 2];
		idx += 3;
	}
	CHECK_END(p_arr, idx, "OutputError");
	return true;
}

Array DebuggerMarshalls::serialize_key_shortcut(const Ref<Shortcut> &p_shortcut) {
	ERR_FAIL_COND_V(p_shortcut.is_null(), Array());
	Array keys;
	for (const Ref<InputEvent> ev : p_shortcut->get_events()) {
		const Ref<InputEventKey> kev = ev;
		ERR_CONTINUE(kev.is_null());
		if (kev->get_physical_keycode() != Key::NONE) {
			keys.push_back(true);
			keys.push_back(kev->get_physical_keycode_with_modifiers());
		} else {
			keys.push_back(false);
			keys.push_back(kev->get_keycode_with_modifiers());
		}
	}
	return keys;
}

Ref<Shortcut> DebuggerMarshalls::deserialize_key_shortcut(const Array &p_keys) {
	Array key_events;
	ERR_FAIL_COND_V(p_keys.size() % 2 != 0, Ref<Shortcut>());
	for (int i = 0; i < p_keys.size(); i += 2) {
		ERR_CONTINUE(p_keys[i].get_type() != Variant::BOOL);
		ERR_CONTINUE(p_keys[i + 1].get_type() != Variant::INT);
		key_events.push_back(InputEventKey::create_reference((Key)p_keys[i + 1].operator int(), p_keys[i].operator bool()));
	}
	if (key_events.is_empty()) {
		return Ref<Shortcut>();
	}
	Ref<Shortcut> shortcut;
	shortcut.instantiate();
	shortcut->set_events(key_events);
	return shortcut;
}
