/*************************************************************************/
/*  debugger_marshalls.cpp                                               */
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

#include "debugger_marshalls.h"

#include "core/io/marshalls.h"

#define CHECK_SIZE(arr, expected, what) ERR_FAIL_COND_V_MSG((uint32_t)(arr).size() < (uint32_t)(expected), false, String("Malformed ") + (what) + " message from script debugger, message too short. Expected size: " + itos(expected) + ", actual size: " + itos((arr).size()))
#define CHECK_END(arr, expected, what) ERR_FAIL_COND_V_MSG((uint32_t)(arr).size() > (uint32_t)(expected), false, String("Malformed ") + (what) + " message from script debugger, message too long. Expected size: " + itos(expected) + ", actual size: " + itos((arr).size()))

Array DebuggerMarshalls::ScriptStackDump::serialize() const {
	Array arr;
	arr.push_back(frames.size() * 3);
	for (int i = 0; i < frames.size(); i++) {
		arr.push_back(frames[i].file);
		arr.push_back(frames[i].line);
		arr.push_back(frames[i].func);
	}
	if (!tid.is_zero()) {
		arr.push_back(tid);
	}
	return arr;
}

bool DebuggerMarshalls::ScriptStackDump::deserialize(const Array &p_arr) {
	CHECK_SIZE(p_arr, 1, "ScriptStackDump");
	const uint32_t size = p_arr[0];
	CHECK_SIZE(p_arr, size, "ScriptStackDump");
	int idx = 1;
	for (uint32_t i = 0; i < size / 3; i++) {
		StackInfo sf;
		sf.file = p_arr[idx];
		sf.line = p_arr[idx + 1];
		sf.func = p_arr[idx + 2];
		frames.push_back(sf);
		idx += 3;
	}
	if (idx == p_arr.size() - 1) {
		// one extra data item is optional TID
		tid = p_arr[idx];
		++idx;
	} else {
		tid.zero();
	}
	CHECK_END(p_arr, idx, "ScriptStackDump");
	return true;
}

void DebuggerMarshalls::ScriptStackDump::populate(const ScriptLanguageThreadContext &p_context) {
	tid = p_context.debug_get_thread_id();
	const int slc = p_context.debug_get_stack_level_count();
	for (int i = 0; i < slc; i++) {
		StackInfo frame;
		frame.file = p_context.debug_get_stack_level_source(i);
		frame.line = p_context.debug_get_stack_level_line(i);
		frame.func = p_context.debug_get_stack_level_function(i);
		frames.push_back(frame);
	}
}

void DebuggerMarshalls::ScriptStackDump::clear() {
	tid = DebugThreadID();
	frames.clear();
}

Array DebuggerMarshalls::ScriptStackVariable::serialize(int max_size) const {
	Array arr;
	arr.push_back(name);
	arr.push_back(type);

	Variant var = value;
	if (value.get_type() == Variant::OBJECT && value.get_validated_object() == nullptr) {
		var = Variant();
	}

	int len = 0;
	const Error err = encode_variant(var, nullptr, len, true);
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
	CHECK_SIZE(p_arr, 3, "ScriptStackVariable");
	name = p_arr[0];
	type = p_arr[1];
	value = p_arr[2];
	CHECK_END(p_arr, 3, "ScriptStackVariable");
	return true;
}

Array DebuggerMarshalls::OutputError::serialize() const {
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
	const unsigned int size = callstack.size();
	const StackInfo *r = callstack.ptr();
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
	const unsigned int stack_size = p_arr[10];
	CHECK_SIZE(p_arr, stack_size, "OutputError");
	int idx = 11;
	callstack.resize(static_cast<int>(stack_size) / 3);
	StackInfo *w = callstack.ptrw();
	for (unsigned int i = 0; i < stack_size / 3; i++) {
		w[i].file = p_arr[idx];
		w[i].func = p_arr[idx + 1];
		w[i].line = p_arr[idx + 2];
		idx += 3;
	}
	CHECK_END(p_arr, idx, "OutputError");
	return true;
}
