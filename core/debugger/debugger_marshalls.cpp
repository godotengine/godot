/*************************************************************************/
/*  debugger_marshalls.cpp                                               */
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

#include "debugger_marshalls.h"

#include "core/io/marshalls.h"

#define CHECK_SIZE(arr, expected, what) ERR_FAIL_COND_V_MSG((uint32_t)arr.size() < (uint32_t)(expected), false, String("Malformed ") + what + " message from script debugger, message too short. Expected size: " + itos(expected) + ", actual size: " + itos(arr.size()))
#define CHECK_END(arr, expected, what) ERR_FAIL_COND_V_MSG((uint32_t)arr.size() > (uint32_t)expected, false, String("Malformed ") + what + " message from script debugger, message too long. Expected size: " + itos(expected) + ", actual size: " + itos(arr.size()))

Array DebuggerMarshalls::ResourceUsage::serialize() {
	infos.sort();

	Array arr;
	arr.push_back(infos.size() * 4);
	for (List<ResourceInfo>::Element *E = infos.front(); E; E = E->next()) {
		arr.push_back(E->get().path);
		arr.push_back(E->get().format);
		arr.push_back(E->get().type);
		arr.push_back(E->get().vram);
	}
	return arr;
}

bool DebuggerMarshalls::ResourceUsage::deserialize(const Array &p_arr) {
	CHECK_SIZE(p_arr, 1, "ResourceUsage");
	uint32_t size = p_arr[0];
	CHECK_SIZE(p_arr, size, "ResourceUsage");
	int idx = 1;
	for (uint32_t i = 0; i < size / 4; i++) {
		ResourceInfo info;
		info.path = p_arr[idx];
		info.format = p_arr[idx + 1];
		info.type = p_arr[idx + 2];
		info.vram = p_arr[idx + 3];
		infos.push_back(info);
	}
	CHECK_END(p_arr, idx, "ResourceUsage");
	return true;
}

Array DebuggerMarshalls::ScriptFunctionSignature::serialize() {
	Array arr;
	arr.push_back(name);
	arr.push_back(id);
	return arr;
}

bool DebuggerMarshalls::ScriptFunctionSignature::deserialize(const Array &p_arr) {
	CHECK_SIZE(p_arr, 2, "ScriptFunctionSignature");
	name = p_arr[0];
	id = p_arr[1];
	CHECK_END(p_arr, 2, "ScriptFunctionSignature");
	return true;
}

Array DebuggerMarshalls::NetworkProfilerFrame::serialize() {
	Array arr;
	arr.push_back(infos.size() * 6);
	for (int i = 0; i < infos.size(); ++i) {
		arr.push_back(uint64_t(infos[i].node));
		arr.push_back(infos[i].node_path);
		arr.push_back(infos[i].incoming_rpc);
		arr.push_back(infos[i].incoming_rset);
		arr.push_back(infos[i].outgoing_rpc);
		arr.push_back(infos[i].outgoing_rset);
	}
	return arr;
}

bool DebuggerMarshalls::NetworkProfilerFrame::deserialize(const Array &p_arr) {
	CHECK_SIZE(p_arr, 1, "NetworkProfilerFrame");
	uint32_t size = p_arr[0];
	CHECK_SIZE(p_arr, size, "NetworkProfilerFrame");
	infos.resize(size);
	int idx = 1;
	for (uint32_t i = 0; i < size / 6; ++i) {
		infos.write[i].node = uint64_t(p_arr[idx]);
		infos.write[i].node_path = p_arr[idx + 1];
		infos.write[i].incoming_rpc = p_arr[idx + 2];
		infos.write[i].incoming_rset = p_arr[idx + 3];
		infos.write[i].outgoing_rpc = p_arr[idx + 4];
		infos.write[i].outgoing_rset = p_arr[idx + 5];
	}
	CHECK_END(p_arr, idx, "NetworkProfilerFrame");
	return true;
}

Array DebuggerMarshalls::ServersProfilerFrame::serialize() {
	Array arr;
	arr.push_back(frame_number);
	arr.push_back(frame_time);
	arr.push_back(idle_time);
	arr.push_back(physics_time);
	arr.push_back(physics_frame_time);
	arr.push_back(script_time);

	arr.push_back(servers.size());
	for (int i = 0; i < servers.size(); i++) {
		ServerInfo &s = servers[i];
		arr.push_back(s.name);
		arr.push_back(s.functions.size() * 2);
		for (int j = 0; j < s.functions.size(); j++) {
			ServerFunctionInfo &f = s.functions[j];
			arr.push_back(f.name);
			arr.push_back(f.time);
		}
	}

	arr.push_back(script_functions.size() * 4);
	for (int i = 0; i < script_functions.size(); i++) {
		arr.push_back(script_functions[i].sig_id);
		arr.push_back(script_functions[i].call_count);
		arr.push_back(script_functions[i].self_time);
		arr.push_back(script_functions[i].total_time);
	}
	return arr;
}

bool DebuggerMarshalls::ServersProfilerFrame::deserialize(const Array &p_arr) {
	CHECK_SIZE(p_arr, 7, "ServersProfilerFrame");
	frame_number = p_arr[0];
	frame_time = p_arr[1];
	idle_time = p_arr[2];
	physics_time = p_arr[3];
	physics_frame_time = p_arr[4];
	script_time = p_arr[5];
	int servers_size = p_arr[6];
	int idx = 7;
	while (servers_size) {
		CHECK_SIZE(p_arr, idx + 2, "ServersProfilerFrame");
		servers_size--;
		ServerInfo si;
		si.name = p_arr[idx];
		int sub_data_size = p_arr[idx + 1];
		idx += 2;
		CHECK_SIZE(p_arr, idx + sub_data_size, "ServersProfilerFrame");
		for (int j = 0; j < sub_data_size / 2; j++) {
			ServerFunctionInfo sf;
			sf.name = p_arr[idx];
			sf.time = p_arr[idx + 1];
			idx += 2;
			si.functions.push_back(sf);
		}
		servers.push_back(si);
	}
	CHECK_SIZE(p_arr, idx + 1, "ServersProfilerFrame");
	int func_size = p_arr[idx];
	idx += 1;
	CHECK_SIZE(p_arr, idx + func_size, "ServersProfilerFrame");
	for (int i = 0; i < func_size / 4; i++) {
		ScriptFunctionInfo fi;
		fi.sig_id = p_arr[idx];
		fi.call_count = p_arr[idx + 1];
		fi.self_time = p_arr[idx + 2];
		fi.total_time = p_arr[idx + 3];
		script_functions.push_back(fi);
		idx += 4;
	}
	CHECK_END(p_arr, idx, "ServersProfilerFrame");
	return true;
}

Array DebuggerMarshalls::ScriptStackDump::serialize() {
	Array arr;
	arr.push_back(frames.size() * 3);
	for (int i = 0; i < frames.size(); i++) {
		arr.push_back(frames[i].file);
		arr.push_back(frames[i].line);
		arr.push_back(frames[i].func);
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

	Variant var = value;
	if (value.get_type() == Variant::OBJECT && value.get_validated_object() == nullptr) {
		var = Variant();
	}

	int len = 0;
	Error err = encode_variant(var, nullptr, len, true);
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

Array DebuggerMarshalls::VisualProfilerFrame::serialize() {
	Array arr;
	arr.push_back(frame_number);
	arr.push_back(areas.size() * 3);
	for (int i = 0; i < areas.size(); i++) {
		arr.push_back(areas[i].name);
		arr.push_back(areas[i].cpu_msec);
		arr.push_back(areas[i].gpu_msec);
	}
	return arr;
}

bool DebuggerMarshalls::VisualProfilerFrame::deserialize(const Array &p_arr) {
	CHECK_SIZE(p_arr, 2, "VisualProfilerFrame");
	frame_number = p_arr[0];
	int size = p_arr[1];
	CHECK_SIZE(p_arr, size, "VisualProfilerFrame");
	int idx = 2;
	areas.resize(size / 3);
	RS::FrameProfileArea *w = areas.ptrw();
	for (int i = 0; i < size / 3; i++) {
		w[i].name = p_arr[idx];
		w[i].cpu_msec = p_arr[idx + 1];
		w[i].gpu_msec = p_arr[idx + 2];
		idx += 3;
	}
	CHECK_END(p_arr, idx, "VisualProfilerFrame");
	return true;
}
