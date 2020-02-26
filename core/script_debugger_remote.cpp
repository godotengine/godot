/*************************************************************************/
/*  script_debugger_remote.cpp                                           */
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

#include "script_debugger_remote.h"

#include "core/engine.h"
#include "core/io/ip.h"
#include "core/io/marshalls.h"
#include "core/os/input.h"
#include "core/os/os.h"
#include "core/project_settings.h"
#include "servers/visual_server.h"

#define CHECK_SIZE(arr, expected, what) ERR_FAIL_COND_V_MSG((uint32_t)arr.size() < (uint32_t)(expected), false, String("Malformed ") + what + " message from script debugger, message too short. Exptected size: " + itos(expected) + ", actual size: " + itos(arr.size()))
#define CHECK_END(arr, expected, what) ERR_FAIL_COND_V_MSG((uint32_t)arr.size() > (uint32_t)expected, false, String("Malformed ") + what + " message from script debugger, message too short. Exptected size: " + itos(expected) + ", actual size: " + itos(arr.size()))

Array ScriptDebuggerRemote::ScriptStackDump::serialize() {
	Array arr;
	arr.push_back(frames.size() * 3);
	for (int i = 0; i < frames.size(); i++) {
		arr.push_back(frames[i].file);
		arr.push_back(frames[i].line);
		arr.push_back(frames[i].func);
	}
	return arr;
}

bool ScriptDebuggerRemote::ScriptStackDump::deserialize(const Array &p_arr) {
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

Array ScriptDebuggerRemote::ScriptStackVariable::serialize(int max_size) {
	Array arr;
	arr.push_back(name);
	arr.push_back(type);

	Variant var = value;
	if (value.get_type() == Variant::OBJECT && value.get_validated_object() == nullptr) {
		var = Variant();
	}

	int len = 0;
	Error err = encode_variant(var, NULL, len, true);
	if (err != OK)
		ERR_PRINT("Failed to encode variant.");

	if (len > max_size) {
		arr.push_back(Variant());
	} else {
		arr.push_back(var);
	}
	return arr;
}

bool ScriptDebuggerRemote::ScriptStackVariable::deserialize(const Array &p_arr) {
	CHECK_SIZE(p_arr, 3, "ScriptStackVariable");
	name = p_arr[0];
	type = p_arr[1];
	value = p_arr[2];
	CHECK_END(p_arr, 3, "ScriptStackVariable");
	return true;
}

Array ScriptDebuggerRemote::OutputError::serialize() {
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

bool ScriptDebuggerRemote::OutputError::deserialize(const Array &p_arr) {
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

Array ScriptDebuggerRemote::ResourceUsage::serialize() {
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

bool ScriptDebuggerRemote::ResourceUsage::deserialize(const Array &p_arr) {
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

Array ScriptDebuggerRemote::ProfilerSignature::serialize() {
	Array arr;
	arr.push_back(name);
	arr.push_back(id);
	return arr;
}

bool ScriptDebuggerRemote::ProfilerSignature::deserialize(const Array &p_arr) {
	CHECK_SIZE(p_arr, 2, "ProfilerSignature");
	name = p_arr[0];
	id = p_arr[1];
	CHECK_END(p_arr, 2, "ProfilerSignature");
	return true;
}

Array ScriptDebuggerRemote::ProfilerFrame::serialize() {
	Array arr;
	arr.push_back(frame_number);
	arr.push_back(frame_time);
	arr.push_back(idle_time);
	arr.push_back(physics_time);
	arr.push_back(physics_frame_time);
	arr.push_back(USEC_TO_SEC(script_time));

	arr.push_back(frames_data.size());
	arr.push_back(frame_functions.size() * 4);

	// Servers profiling info.
	for (int i = 0; i < frames_data.size(); i++) {
		arr.push_back(frames_data[i].name); // Type (physics/process/audio/...)
		arr.push_back(frames_data[i].data.size());
		for (int j = 0; j < frames_data[i].data.size() / 2; j++) {
			arr.push_back(frames_data[i].data[2 * j]); // NAME
			arr.push_back(frames_data[i].data[2 * j + 1]); // TIME
		}
	}
	for (int i = 0; i < frame_functions.size(); i++) {
		arr.push_back(frame_functions[i].sig_id);
		arr.push_back(frame_functions[i].call_count);
		arr.push_back(frame_functions[i].self_time);
		arr.push_back(frame_functions[i].total_time);
	}
	return arr;
}

bool ScriptDebuggerRemote::ProfilerFrame::deserialize(const Array &p_arr) {
	CHECK_SIZE(p_arr, 8, "ProfilerFrame");
	frame_number = p_arr[0];
	frame_time = p_arr[1];
	idle_time = p_arr[2];
	physics_time = p_arr[3];
	physics_frame_time = p_arr[4];
	script_time = p_arr[5];
	uint32_t frame_data_size = p_arr[6];
	int frame_func_size = p_arr[7];
	int idx = 8;
	while (frame_data_size) {
		CHECK_SIZE(p_arr, idx + 2, "ProfilerFrame");
		frame_data_size--;
		FrameData fd;
		fd.name = p_arr[idx];
		int sub_data_size = p_arr[idx + 1];
		idx += 2;
		CHECK_SIZE(p_arr, idx + sub_data_size, "ProfilerFrame");
		for (int j = 0; j < sub_data_size / 2; j++) {
			fd.data.push_back(p_arr[idx]); // NAME
			fd.data.push_back(p_arr[idx + 1]); // TIME
			idx += 2;
		}
		frames_data.push_back(fd);
	}
	CHECK_SIZE(p_arr, idx + frame_func_size, "ProfilerFrame");
	for (int i = 0; i < frame_func_size / 4; i++) {
		FrameFunction ff;
		ff.sig_id = p_arr[idx];
		ff.call_count = p_arr[idx + 1];
		ff.self_time = p_arr[idx + 2];
		ff.total_time = p_arr[idx + 3];
		frame_functions.push_back(ff);
		idx += 4;
	}
	CHECK_END(p_arr, idx, "ProfilerFrame");
	return true;
}

Array ScriptDebuggerRemote::NetworkProfilerFrame::serialize() {
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

bool ScriptDebuggerRemote::NetworkProfilerFrame::deserialize(const Array &p_arr) {
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

void ScriptDebuggerRemote::_put_msg(String p_message, Array p_data) {
	Array msg;
	msg.push_back(p_message);
	msg.push_back(p_data);
	packet_peer_stream->put_var(msg);
}

bool ScriptDebuggerRemote::is_peer_connected() {
	return tcp_client->is_connected_to_host() && tcp_client->get_status() == StreamPeerTCP::STATUS_CONNECTED;
}

void ScriptDebuggerRemote::_send_video_memory() {

	ResourceUsage usage;
	if (resource_usage_func)
		resource_usage_func(&usage);

	_put_msg("message:video_mem", usage.serialize());
}

Error ScriptDebuggerRemote::connect_to_host(const String &p_host, uint16_t p_port) {

	IP_Address ip;
	if (p_host.is_valid_ip_address())
		ip = p_host;
	else
		ip = IP::get_singleton()->resolve_hostname(p_host);

	int port = p_port;

	const int tries = 6;
	int waits[tries] = { 1, 10, 100, 1000, 1000, 1000 };

	tcp_client->connect_to_host(ip, port);

	for (int i = 0; i < tries; i++) {

		if (tcp_client->get_status() == StreamPeerTCP::STATUS_CONNECTED) {
			print_verbose("Remote Debugger: Connected!");
			break;
		} else {

			const int ms = waits[i];
			OS::get_singleton()->delay_usec(ms * 1000);
			print_verbose("Remote Debugger: Connection failed with status: '" + String::num(tcp_client->get_status()) + "', retrying in " + String::num(ms) + " msec.");
		};
	};

	if (tcp_client->get_status() != StreamPeerTCP::STATUS_CONNECTED) {

		ERR_PRINT("Remote Debugger: Unable to connect. Status: " + String::num(tcp_client->get_status()) + ".");
		return FAILED;
	};

	packet_peer_stream->set_stream_peer(tcp_client);
	Array msg;
	msg.push_back(OS::get_singleton()->get_process_id());
	send_message("set_pid", msg);

	return OK;
}

void ScriptDebuggerRemote::_parse_message(const String p_command, const Array &p_data, ScriptLanguage *p_script) {

	if (p_command == "request_video_mem") {
		_send_video_memory();

	} else if (p_command == "start_profiling") {
		ERR_FAIL_COND(p_data.size() < 1);

		for (int i = 0; i < ScriptServer::get_language_count(); i++) {
			ScriptServer::get_language(i)->profiling_start();
		}

		max_frame_functions = p_data[0];
		profiler_function_signature_map.clear();
		profiling = true;
		frame_time = 0;
		idle_time = 0;
		physics_time = 0;
		physics_frame_time = 0;
		print_line("PROFILING ALRIGHT!");

	} else if (p_command == "stop_profiling") {
		for (int i = 0; i < ScriptServer::get_language_count(); i++) {
			ScriptServer::get_language(i)->profiling_stop();
		}
		profiling = false;
		_send_profiling_data(false);
		print_line("PROFILING END!");

	} else if (p_command == "start_visual_profiling") {

		visual_profiling = true;
		VS::get_singleton()->set_frame_profiling_enabled(true);
	} else if (p_command == "stop_visual_profiling") {

		visual_profiling = false;
		VS::get_singleton()->set_frame_profiling_enabled(false);

	} else if (p_command == "start_network_profiling") {

		network_profiling = true;
		multiplayer->profiling_start();

	} else if (p_command == "stop_network_profiling") {

		network_profiling = false;
		multiplayer->profiling_end();

	} else if (p_command == "reload_scripts") {
		reload_all_scripts = true;

	} else if (p_command == "breakpoint") {
		ERR_FAIL_COND(p_data.size() < 3);
		bool set = p_data[2];
		if (set)
			insert_breakpoint(p_data[1], p_data[0]);
		else
			remove_breakpoint(p_data[1], p_data[0]);

	} else if (p_command == "set_skip_breakpoints") {
		ERR_FAIL_COND(p_data.size() < 1);
		skip_breakpoints = p_data[0];

	} else if (p_command == "get_stack_dump") {
		ERR_FAIL_COND(!p_script);
		ScriptStackDump dump;
		int slc = p_script->debug_get_stack_level_count();
		for (int i = 0; i < slc; i++) {
			ScriptLanguage::StackInfo frame;
			frame.file = p_script->debug_get_stack_level_source(i);
			frame.line = p_script->debug_get_stack_level_line(i);
			frame.func = p_script->debug_get_stack_level_function(i);
			dump.frames.push_back(frame);
		}
		_put_msg("stack_dump", dump.serialize());

	} else if (p_command == "get_stack_frame_vars") {
		ERR_FAIL_COND(p_data.size() != 1);
		ERR_FAIL_COND(!p_script);
		int lv = p_data[0];

		List<String> members;
		List<Variant> member_vals;
		if (ScriptInstance *inst = p_script->debug_get_stack_level_instance(lv)) {
			members.push_back("self");
			member_vals.push_back(inst->get_owner());
		}
		p_script->debug_get_stack_level_members(lv, &members, &member_vals);
		ERR_FAIL_COND(members.size() != member_vals.size());

		List<String> locals;
		List<Variant> local_vals;
		p_script->debug_get_stack_level_locals(lv, &locals, &local_vals);
		ERR_FAIL_COND(locals.size() != local_vals.size());

		List<String> globals;
		List<Variant> globals_vals;
		p_script->debug_get_globals(&globals, &globals_vals);
		ERR_FAIL_COND(globals.size() != globals_vals.size());

		_put_msg("stack_frame_vars", Array());

		ScriptStackVariable stvar;
		{ //locals
			List<String>::Element *E = locals.front();
			List<Variant>::Element *F = local_vals.front();
			while (E) {
				stvar.name = E->get();
				stvar.value = F->get();
				stvar.type = 0;
				_put_msg("stack_frame_var", stvar.serialize());

				E = E->next();
				F = F->next();
			}
		}

		{ //members
			List<String>::Element *E = members.front();
			List<Variant>::Element *F = member_vals.front();
			while (E) {
				stvar.name = E->get();
				stvar.value = F->get();
				stvar.type = 1;
				_put_msg("stack_frame_var", stvar.serialize());

				E = E->next();
				F = F->next();
			}
		}

		{ //globals
			List<String>::Element *E = globals.front();
			List<Variant>::Element *F = globals_vals.front();
			while (E) {
				stvar.name = E->get();
				stvar.value = F->get();
				stvar.type = 2;
				_put_msg("stack_frame_var", stvar.serialize());

				E = E->next();
				F = F->next();
			}
		}

	} else {
		if (scene_tree_parse_func) {
			scene_tree_parse_func(p_command, p_data);
		}
		// Unknown message...
	}
}

void ScriptDebuggerRemote::debug(ScriptLanguage *p_script, bool p_can_continue, bool p_is_error_breakpoint) {

	//this function is called when there is a debugger break (bug on script)
	//or when execution is paused from editor

	if (skip_breakpoints && !p_is_error_breakpoint)
		return;

	ERR_FAIL_COND_MSG(!is_peer_connected(), "Script Debugger failed to connect, but being used anyway.");

	Array msg;
	msg.push_back(p_can_continue);
	msg.push_back(p_script->debug_get_error());
	_put_msg("debug_enter", msg);

	skip_profile_frame = true; // to avoid super long frame time for the frame

	Input::MouseMode mouse_mode = Input::get_singleton()->get_mouse_mode();
	if (mouse_mode != Input::MOUSE_MODE_VISIBLE)
		Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_VISIBLE);

	uint64_t loop_begin_usec = 0;
	uint64_t loop_time_sec = 0;
	while (true) {
		loop_begin_usec = OS::get_singleton()->get_ticks_usec();

		_get_output();

		if (packet_peer_stream->get_available_packet_count() > 0) {

			Variant var;
			Error err = packet_peer_stream->get_var(var);

			ERR_CONTINUE(err != OK);
			ERR_CONTINUE(var.get_type() != Variant::ARRAY);

			Array cmd = var;

			ERR_CONTINUE(cmd.size() != 2);
			ERR_CONTINUE(cmd[0].get_type() != Variant::STRING);
			ERR_CONTINUE(cmd[1].get_type() != Variant::ARRAY);

			String command = cmd[0];
			Array data = cmd[1];
			if (command == "step") {

				set_depth(-1);
				set_lines_left(1);
				break;
			} else if (command == "next") {

				set_depth(0);
				set_lines_left(1);
				break;

			} else if (command == "continue") {
				set_depth(-1);
				set_lines_left(-1);
				OS::get_singleton()->move_window_to_foreground();
				break;
			} else if (command == "break") {
				ERR_PRINT("Got break when already broke!");
				break;
			}

			_parse_message(command, data, p_script);
		} else {
			OS::get_singleton()->delay_usec(10000);
			OS::get_singleton()->process_and_drop_events();
		}

		// This is for the camera override to stay live even when the game is paused from the editor
		loop_time_sec = (OS::get_singleton()->get_ticks_usec() - loop_begin_usec) / 1000000.0f;
		VisualServer::get_singleton()->sync();
		if (VisualServer::get_singleton()->has_changed()) {
			VisualServer::get_singleton()->draw(true, loop_time_sec * Engine::get_singleton()->get_time_scale());
		}
	}

	_put_msg("debug_exit", Array());

	if (mouse_mode != Input::MOUSE_MODE_VISIBLE)
		Input::get_singleton()->set_mouse_mode(mouse_mode);
}

void ScriptDebuggerRemote::_get_output() {

	MutexLock lock(mutex);

	if (output_strings.size()) {

		locking = true;

		while (output_strings.size()) {

			Array arr;
			arr.push_back(output_strings.front()->get());
			_put_msg("output", arr);
			output_strings.pop_front();
		}
		locking = false;
	}

	if (n_messages_dropped > 0) {
		Message msg;
		msg.message = "Too many messages! " + String::num_int64(n_messages_dropped) + " messages were dropped.";
		messages.push_back(msg);
		n_messages_dropped = 0;
	}

	while (messages.size()) {
		locking = true;
		Message msg = messages.front()->get();
		_put_msg("message:" + msg.message, msg.data);
		messages.pop_front();
		locking = false;
	}

	if (n_errors_dropped == 1) {
		// Only print one message about dropping per second
		OutputError oe;
		oe.error = "TOO_MANY_ERRORS";
		oe.error_descr = "Too many errors! Ignoring errors for up to 1 second.";
		oe.warning = false;
		uint64_t time = OS::get_singleton()->get_ticks_msec();
		oe.hr = time / 3600000;
		oe.min = (time / 60000) % 60;
		oe.sec = (time / 1000) % 60;
		oe.msec = time % 1000;
		errors.push_back(oe);
	}

	if (n_warnings_dropped == 1) {
		// Only print one message about dropping per second
		OutputError oe;
		oe.error = "TOO_MANY_WARNINGS";
		oe.error_descr = "Too many warnings! Ignoring warnings for up to 1 second.";
		oe.warning = true;
		uint64_t time = OS::get_singleton()->get_ticks_msec();
		oe.hr = time / 3600000;
		oe.min = (time / 60000) % 60;
		oe.sec = (time / 1000) % 60;
		oe.msec = time % 1000;
		errors.push_back(oe);
	}

	while (errors.size()) {
		locking = true;
		OutputError oe = errors.front()->get();
		_put_msg("error", oe.serialize());
		errors.pop_front();
		locking = false;
	}
}

void ScriptDebuggerRemote::line_poll() {

	//the purpose of this is just processing events every now and then when the script might get too busy
	//otherwise bugs like infinite loops can't be caught
	if (poll_every % 2048 == 0)
		_poll_events();
	poll_every++;
}

void ScriptDebuggerRemote::_err_handler(void *ud, const char *p_func, const char *p_file, int p_line, const char *p_err, const char *p_descr, ErrorHandlerType p_type) {

	if (p_type == ERR_HANDLER_SCRIPT)
		return; //ignore script errors, those go through debugger

	Vector<ScriptLanguage::StackInfo> si;

	for (int i = 0; i < ScriptServer::get_language_count(); i++) {
		si = ScriptServer::get_language(i)->debug_get_current_stack_info();
		if (si.size())
			break;
	}

	ScriptDebuggerRemote *sdr = (ScriptDebuggerRemote *)ud;
	sdr->send_error(p_func, p_file, p_line, p_err, p_descr, p_type, si);
}

void ScriptDebuggerRemote::_poll_events() {

	//this si called from ::idle_poll, happens only when running the game,
	//does not get called while on debug break

	while (packet_peer_stream->get_available_packet_count() > 0) {

		_get_output();

		//send over output_strings

		Variant var;
		Error err = packet_peer_stream->get_var(var);

		ERR_CONTINUE(err != OK);
		ERR_CONTINUE(var.get_type() != Variant::ARRAY);

		Array cmd = var;

		ERR_CONTINUE(cmd.size() < 2);
		ERR_CONTINUE(cmd[0].get_type() != Variant::STRING);
		ERR_CONTINUE(cmd[1].get_type() != Variant::ARRAY);

		String command = cmd[0];
		Array data = cmd[1];

		if (command == "break") {

			if (get_break_language())
				debug(get_break_language());
		} else {
			_parse_message(command, data);
		}
	}
}

void ScriptDebuggerRemote::_send_profiling_data(bool p_for_frame) {

	int ofs = 0;

	for (int i = 0; i < ScriptServer::get_language_count(); i++) {
		if (p_for_frame)
			ofs += ScriptServer::get_language(i)->profiling_get_frame_data(&profile_info.write[ofs], profile_info.size() - ofs);
		else
			ofs += ScriptServer::get_language(i)->profiling_get_accumulated_data(&profile_info.write[ofs], profile_info.size() - ofs);
	}

	for (int i = 0; i < ofs; i++) {
		profile_info_ptrs.write[i] = &profile_info.write[i];
	}

	SortArray<ScriptLanguage::ProfilingInfo *, ProfileInfoSort> sa;
	sa.sort(profile_info_ptrs.ptrw(), ofs);

	int to_send = MIN(ofs, max_frame_functions);

	//check signatures first
	uint64_t total_script_time = 0;

	for (int i = 0; i < to_send; i++) {

		if (!profiler_function_signature_map.has(profile_info_ptrs[i]->signature)) {

			int idx = profiler_function_signature_map.size();
			ProfilerSignature sig;
			sig.name = profile_info_ptrs[i]->signature;
			sig.id = idx;
			_put_msg("profile_sig", sig.serialize());
			profiler_function_signature_map[profile_info_ptrs[i]->signature] = idx;
		}

		total_script_time += profile_info_ptrs[i]->self_time;
	}

	//send frames then
	ProfilerFrame metric;
	metric.frame_number = Engine::get_singleton()->get_frames_drawn();
	metric.frame_time = frame_time;
	metric.idle_time = idle_time;
	metric.physics_time = physics_time;
	metric.physics_frame_time = physics_frame_time;
	metric.script_time = total_script_time;

	// Add script functions information.
	metric.frame_functions.resize(to_send);
	FrameFunction *w = metric.frame_functions.ptrw();
	for (int i = 0; i < to_send; i++) {

		if (profiler_function_signature_map.has(profile_info_ptrs[i]->signature)) {
			w[i].sig_id = profiler_function_signature_map[profile_info_ptrs[i]->signature];
		}

		w[i].call_count = profile_info_ptrs[i]->call_count;
		w[i].total_time = profile_info_ptrs[i]->total_time / 1000000.0;
		w[i].self_time = profile_info_ptrs[i]->self_time / 1000000.0;
	}
	if (p_for_frame) {
		// Add profile frame data information.
		metric.frames_data.append_array(profile_frame_data);
		_put_msg("profile_frame", metric.serialize());
		profile_frame_data.clear();
	} else {
		_put_msg("profile_total", metric.serialize());
	}
}

void ScriptDebuggerRemote::idle_poll() {

	// this function is called every frame, except when there is a debugger break (::debug() in this class)
	// execution stops and remains in the ::debug function

	_get_output();

	if (requested_quit) {

		_put_msg("kill_me", Array());
		requested_quit = false;
	}

	if (performance) {

		uint64_t pt = OS::get_singleton()->get_ticks_msec();
		if (pt - last_perf_time > 1000) {

			last_perf_time = pt;
			int max = performance->get("MONITOR_MAX");
			Array arr;
			arr.resize(max);
			for (int i = 0; i < max; i++) {
				arr[i] = performance->call("get_monitor", i);
			}
			_put_msg("performance", arr);
		}
	}

	if (visual_profiling) {
		Vector<VS::FrameProfileArea> profile_areas = VS::get_singleton()->get_frame_profile();
		if (profile_areas.size()) {
			Vector<String> area_names;
			Vector<real_t> area_times;
			area_names.resize(profile_areas.size());
			area_times.resize(profile_areas.size() * 2);
			{
				String *area_namesw = area_names.ptrw();
				real_t *area_timesw = area_times.ptrw();

				for (int i = 0; i < profile_areas.size(); i++) {
					area_namesw[i] = profile_areas[i].name;
					area_timesw[i * 2 + 0] = profile_areas[i].cpu_msec;
					area_timesw[i * 2 + 1] = profile_areas[i].gpu_msec;
				}
			}
			Array msg;
			msg.push_back(VS::get_singleton()->get_frame_profile_frame());
			msg.push_back(area_names);
			msg.push_back(area_times);
			_put_msg("visual_profile", msg);
		}
	}

	if (profiling) {

		if (skip_profile_frame) {
			skip_profile_frame = false;
		} else {
			//send profiling info normally
			_send_profiling_data(true);
		}
	}

	if (network_profiling) {
		uint64_t pt = OS::get_singleton()->get_ticks_msec();
		if (pt - last_net_bandwidth_time > 200) {
			last_net_bandwidth_time = pt;
			_send_network_bandwidth_usage();
		}
		if (pt - last_net_prof_time > 100) {
			last_net_prof_time = pt;
			_send_network_profiling_data();
		}
	}

	if (reload_all_scripts) {

		for (int i = 0; i < ScriptServer::get_language_count(); i++) {
			ScriptServer::get_language(i)->reload_all_scripts();
		}
		reload_all_scripts = false;
	}

	_poll_events();
}

void ScriptDebuggerRemote::_send_network_profiling_data() {
	ERR_FAIL_COND(multiplayer.is_null());

	int n_nodes = multiplayer->get_profiling_frame(&network_profile_info.write[0]);

	NetworkProfilerFrame frame;
	for (int i = 0; i < n_nodes; i++) {
		frame.infos.push_back(network_profile_info[i]);
	}
	_put_msg("network_profile", frame.serialize());
}

void ScriptDebuggerRemote::_send_network_bandwidth_usage() {
	ERR_FAIL_COND(multiplayer.is_null());

	int incoming_bandwidth = multiplayer->get_incoming_bandwidth_usage();
	int outgoing_bandwidth = multiplayer->get_outgoing_bandwidth_usage();

	Array arr;
	arr.push_back(incoming_bandwidth);
	arr.push_back(outgoing_bandwidth);
	_put_msg("network_bandwidth", arr);
}

void ScriptDebuggerRemote::send_message(const String &p_message, const Array &p_args) {

	MutexLock lock(mutex);

	if (!locking && is_peer_connected()) {

		if (messages.size() >= max_messages_per_frame) {
			n_messages_dropped++;
		} else {
			Message msg;
			msg.message = p_message;
			msg.data = p_args;
			messages.push_back(msg);
		}
	}
}

void ScriptDebuggerRemote::send_error(const String &p_func, const String &p_file, int p_line, const String &p_err, const String &p_descr, ErrorHandlerType p_type, const Vector<ScriptLanguage::StackInfo> &p_stack_info) {

	OutputError oe;
	oe.error = p_err;
	oe.error_descr = p_descr;
	oe.source_file = p_file;
	oe.source_line = p_line;
	oe.source_func = p_func;
	oe.warning = p_type == ERR_HANDLER_WARNING;
	uint64_t time = OS::get_singleton()->get_ticks_msec();
	oe.hr = time / 3600000;
	oe.min = (time / 60000) % 60;
	oe.sec = (time / 1000) % 60;
	oe.msec = time % 1000;
	Array cstack;

	uint64_t ticks = OS::get_singleton()->get_ticks_usec() / 1000;
	msec_count += ticks - last_msec;
	last_msec = ticks;

	if (msec_count > 1000) {
		msec_count = 0;

		err_count = 0;
		n_errors_dropped = 0;
		warn_count = 0;
		n_warnings_dropped = 0;
	}

	cstack.resize(p_stack_info.size() * 3);
	for (int i = 0; i < p_stack_info.size(); i++) {
		cstack[i * 3 + 0] = p_stack_info[i].file;
		cstack[i * 3 + 1] = p_stack_info[i].func;
		cstack[i * 3 + 2] = p_stack_info[i].line;
	}

	//oe.callstack = cstack;
	if (oe.warning) {
		warn_count++;
	} else {
		err_count++;
	}

	MutexLock lock(mutex);

	if (!locking && is_peer_connected()) {

		if (oe.warning) {
			if (warn_count > max_warnings_per_second) {
				n_warnings_dropped++;
			} else {
				errors.push_back(oe);
			}
		} else {
			if (err_count > max_errors_per_second) {
				n_errors_dropped++;
			} else {
				errors.push_back(oe);
			}
		}
	}
}

void ScriptDebuggerRemote::_print_handler(void *p_this, const String &p_string, bool p_error) {

	ScriptDebuggerRemote *sdr = (ScriptDebuggerRemote *)p_this;

	uint64_t ticks = OS::get_singleton()->get_ticks_usec() / 1000;
	sdr->msec_count += ticks - sdr->last_msec;
	sdr->last_msec = ticks;

	if (sdr->msec_count > 1000) {
		sdr->char_count = 0;
		sdr->msec_count = 0;
	}

	String s = p_string;
	int allowed_chars = MIN(MAX(sdr->max_cps - sdr->char_count, 0), s.length());

	if (allowed_chars == 0)
		return;

	if (allowed_chars < s.length()) {
		s = s.substr(0, allowed_chars);
	}

	sdr->char_count += allowed_chars;
	bool overflowed = sdr->char_count >= sdr->max_cps;

	{
		MutexLock lock(sdr->mutex);

		if (!sdr->locking && sdr->is_peer_connected()) {

			if (overflowed)
				s += "[...]";

			sdr->output_strings.push_back(s);

			if (overflowed) {
				sdr->output_strings.push_back("[output overflow, print less text!]");
			}
		}
	}
}

void ScriptDebuggerRemote::request_quit() {

	requested_quit = true;
}

void ScriptDebuggerRemote::set_multiplayer(Ref<MultiplayerAPI> p_multiplayer) {
	multiplayer = p_multiplayer;
}

bool ScriptDebuggerRemote::is_profiling() const {

	return profiling;
}
void ScriptDebuggerRemote::add_profiling_frame_data(const StringName &p_name, const Array &p_data) {

	int idx = -1;
	for (int i = 0; i < profile_frame_data.size(); i++) {
		if (profile_frame_data[i].name == p_name) {
			idx = i;
			break;
		}
	}

	FrameData fd;
	fd.name = p_name;
	fd.data = p_data;

	if (idx == -1) {
		profile_frame_data.push_back(fd);
	} else {
		profile_frame_data.write[idx] = fd;
	}
}

void ScriptDebuggerRemote::profiling_start() {
	//ignores this, uses it via connection
}

void ScriptDebuggerRemote::profiling_end() {
	//ignores this, uses it via connection
}

void ScriptDebuggerRemote::profiling_set_frame_times(float p_frame_time, float p_idle_time, float p_physics_time, float p_physics_frame_time) {

	frame_time = p_frame_time;
	idle_time = p_idle_time;
	physics_time = p_physics_time;
	physics_frame_time = p_physics_frame_time;
}

void ScriptDebuggerRemote::set_skip_breakpoints(bool p_skip_breakpoints) {
	skip_breakpoints = p_skip_breakpoints;
}

ScriptDebuggerRemote::ResourceUsageFunc ScriptDebuggerRemote::resource_usage_func = NULL;
ScriptDebuggerRemote::ParseMessageFunc ScriptDebuggerRemote::scene_tree_parse_func = NULL;

ScriptDebuggerRemote::ScriptDebuggerRemote() :
		profiling(false),
		visual_profiling(false),
		network_profiling(false),
		max_frame_functions(16),
		skip_profile_frame(false),
		reload_all_scripts(false),
		tcp_client(Ref<StreamPeerTCP>(memnew(StreamPeerTCP))),
		packet_peer_stream(Ref<PacketPeerStream>(memnew(PacketPeerStream))),
		last_perf_time(0),
		last_net_prof_time(0),
		last_net_bandwidth_time(0),
		performance(Engine::get_singleton()->get_singleton_object("Performance")),
		requested_quit(false),
		max_messages_per_frame(GLOBAL_GET("network/limits/debugger_stdout/max_messages_per_frame")),
		n_messages_dropped(0),
		max_errors_per_second(GLOBAL_GET("network/limits/debugger_stdout/max_errors_per_second")),
		max_warnings_per_second(GLOBAL_GET("network/limits/debugger_stdout/max_warnings_per_second")),
		n_errors_dropped(0),
		max_cps(GLOBAL_GET("network/limits/debugger_stdout/max_chars_per_second")),
		char_count(0),
		err_count(0),
		warn_count(0),
		last_msec(0),
		msec_count(0),
		locking(false),
		poll_every(0) {

	packet_peer_stream->set_stream_peer(tcp_client);
	packet_peer_stream->set_output_buffer_max_size((1024 * 1024 * 8) - 4); // 8 MiB should be way more than enough, minus 4 bytes for separator.

	phl.printfunc = _print_handler;
	phl.userdata = this;
	add_print_handler(&phl);

	eh.errfunc = _err_handler;
	eh.userdata = this;
	add_error_handler(&eh);

	profile_info.resize(GLOBAL_GET("debug/settings/profiler/max_functions"));
	network_profile_info.resize(GLOBAL_GET("debug/settings/profiler/max_functions"));
	profile_info_ptrs.resize(profile_info.size());
}

ScriptDebuggerRemote::~ScriptDebuggerRemote() {

	remove_print_handler(&phl);
	remove_error_handler(&eh);
}
