/*************************************************************************/
/*  remote_debugger.cpp                                                  */
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

#include "remote_debugger.h"

#include "core/config/project_settings.h"
#include "core/debugger/debugger_marshalls.h"
#include "core/debugger/engine_debugger.h"
#include "core/debugger/script_debugger.h"
#include "core/input/input.h"
#include "core/object/script_language.h"
#include "core/os/os.h"
#include "scene/main/node.h"
#include "servers/display_server.h"

template <typename T>
void RemoteDebugger::_bind_profiler(const String &p_name, T *p_prof) {
	EngineDebugger::Profiler prof(
			p_prof,
			[](void *p_user, bool p_enable, const Array &p_opts) {
				((T *)p_user)->toggle(p_enable, p_opts);
			},
			[](void *p_user, const Array &p_data) {
				((T *)p_user)->add(p_data);
			},
			[](void *p_user, double p_frame_time, double p_idle_time, double p_physics_time, double p_physics_frame_time) {
				((T *)p_user)->tick(p_frame_time, p_idle_time, p_physics_time, p_physics_frame_time);
			});
	EngineDebugger::register_profiler(p_name, prof);
}

struct RemoteDebugger::NetworkProfiler {
public:
	typedef DebuggerMarshalls::MultiplayerNodeInfo NodeInfo;
	struct BandwidthFrame {
		uint32_t timestamp;
		int packet_size;
	};

	int bandwidth_in_ptr = 0;
	Vector<BandwidthFrame> bandwidth_in;
	int bandwidth_out_ptr = 0;
	Vector<BandwidthFrame> bandwidth_out;
	uint64_t last_bandwidth_time = 0;

	Map<ObjectID, NodeInfo> multiplayer_node_data;
	uint64_t last_profile_time = 0;

	NetworkProfiler() {}

	int bandwidth_usage(const Vector<BandwidthFrame> &p_buffer, int p_pointer) {
		ERR_FAIL_COND_V(p_buffer.size() == 0, 0);
		int total_bandwidth = 0;

		uint32_t timestamp = OS::get_singleton()->get_ticks_msec();
		uint32_t final_timestamp = timestamp - 1000;

		int i = (p_pointer + p_buffer.size() - 1) % p_buffer.size();

		while (i != p_pointer && p_buffer[i].packet_size > 0) {
			if (p_buffer[i].timestamp < final_timestamp) {
				return total_bandwidth;
			}
			total_bandwidth += p_buffer[i].packet_size;
			i = (i + p_buffer.size() - 1) % p_buffer.size();
		}

		ERR_FAIL_COND_V_MSG(i == p_pointer, total_bandwidth, "Reached the end of the bandwidth profiler buffer, values might be inaccurate.");
		return total_bandwidth;
	}

	void init_node(const ObjectID p_node) {
		if (multiplayer_node_data.has(p_node)) {
			return;
		}
		multiplayer_node_data.insert(p_node, DebuggerMarshalls::MultiplayerNodeInfo());
		multiplayer_node_data[p_node].node = p_node;
		multiplayer_node_data[p_node].node_path = Object::cast_to<Node>(ObjectDB::get_instance(p_node))->get_path();
		multiplayer_node_data[p_node].incoming_rpc = 0;
		multiplayer_node_data[p_node].incoming_rset = 0;
		multiplayer_node_data[p_node].outgoing_rpc = 0;
		multiplayer_node_data[p_node].outgoing_rset = 0;
	}

	void toggle(bool p_enable, const Array &p_opts) {
		multiplayer_node_data.clear();

		if (!p_enable) {
			bandwidth_in.clear();
			bandwidth_out.clear();
		} else {
			bandwidth_in_ptr = 0;
			bandwidth_in.resize(16384); // ~128kB
			for (int i = 0; i < bandwidth_in.size(); ++i) {
				bandwidth_in.write[i].packet_size = -1;
			}
			bandwidth_out_ptr = 0;
			bandwidth_out.resize(16384); // ~128kB
			for (int i = 0; i < bandwidth_out.size(); ++i) {
				bandwidth_out.write[i].packet_size = -1;
			}
		}
	}

	void add(const Array &p_data) {
		ERR_FAIL_COND(p_data.size() < 1);
		const String type = p_data[0];
		if (type == "node") {
			ERR_FAIL_COND(p_data.size() < 3);
			const ObjectID id = p_data[1];
			const String what = p_data[2];
			init_node(id);
			NodeInfo &info = multiplayer_node_data[id];
			if (what == "rpc_in") {
				info.incoming_rpc++;
			} else if (what == "rpc_out") {
				info.outgoing_rpc++;
			} else if (what == "rset_in") {
				info.incoming_rset = 0;
			} else if (what == "rset_out") {
				info.outgoing_rset++;
			}
		} else if (type == "bandwidth") {
			ERR_FAIL_COND(p_data.size() < 4);
			const String inout = p_data[1];
			int time = p_data[2];
			int size = p_data[3];
			if (inout == "in") {
				bandwidth_in.write[bandwidth_in_ptr].timestamp = time;
				bandwidth_in.write[bandwidth_in_ptr].packet_size = size;
				bandwidth_in_ptr = (bandwidth_in_ptr + 1) % bandwidth_in.size();
			} else if (inout == "out") {
				bandwidth_out.write[bandwidth_out_ptr].timestamp = time;
				bandwidth_out.write[bandwidth_out_ptr].packet_size = size;
				bandwidth_out_ptr = (bandwidth_out_ptr + 1) % bandwidth_out.size();
			}
		}
	}

	void tick(double p_frame_time, double p_idle_time, double p_physics_time, double p_physics_frame_time) {
		uint64_t pt = OS::get_singleton()->get_ticks_msec();
		if (pt - last_bandwidth_time > 200) {
			last_bandwidth_time = pt;
			int incoming_bandwidth = bandwidth_usage(bandwidth_in, bandwidth_in_ptr);
			int outgoing_bandwidth = bandwidth_usage(bandwidth_out, bandwidth_out_ptr);

			Array arr;
			arr.push_back(incoming_bandwidth);
			arr.push_back(outgoing_bandwidth);
			EngineDebugger::get_singleton()->send_message("network:bandwidth", arr);
		}
		if (pt - last_profile_time > 100) {
			last_profile_time = pt;
			DebuggerMarshalls::NetworkProfilerFrame frame;
			for (const KeyValue<ObjectID, NodeInfo> &E : multiplayer_node_data) {
				frame.infos.push_back(E.value);
			}
			multiplayer_node_data.clear();
			EngineDebugger::get_singleton()->send_message("network:profile_frame", frame.serialize());
		}
	}
};

struct RemoteDebugger::ScriptsProfiler {
	typedef DebuggerMarshalls::ScriptFunctionSignature FunctionSignature;
	typedef DebuggerMarshalls::ScriptFunctionInfo FunctionInfo;
	struct ProfileInfoSort {
		bool operator()(ScriptLanguage::ProfilingInfo *A, ScriptLanguage::ProfilingInfo *B) const {
			return A->total_time < B->total_time;
		}
	};
	Vector<ScriptLanguage::ProfilingInfo> info;
	Vector<ScriptLanguage::ProfilingInfo *> ptrs;
	Map<StringName, int> sig_map;
	int max_frame_functions = 16;

	void toggle(bool p_enable, const Array &p_opts) {
		if (p_enable) {
			sig_map.clear();
			for (int i = 0; i < ScriptServer::get_language_count(); i++) {
				ScriptServer::get_language(i)->profiling_start();
			}
			if (p_opts.size() == 1 && p_opts[0].get_type() == Variant::INT) {
				max_frame_functions = MAX(0, int(p_opts[0]));
			}
		} else {
			for (int i = 0; i < ScriptServer::get_language_count(); i++) {
				ScriptServer::get_language(i)->profiling_stop();
			}
		}
	}

	void write_frame_data(Vector<FunctionInfo> &r_funcs, uint64_t &r_total, bool p_accumulated) {
		int ofs = 0;
		for (int i = 0; i < ScriptServer::get_language_count(); i++) {
			if (p_accumulated) {
				ofs += ScriptServer::get_language(i)->profiling_get_accumulated_data(&info.write[ofs], info.size() - ofs);
			} else {
				ofs += ScriptServer::get_language(i)->profiling_get_frame_data(&info.write[ofs], info.size() - ofs);
			}
		}

		for (int i = 0; i < ofs; i++) {
			ptrs.write[i] = &info.write[i];
		}

		SortArray<ScriptLanguage::ProfilingInfo *, ProfileInfoSort> sa;
		sa.sort(ptrs.ptrw(), ofs);

		int to_send = MIN(ofs, max_frame_functions);

		// Check signatures first, and compute total time.
		r_total = 0;
		for (int i = 0; i < to_send; i++) {
			if (!sig_map.has(ptrs[i]->signature)) {
				int idx = sig_map.size();
				FunctionSignature sig;
				sig.name = ptrs[i]->signature;
				sig.id = idx;
				EngineDebugger::get_singleton()->send_message("servers:function_signature", sig.serialize());
				sig_map[ptrs[i]->signature] = idx;
			}
			r_total += ptrs[i]->self_time;
		}

		// Send frame, script time, functions information then
		r_funcs.resize(to_send);

		FunctionInfo *w = r_funcs.ptrw();
		for (int i = 0; i < to_send; i++) {
			if (sig_map.has(ptrs[i]->signature)) {
				w[i].sig_id = sig_map[ptrs[i]->signature];
			}
			w[i].call_count = ptrs[i]->call_count;
			w[i].total_time = ptrs[i]->total_time / 1000000.0;
			w[i].self_time = ptrs[i]->self_time / 1000000.0;
		}
	}

	ScriptsProfiler() {
		info.resize(GLOBAL_GET("debug/settings/profiler/max_functions"));
		ptrs.resize(info.size());
	}
};

struct RemoteDebugger::ServersProfiler {
	bool skip_profile_frame = false;
	typedef DebuggerMarshalls::ServerInfo ServerInfo;
	typedef DebuggerMarshalls::ServerFunctionInfo ServerFunctionInfo;

	Map<StringName, ServerInfo> server_data;
	ScriptsProfiler scripts_profiler;

	double frame_time = 0;
	double idle_time = 0;
	double physics_time = 0;
	double physics_frame_time = 0;

	void toggle(bool p_enable, const Array &p_opts) {
		skip_profile_frame = false;
		if (p_enable) {
			server_data.clear(); // Clear old profiling data.
		} else {
			_send_frame_data(true); // Send final frame.
		}
		scripts_profiler.toggle(p_enable, p_opts);
	}

	void add(const Array &p_data) {
		String name = p_data[0];
		if (!server_data.has(name)) {
			ServerInfo info;
			info.name = name;
			server_data[name] = info;
		}
		ServerInfo &srv = server_data[name];

		ServerFunctionInfo fi;
		fi.name = p_data[1];
		fi.time = p_data[2];
		srv.functions.push_back(fi);
	}

	void tick(double p_frame_time, double p_idle_time, double p_physics_time, double p_physics_frame_time) {
		frame_time = p_frame_time;
		idle_time = p_idle_time;
		physics_time = p_physics_time;
		physics_frame_time = p_physics_frame_time;
		_send_frame_data(false);
	}

	void _send_frame_data(bool p_final) {
		DebuggerMarshalls::ServersProfilerFrame frame;
		frame.frame_number = Engine::get_singleton()->get_process_frames();
		frame.frame_time = frame_time;
		frame.idle_time = idle_time;
		frame.physics_time = physics_time;
		frame.physics_frame_time = physics_frame_time;
		Map<StringName, ServerInfo>::Element *E = server_data.front();
		while (E) {
			if (!p_final) {
				frame.servers.push_back(E->get());
			}
			E->get().functions.clear();
			E = E->next();
		}
		uint64_t time = 0;
		scripts_profiler.write_frame_data(frame.script_functions, time, p_final);
		frame.script_time = USEC_TO_SEC(time);
		if (skip_profile_frame) {
			skip_profile_frame = false;
			return;
		}
		if (p_final) {
			EngineDebugger::get_singleton()->send_message("servers:profile_total", frame.serialize());
		} else {
			EngineDebugger::get_singleton()->send_message("servers:profile_frame", frame.serialize());
		}
	}
};

struct RemoteDebugger::VisualProfiler {
	typedef DebuggerMarshalls::ServerInfo ServerInfo;
	typedef DebuggerMarshalls::ServerFunctionInfo ServerFunctionInfo;

	Map<StringName, ServerInfo> server_data;

	void toggle(bool p_enable, const Array &p_opts) {
		RS::get_singleton()->set_frame_profiling_enabled(p_enable);
	}

	void add(const Array &p_data) {}

	void tick(double p_frame_time, double p_idle_time, double p_physics_time, double p_physics_frame_time) {
		Vector<RS::FrameProfileArea> profile_areas = RS::get_singleton()->get_frame_profile();
		DebuggerMarshalls::VisualProfilerFrame frame;
		if (!profile_areas.size()) {
			return;
		}

		frame.frame_number = RS::get_singleton()->get_frame_profile_frame();
		frame.areas.append_array(profile_areas);
		EngineDebugger::get_singleton()->send_message("visual:profile_frame", frame.serialize());
	}
};

struct RemoteDebugger::PerformanceProfiler {
	Object *performance = nullptr;
	int last_perf_time = 0;
	uint64_t last_monitor_modification_time = 0;

	void toggle(bool p_enable, const Array &p_opts) {}
	void add(const Array &p_data) {}
	void tick(double p_frame_time, double p_idle_time, double p_physics_time, double p_physics_frame_time) {
		if (!performance) {
			return;
		}

		uint64_t pt = OS::get_singleton()->get_ticks_msec();
		if (pt - last_perf_time < 1000) {
			return;
		}
		last_perf_time = pt;

		Array custom_monitor_names = performance->call("get_custom_monitor_names");

		uint64_t monitor_modification_time = performance->call("get_monitor_modification_time");
		if (monitor_modification_time > last_monitor_modification_time) {
			last_monitor_modification_time = monitor_modification_time;
			EngineDebugger::get_singleton()->send_message("performance:profile_names", custom_monitor_names);
		}

		int max = performance->get("MONITOR_MAX");
		Array arr;
		arr.resize(max + custom_monitor_names.size());
		for (int i = 0; i < max; i++) {
			arr[i] = performance->call("get_monitor", i);
		}

		for (int i = 0; i < custom_monitor_names.size(); i++) {
			Variant monitor_value = performance->call("get_custom_monitor", custom_monitor_names[i]);
			if (!monitor_value.is_num()) {
				ERR_PRINT("Value of custom monitor '" + String(custom_monitor_names[i]) + "' is not a number");
				arr[i + max] = Variant();
			}
			arr[i + max] = monitor_value;
		}

		EngineDebugger::get_singleton()->send_message("performance:profile_frame", arr);
	}

	PerformanceProfiler(Object *p_performance) {
		performance = p_performance;
	}
};

void RemoteDebugger::_send_resource_usage() {
	DebuggerMarshalls::ResourceUsage usage;

	List<RS::TextureInfo> tinfo;
	RS::get_singleton()->texture_debug_usage(&tinfo);

	for (const RS::TextureInfo &E : tinfo) {
		DebuggerMarshalls::ResourceInfo info;
		info.path = E.path;
		info.vram = E.bytes;
		info.id = E.texture;
		info.type = "Texture";
		if (E.depth == 0) {
			info.format = itos(E.width) + "x" + itos(E.height) + " " + Image::get_format_name(E.format);
		} else {
			info.format = itos(E.width) + "x" + itos(E.height) + "x" + itos(E.depth) + " " + Image::get_format_name(E.format);
		}
		usage.infos.push_back(info);
	}

	EngineDebugger::get_singleton()->send_message("memory:usage", usage.serialize());
}

Error RemoteDebugger::_put_msg(String p_message, Array p_data) {
	Array msg;
	msg.push_back(p_message);
	msg.push_back(p_data);
	Error err = peer->put_message(msg);
	if (err != OK) {
		n_messages_dropped++;
	}
	return err;
}

void RemoteDebugger::_err_handler(void *p_this, const char *p_func, const char *p_file, int p_line, const char *p_err, const char *p_descr, bool p_editor_notify, ErrorHandlerType p_type) {
	if (p_type == ERR_HANDLER_SCRIPT) {
		return; //ignore script errors, those go through debugger
	}

	RemoteDebugger *rd = (RemoteDebugger *)p_this;
	if (rd->flushing && Thread::get_caller_id() == rd->flush_thread) { // Can't handle recursive errors during flush.
		return;
	}

	Vector<ScriptLanguage::StackInfo> si;

	for (int i = 0; i < ScriptServer::get_language_count(); i++) {
		si = ScriptServer::get_language(i)->debug_get_current_stack_info();
		if (si.size()) {
			break;
		}
	}

	// send_error will lock internally.
	rd->script_debugger->send_error(p_func, p_file, p_line, p_err, p_descr, p_editor_notify, p_type, si);
}

void RemoteDebugger::_print_handler(void *p_this, const String &p_string, bool p_error) {
	RemoteDebugger *rd = (RemoteDebugger *)p_this;

	if (rd->flushing && Thread::get_caller_id() == rd->flush_thread) { // Can't handle recursive prints during flush.
		return;
	}

	String s = p_string;
	int allowed_chars = MIN(MAX(rd->max_chars_per_second - rd->char_count, 0), s.length());

	if (allowed_chars == 0 && s.length() > 0) {
		return;
	}

	if (allowed_chars < s.length()) {
		s = s.substr(0, allowed_chars);
	}

	MutexLock lock(rd->mutex);

	rd->char_count += allowed_chars;
	bool overflowed = rd->char_count >= rd->max_chars_per_second;
	if (rd->is_peer_connected()) {
		if (overflowed) {
			s += "[...]";
		}

		OutputString output_string;
		output_string.message = s;
		output_string.type = p_error ? MESSAGE_TYPE_ERROR : MESSAGE_TYPE_LOG;
		rd->output_strings.push_back(output_string);

		if (overflowed) {
			output_string.message = "[output overflow, print less text!]";
			output_string.type = MESSAGE_TYPE_ERROR;
			rd->output_strings.push_back(output_string);
		}
	}
}

RemoteDebugger::ErrorMessage RemoteDebugger::_create_overflow_error(const String &p_what, const String &p_descr) {
	ErrorMessage oe;
	oe.error = p_what;
	oe.error_descr = p_descr;
	oe.warning = false;
	uint64_t time = OS::get_singleton()->get_ticks_msec();
	oe.hr = time / 3600000;
	oe.min = (time / 60000) % 60;
	oe.sec = (time / 1000) % 60;
	oe.msec = time % 1000;
	return oe;
}

void RemoteDebugger::flush_output() {
	flush_thread = Thread::get_caller_id();
	flushing = true;
	MutexLock lock(mutex);
	if (!is_peer_connected()) {
		return;
	}

	if (n_messages_dropped > 0) {
		ErrorMessage err_msg = _create_overflow_error("TOO_MANY_MESSAGES", "Too many messages! " + String::num_int64(n_messages_dropped) + " messages were dropped. Profiling might misbheave, try raising 'network/limits/debugger/max_queued_messages' in project setting.");
		if (_put_msg("error", err_msg.serialize()) == OK) {
			n_messages_dropped = 0;
		}
	}

	if (output_strings.size()) {
		// Join output strings so we generate less messages.
		Vector<String> joined_log_strings;
		Vector<String> strings;
		Vector<int> types;
		for (int i = 0; i < output_strings.size(); i++) {
			const OutputString &output_string = output_strings[i];
			if (output_string.type == MESSAGE_TYPE_ERROR) {
				if (!joined_log_strings.is_empty()) {
					strings.push_back(String("\n").join(joined_log_strings));
					types.push_back(MESSAGE_TYPE_LOG);
					joined_log_strings.clear();
				}
				strings.push_back(output_string.message);
				types.push_back(MESSAGE_TYPE_ERROR);
			} else {
				joined_log_strings.push_back(output_string.message);
			}
		}

		if (!joined_log_strings.is_empty()) {
			strings.push_back(String("\n").join(joined_log_strings));
			types.push_back(MESSAGE_TYPE_LOG);
		}

		Array arr;
		arr.push_back(strings);
		arr.push_back(types);
		_put_msg("output", arr);
		output_strings.clear();
	}

	while (errors.size()) {
		ErrorMessage oe = errors.front()->get();
		_put_msg("error", oe.serialize());
		errors.pop_front();
	}

	// Update limits
	uint64_t ticks = OS::get_singleton()->get_ticks_usec() / 1000;

	if (ticks - last_reset > 1000) {
		last_reset = ticks;
		char_count = 0;
		err_count = 0;
		n_errors_dropped = 0;
		warn_count = 0;
		n_warnings_dropped = 0;
	}
	flushing = false;
}

void RemoteDebugger::send_message(const String &p_message, const Array &p_args) {
	MutexLock lock(mutex);
	if (is_peer_connected()) {
		_put_msg(p_message, p_args);
	}
}

void RemoteDebugger::send_error(const String &p_func, const String &p_file, int p_line, const String &p_err, const String &p_descr, bool p_editor_notify, ErrorHandlerType p_type) {
	ErrorMessage oe;
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
	oe.callstack.append_array(script_debugger->get_error_stack_info());

	if (flushing && Thread::get_caller_id() == flush_thread) { // Can't handle recursive errors during flush.
		return;
	}

	MutexLock lock(mutex);

	if (oe.warning) {
		warn_count++;
	} else {
		err_count++;
	}

	if (is_peer_connected()) {
		if (oe.warning) {
			if (warn_count > max_warnings_per_second) {
				n_warnings_dropped++;
				if (n_warnings_dropped == 1) {
					// Only print one message about dropping per second
					ErrorMessage overflow = _create_overflow_error("TOO_MANY_WARNINGS", "Too many warnings! Ignoring warnings for up to 1 second.");
					errors.push_back(overflow);
				}
			} else {
				errors.push_back(oe);
			}
		} else {
			if (err_count > max_errors_per_second) {
				n_errors_dropped++;
				if (n_errors_dropped == 1) {
					// Only print one message about dropping per second
					ErrorMessage overflow = _create_overflow_error("TOO_MANY_ERRORS", "Too many errors! Ignoring errors for up to 1 second.");
					errors.push_back(overflow);
				}
			} else {
				errors.push_back(oe);
			}
		}
	}
}

void RemoteDebugger::_send_stack_vars(List<String> &p_names, List<Variant> &p_vals, int p_type) {
	DebuggerMarshalls::ScriptStackVariable stvar;
	List<String>::Element *E = p_names.front();
	List<Variant>::Element *F = p_vals.front();
	while (E) {
		stvar.name = E->get();
		stvar.value = F->get();
		stvar.type = p_type;
		send_message("stack_frame_var", stvar.serialize());
		E = E->next();
		F = F->next();
	}
}

Error RemoteDebugger::_try_capture(const String &p_msg, const Array &p_data, bool &r_captured) {
	const int idx = p_msg.find(":");
	r_captured = false;
	if (idx < 0) { // No prefix, unknown message.
		return OK;
	}
	const String cap = p_msg.substr(0, idx);
	if (!has_capture(cap)) {
		return ERR_UNAVAILABLE; // Unknown message...
	}
	const String msg = p_msg.substr(idx + 1);
	return capture_parse(cap, msg, p_data, r_captured);
}

void RemoteDebugger::debug(bool p_can_continue, bool p_is_error_breakpoint) {
	//this function is called when there is a debugger break (bug on script)
	//or when execution is paused from editor

	if (script_debugger->is_skipping_breakpoints() && !p_is_error_breakpoint) {
		return;
	}

	ERR_FAIL_COND_MSG(!is_peer_connected(), "Script Debugger failed to connect, but being used anyway.");

	if (!peer->can_block()) {
		return; // Peer does not support blocking IO. We could at least send the error though.
	}

	ScriptLanguage *script_lang = script_debugger->get_break_language();
	const String error_str = script_lang ? script_lang->debug_get_error() : "";
	Array msg;
	msg.push_back(p_can_continue);
	msg.push_back(error_str);
	ERR_FAIL_COND(!script_lang);
	msg.push_back(script_lang->debug_get_stack_level_count() > 0);
	send_message("debug_enter", msg);

	servers_profiler->skip_profile_frame = true; // Avoid frame time spike in debug.

	Input::MouseMode mouse_mode = Input::get_singleton()->get_mouse_mode();
	if (mouse_mode != Input::MOUSE_MODE_VISIBLE) {
		Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_VISIBLE);
	}

	uint64_t loop_begin_usec = 0;
	uint64_t loop_time_sec = 0;
	while (is_peer_connected()) {
		loop_begin_usec = OS::get_singleton()->get_ticks_usec();

		flush_output();
		peer->poll();

		if (peer->has_message()) {
			Array cmd = peer->get_message();

			ERR_CONTINUE(cmd.size() != 2);
			ERR_CONTINUE(cmd[0].get_type() != Variant::STRING);
			ERR_CONTINUE(cmd[1].get_type() != Variant::ARRAY);

			String command = cmd[0];
			Array data = cmd[1];

			if (command == "step") {
				script_debugger->set_depth(-1);
				script_debugger->set_lines_left(1);
				break;

			} else if (command == "next") {
				script_debugger->set_depth(0);
				script_debugger->set_lines_left(1);
				break;

			} else if (command == "continue") {
				script_debugger->set_depth(-1);
				script_debugger->set_lines_left(-1);
				DisplayServer::get_singleton()->window_move_to_foreground();
				break;

			} else if (command == "break") {
				ERR_PRINT("Got break when already broke!");
				break;

			} else if (command == "get_stack_dump") {
				DebuggerMarshalls::ScriptStackDump dump;
				int slc = script_lang->debug_get_stack_level_count();
				for (int i = 0; i < slc; i++) {
					ScriptLanguage::StackInfo frame;
					frame.file = script_lang->debug_get_stack_level_source(i);
					frame.line = script_lang->debug_get_stack_level_line(i);
					frame.func = script_lang->debug_get_stack_level_function(i);
					dump.frames.push_back(frame);
				}
				send_message("stack_dump", dump.serialize());

			} else if (command == "get_stack_frame_vars") {
				ERR_FAIL_COND(data.size() != 1);
				ERR_FAIL_COND(!script_lang);
				int lv = data[0];

				List<String> members;
				List<Variant> member_vals;
				if (ScriptInstance *inst = script_lang->debug_get_stack_level_instance(lv)) {
					members.push_back("self");
					member_vals.push_back(inst->get_owner());
				}
				script_lang->debug_get_stack_level_members(lv, &members, &member_vals);
				ERR_FAIL_COND(members.size() != member_vals.size());

				List<String> locals;
				List<Variant> local_vals;
				script_lang->debug_get_stack_level_locals(lv, &locals, &local_vals);
				ERR_FAIL_COND(locals.size() != local_vals.size());

				List<String> globals;
				List<Variant> globals_vals;
				script_lang->debug_get_globals(&globals, &globals_vals);
				ERR_FAIL_COND(globals.size() != globals_vals.size());

				Array var_size;
				var_size.push_back(local_vals.size() + member_vals.size() + globals_vals.size());
				send_message("stack_frame_vars", var_size);
				_send_stack_vars(locals, local_vals, 0);
				_send_stack_vars(members, member_vals, 1);
				_send_stack_vars(globals, globals_vals, 2);

			} else if (command == "reload_scripts") {
				reload_all_scripts = true;

			} else if (command == "breakpoint") {
				ERR_FAIL_COND(data.size() < 3);
				bool set = data[2];
				if (set) {
					script_debugger->insert_breakpoint(data[1], data[0]);
				} else {
					script_debugger->remove_breakpoint(data[1], data[0]);
				}

			} else if (command == "set_skip_breakpoints") {
				ERR_FAIL_COND(data.size() < 1);
				script_debugger->set_skip_breakpoints(data[0]);
			} else {
				bool captured = false;
				ERR_CONTINUE(_try_capture(command, data, captured) != OK);
				if (!captured) {
					WARN_PRINT("Unknown message received from debugger: " + command);
				}
			}
		} else {
			OS::get_singleton()->delay_usec(10000);
			OS::get_singleton()->process_and_drop_events();
		}

		// This is for the camera override to stay live even when the game is paused from the editor
		loop_time_sec = (OS::get_singleton()->get_ticks_usec() - loop_begin_usec) / 1000000.0f;
		RenderingServer::get_singleton()->sync();
		if (RenderingServer::get_singleton()->has_changed()) {
			RenderingServer::get_singleton()->draw(true, loop_time_sec * Engine::get_singleton()->get_time_scale());
		}
	}

	send_message("debug_exit", Array());

	if (mouse_mode != Input::MOUSE_MODE_VISIBLE) {
		Input::get_singleton()->set_mouse_mode(mouse_mode);
	}
}

void RemoteDebugger::poll_events(bool p_is_idle) {
	if (peer.is_null()) {
		return;
	}

	flush_output();
	peer->poll();
	while (peer->has_message()) {
		Array arr = peer->get_message();

		ERR_CONTINUE(arr.size() != 2);
		ERR_CONTINUE(arr[0].get_type() != Variant::STRING);
		ERR_CONTINUE(arr[1].get_type() != Variant::ARRAY);

		const String cmd = arr[0];
		const int idx = cmd.find(":");
		bool parsed = false;
		if (idx < 0) { // Not prefix, use scripts capture.
			capture_parse("core", cmd, arr[1], parsed);
			continue;
		}

		const String cap = cmd.substr(0, idx);
		if (!has_capture(cap)) {
			continue; // Unknown message...
		}

		const String msg = cmd.substr(idx + 1);
		capture_parse(cap, msg, arr[1], parsed);
	}

	// Reload scripts during idle poll only.
	if (p_is_idle && reload_all_scripts) {
		for (int i = 0; i < ScriptServer::get_language_count(); i++) {
			ScriptServer::get_language(i)->reload_all_scripts();
		}
		reload_all_scripts = false;
	}
}

Error RemoteDebugger::_core_capture(const String &p_cmd, const Array &p_data, bool &r_captured) {
	r_captured = true;
	if (p_cmd == "reload_scripts") {
		reload_all_scripts = true;

	} else if (p_cmd == "breakpoint") {
		ERR_FAIL_COND_V(p_data.size() < 3, ERR_INVALID_DATA);
		bool set = p_data[2];
		if (set) {
			script_debugger->insert_breakpoint(p_data[1], p_data[0]);
		} else {
			script_debugger->remove_breakpoint(p_data[1], p_data[0]);
		}

	} else if (p_cmd == "set_skip_breakpoints") {
		ERR_FAIL_COND_V(p_data.size() < 1, ERR_INVALID_DATA);
		script_debugger->set_skip_breakpoints(p_data[0]);
	} else if (p_cmd == "memory") {
		_send_resource_usage();
	} else if (p_cmd == "break") {
		script_debugger->debug(script_debugger->get_break_language());
	} else {
		r_captured = false;
	}
	return OK;
}

Error RemoteDebugger::_profiler_capture(const String &p_cmd, const Array &p_data, bool &r_captured) {
	r_captured = false;
	ERR_FAIL_COND_V(p_data.size() < 1, ERR_INVALID_DATA);
	ERR_FAIL_COND_V(p_data[0].get_type() != Variant::BOOL, ERR_INVALID_DATA);
	ERR_FAIL_COND_V(!has_profiler(p_cmd), ERR_UNAVAILABLE);
	Array opts;
	if (p_data.size() > 1) { // Optional profiler parameters.
		ERR_FAIL_COND_V(p_data[1].get_type() != Variant::ARRAY, ERR_INVALID_DATA);
		opts = p_data[1];
	}
	r_captured = true;
	profiler_enable(p_cmd, p_data[0], opts);
	return OK;
}

RemoteDebugger::RemoteDebugger(Ref<RemoteDebuggerPeer> p_peer) {
	peer = p_peer;
	max_chars_per_second = GLOBAL_GET("network/limits/debugger/max_chars_per_second");
	max_errors_per_second = GLOBAL_GET("network/limits/debugger/max_errors_per_second");
	max_warnings_per_second = GLOBAL_GET("network/limits/debugger/max_warnings_per_second");

	// Network Profiler
	network_profiler = memnew(NetworkProfiler);
	_bind_profiler("network", network_profiler);

	// Servers Profiler (audio/physics/...)
	servers_profiler = memnew(ServersProfiler);
	_bind_profiler("servers", servers_profiler);

	// Visual Profiler (cpu/gpu times)
	visual_profiler = memnew(VisualProfiler);
	_bind_profiler("visual", visual_profiler);

	// Performance Profiler
	Object *perf = Engine::get_singleton()->get_singleton_object("Performance");
	if (perf) {
		performance_profiler = memnew(PerformanceProfiler(perf));
		_bind_profiler("performance", performance_profiler);
		profiler_enable("performance", true);
	}

	// Core and profiler captures.
	Capture core_cap(this,
			[](void *p_user, const String &p_cmd, const Array &p_data, bool &r_captured) {
				return ((RemoteDebugger *)p_user)->_core_capture(p_cmd, p_data, r_captured);
			});
	register_message_capture("core", core_cap);
	Capture profiler_cap(this,
			[](void *p_user, const String &p_cmd, const Array &p_data, bool &r_captured) {
				return ((RemoteDebugger *)p_user)->_profiler_capture(p_cmd, p_data, r_captured);
			});
	register_message_capture("profiler", profiler_cap);

	// Error handlers
	phl.printfunc = _print_handler;
	phl.userdata = this;
	add_print_handler(&phl);

	eh.errfunc = _err_handler;
	eh.userdata = this;
	add_error_handler(&eh);
}

RemoteDebugger::~RemoteDebugger() {
	remove_print_handler(&phl);
	remove_error_handler(&eh);

	EngineDebugger::get_singleton()->unregister_profiler("servers");
	EngineDebugger::get_singleton()->unregister_profiler("network");
	EngineDebugger::get_singleton()->unregister_profiler("visual");
	if (EngineDebugger::has_profiler("performance")) {
		EngineDebugger::get_singleton()->unregister_profiler("performance");
	}
	memdelete(servers_profiler);
	memdelete(network_profiler);
	memdelete(visual_profiler);
	if (performance_profiler) {
		memdelete(performance_profiler);
	}
}
