/**************************************************************************/
/*  script_debugger_remote.cpp                                            */
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

#include "script_debugger_remote.h"

#include "core/engine.h"
#include "core/io/marshalls.h"
#include "core/os/input.h"
#include "core/os/os.h"
#include "core/project_settings.h"
#include "scene/main/node.h"
#include "scene/main/scene_tree.h"
#include "scene/main/viewport.h"
#include "scene/resources/packed_scene.h"
#include "servers/visual_server.h"

void ScriptDebuggerRemote::_send_video_memory(uint64_t p_request_id, uint64_t p_stop_id) {
	List<ResourceUsage> usage;
	if (resource_usage_func) {
		resource_usage_func(&usage);
	}

	usage.sort();

	Array resources;
	for (List<ResourceUsage>::Element *E = usage.front(); E; E = E->next()) {
		Array descriptor;
		descriptor.push_back(E->get().path);
		descriptor.push_back(E->get().type);
		descriptor.push_back(E->get().format);
		descriptor.push_back(E->get().vram);
		resources.push_back(descriptor);
	}
	Array payload;
	payload.push_back(resources);
	transport.send_data("video_memory_result", payload, p_request_id, p_stop_id, DebugTransportV2::DATA_PRIORITY_INTERACTIVE);
}

Error ScriptDebuggerRemote::connect_to_host(const String &p_host, uint16_t p_port) {
	Error error = transport.connect_to_host(p_host, p_port);
	if (error == OK) {
		transport.trace_state(session.get_state(), DebugProtocolV2::STATE_RUNNING, "connected");
		session.connected(transport.get_generation(), OS::get_singleton()->get_ticks_msec());
	}
	return error;
}

void ScriptDebuggerRemote::_put_variable(const String &p_name, const Variant &p_variable) {
	packet_peer_stream->put_var(p_name);

	Variant var = p_variable;
	if (p_variable.get_type() == Variant::OBJECT && p_variable.operator Object *() == nullptr) {
		var = Variant();
	}

	int len = 0;
	Error err = encode_variant(var, nullptr, len, true);
	if (err != OK)
		ERR_PRINT("Failed to encode variant.");

	if (len > packet_peer_stream->get_output_buffer_max_size()) { //limit to max size
		packet_peer_stream->put_var(Variant());
	} else {
		packet_peer_stream->put_var(var);
	}
}

void ScriptDebuggerRemote::_save_node(ObjectID id, const String &p_path) {
	Node *node = Object::cast_to<Node>(ObjectDB::get_instance(id));
	ERR_FAIL_COND(!node);

	Ref<PackedScene> ps = memnew(PackedScene);
	ps->pack(node);
	ResourceSaver::save(p_path, ps);
}

void ScriptDebuggerRemote::debug(ScriptLanguage *p_script, bool p_can_continue, bool p_is_error_breakpoint) {
	//this function is called when there is a debugger break (bug on script)
	//or when execution is paused from editor
	// A stop reached through any path other than the one-shot return hook must
	// cancel Step Out so it cannot trigger again after the next resume.
	set_break_on_function_return(false);
	// Protocol v2 never enters a terminal pause for an error the VM cannot
	// resume from. Older frontends exposed a disabled Continue button here,
	// leaving the game alive but permanently frozen.
	if (p_is_error_breakpoint && !p_can_continue) {
		if (transport.is_data_connected()) {
			Array payload;
			payload.push_back(p_script->debug_get_error());
			Array frames;
			const int frame_count = p_script->debug_get_stack_level_count();
			for (int frame = 0; frame < frame_count; frame++) {
				Array descriptor;
				descriptor.push_back(p_script->debug_get_stack_level_source(frame));
				descriptor.push_back(p_script->debug_get_stack_level_line(frame));
				descriptor.push_back(p_script->debug_get_stack_level_function(frame));
				frames.push_back(descriptor);
			}
			payload.push_back(frames);
			transport.send_data("noncontinuable_error", payload, 0, session.get_current_stop_id(), DebugTransportV2::DATA_PRIORITY_INTERACTIVE);
		}
		return;
	}

	if (skip_breakpoints && !p_is_error_breakpoint) {
		return;
	}

	ERR_FAIL_COND_MSG(!transport.is_connected(), "Script Debugger v2 requires connected control and data channels.");

	if (allow_focus_steal_pid) {
		OS::get_singleton()->enable_for_stealing_focus(allow_focus_steal_pid);
	}

	const DebugProtocolV2::State previous_state = session.get_state();
	const uint64_t stop_id = session.enter_paused(p_can_continue, OS::get_singleton()->get_ticks_msec());
	transport.trace_state(previous_state, session.get_state(), "stopped");
	variable_store.start_stop(stop_id, transport.get_variable_page_size());
	Array stopped_payload;
	stopped_payload.push_back(p_can_continue);
	stopped_payload.push_back(p_script->debug_get_error());
	stopped_payload.push_back(p_is_error_breakpoint);
	if (transport.send_control("stopped", stopped_payload, 0, stop_id) != OK) {
		session.control_lost();
		set_depth(-1);
		set_lines_left(-1);
		set_break_on_function_return(false);
		return;
	}

	skip_profile_frame = true; // to avoid super long frame time for the frame

	Input::MouseMode mouse_mode = Input::get_singleton()->get_mouse_mode();
	if (mouse_mode != Input::MOUSE_MODE_VISIBLE) {
		Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_VISIBLE);
	}

	bool resumed = false;
	uint64_t last_draw_usec = OS::get_singleton()->get_ticks_usec();
	while (true) {
		transport.flush_data(4);
		const uint64_t now_msec = OS::get_singleton()->get_ticks_msec();
		if (!transport.is_control_connected() || session.heartbeat_expired(now_msec, transport.get_heartbeat_timeout_msec())) {
			if (transport.is_control_connected()) {
				Array diagnostic;
				diagnostic.push_back("control_timeout_auto_continue");
				transport.send_control("diagnostic", diagnostic, 0, stop_id);
			}
			WARN_PRINT("Debugger v2 control channel was lost or timed out while paused; automatically continuing the game.");
			transport.trace_state(session.get_state(), DebugProtocolV2::STATE_DISCONNECTED, "control_lost");
			session.control_lost();
			variable_store.invalidate();
			set_depth(-1);
			set_lines_left(-1);
			set_break_on_function_return(false);
			break;
		}

		DebugProtocolV2::Envelope message;
		Error poll_error = transport.poll_control(message);
		if (poll_error == OK) {
			DebugSessionV2::Action action = _handle_control_message(message);
			if (action == DebugSessionV2::ACTION_STEP) {
				set_break_on_function_return(false);
				set_depth(-1);
				set_lines_left(1);
				resumed = true;
			} else if (action == DebugSessionV2::ACTION_NEXT) {
				set_break_on_function_return(false);
				set_depth(0);
				set_lines_left(1);
				resumed = true;
			} else if (action == DebugSessionV2::ACTION_STEP_OUT) {
				set_depth(1);
				set_lines_left(1);
				set_break_on_function_return(true);
				resumed = true;
			} else if (action == DebugSessionV2::ACTION_CONTINUE) {
				set_break_on_function_return(false);
				set_depth(-1);
				set_lines_left(-1);
				OS::get_singleton()->move_window_to_foreground();
				resumed = true;
			}
			if (resumed) {
				variable_store.invalidate();
				transport.trace_state(session.get_state(), DebugProtocolV2::STATE_RUNNING, "resume_acknowledged");
				session.mark_running(now_msec);
				Array running_payload;
				running_payload.push_back(DebugProtocolV2::state_to_string(session.get_state()));
				transport.send_control("running", running_payload, message.request_id, stop_id);
				break;
			}
		} else if (poll_error != ERR_UNAVAILABLE) {
			continue;
		} else {
			if (!_poll_data_request(p_script)) {
				OS::get_singleton()->delay_usec(1000);
			}
		}

		OS::get_singleton()->process_and_drop_events();
		const uint64_t now_usec = OS::get_singleton()->get_ticks_usec();
		if (now_usec - last_draw_usec >= 16000) {
			const float elapsed_seconds = (now_usec - last_draw_usec) / 1000000.0f;
			last_draw_usec = now_usec;
			VisualServer::get_singleton()->sync();
			if (VisualServer::get_singleton()->has_changed()) {
				VisualServer::get_singleton()->draw(true, elapsed_seconds * Engine::get_singleton()->get_time_scale());
			}
		}
	}

	if (mouse_mode != Input::MOUSE_MODE_VISIBLE) {
		Input::get_singleton()->set_mouse_mode(mouse_mode);
	}
}

void ScriptDebuggerRemote::_send_control_ack(const DebugSessionV2::Decision &p_decision) {
	Array payload;
	payload.push_back(p_decision.accepted);
	payload.push_back(p_decision.duplicate);
	payload.push_back(p_decision.reason);
	payload.push_back(DebugProtocolV2::state_to_string(session.get_state()));
	transport.send_control("ack", payload, p_decision.request_id, p_decision.stop_id);
}

DebugSessionV2::Action ScriptDebuggerRemote::_handle_control_message(const DebugProtocolV2::Envelope &p_message) {
	DebugSessionV2::Decision decision = session.handle_control(p_message, OS::get_singleton()->get_ticks_msec());
	_send_control_ack(decision);
	return decision.action;
}

void ScriptDebuggerRemote::_send_data_error(const DebugProtocolV2::Envelope &p_request, const String &p_code, const String &p_message) {
	Array payload;
	payload.push_back(p_code);
	payload.push_back(p_message);
	transport.send_data("request_error", payload, p_request.request_id, session.get_current_stop_id());
}

bool ScriptDebuggerRemote::_poll_data_request(ScriptLanguage *p_script) {
	if (!transport.is_data_connected()) {
		return false;
	}
	DebugProtocolV2::Envelope request;
	Error error = transport.poll_data(request);
	if (error == ERR_UNAVAILABLE) {
		return false;
	}
	if (error != OK) {
		return true;
	}

	if (request.type == "queue_stats") {
		Array payload;
		payload.push_back(transport.get_queued_data_messages());
		payload.push_back(transport.get_queued_data_bytes());
		payload.push_back(transport.get_data_queue_max_bytes());
		payload.push_back((int64_t)transport.get_dropped_data_messages());
		payload.push_back((int64_t)transport.get_dropped_data_bytes());
		transport.send_data("queue_stats_result", payload, request.request_id, request.stop_id, DebugTransportV2::DATA_PRIORITY_ACTIVE);
		return true;
	}
	if (request.type == "stress_data_queue") {
		const int messages_to_queue = request.payload.size() > 0 ? CLAMP((int)request.payload[0], 1, 4096) : 1024;
		for (int index = 0; index < messages_to_queue; index++) {
			Array probe;
			probe.push_back(index);
			transport.send_data("stress_output", probe, 0, request.stop_id, DebugTransportV2::DATA_PRIORITY_OUTPUT);
		}
		Array payload;
		payload.push_back(messages_to_queue);
		transport.send_data("stress_data_queue_result", payload, request.request_id, request.stop_id, DebugTransportV2::DATA_PRIORITY_ACTIVE);
		return true;
	}
	if (request.type == "video_memory") {
		_send_video_memory(request.request_id, request.stop_id);
		return true;
	}
	if (request.type == "breakpoint" && request.payload.size() >= 3) {
		const String source = request.payload[0];
		const int line = request.payload[1];
		const bool enabled = request.payload[2];
		if (enabled) {
			insert_breakpoint(line, source);
		} else {
			remove_breakpoint(line, source);
		}
		transport.send_data("breakpoint_result", Array(), request.request_id, request.stop_id, DebugTransportV2::DATA_PRIORITY_ACTIVE);
		return true;
	}
	if (request.type == "set_skip_breakpoints" && request.payload.size() >= 1) {
		skip_breakpoints = request.payload[0];
		transport.send_data("set_skip_breakpoints_result", Array(), request.request_id, request.stop_id, DebugTransportV2::DATA_PRIORITY_ACTIVE);
		return true;
	}
	if (request.type == "reload_scripts") {
		reload_all_scripts = true;
		transport.send_data("reload_scripts_result", Array(), request.request_id, request.stop_id, DebugTransportV2::DATA_PRIORITY_ACTIVE);
		return true;
	}
	if (request.type == "set_object_property" && request.payload.size() >= 3) {
		_set_object_property((ObjectID)(int64_t)request.payload[0], request.payload[1], request.payload[2]);
		transport.send_data("set_object_property_result", Array(), request.request_id, request.stop_id, DebugTransportV2::DATA_PRIORITY_ACTIVE);
		return true;
	}
	if (request.type.begins_with("live_")) {
		Array command;
		command.push_back(request.type);
		for (int index = 0; index < request.payload.size(); index++) {
			command.push_back(request.payload[index]);
		}
		_parse_live_edit(command);
		transport.send_data("live_edit_result", Array(), request.request_id, request.stop_id, DebugTransportV2::DATA_PRIORITY_ACTIVE);
		return true;
	}
	if (request.type == "export_trace") {
		Array payload;
		payload.push_back(transport.export_trace());
		transport.send_data("export_trace_result", payload, request.request_id, request.stop_id, DebugTransportV2::DATA_PRIORITY_ACTIVE);
		return true;
	}
	if (request.type == "scene_snapshot") {
#ifdef DEBUG_ENABLED
		if (scene_tree) {
			pending_scene_request_id = request.request_id;
			pending_scene_stop_id = request.stop_id;
			scene_tree->_debugger_request_tree();
			pending_scene_request_id = 0;
			pending_scene_stop_id = 0;
		} else {
			_send_data_error(request, "scene_unavailable", "The running project has no SceneTree.");
		}
#else
		_send_data_error(request, "scene_unavailable", "Scene snapshots require a debug build.");
#endif
		return true;
	}
	if (request.type == "inspect_object_id" && request.payload.size() >= 1) {
		if (!_send_object_id((ObjectID)(int64_t)request.payload[0], request.request_id, request.stop_id)) {
			_send_data_error(request, "object_unavailable", "The requested remote object no longer exists.");
		}
		return true;
	}
	if (request.stop_id != session.get_current_stop_id() || variable_store.get_active_stop_id() != session.get_current_stop_id()) {
		_send_data_error(request, "stale_stop_id", "The requested debugger stop is no longer active.");
		return true;
	}
	if (!p_script) {
		_send_data_error(request, "not_paused", "Stack and variable requests require an active debugger stop.");
		return true;
	}

	if (request.type == "stack_trace") {
		const int total = p_script->debug_get_stack_level_count();
		const int start = request.payload.size() > 0 ? CLAMP((int)request.payload[0], 0, total) : 0;
		const int requested_count = request.payload.size() > 1 ? (int)request.payload[1] : transport.get_variable_page_size();
		const int count = CLAMP(requested_count <= 0 ? transport.get_variable_page_size() : requested_count, 1, (int)transport.get_variable_page_size());
		const int end = MIN(start + count, total);
		Array frames;
		for (int frame = start; frame < end; frame++) {
			Array descriptor;
			descriptor.push_back(frame);
			descriptor.push_back(p_script->debug_get_stack_level_source(frame));
			descriptor.push_back(p_script->debug_get_stack_level_line(frame));
			descriptor.push_back(p_script->debug_get_stack_level_function(frame));
			frames.push_back(descriptor);
		}
		Array payload;
		payload.push_back(total);
		payload.push_back(frames);
		payload.push_back(end < total);
		transport.send_data("stack_trace_result", payload, request.request_id, request.stop_id);
		return true;
	}

	if (request.type == "scopes") {
		if (request.payload.size() < 1) {
			_send_data_error(request, "invalid_request", "A stack frame is required.");
			return true;
		}
		const int frame = request.payload[0];
		if (frame < 0 || frame >= p_script->debug_get_stack_level_count()) {
			_send_data_error(request, "invalid_frame", "The requested stack frame does not exist.");
			return true;
		}
		Array scopes;
		const char *scope_names[3] = { "Locals", "Members", "Globals" };
		for (int scope = 0; scope < 3; scope++) {
			Array descriptor;
			descriptor.push_back(scope_names[scope]);
			descriptor.push_back((int64_t)variable_store.create_scope_handle(frame, (DebugVariableStoreV2::ScopeKind)scope));
			descriptor.push_back(scope == DebugVariableStoreV2::SCOPE_GLOBALS);
			scopes.push_back(descriptor);
		}
		Array payload;
		payload.push_back(scopes);
		transport.send_data("scopes_result", payload, request.request_id, request.stop_id);
		return true;
	}

	if (request.type == "variables" || request.type == "inspect_object") {
		if (request.payload.size() < 1) {
			_send_data_error(request, "invalid_request", "A variable handle is required.");
			return true;
		}
		const uint64_t handle = (uint64_t)(int64_t)request.payload[0];
		const int start = request.payload.size() > 1 ? (int)request.payload[1] : 0;
		const int count = request.payload.size() > 2 ? (int)request.payload[2] : variable_store.get_page_size();
		Array items;
		int total = 0;
		bool has_more = false;

		int frame = 0;
		DebugVariableStoreV2::ScopeKind scope;
		if (variable_store.get_scope(handle, frame, scope)) {
			List<String> names;
			List<Variant> values;
			if (scope == DebugVariableStoreV2::SCOPE_LOCALS) {
				p_script->debug_get_stack_level_locals(frame, &names, &values, 0, 0);
			} else if (scope == DebugVariableStoreV2::SCOPE_MEMBERS) {
				if (ScriptInstance *instance = p_script->debug_get_stack_level_instance(frame)) {
					names.push_back("self");
					values.push_back(instance->get_owner());
				}
				p_script->debug_get_stack_level_members(frame, &names, &values, 0, 0);
			} else {
				p_script->debug_get_globals(&names, &values, 0, 0);
			}
			error = variable_store.describe_named(names, values, start, count, items, total, has_more);
		} else {
			error = variable_store.describe_handle(handle, start, count, items, total, has_more);
		}
		if (error != OK) {
			_send_data_error(request, "stale_handle", "The variable handle is stale or invalid.");
			return true;
		}

		Array payload;
		payload.push_back(total);
		payload.push_back(MAX(0, start));
		payload.push_back(items);
		payload.push_back(has_more);
		transport.send_data(request.type == "inspect_object" ? "inspect_object_result" : "variables_result", payload, request.request_id, request.stop_id);
		return true;
	}

	_send_data_error(request, "unknown_data_request", "Unknown debugger data request '" + request.type + "'.");
	return true;
}

void ScriptDebuggerRemote::_get_output() {
	mutex.lock();
	locking = true;
	while (output_strings.size()) {
		const OutputString output_string = output_strings.front()->get();
		output_strings.pop_front();
		Array payload;
		payload.push_back(output_string.message);
		payload.push_back(output_string.type);
		transport.send_data("output", payload, 0, session.get_current_stop_id(), DebugTransportV2::DATA_PRIORITY_OUTPUT);
	}

	if (n_messages_dropped > 0) {
		Message msg;
		msg.message = "Too many messages! " + String::num_int64(n_messages_dropped) + " messages were dropped.";
		messages.push_back(msg);
		n_messages_dropped = 0;
	}

	while (messages.size()) {
		const Message message = messages.front()->get();
		messages.pop_front();
		transport.send_data("message:" + message.message, message.data, 0, session.get_current_stop_id(), DebugTransportV2::DATA_PRIORITY_BACKGROUND);
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
		const OutputError oe = errors.front()->get();
		errors.pop_front();
		Array error_data;
		error_data.push_back(oe.hr);
		error_data.push_back(oe.min);
		error_data.push_back(oe.sec);
		error_data.push_back(oe.msec);
		error_data.push_back(oe.source_func);
		error_data.push_back(oe.source_file);
		error_data.push_back(oe.source_line);
		error_data.push_back(oe.error);
		error_data.push_back(oe.error_descr);
		error_data.push_back(oe.warning);
		Array payload;
		payload.push_back(error_data);
		payload.push_back(oe.callstack);
		transport.send_data("error", payload, 0, session.get_current_stop_id(), DebugTransportV2::DATA_PRIORITY_BACKGROUND);
	}

	const uint64_t transport_dropped = transport.get_dropped_data_messages();
	if (transport_dropped != last_transport_dropped) {
		Array payload;
		payload.push_back((int64_t)transport_dropped);
		payload.push_back((int64_t)transport.get_dropped_data_bytes());
		payload.push_back(transport.get_queued_data_bytes());
		transport.send_data("data_drop_summary", payload, 0, session.get_current_stop_id(), DebugTransportV2::DATA_PRIORITY_INTERACTIVE);
		last_transport_dropped = transport_dropped;
	}
	locking = false;
	mutex.unlock();
}

void ScriptDebuggerRemote::line_poll() {
	//the purpose of this is just processing events every now and then when the script might get too busy
	//otherwise bugs like infinite loops can't be caught
	if (poll_every % 2048 == 0) {
		_poll_events();
	}
	poll_every++;
}

void ScriptDebuggerRemote::_err_handler(void *ud, const char *p_func, const char *p_file, int p_line, const char *p_err, const char *p_descr, ErrorHandlerType p_type) {
	if (p_type == ERR_HANDLER_SCRIPT) {
		return; //ignore script errors, those go through debugger
	}

	Vector<ScriptLanguage::StackInfo> si;

	for (int i = 0; i < ScriptServer::get_language_count(); i++) {
		si = ScriptServer::get_language(i)->debug_get_current_stack_info();
		if (si.size()) {
			break;
		}
	}

	ScriptDebuggerRemote *sdr = (ScriptDebuggerRemote *)ud;
	sdr->send_error(String::utf8(p_func), String::utf8(p_file), p_line, String::utf8(p_err), String::utf8(p_descr), p_type, si);
}

bool ScriptDebuggerRemote::_parse_live_edit(const Array &p_command) {
#ifdef DEBUG_ENABLED

	String cmdstr = p_command[0];
	if (!scene_tree || !cmdstr.begins_with("live_")) {
		return false;
	}

	if (cmdstr == "live_set_root") {
		scene_tree->_live_edit_root_func(p_command[1], p_command[2]);

	} else if (cmdstr == "live_node_path") {
		scene_tree->_live_edit_node_path_func(p_command[1], p_command[2]);

	} else if (cmdstr == "live_res_path") {
		scene_tree->_live_edit_res_path_func(p_command[1], p_command[2]);

	} else if (cmdstr == "live_node_prop_res") {
		scene_tree->_live_edit_node_set_res_func(p_command[1], p_command[2], p_command[3]);

	} else if (cmdstr == "live_node_prop") {
		scene_tree->_live_edit_node_set_func(p_command[1], p_command[2], p_command[3]);

	} else if (cmdstr == "live_res_prop_res") {
		scene_tree->_live_edit_res_set_res_func(p_command[1], p_command[2], p_command[3]);

	} else if (cmdstr == "live_res_prop") {
		scene_tree->_live_edit_res_set_func(p_command[1], p_command[2], p_command[3]);

	} else if (cmdstr == "live_node_call") {
		scene_tree->_live_edit_node_call_func(p_command[1], p_command[2], p_command[3], p_command[4], p_command[5], p_command[6], p_command[7], p_command[8], p_command[9], p_command[10]);

	} else if (cmdstr == "live_res_call") {
		scene_tree->_live_edit_res_call_func(p_command[1], p_command[2], p_command[3], p_command[4], p_command[5], p_command[6], p_command[7], p_command[8], p_command[9], p_command[10]);

	} else if (cmdstr == "live_create_node") {
		scene_tree->_live_edit_create_node_func(p_command[1], p_command[2], p_command[3]);

	} else if (cmdstr == "live_instance_node") {
		scene_tree->_live_edit_instance_node_func(p_command[1], p_command[2], p_command[3]);

	} else if (cmdstr == "live_remove_node") {
		scene_tree->_live_edit_remove_node_func(p_command[1]);

	} else if (cmdstr == "live_remove_and_keep_node") {
		scene_tree->_live_edit_remove_and_keep_node_func(p_command[1], p_command[2]);

	} else if (cmdstr == "live_restore_node") {
		scene_tree->_live_edit_restore_node_func(p_command[1], p_command[2], p_command[3]);

	} else if (cmdstr == "live_duplicate_node") {
		scene_tree->_live_edit_duplicate_node_func(p_command[1], p_command[2]);

	} else if (cmdstr == "live_reparent_node") {
		scene_tree->_live_edit_reparent_node_func(p_command[1], p_command[2], p_command[3], p_command[4]);

	} else {
		return false;
	}

	return true;
#else

	return false;
#endif
}

bool ScriptDebuggerRemote::_send_object_id(ObjectID p_id, uint64_t p_request_id, uint64_t p_stop_id) {
	Object *obj = ObjectDB::get_instance(p_id);
	if (!obj) {
		return false;
	}

	typedef Pair<PropertyInfo, Variant> PropertyDesc;
	List<PropertyDesc> properties;

	if (ScriptInstance *si = obj->get_script_instance()) {
		if (!si->get_script().is_null()) {
			typedef Map<const Script *, Set<StringName>> ScriptMemberMap;
			typedef Map<const Script *, Map<StringName, Variant>> ScriptConstantsMap;

			ScriptMemberMap members;
			members[si->get_script().ptr()] = Set<StringName>();
			si->get_script()->get_members(&(members[si->get_script().ptr()]));

			ScriptConstantsMap constants;
			constants[si->get_script().ptr()] = Map<StringName, Variant>();
			si->get_script()->get_constants(&(constants[si->get_script().ptr()]));

			Ref<Script> base = si->get_script()->get_base_script();
			while (base.is_valid()) {
				members[base.ptr()] = Set<StringName>();
				base->get_members(&(members[base.ptr()]));

				constants[base.ptr()] = Map<StringName, Variant>();
				base->get_constants(&(constants[base.ptr()]));

				base = base->get_base_script();
			}

			for (ScriptMemberMap::Element *sm = members.front(); sm; sm = sm->next()) {
				for (Set<StringName>::Element *E = sm->get().front(); E; E = E->next()) {
					Variant m;
					if (si->get(E->get(), m)) {
						String script_path = sm->key() == si->get_script().ptr() ? "" : sm->key()->get_path().get_file() + "/";
						PropertyInfo pi(m.get_type(), "Members/" + script_path + E->get());
						properties.push_back(PropertyDesc(pi, m));
					}
				}
			}

			for (ScriptConstantsMap::Element *sc = constants.front(); sc; sc = sc->next()) {
				for (Map<StringName, Variant>::Element *E = sc->get().front(); E; E = E->next()) {
					String script_path = sc->key() == si->get_script().ptr() ? "" : sc->key()->get_path().get_file() + "/";
					if (E->value().get_type() == Variant::OBJECT) {
						Variant id = ((Object *)E->value())->get_instance_id();
						PropertyInfo pi(id.get_type(), "Constants/" + E->key(), PROPERTY_HINT_OBJECT_ID, "Object");
						properties.push_back(PropertyDesc(pi, id));
					} else {
						PropertyInfo pi(E->value().get_type(), "Constants/" + script_path + E->key());
						properties.push_back(PropertyDesc(pi, E->value()));
					}
				}
			}
		}
	}

	if (Node *node = Object::cast_to<Node>(obj)) {
		// in some cases node will not be in tree here
		// for instance where it created as variable and not yet added to tree
		// in such cases we can't ask for it's path
		if (node->is_inside_tree()) {
			PropertyInfo pi(Variant::NODE_PATH, String("Node/path"));
			properties.push_front(PropertyDesc(pi, node->get_path()));
		} else {
			PropertyInfo pi(Variant::STRING, String("Node/path"));
			properties.push_front(PropertyDesc(pi, "[Orphan]"));
		}

	} else if (Resource *res = Object::cast_to<Resource>(obj)) {
		if (Script *s = Object::cast_to<Script>(res)) {
			Map<StringName, Variant> constants;
			s->get_constants(&constants);
			for (Map<StringName, Variant>::Element *E = constants.front(); E; E = E->next()) {
				if (E->value().get_type() == Variant::OBJECT) {
					Variant id = ((Object *)E->value())->get_instance_id();
					PropertyInfo pi(id.get_type(), "Constants/" + E->key(), PROPERTY_HINT_OBJECT_ID, "Object");
					properties.push_front(PropertyDesc(pi, E->value()));
				} else {
					PropertyInfo pi(E->value().get_type(), String("Constants/") + E->key());
					properties.push_front(PropertyDesc(pi, E->value()));
				}
			}
		}
	}

	List<PropertyInfo> pinfo;
	obj->get_property_list(&pinfo, true);
	for (List<PropertyInfo>::Element *E = pinfo.front(); E; E = E->next()) {
		if (E->get().usage & (PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_CATEGORY)) {
			properties.push_back(PropertyDesc(E->get(), obj->get(E->get().name)));
		}
	}

	Array send_props;
	for (int i = 0; i < properties.size(); i++) {
		const PropertyInfo &pi = properties[i].first;
		Variant &var = properties[i].second;

		RES res = var;

		Array prop;
		prop.push_back(pi.name);
		prop.push_back(pi.type);

		//only send information that can be sent..
		int len = 0; //test how big is this to encode
		encode_variant(var, nullptr, len);
		if (len > packet_peer_stream->get_output_buffer_max_size()) { //limit to max size
			prop.push_back(PROPERTY_HINT_OBJECT_TOO_BIG);
			prop.push_back("");
			prop.push_back(pi.usage);
			prop.push_back(Variant());
		} else {
			prop.push_back(pi.hint);
			prop.push_back(pi.hint_string);
			prop.push_back(pi.usage);

			if (!res.is_null() && !res->get_path().empty()) {
				var = res->get_path();
			}

			prop.push_back(var);
		}
		send_props.push_back(prop);
	}

	Array payload;
	payload.push_back((int64_t)p_id);
	payload.push_back(obj->get_class());
	payload.push_back(send_props);
	transport.send_data("inspect_object_id_result", payload, p_request_id, p_stop_id, DebugTransportV2::DATA_PRIORITY_INTERACTIVE);
	return true;
}

void ScriptDebuggerRemote::_set_object_property(ObjectID p_id, const String &p_property, const Variant &p_value) {
	Object *obj = ObjectDB::get_instance(p_id);
	if (!obj) {
		return;
	}

	String prop_name = p_property;
	if (p_property.begins_with("Members/")) {
		Vector<String> ss = p_property.split("/");
		prop_name = ss[ss.size() - 1];
	}

	obj->set(prop_name, p_value);
}

void ScriptDebuggerRemote::_poll_events() {
	//this si called from ::idle_poll, happens only when running the game,
	//does not get called while on debug break
	while (transport.is_control_connected()) {
		DebugProtocolV2::Envelope message;
		Error error = transport.poll_control(message);
		if (error == ERR_UNAVAILABLE) {
			break;
		}
		if (error != OK) {
			continue;
		}

		DebugSessionV2::Action action = _handle_control_message(message);
		if (action == DebugSessionV2::ACTION_PAUSE && get_break_language()) {
			debug(get_break_language());
			return;
		}
	}

	if (!transport.is_data_connected()) {
		return;
	}

	for (int request = 0; request < 8; request++) {
		if (!_poll_data_request(nullptr)) {
			break;
		}
	}
}

void ScriptDebuggerRemote::_send_profiling_data(bool p_for_frame) {
	int ofs = 0;

	for (int i = 0; i < ScriptServer::get_language_count(); i++) {
		if (p_for_frame) {
			ofs += ScriptServer::get_language(i)->profiling_get_frame_data(&profile_info.write[ofs], profile_info.size() - ofs);
		} else {
			ofs += ScriptServer::get_language(i)->profiling_get_accumulated_data(&profile_info.write[ofs], profile_info.size() - ofs);
		}
	}

	for (int i = 0; i < ofs; i++) {
		profile_info_ptrs.write[i] = &profile_info.write[i];
	}

	SortArray<ScriptLanguage::ProfilingInfo *, ProfileInfoSort> sa;
	sa.sort(profile_info_ptrs.ptrw(), ofs);

	int to_send = MIN(ofs, max_frame_functions);

	//check signatures first
	uint64_t total_script_time = 0;
	Array signatures;

	for (int i = 0; i < to_send; i++) {
		if (!profiler_function_signature_map.has(profile_info_ptrs[i]->signature)) {
			int idx = profiler_function_signature_map.size();
			profiler_function_signature_map[profile_info_ptrs[i]->signature] = idx;
			Array signature;
			signature.push_back(profile_info_ptrs[i]->signature);
			signature.push_back(idx);
			signatures.push_back(signature);
		}

		total_script_time += profile_info_ptrs[i]->self_time;
	}

	Array custom_frames;
	if (p_for_frame) {
		for (int i = 0; i < profile_frame_data.size(); i++) {
			Array custom;
			custom.push_back(profile_frame_data[i].name);
			custom.push_back(profile_frame_data[i].data);
			custom_frames.push_back(custom);
		}
	}

	Array functions;
	for (int i = 0; i < to_send; i++) {
		int sig_id = -1;

		if (profiler_function_signature_map.has(profile_info_ptrs[i]->signature)) {
			sig_id = profiler_function_signature_map[profile_info_ptrs[i]->signature];
		}

		Array function;
		function.push_back(sig_id);
		function.push_back(profile_info_ptrs[i]->call_count);
		function.push_back(profile_info_ptrs[i]->total_time / 1000000.0);
		function.push_back(profile_info_ptrs[i]->self_time / 1000000.0);
		functions.push_back(function);
	}

	Array payload;
	payload.push_back(signatures);
	payload.push_back(Engine::get_singleton()->get_idle_frames());
	payload.push_back(frame_time);
	payload.push_back(process_time);
	payload.push_back(physics_time);
	payload.push_back(physics_frame_time);
	payload.push_back(USEC_TO_SEC(total_script_time));
	payload.push_back(custom_frames);
	payload.push_back(functions);
	transport.send_data(p_for_frame ? "profile_frame" : "profile_total", payload, 0, session.get_current_stop_id(), DebugTransportV2::DATA_PRIORITY_BACKGROUND);

	if (p_for_frame) {
		profile_frame_data.clear();
	}
}

void ScriptDebuggerRemote::idle_poll() {
	// this function is called every frame, except when there is a debugger break (::debug() in this class)
	// execution stops and remains in the ::debug function
	if (!transport.is_connected()) {
		const uint64_t now_msec = OS::get_singleton()->get_ticks_msec();
		if (transport.is_reconnecting()) {
			Error reconnect_error = transport.poll_reconnect();
			if (reconnect_error == OK) {
				transport.trace_state(session.get_state(), DebugProtocolV2::STATE_RUNNING, "reconnected");
				session.connected(transport.get_generation(), now_msec);
			} else if (reconnect_error != ERR_BUSY) {
				next_reconnect_msec = now_msec + 250;
			}
		} else {
			_poll_events();
			if (!transport.is_connected() && now_msec >= next_reconnect_msec) {
				session.control_lost();
				Error reconnect_error = transport.start_reconnect();
				if (reconnect_error != ERR_BUSY) {
					next_reconnect_msec = now_msec + 250;
				}
			}
		}
		return;
	}

	_get_output();
	transport.flush_data(4);

	if (requested_quit) {
		transport.send_data("kill_me", Array(), 0, session.get_current_stop_id(), DebugTransportV2::DATA_PRIORITY_INTERACTIVE);
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
			Array payload;
			payload.push_back(arr);
			transport.send_data("performance", payload, 0, session.get_current_stop_id(), DebugTransportV2::DATA_PRIORITY_BACKGROUND);
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

	if (profiling_network) {
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
	transport.flush_data(4);
}

void ScriptDebuggerRemote::_send_network_profiling_data() {
	ERR_FAIL_COND(multiplayer.is_null());

	int n_nodes = multiplayer->get_profiling_frame(&network_profile_info.write[0]);

	Array nodes;
	for (int i = 0; i < n_nodes; ++i) {
		Array node;
		node.push_back(network_profile_info[i].node);
		node.push_back(network_profile_info[i].node_path);
		node.push_back(network_profile_info[i].incoming_rpc);
		node.push_back(network_profile_info[i].incoming_rset);
		node.push_back(network_profile_info[i].outgoing_rpc);
		node.push_back(network_profile_info[i].outgoing_rset);
		nodes.push_back(node);
	}
	Array payload;
	payload.push_back(nodes);
	transport.send_data("network_profile", payload, 0, session.get_current_stop_id(), DebugTransportV2::DATA_PRIORITY_BACKGROUND);
}

void ScriptDebuggerRemote::_send_network_bandwidth_usage() {
	ERR_FAIL_COND(multiplayer.is_null());

	int incoming_bandwidth = multiplayer->get_incoming_bandwidth_usage();
	int outgoing_bandwidth = multiplayer->get_outgoing_bandwidth_usage();

	Array payload;
	payload.push_back(incoming_bandwidth);
	payload.push_back(outgoing_bandwidth);
	transport.send_data("network_bandwidth", payload, 0, session.get_current_stop_id(), DebugTransportV2::DATA_PRIORITY_BACKGROUND);
}

void ScriptDebuggerRemote::send_message(const String &p_message, const Array &p_args) {
	if (p_message == "scene_tree") {
		if (pending_scene_request_id != 0) {
			Array payload;
			payload.push_back((int64_t)++scene_revision);
			payload.push_back(p_args);
			transport.send_data("scene_snapshot_result", payload, pending_scene_request_id, pending_scene_stop_id, DebugTransportV2::DATA_PRIORITY_INTERACTIVE);
		}
		return;
	}
	mutex.lock();
	if (!locking && transport.is_data_connected()) {
		if (messages.size() >= max_messages_per_frame) {
			n_messages_dropped++;
		} else {
			Message msg;
			msg.message = p_message;
			msg.data = p_args;
			messages.push_back(msg);
		}
	}
	mutex.unlock();
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

	oe.callstack = cstack;
	if (oe.warning) {
		warn_count++;
	} else {
		err_count++;
	}

	mutex.lock();

	if (!locking && transport.is_data_connected()) {
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

	mutex.unlock();
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

	if (allowed_chars == 0 && s.length() > 0) {
		return;
	}

	if (allowed_chars < s.length()) {
		s = s.substr(0, allowed_chars);
	}

	sdr->char_count += allowed_chars;
	bool overflowed = sdr->char_count >= sdr->max_cps;

	sdr->mutex.lock();
	if (!sdr->locking && sdr->transport.is_data_connected()) {
		if (overflowed) {
			s += "[...]";
		}

		OutputString output_string;
		output_string.message = s;
		output_string.type = p_error ? MESSAGE_TYPE_ERROR : MESSAGE_TYPE_LOG;
		sdr->output_strings.push_back(output_string);

		if (overflowed) {
			output_string.message = "[output overflow, print less text!]";
			output_string.type = MESSAGE_TYPE_ERROR;
			sdr->output_strings.push_back(output_string);
		}
	}
	sdr->mutex.unlock();
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

void ScriptDebuggerRemote::profiling_set_frame_times(float p_frame_time, float p_process_time, float p_physics_time, float p_physics_frame_time) {
	frame_time = p_frame_time;
	process_time = p_process_time;
	physics_time = p_physics_time;
	physics_frame_time = p_physics_frame_time;
}

void ScriptDebuggerRemote::set_skip_breakpoints(bool p_skip_breakpoints) {
	skip_breakpoints = p_skip_breakpoints;
}

void ScriptDebuggerRemote::set_allow_focus_steal_pid(OS::ProcessID p_pid) {
	allow_focus_steal_pid = p_pid;
}

ScriptDebuggerRemote::ResourceUsageFunc ScriptDebuggerRemote::resource_usage_func = nullptr;

ScriptDebuggerRemote::ScriptDebuggerRemote() :
		profiling(false),
		profiling_network(false),
		max_frame_functions(16),
		skip_profile_frame(false),
		reload_all_scripts(false),
		packet_peer_stream(transport.get_legacy_data_peer()),
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
		n_warnings_dropped(0),
		last_transport_dropped(0),
		max_cps(GLOBAL_GET("network/limits/debugger_stdout/max_chars_per_second")),
		char_count(0),
		err_count(0),
		warn_count(0),
		last_msec(0),
		msec_count(0),
		allow_focus_steal_pid(0),
		locking(false),
		poll_every(0),
		next_reconnect_msec(0),
		scene_revision(0),
		pending_scene_request_id(0),
		pending_scene_stop_id(0),
		scene_tree(nullptr) {
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
