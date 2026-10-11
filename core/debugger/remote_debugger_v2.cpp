/**************************************************************************/
/*  remote_debugger_v2.cpp                                                */
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
/* "Software"), to deal in the Software without restriction, including   */
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

#include "remote_debugger_v2.h"

#include "core/debugger/debugger_marshalls.h"
#include "core/debugger/script_debugger.h"
#include "core/input/input.h"
#include "core/object/script_language.h"
#include "core/os/os.h"
#include "servers/display/display_server.h"

RemoteDebuggerV2::RemoteDebuggerV2() {
	print_handler.printfunc = _print_handler;
	print_handler.userdata = this;
	add_print_handler(&print_handler);
	error_handler.errfunc = _error_handler;
	error_handler.userdata = this;
	add_error_handler(&error_handler);
}

RemoteDebuggerV2::~RemoteDebuggerV2() {
	stopping = true;
	remove_print_handler(&print_handler);
	remove_error_handler(&error_handler);
	transport.disconnect();
}

Error RemoteDebuggerV2::connect(const String &p_host, uint16_t p_port) {
	Error err = transport.connect_to_host(p_host, p_port);
	if (err == OK) {
		session.connected(transport.get_generation(), OS::get_singleton()->get_ticks_msec());
	}
	return err;
}

void RemoteDebuggerV2::_print_handler(void *p_self, const String &p_text, bool p_error, bool p_rich) {
	RemoteDebuggerV2 *self = static_cast<RemoteDebuggerV2 *>(p_self);
	if (self->stopping) {
		return;
	}
	const int output_type = p_error ? 1 : (p_rich ? 2 : 0);
	self->send_message("output", { p_text, output_type });
}

void RemoteDebuggerV2::_error_handler(void *p_self, const char *p_function, const char *p_file, int p_line, const char *p_error, const char *p_description, bool p_editor_notify, ErrorHandlerType p_type) {
	RemoteDebuggerV2 *self = static_cast<RemoteDebuggerV2 *>(p_self);
	if (!self->stopping && script_debugger) {
		self->send_error(String::utf8(p_function), String::utf8(p_file), p_line, String::utf8(p_error), String::utf8(p_description), p_editor_notify, p_type);
	}
}

void RemoteDebuggerV2::send_message(const String &p_message, const Array &p_data) {
	MutexLock lock(queue_mutex);
	if (queued_messages.size() >= 2048) {
		queued_messages.pop_front();
	}
	queued_messages.push_back({ p_message, p_data });
}

void RemoteDebuggerV2::send_error(const String &p_function, const String &p_file, int p_line, const String &p_error, const String &p_description, bool p_editor_notify, ErrorHandlerType p_type) {
	DebuggerMarshalls::OutputError error;
	error.error = p_error;
	error.error_descr = p_description;
	error.source_file = p_file;
	error.source_line = p_line;
	error.source_func = p_function;
	error.warning = p_type == ERR_HANDLER_WARNING;
	const uint64_t now = OS::get_singleton()->get_ticks_msec();
	error.hr = now / 3600000;
	error.min = (now / 60000) % 60;
	error.sec = (now / 1000) % 60;
	error.msec = now % 1000;
	if (script_debugger) {
		error.callstack.append_array(script_debugger->get_error_stack_info());
	}
	send_message("error", error.serialize());
}

void RemoteDebuggerV2::_flush_messages() {
	List<QueuedMessage> pending;
	{
		MutexLock lock(queue_mutex);
		while (!queued_messages.is_empty()) {
			pending.push_back(queued_messages.front()->get());
			queued_messages.pop_front();
		}
	}
	if (!transport.is_data_connected()) {
		return;
	}
	for (const QueuedMessage &message : pending) {
		if (message.type == "scene:scene_tree" && pending_scene_request_id != 0) {
			transport.send_data("scene_snapshot_result", { ++scene_revision, message.payload }, pending_scene_request_id, pending_scene_stop_id, DebugTransportV2::DATA_PRIORITY_INTERACTIVE);
			pending_scene_request_id = 0;
		} else if (message.type == "scene:inspect_object" && pending_inspect_request_id != 0) {
			transport.send_data("inspect_object_id_result", message.payload, pending_inspect_request_id, pending_inspect_stop_id, DebugTransportV2::DATA_PRIORITY_INTERACTIVE);
			pending_inspect_request_id = 0;
		} else if (message.type == "output" || message.type == "error") {
			Array payload;
			payload.push_back(message.payload);
			transport.send_data(message.type, payload, 0, session.get_current_stop_id(), DebugTransportV2::DATA_PRIORITY_OUTPUT);
		} else {
			transport.send_data("message:" + message.type, message.payload, 0, session.get_current_stop_id(), DebugTransportV2::DATA_PRIORITY_BACKGROUND);
		}
	}
}

void RemoteDebuggerV2::_request_error(const DebugProtocolV2::Envelope &p_request, const String &p_code, const String &p_description) {
	transport.send_data("request_error", { p_code, p_description }, p_request.request_id, session.get_current_stop_id());
}

DebugSessionV2::Action RemoteDebuggerV2::_control(const DebugProtocolV2::Envelope &p_message) {
	const DebugSessionV2::Decision decision = session.handle_control(p_message, OS::get_singleton()->get_ticks_msec());
	transport.send_control("ack", { decision.accepted, decision.duplicate, decision.reason, DebugProtocolV2::state_to_string(session.get_state()) }, decision.request_id, decision.stop_id);
	return decision.action;
}

void RemoteDebuggerV2::_resume_without_frontend() {
	variables.invalidate();
	session.control_lost();
	script_debugger->set_depth(-1);
	script_debugger->set_lines_left(-1);
}

bool RemoteDebuggerV2::_poll_data(ScriptLanguage *p_language) {
	if (!transport.is_data_connected()) {
		return false;
	}
	DebugProtocolV2::Envelope request;
	const Error poll_error = transport.poll_data(request);
	if (poll_error == ERR_UNAVAILABLE) {
		return false;
	}
	if (poll_error != OK) {
		return true;
	}
	if (request.type == "breakpoint" && request.payload.size() >= 3) {
		const String source = request.payload[0];
		const int line = request.payload[1];
		if ((bool)request.payload[2]) {
			script_debugger->insert_breakpoint(line, source);
		} else {
			script_debugger->remove_breakpoint(line, source);
		}
		transport.send_data("breakpoint_result", Array(), request.request_id, request.stop_id);
		return true;
	}
	if (request.type == "queue_stats") {
		transport.send_data("queue_stats_result", { transport.get_queued_data_messages(), transport.get_queued_data_bytes(), transport.get_data_queue_max_bytes(), (int64_t)transport.get_dropped_data_messages(), (int64_t)transport.get_dropped_data_bytes() }, request.request_id, request.stop_id);
		return true;
	}
	if (request.type == "export_trace") {
		Array payload;
		payload.push_back(transport.export_trace());
		transport.send_data("export_trace_result", payload, request.request_id, request.stop_id);
		return true;
	}
	if (request.type == "scene_snapshot") {
		pending_scene_request_id = request.request_id;
		pending_scene_stop_id = request.stop_id;
		bool captured = false;
		const Error err = capture_parse("scene", "request_scene_tree", Array(), captured);
		if (err != OK || !captured) {
			pending_scene_request_id = 0;
			_request_error(request, "scene_unavailable", "Scene debugger is unavailable.");
		}
		return true;
	}
	if (request.type == "inspect_object_id" && request.payload.size() >= 1) {
		pending_inspect_request_id = request.request_id;
		pending_inspect_stop_id = request.stop_id;
		bool captured = false;
		const Error err = capture_parse("scene", "inspect_object", { request.payload[0] }, captured);
		if (err != OK || !captured) {
			pending_inspect_request_id = 0;
			_request_error(request, "object_unavailable", "The remote object is unavailable.");
		}
		return true;
	}
	if (request.type == "set_object_property" && request.payload.size() >= 3) {
		bool captured = false;
		const Error err = capture_parse("scene", "set_object_property", request.payload, captured);
		if (err != OK || !captured) {
			_request_error(request, "object_unavailable", "The remote object property could not be set.");
		} else {
			transport.send_data("set_object_property_result", Array(), request.request_id, request.stop_id);
		}
		return true;
	}
	if (request.stop_id != session.get_current_stop_id() || variables.get_active_stop_id() != session.get_current_stop_id()) {
		_request_error(request, "stale_stop_id", "The debugger stop is no longer active.");
		return true;
	}
	if (!p_language) {
		_request_error(request, "not_paused", "Stack and variable requests require a paused script.");
		return true;
	}
	if (request.type == "stack_trace") {
		const int total = p_language->debug_get_stack_level_count();
		const int start = request.payload.size() > 0 ? CLAMP((int)request.payload[0], 0, total) : 0;
		const int count = request.payload.size() > 1 ? CLAMP((int)request.payload[1], 1, 100) : 100;
		Array frames;
		for (int index = start; index < MIN(start + count, total); index++) {
			frames.push_back(Array{ index, p_language->debug_get_stack_level_source(index), p_language->debug_get_stack_level_line(index), p_language->debug_get_stack_level_function(index) });
		}
		transport.send_data("stack_trace_result", { total, frames, start + count < total }, request.request_id, request.stop_id);
		return true;
	}
	if (request.type == "scopes" && request.payload.size() >= 1) {
		const int frame = request.payload[0];
		if (frame < 0 || frame >= p_language->debug_get_stack_level_count()) {
			_request_error(request, "invalid_frame", "The requested frame does not exist.");
			return true;
		}
		Array scopes;
		const char *names[3] = { "Locals", "Members", "Globals" };
		for (int index = 0; index < 3; index++) {
			scopes.push_back(Array{ names[index], (int64_t)variables.create_scope_handle(frame, (DebugVariableStoreV2::ScopeKind)index), index == 2 });
		}
		Array payload;
		payload.push_back(scopes);
		transport.send_data("scopes_result", payload, request.request_id, request.stop_id);
		return true;
	}
	if ((request.type == "variables" || request.type == "inspect_object") && request.payload.size() >= 1) {
		const uint64_t handle = (uint64_t)(int64_t)request.payload[0];
		const int start = request.payload.size() > 1 ? (int)request.payload[1] : 0;
		const int count = request.payload.size() > 2 ? (int)request.payload[2] : variables.get_page_size();
		Array items;
		int total = 0;
		bool more = false;
		int frame = 0;
		DebugVariableStoreV2::ScopeKind scope;
		Error error = OK;
		if (variables.get_scope(handle, frame, scope)) {
			List<String> names;
			List<Variant> values;
			if (scope == DebugVariableStoreV2::SCOPE_LOCALS) {
				p_language->debug_get_stack_level_locals(frame, &names, &values, 0, 0);
			} else if (scope == DebugVariableStoreV2::SCOPE_MEMBERS) {
				if (ScriptInstance *instance = p_language->debug_get_stack_level_instance(frame)) {
					names.push_back("self");
					values.push_back(instance->get_owner());
				}
				p_language->debug_get_stack_level_members(frame, &names, &values, 0, 0);
			} else {
				p_language->debug_get_globals(&names, &values, 0, 0);
			}
			error = variables.describe_named(names, values, start, count, items, total, more);
		} else {
			error = variables.describe_handle(handle, start, count, items, total, more);
		}
		if (error != OK) {
			_request_error(request, "stale_handle", "The variable handle is stale or invalid.");
			return true;
		}
		transport.send_data(request.type == "variables" ? "variables_result" : "inspect_object_result", { total, MAX(start, 0), items, more }, request.request_id, request.stop_id);
		return true;
	}
	_request_error(request, "unknown_data_request", "Unknown debugger request: " + request.type);
	return true;
}

void RemoteDebuggerV2::_poll_control_running() {
	while (transport.is_control_connected()) {
		DebugProtocolV2::Envelope message;
		const Error err = transport.poll_control(message);
		if (err == ERR_UNAVAILABLE) {
			break;
		}
		if (err != OK) {
			continue;
		}
		if (_control(message) == DebugSessionV2::ACTION_PAUSE) {
			debug();
			break;
		}
	}
}

void RemoteDebuggerV2::poll_events(bool p_is_idle) {
	if (in_break) {
		return;
	}
	const uint64_t now = OS::get_singleton()->get_ticks_msec();
	if (!transport.is_connected()) {
		if (transport.is_reconnecting()) {
			const Error err = transport.poll_reconnect();
			if (err == OK) {
				session.connected(transport.get_generation(), now);
			} else if (err != ERR_BUSY) {
				next_reconnect_msec = now + 250;
			}
		} else if (now >= next_reconnect_msec) {
			if (transport.start_reconnect() != ERR_BUSY) {
				next_reconnect_msec = now + 250;
			}
		}
		return;
	}
	_flush_messages();
	transport.flush_data(4);
	_poll_control_running();
	for (int index = 0; index < 8 && _poll_data(nullptr); index++) {
	}
	transport.flush_data(4);
}

void RemoteDebuggerV2::debug(bool p_can_continue, bool p_is_error_breakpoint) {
	if (script_debugger->is_skipping_breakpoints() && !p_is_error_breakpoint) {
		return;
	}
	if (script_debugger->is_ignoring_error_breaks() && p_is_error_breakpoint) {
		return;
	}
	ScriptLanguage *language = script_debugger->get_break_language();
	if (p_is_error_breakpoint && !p_can_continue) {
		transport.send_data("noncontinuable_error", { language ? language->debug_get_error() : String() }, 0, session.get_current_stop_id(), DebugTransportV2::DATA_PRIORITY_INTERACTIVE);
		return;
	}
	if (!transport.is_connected()) {
		_resume_without_frontend();
		return;
	}
	in_break = true;
	const uint64_t stop_id = session.enter_paused(p_can_continue, OS::get_singleton()->get_ticks_msec());
	variables.start_stop(stop_id, transport.get_variable_page_size());
	const String error = language ? language->debug_get_error() : String();
	if (transport.send_control("stopped", { p_can_continue, error, p_is_error_breakpoint }, 0, stop_id) != OK) {
		_resume_without_frontend();
		in_break = false;
		return;
	}
	Input::MouseMode mouse_mode = Input::MOUSE_MODE_VISIBLE;
	if (Thread::is_main_thread()) {
		mouse_mode = Input::get_singleton()->get_mouse_mode();
		Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_VISIBLE);
	}
	while (true) {
		_flush_messages();
		transport.flush_data(4);
		const uint64_t now = OS::get_singleton()->get_ticks_msec();
		if (!transport.is_control_connected() || session.heartbeat_expired(now, transport.get_heartbeat_timeout_msec())) {
			_resume_without_frontend();
			break;
		}
		DebugProtocolV2::Envelope message;
		const Error err = transport.poll_control(message);
		if (err == OK) {
			const DebugSessionV2::Action action = _control(message);
			if (action == DebugSessionV2::ACTION_CONTINUE || action == DebugSessionV2::ACTION_STEP || action == DebugSessionV2::ACTION_NEXT || action == DebugSessionV2::ACTION_STEP_OUT) {
				if (action == DebugSessionV2::ACTION_CONTINUE) {
					script_debugger->set_depth(-1);
					script_debugger->set_lines_left(-1);
				} else {
					int depth = -1;
					if (action == DebugSessionV2::ACTION_NEXT) {
						depth = 0;
					} else if (action == DebugSessionV2::ACTION_STEP_OUT) {
						depth = 1;
					}
					script_debugger->set_depth(depth);
					script_debugger->set_lines_left(1);
				}
				variables.invalidate();
				session.mark_running(now);
				transport.send_control("running", { "running" }, message.request_id, stop_id);
				break;
			}
		} else if (err == ERR_UNAVAILABLE) {
			if (!_poll_data(language)) {
				OS::get_singleton()->delay_usec(1000);
			}
		}
		if (Thread::is_main_thread()) {
			DisplayServer::get_singleton()->force_process_and_drop_events();
		}
	}
	if (Thread::is_main_thread()) {
		Input::get_singleton()->set_mouse_mode(mouse_mode);
	}
	in_break = false;
}
