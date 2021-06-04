/*************************************************************************/
/*  debug_adapter_parser.cpp                                             */
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

#include "debug_adapter_parser.h"

#include "editor/editor_node.h"

void DebugAdapterParser::_bind_methods() {
	// Requests
	ClassDB::bind_method(D_METHOD("req_initialize", "peer", "params"), &DebugAdapterParser::req_initialize);
	ClassDB::bind_method(D_METHOD("req_disconnect", "peer", "params"), &DebugAdapterParser::prepare_success_response);
	ClassDB::bind_method(D_METHOD("req_launch", "peer", "params"), &DebugAdapterParser::req_launch);
	ClassDB::bind_method(D_METHOD("req_terminate", "peer", "params"), &DebugAdapterParser::req_terminate);
	ClassDB::bind_method(D_METHOD("req_configurationDone", "peer", "params"), &DebugAdapterParser::prepare_success_response);
	ClassDB::bind_method(D_METHOD("req_pause", "peer", "params"), &DebugAdapterParser::req_pause);
	ClassDB::bind_method(D_METHOD("req_continue", "peer", "params"), &DebugAdapterParser::req_continue);
	ClassDB::bind_method(D_METHOD("req_threads", "peer", "params"), &DebugAdapterParser::req_threads);
	ClassDB::bind_method(D_METHOD("req_stackTrace", "peer", "params"), &DebugAdapterParser::req_stackTrace);
}

Dictionary DebugAdapterParser::prepare_base_message(Ref<DAPeer> p_peer) const {
	Dictionary message;
	message["seq"] = ++(p_peer->seq);

	return message;
}

Dictionary DebugAdapterParser::prepare_base_event(Ref<DAPeer> p_peer) const {
	Dictionary event = prepare_base_message(p_peer);
	event["type"] = "event";

	return event;
}

Dictionary DebugAdapterParser::prepare_success_response(Ref<DAPeer> p_peer, const Dictionary &p_params) const {
	Dictionary response = prepare_base_message(p_peer);
	response["type"] = "response";
	response["request_seq"] = p_params["seq"];
	response["command"] = p_params["command"];
	response["success"] = true;

	return response;
}

Dictionary DebugAdapterParser::prepare_error_response(Ref<DAPeer> p_peer, const Dictionary &p_params, DAP::ErrorType err_type) {
	Dictionary response = prepare_base_message(p_peer);
	response["type"] = "response";
	response["request_seq"] = p_params["seq"];
	response["command"] = p_params["command"];
	response["success"] = false;

	DAP::Message message;
	message.id = generate_message_id();
	String error, error_desc;
	switch (err_type) {
		case DAP::ErrorType::UNKNOWN:
			error = "unknown";
			error_desc = "An unknown error has ocurred when processing the request.";
			break;
	}

	message.format = error_desc;
	response["message"] = error;
	response["body"] = message.to_json();

	return response;
}

Dictionary DebugAdapterParser::req_initialize(Ref<DAPeer> p_peer, const Dictionary &p_params) const {
	Dictionary response = prepare_success_response(p_peer, p_params);
	Dictionary args = p_params["arguments"];

	p_peer->linesStartAt1 = args.get("linesStartAt1", false);
	p_peer->columnsStartAt1 = args.get("columnsStartAt1", false);
	p_peer->supportsVariableType = args.get("supportsVariableType", false);
	p_peer->supportsInvalidatedEvent = args.get("supportsInvalidatedEvent", false);

	DAP::Capabilities caps;
	response["body"] = caps.to_json();

	DebugAdapterProtocol::get_singleton()->notify_initialized();

	return response;
}

Dictionary DebugAdapterParser::req_launch(Ref<DAPeer> p_peer, const Dictionary &p_params) const {
	Dictionary response = prepare_success_response(p_peer, p_params);
	EditorNode::get_singleton()->run_play();

	DebugAdapterProtocol::get_singleton()->notify_process();

	return response;
}

Dictionary DebugAdapterParser::req_terminate(Ref<DAPeer> p_peer, const Dictionary &p_params) const {
	Dictionary response = prepare_success_response(p_peer, p_params);
	EditorNode::get_singleton()->run_stop();

	DebugAdapterProtocol::get_singleton()->notify_terminated();
	DebugAdapterProtocol::get_singleton()->notify_exited();

	return response;
}

Dictionary DebugAdapterParser::req_pause(Ref<DAPeer> p_peer, const Dictionary &p_params) const {
	Dictionary response = prepare_success_response(p_peer, p_params);
	EditorNode::get_singleton()->get_pause_button()->set_pressed(true);
	EditorDebuggerNode::get_singleton()->_paused();

	DebugAdapterProtocol::get_singleton()->notify_stopped(DAP::StopReason::PAUSE);

	return response;
}

Dictionary DebugAdapterParser::req_continue(Ref<DAPeer> p_peer, const Dictionary &p_params) const {
	Dictionary response = prepare_success_response(p_peer, p_params);
	EditorNode::get_singleton()->get_pause_button()->set_pressed(false);
	EditorDebuggerNode::get_singleton()->_paused();

	DebugAdapterProtocol::get_singleton()->notify_continued();

	return response;
}

Dictionary DebugAdapterParser::req_threads(Ref<DAPeer> p_peer, const Dictionary &p_params) const {
	Dictionary response = prepare_success_response(p_peer, p_params), body;
	response["body"] = body;

	Array arr;
	DAP::Thread thread;

	thread.id = 1; // Hardcoded because Godot only supports debugging one thread at the moment
	thread.name = "Main";
	arr.push_back(thread.to_json());
	body["threads"] = arr;

	return response;
}

Dictionary DebugAdapterParser::req_stackTrace(Ref<DAPeer> p_peer, const Dictionary &p_params) const {
	Dictionary response = prepare_success_response(p_peer, p_params), body;
	response["body"] = body;

	Array arr;
	body["stackFrames"] = arr;

	return response;
}

Dictionary DebugAdapterParser::ev_initialized(Ref<DAPeer> p_peer) const {
	Dictionary event = prepare_base_event(p_peer);
	event["event"] = "initialized";

	return event;
}

Dictionary DebugAdapterParser::ev_process(Ref<DAPeer> p_peer, const String &p_command) const {
	Dictionary event = prepare_base_event(p_peer), body;
	event["event"] = "process";
	event["body"] = body;

	body["name"] = OS::get_singleton()->get_executable_path();
	body["startMethod"] = p_command;

	return event;
}

Dictionary DebugAdapterParser::ev_terminated(Ref<DAPeer> p_peer) const {
	Dictionary event = prepare_base_event(p_peer);
	event["event"] = "terminated";

	return event;
}

Dictionary DebugAdapterParser::ev_exited(Ref<DAPeer> p_peer, const int &p_exitcode) const {
	Dictionary event = prepare_base_event(p_peer), body;
	event["event"] = "exited";
	event["body"] = body;

	body["exitCode"] = p_exitcode;

	return event;
}

Dictionary DebugAdapterParser::ev_stopped(Ref<DAPeer> p_peer, DAP::StopReason p_reason) const {
	Dictionary event = prepare_base_event(p_peer), body;
	event["event"] = "stopped";
	event["body"] = body;

	switch (p_reason) {
		case DAP::StopReason::STEP:
			body["reason"] = "step";
			break;
		case DAP::StopReason::BREAKPOINT:
			body["reason"] = "breakpoint";
			break;
		case DAP::StopReason::EXCEPTION:
			body["reason"] = "exception";
			break;
		case DAP::StopReason::PAUSE:
			body["reason"] = "pause";
			break;
	}

	body["threadId"] = 1;

	return event;
}

Dictionary DebugAdapterParser::ev_continued(Ref<DAPeer> p_peer) const {
	Dictionary event = prepare_base_event(p_peer), body;
	event["event"] = "continued";
	event["body"] = body;

	body["threadId"] = 1;

	return event;
}

DebugAdapterParser::DebugAdapterParser() {
	reset_ids();
}

void DebugAdapterParser::reset_ids() {
	messageId = 0;
}

int DebugAdapterParser::generate_message_id() {
	return ++messageId;
}
