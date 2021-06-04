/*************************************************************************/
/*  debug_adapter_protocol.cpp                                           */
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

#include "debug_adapter_protocol.h"

#include "core/config/project_settings.h"
#include "core/io/json.h"
#include "editor/debugger/script_editor_debugger.h"
#include "editor/doc_tools.h"
#include "editor/editor_log.h"
#include "editor/editor_node.h"

DebugAdapterProtocol *DebugAdapterProtocol::singleton = nullptr;

Error DAPeer::handle_data() {
	int read = 0;
	// Read headers
	if (!has_header) {
		while (true) {
			if (req_pos >= DAP_MAX_BUFFER_SIZE) {
				req_pos = 0;
				ERR_FAIL_COND_V_MSG(true, ERR_OUT_OF_MEMORY, "Response header too big");
			}
			Error err = connection->get_partial_data(&req_buf[req_pos], 1, read);
			if (err != OK) {
				return FAILED;
			} else if (read != 1) { // Busy, wait until next poll
				return ERR_BUSY;
			}
			char *r = (char *)req_buf;
			int l = req_pos;

			// End of headers
			if (l > 3 && r[l] == '\n' && r[l - 1] == '\r' && r[l - 2] == '\n' && r[l - 3] == '\r') {
				r[l - 3] = '\0'; // Null terminate to read string
				String header;
				header.parse_utf8(r);
				content_length = header.substr(16).to_int();
				has_header = true;
				req_pos = 0;
				break;
			}
			req_pos++;
		}
	}
	if (has_header) {
		while (req_pos < content_length) {
			if (content_length >= DAP_MAX_BUFFER_SIZE) {
				req_pos = 0;
				has_header = false;
				ERR_FAIL_COND_V_MSG(req_pos >= DAP_MAX_BUFFER_SIZE, ERR_OUT_OF_MEMORY, "Response content too big");
			}
			Error err = connection->get_partial_data(&req_buf[req_pos], content_length - req_pos, read);
			if (err != OK) {
				return FAILED;
			} else if (read < content_length - req_pos) {
				return ERR_BUSY;
			}
			req_pos += read;
		}

		// Parse data
		String msg;
		msg.parse_utf8((const char *)req_buf, req_pos);

		// Reset to read again
		req_pos = 0;
		has_header = false;

		// Response
		DebugAdapterProtocol::get_singleton()->process_message(msg);
	}
	return OK;
}

Error DAPeer::send_data() {
	while (res_queue.size()) {
		Dictionary data = res_queue.front()->get();
		String formatted_data = format_output(data);

		int data_sent = 0;
		while (data_sent < formatted_data.length()) {
			int curr_sent = 0;
			Error err = connection->put_partial_data((const uint8_t *)formatted_data.utf8().get_data(), formatted_data.size() - data_sent - 1, curr_sent);
			if (err != OK) {
				return err;
			}
			data_sent += curr_sent;
		}
		res_queue.pop_front();
	}
	return OK;
}

String DAPeer::format_output(const Dictionary &p_params) const {
	String response = Variant(p_params).to_json_string();
	String header = "Content-Length: ";
	CharString charstr = response.utf8();
	size_t len = charstr.length();
	header += itos(len);
	header += "\r\n\r\n";

	return header + response;
}

Error DebugAdapterProtocol::on_client_connected() {
	ERR_FAIL_COND_V_MSG(clients.size() >= DAP_MAX_CLIENTS, FAILED, "Max client limits reached");
	Ref<StreamPeerTCP> tcp_peer = server->take_connection();
	tcp_peer->set_no_delay(true);
	Ref<DAPeer> peer = memnew(DAPeer);
	peer->connection = tcp_peer;
	clients.push_back(peer);
	EditorNode::get_log()->add_message("[DAP] Connection Taken", EditorLog::MSG_TYPE_EDITOR);
	return OK;
}

void DebugAdapterProtocol::on_client_disconnected(const Ref<DAPeer> &p_peer) {
	clients.erase(p_peer);
	if (!clients.size()) {
		parser->reset_ids();
	}
	EditorNode::get_log()->add_message("[DAP] Disconnected", EditorLog::MSG_TYPE_EDITOR);
}

void DebugAdapterProtocol::reset_current_info() {
	_current_request = "";
	_current_args = Dictionary();
	_current_peer = Ref<DAPeer>(nullptr);
}

void DebugAdapterProtocol::process_message(const String &p_text) {
	JSON json;
	ERR_FAIL_COND_MSG(json.parse(p_text) != OK, "Mal-formed message!");
	Dictionary params = json.get_data();

	// Append "req_" to any command received; prevents name clash with existing functions, and possibly exploiting
	String command = "req_" + (String)params["command"];
	if (parser->has_method(command)) {
		_current_request = params["command"];
		_current_args = params["arguments"];

		Array args;
		args.push_back(_current_peer);
		args.push_back(params);
		Dictionary response = parser->callv(command, args);
		_current_peer->res_queue.push_front(response);
	}

	reset_current_info();
}

void DebugAdapterProtocol::notify_initialized() {
	Dictionary event = parser->ev_initialized(_current_peer);
	_current_peer->res_queue.push_back(event);
}

void DebugAdapterProtocol::notify_process() {
	String launch_mode = _current_request.is_empty() ? "launch" : _current_request;

	for (List<Ref<DAPeer>>::Element *E = clients.front(); E; E = E->next()) {
		Dictionary event = parser->ev_process(E->get(), launch_mode);
		E->get()->res_queue.push_back(event);
	}
}

void DebugAdapterProtocol::notify_terminated() {
	for (List<Ref<DAPeer>>::Element *E = clients.front(); E; E = E->next()) {
		if (_current_request == "launch" && _current_peer == E->get()) {
			continue;
		}
		Dictionary event = parser->ev_terminated(E->get());
		E->get()->res_queue.push_back(event);
	}
}

void DebugAdapterProtocol::notify_exited(const int &p_exitcode) {
	for (List<Ref<DAPeer>>::Element *E = clients.front(); E; E = E->next()) {
		if (_current_request == "launch" && _current_peer == E->get()) {
			continue;
		}
		Dictionary event = parser->ev_exited(E->get(), p_exitcode);
		E->get()->res_queue.push_back(event);
	}
}

void DebugAdapterProtocol::notify_stopped(const DAP::StopReason &p_reason) {
	for (List<Ref<DAPeer>>::Element *E = clients.front(); E; E = E->next()) {
		Dictionary event = parser->ev_stopped(E->get(), p_reason);
		E->get()->res_queue.push_back(event);
	}
}

void DebugAdapterProtocol::notify_continued() {
	for (List<Ref<DAPeer>>::Element *E = clients.front(); E; E = E->next()) {
		if (_current_request == "continue" && E->get() == _current_peer) {
			continue;
		}
		Dictionary event = parser->ev_continued(E->get());
		E->get()->res_queue.push_back(event);
	}
}

void DebugAdapterProtocol::on_debug_paused() {
	if (EditorNode::get_singleton()->get_pause_button()->is_pressed()) {
		notify_stopped(DAP::StopReason::PAUSE);
	} else {
		notify_continued();
	}
}

void DebugAdapterProtocol::on_debug_stopped() {
	notify_exited();
	notify_terminated();
}

void DebugAdapterProtocol::poll() {
	if (server->is_connection_available()) {
		on_client_connected();
	}
	List<Ref<DAPeer>> to_delete;
	for (List<Ref<DAPeer>>::Element *E = clients.front(); E; E = E->next()) {
		Ref<DAPeer> peer = E->get();
		StreamPeerTCP::Status status = peer->connection->get_status();
		if (status == StreamPeerTCP::STATUS_NONE || status == StreamPeerTCP::STATUS_ERROR) {
			to_delete.push_back(peer);
		} else {
			_current_peer = peer;
			if (peer->connection->get_available_bytes() > 0) {
				Error err = peer->handle_data();
				if (err != OK && err != ERR_BUSY) {
					to_delete.push_back(peer);
				}
			}
			Error err = peer->send_data();
			if (err != OK && err != ERR_BUSY) {
				to_delete.push_back(peer);
			}
		}
	}

	for (List<Ref<DAPeer>>::Element *E = to_delete.front(); E; E = E->next()) {
		on_client_disconnected(E->get());
	}
	to_delete.clear();
}

Error DebugAdapterProtocol::start(int p_port, const IPAddress &p_bind_ip) {
	_initialized = true;
	return server->listen(p_port, p_bind_ip);
}

void DebugAdapterProtocol::stop() {
	for (List<Ref<DAPeer>>::Element *E = clients.front(); E; E = E->next()) {
		E->get()->connection->disconnect_from_host();
	}

	clients.clear();
	server->stop();
	_initialized = false;
}

DebugAdapterProtocol::DebugAdapterProtocol() {
	server.instantiate();
	singleton = this;
	parser = memnew(DebugAdapterParser);

	EditorNode *node = EditorNode::get_singleton();
	node->get_pause_button()->connect("pressed", callable_mp(this, &DebugAdapterProtocol::on_debug_paused));

	EditorDebuggerNode *debugger_node = EditorDebuggerNode::get_singleton();
	debugger_node->get_default_debugger()->connect("stopped", callable_mp(this, &DebugAdapterProtocol::on_debug_stopped));
}

DebugAdapterProtocol::~DebugAdapterProtocol() {
	memdelete(parser);
}
