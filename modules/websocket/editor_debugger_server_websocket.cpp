/*************************************************************************/
/*  editor_debugger_server_websocket.cpp                                 */
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

#include "editor_debugger_server_websocket.h"

#include "core/config/project_settings.h"
#include "editor/editor_log.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "modules/websocket/remote_debugger_peer_websocket.h"

void EditorDebuggerServerWebSocket::_peer_connected(int p_id, String _protocol) {
	pending_peers.push_back(p_id);
}

void EditorDebuggerServerWebSocket::_peer_disconnected(int p_id, bool p_was_clean) {
	if (pending_peers.find(p_id)) {
		pending_peers.erase(p_id);
	}
}

void EditorDebuggerServerWebSocket::poll() {
	server->poll();
}

String EditorDebuggerServerWebSocket::get_uri() const {
	return endpoint;
}

Error EditorDebuggerServerWebSocket::start(const String &p_uri) {
	// Default host and port
	String bind_host = (String)EditorSettings::get_singleton()->get("network/debug/remote_host");
	int bind_port = (int)EditorSettings::get_singleton()->get("network/debug/remote_port");

	// Optionally override
	if (!p_uri.is_empty() && p_uri != "ws://") {
		String scheme, path;
		Error err = p_uri.parse_url(scheme, bind_host, bind_port, path);
		ERR_FAIL_COND_V(err != OK, ERR_INVALID_PARAMETER);
		ERR_FAIL_COND_V(!bind_host.is_valid_ip_address() && bind_host != "*", ERR_INVALID_PARAMETER);
	}

	// Set up the server
	server->set_bind_ip(bind_host);
	Vector<String> compatible_protocols;
	compatible_protocols.push_back("binary"); // compatibility with EMSCRIPTEN TCP-to-WebSocket layer.

	// Try listening on ports
	const int max_attempts = 5;
	for (int attempt = 1;; ++attempt) {
		const Error err = server->listen(bind_port, compatible_protocols);
		if (err == OK) {
			break;
		}
		if (attempt >= max_attempts) {
			EditorNode::get_log()->add_message(vformat("Cannot listen on port %d, remote debugging unavailable.", bind_port), EditorLog::MSG_TYPE_ERROR);
			return err;
		}
		int last_port = bind_port++;
		EditorNode::get_log()->add_message(vformat("Cannot listen on port %d, trying %d instead.", last_port, bind_port), EditorLog::MSG_TYPE_WARNING);
	}

	// Endpoint that the client should connect to
	endpoint = vformat("ws://%s:%d", bind_host, bind_port);

	return OK;
}

void EditorDebuggerServerWebSocket::stop() {
	server->stop();
	pending_peers.clear();
}

bool EditorDebuggerServerWebSocket::is_active() const {
	return server->is_listening();
}

bool EditorDebuggerServerWebSocket::is_connection_available() const {
	return pending_peers.size() > 0;
}

Ref<RemoteDebuggerPeer> EditorDebuggerServerWebSocket::take_connection() {
	ERR_FAIL_COND_V(!is_connection_available(), Ref<RemoteDebuggerPeer>());
	RemoteDebuggerPeer *peer = memnew(RemoteDebuggerPeerWebSocket(server->get_peer(pending_peers[0])));
	pending_peers.pop_front();
	return peer;
}

EditorDebuggerServerWebSocket::EditorDebuggerServerWebSocket() {
	server = Ref<WebSocketServer>(WebSocketServer::create());
	int max_pkts = (int)GLOBAL_GET("network/limits/debugger/max_queued_messages");
	server->set_buffers(8192, max_pkts, 8192, max_pkts);
	server->connect("client_connected", callable_mp(this, &EditorDebuggerServerWebSocket::_peer_connected));
	server->connect("client_disconnected", callable_mp(this, &EditorDebuggerServerWebSocket::_peer_disconnected));
}

EditorDebuggerServerWebSocket::~EditorDebuggerServerWebSocket() {
	stop();
}

EditorDebuggerServer *EditorDebuggerServerWebSocket::create(const String &p_protocol) {
	ERR_FAIL_COND_V(p_protocol != "ws://", nullptr);
	return memnew(EditorDebuggerServerWebSocket);
}
