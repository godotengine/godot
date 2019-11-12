/*************************************************************************/
/*  script_editor_debugger_websocket.cpp                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "script_editor_debugger_websocket.h"

void ScriptEditorDebuggerWebSocket::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_peer_connected"), &ScriptEditorDebuggerWebSocket::_peer_connected);
}

void ScriptEditorDebuggerWebSocket::_peer_connected(int p_id, String _protocol) {
	if (peer_id) {
		server->disconnect_peer(p_id);
		return;
	}
	peer_id = p_id;
	just_connected = true;
}

Error ScriptEditorDebuggerWebSocket::start_server(int p_port) {
	Vector<String> protocols;
	protocols.push_back("binary"); // compatibility with EMSCRIPTEN TCP-to-WebSocket layer.
	return server->listen(p_port, protocols);
}

void ScriptEditorDebuggerWebSocket::stop_server() {
	server->stop();
	peer_id = 0;
	just_connected = false;
}

void ScriptEditorDebuggerWebSocket::handle_connections(bool &r_connected, bool &r_disconnected) {
	r_connected = false;
	r_disconnected = false;

	server->poll();

	// Was connected but got a disconnection.
	if (peer_id && !server->has_peer(peer_id)) {
		r_disconnected = true;
		peer_id = 0;
		return;
	}

	if (just_connected) {
		just_connected = false;
		r_connected = true;
	}
}

bool ScriptEditorDebuggerWebSocket::has_peer() {
	return server->has_peer(peer_id);
}

Ref<PacketPeer> ScriptEditorDebuggerWebSocket::get_peer() {
	return server->get_peer(peer_id);
}

ScriptEditorDebuggerWebSocket::ScriptEditorDebuggerWebSocket() :
		server(WebSocketServer::create()),
		peer_id(0),
		just_connected(false) {
	server->connect("client_connected", this, "_peer_connected");
}

ScriptEditorDebuggerWebSocket::~ScriptEditorDebuggerWebSocket() {
	stop_server();
}
