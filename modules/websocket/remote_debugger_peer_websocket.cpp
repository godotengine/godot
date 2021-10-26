/*************************************************************************/
/*  remote_debugger_peer_websocket.cpp                                   */
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

#include "remote_debugger_peer_websocket.h"

#include "core/config/project_settings.h"

Error RemoteDebuggerPeerWebSocket::connect_to_host(const String &p_uri) {
	Vector<String> protocols;
	protocols.push_back("binary"); // Compatibility for emscripten TCP-to-WebSocket.

	ws_client->connect_to_url(p_uri, protocols);
	ws_client->poll();

	if (ws_client->get_connection_status() == WebSocketClient::CONNECTION_DISCONNECTED) {
		ERR_PRINT("Remote Debugger: Unable to connect. Status: " + String::num(ws_client->get_connection_status()) + ".");
		return FAILED;
	}

	ws_peer = ws_client->get_peer(1);

	return OK;
}

bool RemoteDebuggerPeerWebSocket::is_peer_connected() {
	return ws_peer.is_valid() && ws_peer->is_connected_to_host();
}

void RemoteDebuggerPeerWebSocket::poll() {
	ws_client->poll();

	while (ws_peer->is_connected_to_host() && ws_peer->get_available_packet_count() > 0 && in_queue.size() < max_queued_messages) {
		Variant var;
		Error err = ws_peer->get_var(var);
		ERR_CONTINUE(err != OK);
		ERR_CONTINUE(var.get_type() != Variant::ARRAY);
		in_queue.push_back(var);
	}

	while (ws_peer->is_connected_to_host() && out_queue.size() > 0) {
		Array var = out_queue[0];
		Error err = ws_peer->put_var(var);
		ERR_BREAK(err != OK); // Peer buffer full?
		out_queue.pop_front();
	}
}

int RemoteDebuggerPeerWebSocket::get_max_message_size() const {
	return 8 << 20; // 8 Mib
}

bool RemoteDebuggerPeerWebSocket::has_message() {
	return in_queue.size() > 0;
}

Array RemoteDebuggerPeerWebSocket::get_message() {
	ERR_FAIL_COND_V(in_queue.size() < 1, Array());
	Array msg = in_queue[0];
	in_queue.pop_front();
	return msg;
}

Error RemoteDebuggerPeerWebSocket::put_message(const Array &p_arr) {
	if (out_queue.size() >= max_queued_messages) {
		return ERR_OUT_OF_MEMORY;
	}
	out_queue.push_back(p_arr);
	return OK;
}

void RemoteDebuggerPeerWebSocket::close() {
	if (ws_peer.is_valid()) {
		ws_peer.unref();
	}
	ws_client->disconnect_from_host();
}

bool RemoteDebuggerPeerWebSocket::can_block() const {
#ifdef JAVASCRIPT_ENABLED
	return false;
#else
	return true;
#endif
}

RemoteDebuggerPeerWebSocket::RemoteDebuggerPeerWebSocket(Ref<WebSocketPeer> p_peer) {
#ifdef JAVASCRIPT_ENABLED
	ws_client = Ref<WebSocketClient>(memnew(EMWSClient));
#else
	ws_client = Ref<WebSocketClient>(memnew(WSLClient));
#endif
	max_queued_messages = (int)GLOBAL_GET("network/limits/debugger/max_queued_messages");
	ws_client->set_buffers(8192, max_queued_messages, 8192, max_queued_messages);
	ws_peer = p_peer;
}

RemoteDebuggerPeer *RemoteDebuggerPeerWebSocket::create(const String &p_uri) {
	ERR_FAIL_COND_V(!p_uri.begins_with("ws://") && !p_uri.begins_with("wss://"), nullptr);
	RemoteDebuggerPeerWebSocket *peer = memnew(RemoteDebuggerPeerWebSocket);
	Error err = peer->connect_to_host(p_uri);
	if (err != OK) {
		memdelete(peer);
		return nullptr;
	}
	return peer;
}
