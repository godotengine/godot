/*************************************************************************/
/*  script_debugger_websocket.cpp                                        */
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

#include "script_debugger_websocket.h"

#include "core/project_settings.h"

ScriptDebuggerConnection *ScriptDebuggerWebSocket::_create() {
	return memnew(ScriptDebuggerWebSocket);
}

void ScriptDebuggerWebSocket::make_default() {
	ScriptDebuggerConnection::_create = _create;
}

Error ScriptDebuggerWebSocket::connect_to_host(const String &p_host, uint16_t p_port) {

	IP_Address ip;
	if (p_host.is_valid_ip_address())
		ip = p_host;
	else
		ip = IP::get_singleton()->resolve_hostname(p_host);

	int port = p_port;
	Vector<String> protocols;
	protocols.push_back("binary"); // Compatibility for emscripten TCP-to-WebSocket.

	const int tries = 6;
	int waits[tries] = { 10, 10, 100, 1000, 1000, 1000 };

	for (int i = 0; i < tries; i++) {

		if (ws_client->get_connection_status() == WebSocketClient::CONNECTION_DISCONNECTED)
			ws_client->connect_to_url("ws://" + p_host + ":" + itos(port), protocols);
		ws_client->poll();

		// Wait and poll
		const int ms = waits[i];
		for (int j = 0; j < ms; j++) {
			ws_client->poll();
			if (is_connected_to_host()) {
				print_verbose("Remote Debugger: Connected!");
				return OK;
			}
			OS::get_singleton()->delay_usec(1000);
		}
		print_verbose("Remote Debugger: Connection failed with status: '" + String::num(ws_client->get_connection_status()) + "', retrying in " + String::num(ms) + " msec.");
	};

	if (!is_connected_to_host()) {

		ERR_PRINTS("Remote Debugger: Unable to connect. Status: " + String::num(ws_client->get_connection_status()) + ".");
		return FAILED;
	};

	return FAILED;
}

void ScriptDebuggerWebSocket::poll() {
	ws_client->poll();
}

bool ScriptDebuggerWebSocket::is_connected_to_host() {
	return ws_client->get_connection_status() == WebSocketClient::CONNECTION_CONNECTED;
}

Ref<PacketPeer> ScriptDebuggerWebSocket::get_peer() {
	if (!is_connected_to_host())
		return memnew(PacketPeerStream); // avoid returning null.
	return ws_client->get_peer(1); // should not be cached.
}

ScriptDebuggerWebSocket::ScriptDebuggerWebSocket() {
#define _SET_HINT(NAME, _VAL_, _MAX_) \
	GLOBAL_DEF(NAME, _VAL_);          \
	ProjectSettings::get_singleton()->set_custom_property_info(NAME, PropertyInfo(Variant::INT, NAME, PROPERTY_HINT_RANGE, "2," #_MAX_ ",1,or_greater"));

	// Client buffers project settings
	_SET_HINT(WSC_IN_BUF, 64, 4096);
	_SET_HINT(WSC_IN_PKT, 1024, 16384);
	_SET_HINT(WSC_OUT_BUF, 64, 4096);
	_SET_HINT(WSC_OUT_PKT, 1024, 16384);

	// Server buffers project settings
	_SET_HINT(WSS_IN_BUF, 64, 4096);
	_SET_HINT(WSS_IN_PKT, 1024, 16384);
	_SET_HINT(WSS_OUT_BUF, 64, 4096);
	_SET_HINT(WSS_OUT_PKT, 1024, 16384);

#ifdef JAVASCRIPT_ENABLED
	ws_client = Ref<WebSocketClient>(memnew(EMWSClient));
#else
	ws_client = Ref<WebSocketClient>(memnew(WSLClient));
#endif
}
