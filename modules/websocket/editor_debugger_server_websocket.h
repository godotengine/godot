/*************************************************************************/
/*  editor_debugger_server_websocket.h                                   */
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

#ifndef SCRIPT_EDITOR_DEBUGGER_WEBSOCKET_H
#define SCRIPT_EDITOR_DEBUGGER_WEBSOCKET_H

#include "modules/websocket/websocket_server.h"

#include "editor/debugger/editor_debugger_server.h"

class EditorDebuggerServerWebSocket : public EditorDebuggerServer {
	GDCLASS(EditorDebuggerServerWebSocket, EditorDebuggerServer);

private:
	Ref<WebSocketServer> server;
	List<int> pending_peers;

public:
	static EditorDebuggerServer *create(const String &p_protocol);

	void _peer_connected(int p_peer, String p_protocol);
	void _peer_disconnected(int p_peer, bool p_was_clean);

	void poll() override;
	Error start() override;
	void stop() override;
	bool is_active() const override;
	bool is_connection_available() const override;
	Ref<RemoteDebuggerPeer> take_connection() override;

	EditorDebuggerServerWebSocket();
	~EditorDebuggerServerWebSocket();
};

#endif // SCRIPT_EDITOR_DEBUGGER_WEBSOCKET_H
