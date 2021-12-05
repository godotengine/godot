/*************************************************************************/
/*  editor_debugger_server_websocket.h                                   */
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

#ifndef EDITOR_DEBUGGER_SERVER_WEBSOCKET_H
#define EDITOR_DEBUGGER_SERVER_WEBSOCKET_H

#include "editor/debugger/editor_debugger_server.h"
#include "modules/websocket/websocket_server.h"

class EditorDebuggerServerWebSocket : public EditorDebuggerServer {
	GDCLASS(EditorDebuggerServerWebSocket, EditorDebuggerServer);

private:
	Ref<WebSocketServer> server;
	List<int> pending_peers;
	String endpoint;

public:
	static EditorDebuggerServer *create(const String &p_protocol);

	void _peer_connected(int p_peer, String p_protocol);
	void _peer_disconnected(int p_peer, bool p_was_clean);

	virtual void poll() override;
	virtual String get_uri() const override;
	virtual Error start(const String &p_uri = "") override;
	virtual void stop() override;
	virtual bool is_active() const override;
	virtual bool is_connection_available() const override;
	virtual Ref<RemoteDebuggerPeer> take_connection() override;

	EditorDebuggerServerWebSocket();
	~EditorDebuggerServerWebSocket();
};

#endif // EDITOR_DEBUGGER_SERVER_WEBSOCKET_H
