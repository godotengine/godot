/*************************************************************************/
/*  script_editor_debugger_websocket.h                                   */
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

#ifndef SCRIPT_EDITOR_DEBUGGER_WEBSOCKET_H
#define SCRIPT_EDITOR_DEBUGGER_WEBSOCKET_H

#include "modules/websocket/websocket_server.h"

#include "editor/script_editor_debugger.h"

class ScriptEditorDebuggerWebSocket : public ScriptEditorDebuggerServer {

	GDCLASS(ScriptEditorDebuggerWebSocket, ScriptEditorDebuggerServer);

private:
	Ref<WebSocketServer> server;
	int peer_id;
	bool just_connected;

protected:
	static void _bind_methods();

	virtual Error start_server(int p_port);
	virtual void stop_server();
	virtual void handle_connections(bool &r_connected, bool &r_disconnected);
	virtual bool has_peer();
	virtual Ref<PacketPeer> get_peer();

public:
	void _peer_connected(int p_peer, String p_protocol);

	ScriptEditorDebuggerWebSocket();
	~ScriptEditorDebuggerWebSocket();
};

#endif // SCRIPT_EDITOR_DEBUGGER_WEBSOCKET_H
