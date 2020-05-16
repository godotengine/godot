/*************************************************************************/
/*  register_types.cpp                                                   */
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

#include "register_types.h"
#include "core/error_macros.h"
#include "core/project_settings.h"
#ifdef JAVASCRIPT_ENABLED
#include "emscripten.h"
#include "emws_client.h"
#include "emws_peer.h"
#include "emws_server.h"
#else
#include "wsl_client.h"
#include "wsl_server.h"
#endif
#ifdef TOOLS_ENABLED
#include "editor/debugger/editor_debugger_server.h"
#include "editor/editor_node.h"
#include "editor_debugger_server_websocket.h"
#endif

#ifdef TOOLS_ENABLED
static void _editor_init_callback() {
	EditorDebuggerServer::register_protocol_handler("ws://", EditorDebuggerServerWebSocket::create);
}
#endif

void register_websocket_types() {
#ifdef JAVASCRIPT_ENABLED
	EMWSPeer::make_default();
	EMWSClient::make_default();
	EMWSServer::make_default();
#else
	WSLPeer::make_default();
	WSLClient::make_default();
	WSLServer::make_default();
#endif

	ClassDB::register_virtual_class<WebSocketMultiplayerPeer>();
	ClassDB::register_custom_instance_class<WebSocketServer>();
	ClassDB::register_custom_instance_class<WebSocketClient>();
	ClassDB::register_custom_instance_class<WebSocketPeer>();

#ifdef TOOLS_ENABLED
	EditorNode::add_init_callback(&_editor_init_callback);
#endif
}

void unregister_websocket_types() {}
