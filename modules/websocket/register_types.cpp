/**************************************************************************/
/*  register_types.cpp                                                    */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "register_types.h"

#include "core/error_macros.h"
#include "core/project_settings.h"

#include "websocket_client.h"
#include "websocket_server.h"

#ifdef JAVASCRIPT_ENABLED
#include "emscripten.h"
#include "emws_client.h"
#include "emws_peer.h"
#else
#include "wsl_client.h"
#include "wsl_server.h"
#endif

void register_websocket_types() {
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
	EMWSPeer::make_default();
	EMWSClient::make_default();
#else
	WSLPeer::make_default();
	WSLClient::make_default();
	WSLServer::make_default();
#endif

	ClassDB::register_virtual_class<WebSocketMultiplayerPeer>();
	ClassDB::register_custom_instance_class<WebSocketServer>();
	ClassDB::register_custom_instance_class<WebSocketClient>();
	ClassDB::register_custom_instance_class<WebSocketPeer>();
}

void unregister_websocket_types() {}
