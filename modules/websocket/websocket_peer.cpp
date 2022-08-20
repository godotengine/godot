/*************************************************************************/
/*  websocket_peer.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "websocket_peer.h"

GDCINULL(WebSocketPeer);

WebSocketPeer::WebSocketPeer() {
}

WebSocketPeer::~WebSocketPeer() {
}

void WebSocketPeer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_write_mode"), &WebSocketPeer::get_write_mode);
	ClassDB::bind_method(D_METHOD("set_write_mode", "mode"), &WebSocketPeer::set_write_mode);
	ClassDB::bind_method(D_METHOD("is_connected_to_host"), &WebSocketPeer::is_connected_to_host);
	ClassDB::bind_method(D_METHOD("was_string_packet"), &WebSocketPeer::was_string_packet);
	ClassDB::bind_method(D_METHOD("close", "code", "reason"), &WebSocketPeer::close, DEFVAL(1000), DEFVAL(""));
	ClassDB::bind_method(D_METHOD("get_connected_host"), &WebSocketPeer::get_connected_host);
	ClassDB::bind_method(D_METHOD("get_connected_port"), &WebSocketPeer::get_connected_port);
	ClassDB::bind_method(D_METHOD("set_no_delay", "enabled"), &WebSocketPeer::set_no_delay);
	ClassDB::bind_method(D_METHOD("get_current_outbound_buffered_amount"), &WebSocketPeer::get_current_outbound_buffered_amount);

	BIND_ENUM_CONSTANT(WRITE_MODE_TEXT);
	BIND_ENUM_CONSTANT(WRITE_MODE_BINARY);
}
