/*************************************************************************/
/*  webrtc_peer.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "webrtc_peer.h"

WebRTCPeer *(*WebRTCPeer::_create)() = NULL;

Ref<WebRTCPeer> WebRTCPeer::create_ref() {

	if (!_create)
		return Ref<WebRTCPeer>();
	return Ref<WebRTCPeer>(_create());
}

WebRTCPeer *WebRTCPeer::create() {

	if (!_create)
		return NULL;
	return _create();
}

void WebRTCPeer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("create_offer"), &WebRTCPeer::create_offer);
	ClassDB::bind_method(D_METHOD("set_local_description", "type", "sdp"), &WebRTCPeer::set_local_description);
	ClassDB::bind_method(D_METHOD("set_remote_description", "type", "sdp"), &WebRTCPeer::set_remote_description);
	ClassDB::bind_method(D_METHOD("poll"), &WebRTCPeer::poll);
	ClassDB::bind_method(D_METHOD("add_ice_candidate", "media", "index", "name"), &WebRTCPeer::add_ice_candidate);

	ClassDB::bind_method(D_METHOD("was_string_packet"), &WebRTCPeer::was_string_packet);
	ClassDB::bind_method(D_METHOD("set_write_mode", "write_mode"), &WebRTCPeer::set_write_mode);
	ClassDB::bind_method(D_METHOD("get_write_mode"), &WebRTCPeer::get_write_mode);
	ClassDB::bind_method(D_METHOD("get_connection_state"), &WebRTCPeer::get_connection_state);

	ADD_SIGNAL(MethodInfo("offer_created", PropertyInfo(Variant::STRING, "type"), PropertyInfo(Variant::STRING, "sdp")));
	ADD_SIGNAL(MethodInfo("new_ice_candidate", PropertyInfo(Variant::STRING, "media"), PropertyInfo(Variant::INT, "index"), PropertyInfo(Variant::STRING, "name")));

	ADD_PROPERTY(PropertyInfo(Variant::INT, "write_mode", PROPERTY_HINT_ENUM), "set_write_mode", "get_write_mode");

	BIND_ENUM_CONSTANT(WRITE_MODE_TEXT);
	BIND_ENUM_CONSTANT(WRITE_MODE_BINARY);

	BIND_ENUM_CONSTANT(STATE_NEW);
	BIND_ENUM_CONSTANT(STATE_CONNECTING);
	BIND_ENUM_CONSTANT(STATE_CONNECTED);
	BIND_ENUM_CONSTANT(STATE_DISCONNECTED);
	BIND_ENUM_CONSTANT(STATE_FAILED);
	BIND_ENUM_CONSTANT(STATE_CLOSED);
}

WebRTCPeer::WebRTCPeer() {
}

WebRTCPeer::~WebRTCPeer() {
}
