/**************************************************************************/
/*  webrtc_peer_connection.cpp                                            */
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

#include "webrtc_peer_connection.h"

#ifdef WEB_ENABLED
#include "webrtc_peer_connection_js.h"
#endif

#include "webrtc_peer_connection_extension.h"

StringName WebRTCPeerConnection::default_extension;

void WebRTCPeerConnection::set_default_extension(const StringName &p_extension) {
	ERR_FAIL_COND_MSG(!ClassDB::is_parent_class(p_extension, WebRTCPeerConnectionExtension::get_class_static()), vformat("Can't make %s the default WebRTC extension since it does not extend WebRTCPeerConnectionExtension.", p_extension));
	default_extension = StringName(p_extension, true);
}

WebRTCPeerConnection *WebRTCPeerConnection::create() {
#ifdef WEB_ENABLED
	return memnew(WebRTCPeerConnectionJS);
#else
	if (default_extension == StringName()) {
		WARN_PRINT_ONCE("No default WebRTC extension configured.");
		return memnew(WebRTCPeerConnectionExtension);
	}
	Object *obj = ClassDB::instantiate(default_extension);
	return Object::cast_to<WebRTCPeerConnectionExtension>(obj);
#endif
}

void WebRTCPeerConnection::_bind_methods() {
	ClassDB::bind_static_method(get_class_static(), D_METHOD("set_default_extension", "extension_class"), &WebRTCPeerConnectionExtension::set_default_extension);

	ClassDB::bind_method(D_METHOD("initialize", "configuration"), &WebRTCPeerConnection::initialize, DEFVAL(Dictionary()));
	ClassDB::bind_method(D_METHOD("create_data_channel", "label", "options"), &WebRTCPeerConnection::create_data_channel, DEFVAL(Dictionary()));
	ClassDB::bind_method(D_METHOD("create_offer"), &WebRTCPeerConnection::create_offer);
	ClassDB::bind_method(D_METHOD("set_local_description", "type", "sdp"), &WebRTCPeerConnection::set_local_description);
	ClassDB::bind_method(D_METHOD("set_remote_description", "type", "sdp"), &WebRTCPeerConnection::set_remote_description);
	ClassDB::bind_method(D_METHOD("add_ice_candidate", "media", "index", "name"), &WebRTCPeerConnection::add_ice_candidate);
	ClassDB::bind_method(D_METHOD("poll"), &WebRTCPeerConnection::poll);
	ClassDB::bind_method(D_METHOD("close"), &WebRTCPeerConnection::close);

	ClassDB::bind_method(D_METHOD("get_connection_state"), &WebRTCPeerConnection::get_connection_state);
	ClassDB::bind_method(D_METHOD("get_gathering_state"), &WebRTCPeerConnection::get_gathering_state);
	ClassDB::bind_method(D_METHOD("get_signaling_state"), &WebRTCPeerConnection::get_signaling_state);

	ADD_SIGNAL(MethodInfo("session_description_created", PropertyInfo(Variant::STRING, "type"), PropertyInfo(Variant::STRING, "sdp")));
	ADD_SIGNAL(MethodInfo("ice_candidate_created", PropertyInfo(Variant::STRING, "media"), PropertyInfo(Variant::INT, "index"), PropertyInfo(Variant::STRING, "name")));
	ADD_SIGNAL(MethodInfo("data_channel_received", PropertyInfo(Variant::OBJECT, "channel", PROPERTY_HINT_RESOURCE_TYPE, "WebRTCDataChannel")));

	BIND_ENUM_CONSTANT(STATE_NEW);
	BIND_ENUM_CONSTANT(STATE_CONNECTING);
	BIND_ENUM_CONSTANT(STATE_CONNECTED);
	BIND_ENUM_CONSTANT(STATE_DISCONNECTED);
	BIND_ENUM_CONSTANT(STATE_FAILED);
	BIND_ENUM_CONSTANT(STATE_CLOSED);

	BIND_ENUM_CONSTANT(GATHERING_STATE_NEW);
	BIND_ENUM_CONSTANT(GATHERING_STATE_GATHERING);
	BIND_ENUM_CONSTANT(GATHERING_STATE_COMPLETE);

	BIND_ENUM_CONSTANT(SIGNALING_STATE_STABLE);
	BIND_ENUM_CONSTANT(SIGNALING_STATE_HAVE_LOCAL_OFFER);
	BIND_ENUM_CONSTANT(SIGNALING_STATE_HAVE_REMOTE_OFFER);
	BIND_ENUM_CONSTANT(SIGNALING_STATE_HAVE_LOCAL_PRANSWER);
	BIND_ENUM_CONSTANT(SIGNALING_STATE_HAVE_REMOTE_PRANSWER);
	BIND_ENUM_CONSTANT(SIGNALING_STATE_CLOSED);
}

WebRTCPeerConnection::WebRTCPeerConnection() {
}

WebRTCPeerConnection::~WebRTCPeerConnection() {
}
