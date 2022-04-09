/*************************************************************************/
/*  webrtc_peer_connection_extension.cpp                                 */
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

#include "webrtc_peer_connection_extension.h"

void WebRTCPeerConnectionExtension::_bind_methods() {
	ClassDB::bind_method(D_METHOD("make_default"), &WebRTCPeerConnectionExtension::make_default);

	GDVIRTUAL_BIND(_get_connection_state);
	GDVIRTUAL_BIND(_initialize, "p_config");
	GDVIRTUAL_BIND(_create_data_channel, "p_label", "p_config");
	GDVIRTUAL_BIND(_create_offer);
	GDVIRTUAL_BIND(_set_remote_description, "p_type", "p_sdp");
	GDVIRTUAL_BIND(_set_local_description, "p_type", "p_sdp");
	GDVIRTUAL_BIND(_add_ice_candidate, "p_sdp_mid_name", "p_sdp_mline_index", "p_sdp_name");
	GDVIRTUAL_BIND(_poll);
	GDVIRTUAL_BIND(_close);
}

void WebRTCPeerConnectionExtension::make_default() {
	ERR_FAIL_COND_MSG(!_get_extension(), vformat("Can't make %s the default without extending it.", get_class()));
	WebRTCPeerConnection::set_default_extension(get_class());
}

WebRTCPeerConnection::ConnectionState WebRTCPeerConnectionExtension::get_connection_state() const {
	int state;
	if (GDVIRTUAL_CALL(_get_connection_state, state)) {
		return (ConnectionState)state;
	}
	WARN_PRINT_ONCE("WebRTCPeerConnectionExtension::_get_connection_state is unimplemented!");
	return STATE_DISCONNECTED;
}

Error WebRTCPeerConnectionExtension::initialize(Dictionary p_config) {
	int err;
	if (GDVIRTUAL_CALL(_initialize, p_config, err)) {
		return (Error)err;
	}
	WARN_PRINT_ONCE("WebRTCPeerConnectionExtension::_initialize is unimplemented!");
	return ERR_UNCONFIGURED;
}

Ref<WebRTCDataChannel> WebRTCPeerConnectionExtension::create_data_channel(String p_label, Dictionary p_options) {
	Object *ret = nullptr;
	if (GDVIRTUAL_CALL(_create_data_channel, p_label, p_options, ret)) {
		WebRTCDataChannel *ch = Object::cast_to<WebRTCDataChannel>(ret);
		ERR_FAIL_COND_V_MSG(ret && !ch, nullptr, "Returned object must be an instance of WebRTCDataChannel.");
		return ch;
	}
	WARN_PRINT_ONCE("WebRTCPeerConnectionExtension::_create_data_channel is unimplemented!");
	return nullptr;
}

Error WebRTCPeerConnectionExtension::create_offer() {
	int err;
	if (GDVIRTUAL_CALL(_create_offer, err)) {
		return (Error)err;
	}
	WARN_PRINT_ONCE("WebRTCPeerConnectionExtension::_create_offer is unimplemented!");
	return ERR_UNCONFIGURED;
}

Error WebRTCPeerConnectionExtension::set_local_description(String p_type, String p_sdp) {
	int err;
	if (GDVIRTUAL_CALL(_set_local_description, p_type, p_sdp, err)) {
		return (Error)err;
	}
	WARN_PRINT_ONCE("WebRTCPeerConnectionExtension::_set_local_description is unimplemented!");
	return ERR_UNCONFIGURED;
}

Error WebRTCPeerConnectionExtension::set_remote_description(String p_type, String p_sdp) {
	int err;
	if (GDVIRTUAL_CALL(_set_remote_description, p_type, p_sdp, err)) {
		return (Error)err;
	}
	WARN_PRINT_ONCE("WebRTCPeerConnectionExtension::_set_remote_description is unimplemented!");
	return ERR_UNCONFIGURED;
}

Error WebRTCPeerConnectionExtension::add_ice_candidate(String p_sdp_mid_name, int p_sdp_mline_index, String p_sdp_name) {
	int err;
	if (GDVIRTUAL_CALL(_add_ice_candidate, p_sdp_mid_name, p_sdp_mline_index, p_sdp_name, err)) {
		return (Error)err;
	}
	WARN_PRINT_ONCE("WebRTCPeerConnectionExtension::_add_ice_candidate is unimplemented!");
	return ERR_UNCONFIGURED;
}

Error WebRTCPeerConnectionExtension::poll() {
	int err;
	if (GDVIRTUAL_CALL(_poll, err)) {
		return (Error)err;
	}
	WARN_PRINT_ONCE("WebRTCPeerConnectionExtension::_poll is unimplemented!");
	return ERR_UNCONFIGURED;
}

void WebRTCPeerConnectionExtension::close() {
	if (GDVIRTUAL_CALL(_close)) {
		return;
	}
	WARN_PRINT_ONCE("WebRTCPeerConnectionExtension::_close is unimplemented!");
}
