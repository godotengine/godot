/**************************************************************************/
/*  web_rtc_peer_connection_extension.cpp                                 */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#include <godot_cpp/classes/web_rtc_peer_connection_extension.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/web_rtc_data_channel.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>

namespace godot {

WebRTCPeerConnection::ConnectionState WebRTCPeerConnectionExtension::_get_connection_state() const {
	return WebRTCPeerConnection::ConnectionState(0);
}

WebRTCPeerConnection::GatheringState WebRTCPeerConnectionExtension::_get_gathering_state() const {
	return WebRTCPeerConnection::GatheringState(0);
}

WebRTCPeerConnection::SignalingState WebRTCPeerConnectionExtension::_get_signaling_state() const {
	return WebRTCPeerConnection::SignalingState(0);
}

Error WebRTCPeerConnectionExtension::_initialize(const Dictionary &p_config) {
	return Error(0);
}

Ref<WebRTCDataChannel> WebRTCPeerConnectionExtension::_create_data_channel(const String &p_label, const Dictionary &p_config) {
	return Ref<WebRTCDataChannel>();
}

Error WebRTCPeerConnectionExtension::_create_offer() {
	return Error(0);
}

Error WebRTCPeerConnectionExtension::_set_remote_description(const String &p_type, const String &p_sdp) {
	return Error(0);
}

Error WebRTCPeerConnectionExtension::_set_local_description(const String &p_type, const String &p_sdp) {
	return Error(0);
}

Error WebRTCPeerConnectionExtension::_add_ice_candidate(const String &p_sdp_mid_name, int32_t p_sdp_mline_index, const String &p_sdp_name) {
	return Error(0);
}

Error WebRTCPeerConnectionExtension::_poll() {
	return Error(0);
}

void WebRTCPeerConnectionExtension::_close() {}

} // namespace godot
