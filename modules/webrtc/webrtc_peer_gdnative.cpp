/*************************************************************************/
/*  webrtc_peer_gdnative.cpp                                             */
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

#ifdef WEBRTC_GDNATIVE_ENABLED

#include "webrtc_peer_gdnative.h"

void WebRTCPeerGDNative::_bind_methods() {
}

WebRTCPeerGDNative::WebRTCPeerGDNative() {
	interface = NULL;
}

WebRTCPeerGDNative::~WebRTCPeerGDNative() {
}

Error WebRTCPeerGDNative::create_offer() {
	ERR_FAIL_COND_V(interface == NULL, ERR_UNCONFIGURED);
	return (Error)interface->create_offer(interface->data);
}

Error WebRTCPeerGDNative::set_local_description(String p_type, String p_sdp) {
	ERR_FAIL_COND_V(interface == NULL, ERR_UNCONFIGURED);
	return (Error)interface->set_local_description(interface->data, p_type.utf8().get_data(), p_sdp.utf8().get_data());
}

Error WebRTCPeerGDNative::set_remote_description(String p_type, String p_sdp) {
	ERR_FAIL_COND_V(interface == NULL, ERR_UNCONFIGURED);
	return (Error)interface->set_remote_description(interface->data, p_type.utf8().get_data(), p_sdp.utf8().get_data());
}

Error WebRTCPeerGDNative::add_ice_candidate(String sdpMidName, int sdpMlineIndexName, String sdpName) {
	ERR_FAIL_COND_V(interface == NULL, ERR_UNCONFIGURED);
	return (Error)interface->add_ice_candidate(interface->data, sdpMidName.utf8().get_data(), sdpMlineIndexName, sdpName.utf8().get_data());
}

Error WebRTCPeerGDNative::poll() {
	ERR_FAIL_COND_V(interface == NULL, ERR_UNCONFIGURED);
	return (Error)interface->poll(interface->data);
}

void WebRTCPeerGDNative::set_write_mode(WriteMode p_mode) {
	ERR_FAIL_COND(interface == NULL);
	interface->set_write_mode(interface->data, p_mode);
}

WebRTCPeer::WriteMode WebRTCPeerGDNative::get_write_mode() const {
	ERR_FAIL_COND_V(interface == NULL, WRITE_MODE_BINARY);
	return (WriteMode)interface->get_write_mode(interface->data);
}

bool WebRTCPeerGDNative::was_string_packet() const {
	ERR_FAIL_COND_V(interface == NULL, false);
	return interface->was_string_packet(interface->data);
}

WebRTCPeer::ConnectionState WebRTCPeerGDNative::get_connection_state() const {
	ERR_FAIL_COND_V(interface == NULL, STATE_DISCONNECTED);
	return STATE_DISCONNECTED;
}

Error WebRTCPeerGDNative::get_packet(const uint8_t **r_buffer, int &r_buffer_size) {
	ERR_FAIL_COND_V(interface == NULL, ERR_UNCONFIGURED);
	return (Error)interface->get_packet(interface->data, r_buffer, &r_buffer_size);
}

Error WebRTCPeerGDNative::put_packet(const uint8_t *p_buffer, int p_buffer_size) {
	ERR_FAIL_COND_V(interface == NULL, ERR_UNCONFIGURED);
	return (Error)interface->put_packet(interface->data, p_buffer, p_buffer_size);
}

int WebRTCPeerGDNative::get_max_packet_size() const {
	ERR_FAIL_COND_V(interface == NULL, 0);
	return interface->get_max_packet_size(interface->data);
}

int WebRTCPeerGDNative::get_available_packet_count() const {
	ERR_FAIL_COND_V(interface == NULL, 0);
	return interface->get_available_packet_count(interface->data);
}

void WebRTCPeerGDNative::set_native_webrtc_peer(const godot_net_webrtc_peer *p_impl) {
	interface = p_impl;
}

#endif // WEBRTC_GDNATIVE_ENABLED
