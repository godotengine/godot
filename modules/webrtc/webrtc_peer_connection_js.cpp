/**************************************************************************/
/*  webrtc_peer_connection_js.cpp                                         */
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

#include "webrtc_peer_connection_js.h"

#ifdef WEB_ENABLED

#include "webrtc_data_channel_js.h"

#include <emscripten.h>

void WebRTCPeerConnectionJS::_on_ice_candidate(void *p_obj, const char *p_mid_name, int p_mline_idx, const char *p_candidate) {
	WebRTCPeerConnectionJS *peer = static_cast<WebRTCPeerConnectionJS *>(p_obj);
	peer->emit_signal(SNAME("ice_candidate_created"), String(p_mid_name), p_mline_idx, String(p_candidate));
}

void WebRTCPeerConnectionJS::_on_session_created(void *p_obj, const char *p_type, const char *p_session) {
	WebRTCPeerConnectionJS *peer = static_cast<WebRTCPeerConnectionJS *>(p_obj);
	peer->emit_signal(SNAME("session_description_created"), String(p_type), String(p_session));
}

void WebRTCPeerConnectionJS::_on_connection_state_changed(void *p_obj, int p_state) {
	WebRTCPeerConnectionJS *peer = static_cast<WebRTCPeerConnectionJS *>(p_obj);
	peer->_conn_state = (ConnectionState)p_state;
}

void WebRTCPeerConnectionJS::_on_gathering_state_changed(void *p_obj, int p_state) {
	WebRTCPeerConnectionJS *peer = static_cast<WebRTCPeerConnectionJS *>(p_obj);
	peer->_gathering_state = (GatheringState)p_state;
}

void WebRTCPeerConnectionJS::_on_signaling_state_changed(void *p_obj, int p_state) {
	WebRTCPeerConnectionJS *peer = static_cast<WebRTCPeerConnectionJS *>(p_obj);
	peer->_signaling_state = (SignalingState)p_state;
}

void WebRTCPeerConnectionJS::_on_error(void *p_obj) {
	ERR_PRINT("RTCPeerConnection error!");
}

void WebRTCPeerConnectionJS::_on_data_channel(void *p_obj, int p_id) {
	WebRTCPeerConnectionJS *peer = static_cast<WebRTCPeerConnectionJS *>(p_obj);
	peer->emit_signal(SNAME("data_channel_received"), Ref<WebRTCDataChannel>(memnew(WebRTCDataChannelJS(p_id))));
}

void WebRTCPeerConnectionJS::close() {
	godot_js_rtc_pc_close(_js_id);
	_conn_state = STATE_CLOSED;
}

Error WebRTCPeerConnectionJS::create_offer() {
	ERR_FAIL_COND_V(_conn_state != STATE_NEW, FAILED);

	_conn_state = STATE_CONNECTING;
	godot_js_rtc_pc_offer_create(_js_id, this, &_on_session_created, &_on_error);
	return OK;
}

Error WebRTCPeerConnectionJS::set_local_description(String type, String sdp) {
	godot_js_rtc_pc_local_description_set(_js_id, type.utf8().get_data(), sdp.utf8().get_data(), this, &_on_error);
	return OK;
}

Error WebRTCPeerConnectionJS::set_remote_description(String type, String sdp) {
	if (type == "offer") {
		ERR_FAIL_COND_V(_conn_state != STATE_NEW, FAILED);
		_conn_state = STATE_CONNECTING;
	}
	godot_js_rtc_pc_remote_description_set(_js_id, type.utf8().get_data(), sdp.utf8().get_data(), this, &_on_session_created, &_on_error);
	return OK;
}

Error WebRTCPeerConnectionJS::add_ice_candidate(String sdpMidName, int sdpMlineIndexName, String sdpName) {
	godot_js_rtc_pc_ice_candidate_add(_js_id, sdpMidName.utf8().get_data(), sdpMlineIndexName, sdpName.utf8().get_data());
	return OK;
}

Error WebRTCPeerConnectionJS::initialize(Dictionary p_config) {
	if (_js_id) {
		godot_js_rtc_pc_destroy(_js_id);
		_js_id = 0;
	}
	_conn_state = STATE_NEW;

	String config = Variant(p_config).to_json_string();
	_js_id = godot_js_rtc_pc_create(config.utf8().get_data(), this, &_on_connection_state_changed, &_on_gathering_state_changed, &_on_signaling_state_changed, &_on_ice_candidate, &_on_data_channel);
	return _js_id ? OK : FAILED;
}

Ref<WebRTCDataChannel> WebRTCPeerConnectionJS::create_data_channel(String p_channel, Dictionary p_channel_config) {
	ERR_FAIL_COND_V(_conn_state != STATE_NEW, nullptr);

	String config = Variant(p_channel_config).to_json_string();
	int id = godot_js_rtc_pc_datachannel_create(_js_id, p_channel.utf8().get_data(), config.utf8().get_data());
	ERR_FAIL_COND_V(id == 0, nullptr);
	return memnew(WebRTCDataChannelJS(id));
}

Error WebRTCPeerConnectionJS::poll() {
	return OK;
}

WebRTCPeerConnection::GatheringState WebRTCPeerConnectionJS::get_gathering_state() const {
	return _gathering_state;
}

WebRTCPeerConnection::SignalingState WebRTCPeerConnectionJS::get_signaling_state() const {
	return _signaling_state;
}

WebRTCPeerConnection::ConnectionState WebRTCPeerConnectionJS::get_connection_state() const {
	return _conn_state;
}

WebRTCPeerConnectionJS::WebRTCPeerConnectionJS() {
	Dictionary config;
	initialize(config);
}

WebRTCPeerConnectionJS::~WebRTCPeerConnectionJS() {
	close();
	if (_js_id) {
		godot_js_rtc_pc_destroy(_js_id);
		_js_id = 0;
	}
}
#endif
