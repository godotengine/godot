/*************************************************************************/
/*  webrtc_peer_connection_js.cpp                                        */
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

#ifdef JAVASCRIPT_ENABLED

#include "webrtc_peer_connection_js.h"

#include "webrtc_data_channel_js.h"

#include "core/io/json.h"
#include "emscripten.h"

extern "C" {
EMSCRIPTEN_KEEPALIVE void _emrtc_on_ice_candidate(void *obj, char *p_MidName, int p_MlineIndexName, char *p_sdpName) {
	WebRTCPeerConnectionJS *peer = static_cast<WebRTCPeerConnectionJS *>(obj);
	peer->emit_signal("ice_candidate_created", String(p_MidName), p_MlineIndexName, String(p_sdpName));
}

EMSCRIPTEN_KEEPALIVE void _emrtc_session_description_created(void *obj, char *p_type, char *p_offer) {
	WebRTCPeerConnectionJS *peer = static_cast<WebRTCPeerConnectionJS *>(obj);
	peer->emit_signal("session_description_created", String(p_type), String(p_offer));
}

EMSCRIPTEN_KEEPALIVE void _emrtc_on_connection_state_changed(void *obj) {
	WebRTCPeerConnectionJS *peer = static_cast<WebRTCPeerConnectionJS *>(obj);
	peer->_on_connection_state_changed();
}

EMSCRIPTEN_KEEPALIVE void _emrtc_on_error() {
	ERR_PRINT("RTCPeerConnection error!");
}

EMSCRIPTEN_KEEPALIVE void _emrtc_emit_channel(void *obj, int p_id) {
	WebRTCPeerConnectionJS *peer = static_cast<WebRTCPeerConnectionJS *>(obj);
	peer->emit_signal("data_channel_received", Ref<WebRTCDataChannelJS>(new WebRTCDataChannelJS(p_id)));
}
}

void _emrtc_create_pc(int p_id, const Dictionary &p_config) {
	String config = JSON::print(p_config);
	/* clang-format off */
	EM_ASM({
		var dict = Module.IDHandler.get($0);
		var c_ptr = dict["ptr"];
		var config = JSON.parse(UTF8ToString($1));
		// Setup local connaction
		var conn = null;
		try {
			conn = new RTCPeerConnection(config);
		} catch (e) {
			console.log(e);
			return;
		}
		conn.oniceconnectionstatechange = function(event) {
			if (!Module.IDHandler.get($0)) return;
			ccall("_emrtc_on_connection_state_changed", "void", ["number"], [c_ptr]);
		};
		conn.onicecandidate = function(event) {
			if (!Module.IDHandler.get($0)) return;
			if (!event.candidate) return;

			var c = event.candidate;
			// should emit on ice candidate
			ccall("_emrtc_on_ice_candidate",
				"void",
				["number", "string", "number", "string"],
				[c_ptr, c.sdpMid, c.sdpMLineIndex, c.candidate]
			);
		};
		conn.ondatachannel = function (evt) {
			var dict = Module.IDHandler.get($0);
			if (!dict) {
				return;
			}
			var id = Module.IDHandler.add({"channel": evt.channel, "ptr": null});
			ccall("_emrtc_emit_channel",
				"void",
				["number", "number"],
				[c_ptr, id]
			);
		};
		dict["conn"] = conn;
	}, p_id, config.utf8().get_data());
	/* clang-format on */
}

void WebRTCPeerConnectionJS::_on_connection_state_changed() {
	/* clang-format off */
	_conn_state = (ConnectionState)EM_ASM_INT({
		var dict = Module.IDHandler.get($0);
		if (!dict) return 5; // CLOSED
		var conn = dict["conn"];
		switch(conn.iceConnectionState) {
			case "new":
				return 0;
			case "checking":
				return 1;
			case "connected":
			case "completed":
				return 2;
			case "disconnected":
				return 3;
			case "failed":
				return 4;
			case "closed":
				return 5;
		}
		return 5; // CLOSED
	}, _js_id);
	/* clang-format on */
}

void WebRTCPeerConnectionJS::close() {
	/* clang-format off */
	EM_ASM({
		var dict = Module.IDHandler.get($0);
		if (!dict) return;
		if (dict["conn"]) {
			dict["conn"].close();
		}
	}, _js_id);
	/* clang-format on */
	_conn_state = STATE_CLOSED;
}

Error WebRTCPeerConnectionJS::create_offer() {
	ERR_FAIL_COND_V(_conn_state != STATE_NEW, FAILED);

	_conn_state = STATE_CONNECTING;
	/* clang-format off */
	EM_ASM({
		var dict = Module.IDHandler.get($0);
		var conn = dict["conn"];
		var c_ptr = dict["ptr"];
		var onError = function(error) {
			console.error(error);
			ccall("_emrtc_on_error", "void", [], []);
		};
		var onCreated = function(offer) {
			ccall("_emrtc_session_description_created",
				"void",
				["number", "string", "string"],
				[c_ptr, offer.type, offer.sdp]
			);
		};
		conn.createOffer().then(onCreated).catch(onError);
	}, _js_id);
	/* clang-format on */
	return OK;
}

Error WebRTCPeerConnectionJS::set_local_description(String type, String sdp) {
	/* clang-format off */
	EM_ASM({
		var dict = Module.IDHandler.get($0);
		var conn = dict["conn"];
		var c_ptr = dict["ptr"];
		var type = UTF8ToString($1);
		var sdp = UTF8ToString($2);
		var onError = function(error) {
			console.error(error);
			ccall("_emrtc_on_error", "void", [], []);
		};
		conn.setLocalDescription({
			"sdp": sdp,
			"type": type
		}).catch(onError);
	}, _js_id, type.utf8().get_data(), sdp.utf8().get_data());
	/* clang-format on */
	return OK;
}

Error WebRTCPeerConnectionJS::set_remote_description(String type, String sdp) {
	if (type == "offer") {
		ERR_FAIL_COND_V(_conn_state != STATE_NEW, FAILED);
		_conn_state = STATE_CONNECTING;
	}
	/* clang-format off */
	EM_ASM({
		var dict = Module.IDHandler.get($0);
		var conn = dict["conn"];
		var c_ptr = dict["ptr"];
		var type = UTF8ToString($1);
		var sdp = UTF8ToString($2);

		var onError = function(error) {
			console.error(error);
			ccall("_emrtc_on_error", "void", [], []);
		};
		var onCreated = function(offer) {
			ccall("_emrtc_session_description_created",
				"void",
				["number", "string", "string"],
				[c_ptr, offer.type, offer.sdp]
			);
		};
		var onSet = function() {
			if (type != "offer") {
				return;
			}
			conn.createAnswer().then(onCreated);
		};
		conn.setRemoteDescription({
			"sdp": sdp,
			"type": type
		}).then(onSet).catch(onError);
	}, _js_id, type.utf8().get_data(), sdp.utf8().get_data());
	/* clang-format on */
	return OK;
}

Error WebRTCPeerConnectionJS::add_ice_candidate(String sdpMidName, int sdpMlineIndexName, String sdpName) {
	/* clang-format off */
	EM_ASM({
		var dict = Module.IDHandler.get($0);
		var conn = dict["conn"];
		var c_ptr = dict["ptr"];
		var sdpMidName = UTF8ToString($1);
		var sdpMlineIndexName = UTF8ToString($2);
		var sdpName = UTF8ToString($3);
		conn.addIceCandidate(new RTCIceCandidate({
			"candidate": sdpName,
			"sdpMid": sdpMidName,
			"sdpMlineIndex": sdpMlineIndexName
		}));
	}, _js_id, sdpMidName.utf8().get_data(), sdpMlineIndexName, sdpName.utf8().get_data());
	/* clang-format on */
	return OK;
}

Error WebRTCPeerConnectionJS::initialize(Dictionary p_config) {
	_emrtc_create_pc(_js_id, p_config);
	return OK;
}

Ref<WebRTCDataChannel> WebRTCPeerConnectionJS::create_data_channel(String p_channel, Dictionary p_channel_config) {
	String config = JSON::print(p_channel_config);
	/* clang-format off */
	int id = EM_ASM_INT({
		try {
			var dict = Module.IDHandler.get($0);
			if (!dict) return 0;
			var label = UTF8ToString($1);
			var config = JSON.parse(UTF8ToString($2));
			var conn = dict["conn"];
			return Module.IDHandler.add({
				"channel": conn.createDataChannel(label, config),
				"ptr": null
			})
		} catch (e) {
			return 0;
		}
	}, _js_id, p_channel.utf8().get_data(), config.utf8().get_data());
	/* clang-format on */
	ERR_FAIL_COND_V(id == 0, NULL);
	return memnew(WebRTCDataChannelJS(id));
}

Error WebRTCPeerConnectionJS::poll() {
	return OK;
}

WebRTCPeerConnection::ConnectionState WebRTCPeerConnectionJS::get_connection_state() const {
	return _conn_state;
}

WebRTCPeerConnectionJS::WebRTCPeerConnectionJS() {
	_conn_state = STATE_NEW;

	/* clang-format off */
	_js_id = EM_ASM_INT({
		return Module.IDHandler.add({"conn": null, "ptr": $0});
	}, this);
	/* clang-format on */
	Dictionary config;
	_emrtc_create_pc(_js_id, config);
}

WebRTCPeerConnectionJS::~WebRTCPeerConnectionJS() {
	close();
	/* clang-format off */
	EM_ASM({
		Module.IDHandler.remove($0);
	}, _js_id);
	/* clang-format on */
};
#endif
