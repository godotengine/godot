/*************************************************************************/
/*  webrtc_peer_js.cpp                                                   */
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

#ifdef JAVASCRIPT_ENABLED

#include "webrtc_peer_js.h"
#include "emscripten.h"

extern "C" {
EMSCRIPTEN_KEEPALIVE void _emrtc_on_ice_candidate(void *obj, char *p_MidName, int p_MlineIndexName, char *p_sdpName) {
	WebRTCPeerJS *peer = static_cast<WebRTCPeerJS *>(obj);
	peer->emit_signal("new_ice_candidate", String(p_MidName), p_MlineIndexName, String(p_sdpName));
}

EMSCRIPTEN_KEEPALIVE void _emrtc_offer_created(void *obj, char *p_type, char *p_offer) {
	WebRTCPeerJS *peer = static_cast<WebRTCPeerJS *>(obj);
	peer->emit_signal("offer_created", String(p_type), String(p_offer));
}

EMSCRIPTEN_KEEPALIVE void _emrtc_on_error(void *obj) {
	WebRTCPeerJS *peer = static_cast<WebRTCPeerJS *>(obj);
	peer->_on_error();
}

EMSCRIPTEN_KEEPALIVE void _emrtc_on_open(void *obj) {
	WebRTCPeerJS *peer = static_cast<WebRTCPeerJS *>(obj);
	peer->_on_open();
}

EMSCRIPTEN_KEEPALIVE void _emrtc_on_close(void *obj) {
	WebRTCPeerJS *peer = static_cast<WebRTCPeerJS *>(obj);
	peer->_on_close();
}

EMSCRIPTEN_KEEPALIVE void _emrtc_on_message(void *obj, uint8_t *p_data, uint32_t p_size, bool p_is_string) {
	WebRTCPeerJS *peer = static_cast<WebRTCPeerJS *>(obj);
	peer->_on_message(p_data, p_size, p_is_string);
}

EMSCRIPTEN_KEEPALIVE void _emrtc_bind_channel(int p_id) {
	/* clang-format off */
	EM_ASM({
		if (!Module.IDHandler.has($0)) {
			return; // Godot Object is gone!
		}
		var dict = Module.IDHandler.get($0);
		var channel = dict["channel"];
		var c_ptr = dict["ptr"];

		channel.onopen = function (evt) {
			ccall("_emrtc_on_open",
				"void",
				["number"],
				[c_ptr]
			);
		};
		channel.onclose = function (evt) {
			ccall("_emrtc_on_close",
				"void",
				["number"],
				[c_ptr]
			);
		};
		channel.onerror = function (evt) {
			ccall("_emrtc_on_error",
				"void",
				["number"],
				[c_ptr]
			);
		};

		channel.binaryType = "arraybuffer";
		channel.onmessage = function(event) {
			var buffer;
			var is_string = 0;
			if (event.data instanceof ArrayBuffer) {

				buffer = new Uint8Array(event.data);

			} else if (event.data instanceof Blob) {

				alert("Blob type not supported");
				return;

			} else if (typeof event.data === "string") {

				is_string = 1;
				var enc = new TextEncoder("utf-8");
				buffer = new Uint8Array(enc.encode(event.data));

			} else {

				alert("Unknown message type");
				return;

			}
			var len = buffer.length*buffer.BYTES_PER_ELEMENT;
			var out = Module._malloc(len);
			Module.HEAPU8.set(buffer, out);
			ccall("_emrtc_on_message",
				"void",
				["number", "number", "number", "number"],
				[c_ptr, out, len, is_string]
			);
			Module._free(out);
		}
	}, p_id);
	/* clang-format on */
}
}

void _emrtc_create_pc(int p_id) {
	/* clang-format off */
	EM_ASM({
		var dict = Module.IDHandler.get($0);
		var c_ptr = dict["ptr"];
		// Setup local connaction
		var conn = new RTCPeerConnection();
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
			if (!dict || dict["channel"]) {
				return;
			}
			var channel = evt.channel;
			dict["channel"] = channel;
			ccall("_emrtc_bind_channel",
				"void",
				["number"],
				[$0]
			);
		};
		dict["conn"] = conn;
	}, p_id);
	/* clang-format on */
}

void WebRTCPeerJS::_on_open() {
	in_buffer.resize(16);
	_conn_state = STATE_CONNECTED;
}

void WebRTCPeerJS::_on_close() {
	close();
}

void WebRTCPeerJS::_on_error() {
	close();
	_conn_state = STATE_FAILED;
}

void WebRTCPeerJS::_on_message(uint8_t *p_data, uint32_t p_size, bool p_is_string) {
	if (in_buffer.space_left() < p_size + 5) {
		ERR_EXPLAIN("Buffer full! Dropping data");
		ERR_FAIL();
	}

	uint8_t is_string = p_is_string ? 1 : 0;
	in_buffer.write((uint8_t *)&p_size, 4);
	in_buffer.write((uint8_t *)&is_string, 1);
	in_buffer.write(p_data, p_size);
	queue_count++;
}

void WebRTCPeerJS::close() {
	in_buffer.resize(0);
	queue_count = 0;
	_was_string = false;
	/* clang-format off */
	EM_ASM({
		var dict = Module.IDHandler.get($0);
		if (!dict) return;
		if (dict["channel"]) {
			dict["channel"].close();
			dict["channel"] = null;
		}
		if (dict["conn"]) {
			dict["conn"].close();
		}
	}, _js_id);
	/* clang-format on */
	_conn_state = STATE_CLOSED;
}

Error WebRTCPeerJS::create_offer() {
	ERR_FAIL_COND_V(_conn_state != STATE_NEW, FAILED);

	_conn_state = STATE_CONNECTING;
	/* clang-format off */
	EM_ASM({
		var dict = Module.IDHandler.get($0);
		var conn = dict["conn"];
		var c_ptr = dict["ptr"];
		var onError = function(error) {
			console.log(error);
			ccall("_emrtc_on_error",
				"void",
				["number"],
				[c_ptr]
			);
		};
		var onCreated = function(offer) {
			ccall("_emrtc_offer_created",
				"void",
				["number", "string", "string"],
				[c_ptr, offer.type, offer.sdp]
			);
		};

		var channel = conn.createDataChannel("default");
		dict["channel"] = channel;
		ccall("_emrtc_bind_channel",
			"void",
			["number"],
			[$0]
		);
		conn.createOffer().then(onCreated).catch(onError);
	}, _js_id);
	/* clang-format on */
	return OK;
}

Error WebRTCPeerJS::set_local_description(String type, String sdp) {
	/* clang-format off */
	EM_ASM({
		var dict = Module.IDHandler.get($0);
		var conn = dict["conn"];
		var c_ptr = dict["ptr"];
		var type = UTF8ToString($1);
		var sdp = UTF8ToString($2);
		var onError = function(error) {
			console.log(error);
			ccall("_emrtc_on_error",
				"void",
				["number"],
				[c_ptr]
			);
		};
		conn.setLocalDescription({
			"sdp": sdp,
			"type": type
		}).catch(onError);
	}, _js_id, type.utf8().get_data(), sdp.utf8().get_data());
	/* clang-format on */
	return OK;
}

Error WebRTCPeerJS::set_remote_description(String type, String sdp) {
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
			console.log(error);
			ccall("_emrtc_on_error",
				"void",
				["number"],
				[c_ptr]
			);
		};
		var onCreated = function(offer) {
			ccall("_emrtc_offer_created",
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

Error WebRTCPeerJS::add_ice_candidate(String sdpMidName, int sdpMlineIndexName, String sdpName) {
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

Error WebRTCPeerJS::poll() {
	return OK;
}

WebRTCPeer::ConnectionState WebRTCPeerJS::get_connection_state() const {
	return _conn_state;
}

int WebRTCPeerJS::get_available_packet_count() const {
	return queue_count;
}

Error WebRTCPeerJS::get_packet(const uint8_t **r_buffer, int &r_buffer_size) {
	ERR_FAIL_COND_V(_conn_state != STATE_CONNECTED, ERR_UNCONFIGURED);

	if (queue_count == 0)
		return ERR_UNAVAILABLE;

	uint32_t to_read = 0;
	uint32_t left = 0;
	uint8_t is_string = 0;
	r_buffer_size = 0;

	in_buffer.read((uint8_t *)&to_read, 4);
	--queue_count;
	left = in_buffer.data_left();

	if (left < to_read + 1) {
		in_buffer.advance_read(left);
		return FAILED;
	}

	in_buffer.read(&is_string, 1);
	_was_string = is_string == 1;
	in_buffer.read(packet_buffer, to_read);
	*r_buffer = packet_buffer;
	r_buffer_size = to_read;

	return OK;
}

Error WebRTCPeerJS::put_packet(const uint8_t *p_buffer, int p_buffer_size) {
	ERR_FAIL_COND_V(_conn_state != STATE_CONNECTED, ERR_UNCONFIGURED);

	int is_bin = _write_mode == WebRTCPeer::WRITE_MODE_BINARY ? 1 : 0;

	/* clang-format off */
	EM_ASM({
		var dict = Module.IDHandler.get($0);
		var channel = dict["channel"];
		var bytes_array = new Uint8Array($2);
		var i = 0;

		for(i=0; i<$2; i++) {
			bytes_array[i] = getValue($1+i, 'i8');
		}

		if ($3) {
			channel.send(bytes_array.buffer);
		} else {
			var string = new TextDecoder("utf-8").decode(bytes_array);
			channel.send(string);
		}
	}, _js_id, p_buffer, p_buffer_size, is_bin);
	/* clang-format on */

	return OK;
}

int WebRTCPeerJS::get_max_packet_size() const {
	return 1200;
}

void WebRTCPeerJS::set_write_mode(WriteMode p_mode) {
	_write_mode = p_mode;
}

WebRTCPeer::WriteMode WebRTCPeerJS::get_write_mode() const {
	return _write_mode;
}

bool WebRTCPeerJS::was_string_packet() const {
	return _was_string;
}

WebRTCPeerJS::WebRTCPeerJS() {
	queue_count = 0;
	_was_string = false;
	_write_mode = WRITE_MODE_BINARY;
	_conn_state = STATE_NEW;

	/* clang-format off */
	_js_id = EM_ASM_INT({
		return Module.IDHandler.add({"conn": null, "ptr": $0, "channel": null});
	}, this);
	/* clang-format on */
	_emrtc_create_pc(_js_id);
}

WebRTCPeerJS::~WebRTCPeerJS() {
	close();
	/* clang-format off */
	EM_ASM({
		Module.IDHandler.remove($0);
	}, _js_id);
	/* clang-format on */
};
#endif
