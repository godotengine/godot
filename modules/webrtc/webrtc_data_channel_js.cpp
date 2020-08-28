/*************************************************************************/
/*  webrtc_data_channel_js.cpp                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "webrtc_data_channel_js.h"
#include "emscripten.h"

extern "C" {
EMSCRIPTEN_KEEPALIVE void _emrtc_on_ch_error(void *obj) {
	WebRTCDataChannelJS *peer = static_cast<WebRTCDataChannelJS *>(obj);
	peer->_on_error();
}

EMSCRIPTEN_KEEPALIVE void _emrtc_on_ch_open(void *obj) {
	WebRTCDataChannelJS *peer = static_cast<WebRTCDataChannelJS *>(obj);
	peer->_on_open();
}

EMSCRIPTEN_KEEPALIVE void _emrtc_on_ch_close(void *obj) {
	WebRTCDataChannelJS *peer = static_cast<WebRTCDataChannelJS *>(obj);
	peer->_on_close();
}

EMSCRIPTEN_KEEPALIVE void _emrtc_on_ch_message(void *obj, uint8_t *p_data, uint32_t p_size, bool p_is_string) {
	WebRTCDataChannelJS *peer = static_cast<WebRTCDataChannelJS *>(obj);
	peer->_on_message(p_data, p_size, p_is_string);
}
}

void WebRTCDataChannelJS::_on_open() {
	in_buffer.resize(_in_buffer_shift);
}

void WebRTCDataChannelJS::_on_close() {
	close();
}

void WebRTCDataChannelJS::_on_error() {
	close();
}

void WebRTCDataChannelJS::_on_message(uint8_t *p_data, uint32_t p_size, bool p_is_string) {
	ERR_FAIL_COND_MSG(in_buffer.space_left() < (int)(p_size + 5), "Buffer full! Dropping data.");

	uint8_t is_string = p_is_string ? 1 : 0;
	in_buffer.write((uint8_t *)&p_size, 4);
	in_buffer.write((uint8_t *)&is_string, 1);
	in_buffer.write(p_data, p_size);
	queue_count++;
}

void WebRTCDataChannelJS::close() {
	in_buffer.resize(0);
	queue_count = 0;
	_was_string = false;
	/* clang-format off */
	EM_ASM({
		var dict = Module.IDHandler.get($0);
		if (!dict) return;
		var channel = dict["channel"];
		channel.onopen = null;
		channel.onclose = null;
		channel.onerror = null;
		channel.onmessage = null;
		channel.close();
	}, _js_id);
	/* clang-format on */
}

Error WebRTCDataChannelJS::poll() {
	return OK;
}

WebRTCDataChannelJS::ChannelState WebRTCDataChannelJS::get_ready_state() const {
	/* clang-format off */
	return (ChannelState) EM_ASM_INT({
		var dict = Module.IDHandler.get($0);
		if (!dict) return 3; // CLOSED
		var channel = dict["channel"];
		switch(channel.readyState) {
			case "connecting":
				return 0;
			case "open":
				return 1;
			case "closing":
				return 2;
			case "closed":
				return 3;
		}
		return 3; // CLOSED
	}, _js_id);
	/* clang-format on */
}

int WebRTCDataChannelJS::get_available_packet_count() const {
	return queue_count;
}

Error WebRTCDataChannelJS::get_packet(const uint8_t **r_buffer, int &r_buffer_size) {
	ERR_FAIL_COND_V(get_ready_state() != STATE_OPEN, ERR_UNCONFIGURED);

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

Error WebRTCDataChannelJS::put_packet(const uint8_t *p_buffer, int p_buffer_size) {
	ERR_FAIL_COND_V(get_ready_state() != STATE_OPEN, ERR_UNCONFIGURED);

	int is_bin = _write_mode == WebRTCDataChannel::WRITE_MODE_BINARY ? 1 : 0;

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

int WebRTCDataChannelJS::get_max_packet_size() const {
	return 1200;
}

void WebRTCDataChannelJS::set_write_mode(WriteMode p_mode) {
	_write_mode = p_mode;
}

WebRTCDataChannel::WriteMode WebRTCDataChannelJS::get_write_mode() const {
	return _write_mode;
}

bool WebRTCDataChannelJS::was_string_packet() const {
	return _was_string;
}

String WebRTCDataChannelJS::get_label() const {
	return _label;
}

/* clang-format off */
#define _JS_GET(PROP, DEF)			\
EM_ASM_INT({					\
	var dict = Module.IDHandler.get($0);	\
	if (!dict || !dict["channel"]) {	\
		return DEF;			\
	}					\
	var out = dict["channel"].PROP;		\
	return out === null ? DEF : out;	\
}, _js_id)
/* clang-format on */

bool WebRTCDataChannelJS::is_ordered() const {
	return _JS_GET(ordered, true);
}

int WebRTCDataChannelJS::get_id() const {
	return _JS_GET(id, 65535);
}

int WebRTCDataChannelJS::get_max_packet_life_time() const {
	// Can't use macro, webkit workaround.
	/* clang-format off */
	return EM_ASM_INT({
		var dict = Module.IDHandler.get($0);
		if (!dict || !dict["channel"]) {
			return 65535;
		}
		if (dict["channel"].maxRetransmitTime !== undefined) {
			// Guess someone didn't appreciate the standardization process.
			return dict["channel"].maxRetransmitTime;
		}
		var out = dict["channel"].maxPacketLifeTime;
		return out === null ? 65535 : out;
	}, _js_id);
	/* clang-format on */
}

int WebRTCDataChannelJS::get_max_retransmits() const {
	return _JS_GET(maxRetransmits, 65535);
}

String WebRTCDataChannelJS::get_protocol() const {
	return _protocol;
}

bool WebRTCDataChannelJS::is_negotiated() const {
	return _JS_GET(negotiated, false);
}

WebRTCDataChannelJS::WebRTCDataChannelJS() {
	queue_count = 0;
	_was_string = false;
	_write_mode = WRITE_MODE_BINARY;
	_js_id = 0;
}

WebRTCDataChannelJS::WebRTCDataChannelJS(int js_id) {
	queue_count = 0;
	_was_string = false;
	_write_mode = WRITE_MODE_BINARY;
	_js_id = js_id;

	/* clang-format off */
	EM_ASM({
		var c_ptr = $0;
		var dict = Module.IDHandler.get($1);
		if (!dict) return;
		var channel = dict["channel"];
		dict["ptr"] = c_ptr;

		channel.binaryType = "arraybuffer";
		channel.onopen = function (evt) {
			ccall("_emrtc_on_ch_open",
				"void",
				["number"],
				[c_ptr]
			);
		};
		channel.onclose = function (evt) {
			ccall("_emrtc_on_ch_close",
				"void",
				["number"],
				[c_ptr]
			);
		};
		channel.onerror = function (evt) {
			ccall("_emrtc_on_ch_error",
				"void",
				["number"],
				[c_ptr]
			);
		};
		channel.onmessage = function(event) {
			var buffer;
			var is_string = 0;
			if (event.data instanceof ArrayBuffer) {
				buffer = new Uint8Array(event.data);
			} else if (event.data instanceof Blob) {
				console.error("Blob type not supported");
				return;
			} else if (typeof event.data === "string") {
				is_string = 1;
				var enc = new TextEncoder("utf-8");
				buffer = new Uint8Array(enc.encode(event.data));
			} else {
				console.error("Unknown message type");
				return;
			}
			var len = buffer.length*buffer.BYTES_PER_ELEMENT;
			var out = _malloc(len);
			HEAPU8.set(buffer, out);
			ccall("_emrtc_on_ch_message",
				"void",
				["number", "number", "number", "number"],
				[c_ptr, out, len, is_string]
			);
			_free(out);
		}

	}, this, js_id);
	// Parse label
	char *str;
	str = (char *)EM_ASM_INT({
		var dict = Module.IDHandler.get($0);
		if (!dict || !dict["channel"]) return 0;
		var str = dict["channel"].label;
		var len = lengthBytesUTF8(str)+1;
		var ptr = _malloc(str);
		stringToUTF8(str, ptr, len+1);
		return ptr;
	}, js_id);
	if(str != nullptr) {
		_label.parse_utf8(str);
		EM_ASM({ _free($0) }, str);
	}
	str = (char *)EM_ASM_INT({
		var dict = Module.IDHandler.get($0);
		if (!dict || !dict["channel"]) return 0;
		var str = dict["channel"].protocol;
		var len = lengthBytesUTF8(str)+1;
		var ptr = _malloc(str);
		stringToUTF8(str, ptr, len+1);
		return ptr;
	}, js_id);
	if(str != nullptr) {
		_protocol.parse_utf8(str);
		EM_ASM({ _free($0) }, str);
	}
	/* clang-format on */
}

WebRTCDataChannelJS::~WebRTCDataChannelJS() {
	close();
	/* clang-format off */
	EM_ASM({
		Module.IDHandler.remove($0);
	}, _js_id);
	/* clang-format on */
};
#endif
