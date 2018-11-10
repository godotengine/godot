/*************************************************************************/
/*  emws_peer.cpp                                                        */
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

#include "emws_peer.h"
#include "core/io/ip.h"

void EMWSPeer::set_sock(int p_sock) {

	peer_sock = p_sock;
	in_buffer.clear();
	queue_count = 0;
}

void EMWSPeer::set_write_mode(WriteMode p_mode) {
	write_mode = p_mode;
}

EMWSPeer::WriteMode EMWSPeer::get_write_mode() const {
	return write_mode;
}

void EMWSPeer::read_msg(uint8_t *p_data, uint32_t p_size, bool p_is_string) {

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

Error EMWSPeer::put_packet(const uint8_t *p_buffer, int p_buffer_size) {

	int is_bin = write_mode == WebSocketPeer::WRITE_MODE_BINARY ? 1 : 0;

	/* clang-format off */
	EM_ASM({
		var sock = Module.IDHandler.get($0);
		var bytes_array = new Uint8Array($2);
		var i = 0;

		for(i=0; i<$2; i++) {
			bytes_array[i] = getValue($1+i, 'i8');
		}

		if ($3) {
			sock.send(bytes_array.buffer);
		} else {
			var string = new TextDecoder("utf-8").decode(bytes_array);
			sock.send(string);
		}
	}, peer_sock, p_buffer, p_buffer_size, is_bin);
	/* clang-format on */

	return OK;
};

Error EMWSPeer::get_packet(const uint8_t **r_buffer, int &r_buffer_size) {

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
};

int EMWSPeer::get_available_packet_count() const {

	return queue_count;
};

bool EMWSPeer::was_string_packet() const {

	return _was_string;
};

bool EMWSPeer::is_connected_to_host() const {

	return peer_sock != -1;
};

void EMWSPeer::close(int p_code, String p_reason) {

	if (peer_sock != -1) {
		/* clang-format off */
		EM_ASM({
			var sock = Module.IDHandler.get($0);
			var code = $1;
			var reason = UTF8ToString($2);
			sock.close(code, reason);
			Module.IDHandler.remove($0);
		}, peer_sock, p_code, p_reason.utf8().get_data());
		/* clang-format on */
	}
	peer_sock = -1;
	queue_count = 0;
	in_buffer.clear();
};

IP_Address EMWSPeer::get_connected_host() const {

	ERR_EXPLAIN("Not supported in HTML5 export");
	ERR_FAIL_V(IP_Address());
};

uint16_t EMWSPeer::get_connected_port() const {

	ERR_EXPLAIN("Not supported in HTML5 export");
	ERR_FAIL_V(0);
};

EMWSPeer::EMWSPeer() {
	peer_sock = -1;
	queue_count = 0;
	_was_string = false;
	in_buffer.resize(16);
	write_mode = WRITE_MODE_BINARY;
};

EMWSPeer::~EMWSPeer() {

	in_buffer.resize(0);
	close();
};

#endif // JAVASCRIPT_ENABLED
