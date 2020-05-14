/*************************************************************************/
/*  emws_peer.cpp                                                        */
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

#include "emws_peer.h"
#include "core/io/ip.h"

void EMWSPeer::set_sock(int p_sock, unsigned int p_in_buf_size, unsigned int p_in_pkt_size) {
	peer_sock = p_sock;
	_in_buffer.resize(p_in_pkt_size, p_in_buf_size);
	_packet_buffer.resize((1 << p_in_buf_size));
}

void EMWSPeer::set_write_mode(WriteMode p_mode) {
	write_mode = p_mode;
}

EMWSPeer::WriteMode EMWSPeer::get_write_mode() const {
	return write_mode;
}

Error EMWSPeer::read_msg(uint8_t *p_data, uint32_t p_size, bool p_is_string) {
	uint8_t is_string = p_is_string ? 1 : 0;
	return _in_buffer.write_packet(p_data, p_size, &is_string);
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

		try {
			if ($3) {
				sock.send(bytes_array.buffer);
			} else {
				var string = new TextDecoder("utf-8").decode(bytes_array);
				sock.send(string);
			}
		} catch (e) {
			return 1;
		}
		return 0;
	}, peer_sock, p_buffer, p_buffer_size, is_bin);
	/* clang-format on */

	return OK;
};

Error EMWSPeer::get_packet(const uint8_t **r_buffer, int &r_buffer_size) {
	if (_in_buffer.packets_left() == 0)
		return ERR_UNAVAILABLE;

	int read = 0;
	Error err = _in_buffer.read_packet(_packet_buffer.ptrw(), _packet_buffer.size(), &_is_string, read);
	ERR_FAIL_COND_V(err != OK, err);

	*r_buffer = _packet_buffer.ptr();
	r_buffer_size = read;

	return OK;
};

int EMWSPeer::get_available_packet_count() const {
	return _in_buffer.packets_left();
};

bool EMWSPeer::was_string_packet() const {
	return _is_string;
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
	_is_string = 0;
	_in_buffer.clear();
	peer_sock = -1;
};

IP_Address EMWSPeer::get_connected_host() const {
	ERR_FAIL_V_MSG(IP_Address(), "Not supported in HTML5 export.");
};

uint16_t EMWSPeer::get_connected_port() const {
	ERR_FAIL_V_MSG(0, "Not supported in HTML5 export.");
};

void EMWSPeer::set_no_delay(bool p_enabled) {
	ERR_FAIL_MSG("'set_no_delay' is not supported in HTML5 export.");
}

EMWSPeer::EMWSPeer() {
	peer_sock = -1;
	write_mode = WRITE_MODE_BINARY;
	close();
};

EMWSPeer::~EMWSPeer() {
	close();
};

#endif // JAVASCRIPT_ENABLED
