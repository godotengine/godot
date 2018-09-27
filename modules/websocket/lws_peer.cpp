/*************************************************************************/
/*  lws_peer.cpp                                                         */
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
#ifndef JAVASCRIPT_ENABLED

#include "lws_peer.h"

#include "core/io/ip.h"

// Needed for socket_helpers on Android at least. UNIXes has it, just include if not windows
#if !defined(WINDOWS_ENABLED)
#include <netinet/in.h>
#include <sys/socket.h>
#endif

#include "drivers/unix/net_socket_posix.h"

void LWSPeer::set_wsi(struct lws *p_wsi) {
	ERR_FAIL_COND(wsi != NULL);

	rbw.resize(16);
	rbr.resize(16);
	wsi = p_wsi;
};

void LWSPeer::set_write_mode(WriteMode p_mode) {
	write_mode = p_mode;
}

LWSPeer::WriteMode LWSPeer::get_write_mode() const {
	return write_mode;
}

Error LWSPeer::read_wsi(void *in, size_t len) {

	ERR_FAIL_COND_V(!is_connected_to_host(), FAILED);

	uint32_t size = in_size;
	uint8_t is_string = lws_frame_is_binary(wsi) ? 0 : 1;

	if (rbr.space_left() < len + 5) {
		ERR_EXPLAIN("Buffer full! Dropping data");
		ERR_FAIL_V(FAILED);
	}

	copymem(&(input_buffer[size]), in, len);
	size += len;

	in_size = size;
	if (lws_is_final_fragment(wsi)) {
		rbr.write((uint8_t *)&size, 4);
		rbr.write((uint8_t *)&is_string, 1);
		rbr.write(input_buffer, size);
		in_count++;
		in_size = 0;
	}

	return OK;
}

Error LWSPeer::write_wsi() {

	ERR_FAIL_COND_V(!is_connected_to_host(), FAILED);

	PoolVector<uint8_t> tmp;
	int left = rbw.data_left();
	uint32_t to_write = 0;

	if (left == 0 || out_count == 0)
		return OK;

	rbw.read((uint8_t *)&to_write, 4);
	out_count--;

	if (left < to_write) {
		rbw.advance_read(left);
		return FAILED;
	}

	tmp.resize(LWS_PRE + to_write);
	rbw.read(&(tmp.write()[LWS_PRE]), to_write);
	lws_write(wsi, &(tmp.write()[LWS_PRE]), to_write, (enum lws_write_protocol)write_mode);
	tmp.resize(0);

	if (out_count > 0)
		lws_callback_on_writable(wsi); // we want to write more!

	return OK;
}

Error LWSPeer::put_packet(const uint8_t *p_buffer, int p_buffer_size) {

	ERR_FAIL_COND_V(!is_connected_to_host(), FAILED);

	rbw.write((uint8_t *)&p_buffer_size, 4);
	rbw.write(p_buffer, MIN(p_buffer_size, rbw.space_left()));
	out_count++;

	lws_callback_on_writable(wsi); // notify that we want to write
	return OK;
};

Error LWSPeer::get_packet(const uint8_t **r_buffer, int &r_buffer_size) {

	ERR_FAIL_COND_V(!is_connected_to_host(), FAILED);

	if (in_count == 0)
		return ERR_UNAVAILABLE;

	uint32_t to_read = 0;
	uint32_t left = 0;
	uint8_t is_string = 0;
	r_buffer_size = 0;

	rbr.read((uint8_t *)&to_read, 4);
	in_count--;
	left = rbr.data_left();

	if (left < to_read + 1) {
		rbr.advance_read(left);
		return FAILED;
	}

	rbr.read(&is_string, 1);
	rbr.read(packet_buffer, to_read);
	*r_buffer = packet_buffer;
	r_buffer_size = to_read;
	_was_string = is_string;

	return OK;
};

int LWSPeer::get_available_packet_count() const {

	if (!is_connected_to_host())
		return 0;

	return in_count;
};

bool LWSPeer::was_string_packet() const {

	return _was_string;
};

bool LWSPeer::is_connected_to_host() const {

	return wsi != NULL;
};

String LWSPeer::get_close_reason(void *in, size_t len, int &r_code) {
	String s;
	r_code = 0;
	if (len < 2) // From docs this should not happen
		return s;

	const uint8_t *b = (const uint8_t *)in;
	r_code = b[0] << 8 | b[1];

	if (len > 2) {
		const char *utf8 = (const char *)&b[2];
		s.parse_utf8(utf8, len - 2);
	}
	return s;
}

void LWSPeer::send_close_status(struct lws *p_wsi) {
	if (close_code == -1)
		return;

	int len = close_reason.size();
	ERR_FAIL_COND(len > 123); // Maximum allowed reason size in bytes

	lws_close_status code = (lws_close_status)close_code;
	unsigned char *reason = len > 0 ? (unsigned char *)close_reason.utf8().ptrw() : NULL;

	lws_close_reason(p_wsi, code, reason, len);

	close_code = -1;
	close_reason = "";
}

void LWSPeer::close(int p_code, String p_reason) {
	if (wsi != NULL) {
		close_code = p_code;
		close_reason = p_reason;
		PeerData *data = ((PeerData *)lws_wsi_user(wsi));
		data->force_close = true;
		data->clean_close = true;
		lws_callback_on_writable(wsi); // Notify that we want to disconnect
	} else {
		close_code = -1;
		close_reason = "";
	}
	wsi = NULL;
	rbw.resize(0);
	rbr.resize(0);
	in_count = 0;
	in_size = 0;
	out_count = 0;
	_was_string = false;
};

IP_Address LWSPeer::get_connected_host() const {

	ERR_FAIL_COND_V(!is_connected_to_host(), IP_Address());

	IP_Address ip;
	uint16_t port = 0;

	struct sockaddr_storage addr;
	socklen_t len = sizeof(addr);

	int fd = lws_get_socket_fd(wsi);
	ERR_FAIL_COND_V(fd == -1, IP_Address());

	int ret = getpeername(fd, (struct sockaddr *)&addr, &len);
	ERR_FAIL_COND_V(ret != 0, IP_Address());

	NetSocketPosix::_set_ip_port(&addr, ip, port);

	return ip;
};

uint16_t LWSPeer::get_connected_port() const {

	ERR_FAIL_COND_V(!is_connected_to_host(), 0);

	IP_Address ip;
	uint16_t port = 0;

	struct sockaddr_storage addr;
	socklen_t len = sizeof(addr);

	int fd = lws_get_socket_fd(wsi);
	ERR_FAIL_COND_V(fd == -1, 0);

	int ret = getpeername(fd, (struct sockaddr *)&addr, &len);
	ERR_FAIL_COND_V(ret != 0, 0);

	NetSocketPosix::_set_ip_port(&addr, ip, port);

	return port;
};

LWSPeer::LWSPeer() {
	wsi = NULL;
	write_mode = WRITE_MODE_BINARY;
	close();
};

LWSPeer::~LWSPeer() {

	close();
};

#endif // JAVASCRIPT_ENABLED
