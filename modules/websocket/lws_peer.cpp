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

void LWSPeer::set_wsi(struct lws *p_wsi) {
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

	PeerData *peer_data = (PeerData *)(lws_wsi_user(wsi));
	uint32_t size = peer_data->in_size;
	uint8_t is_string = lws_frame_is_binary(wsi) ? 0 : 1;

	if (peer_data->rbr.space_left() < len + 5) {
		ERR_EXPLAIN("Buffer full! Dropping data");
		ERR_FAIL_V(FAILED);
	}

	copymem(&(peer_data->input_buffer[size]), in, len);
	size += len;

	peer_data->in_size = size;
	if (lws_is_final_fragment(wsi)) {
		peer_data->rbr.write((uint8_t *)&size, 4);
		peer_data->rbr.write((uint8_t *)&is_string, 1);
		peer_data->rbr.write(peer_data->input_buffer, size);
		peer_data->in_count++;
		peer_data->in_size = 0;
	}

	return OK;
}

Error LWSPeer::write_wsi() {

	ERR_FAIL_COND_V(!is_connected_to_host(), FAILED);

	PeerData *peer_data = (PeerData *)(lws_wsi_user(wsi));
	PoolVector<uint8_t> tmp;
	int left = peer_data->rbw.data_left();
	uint32_t to_write = 0;

	if (left == 0 || peer_data->out_count == 0)
		return OK;

	peer_data->rbw.read((uint8_t *)&to_write, 4);
	peer_data->out_count--;

	if (left < to_write) {
		peer_data->rbw.advance_read(left);
		return FAILED;
	}

	tmp.resize(LWS_PRE + to_write);
	peer_data->rbw.read(&(tmp.write()[LWS_PRE]), to_write);
	lws_write(wsi, &(tmp.write()[LWS_PRE]), to_write, (enum lws_write_protocol)write_mode);
	tmp.resize(0);

	if (peer_data->out_count > 0)
		lws_callback_on_writable(wsi); // we want to write more!

	return OK;
}

Error LWSPeer::put_packet(const uint8_t *p_buffer, int p_buffer_size) {

	ERR_FAIL_COND_V(!is_connected_to_host(), FAILED);

	PeerData *peer_data = (PeerData *)lws_wsi_user(wsi);
	peer_data->rbw.write((uint8_t *)&p_buffer_size, 4);
	peer_data->rbw.write(p_buffer, MIN(p_buffer_size, peer_data->rbw.space_left()));
	peer_data->out_count++;

	lws_callback_on_writable(wsi); // notify that we want to write
	return OK;
};

Error LWSPeer::get_packet(const uint8_t **r_buffer, int &r_buffer_size) {

	ERR_FAIL_COND_V(!is_connected_to_host(), FAILED);

	PeerData *peer_data = (PeerData *)lws_wsi_user(wsi);

	if (peer_data->in_count == 0)
		return ERR_UNAVAILABLE;

	uint32_t to_read = 0;
	uint32_t left = 0;
	uint8_t is_string = 0;
	r_buffer_size = 0;

	peer_data->rbr.read((uint8_t *)&to_read, 4);
	peer_data->in_count--;
	left = peer_data->rbr.data_left();

	if (left < to_read + 1) {
		peer_data->rbr.advance_read(left);
		return FAILED;
	}

	peer_data->rbr.read(&is_string, 1);
	peer_data->rbr.read(packet_buffer, to_read);
	*r_buffer = packet_buffer;
	r_buffer_size = to_read;
	_was_string = is_string;

	return OK;
};

int LWSPeer::get_available_packet_count() const {

	if (!is_connected_to_host())
		return 0;

	return ((PeerData *)lws_wsi_user(wsi))->in_count;
};

bool LWSPeer::was_string_packet() const {

	return _was_string;
};

bool LWSPeer::is_connected_to_host() const {

	return wsi != NULL;
};

void LWSPeer::close() {
	if (wsi != NULL) {
		struct lws *tmp = wsi;
		PeerData *data = ((PeerData *)lws_wsi_user(wsi));
		data->force_close = true;
		wsi = NULL;
		lws_callback_on_writable(tmp); // notify that we want to disconnect
	}
};

IP_Address LWSPeer::get_connected_host() const {

	return IP_Address();
};

uint16_t LWSPeer::get_connected_port() const {

	return 1025;
};

LWSPeer::LWSPeer() {
	wsi = NULL;
	_was_string = false;
	write_mode = WRITE_MODE_BINARY;
};

LWSPeer::~LWSPeer() {

	close();
};

#endif // JAVASCRIPT_ENABLED
