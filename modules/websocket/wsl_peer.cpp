/*************************************************************************/
/*  wsl_peer.cpp                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "wsl_peer.h"

#include "wsl_client.h"
#include "wsl_server.h"

#include "core/crypto/crypto_core.h"
#include "core/math/random_number_generator.h"
#include "core/os/os.h"

String WSLPeer::generate_key() {
	// Random key
	RandomNumberGenerator rng;
	rng.set_seed(OS::get_singleton()->get_unix_time());
	PoolVector<uint8_t> bkey;
	int len = 16; // 16 bytes, as per RFC
	bkey.resize(len);
	PoolVector<uint8_t>::Write w = bkey.write();
	for (int i = 0; i < len; i++) {
		w[i] = (uint8_t)rng.randi_range(0, 255);
	}
	return CryptoCore::b64_encode_str(&w[0], len);
}

String WSLPeer::compute_key_response(String p_key) {
	String key = p_key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"; // Magic UUID as per RFC
	Vector<uint8_t> sha = key.sha1_buffer();
	return CryptoCore::b64_encode_str(sha.ptr(), sha.size());
}

void WSLPeer::_wsl_destroy(struct PeerData **p_data) {
	if (!p_data || !(*p_data)) {
		return;
	}
	struct PeerData *data = *p_data;
	if (data->polling) {
		data->destroy = true;
		return;
	}
	wslay_event_context_free(data->ctx);
	memdelete(data);
	*p_data = nullptr;
}

bool WSLPeer::_wsl_poll(struct PeerData *p_data) {
	p_data->polling = true;
	int err = 0;
	if ((err = wslay_event_recv(p_data->ctx)) != 0 || (err = wslay_event_send(p_data->ctx)) != 0) {
		print_verbose("Websocket (wslay) poll error: " + itos(err));
		p_data->destroy = true;
	}
	p_data->polling = false;

	if (p_data->destroy || (wslay_event_get_close_sent(p_data->ctx) && wslay_event_get_close_received(p_data->ctx))) {
		bool valid = p_data->valid;
		_wsl_destroy(&p_data);
		return valid;
	}
	return false;
}

ssize_t wsl_recv_callback(wslay_event_context_ptr ctx, uint8_t *data, size_t len, int flags, void *user_data) {
	struct WSLPeer::PeerData *peer_data = (struct WSLPeer::PeerData *)user_data;
	if (!peer_data->valid) {
		wslay_event_set_error(ctx, WSLAY_ERR_CALLBACK_FAILURE);
		return -1;
	}
	Ref<StreamPeer> conn = peer_data->conn;
	int read = 0;
	Error err = conn->get_partial_data(data, len, read);
	if (err != OK) {
		print_verbose("Websocket get data error: " + itos(err) + ", read (should be 0!): " + itos(read));
		wslay_event_set_error(ctx, WSLAY_ERR_CALLBACK_FAILURE);
		return -1;
	}
	if (read == 0) {
		wslay_event_set_error(ctx, WSLAY_ERR_WOULDBLOCK);
		return -1;
	}
	return read;
}

ssize_t wsl_send_callback(wslay_event_context_ptr ctx, const uint8_t *data, size_t len, int flags, void *user_data) {
	struct WSLPeer::PeerData *peer_data = (struct WSLPeer::PeerData *)user_data;
	if (!peer_data->valid) {
		wslay_event_set_error(ctx, WSLAY_ERR_CALLBACK_FAILURE);
		return -1;
	}
	Ref<StreamPeer> conn = peer_data->conn;
	int sent = 0;
	Error err = conn->put_partial_data(data, len, sent);
	if (err != OK) {
		wslay_event_set_error(ctx, WSLAY_ERR_CALLBACK_FAILURE);
		return -1;
	}
	if (sent == 0) {
		wslay_event_set_error(ctx, WSLAY_ERR_WOULDBLOCK);
		return -1;
	}
	return sent;
}

int wsl_genmask_callback(wslay_event_context_ptr ctx, uint8_t *buf, size_t len, void *user_data) {
	RandomNumberGenerator rng;
	// TODO maybe use crypto in the future?
	rng.set_seed(OS::get_singleton()->get_unix_time());
	for (unsigned int i = 0; i < len; i++) {
		buf[i] = (uint8_t)rng.randi_range(0, 255);
	}
	return 0;
}

void wsl_msg_recv_callback(wslay_event_context_ptr ctx, const struct wslay_event_on_msg_recv_arg *arg, void *user_data) {
	struct WSLPeer::PeerData *peer_data = (struct WSLPeer::PeerData *)user_data;
	if (!peer_data->valid || peer_data->closing) {
		return;
	}
	WSLPeer *peer = (WSLPeer *)peer_data->peer;

	if (peer->parse_message(arg) != OK) {
		return;
	}

	if (peer_data->is_server) {
		WSLServer *helper = (WSLServer *)peer_data->obj;
		helper->_on_peer_packet(peer_data->id);
	} else {
		WSLClient *helper = (WSLClient *)peer_data->obj;
		helper->_on_peer_packet();
	}
}

wslay_event_callbacks wsl_callbacks = {
	wsl_recv_callback,
	wsl_send_callback,
	wsl_genmask_callback,
	nullptr, /* on_frame_recv_start_callback */
	nullptr, /* on_frame_recv_callback */
	nullptr, /* on_frame_recv_end_callback */
	wsl_msg_recv_callback
};

Error WSLPeer::parse_message(const wslay_event_on_msg_recv_arg *arg) {
	uint8_t is_string = 0;
	if (arg->opcode == WSLAY_TEXT_FRAME) {
		is_string = 1;
	} else if (arg->opcode == WSLAY_CONNECTION_CLOSE) {
		close_code = arg->status_code;
		size_t len = arg->msg_length;
		close_reason = "";
		if (len > 2 /* first 2 bytes = close code */) {
			close_reason.parse_utf8((char *)arg->msg + 2, len - 2);
		}
		if (!wslay_event_get_close_sent(_data->ctx)) {
			if (_data->is_server) {
				WSLServer *helper = (WSLServer *)_data->obj;
				helper->_on_close_request(_data->id, close_code, close_reason);
			} else {
				WSLClient *helper = (WSLClient *)_data->obj;
				helper->_on_close_request(close_code, close_reason);
			}
		}
		return ERR_FILE_EOF;
	} else if (arg->opcode != WSLAY_BINARY_FRAME) {
		// Ping or pong
		return ERR_SKIP;
	}
	_in_buffer.write_packet(arg->msg, arg->msg_length, &is_string);
	return OK;
}

void WSLPeer::make_context(PeerData *p_data, unsigned int p_in_buf_size, unsigned int p_in_pkt_size, unsigned int p_out_buf_size, unsigned int p_out_pkt_size) {
	ERR_FAIL_COND(_data != nullptr);
	ERR_FAIL_COND(p_data == nullptr);

	_in_buffer.resize(p_in_pkt_size, p_in_buf_size);
	_packet_buffer.resize(1 << p_in_buf_size);
	_out_buf_size = p_out_buf_size;
	_out_pkt_size = p_out_pkt_size;

	_data = p_data;
	_data->peer = this;
	_data->valid = true;

	if (_data->is_server) {
		wslay_event_context_server_init(&(_data->ctx), &wsl_callbacks, _data);
	} else {
		wslay_event_context_client_init(&(_data->ctx), &wsl_callbacks, _data);
	}
	wslay_event_config_set_max_recv_msg_length(_data->ctx, (1ULL << p_in_buf_size));
}

void WSLPeer::set_write_mode(WriteMode p_mode) {
	write_mode = p_mode;
}

WSLPeer::WriteMode WSLPeer::get_write_mode() const {
	return write_mode;
}

void WSLPeer::poll() {
	if (!_data) {
		return;
	}

	if (_wsl_poll(_data)) {
		_data = nullptr;
	}
}

Error WSLPeer::put_packet(const uint8_t *p_buffer, int p_buffer_size) {
	ERR_FAIL_COND_V(!is_connected_to_host(), FAILED);
	ERR_FAIL_COND_V(_out_pkt_size && (wslay_event_get_queued_msg_count(_data->ctx) >= (1ULL << _out_pkt_size)), ERR_OUT_OF_MEMORY);
	ERR_FAIL_COND_V(_out_buf_size && (wslay_event_get_queued_msg_length(_data->ctx) + p_buffer_size >= (1ULL << _out_buf_size)), ERR_OUT_OF_MEMORY);

	struct wslay_event_msg msg;
	msg.opcode = write_mode == WRITE_MODE_TEXT ? WSLAY_TEXT_FRAME : WSLAY_BINARY_FRAME;
	msg.msg = p_buffer;
	msg.msg_length = p_buffer_size;

	if (wslay_event_queue_msg(_data->ctx, &msg) != 0 || wslay_event_send(_data->ctx) != 0) {
		close_now();
		return FAILED;
	}
	return OK;
}

Error WSLPeer::get_packet(const uint8_t **r_buffer, int &r_buffer_size) {
	r_buffer_size = 0;

	ERR_FAIL_COND_V(!is_connected_to_host(), FAILED);

	if (_in_buffer.packets_left() == 0) {
		return ERR_UNAVAILABLE;
	}

	int read = 0;
	PoolVector<uint8_t>::Write rw = _packet_buffer.write();
	_in_buffer.read_packet(rw.ptr(), _packet_buffer.size(), &_is_string, read);

	*r_buffer = rw.ptr();
	r_buffer_size = read;

	return OK;
}

int WSLPeer::get_available_packet_count() const {
	if (!is_connected_to_host()) {
		return 0;
	}

	return _in_buffer.packets_left();
}

int WSLPeer::get_current_outbound_buffered_amount() const {
	ERR_FAIL_COND_V(!_data, 0);

	return wslay_event_get_queued_msg_length(_data->ctx);
}

bool WSLPeer::was_string_packet() const {
	return _is_string;
}

bool WSLPeer::is_connected_to_host() const {
	return _data != nullptr;
}

void WSLPeer::close_now() {
	close(1000, "");
	_wsl_destroy(&_data);
}

void WSLPeer::close(int p_code, String p_reason) {
	if (_data && !wslay_event_get_close_sent(_data->ctx)) {
		CharString cs = p_reason.utf8();
		wslay_event_queue_close(_data->ctx, p_code, (uint8_t *)cs.ptr(), cs.size());
		wslay_event_send(_data->ctx);
		_data->closing = true;
	}

	_in_buffer.clear();
	_packet_buffer.resize(0);
}

IP_Address WSLPeer::get_connected_host() const {
	ERR_FAIL_COND_V(!is_connected_to_host() || _data->tcp.is_null(), IP_Address());

	return _data->tcp->get_connected_host();
}

uint16_t WSLPeer::get_connected_port() const {
	ERR_FAIL_COND_V(!is_connected_to_host() || _data->tcp.is_null(), 0);

	return _data->tcp->get_connected_port();
}

void WSLPeer::set_no_delay(bool p_enabled) {
	ERR_FAIL_COND(!is_connected_to_host() || _data->tcp.is_null());
	_data->tcp->set_no_delay(p_enabled);
}

void WSLPeer::invalidate() {
	if (_data) {
		_data->valid = false;
	}
}

WSLPeer::WSLPeer() {
	_data = nullptr;
	_is_string = 0;
	close_code = -1;
	write_mode = WRITE_MODE_BINARY;
	_out_buf_size = 0;
	_out_pkt_size = 0;
}

WSLPeer::~WSLPeer() {
	close();
	invalidate();
	_wsl_destroy(&_data);
	_data = nullptr;
}

#endif // JAVASCRIPT_ENABLED
