/**************************************************************************/
/*  stream_peer_uds.cpp                                                   */
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

#include "stream_peer_uds.h"

#include "core/config/project_settings.h"
#include "core/os/os.h"

Error StreamPeerUDS::poll() {
	if (status == STATUS_CONNECTED) {
		Error err;
		err = _sock->poll(UDSSocket::POLL_TYPE_IN, 0);
		if (err == OK) {
			// FIN received
			if (_sock->get_available_bytes() == 0) {
				disconnect_from_host();
				return OK;
			}
		}
		// Also poll write
		err = _sock->poll(UDSSocket::POLL_TYPE_IN_OUT, 0);
		if (err != OK && err != ERR_BUSY) {
			// Got an error
			disconnect_from_host();
			status = STATUS_ERROR;
			return err;
		}
		return OK;
	} else if (status != STATUS_CONNECTING) {
		return OK;
	}

	Error err = _sock->connect_to_host(path);

	if (err == OK) {
		status = STATUS_CONNECTED;
		return OK;
	} else if (err == ERR_BUSY) {
		// Check for connect timeout
		if (OS::get_singleton()->get_ticks_msec() > timeout) {
			disconnect_from_host();
			status = STATUS_ERROR;
			return ERR_CONNECTION_ERROR;
		}
		// Still trying to connect
		return OK;
	}

	disconnect_from_host();
	status = STATUS_ERROR;
	return ERR_CONNECTION_ERROR;
}

void StreamPeerUDS::accept_socket(Ref<UDSSocket> p_sock) {
	_sock = p_sock;
	_sock->set_blocking_enabled(false);

	timeout = OS::get_singleton()->get_ticks_msec() + (((uint64_t)GLOBAL_GET("network/limits/tcp/connect_timeout_seconds")) * 1000);
	status = STATUS_CONNECTED;
}

Error StreamPeerUDS::bind(const String &p_path) {
	ERR_FAIL_COND_V(!_sock.is_valid(), ERR_UNAVAILABLE);
	ERR_FAIL_COND_V(_sock->is_open(), ERR_ALREADY_IN_USE);

	Error err = _sock->open();
	if (err != OK) {
		return err;
	}
	_sock->set_blocking_enabled(false);
	return _sock->bind(p_path);
}

Error StreamPeerUDS::connect_to_host(const String &p_path) {
	ERR_FAIL_COND_V(!_sock.is_valid(), ERR_UNAVAILABLE);
	ERR_FAIL_COND_V(_sock->is_open(), ERR_ALREADY_IN_USE);
	ERR_FAIL_COND_V(p_path.is_empty(), ERR_INVALID_PARAMETER);

	if (!_sock->is_open()) {
		Error err = _sock->open();
		if (err != OK) {
			return err;
		}
		_sock->set_blocking_enabled(false);
	}

	timeout = OS::get_singleton()->get_ticks_msec() + (((uint64_t)GLOBAL_GET("network/limits/tcp/connect_timeout_seconds")) * 1000);
	Error err = _sock->connect_to_host(p_path);

	if (err == OK) {
		status = STATUS_CONNECTED;
	} else if (err == ERR_BUSY) {
		status = STATUS_CONNECTING;
	} else {
		ERR_PRINT("Connection to remote host failed!");
		disconnect_from_host();
		return FAILED;
	}

	path = p_path;

	return OK;
}

Error StreamPeerUDS::write(const uint8_t *p_data, int p_bytes, int &r_sent, bool p_block) {
	ERR_FAIL_COND_V(_sock.is_null(), ERR_UNAVAILABLE);

	if (status != STATUS_CONNECTED) {
		return FAILED;
	}

	Error err;
	int data_to_send = p_bytes;
	const uint8_t *offset = p_data;
	int total_sent = 0;

	while (data_to_send) {
		int sent_amount = 0;
		err = _sock->send(offset, data_to_send, sent_amount);

		if (err != OK) {
			if (err != ERR_BUSY) {
				disconnect_from_host();
				return FAILED;
			}

			if (!p_block) {
				r_sent = total_sent;
				return OK;
			}

			// Block and wait for the socket to accept more data.
			err = _sock->poll(UDSSocket::POLL_TYPE_OUT, -1);
			if (err != OK) {
				disconnect_from_host();
				return FAILED;
			}
		} else {
			data_to_send -= sent_amount;
			offset += sent_amount;
			total_sent += sent_amount;
		}
	}

	r_sent = total_sent;

	return OK;
}

Error StreamPeerUDS::read(uint8_t *p_buffer, int p_bytes, int &r_received, bool p_block) {
	if (status != STATUS_CONNECTED) {
		return FAILED;
	}

	Error err;
	int to_read = p_bytes;
	int total_read = 0;
	r_received = 0;

	while (to_read) {
		int read = 0;
		err = _sock->recv(p_buffer + total_read, to_read, read);

		if (err != OK) {
			if (err != ERR_BUSY) {
				disconnect_from_host();
				return FAILED;
			}

			if (!p_block) {
				r_received = total_read;
				return OK;
			}

			err = _sock->poll(UDSSocket::POLL_TYPE_IN, -1);

			if (err != OK) {
				disconnect_from_host();
				return FAILED;
			}

		} else if (read == 0) {
			disconnect_from_host();
			r_received = total_read;
			return ERR_FILE_EOF;

		} else {
			to_read -= read;
			total_read += read;

			if (!p_block) {
				r_received = total_read;
				return OK;
			}
		}
	}

	r_received = total_read;

	return OK;
}

StreamPeerUDS::Status StreamPeerUDS::get_status() const {
	return status;
}

void StreamPeerUDS::disconnect_from_host() {
	if (_sock.is_valid()) {
		_sock->close();
	}

	timeout = 0;
	status = STATUS_NONE;
	path.clear();
}

Error StreamPeerUDS::wait(UDSSocket::PollType p_type, int p_timeout) {
	ERR_FAIL_COND_V(_sock.is_null() || !_sock->is_open(), ERR_UNAVAILABLE);
	return _sock->poll(p_type, p_timeout);
}

Error StreamPeerUDS::put_data(const uint8_t *p_data, int p_bytes) {
	int total;
	return write(p_data, p_bytes, total, true);
}

Error StreamPeerUDS::put_partial_data(const uint8_t *p_data, int p_bytes, int &r_sent) {
	return write(p_data, p_bytes, r_sent, false);
}

Error StreamPeerUDS::get_data(uint8_t *p_buffer, int p_bytes) {
	int total;
	return read(p_buffer, p_bytes, total, true);
}

Error StreamPeerUDS::get_partial_data(uint8_t *p_buffer, int p_bytes, int &r_received) {
	return read(p_buffer, p_bytes, r_received, false);
}

int StreamPeerUDS::get_available_bytes() const {
	ERR_FAIL_COND_V(_sock.is_null(), -1);
	return _sock->get_available_bytes();
}

void StreamPeerUDS::_bind_methods() {
	ClassDB::bind_method(D_METHOD("bind", "path"), &StreamPeerUDS::bind);
	ClassDB::bind_method(D_METHOD("connect_to_host", "path"), &StreamPeerUDS::connect_to_host);
	ClassDB::bind_method(D_METHOD("poll"), &StreamPeerUDS::poll);
	ClassDB::bind_method(D_METHOD("get_status"), &StreamPeerUDS::get_status);
	ClassDB::bind_method(D_METHOD("get_connected_path"), &StreamPeerUDS::get_connected_path);
	ClassDB::bind_method(D_METHOD("disconnect_from_host"), &StreamPeerUDS::disconnect_from_host);

	BIND_ENUM_CONSTANT(STATUS_NONE);
	BIND_ENUM_CONSTANT(STATUS_CONNECTING);
	BIND_ENUM_CONSTANT(STATUS_CONNECTED);
	BIND_ENUM_CONSTANT(STATUS_ERROR);
}

StreamPeerUDS::StreamPeerUDS() :
		_sock(Ref<UDSSocket>(UDSSocket::create())) {
}

StreamPeerUDS::~StreamPeerUDS() {
	disconnect_from_host();
}
