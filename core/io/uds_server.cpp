/**************************************************************************/
/*  uds_server.cpp                                                        */
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

#include "uds_server.h"

void UDSServer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("listen", "path"), &UDSServer::listen);
	ClassDB::bind_method(D_METHOD("is_listening"), &UDSServer::is_listening);
	ClassDB::bind_method(D_METHOD("is_connection_available"), &UDSServer::is_connection_available);
	ClassDB::bind_method(D_METHOD("take_connection"), &UDSServer::take_connection);
	ClassDB::bind_method(D_METHOD("stop"), &UDSServer::stop);
}

Error UDSServer::listen(const String &p_path) {
	ERR_FAIL_COND_V(!_sock.is_valid(), ERR_UNAVAILABLE);
	ERR_FAIL_COND_V(_sock->is_open(), ERR_ALREADY_IN_USE);
	ERR_FAIL_COND_V(p_path.is_empty(), ERR_INVALID_PARAMETER);

	Error err = _sock->open();
	ERR_FAIL_COND_V(err != OK, ERR_CANT_CREATE);

	_sock->set_blocking_enabled(false);

	err = _sock->bind(p_path);

	if (err != OK) {
		_sock->close();
		return FAILED;
	}

	err = _sock->listen(MAX_PENDING_CONNECTIONS);

	if (err != OK) {
		_sock->close();
		return FAILED;
	}
	return OK;
}

bool UDSServer::is_listening() const {
	ERR_FAIL_COND_V(!_sock.is_valid(), false);

	return _sock->is_open();
}

bool UDSServer::is_connection_available() const {
	ERR_FAIL_COND_V(!_sock.is_valid(), false);

	if (!_sock->is_open()) {
		return false;
	}

	Error err = _sock->poll(UDSSocket::POLL_TYPE_IN, 0);
	return err == OK;
}

Ref<StreamPeerUDS> UDSServer::take_connection() {
	Ref<StreamPeerUDS> conn;
	if (!is_connection_available()) {
		return conn;
	}

	Ref<UDSSocket> sock = _sock->accept();
	if (sock.is_null()) {
		return conn;
	}

	conn.instantiate();
	conn->accept_socket(sock);
	return conn;
}

void UDSServer::stop() {
	if (_sock.is_valid()) {
		_sock->close();
	}
}

UDSServer::UDSServer() :
		_sock(Ref<UDSSocket>(UDSSocket::create())) {
}

UDSServer::~UDSServer() {
	stop();
}
