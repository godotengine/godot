/**************************************************************************/
/*  socket_server.cpp                                                     */
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

#include "socket_server.h"

void SocketServer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("is_connection_available"), &SocketServer::is_connection_available);
	ClassDB::bind_method(D_METHOD("is_listening"), &SocketServer::is_listening);
	ClassDB::bind_method(D_METHOD("stop"), &SocketServer::stop);
	ClassDB::bind_method(D_METHOD("take_socket_connection"), &SocketServer::take_socket_connection);
}

Error SocketServer::_listen(const NetSocket::Address &p_addr) {
	DEV_ASSERT(_sock.is_valid());
	DEV_ASSERT(_sock->is_open());

	_sock->set_blocking_enabled(false);
	Error err = _sock->bind(p_addr);

	if (err != OK) {
		_sock->close();
		return ERR_ALREADY_IN_USE;
	}

	err = _sock->listen(MAX_PENDING_CONNECTIONS);

	if (err != OK) {
		_sock->close();
		return FAILED;
	}
	return OK;
}

bool SocketServer::is_listening() const {
	ERR_FAIL_COND_V(_sock.is_null(), false);

	return _sock->is_open();
}

bool SocketServer::is_connection_available() const {
	ERR_FAIL_COND_V(_sock.is_null(), false);

	if (!_sock->is_open()) {
		return false;
	}

	Error err = _sock->poll(NetSocket::POLL_TYPE_IN, 0);
	return (err == OK);
}

void SocketServer::stop() {
	if (_sock.is_valid()) {
		_sock->close();
	}
}

SocketServer::SocketServer() :
		_sock(NetSocket::create()) {
}

SocketServer::~SocketServer() {
	stop();
}
