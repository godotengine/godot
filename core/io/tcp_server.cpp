/**************************************************************************/
/*  tcp_server.cpp                                                        */
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

#include "tcp_server.h"

void SocketServer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("is_connection_available"), &TCPServer::is_connection_available);
	ClassDB::bind_method(D_METHOD("is_listening"), &TCPServer::is_listening);
	ClassDB::bind_method(D_METHOD("stop"), &TCPServer::stop);
}

void TCPServer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("listen", "port", "bind_address"), &TCPServer::listen, DEFVAL("*"));
	ClassDB::bind_method(D_METHOD("get_local_port"), &TCPServer::get_local_port);
	ClassDB::bind_method(D_METHOD("take_connection"), &TCPServer::take_connection);
}

Error TCPServer::listen(uint16_t p_port, const IPAddress &p_bind_address) {
	ERR_FAIL_COND_V(_sock.is_null(), ERR_UNAVAILABLE);
	ERR_FAIL_COND_V(_sock->is_open(), ERR_ALREADY_IN_USE);
	ERR_FAIL_COND_V(!p_bind_address.is_valid() && !p_bind_address.is_wildcard(), ERR_INVALID_PARAMETER);

	Error err;
	IP::Type ip_type = IP::TYPE_ANY;

	// If the bind address is valid use its type as the socket type
	if (p_bind_address.is_valid()) {
		ip_type = p_bind_address.is_ipv4() ? IP::TYPE_IPV4 : IP::TYPE_IPV6;
	}

	err = _sock->open(NetSocket::Family::INET, NetSocket::TYPE_TCP, ip_type);

	ERR_FAIL_COND_V(err != OK, ERR_CANT_CREATE);

	_sock->set_blocking_enabled(false);
	_sock->set_reuse_address_enabled(true);

	NetSocket::Address addr(p_bind_address, p_port);
	err = _sock->bind(addr);

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

int TCPServer::get_local_port() const {
	NetSocket::Address addr;
	_sock->get_socket_address(&addr);
	return addr.port();
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

Ref<StreamPeerTCP> TCPServer::take_connection() {
	Ref<StreamPeerTCP> conn;
	if (!is_connection_available()) {
		return conn;
	}

	Ref<NetSocket> ns;
	NetSocket::Address addr;
	ns = _sock->accept(addr);
	if (ns.is_null()) {
		return conn;
	}

	conn.instantiate();
	conn->accept_socket(ns, addr.ip(), addr.port());
	return conn;
}

void SocketServer::stop() {
	if (_sock.is_valid()) {
		_sock->close();
	}
}

SocketServer::SocketServer() :
		_sock(Ref<NetSocket>(NetSocket::create())) {
}

SocketServer::~SocketServer() {
	stop();
}

void UDSServer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("listen", "path"), &UDSServer::listen);
	ClassDB::bind_method(D_METHOD("take_connection"), &UDSServer::take_connection);
}

Error UDSServer::listen(const String &p_path) {
	ERR_FAIL_COND_V(!_sock.is_valid(), ERR_UNAVAILABLE);
	ERR_FAIL_COND_V(_sock->is_open(), ERR_ALREADY_IN_USE);
	ERR_FAIL_COND_V(p_path.is_empty(), ERR_INVALID_PARAMETER);

	IP::Type ip_type = IP::TYPE_NONE;
	Error err = _sock->open(NetSocket::Family::UNIX, NetSocket::TYPE_NONE, ip_type);
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

Ref<StreamPeerUDS> UDSServer::take_connection() {
	Ref<StreamPeerUDS> conn;
	if (!is_connection_available()) {
		return conn;
	}

	Ref<NetSocket> ns;
	NetSocket::Address addr;
	ns = _sock->accept(addr);
	if (ns.is_null()) {
		return conn;
	}

	conn.instantiate();
	conn->accept_socket(ns);
	return conn;
}
