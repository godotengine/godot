/*************************************************************************/
/*  udp_server.cpp                                                       */
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

#include "udp_server.h"

void UDPServer::_bind_methods() {

	ClassDB::bind_method(D_METHOD("listen", "port", "bind_address"), &UDPServer::listen, DEFVAL("*"));
	ClassDB::bind_method(D_METHOD("is_connection_available"), &UDPServer::is_connection_available);
	ClassDB::bind_method(D_METHOD("is_listening"), &UDPServer::is_listening);
	ClassDB::bind_method(D_METHOD("take_connection"), &UDPServer::take_connection);
	ClassDB::bind_method(D_METHOD("stop"), &UDPServer::stop);
}

Error UDPServer::listen(uint16_t p_port, const IP_Address &p_bind_address) {

	ERR_FAIL_COND_V(!_sock.is_valid(), ERR_UNAVAILABLE);
	ERR_FAIL_COND_V(_sock->is_open(), ERR_ALREADY_IN_USE);
	ERR_FAIL_COND_V(!p_bind_address.is_valid() && !p_bind_address.is_wildcard(), ERR_INVALID_PARAMETER);

	Error err;
	IP::Type ip_type = IP::TYPE_ANY;

	if (p_bind_address.is_valid())
		ip_type = p_bind_address.is_ipv4() ? IP::TYPE_IPV4 : IP::TYPE_IPV6;

	err = _sock->open(NetSocket::TYPE_UDP, ip_type);

	if (err != OK)
		return ERR_CANT_CREATE;

	_sock->set_blocking_enabled(false);
	_sock->set_reuse_address_enabled(true);
	err = _sock->bind(p_bind_address, p_port);

	if (err != OK) {
		stop();
		return err;
	}
	bind_address = p_bind_address;
	bind_port = p_port;
	return OK;
}

bool UDPServer::is_listening() const {
	ERR_FAIL_COND_V(!_sock.is_valid(), false);

	return _sock->is_open();
}

bool UDPServer::is_connection_available() const {

	ERR_FAIL_COND_V(!_sock.is_valid(), false);

	if (!_sock->is_open())
		return false;

	Error err = _sock->poll(NetSocket::POLL_TYPE_IN, 0);
	return (err == OK);
}

Ref<PacketPeerUDP> UDPServer::take_connection() {

	Ref<PacketPeerUDP> conn;
	if (!is_connection_available()) {
		return conn;
	}

	conn = Ref<PacketPeerUDP>(memnew(PacketPeerUDP));
	conn->connect_socket(_sock);
	_sock = Ref<NetSocket>(NetSocket::create());
	listen(bind_port, bind_address);
	return conn;
}

void UDPServer::stop() {

	if (_sock.is_valid()) {
		_sock->close();
	}
	bind_port = 0;
	bind_address = IP_Address();
}

UDPServer::UDPServer() :
		_sock(Ref<NetSocket>(NetSocket::create())) {
}

UDPServer::~UDPServer() {

	stop();
}
