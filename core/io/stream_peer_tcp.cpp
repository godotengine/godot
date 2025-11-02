/**************************************************************************/
/*  stream_peer_tcp.cpp                                                   */
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

#include "stream_peer_tcp.h"

#include "core/config/project_settings.h"

void StreamPeerTCP::accept_socket(Ref<NetSocket> p_sock, const NetSocket::Address &p_addr) {
	_sock = p_sock;
	_sock->set_blocking_enabled(false);

	timeout = OS::get_singleton()->get_ticks_msec() + (((uint64_t)GLOBAL_GET("network/limits/tcp/connect_timeout_seconds")) * 1000);
	status = STATUS_CONNECTED;

	peer_address = p_addr;
}

Error StreamPeerTCP::bind(int p_port, const IPAddress &p_host) {
	ERR_FAIL_COND_V(_sock.is_null(), ERR_UNAVAILABLE);
	ERR_FAIL_COND_V(_sock->is_open(), ERR_ALREADY_IN_USE);
	ERR_FAIL_COND_V_MSG(p_port < 0 || p_port > 65535, ERR_INVALID_PARAMETER, "The local port number must be between 0 and 65535 (inclusive).");

	IP::Type ip_type = p_host.is_ipv4() ? IP::TYPE_IPV4 : IP::TYPE_IPV6;
	if (p_host.is_wildcard()) {
		ip_type = IP::TYPE_ANY;
	}
	Error err = _sock->open(NetSocket::Family::INET, NetSocket::TYPE_TCP, ip_type);
	if (err != OK) {
		return err;
	}
	_sock->set_blocking_enabled(false);
	NetSocket::Address addr(p_host, p_port);
	return _sock->bind(addr);
}

Error StreamPeerTCP::connect_to_host(const IPAddress &p_host, int p_port) {
	ERR_FAIL_COND_V(_sock.is_null(), ERR_UNAVAILABLE);
	ERR_FAIL_COND_V(status != STATUS_NONE, ERR_ALREADY_IN_USE);
	ERR_FAIL_COND_V(!p_host.is_valid(), ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V_MSG(p_port < 1 || p_port > 65535, ERR_INVALID_PARAMETER, "The remote port number must be between 1 and 65535 (inclusive).");

	if (!_sock->is_open()) {
		IP::Type ip_type = p_host.is_ipv4() ? IP::TYPE_IPV4 : IP::TYPE_IPV6;
		Error err = _sock->open(NetSocket::Family::INET, NetSocket::TYPE_TCP, ip_type);
		if (err != OK) {
			return err;
		}
		_sock->set_blocking_enabled(false);
	}

	timeout = OS::get_singleton()->get_ticks_msec() + (((uint64_t)GLOBAL_GET("network/limits/tcp/connect_timeout_seconds")) * 1000);

	NetSocket::Address addr(p_host, p_port);
	Error err = _sock->connect_to_host(addr);

	if (err == OK) {
		status = STATUS_CONNECTED;
	} else if (err == ERR_BUSY) {
		status = STATUS_CONNECTING;
	} else {
		ERR_PRINT("Connection to remote host failed!");
		disconnect_from_host();
		return FAILED;
	}

	peer_address = addr;

	return OK;
}

void StreamPeerTCP::set_no_delay(bool p_enabled) {
	ERR_FAIL_COND(_sock.is_null() || !_sock->is_open());
	_sock->set_tcp_no_delay_enabled(p_enabled);
}

IPAddress StreamPeerTCP::get_connected_host() const {
	return peer_address.ip();
}

int StreamPeerTCP::get_connected_port() const {
	return peer_address.port();
}

int StreamPeerTCP::get_local_port() const {
	NetSocket::Address addr;
	_sock->get_socket_address(&addr);
	return addr.port();
}

Error StreamPeerTCP::_connect(const String &p_address, int p_port) {
	IPAddress ip;
	if (p_address.is_valid_ip_address()) {
		ip = p_address;
	} else {
		ip = IP::get_singleton()->resolve_hostname(p_address);
		if (!ip.is_valid()) {
			return ERR_CANT_RESOLVE;
		}
	}

	return connect_to_host(ip, p_port);
}

void StreamPeerTCP::_bind_methods() {
	ClassDB::bind_method(D_METHOD("bind", "port", "host"), &StreamPeerTCP::bind, DEFVAL("*"));
	ClassDB::bind_method(D_METHOD("connect_to_host", "host", "port"), &StreamPeerTCP::_connect);
	ClassDB::bind_method(D_METHOD("get_connected_host"), &StreamPeerTCP::get_connected_host);
	ClassDB::bind_method(D_METHOD("get_connected_port"), &StreamPeerTCP::get_connected_port);
	ClassDB::bind_method(D_METHOD("get_local_port"), &StreamPeerTCP::get_local_port);
	ClassDB::bind_method(D_METHOD("set_no_delay", "enabled"), &StreamPeerTCP::set_no_delay);
}
