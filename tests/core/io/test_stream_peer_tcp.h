/**************************************************************************/
/*  test_stream_peer_tcp.h                                                */
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

#pragma once

#include "core/io/net_socket.h"
#include "core/io/stream_peer_tcp.h"
#include "tests/test_macros.h"

namespace TestStreamPeerTCP {

class MockNetSocket : public NetSocket {
public:
	static void make_default();

	virtual Error open(Type p_type, IP::Type &ip_type) override;
	virtual void close() override;
	virtual Error bind(IPAddress p_addr, uint16_t p_port) override;
	virtual Error listen(int p_max_pending) override;
	virtual Error connect_to_host(IPAddress p_addr, uint16_t p_port) override;
	virtual Error poll(PollType p_type, int timeout) const override;
	virtual Error recv(uint8_t *p_buffer, int p_len, int &r_read) override;
	virtual Error recvfrom(uint8_t *p_buffer, int p_len, int &r_read, IPAddress &r_ip, uint16_t &r_port, bool p_peek = false) override;
	virtual Error send(const uint8_t *p_buffer, int p_len, int &r_sent) override;
	virtual Error sendto(const uint8_t *p_buffer, int p_len, int &r_sent, IPAddress p_ip, uint16_t p_port) override;
	virtual Ref<NetSocket> accept(IPAddress &r_ip, uint16_t &r_port) override;

	virtual bool is_open() const override;
	virtual int get_available_bytes() const override;
	virtual Error get_socket_address(IPAddress *r_ip, uint16_t *r_port) const override;

	virtual Error set_broadcasting_enabled(bool p_enabled) override;
	virtual void set_blocking_enabled(bool p_enabled) override;
	virtual void set_ipv6_only_enabled(bool p_enabled) override;
	virtual void set_tcp_no_delay_enabled(bool p_enabled) override;
	virtual void set_reuse_address_enabled(bool p_enabled) override;
	virtual Error join_multicast_group(const IPAddress &p_multi_address, const String &p_if_name) override;
	virtual Error leave_multicast_group(const IPAddress &p_multi_address, const String &p_if_name) override;

	IPAddress host_ip = IPAddress();
	uint16_t host_port = 0;
	IPAddress dest_ip = IPAddress();
	uint16_t dest_port = 0;
	bool blocking_enabled = true;

	// Helper methods for testing.
	void _set_available_bytes(int p_available_bytes);
	void _set_send_data(uint8_t *p_sent_data);
	void _set_recv_data(uint8_t *p_recv_data);

	MockNetSocket();
	~MockNetSocket() override;

protected:
	static NetSocket *_create_func();

private:
	bool _is_open = false;
	int _available_bytes = 0;
	uint8_t *_sent_data;
	uint8_t *_recv_data;
};

NetSocket *MockNetSocket::_create_func() {
	return memnew(MockNetSocket);
}

void MockNetSocket::_set_available_bytes(int p_available_bytes) {
	_available_bytes = p_available_bytes;
}

void MockNetSocket::_set_send_data(uint8_t *p_sent_data) {
	_sent_data = p_sent_data;
}

void MockNetSocket::_set_recv_data(uint8_t *p_recv_data) {
	_recv_data = p_recv_data;
}

void MockNetSocket::make_default() {
	_create = _create_func;
}

Error MockNetSocket::open(Type p_type, IP::Type &ip_type) {
	_is_open = true;
	return OK;
}
void MockNetSocket::close() {
	_is_open = false;
}
Error MockNetSocket::bind(IPAddress p_addr, uint16_t p_port) {
	host_ip = p_addr;
	host_port = p_port;
	return OK;
}
Error MockNetSocket::listen(int p_max_pending) {
	return OK;
}
Error MockNetSocket::connect_to_host(IPAddress p_addr, uint16_t p_port) {
	dest_ip = p_addr;
	dest_port = p_port;
	return OK;
}
Error MockNetSocket::poll(PollType p_type, int timeout) const {
	return OK;
}
Error MockNetSocket::recv(uint8_t *p_buffer, int p_len, int &r_read) {
	// Receives one byte of _recv_data on each invocation.
	p_buffer[0] = _recv_data[0];
	_recv_data += 1;
	r_read = 1;
	return OK;
}
Error MockNetSocket::recvfrom(uint8_t *p_buffer, int p_len, int &r_read, IPAddress &r_ip, uint16_t &r_port, bool p_peek) {
	return OK;
}
Error MockNetSocket::send(const uint8_t *p_buffer, int p_len, int &r_sent) {
	// Sends one byte to _sent_data on each invocation.
	_sent_data[0] = p_buffer[0];
	_sent_data += 1;
	r_sent = 1;
	return OK;
}
Error MockNetSocket::sendto(const uint8_t *p_buffer, int p_len, int &r_sent, IPAddress p_ip, uint16_t p_port) {
	return OK;
}
Ref<NetSocket> MockNetSocket::accept(IPAddress &r_ip, uint16_t &r_port) {
	return this;
}

bool MockNetSocket::is_open() const {
	return _is_open;
}
int MockNetSocket::get_available_bytes() const {
	return _available_bytes;
}
Error MockNetSocket::get_socket_address(IPAddress *r_ip, uint16_t *r_port) const {
	return OK;
}

Error MockNetSocket::set_broadcasting_enabled(bool p_enabled) {
	return OK;
}
void MockNetSocket::set_blocking_enabled(bool p_enabled) {
	blocking_enabled = p_enabled;
}
void MockNetSocket::set_ipv6_only_enabled(bool p_enabled) {}
void MockNetSocket::set_tcp_no_delay_enabled(bool p_enabled) {}
void MockNetSocket::set_reuse_address_enabled(bool p_enabled) {}
Error MockNetSocket::join_multicast_group(const IPAddress &p_multi_address, const String &p_if_name) {
	return OK;
}
Error MockNetSocket::leave_multicast_group(const IPAddress &p_multi_address, const String &p_if_name) {
	return OK;
}

MockNetSocket::MockNetSocket() {}
MockNetSocket::~MockNetSocket() {}

TEST_CASE("[StreamPeerTCP] basics") {
	Ref<MockNetSocket> ns = memnew(MockNetSocket);
	Ref<StreamPeerTCP> spt = memnew(StreamPeerTCP);
	IPAddress peer_ip = IPAddress("127.0.1.1");
	int peer_port = 5678;
	spt->accept_socket(ns, peer_ip, peer_port);
	REQUIRE(ns->blocking_enabled == false);

	IPAddress bind_ip = IPAddress("127.0.0.1");

	SUBCASE("invalid port numbers returns an Error") {
		ERR_PRINT_OFF;
		Error negative_ret = spt->bind(-901, bind_ip);
		REQUIRE(negative_ret != OK);

		Error too_high_ret = spt->bind(70000, bind_ip);
		REQUIRE(too_high_ret != OK);
		ERR_PRINT_ON;
	}

	SUBCASE("bind calls open and bind on NetSocket") {
		int bind_port = 7890;
		Error bind_ret = spt->bind(bind_port, bind_ip);
		REQUIRE(bind_ret == OK);
		REQUIRE(ns->is_open() == true);
		REQUIRE(ns->host_ip == bind_ip);
		REQUIRE(ns->host_port == bind_port);
	}

	SUBCASE("disconnect_from_host closes NetSocket") {
		spt->disconnect_from_host();
		REQUIRE(ns->is_open() == false);
	}
}

TEST_CASE("[StreamPeerTCP] poll") {
	Ref<MockNetSocket> ns = memnew(MockNetSocket);
	Ref<StreamPeerTCP> spt = memnew(StreamPeerTCP);
	IPAddress peer_ip = IPAddress("127.2.2.2");
	int peer_port = 45878;
	spt->accept_socket(ns, peer_ip, peer_port);
	IPAddress bind_ip = IPAddress("127.0.0.1");
	int bind_port = 9043;
	Error bind_ret = spt->bind(bind_port, bind_ip);
	REQUIRE(bind_ret == OK);
	ns->_set_available_bytes(100);

	SUBCASE("Unconnected status causes connect_to_host") {
		Error poll_ret = spt->poll();
		REQUIRE(poll_ret == OK);
		REQUIRE(spt->get_status() == StreamPeerTCP::STATUS_CONNECTED);
		REQUIRE(ns->is_open() == true);
	}

	SUBCASE("FIN causes disconnect_from_host") {
		// This is the condition for FIN.
		ns->_set_available_bytes(0);
		Error fin_ret = spt->poll();
		REQUIRE(fin_ret == OK);
		REQUIRE(ns->is_open() == false);
	}
}

TEST_CASE("[StreamPeerTCP] data") {
	Ref<MockNetSocket> ns = memnew(MockNetSocket);
	Ref<StreamPeerTCP> spt = memnew(StreamPeerTCP);
	IPAddress peer_ip = IPAddress("127.5.4.3");
	int peer_port = 8908;
	spt->accept_socket(ns, peer_ip, peer_port);
	IPAddress bind_ip = IPAddress("127.0.0.1");
	int bind_port = 2039;
	Error bind_ret = spt->bind(bind_port, bind_ip);
	REQUIRE(bind_ret == OK);

	SUBCASE("put_data an ascii-encoded stream") {
		const String expected = "hello, world";
		Vector<uint8_t> send_data;
		send_data.resize_zeroed(expected.length());
		ns->_set_send_data(send_data.ptrw());
		REQUIRE_EQ(spt->put_data(expected.to_ascii_buffer().ptrw(), expected.length()), Error::OK);
		CHECK_EQ(send_data, expected.to_ascii_buffer());
	}

	SUBCASE("get_data an ascii-encoded stream") {
		const String expected = "I - too - say hello!";
		Vector<uint8_t> recv_data = expected.to_ascii_buffer();
		ns->_set_recv_data(recv_data.ptrw());
		Vector<uint8_t> in_data;
		in_data.resize(expected.length());
		REQUIRE_EQ(spt->get_data(in_data.ptrw(), expected.length()), Error::OK);
		CHECK_EQ(in_data, expected.to_ascii_buffer());
	}
}

} // namespace TestStreamPeerTCP
