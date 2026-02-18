/**************************************************************************/
/*  net_socket_unix.h                                                     */
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

#if defined(UNIX_ENABLED) && !defined(UNIX_SOCKET_UNAVAILABLE)

#include "core/io/net_socket.h"

#include <sys/socket.h>
#include <sys/un.h>

class NetSocketUnix : public NetSocket {
	GDSOFTCLASS(NetSocketUnix, NetSocket);

private:
	int _sock = -1;
	Family _family = Family::NONE;
	IP::Type _ip_type = IP::TYPE_NONE;
	bool _is_stream = false;
	CharString _unix_path;
	// If this is Family::UNIX,
	bool _unlink_on_close = false;

	enum NetError {
		ERR_NET_WOULD_BLOCK,
		ERR_NET_IS_CONNECTED,
		ERR_NET_IN_PROGRESS,
		ERR_NET_ADDRESS_INVALID_OR_UNAVAILABLE,
		ERR_NET_UNAUTHORIZED,
		ERR_NET_BUFFER_TOO_SMALL,
		ERR_NET_OTHER,
	};

	NetError _get_socket_error() const;
	void _set_socket(int p_sock, IP::Type p_ip_type, bool p_is_stream);
	_FORCE_INLINE_ Error _change_multicast_group(IPAddress p_ip, String p_if_name, bool p_add);
	_FORCE_INLINE_ void _set_close_exec_enabled(bool p_enabled);

protected:
	static NetSocket *_create_func();

	bool _can_use_ip(const IPAddress &p_ip, const bool p_for_bind) const;
	bool _can_use_path(const CharString &p_path) const;

	Error _inet_open(Type p_sock_type, IP::Type &r_ip_type);
	Error _inet_bind(IPAddress p_addr, uint16_t p_port);
	Error _inet_connect_to_host(IPAddress p_addr, uint16_t p_port);
	Error _inet_get_socket_address(IPAddress *r_ip, uint16_t *r_port) const;
	Ref<NetSocket> _inet_accept(IPAddress &r_ip, uint16_t &r_port);

	static socklen_t _unix_set_sockaddr(struct sockaddr_un *p_addr, const CharString &p_path);
	Error _unix_open();
	Error _unix_bind(const CharString &p_path);
	Error _unix_connect_to_host(const CharString &p_path);
	Ref<NetSocket> _unix_accept();

public:
	static void make_default();
	static void cleanup();
	static void _set_ip_port(struct sockaddr_storage *p_addr, IPAddress *r_ip, uint16_t *r_port);
	static size_t _set_addr_storage(struct sockaddr_storage *p_addr, const IPAddress &p_ip, uint16_t p_port, IP::Type p_ip_type);

	virtual Error open(Family p_family, Type p_sock_type, IP::Type &r_ip_type) override;
	virtual void close() override;
	virtual Error bind(Address p_addr) override;
	virtual Error listen(int p_max_pending) override;
	virtual Error connect_to_host(Address p_addr) override;
	virtual Error poll(PollType p_type, int timeout) const override;
	virtual Error recv(uint8_t *p_buffer, int p_len, int &r_read) override;
	virtual Error recvfrom(uint8_t *p_buffer, int p_len, int &r_read, IPAddress &r_ip, uint16_t &r_port, bool p_peek = false) override;
	virtual Error send(const uint8_t *p_buffer, int p_len, int &r_sent) override;
	virtual Error sendto(const uint8_t *p_buffer, int p_len, int &r_sent, IPAddress p_ip, uint16_t p_port) override;
	virtual Ref<NetSocket> accept(Address &r_addr) override;

	virtual bool is_open() const override;
	virtual int get_available_bytes() const override;
	virtual Error get_socket_address(Address *r_addr) const override;

	virtual Error set_broadcasting_enabled(bool p_enabled) override;
	virtual void set_blocking_enabled(bool p_enabled) override;
	virtual void set_ipv6_only_enabled(bool p_enabled) override;
	virtual void set_tcp_no_delay_enabled(bool p_enabled) override;
	virtual void set_reuse_address_enabled(bool p_enabled) override;
	virtual void set_reuse_port_enabled(bool p_enabled) override;

	virtual Error join_multicast_group(const IPAddress &p_multi_address, const String &p_if_name) override;
	virtual Error leave_multicast_group(const IPAddress &p_multi_address, const String &p_if_name) override;

	NetSocketUnix();
	~NetSocketUnix() override;
};

#endif // UNIX_ENABLED && !UNIX_SOCKET_UNAVAILABLE
