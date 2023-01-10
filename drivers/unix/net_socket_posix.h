/**************************************************************************/
/*  net_socket_posix.h                                                    */
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

#ifndef NET_SOCKET_POSIX_H
#define NET_SOCKET_POSIX_H

#include "core/io/net_socket.h"

#if defined(WINDOWS_ENABLED)
#include <winsock2.h>
#include <ws2tcpip.h>
#define SOCKET_TYPE SOCKET

#else
#include <sys/socket.h>
#define SOCKET_TYPE int

#endif

class NetSocketPosix : public NetSocket {
private:
	SOCKET_TYPE _sock;
	IP::Type _ip_type;
	bool _is_stream;

	enum NetError {
		ERR_NET_WOULD_BLOCK,
		ERR_NET_IS_CONNECTED,
		ERR_NET_IN_PROGRESS,
		ERR_NET_OTHER
	};

	NetError _get_socket_error() const;
	void _set_socket(SOCKET_TYPE p_sock, IP::Type p_ip_type, bool p_is_stream);
	_FORCE_INLINE_ Error _change_multicast_group(IP_Address p_ip, String p_if_name, bool p_add);
	_FORCE_INLINE_ void _set_close_exec_enabled(bool p_enabled);

protected:
	static NetSocket *_create_func();

	bool _can_use_ip(const IP_Address &p_ip, const bool p_for_bind) const;

public:
	static void make_default();
	static void cleanup();
	static void _set_ip_port(struct sockaddr_storage *p_addr, IP_Address &r_ip, uint16_t &r_port);
	static size_t _set_addr_storage(struct sockaddr_storage *p_addr, const IP_Address &p_ip, uint16_t p_port, IP::Type p_ip_type);

	virtual Error open(Type p_sock_type, IP::Type &ip_type);
	virtual void close();
	virtual Error bind(IP_Address p_addr, uint16_t p_port);
	virtual Error listen(int p_max_pending);
	virtual Error connect_to_host(IP_Address p_host, uint16_t p_port);
	virtual Error poll(PollType p_type, int timeout) const;
	virtual Error recv(uint8_t *p_buffer, int p_len, int &r_read);
	virtual Error recvfrom(uint8_t *p_buffer, int p_len, int &r_read, IP_Address &r_ip, uint16_t &r_port, bool p_peek = false);
	virtual Error send(const uint8_t *p_buffer, int p_len, int &r_sent);
	virtual Error sendto(const uint8_t *p_buffer, int p_len, int &r_sent, IP_Address p_ip, uint16_t p_port);
	virtual Ref<NetSocket> accept(IP_Address &r_ip, uint16_t &r_port);

	virtual bool is_open() const;
	virtual int get_available_bytes() const;

	virtual Error set_broadcasting_enabled(bool p_enabled);
	virtual void set_blocking_enabled(bool p_enabled);
	virtual void set_ipv6_only_enabled(bool p_enabled);
	virtual void set_tcp_no_delay_enabled(bool p_enabled);
	virtual void set_reuse_address_enabled(bool p_enabled);
	virtual void set_reuse_port_enabled(bool p_enabled);
	virtual Error join_multicast_group(const IP_Address &p_multi_address, String p_if_name);
	virtual Error leave_multicast_group(const IP_Address &p_multi_address, String p_if_name);

	NetSocketPosix();
	~NetSocketPosix();
};

#endif // NET_SOCKET_POSIX_H
