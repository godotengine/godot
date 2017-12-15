/*************************************************************************/
/*  stream_peer_winsock.h                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifdef WINDOWS_ENABLED

#ifndef STREAM_PEER_TCP_WINSOCK_H
#define STREAM_PEER_TCP_WINSOCK_H

#include "error_list.h"

#include "core/io/ip_address.h"
#include "core/io/stream_peer_tcp.h"

class StreamPeerTCPWinsock : public StreamPeerTCP {

protected:
	mutable Status status;
	IP::Type sock_type;

	int sockfd;

	Error _block(int p_sockfd, bool p_read, bool p_write) const;

	Error _poll_connection() const;

	IP_Address peer_host;
	int peer_port;

	Error write(const uint8_t *p_data, int p_bytes, int &r_sent, bool p_block);
	Error read(uint8_t *p_buffer, int p_bytes, int &r_received, bool p_block);

	static StreamPeerTCP *_create();

public:
	virtual Error connect_to_host(const IP_Address &p_host, uint16_t p_port);

	virtual Error put_data(const uint8_t *p_data, int p_bytes);
	virtual Error put_partial_data(const uint8_t *p_data, int p_bytes, int &r_sent);

	virtual Error get_data(uint8_t *p_buffer, int p_bytes);
	virtual Error get_partial_data(uint8_t *p_buffer, int p_bytes, int &r_received);

	virtual int get_available_bytes() const;

	void set_socket(int p_sockfd, IP_Address p_host, int p_port, IP::Type p_sock_type);

	virtual IP_Address get_connected_host() const;
	virtual uint16_t get_connected_port() const;

	virtual bool is_connected_to_host() const;
	virtual Status get_status() const;
	virtual void disconnect_from_host();

	static void make_default();
	static void cleanup();

	virtual void set_nodelay(bool p_enabled);

	StreamPeerTCPWinsock();
	~StreamPeerTCPWinsock();
};

#endif // STREAM_PEER_TCP_WINSOCK_H

#endif
