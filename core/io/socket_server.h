/**************************************************************************/
/*  socket_server.h                                                       */
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
#include "core/io/stream_peer_socket.h"

class SocketServer : public RefCounted {
	GDCLASS(SocketServer, RefCounted);

protected:
	enum {
		MAX_PENDING_CONNECTIONS = 8,
	};

	Ref<NetSocket> _sock;
	static void _bind_methods();

	Error _listen(const NetSocket::Address &p_addr);

	template <typename T>
	Ref<T> _take_connection() {
		Ref<T> conn;
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
		conn->accept_socket(ns, addr);
		return conn;
	}

public:
	bool is_listening() const;
	bool is_connection_available() const;
	virtual Ref<StreamPeerSocket> take_socket_connection() = 0;

	void stop(); // Stop listening

	SocketServer();
	~SocketServer();
};
