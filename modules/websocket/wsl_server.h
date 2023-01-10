/**************************************************************************/
/*  wsl_server.h                                                          */
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

#ifndef WSL_SERVER_H
#define WSL_SERVER_H

#ifndef JAVASCRIPT_ENABLED

#include "websocket_server.h"
#include "wsl_peer.h"

#include "core/io/stream_peer_ssl.h"
#include "core/io/stream_peer_tcp.h"
#include "core/io/tcp_server.h"

class WSLServer : public WebSocketServer {
	GDCIIMPL(WSLServer, WebSocketServer);

private:
	class PendingPeer : public Reference {
	private:
		bool _parse_request(const Vector<String> p_protocols);

	public:
		Ref<StreamPeerTCP> tcp;
		Ref<StreamPeer> connection;
		bool use_ssl;

		uint64_t time;
		uint8_t req_buf[WSL_MAX_HEADER_SIZE];
		int req_pos;
		String key;
		String protocol;
		bool has_request;
		CharString response;
		int response_sent;

		PendingPeer();

		Error do_handshake(const Vector<String> p_protocols, uint64_t p_timeout, const Vector<String> &p_extra_headers);
	};

	int _in_buf_size;
	int _in_pkt_size;
	int _out_buf_size;
	int _out_pkt_size;

	List<Ref<PendingPeer>> _pending;
	Ref<TCP_Server> _server;
	Vector<String> _protocols;
	Vector<String> _extra_headers;

public:
	Error set_buffers(int p_in_buffer, int p_in_packets, int p_out_buffer, int p_out_packets);
	void set_extra_headers(const Vector<String> &p_headers);
	Error listen(int p_port, const Vector<String> p_protocols = Vector<String>(), bool gd_mp_api = false);
	void stop();
	bool is_listening() const;
	int get_max_packet_size() const;
	bool has_peer(int p_id) const;
	Ref<WebSocketPeer> get_peer(int p_id) const;
	IP_Address get_peer_address(int p_peer_id) const;
	int get_peer_port(int p_peer_id) const;
	void disconnect_peer(int p_peer_id, int p_code = 1000, String p_reason = "");
	virtual void poll();

	WSLServer();
	~WSLServer();
};

#endif // JAVASCRIPT_ENABLED

#endif // WSL_SERVER_H
