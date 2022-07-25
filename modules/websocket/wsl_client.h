/*************************************************************************/
/*  wsl_client.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef WSL_CLIENT_H
#define WSL_CLIENT_H

#ifndef JAVASCRIPT_ENABLED

#include "core/error_list.h"
#include "core/io/stream_peer_ssl.h"
#include "core/io/stream_peer_tcp.h"
#include "websocket_client.h"
#include "wsl_peer.h"
#include "wslay/wslay.h"

class WSLClient : public WebSocketClient {
	GDCIIMPL(WSLClient, WebSocketClient);

private:
	int _in_buf_size;
	int _in_pkt_size;
	int _out_buf_size;
	int _out_pkt_size;

	Ref<WSLPeer> _peer;
	Ref<StreamPeerTCP> _tcp;
	Ref<StreamPeer> _connection;

	CharString _request;
	int _requested;

	uint8_t _resp_buf[WSL_MAX_HEADER_SIZE];
	int _resp_pos;

	String _response;

	String _key;
	String _host;
	uint16_t _port;
	Array _ip_candidates;
	Vector<String> _protocols;
	bool _use_ssl = false;
	IP::ResolverID _resolver_id = IP::RESOLVER_INVALID_ID;

	void _do_handshake();
	bool _verify_headers(String &r_protocol);

public:
	Error set_buffers(int p_in_buffer, int p_in_packets, int p_out_buffer, int p_out_packets);
	Error connect_to_host(String p_host, String p_path, uint16_t p_port, bool p_ssl, const Vector<String> p_protocol = Vector<String>(), const Vector<String> p_custom_headers = Vector<String>());
	int get_max_packet_size() const;
	Ref<WebSocketPeer> get_peer(int p_peer_id) const;
	void disconnect_from_host(int p_code = 1000, String p_reason = "");
	IP_Address get_connected_host() const;
	uint16_t get_connected_port() const;
	virtual ConnectionStatus get_connection_status() const;
	virtual void poll();

	WSLClient();
	~WSLClient();
};

#endif // JAVASCRIPT_ENABLED

#endif // WSL_CLIENT_H
