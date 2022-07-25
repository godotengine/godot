/*************************************************************************/
/*  emws_client.h                                                        */
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

#ifndef EMWS_CLIENT_H
#define EMWS_CLIENT_H

#ifdef JAVASCRIPT_ENABLED

#include "core/error_list.h"
#include "emws_peer.h"
#include "websocket_client.h"

class EMWSClient : public WebSocketClient {
	GDCIIMPL(EMWSClient, WebSocketClient);

private:
	int _js_id;
	bool _is_connecting;
	int _in_buf_size;
	int _in_pkt_size;
	int _out_buf_size;

	static void _esws_on_connect(void *obj, char *proto);
	static void _esws_on_message(void *obj, const uint8_t *p_data, int p_data_size, int p_is_string);
	static void _esws_on_error(void *obj);
	static void _esws_on_close(void *obj, int code, const char *reason, int was_clean);

public:
	Error set_buffers(int p_in_buffer, int p_in_packets, int p_out_buffer, int p_out_packets);
	Error connect_to_host(String p_host, String p_path, uint16_t p_port, bool p_ssl, const Vector<String> p_protocol = Vector<String>(), const Vector<String> p_custom_headers = Vector<String>());
	Ref<WebSocketPeer> get_peer(int p_peer_id) const;
	void disconnect_from_host(int p_code = 1000, String p_reason = "");
	IP_Address get_connected_host() const;
	uint16_t get_connected_port() const;
	virtual ConnectionStatus get_connection_status() const;
	int get_max_packet_size() const;
	virtual void poll();
	EMWSClient();
	~EMWSClient();
};

#endif // JAVASCRIPT_ENABLED

#endif // EMWS_CLIENT_H
