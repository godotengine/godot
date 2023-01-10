/**************************************************************************/
/*  websocket_client.h                                                    */
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

#ifndef WEBSOCKET_CLIENT_H
#define WEBSOCKET_CLIENT_H

#include "core/crypto/crypto.h"
#include "core/error_list.h"
#include "websocket_multiplayer_peer.h"
#include "websocket_peer.h"

class WebSocketClient : public WebSocketMultiplayerPeer {
	GDCLASS(WebSocketClient, WebSocketMultiplayerPeer);
	GDCICLASS(WebSocketClient);

protected:
	Ref<WebSocketPeer> _peer;
	bool verify_ssl;
	Ref<X509Certificate> ssl_cert;

	static void _bind_methods();

public:
	Error connect_to_url(String p_url, const Vector<String> p_protocols = Vector<String>(), bool gd_mp_api = false, const Vector<String> p_custom_headers = Vector<String>());

	void set_verify_ssl_enabled(bool p_verify_ssl);
	bool is_verify_ssl_enabled() const;
	Ref<X509Certificate> get_trusted_ssl_certificate() const;
	void set_trusted_ssl_certificate(Ref<X509Certificate> p_cert);

	virtual void poll() = 0;
	virtual Error connect_to_host(String p_host, String p_path, uint16_t p_port, bool p_ssl, const Vector<String> p_protocol = Vector<String>(), const Vector<String> p_custom_headers = Vector<String>()) = 0;
	virtual void disconnect_from_host(int p_code = 1000, String p_reason = "") = 0;
	virtual IP_Address get_connected_host() const = 0;
	virtual uint16_t get_connected_port() const = 0;

	virtual bool is_server() const;
	virtual ConnectionStatus get_connection_status() const = 0;

	void _on_peer_packet();
	void _on_connect(String p_protocol);
	void _on_close_request(int p_code, String p_reason);
	void _on_disconnect(bool p_was_clean);
	void _on_error();

	virtual Error set_buffers(int p_in_buffer, int p_in_packets, int p_out_buffer, int p_out_packets) = 0;

	WebSocketClient();
	~WebSocketClient();
};

#endif // WEBSOCKET_CLIENT_H
