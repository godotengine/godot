/*************************************************************************/
/*  emws_client.cpp                                                      */
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

#ifdef JAVASCRIPT_ENABLED

#include "emws_client.h"
#include "core/io/ip.h"
#include "core/project_settings.h"
#include "emscripten.h"

void EMWSClient::_esws_on_connect(void *obj, char *proto) {
	EMWSClient *client = static_cast<EMWSClient *>(obj);
	client->_is_connecting = false;
	client->_on_connect(String(proto));
}

void EMWSClient::_esws_on_message(void *obj, const uint8_t *p_data, int p_data_size, int p_is_string) {
	EMWSClient *client = static_cast<EMWSClient *>(obj);

	Error err = static_cast<EMWSPeer *>(*client->get_peer(1))->read_msg(p_data, p_data_size, p_is_string == 1);
	if (err == OK)
		client->_on_peer_packet();
}

void EMWSClient::_esws_on_error(void *obj) {
	EMWSClient *client = static_cast<EMWSClient *>(obj);
	client->_is_connecting = false;
	client->_on_error();
}

void EMWSClient::_esws_on_close(void *obj, int code, const char *reason, int was_clean) {
	EMWSClient *client = static_cast<EMWSClient *>(obj);
	client->_on_close_request(code, String(reason));
	client->_is_connecting = false;
	client->disconnect_from_host();
	client->_on_disconnect(was_clean != 0);
}

Error EMWSClient::connect_to_host(String p_host, String p_path, uint16_t p_port, bool p_ssl, const Vector<String> p_protocols, const Vector<String> p_custom_headers) {
	if (_js_id) {
		godot_js_websocket_destroy(_js_id);
		_js_id = 0;
	}

	String proto_string;
	for (int i = 0; i < p_protocols.size(); i++) {
		if (i != 0)
			proto_string += ",";
		proto_string += p_protocols[i];
	}

	String str = "ws://";

	if (p_custom_headers.size()) {
		WARN_PRINT_ONCE("Custom headers are not supported in in HTML5 platform.");
	}
	if (p_ssl) {
		str = "wss://";
		if (ssl_cert.is_valid()) {
			WARN_PRINT_ONCE("Custom SSL certificate is not supported in HTML5 platform.");
		}
	}
	str += p_host + ":" + itos(p_port) + p_path;
	_is_connecting = true;

	_js_id = godot_js_websocket_create(this, str.utf8().get_data(), proto_string.utf8().get_data(), &_esws_on_connect, &_esws_on_message, &_esws_on_error, &_esws_on_close);
	if (!_js_id) {
		return FAILED;
	}

	static_cast<Ref<EMWSPeer>>(_peer)->set_sock(_js_id, _in_buf_size, _in_pkt_size, _out_buf_size);

	return OK;
}

void EMWSClient::poll() {
}

Ref<WebSocketPeer> EMWSClient::get_peer(int p_peer_id) const {
	return _peer;
}

NetworkedMultiplayerPeer::ConnectionStatus EMWSClient::get_connection_status() const {
	if (_peer->is_connected_to_host()) {
		if (_is_connecting)
			return CONNECTION_CONNECTING;
		return CONNECTION_CONNECTED;
	}

	return CONNECTION_DISCONNECTED;
}

void EMWSClient::disconnect_from_host(int p_code, String p_reason) {
	_peer->close(p_code, p_reason);
}

IP_Address EMWSClient::get_connected_host() const {
	ERR_FAIL_V_MSG(IP_Address(), "Not supported in HTML5 export.");
}

uint16_t EMWSClient::get_connected_port() const {
	ERR_FAIL_V_MSG(0, "Not supported in HTML5 export.");
}

int EMWSClient::get_max_packet_size() const {
	return (1 << _in_buf_size) - PROTO_SIZE;
}

Error EMWSClient::set_buffers(int p_in_buffer, int p_in_packets, int p_out_buffer, int p_out_packets) {
	_in_buf_size = nearest_shift(p_in_buffer - 1) + 10;
	_in_pkt_size = nearest_shift(p_in_packets - 1);
	_out_buf_size = nearest_shift(p_out_buffer - 1) + 10;
	return OK;
}

EMWSClient::EMWSClient() {
	_in_buf_size = nearest_shift((int)GLOBAL_GET(WSC_IN_BUF) - 1) + 10;
	_in_pkt_size = nearest_shift((int)GLOBAL_GET(WSC_IN_PKT) - 1);
	_out_buf_size = nearest_shift((int)GLOBAL_GET(WSC_OUT_BUF) - 1) + 10;
	_is_connecting = false;
	_peer = Ref<EMWSPeer>(memnew(EMWSPeer));
	_js_id = 0;
}

EMWSClient::~EMWSClient() {
	disconnect_from_host();
	_peer = Ref<EMWSPeer>();
	if (_js_id) {
		godot_js_websocket_destroy(_js_id);
	}
}

#endif // JAVASCRIPT_ENABLED
