/*************************************************************************/
/*  lws_client.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef JAVASCRIPT_ENABLED

#include "lws_client.h"
#include "core/io/ip.h"
#include "core/io/stream_peer_ssl.h"
#include "core/project_settings.h"
#if defined(LWS_OPENSSL_SUPPORT)
// Not openssl, just the mbedtls wrapper
#include "openssl/ssl.h"
#endif

Error LWSClient::connect_to_host(String p_host, String p_path, uint16_t p_port, bool p_ssl, PoolVector<String> p_protocols) {

	ERR_FAIL_COND_V(context != NULL, FAILED);

	IP_Address addr;

	if (!p_host.is_valid_ip_address()) {
		addr = IP::get_singleton()->resolve_hostname(p_host);
	} else {
		addr = p_host;
	}

	ERR_FAIL_COND_V(!addr.is_valid(), ERR_INVALID_PARAMETER);

	// Prepare protocols
	_lws_make_protocols(this, &LWSClient::_lws_gd_callback, p_protocols, &_lws_ref);

	// Init lws client
	struct lws_context_creation_info info;
	struct lws_client_connect_info i;

	memset(&i, 0, sizeof i);
	memset(&info, 0, sizeof info);

	info.port = CONTEXT_PORT_NO_LISTEN;
	info.protocols = _lws_ref->lws_structs;
	info.gid = -1;
	info.uid = -1;
	//info.ws_ping_pong_interval = 5;
	info.user = _lws_ref;
#if defined(LWS_OPENSSL_SUPPORT)
	info.options |= LWS_SERVER_OPTION_DO_SSL_GLOBAL_INIT;
#endif
	context = lws_create_context(&info);

	if (context == NULL) {
		_lws_free_ref(_lws_ref);
		_lws_ref = NULL;
		ERR_EXPLAIN("Unable to create lws context");
		ERR_FAIL_V(FAILED);
	}

	i.context = context;
	if (p_protocols.size() > 0)
		i.protocol = _lws_ref->lws_names;
	else
		i.protocol = NULL;

	if (p_ssl) {
		i.ssl_connection = LCCSCF_USE_SSL;
		if (!verify_ssl)
			i.ssl_connection |= LCCSCF_ALLOW_SELFSIGNED;
	} else {
		i.ssl_connection = 0;
	}

	// These CharStrings needs to survive till we call lws_client_connect_via_info
	CharString addr_ch = ((String)addr).ascii();
	CharString host_ch = p_host.utf8();
	CharString path_ch = p_path.utf8();
	i.address = addr_ch.get_data();
	i.host = host_ch.get_data();
	i.path = path_ch.get_data();
	i.port = p_port;

	lws_client_connect_via_info(&i);

	return OK;
};

int LWSClient::get_max_packet_size() const {
	return (1 << _out_buf_size) - PROTO_SIZE;
}

void LWSClient::poll() {

	_lws_poll();
}

int LWSClient::_handle_cb(struct lws *wsi, enum lws_callback_reasons reason, void *user, void *in, size_t len) {

	Ref<LWSPeer> peer = static_cast<Ref<LWSPeer> >(_peer);
	LWSPeer::PeerData *peer_data = (LWSPeer::PeerData *)user;

	switch (reason) {
#if defined(LWS_OPENSSL_SUPPORT)
		case LWS_CALLBACK_OPENSSL_LOAD_EXTRA_CLIENT_VERIFY_CERTS: {
			PoolByteArray arr = StreamPeerSSL::get_project_cert_array();
			if (arr.size() > 0)
				SSL_CTX_add_client_CA((SSL_CTX *)user, d2i_X509(NULL, &arr.read()[0], arr.size()));
			else if (verify_ssl)
				WARN_PRINTS("No CA cert specified in project settings, SSL will not work");
		} break;
#endif
		case LWS_CALLBACK_CLIENT_ESTABLISHED:
			peer->set_wsi(wsi, _in_buf_size, _in_pkt_size, _out_buf_size, _out_pkt_size);
			peer_data->peer_id = 0;
			peer_data->force_close = false;
			peer_data->clean_close = false;
			_on_connect(lws_get_protocol(wsi)->name);
			break;

		case LWS_CALLBACK_CLIENT_CONNECTION_ERROR:
			_on_error();
			destroy_context();
			return -1; // We should close the connection (would probably happen anyway)

		case LWS_CALLBACK_WS_PEER_INITIATED_CLOSE: {
			int code;
			String reason2 = peer->get_close_reason(in, len, code);
			peer_data->clean_close = true;
			_on_close_request(code, reason2);
			return 0;
		}

		case LWS_CALLBACK_CLIENT_CLOSED:
			peer->close();
			destroy_context();
			_on_disconnect(peer_data->clean_close);
			return 0; // We can end here

		case LWS_CALLBACK_CLIENT_RECEIVE:
			peer->read_wsi(in, len);
			if (peer->get_available_packet_count() > 0)
				_on_peer_packet();
			break;

		case LWS_CALLBACK_CLIENT_WRITEABLE:
			if (peer_data->force_close) {
				peer->send_close_status(wsi);
				return -1;
			}

			peer->write_wsi();
			break;

		default:
			break;
	}

	return 0;
}

Ref<WebSocketPeer> LWSClient::get_peer(int p_peer_id) const {

	return _peer;
}

NetworkedMultiplayerPeer::ConnectionStatus LWSClient::get_connection_status() const {

	if (context == NULL)
		return CONNECTION_DISCONNECTED;

	if (_peer->is_connected_to_host())
		return CONNECTION_CONNECTED;

	return CONNECTION_CONNECTING;
}

void LWSClient::disconnect_from_host(int p_code, String p_reason) {

	if (context == NULL)
		return;

	_peer->close(p_code, p_reason);
};

IP_Address LWSClient::get_connected_host() const {

	return IP_Address();
};

uint16_t LWSClient::get_connected_port() const {

	return 1025;
};

LWSClient::LWSClient() {
	_in_buf_size = nearest_shift((int)GLOBAL_GET(WSC_IN_BUF) - 1) + 10;
	_in_pkt_size = nearest_shift((int)GLOBAL_GET(WSC_IN_PKT) - 1);
	_out_buf_size = nearest_shift((int)GLOBAL_GET(WSC_OUT_BUF) - 1) + 10;
	_out_pkt_size = nearest_shift((int)GLOBAL_GET(WSC_OUT_PKT) - 1);

	context = NULL;
	_lws_ref = NULL;
	_peer = Ref<LWSPeer>(memnew(LWSPeer));
};

LWSClient::~LWSClient() {

	invalidate_lws_ref(); // We do not want any more callback
	disconnect_from_host();
	destroy_context();
	_peer = Ref<LWSPeer>();
};

#endif // JAVASCRIPT_ENABLED
