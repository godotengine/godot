/*************************************************************************/
/*  lws_server.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "lws_server.h"
#include "core/os/os.h"

Error LWSServer::listen(int p_port, PoolVector<String> p_protocols, bool gd_mp_api) {

	ERR_FAIL_COND_V(context != NULL, FAILED);

	_is_multiplayer = gd_mp_api;

	struct lws_context_creation_info info;
	memset(&info, 0, sizeof info);

	// Prepare lws protocol structs
	_lws_make_protocols(this, &LWSServer::_lws_gd_callback, p_protocols, &_lws_ref);

	info.port = p_port;
	info.user = _lws_ref;
	info.protocols = _lws_ref->lws_structs;
	info.gid = -1;
	info.uid = -1;
	//info.ws_ping_pong_interval = 5;

	context = lws_create_context(&info);

	if (context == NULL) {
		_lws_free_ref(_lws_ref);
		_lws_ref = NULL;
		ERR_EXPLAIN("Unable to create LWS context");
		ERR_FAIL_V(FAILED);
	}

	return OK;
}

bool LWSServer::is_listening() const {
	return context != NULL;
}

int LWSServer::_handle_cb(struct lws *wsi, enum lws_callback_reasons reason, void *user, void *in, size_t len) {

	LWSPeer::PeerData *peer_data = (LWSPeer::PeerData *)user;

	switch (reason) {
		case LWS_CALLBACK_HTTP:
			// no http for now
			// closing immediately returning -1;
			return -1;

		case LWS_CALLBACK_FILTER_PROTOCOL_CONNECTION:
			// check header here?
			break;

		case LWS_CALLBACK_ESTABLISHED: {
			int32_t id = _gen_unique_id();

			Ref<LWSPeer> peer = Ref<LWSPeer>(memnew(LWSPeer));
			peer->set_wsi(wsi);
			_peer_map[id] = peer;

			peer_data->peer_id = id;
			peer_data->force_close = false;
			peer_data->clean_close = false;
			_on_connect(id, lws_get_protocol(wsi)->name);
			break;
		}

		case LWS_CALLBACK_WS_PEER_INITIATED_CLOSE: {
			if (peer_data == NULL)
				return 0;

			int32_t id = peer_data->peer_id;
			if (_peer_map.has(id)) {
				int code;
				Ref<LWSPeer> peer = _peer_map[id];
				String reason = peer->get_close_reason(in, len, code);
				peer_data->clean_close = true;
				_on_close_request(id, code, reason);
			}
			return 0;
		}

		case LWS_CALLBACK_CLOSED: {
			if (peer_data == NULL)
				return 0;
			int32_t id = peer_data->peer_id;
			bool clean = peer_data->clean_close;
			if (_peer_map.has(id)) {
				_peer_map[id]->close();
				_peer_map.erase(id);
			}
			_on_disconnect(id, clean);
			return 0; // we can end here
		}

		case LWS_CALLBACK_RECEIVE: {
			int32_t id = peer_data->peer_id;
			if (_peer_map.has(id)) {
				static_cast<Ref<LWSPeer> >(_peer_map[id])->read_wsi(in, len);
				if (_peer_map[id]->get_available_packet_count() > 0)
					_on_peer_packet(id);
			}
			break;
		}

		case LWS_CALLBACK_SERVER_WRITEABLE: {
			int id = peer_data->peer_id;
			if (peer_data->force_close) {
				if (_peer_map.has(id)) {
					Ref<LWSPeer> peer = _peer_map[id];
					peer->send_close_status(wsi);
				}
				return -1;
			}

			if (_peer_map.has(id))
				static_cast<Ref<LWSPeer> >(_peer_map[id])->write_wsi();
			break;
		}

		default:
			break;
	}

	return 0;
}

void LWSServer::stop() {
	if (context == NULL)
		return;

	_peer_map.clear();
	destroy_context();
	context = NULL;
}

bool LWSServer::has_peer(int p_id) const {
	return _peer_map.has(p_id);
}

Ref<WebSocketPeer> LWSServer::get_peer(int p_id) const {
	ERR_FAIL_COND_V(!has_peer(p_id), NULL);
	return _peer_map[p_id];
}

IP_Address LWSServer::get_peer_address(int p_peer_id) const {
	ERR_FAIL_COND_V(!has_peer(p_peer_id), IP_Address());

	return _peer_map[p_peer_id]->get_connected_host();
}

int LWSServer::get_peer_port(int p_peer_id) const {
	ERR_FAIL_COND_V(!has_peer(p_peer_id), 0);

	return _peer_map[p_peer_id]->get_connected_port();
}

void LWSServer::disconnect_peer(int p_peer_id, int p_code, String p_reason) {
	ERR_FAIL_COND(!has_peer(p_peer_id));

	get_peer(p_peer_id)->close(p_code, p_reason);
}

LWSServer::LWSServer() {
	context = NULL;
	_lws_ref = NULL;
}

LWSServer::~LWSServer() {
	invalidate_lws_ref(); // we do not want any more callbacks
	stop();
}

#endif // JAVASCRIPT_ENABLED
