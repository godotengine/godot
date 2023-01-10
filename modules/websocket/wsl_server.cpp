/**************************************************************************/
/*  wsl_server.cpp                                                        */
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

#ifndef JAVASCRIPT_ENABLED

#include "wsl_server.h"
#include "core/os/os.h"
#include "core/project_settings.h"

WSLServer::PendingPeer::PendingPeer() {
	use_ssl = false;
	time = 0;
	has_request = false;
	response_sent = 0;
	req_pos = 0;
	memset(req_buf, 0, sizeof(req_buf));
}

bool WSLServer::PendingPeer::_parse_request(const Vector<String> p_protocols) {
	Vector<String> psa = String((char *)req_buf).split("\r\n");
	int len = psa.size();
	ERR_FAIL_COND_V_MSG(len < 4, false, "Not enough response headers, got: " + itos(len) + ", expected >= 4.");

	Vector<String> req = psa[0].split(" ", false);
	ERR_FAIL_COND_V_MSG(req.size() < 2, false, "Invalid protocol or status code.");

	// Wrong protocol
	ERR_FAIL_COND_V_MSG(req[0] != "GET" || req[2] != "HTTP/1.1", false, "Invalid method or HTTP version.");

	Map<String, String> headers;
	for (int i = 1; i < len; i++) {
		Vector<String> header = psa[i].split(":", false, 1);
		ERR_FAIL_COND_V_MSG(header.size() != 2, false, "Invalid header -> " + psa[i]);
		String name = header[0].to_lower();
		String value = header[1].strip_edges();
		if (headers.has(name)) {
			headers[name] += "," + value;
		} else {
			headers[name] = value;
		}
	}
#define _WSL_CHECK(NAME, VALUE)                                                         \
	ERR_FAIL_COND_V_MSG(!headers.has(NAME) || headers[NAME].to_lower() != VALUE, false, \
			"Missing or invalid header '" + String(NAME) + "'. Expected value '" + VALUE + "'.");
#define _WSL_CHECK_EX(NAME) \
	ERR_FAIL_COND_V_MSG(!headers.has(NAME), false, "Missing header '" + String(NAME) + "'.");
	_WSL_CHECK("upgrade", "websocket");
	_WSL_CHECK("sec-websocket-version", "13");
	_WSL_CHECK_EX("sec-websocket-key");
	_WSL_CHECK_EX("connection");
#undef _WSL_CHECK_EX
#undef _WSL_CHECK
	key = headers["sec-websocket-key"];
	if (headers.has("sec-websocket-protocol")) {
		Vector<String> protos = headers["sec-websocket-protocol"].split(",");
		for (int i = 0; i < protos.size(); i++) {
			String proto = protos[i].strip_edges();
			// Check if we have the given protocol
			for (int j = 0; j < p_protocols.size(); j++) {
				if (proto != p_protocols[j]) {
					continue;
				}
				protocol = proto;
				break;
			}
			// Found a protocol
			if (protocol != "") {
				break;
			}
		}
		if (protocol == "") { // Invalid protocol(s) requested
			return false;
		}
	} else if (p_protocols.size() > 0) { // No protocol requested, but we need one
		return false;
	}
	return true;
}

Error WSLServer::PendingPeer::do_handshake(const Vector<String> p_protocols, uint64_t p_timeout, const Vector<String> &p_extra_headers) {
	if (OS::get_singleton()->get_ticks_msec() - time > p_timeout) {
		print_verbose(vformat("WebSocket handshake timed out after %.3f seconds.", p_timeout * 0.001));
		return ERR_TIMEOUT;
	}

	if (use_ssl) {
		Ref<StreamPeerSSL> ssl = static_cast<Ref<StreamPeerSSL>>(connection);
		if (ssl.is_null()) {
			ERR_FAIL_V_MSG(ERR_BUG, "Couldn't get StreamPeerSSL for WebSocket handshake.");
		}
		ssl->poll();
		if (ssl->get_status() == StreamPeerSSL::STATUS_HANDSHAKING) {
			return ERR_BUSY;
		} else if (ssl->get_status() != StreamPeerSSL::STATUS_CONNECTED) {
			print_verbose(vformat("WebSocket SSL connection error during handshake (StreamPeerSSL status code %d).", ssl->get_status()));
			return FAILED;
		}
	}

	if (!has_request) {
		int read = 0;
		while (true) {
			ERR_FAIL_COND_V_MSG(req_pos >= WSL_MAX_HEADER_SIZE, ERR_OUT_OF_MEMORY, "WebSocket response headers are too big.");
			Error err = connection->get_partial_data(&req_buf[req_pos], 1, read);
			if (err != OK) { // Got an error
				print_verbose(vformat("WebSocket error while getting partial data (StreamPeer error code %d).", err));
				return FAILED;
			} else if (read != 1) { // Busy, wait next poll
				return ERR_BUSY;
			}
			char *r = (char *)req_buf;
			int l = req_pos;
			if (l > 3 && r[l] == '\n' && r[l - 1] == '\r' && r[l - 2] == '\n' && r[l - 3] == '\r') {
				r[l - 3] = '\0';
				if (!_parse_request(p_protocols)) {
					return FAILED;
				}
				String s = "HTTP/1.1 101 Switching Protocols\r\n";
				s += "Upgrade: websocket\r\n";
				s += "Connection: Upgrade\r\n";
				s += "Sec-WebSocket-Accept: " + WSLPeer::compute_key_response(key) + "\r\n";
				if (protocol != "") {
					s += "Sec-WebSocket-Protocol: " + protocol + "\r\n";
				}
				for (int i = 0; i < p_extra_headers.size(); i++) {
					s += p_extra_headers[i] + "\r\n";
				}
				s += "\r\n";
				response = s.utf8();
				has_request = true;
				break;
			}
			req_pos += 1;
		}
	}

	if (has_request && response_sent < response.size() - 1) {
		int sent = 0;
		Error err = connection->put_partial_data((const uint8_t *)response.get_data() + response_sent, response.size() - response_sent - 1, sent);
		if (err != OK) {
			print_verbose(vformat("WebSocket error while putting partial data (StreamPeer error code %d).", err));
			return err;
		}
		response_sent += sent;
	}

	if (response_sent < response.size() - 1) {
		return ERR_BUSY;
	}

	return OK;
}

void WSLServer::set_extra_headers(const Vector<String> &p_headers) {
	_extra_headers = p_headers;
}

Error WSLServer::listen(int p_port, const Vector<String> p_protocols, bool gd_mp_api) {
	ERR_FAIL_COND_V(is_listening(), ERR_ALREADY_IN_USE);

	_is_multiplayer = gd_mp_api;
	// Strip edges from protocols.
	_protocols.resize(p_protocols.size());
	String *pw = _protocols.ptrw();
	for (int i = 0; i < p_protocols.size(); i++) {
		pw[i] = p_protocols[i].strip_edges();
	}
	return _server->listen(p_port, bind_ip);
}

void WSLServer::poll() {
	List<int> remove_ids;
	for (Map<int, Ref<WebSocketPeer>>::Element *E = _peer_map.front(); E; E = E->next()) {
		Ref<WSLPeer> peer = (WSLPeer *)E->get().ptr();
		peer->poll();
		if (!peer->is_connected_to_host()) {
			_on_disconnect(E->key(), peer->close_code != -1);
			remove_ids.push_back(E->key());
		}
	}
	for (List<int>::Element *E = remove_ids.front(); E; E = E->next()) {
		_peer_map.erase(E->get());
	}
	remove_ids.clear();

	List<Ref<PendingPeer>> remove_peers;
	for (List<Ref<PendingPeer>>::Element *E = _pending.front(); E; E = E->next()) {
		Ref<PendingPeer> ppeer = E->get();
		Error err = ppeer->do_handshake(_protocols, handshake_timeout, _extra_headers);
		if (err == ERR_BUSY) {
			continue;
		} else if (err != OK) {
			remove_peers.push_back(ppeer);
			continue;
		}
		// Creating new peer
		int32_t id = _gen_unique_id();

		WSLPeer::PeerData *data = memnew(struct WSLPeer::PeerData);
		data->obj = this;
		data->conn = ppeer->connection;
		data->tcp = ppeer->tcp;
		data->is_server = true;
		data->id = id;

		Ref<WSLPeer> ws_peer = memnew(WSLPeer);
		ws_peer->make_context(data, _in_buf_size, _in_pkt_size, _out_buf_size, _out_pkt_size);
		ws_peer->set_no_delay(true);

		_peer_map[id] = ws_peer;
		remove_peers.push_back(ppeer);
		_on_connect(id, ppeer->protocol);
	}
	for (List<Ref<PendingPeer>>::Element *E = remove_peers.front(); E; E = E->next()) {
		_pending.erase(E->get());
	}
	remove_peers.clear();

	if (!_server->is_listening()) {
		return;
	}

	while (_server->is_connection_available()) {
		Ref<StreamPeerTCP> conn = _server->take_connection();
		if (is_refusing_new_connections()) {
			continue; // Conn will go out-of-scope and be closed.
		}

		Ref<PendingPeer> peer = memnew(PendingPeer);
		if (private_key.is_valid() && ssl_cert.is_valid()) {
			Ref<StreamPeerSSL> ssl = Ref<StreamPeerSSL>(StreamPeerSSL::create());
			ssl->set_blocking_handshake_enabled(false);
			ssl->accept_stream(conn, private_key, ssl_cert, ca_chain);
			peer->connection = ssl;
			peer->use_ssl = true;
		} else {
			peer->connection = conn;
		}
		peer->tcp = conn;
		peer->time = OS::get_singleton()->get_ticks_msec();
		_pending.push_back(peer);
	}
}

bool WSLServer::is_listening() const {
	return _server->is_listening();
}

int WSLServer::get_max_packet_size() const {
	return (1 << _out_buf_size) - PROTO_SIZE;
}

void WSLServer::stop() {
	_server->stop();
	for (Map<int, Ref<WebSocketPeer>>::Element *E = _peer_map.front(); E; E = E->next()) {
		Ref<WSLPeer> peer = (WSLPeer *)E->get().ptr();
		peer->close_now();
	}
	_pending.clear();
	_peer_map.clear();
	_protocols.clear();
}

bool WSLServer::has_peer(int p_id) const {
	return _peer_map.has(p_id);
}

Ref<WebSocketPeer> WSLServer::get_peer(int p_id) const {
	ERR_FAIL_COND_V(!has_peer(p_id), nullptr);
	return _peer_map[p_id];
}

IP_Address WSLServer::get_peer_address(int p_peer_id) const {
	ERR_FAIL_COND_V(!has_peer(p_peer_id), IP_Address());

	return _peer_map[p_peer_id]->get_connected_host();
}

int WSLServer::get_peer_port(int p_peer_id) const {
	ERR_FAIL_COND_V(!has_peer(p_peer_id), 0);

	return _peer_map[p_peer_id]->get_connected_port();
}

void WSLServer::disconnect_peer(int p_peer_id, int p_code, String p_reason) {
	ERR_FAIL_COND(!has_peer(p_peer_id));

	get_peer(p_peer_id)->close(p_code, p_reason);
}

Error WSLServer::set_buffers(int p_in_buffer, int p_in_packets, int p_out_buffer, int p_out_packets) {
	ERR_FAIL_COND_V_MSG(_server->is_listening(), FAILED, "Buffers sizes can only be set before listening or connecting.");

	_in_buf_size = nearest_shift(p_in_buffer - 1) + 10;
	_in_pkt_size = nearest_shift(p_in_packets - 1);
	_out_buf_size = nearest_shift(p_out_buffer - 1) + 10;
	_out_pkt_size = nearest_shift(p_out_packets - 1);
	return OK;
}

WSLServer::WSLServer() {
	_in_buf_size = nearest_shift((int)GLOBAL_GET(WSS_IN_BUF) - 1) + 10;
	_in_pkt_size = nearest_shift((int)GLOBAL_GET(WSS_IN_PKT) - 1);
	_out_buf_size = nearest_shift((int)GLOBAL_GET(WSS_OUT_BUF) - 1) + 10;
	_out_pkt_size = nearest_shift((int)GLOBAL_GET(WSS_OUT_PKT) - 1);
	_server.instance();
}

WSLServer::~WSLServer() {
	stop();
}

#endif // JAVASCRIPT_ENABLED
