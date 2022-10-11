/*************************************************************************/
/*  websocket_multiplayer_peer.cpp                                       */
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

#include "websocket_multiplayer_peer.h"

#include "core/os/os.h"

WebSocketMultiplayerPeer::WebSocketMultiplayerPeer() {
	peer_config = Ref<WebSocketPeer>(WebSocketPeer::create());
}

WebSocketMultiplayerPeer::~WebSocketMultiplayerPeer() {
	_clear();
}

Ref<WebSocketPeer> WebSocketMultiplayerPeer::_create_peer() {
	Ref<WebSocketPeer> peer = Ref<WebSocketPeer>(WebSocketPeer::create());
	peer->set_supported_protocols(get_supported_protocols());
	peer->set_handshake_headers(get_handshake_headers());
	peer->set_inbound_buffer_size(get_inbound_buffer_size());
	peer->set_outbound_buffer_size(get_outbound_buffer_size());
	peer->set_max_queued_packets(get_max_queued_packets());
	return peer;
}

void WebSocketMultiplayerPeer::_clear() {
	connection_status = CONNECTION_DISCONNECTED;
	unique_id = 0;
	peers_map.clear();
	use_tls = false;
	tcp_server.unref();
	pending_peers.clear();
	tls_certificate.unref();
	tls_key.unref();
	if (current_packet.data != nullptr) {
		memfree(current_packet.data);
	}

	for (Packet &E : incoming_packets) {
		memfree(E.data);
		E.data = nullptr;
	}

	incoming_packets.clear();
}

void WebSocketMultiplayerPeer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("create_client", "url", "verify_tls", "tls_certificate"), &WebSocketMultiplayerPeer::create_client, DEFVAL(true), DEFVAL(Ref<X509Certificate>()));
	ClassDB::bind_method(D_METHOD("create_server", "port", "bind_address", "tls_key", "tls_certificate"), &WebSocketMultiplayerPeer::create_server, DEFVAL("*"), DEFVAL(Ref<CryptoKey>()), DEFVAL(Ref<X509Certificate>()));
	ClassDB::bind_method(D_METHOD("close"), &WebSocketMultiplayerPeer::close);

	ClassDB::bind_method(D_METHOD("get_peer", "peer_id"), &WebSocketMultiplayerPeer::get_peer);
	ClassDB::bind_method(D_METHOD("get_peer_address", "id"), &WebSocketMultiplayerPeer::get_peer_address);
	ClassDB::bind_method(D_METHOD("get_peer_port", "id"), &WebSocketMultiplayerPeer::get_peer_port);
	ClassDB::bind_method(D_METHOD("disconnect_peer", "id", "code", "reason"), &WebSocketMultiplayerPeer::disconnect_peer, DEFVAL(1000), DEFVAL(""));

	ClassDB::bind_method(D_METHOD("get_supported_protocols"), &WebSocketMultiplayerPeer::get_supported_protocols);
	ClassDB::bind_method(D_METHOD("set_supported_protocols", "protocols"), &WebSocketMultiplayerPeer::set_supported_protocols);

	ClassDB::bind_method(D_METHOD("get_handshake_headers"), &WebSocketMultiplayerPeer::get_handshake_headers);
	ClassDB::bind_method(D_METHOD("set_handshake_headers", "protocols"), &WebSocketMultiplayerPeer::set_handshake_headers);

	ClassDB::bind_method(D_METHOD("get_inbound_buffer_size"), &WebSocketMultiplayerPeer::get_inbound_buffer_size);
	ClassDB::bind_method(D_METHOD("set_inbound_buffer_size", "buffer_size"), &WebSocketMultiplayerPeer::set_inbound_buffer_size);

	ClassDB::bind_method(D_METHOD("get_outbound_buffer_size"), &WebSocketMultiplayerPeer::get_outbound_buffer_size);
	ClassDB::bind_method(D_METHOD("set_outbound_buffer_size", "buffer_size"), &WebSocketMultiplayerPeer::set_outbound_buffer_size);

	ClassDB::bind_method(D_METHOD("get_handshake_timeout"), &WebSocketMultiplayerPeer::get_handshake_timeout);
	ClassDB::bind_method(D_METHOD("set_handshake_timeout", "timeout"), &WebSocketMultiplayerPeer::set_handshake_timeout);

	ClassDB::bind_method(D_METHOD("set_max_queued_packets", "max_queued_packets"), &WebSocketMultiplayerPeer::set_max_queued_packets);
	ClassDB::bind_method(D_METHOD("get_max_queued_packets"), &WebSocketMultiplayerPeer::get_max_queued_packets);

	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "supported_protocols"), "set_supported_protocols", "get_supported_protocols");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "handshake_headers"), "set_handshake_headers", "get_handshake_headers");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "inbound_buffer_size"), "set_inbound_buffer_size", "get_inbound_buffer_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "outbound_buffer_size"), "set_outbound_buffer_size", "get_outbound_buffer_size");

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "handshake_timeout"), "set_handshake_timeout", "get_handshake_timeout");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_queued_packets"), "set_max_queued_packets", "get_max_queued_packets");
}

//
// PacketPeer
//
int WebSocketMultiplayerPeer::get_available_packet_count() const {
	return incoming_packets.size();
}

Error WebSocketMultiplayerPeer::get_packet(const uint8_t **r_buffer, int &r_buffer_size) {
	ERR_FAIL_COND_V(get_connection_status() != CONNECTION_CONNECTED, ERR_UNCONFIGURED);

	r_buffer_size = 0;

	if (current_packet.data != nullptr) {
		memfree(current_packet.data);
		current_packet.data = nullptr;
	}

	ERR_FAIL_COND_V(incoming_packets.size() == 0, ERR_UNAVAILABLE);

	current_packet = incoming_packets.front()->get();
	incoming_packets.pop_front();

	*r_buffer = current_packet.data;
	r_buffer_size = current_packet.size;

	return OK;
}

Error WebSocketMultiplayerPeer::put_packet(const uint8_t *p_buffer, int p_buffer_size) {
	ERR_FAIL_COND_V(get_connection_status() != CONNECTION_CONNECTED, ERR_UNCONFIGURED);

	Vector<uint8_t> buffer = _make_pkt(SYS_NONE, get_unique_id(), target_peer, p_buffer, p_buffer_size);

	if (is_server()) {
		return _server_relay(1, target_peer, &(buffer.ptr()[0]), buffer.size());
	} else {
		return get_peer(1)->put_packet(&(buffer.ptr()[0]), buffer.size());
	}
}

//
// MultiplayerPeer
//
void WebSocketMultiplayerPeer::set_target_peer(int p_target_peer) {
	target_peer = p_target_peer;
}

int WebSocketMultiplayerPeer::get_packet_peer() const {
	ERR_FAIL_COND_V(incoming_packets.size() == 0, 1);

	return incoming_packets.front()->get().source;
}

int WebSocketMultiplayerPeer::get_unique_id() const {
	return unique_id;
}

int WebSocketMultiplayerPeer::get_max_packet_size() const {
	return get_outbound_buffer_size() - PROTO_SIZE;
}

Error WebSocketMultiplayerPeer::create_server(int p_port, IPAddress p_bind_ip, Ref<CryptoKey> p_tls_key, Ref<X509Certificate> p_tls_certificate) {
	ERR_FAIL_COND_V(get_connection_status() != CONNECTION_DISCONNECTED, ERR_ALREADY_IN_USE);
	_clear();
	tcp_server.instantiate();
	Error err = tcp_server->listen(p_port, p_bind_ip);
	if (err != OK) {
		tcp_server.unref();
		return err;
	}
	unique_id = 1;
	connection_status = CONNECTION_CONNECTED;
	// TLS config
	tls_key = p_tls_key;
	tls_certificate = p_tls_certificate;
	if (tls_key.is_valid() && tls_certificate.is_valid()) {
		use_tls = true;
	}
	return OK;
}

Error WebSocketMultiplayerPeer::create_client(const String &p_url, bool p_verify_tls, Ref<X509Certificate> p_tls_certificate) {
	ERR_FAIL_COND_V(get_connection_status() != CONNECTION_DISCONNECTED, ERR_ALREADY_IN_USE);
	_clear();
	Ref<WebSocketPeer> peer = _create_peer();
	Error err = peer->connect_to_url(p_url, p_verify_tls, p_tls_certificate);
	if (err != OK) {
		return err;
	}
	PendingPeer pending;
	pending.time = OS::get_singleton()->get_ticks_msec();
	pending_peers[1] = pending;
	peers_map[1] = peer;
	connection_status = CONNECTION_CONNECTING;
	return OK;
}

bool WebSocketMultiplayerPeer::is_server() const {
	return tcp_server.is_valid();
}

void WebSocketMultiplayerPeer::_poll_client() {
	ERR_FAIL_COND(connection_status == CONNECTION_DISCONNECTED); // Bug.
	ERR_FAIL_COND(!peers_map.has(1) || peers_map[1].is_null()); // Bug.
	Ref<WebSocketPeer> peer = peers_map[1];
	peer->poll(); // Update state and fetch packets.
	WebSocketPeer::State ready_state = peer->get_ready_state();
	if (ready_state == WebSocketPeer::STATE_OPEN) {
		while (peer->get_available_packet_count()) {
			_process_multiplayer(peer, 1);
		}
	} else if (peer->get_ready_state() == WebSocketPeer::STATE_CLOSED) {
		if (connection_status == CONNECTION_CONNECTED) {
			emit_signal(SNAME("server_disconnected"));
		} else {
			emit_signal(SNAME("connection_failed"));
		}
		_clear();
		return;
	}
	if (connection_status == CONNECTION_CONNECTING) {
		// Still connecting
		ERR_FAIL_COND(!pending_peers.has(1)); // Bug.
		if (OS::get_singleton()->get_ticks_msec() - pending_peers[1].time > handshake_timeout) {
			print_verbose(vformat("WebSocket handshake timed out after %.3f seconds.", handshake_timeout * 0.001));
			emit_signal(SNAME("connection_failed"));
			_clear();
			return;
		}
	}
}

void WebSocketMultiplayerPeer::_poll_server() {
	ERR_FAIL_COND(connection_status != CONNECTION_CONNECTED); // Bug.
	ERR_FAIL_COND(tcp_server.is_null() || !tcp_server->is_listening()); // Bug.

	// Accept new connections.
	if (!is_refusing_new_connections() && tcp_server->is_connection_available()) {
		PendingPeer peer;
		peer.time = OS::get_singleton()->get_ticks_msec();
		peer.tcp = tcp_server->take_connection();
		peer.connection = peer.tcp;
		pending_peers[generate_unique_id()] = peer;
	}

	// Process pending peers.
	HashSet<int> to_remove;
	for (KeyValue<int, PendingPeer> &E : pending_peers) {
		PendingPeer &peer = E.value;
		int id = E.key;

		if (OS::get_singleton()->get_ticks_msec() - peer.time > handshake_timeout) {
			print_verbose(vformat("WebSocket handshake timed out after %.3f seconds.", handshake_timeout * 0.001));
			to_remove.insert(id);
			continue;
		}

		if (peer.ws.is_valid()) {
			peer.ws->poll();
			WebSocketPeer::State state = peer.ws->get_ready_state();
			if (state == WebSocketPeer::STATE_OPEN) {
				// Connected.
				to_remove.insert(id);
				if (is_refusing_new_connections()) {
					// The user does not want new connections, dropping it.
					continue;
				}
				peers_map[id] = peer.ws;
				_send_ack(peer.ws, id);
				emit_signal("peer_connected", id);
				continue;
			} else if (state == WebSocketPeer::STATE_CONNECTING) {
				continue; // Still connecting.
			}
			to_remove.insert(id); // Error.
			continue;
		}
		if (peer.tcp->get_status() != StreamPeerTCP::STATUS_CONNECTED) {
			to_remove.insert(id); // Error.
			continue;
		}
		if (!use_tls) {
			peer.ws = _create_peer();
			peer.ws->accept_stream(peer.tcp);
			continue;
		} else {
			if (peer.connection == peer.tcp) {
				Ref<StreamPeerTLS> tls = Ref<StreamPeerTLS>(StreamPeerTLS::create());
				Error err = tls->accept_stream(peer.tcp, tls_key, tls_certificate);
				if (err != OK) {
					to_remove.insert(id);
					continue;
				}
			}
			Ref<StreamPeerTLS> tls = static_cast<Ref<StreamPeerTLS>>(peer.connection);
			tls->poll();
			if (tls->get_status() == StreamPeerTLS::STATUS_CONNECTED) {
				peer.ws = _create_peer();
				peer.ws->accept_stream(peer.connection);
				continue;
			} else if (tls->get_status() == StreamPeerTLS::STATUS_HANDSHAKING) {
				// Still connecting.
				continue;
			} else {
				// Error
				to_remove.insert(id);
			}
		}
	}

	// Remove disconnected pending peers.
	for (const int &pid : to_remove) {
		pending_peers.erase(pid);
	}
	to_remove.clear();

	// Process connected peers.
	for (KeyValue<int, Ref<WebSocketPeer>> &E : peers_map) {
		Ref<WebSocketPeer> ws = E.value;
		int id = E.key;
		ws->poll();
		if (ws->get_ready_state() != WebSocketPeer::STATE_OPEN) {
			to_remove.insert(id); // Disconnected.
			continue;
		}
		// Fetch packets
		int pkts = ws->get_available_packet_count();
		while (pkts && ws->get_ready_state() == WebSocketPeer::STATE_OPEN) {
			_process_multiplayer(ws, id);
			pkts--;
		}
	}

	// Remove disconnected peers.
	for (const int &pid : to_remove) {
		emit_signal(SNAME("peer_disconnected"), pid);
		peers_map.erase(pid);
	}
}

void WebSocketMultiplayerPeer::poll() {
	if (connection_status == CONNECTION_DISCONNECTED) {
		return;
	}
	if (is_server()) {
		_poll_server();
	} else {
		_poll_client();
	}
}

MultiplayerPeer::ConnectionStatus WebSocketMultiplayerPeer::get_connection_status() const {
	return connection_status;
}

Ref<WebSocketPeer> WebSocketMultiplayerPeer::get_peer(int p_id) const {
	ERR_FAIL_COND_V(!peers_map.has(p_id), Ref<WebSocketPeer>());
	return peers_map[p_id];
}

void WebSocketMultiplayerPeer::_send_sys(Ref<WebSocketPeer> p_peer, uint8_t p_type, int32_t p_peer_id) {
	ERR_FAIL_COND(!p_peer.is_valid());
	ERR_FAIL_COND(p_peer->get_ready_state() != WebSocketPeer::STATE_OPEN);

	Vector<uint8_t> message = _make_pkt(p_type, 1, 0, (uint8_t *)&p_peer_id, 4);
	p_peer->put_packet(&(message.ptr()[0]), message.size());
}

Vector<uint8_t> WebSocketMultiplayerPeer::_make_pkt(uint8_t p_type, int32_t p_from, int32_t p_to, const uint8_t *p_data, uint32_t p_data_size) {
	Vector<uint8_t> out;
	out.resize(PROTO_SIZE + p_data_size);

	uint8_t *w = out.ptrw();
	memcpy(&w[0], &p_type, 1);
	memcpy(&w[1], &p_from, 4);
	memcpy(&w[5], &p_to, 4);
	memcpy(&w[PROTO_SIZE], p_data, p_data_size);

	return out;
}

void WebSocketMultiplayerPeer::_send_ack(Ref<WebSocketPeer> p_peer, int32_t p_peer_id) {
	ERR_FAIL_COND(p_peer.is_null());
	// First of all, confirm the ID!
	_send_sys(p_peer, SYS_ID, p_peer_id);

	// Then send the server peer (which will trigger connection_succeded in client)
	_send_sys(p_peer, SYS_ADD, 1);

	for (const KeyValue<int, Ref<WebSocketPeer>> &E : peers_map) {
		ERR_CONTINUE(E.value.is_null());

		int32_t id = E.key;
		if (p_peer_id == id) {
			continue; // Skip the newly added peer (already confirmed)
		}

		// Send new peer to others
		_send_sys(E.value, SYS_ADD, p_peer_id);
		// Send others to new peer
		_send_sys(E.value, SYS_ADD, id);
	}
}

void WebSocketMultiplayerPeer::_send_del(int32_t p_peer_id) {
	for (const KeyValue<int, Ref<WebSocketPeer>> &E : peers_map) {
		int32_t id = E.key;
		if (p_peer_id != id) {
			_send_sys(E.value, SYS_DEL, p_peer_id);
		}
	}
}

void WebSocketMultiplayerPeer::_store_pkt(int32_t p_source, int32_t p_dest, const uint8_t *p_data, uint32_t p_data_size) {
	Packet packet;
	packet.data = (uint8_t *)memalloc(p_data_size);
	packet.size = p_data_size;
	packet.source = p_source;
	packet.destination = p_dest;
	memcpy(packet.data, &p_data[PROTO_SIZE], p_data_size);
	incoming_packets.push_back(packet);
}

Error WebSocketMultiplayerPeer::_server_relay(int32_t p_from, int32_t p_to, const uint8_t *p_buffer, uint32_t p_buffer_size) {
	if (p_to == 1) {
		return OK; // Will not send to self

	} else if (p_to == 0) {
		for (KeyValue<int, Ref<WebSocketPeer>> &E : peers_map) {
			if (E.key != p_from) {
				E.value->put_packet(p_buffer, p_buffer_size);
			}
		}
		return OK; // Sent to all but sender

	} else if (p_to < 0) {
		for (KeyValue<int, Ref<WebSocketPeer>> &E : peers_map) {
			if (E.key != p_from && E.key != -p_to) {
				E.value->put_packet(p_buffer, p_buffer_size);
			}
		}
		return OK; // Sent to all but sender and excluded

	} else {
		ERR_FAIL_COND_V(p_to == p_from, FAILED);

		Ref<WebSocketPeer> peer_to = get_peer(p_to);
		ERR_FAIL_COND_V(peer_to.is_null(), FAILED);

		return peer_to->put_packet(p_buffer, p_buffer_size); // Sending to specific peer
	}
}

void WebSocketMultiplayerPeer::_process_multiplayer(Ref<WebSocketPeer> p_peer, uint32_t p_peer_id) {
	ERR_FAIL_COND(!p_peer.is_valid());

	const uint8_t *in_buffer;
	int size = 0;
	int data_size = 0;

	Error err = p_peer->get_packet(&in_buffer, size);

	ERR_FAIL_COND(err != OK);
	ERR_FAIL_COND(size < PROTO_SIZE);

	data_size = size - PROTO_SIZE;

	uint8_t type = 0;
	uint32_t from = 0;
	int32_t to = 0;
	memcpy(&type, in_buffer, 1);
	memcpy(&from, &in_buffer[1], 4);
	memcpy(&to, &in_buffer[5], 4);

	if (is_server()) { // Server can resend

		ERR_FAIL_COND(type != SYS_NONE); // Only server sends sys messages
		ERR_FAIL_COND(from != p_peer_id); // Someone is cheating

		if (to == 1) {
			// This is for the server
			_store_pkt(from, to, in_buffer, data_size);

		} else if (to == 0) {
			// Broadcast, for us too
			_store_pkt(from, to, in_buffer, data_size);

		} else if (to < -1) {
			// All but one, for us if not excluded
			_store_pkt(from, to, in_buffer, data_size);
		}
		// Relay if needed (i.e. "to" includes a peer that is not the server)
		_server_relay(from, to, in_buffer, size);

	} else {
		if (type == SYS_NONE) {
			// Payload message
			_store_pkt(from, to, in_buffer, data_size);
			return;
		}

		// System message
		ERR_FAIL_COND(data_size < 4);
		int id = 0;
		memcpy(&id, &in_buffer[PROTO_SIZE], 4);

		switch (type) {
			case SYS_ADD: // Add peer
				if (id != 1) {
					peers_map[id] = Ref<WebSocketPeer>();
				} else {
					pending_peers.clear();
					connection_status = CONNECTION_CONNECTED;
				}
				emit_signal(SNAME("peer_connected"), id);
				if (id == 1) { // We just connected to the server
					emit_signal(SNAME("connection_succeeded"));
				}
				break;

			case SYS_DEL: // Remove peer
				emit_signal(SNAME("peer_disconnected"), id);
				peers_map.erase(id);
				break;
			case SYS_ID: // Hello, server assigned ID
				unique_id = id;
				break;
			default:
				ERR_FAIL_MSG("Invalid multiplayer message.");
				break;
		}
	}
}

void WebSocketMultiplayerPeer::set_supported_protocols(const Vector<String> &p_protocols) {
	peer_config->set_supported_protocols(p_protocols);
}

Vector<String> WebSocketMultiplayerPeer::get_supported_protocols() const {
	return peer_config->get_supported_protocols();
}

void WebSocketMultiplayerPeer::set_handshake_headers(const Vector<String> &p_headers) {
	peer_config->set_handshake_headers(p_headers);
}

Vector<String> WebSocketMultiplayerPeer::get_handshake_headers() const {
	return peer_config->get_handshake_headers();
}

void WebSocketMultiplayerPeer::set_outbound_buffer_size(int p_buffer_size) {
	peer_config->set_outbound_buffer_size(p_buffer_size);
}

int WebSocketMultiplayerPeer::get_outbound_buffer_size() const {
	return peer_config->get_outbound_buffer_size();
}

void WebSocketMultiplayerPeer::set_inbound_buffer_size(int p_buffer_size) {
	peer_config->set_inbound_buffer_size(p_buffer_size);
}

int WebSocketMultiplayerPeer::get_inbound_buffer_size() const {
	return peer_config->get_inbound_buffer_size();
}

void WebSocketMultiplayerPeer::set_max_queued_packets(int p_max_queued_packets) {
	peer_config->set_max_queued_packets(p_max_queued_packets);
}

int WebSocketMultiplayerPeer::get_max_queued_packets() const {
	return peer_config->get_max_queued_packets();
}

float WebSocketMultiplayerPeer::get_handshake_timeout() const {
	return handshake_timeout / 1000.0;
}

void WebSocketMultiplayerPeer::set_handshake_timeout(float p_timeout) {
	ERR_FAIL_COND(p_timeout <= 0.0);
	handshake_timeout = p_timeout * 1000;
}

IPAddress WebSocketMultiplayerPeer::get_peer_address(int p_peer_id) const {
	ERR_FAIL_COND_V(!peers_map.has(p_peer_id), IPAddress());
	return peers_map[p_peer_id]->get_connected_host();
}

int WebSocketMultiplayerPeer::get_peer_port(int p_peer_id) const {
	ERR_FAIL_COND_V(!peers_map.has(p_peer_id), 0);
	return peers_map[p_peer_id]->get_connected_port();
}

void WebSocketMultiplayerPeer::disconnect_peer(int p_peer_id, int p_code, String p_reason) {
	ERR_FAIL_COND(!peers_map.has(p_peer_id));
	peers_map[p_peer_id]->close(p_code, p_reason);
}

void WebSocketMultiplayerPeer::close() {
	_clear();
}
