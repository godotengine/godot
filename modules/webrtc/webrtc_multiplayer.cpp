/*************************************************************************/
/*  webrtc_multiplayer.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "webrtc_multiplayer.h"

#include "core/io/marshalls.h"
#include "core/os/os.h"

void WebRTCMultiplayer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("initialize", "peer_id", "server_compatibility"), &WebRTCMultiplayer::initialize, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("add_peer", "peer", "peer_id", "unreliable_lifetime"), &WebRTCMultiplayer::add_peer, DEFVAL(1));
	ClassDB::bind_method(D_METHOD("remove_peer", "peer_id"), &WebRTCMultiplayer::remove_peer);
	ClassDB::bind_method(D_METHOD("has_peer", "peer_id"), &WebRTCMultiplayer::has_peer);
	ClassDB::bind_method(D_METHOD("get_peer", "peer_id"), &WebRTCMultiplayer::get_peer);
	ClassDB::bind_method(D_METHOD("get_peers"), &WebRTCMultiplayer::get_peers);
	ClassDB::bind_method(D_METHOD("close"), &WebRTCMultiplayer::close);
}

void WebRTCMultiplayer::set_transfer_mode(TransferMode p_mode) {
	transfer_mode = p_mode;
}

NetworkedMultiplayerPeer::TransferMode WebRTCMultiplayer::get_transfer_mode() const {
	return transfer_mode;
}

void WebRTCMultiplayer::set_target_peer(int p_peer_id) {
	target_peer = p_peer_id;
}

/* Returns the ID of the NetworkedMultiplayerPeer who sent the most recent packet: */
int WebRTCMultiplayer::get_packet_peer() const {
	return next_packet_peer;
}

bool WebRTCMultiplayer::is_server() const {
	return unique_id == TARGET_PEER_SERVER;
}

void WebRTCMultiplayer::poll() {
	if (peer_map.size() == 0) {
		return;
	}

	List<int> remove;
	List<int> add;
	for (Map<int, Ref<ConnectedPeer>>::Element *E = peer_map.front(); E; E = E->next()) {
		Ref<ConnectedPeer> peer = E->get();
		peer->connection->poll();
		// Check peer state
		switch (peer->connection->get_connection_state()) {
			case WebRTCPeerConnection::STATE_NEW:
			case WebRTCPeerConnection::STATE_CONNECTING:
				// Go to next peer, not ready yet.
				continue;
			case WebRTCPeerConnection::STATE_CONNECTED:
				// Good to go, go ahead and check channel state.
				break;
			default:
				// Peer is closed or in error state. Got to next peer.
				remove.push_back(E->key());
				continue;
		}
		// Check channels state
		int ready = 0;
		for (List<Ref<WebRTCDataChannel>>::Element *C = peer->channels.front(); C && C->get().is_valid(); C = C->next()) {
			Ref<WebRTCDataChannel> ch = C->get();
			switch (ch->get_ready_state()) {
				case WebRTCDataChannel::STATE_CONNECTING:
					continue;
				case WebRTCDataChannel::STATE_OPEN:
					ready++;
					continue;
				default:
					// Channel was closed or in error state, remove peer id.
					remove.push_back(E->key());
			}
			// We got a closed channel break out, the peer will be removed.
			break;
		}
		// This peer has newly connected, and all channels are now open.
		if (ready == peer->channels.size() && !peer->connected) {
			peer->connected = true;
			add.push_back(E->key());
		}
	}
	// Remove disconnected peers
	for (List<int>::Element *E = remove.front(); E; E = E->next()) {
		remove_peer(E->get());
		if (next_packet_peer == E->get()) {
			next_packet_peer = 0;
		}
	}
	// Signal newly connected peers
	for (List<int>::Element *E = add.front(); E; E = E->next()) {
		// Already connected to server: simply notify new peer.
		// NOTE: Mesh is always connected.
		if (connection_status == CONNECTION_CONNECTED) {
			emit_signal("peer_connected", E->get());
		}

		// Server emulation mode suppresses peer_conencted until server connects.
		if (server_compat && E->get() == TARGET_PEER_SERVER) {
			// Server connected.
			connection_status = CONNECTION_CONNECTED;
			emit_signal("peer_connected", TARGET_PEER_SERVER);
			emit_signal("connection_succeeded");
			// Notify of all previously connected peers
			for (Map<int, Ref<ConnectedPeer>>::Element *F = peer_map.front(); F; F = F->next()) {
				if (F->key() != 1 && F->get()->connected) {
					emit_signal("peer_connected", F->key());
				}
			}
			break; // Because we already notified of all newly added peers.
		}
	}
	// Fetch next packet
	if (next_packet_peer == 0) {
		_find_next_peer();
	}
}

void WebRTCMultiplayer::_find_next_peer() {
	Map<int, Ref<ConnectedPeer>>::Element *E = peer_map.find(next_packet_peer);
	if (E) {
		E = E->next();
	}
	// After last.
	while (E) {
		for (List<Ref<WebRTCDataChannel>>::Element *F = E->get()->channels.front(); F; F = F->next()) {
			if (F->get()->get_available_packet_count()) {
				next_packet_peer = E->key();
				return;
			}
		}
		E = E->next();
	}
	E = peer_map.front();
	// Before last
	while (E) {
		for (List<Ref<WebRTCDataChannel>>::Element *F = E->get()->channels.front(); F; F = F->next()) {
			if (F->get()->get_available_packet_count()) {
				next_packet_peer = E->key();
				return;
			}
		}
		if (E->key() == (int)next_packet_peer) {
			break;
		}
		E = E->next();
	}
	// No packet found
	next_packet_peer = 0;
}

void WebRTCMultiplayer::set_refuse_new_connections(bool p_enable) {
	refuse_connections = p_enable;
}

bool WebRTCMultiplayer::is_refusing_new_connections() const {
	return refuse_connections;
}

NetworkedMultiplayerPeer::ConnectionStatus WebRTCMultiplayer::get_connection_status() const {
	return connection_status;
}

Error WebRTCMultiplayer::initialize(int p_self_id, bool p_server_compat) {
	ERR_FAIL_COND_V(p_self_id < 0 || p_self_id > ~(1 << 31), ERR_INVALID_PARAMETER);
	unique_id = p_self_id;
	server_compat = p_server_compat;

	// Mesh and server are always connected
	if (!server_compat || p_self_id == 1) {
		connection_status = CONNECTION_CONNECTED;
	} else {
		connection_status = CONNECTION_CONNECTING;
	}
	return OK;
}

int WebRTCMultiplayer::get_unique_id() const {
	ERR_FAIL_COND_V(connection_status == CONNECTION_DISCONNECTED, 1);
	return unique_id;
}

void WebRTCMultiplayer::_peer_to_dict(Ref<ConnectedPeer> p_connected_peer, Dictionary &r_dict) {
	Array channels;
	for (List<Ref<WebRTCDataChannel>>::Element *F = p_connected_peer->channels.front(); F; F = F->next()) {
		channels.push_back(F->get());
	}
	r_dict["connection"] = p_connected_peer->connection;
	r_dict["connected"] = p_connected_peer->connected;
	r_dict["channels"] = channels;
}

bool WebRTCMultiplayer::has_peer(int p_peer_id) {
	return peer_map.has(p_peer_id);
}

Dictionary WebRTCMultiplayer::get_peer(int p_peer_id) {
	ERR_FAIL_COND_V(!peer_map.has(p_peer_id), Dictionary());
	Dictionary out;
	_peer_to_dict(peer_map[p_peer_id], out);
	return out;
}

Dictionary WebRTCMultiplayer::get_peers() {
	Dictionary out;
	for (Map<int, Ref<ConnectedPeer>>::Element *E = peer_map.front(); E; E = E->next()) {
		Dictionary d;
		_peer_to_dict(E->get(), d);
		out[E->key()] = d;
	}
	return out;
}

Error WebRTCMultiplayer::add_peer(Ref<WebRTCPeerConnection> p_peer, int p_peer_id, int p_unreliable_lifetime) {
	ERR_FAIL_COND_V(p_peer_id < 0 || p_peer_id > ~(1 << 31), ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_unreliable_lifetime < 0, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(refuse_connections, ERR_UNAUTHORIZED);
	// Peer must be valid, and in new state (to create data channels)
	ERR_FAIL_COND_V(!p_peer.is_valid(), ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_peer->get_connection_state() != WebRTCPeerConnection::STATE_NEW, ERR_INVALID_PARAMETER);

	Ref<ConnectedPeer> peer = memnew(ConnectedPeer);
	peer->connection = p_peer;

	// Initialize data channels
	Dictionary cfg;
	cfg["negotiated"] = true;
	cfg["ordered"] = true;

	cfg["id"] = 1;
	peer->channels[CH_RELIABLE] = p_peer->create_data_channel("reliable", cfg);
	ERR_FAIL_COND_V(!peer->channels[CH_RELIABLE].is_valid(), FAILED);

	cfg["id"] = 2;
	cfg["maxPacketLifetime"] = p_unreliable_lifetime;
	peer->channels[CH_ORDERED] = p_peer->create_data_channel("ordered", cfg);
	ERR_FAIL_COND_V(!peer->channels[CH_ORDERED].is_valid(), FAILED);

	cfg["id"] = 3;
	cfg["ordered"] = false;
	peer->channels[CH_UNRELIABLE] = p_peer->create_data_channel("unreliable", cfg);
	ERR_FAIL_COND_V(!peer->channels[CH_UNRELIABLE].is_valid(), FAILED);

	peer_map[p_peer_id] = peer; // add the new peer connection to the peer_map

	return OK;
}

void WebRTCMultiplayer::remove_peer(int p_peer_id) {
	ERR_FAIL_COND(!peer_map.has(p_peer_id));
	Ref<ConnectedPeer> peer = peer_map[p_peer_id];
	peer_map.erase(p_peer_id);
	if (peer->connected) {
		peer->connected = false;
		emit_signal("peer_disconnected", p_peer_id);
		if (server_compat && p_peer_id == TARGET_PEER_SERVER) {
			emit_signal("server_disconnected");
			connection_status = CONNECTION_DISCONNECTED;
		}
	}
}

Error WebRTCMultiplayer::get_packet(const uint8_t **r_buffer, int &r_buffer_size) {
	// Peer not available
	if (next_packet_peer == 0 || !peer_map.has(next_packet_peer)) {
		_find_next_peer();
		ERR_FAIL_V(ERR_UNAVAILABLE);
	}
	for (List<Ref<WebRTCDataChannel>>::Element *E = peer_map[next_packet_peer]->channels.front(); E; E = E->next()) {
		if (E->get()->get_available_packet_count()) {
			Error err = E->get()->get_packet(r_buffer, r_buffer_size);
			_find_next_peer();
			return err;
		}
	}
	// Channels for that peer were empty. Bug?
	_find_next_peer();
	ERR_FAIL_V(ERR_BUG);
}

Error WebRTCMultiplayer::put_packet(const uint8_t *p_buffer, int p_buffer_size) {
	ERR_FAIL_COND_V(connection_status == CONNECTION_DISCONNECTED, ERR_UNCONFIGURED);

	int ch = CH_RELIABLE;
	switch (transfer_mode) {
		case TRANSFER_MODE_RELIABLE:
			ch = CH_RELIABLE;
			break;
		case TRANSFER_MODE_UNRELIABLE_ORDERED:
			ch = CH_ORDERED;
			break;
		case TRANSFER_MODE_UNRELIABLE:
			ch = CH_UNRELIABLE;
			break;
	}

	Map<int, Ref<ConnectedPeer>>::Element *E = nullptr;

	if (target_peer > 0) {
		E = peer_map.find(target_peer);
		ERR_FAIL_COND_V_MSG(!E, ERR_INVALID_PARAMETER, "Invalid target peer: " + itos(target_peer) + ".");

		ERR_FAIL_COND_V(E->value()->channels.size() <= ch, ERR_BUG);
		ERR_FAIL_COND_V(!E->value()->channels[ch].is_valid(), ERR_BUG);
		return E->value()->channels[ch]->put_packet(p_buffer, p_buffer_size);

	} else {
		int exclude = -target_peer;

		for (Map<int, Ref<ConnectedPeer>>::Element *F = peer_map.front(); F; F = F->next()) {
			// Exclude packet. If target_peer == 0 then don't exclude any packets
			if (target_peer != 0 && F->key() == exclude) {
				continue;
			}

			ERR_CONTINUE(F->value()->channels.size() <= ch || !F->value()->channels[ch].is_valid());
			F->value()->channels[ch]->put_packet(p_buffer, p_buffer_size);
		}
	}
	return OK;
}

int WebRTCMultiplayer::get_available_packet_count() const {
	if (next_packet_peer == 0) {
		return 0; // To be sure next call to get_packet works if size > 0 .
	}
	int size = 0;
	for (Map<int, Ref<ConnectedPeer>>::Element *E = peer_map.front(); E; E = E->next()) {
		for (List<Ref<WebRTCDataChannel>>::Element *F = E->get()->channels.front(); F; F = F->next()) {
			size += F->get()->get_available_packet_count();
		}
	}
	return size;
}

int WebRTCMultiplayer::get_max_packet_size() const {
	return 1200;
}

void WebRTCMultiplayer::close() {
	peer_map.clear();
	unique_id = 0;
	next_packet_peer = 0;
	target_peer = 0;
	connection_status = CONNECTION_DISCONNECTED;
}

WebRTCMultiplayer::WebRTCMultiplayer() {
	unique_id = 0;
	next_packet_peer = 0;
	target_peer = 0;
	client_count = 0;
	transfer_mode = TRANSFER_MODE_RELIABLE;
	refuse_connections = false;
	connection_status = CONNECTION_DISCONNECTED;
	server_compat = false;
}

WebRTCMultiplayer::~WebRTCMultiplayer() {
	close();
}
