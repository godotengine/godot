/*************************************************************************/
/*  webrtc_multiplayer_peer.cpp                                          */
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

#include "webrtc_multiplayer_peer.h"

#include "core/io/marshalls.h"
#include "core/os/os.h"

void WebRTCMultiplayerPeer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("initialize", "peer_id", "server_compatibility", "channels_config"), &WebRTCMultiplayerPeer::initialize, DEFVAL(false), DEFVAL(Array()));
	ClassDB::bind_method(D_METHOD("add_peer", "peer", "peer_id", "unreliable_lifetime"), &WebRTCMultiplayerPeer::add_peer, DEFVAL(1));
	ClassDB::bind_method(D_METHOD("remove_peer", "peer_id"), &WebRTCMultiplayerPeer::remove_peer);
	ClassDB::bind_method(D_METHOD("has_peer", "peer_id"), &WebRTCMultiplayerPeer::has_peer);
	ClassDB::bind_method(D_METHOD("get_peer", "peer_id"), &WebRTCMultiplayerPeer::get_peer);
	ClassDB::bind_method(D_METHOD("get_peers"), &WebRTCMultiplayerPeer::get_peers);
	ClassDB::bind_method(D_METHOD("close"), &WebRTCMultiplayerPeer::close);
}

void WebRTCMultiplayerPeer::set_target_peer(int p_peer_id) {
	target_peer = p_peer_id;
}

/* Returns the ID of the MultiplayerPeer who sent the most recent packet: */
int WebRTCMultiplayerPeer::get_packet_peer() const {
	return next_packet_peer;
}

bool WebRTCMultiplayerPeer::is_server() const {
	return unique_id == TARGET_PEER_SERVER;
}

void WebRTCMultiplayerPeer::poll() {
	if (peer_map.size() == 0) {
		return;
	}

	List<int> remove;
	List<int> add;
	for (KeyValue<int, Ref<ConnectedPeer>> &E : peer_map) {
		Ref<ConnectedPeer> peer = E.value;
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
				remove.push_back(E.key);
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
					remove.push_back(E.key);
			}
			// We got a closed channel break out, the peer will be removed.
			break;
		}
		// This peer has newly connected, and all channels are now open.
		if (ready == peer->channels.size() && !peer->connected) {
			peer->connected = true;
			add.push_back(E.key);
		}
	}
	// Remove disconnected peers
	for (int &E : remove) {
		remove_peer(E);
		if (next_packet_peer == E) {
			next_packet_peer = 0;
		}
	}
	// Signal newly connected peers
	for (int &E : add) {
		// Already connected to server: simply notify new peer.
		// NOTE: Mesh is always connected.
		if (connection_status == CONNECTION_CONNECTED) {
			emit_signal(SNAME("peer_connected"), E);
		}

		// Server emulation mode suppresses peer_conencted until server connects.
		if (server_compat && E == TARGET_PEER_SERVER) {
			// Server connected.
			connection_status = CONNECTION_CONNECTED;
			emit_signal(SNAME("peer_connected"), TARGET_PEER_SERVER);
			emit_signal(SNAME("connection_succeeded"));
			// Notify of all previously connected peers
			for (const KeyValue<int, Ref<ConnectedPeer>> &F : peer_map) {
				if (F.key != 1 && F.value->connected) {
					emit_signal(SNAME("peer_connected"), F.key);
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

void WebRTCMultiplayerPeer::_find_next_peer() {
	Map<int, Ref<ConnectedPeer>>::Element *E = peer_map.find(next_packet_peer);
	if (E) {
		E = E->next();
	}
	// After last.
	while (E) {
		if (!E->get()->connected) {
			E = E->next();
			continue;
		}
		for (const Ref<WebRTCDataChannel> &F : E->get()->channels) {
			if (F->get_available_packet_count()) {
				next_packet_peer = E->key();
				return;
			}
		}
		E = E->next();
	}
	E = peer_map.front();
	// Before last
	while (E) {
		if (!E->get()->connected) {
			E = E->next();
			continue;
		}
		for (const Ref<WebRTCDataChannel> &F : E->get()->channels) {
			if (F->get_available_packet_count()) {
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

MultiplayerPeer::ConnectionStatus WebRTCMultiplayerPeer::get_connection_status() const {
	return connection_status;
}

Error WebRTCMultiplayerPeer::initialize(int p_self_id, bool p_server_compat, Array p_channels_config) {
	ERR_FAIL_COND_V(p_self_id < 1 || p_self_id > ~(1 << 31), ERR_INVALID_PARAMETER);
	channels_config.clear();
	for (int i = 0; i < p_channels_config.size(); i++) {
		ERR_FAIL_COND_V_MSG(p_channels_config[i].get_type() != Variant::INT, ERR_INVALID_PARAMETER, "The 'channels_config' array must contain only enum values from 'MultiplayerPeer.Multiplayer::TransferMode'");
		int mode = p_channels_config[i].operator int();
		// Initialize data channel configurations.
		Dictionary cfg;
		cfg["id"] = CH_RESERVED_MAX + i + 1;
		cfg["negotiated"] = true;
		cfg["ordered"] = true;

		switch (mode) {
			case Multiplayer::TRANSFER_MODE_UNRELIABLE_ORDERED:
				cfg["maxPacketLifetime"] = 1;
				break;
			case Multiplayer::TRANSFER_MODE_UNRELIABLE:
				cfg["maxPacketLifetime"] = 1;
				cfg["ordered"] = false;
				break;
			case Multiplayer::TRANSFER_MODE_RELIABLE:
				break;
			default:
				ERR_FAIL_V_MSG(ERR_INVALID_PARAMETER, vformat("The 'channels_config' array must contain only enum values from 'MultiplayerPeer.Multiplayer::TransferMode'. Got: %d", mode));
		}
		channels_config.push_back(cfg);
	}

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

int WebRTCMultiplayerPeer::get_unique_id() const {
	ERR_FAIL_COND_V(connection_status == CONNECTION_DISCONNECTED, 1);
	return unique_id;
}

void WebRTCMultiplayerPeer::_peer_to_dict(Ref<ConnectedPeer> p_connected_peer, Dictionary &r_dict) {
	Array channels;
	for (Ref<WebRTCDataChannel> &F : p_connected_peer->channels) {
		channels.push_back(F);
	}
	r_dict["connection"] = p_connected_peer->connection;
	r_dict["connected"] = p_connected_peer->connected;
	r_dict["channels"] = channels;
}

bool WebRTCMultiplayerPeer::has_peer(int p_peer_id) {
	return peer_map.has(p_peer_id);
}

Dictionary WebRTCMultiplayerPeer::get_peer(int p_peer_id) {
	ERR_FAIL_COND_V(!peer_map.has(p_peer_id), Dictionary());
	Dictionary out;
	_peer_to_dict(peer_map[p_peer_id], out);
	return out;
}

Dictionary WebRTCMultiplayerPeer::get_peers() {
	Dictionary out;
	for (const KeyValue<int, Ref<ConnectedPeer>> &E : peer_map) {
		Dictionary d;
		_peer_to_dict(E.value, d);
		out[E.key] = d;
	}
	return out;
}

Error WebRTCMultiplayerPeer::add_peer(Ref<WebRTCPeerConnection> p_peer, int p_peer_id, int p_unreliable_lifetime) {
	ERR_FAIL_COND_V(p_peer_id < 0 || p_peer_id > ~(1 << 31), ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_unreliable_lifetime < 0, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(is_refusing_new_connections(), ERR_UNAUTHORIZED);
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
	ERR_FAIL_COND_V(peer->channels[CH_RELIABLE].is_null(), FAILED);

	cfg["id"] = 2;
	cfg["maxPacketLifetime"] = p_unreliable_lifetime;
	peer->channels[CH_ORDERED] = p_peer->create_data_channel("ordered", cfg);
	ERR_FAIL_COND_V(peer->channels[CH_ORDERED].is_null(), FAILED);

	cfg["id"] = 3;
	cfg["ordered"] = false;
	peer->channels[CH_UNRELIABLE] = p_peer->create_data_channel("unreliable", cfg);
	ERR_FAIL_COND_V(peer->channels[CH_UNRELIABLE].is_null(), FAILED);

	for (const Dictionary &dict : channels_config) {
		Ref<WebRTCDataChannel> ch = p_peer->create_data_channel(String::num_int64(dict["id"]), dict);
		ERR_FAIL_COND_V(ch.is_null(), FAILED);
		peer->channels.push_back(ch);
	}

	peer_map[p_peer_id] = peer; // add the new peer connection to the peer_map

	return OK;
}

void WebRTCMultiplayerPeer::remove_peer(int p_peer_id) {
	ERR_FAIL_COND(!peer_map.has(p_peer_id));
	Ref<ConnectedPeer> peer = peer_map[p_peer_id];
	peer_map.erase(p_peer_id);
	if (peer->connected) {
		peer->connected = false;
		emit_signal(SNAME("peer_disconnected"), p_peer_id);
		if (server_compat && p_peer_id == TARGET_PEER_SERVER) {
			emit_signal(SNAME("server_disconnected"));
			connection_status = CONNECTION_DISCONNECTED;
		}
	}
}

Error WebRTCMultiplayerPeer::get_packet(const uint8_t **r_buffer, int &r_buffer_size) {
	// Peer not available
	if (next_packet_peer == 0 || !peer_map.has(next_packet_peer)) {
		_find_next_peer();
		ERR_FAIL_V(ERR_UNAVAILABLE);
	}
	for (Ref<WebRTCDataChannel> &E : peer_map[next_packet_peer]->channels) {
		if (E->get_available_packet_count()) {
			Error err = E->get_packet(r_buffer, r_buffer_size);
			_find_next_peer();
			return err;
		}
	}
	// Channels for that peer were empty. Bug?
	_find_next_peer();
	ERR_FAIL_V(ERR_BUG);
}

Error WebRTCMultiplayerPeer::put_packet(const uint8_t *p_buffer, int p_buffer_size) {
	ERR_FAIL_COND_V(connection_status == CONNECTION_DISCONNECTED, ERR_UNCONFIGURED);

	int ch = get_transfer_channel();
	if (ch == 0) {
		switch (get_transfer_mode()) {
			case Multiplayer::TRANSFER_MODE_RELIABLE:
				ch = CH_RELIABLE;
				break;
			case Multiplayer::TRANSFER_MODE_UNRELIABLE_ORDERED:
				ch = CH_ORDERED;
				break;
			case Multiplayer::TRANSFER_MODE_UNRELIABLE:
				ch = CH_UNRELIABLE;
				break;
		}
	} else {
		ch += CH_RESERVED_MAX - 1;
	}

	Map<int, Ref<ConnectedPeer>>::Element *E = nullptr;

	if (target_peer > 0) {
		E = peer_map.find(target_peer);
		ERR_FAIL_COND_V_MSG(!E, ERR_INVALID_PARAMETER, "Invalid target peer: " + itos(target_peer) + ".");

		ERR_FAIL_COND_V_MSG(E->value()->channels.size() <= ch, ERR_INVALID_PARAMETER, vformat("Unable to send packet on channel %d, max channels: %d", ch, E->value()->channels.size()));
		ERR_FAIL_COND_V(E->value()->channels[ch].is_null(), ERR_BUG);
		return E->value()->channels[ch]->put_packet(p_buffer, p_buffer_size);

	} else {
		int exclude = -target_peer;

		for (KeyValue<int, Ref<ConnectedPeer>> &F : peer_map) {
			// Exclude packet. If target_peer == 0 then don't exclude any packets
			if (target_peer != 0 && F.key == exclude) {
				continue;
			}

			ERR_CONTINUE_MSG(F.value->channels.size() <= ch, vformat("Unable to send packet on channel %d, max channels: %d", ch, E->value()->channels.size()));
			ERR_CONTINUE(F.value->channels[ch].is_null());
			F.value->channels[ch]->put_packet(p_buffer, p_buffer_size);
		}
	}
	return OK;
}

int WebRTCMultiplayerPeer::get_available_packet_count() const {
	if (next_packet_peer == 0) {
		return 0; // To be sure next call to get_packet works if size > 0 .
	}
	int size = 0;
	for (const KeyValue<int, Ref<ConnectedPeer>> &E : peer_map) {
		if (!E.value->connected) {
			continue;
		}
		for (const Ref<WebRTCDataChannel> &F : E.value->channels) {
			size += F->get_available_packet_count();
		}
	}
	return size;
}

int WebRTCMultiplayerPeer::get_max_packet_size() const {
	return 1200;
}

void WebRTCMultiplayerPeer::close() {
	peer_map.clear();
	channels_config.clear();
	unique_id = 0;
	next_packet_peer = 0;
	target_peer = 0;
	connection_status = CONNECTION_DISCONNECTED;
}

WebRTCMultiplayerPeer::~WebRTCMultiplayerPeer() {
	close();
}
