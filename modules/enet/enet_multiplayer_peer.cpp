/*************************************************************************/
/*  enet_multiplayer_peer.cpp                                            */
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

#include "enet_multiplayer_peer.h"
#include "core/io/ip.h"
#include "core/io/marshalls.h"
#include "core/os/os.h"

void ENetMultiplayerPeer::set_target_peer(int p_peer) {
	target_peer = p_peer;
}

int ENetMultiplayerPeer::get_packet_peer() const {
	ERR_FAIL_COND_V_MSG(!_is_active(), 1, "The multiplayer instance isn't currently active.");
	ERR_FAIL_COND_V(incoming_packets.size() == 0, 1);

	return incoming_packets.front()->get().from;
}

Error ENetMultiplayerPeer::create_server(int p_port, int p_max_clients, int p_max_channels, int p_in_bandwidth, int p_out_bandwidth) {
	ERR_FAIL_COND_V_MSG(_is_active(), ERR_ALREADY_IN_USE, "The multiplayer instance is already active.");
	set_refuse_new_connections(false);
	Ref<ENetConnection> host;
	host.instantiate();
	Error err = host->create_host_bound(bind_ip, p_port, p_max_clients, 0, p_max_channels > 0 ? p_max_channels + SYSCH_MAX : 0, p_out_bandwidth);
	if (err != OK) {
		return err;
	}

	active_mode = MODE_SERVER;
	unique_id = 1;
	connection_status = CONNECTION_CONNECTED;
	hosts[0] = host;
	return OK;
}

Error ENetMultiplayerPeer::create_client(const String &p_address, int p_port, int p_channel_count, int p_in_bandwidth, int p_out_bandwidth, int p_local_port) {
	ERR_FAIL_COND_V_MSG(_is_active(), ERR_ALREADY_IN_USE, "The multiplayer instance is already active.");
	set_refuse_new_connections(false);
	Ref<ENetConnection> host;
	host.instantiate();
	Error err;
	if (p_local_port) {
		err = host->create_host_bound(bind_ip, p_local_port, 1, 0, p_in_bandwidth, p_out_bandwidth);
	} else {
		err = host->create_host(1, 0, p_in_bandwidth, p_out_bandwidth);
	}
	if (err != OK) {
		return err;
	}

	unique_id = generate_unique_id();

	Ref<ENetPacketPeer> peer = host->connect_to_host(p_address, p_port, p_channel_count > 0 ? p_channel_count + SYSCH_MAX : 0, unique_id);
	if (peer.is_null()) {
		host->destroy();
		ERR_FAIL_V_MSG(ERR_CANT_CREATE, "Couldn't connect to the ENet multiplayer server.");
	}

	// Need to wait for CONNECT event.
	connection_status = CONNECTION_CONNECTING;
	active_mode = MODE_CLIENT;
	peers[1] = peer;
	hosts[0] = host;

	return OK;
}

Error ENetMultiplayerPeer::create_mesh(int p_id) {
	ERR_FAIL_COND_V_MSG(p_id <= 0, ERR_INVALID_PARAMETER, "The unique ID must be greater then 0");
	ERR_FAIL_COND_V_MSG(_is_active(), ERR_ALREADY_IN_USE, "The multiplayer instance is already active.");
	active_mode = MODE_MESH;
	unique_id = p_id;
	connection_status = CONNECTION_CONNECTED;
	return OK;
}

Error ENetMultiplayerPeer::add_mesh_peer(int p_id, Ref<ENetConnection> p_host) {
	ERR_FAIL_COND_V(p_host.is_null(), ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V_MSG(active_mode != MODE_MESH, ERR_UNCONFIGURED, "The multiplayer instance is not configured as a mesh. Call 'create_mesh' first.");
	List<Ref<ENetPacketPeer>> host_peers;
	p_host->get_peers(host_peers);
	ERR_FAIL_COND_V_MSG(host_peers.size() != 1 || host_peers[0]->get_state() != ENetPacketPeer::STATE_CONNECTED, ERR_INVALID_PARAMETER, "The provided host must have exactly one peer in the connected state.");
	hosts[p_id] = p_host;
	peers[p_id] = host_peers[0];
	emit_signal(SNAME("peer_connected"), p_id);
	return OK;
}

bool ENetMultiplayerPeer::_parse_server_event(ENetConnection::EventType p_type, ENetConnection::Event &p_event) {
	switch (p_type) {
		case ENetConnection::EVENT_CONNECT: {
			if (is_refusing_new_connections()) {
				p_event.peer->reset();
				return false;
			}
			// Client joined with invalid ID, probably trying to exploit us.
			if (p_event.data < 2 || peers.has((int)p_event.data)) {
				p_event.peer->reset();
				return false;
			}
			int id = p_event.data;
			p_event.peer->set_meta(SNAME("_net_id"), id);
			peers[id] = p_event.peer;

			emit_signal(SNAME("peer_connected"), id);
			if (server_relay) {
				_notify_peers(id, true);
			}
			return false;
		}
		case ENetConnection::EVENT_DISCONNECT: {
			int id = p_event.peer->get_meta(SNAME("_net_id"));
			if (!peers.has(id)) {
				// Never fully connected.
				return false;
			}

			emit_signal(SNAME("peer_disconnected"), id);
			peers.erase(id);
			if (server_relay) {
				_notify_peers(id, false);
			}
			return false;
		}
		case ENetConnection::EVENT_RECEIVE: {
			if (p_event.channel_id == SYSCH_CONFIG) {
				_destroy_unused(p_event.packet);
				ERR_FAIL_V_MSG(false, "Only server can send config messages");
			} else {
				if (p_event.packet->dataLength < 8) {
					_destroy_unused(p_event.packet);
					ERR_FAIL_V_MSG(false, "Invalid packet size");
				}

				uint32_t source = decode_uint32(&p_event.packet->data[0]);
				int target = decode_uint32(&p_event.packet->data[4]);

				uint32_t id = p_event.peer->get_meta(SNAME("_net_id"));
				// Someone is cheating and trying to fake the source!
				if (source != id) {
					_destroy_unused(p_event.packet);
					ERR_FAIL_V_MSG(false, "Someone is cheating and trying to fake the source!");
				}

				Packet packet;
				packet.packet = p_event.packet;
				packet.channel = p_event.channel_id;
				packet.from = id;

				// Even if relaying is disabled, these targets are valid as incoming packets.
				if (target == 1 || target == 0 || target < -1) {
					packet.packet->referenceCount++;
					incoming_packets.push_back(packet);
				}

				if (server_relay && target != 1) {
					packet.packet->referenceCount++;
					_relay(source, target, p_event.channel_id, p_event.packet);
					packet.packet->referenceCount--;
					_destroy_unused(p_event.packet);
				}
				// Destroy packet later
			}
			return false;
		}
		default:
			return true;
	}
}

bool ENetMultiplayerPeer::_parse_client_event(ENetConnection::EventType p_type, ENetConnection::Event &p_event) {
	switch (p_type) {
		case ENetConnection::EVENT_CONNECT: {
			connection_status = CONNECTION_CONNECTED;
			emit_signal(SNAME("peer_connected"), 1);
			emit_signal(SNAME("connection_succeeded"));
			return false;
		}
		case ENetConnection::EVENT_DISCONNECT: {
			if (connection_status == CONNECTION_CONNECTED) {
				// Client just disconnected from server.
				emit_signal(SNAME("server_disconnected"));
			} else {
				emit_signal(SNAME("connection_failed"));
			}
			close_connection();
			return true;
		}
		case ENetConnection::EVENT_RECEIVE: {
			if (p_event.channel_id == SYSCH_CONFIG) {
				// Config message
				if (p_event.packet->dataLength != 8) {
					_destroy_unused(p_event.packet);
					ERR_FAIL_V(false);
				}

				int msg = decode_uint32(&p_event.packet->data[0]);
				int id = decode_uint32(&p_event.packet->data[4]);

				switch (msg) {
					case SYSMSG_ADD_PEER: {
						peers[id] = Ref<ENetPacketPeer>();
						emit_signal(SNAME("peer_connected"), id);

					} break;
					case SYSMSG_REMOVE_PEER: {
						peers.erase(id);
						emit_signal(SNAME("peer_disconnected"), id);
					} break;
				}
				_destroy_unused(p_event.packet);
			} else {
				if (p_event.packet->dataLength < 8) {
					_destroy_unused(p_event.packet);
					ERR_FAIL_V_MSG(false, "Invalid packet size");
				}

				uint32_t source = decode_uint32(&p_event.packet->data[0]);
				Packet packet;
				packet.packet = p_event.packet;
				packet.from = source;
				packet.channel = p_event.channel_id;

				packet.packet->referenceCount++;
				incoming_packets.push_back(packet);
				// Destroy packet later
			}
			return false;
		}
		default:
			return true;
	}
}

bool ENetMultiplayerPeer::_parse_mesh_event(ENetConnection::EventType p_type, ENetConnection::Event &p_event, int p_peer_id) {
	switch (p_type) {
		case ENetConnection::EVENT_CONNECT:
			p_event.peer->reset();
			return false;
		case ENetConnection::EVENT_DISCONNECT:
			if (peers.has(p_peer_id)) {
				emit_signal(SNAME("peer_disconnected"), p_peer_id);
				peers.erase(p_peer_id);
			}
			hosts.erase(p_peer_id);
			return true;
		case ENetConnection::EVENT_RECEIVE: {
			if (p_event.packet->dataLength < 8) {
				_destroy_unused(p_event.packet);
				ERR_FAIL_V_MSG(false, "Invalid packet size");
			}

			Packet packet;
			packet.packet = p_event.packet;
			packet.from = p_peer_id;
			packet.channel = p_event.channel_id;

			packet.packet->referenceCount++;
			incoming_packets.push_back(packet);
			return false;
		} break;
		default:
			// Nothing to do
			return true;
	}
}

void ENetMultiplayerPeer::poll() {
	ERR_FAIL_COND_MSG(!_is_active(), "The multiplayer instance isn't currently active.");

	_pop_current_packet();

	switch (active_mode) {
		case MODE_CLIENT: {
			if (peers.has(1) && !peers[1]->is_active()) {
				if (connection_status == CONNECTION_CONNECTED) {
					// Client just disconnected from server.
					emit_signal(SNAME("server_disconnected"));
				} else {
					emit_signal(SNAME("connection_failed"));
				}
				close_connection();
				return;
			}
			ENetConnection::Event event;
			ENetConnection::EventType ret = hosts[0]->service(0, event);
			if (ret == ENetConnection::EVENT_ERROR) {
				return;
			}
			do {
				if (_parse_client_event(ret, event)) {
					return;
				}
			} while (hosts[0]->check_events(ret, event) > 0);
		} break;
		case MODE_SERVER: {
			for (const KeyValue<int, Ref<ENetPacketPeer>> &E : peers) {
				if (!(E.value->is_active())) {
					emit_signal(SNAME("peer_disconnected"), E.value->get_meta(SNAME("_net_id")));
					peers.erase(E.key);
				}
			}
			ENetConnection::Event event;
			ENetConnection::EventType ret = hosts[0]->service(0, event);
			if (ret == ENetConnection::EVENT_ERROR) {
				return;
			}
			do {
				if (_parse_server_event(ret, event)) {
					return;
				}
			} while (hosts[0]->check_events(ret, event) > 0);
		} break;
		case MODE_MESH: {
			for (const KeyValue<int, Ref<ENetPacketPeer>> &E : peers) {
				if (!(E.value->is_active())) {
					emit_signal(SNAME("peer_disconnected"), E.key);
					peers.erase(E.key);
					if (hosts.has(E.key)) {
						hosts.erase(E.key);
					}
				}
			}
			for (KeyValue<int, Ref<ENetConnection>> &E : hosts) {
				ENetConnection::Event event;
				ENetConnection::EventType ret = E.value->service(0, event);
				if (ret == ENetConnection::EVENT_ERROR) {
					if (peers.has(E.key)) {
						emit_signal(SNAME("peer_disconnected"), E.key);
						peers.erase(E.key);
					}
					hosts.erase(E.key);
					continue;
				}
				do {
					if (_parse_mesh_event(ret, event, E.key)) {
						break; // Keep polling the others.
					}
				} while (E.value->check_events(ret, event) > 0);
			}
		} break;
		default:
			return;
	}
}

bool ENetMultiplayerPeer::is_server() const {
	return active_mode == MODE_SERVER;
}

void ENetMultiplayerPeer::close_connection(uint32_t wait_usec) {
	if (!_is_active()) {
		return;
	}

	_pop_current_packet();

	bool peers_disconnected = false;
	for (KeyValue<int, Ref<ENetPacketPeer>> &E : peers) {
		if (E.value.is_valid() && E.value->get_state() == ENetPacketPeer::STATE_CONNECTED) {
			E.value->peer_disconnect_now(unique_id);
			peers_disconnected = true;
		}
	}

	if (peers_disconnected) {
		for (KeyValue<int, Ref<ENetConnection>> &E : hosts) {
			E.value->flush();
		}

		if (wait_usec > 0) {
			OS::get_singleton()->delay_usec(wait_usec); // Wait for disconnection packets to send
		}
	}

	active_mode = MODE_NONE;
	incoming_packets.clear();
	peers.clear();
	hosts.clear();
	unique_id = 0;
	connection_status = CONNECTION_DISCONNECTED;
	set_refuse_new_connections(false);
}

int ENetMultiplayerPeer::get_available_packet_count() const {
	return incoming_packets.size();
}

Error ENetMultiplayerPeer::get_packet(const uint8_t **r_buffer, int &r_buffer_size) {
	ERR_FAIL_COND_V_MSG(incoming_packets.size() == 0, ERR_UNAVAILABLE, "No incoming packets available.");

	_pop_current_packet();

	current_packet = incoming_packets.front()->get();
	incoming_packets.pop_front();

	*r_buffer = (const uint8_t *)(&current_packet.packet->data[8]);
	r_buffer_size = current_packet.packet->dataLength - 8;

	return OK;
}

Error ENetMultiplayerPeer::put_packet(const uint8_t *p_buffer, int p_buffer_size) {
	ERR_FAIL_COND_V_MSG(!_is_active(), ERR_UNCONFIGURED, "The multiplayer instance isn't currently active.");
	ERR_FAIL_COND_V_MSG(connection_status != CONNECTION_CONNECTED, ERR_UNCONFIGURED, "The multiplayer instance isn't currently connected to any server or client.");
	ERR_FAIL_COND_V_MSG(target_peer != 0 && !peers.has(ABS(target_peer)), ERR_INVALID_PARAMETER, vformat("Invalid target peer: %d", target_peer));
	ERR_FAIL_COND_V(active_mode == MODE_CLIENT && !peers.has(1), ERR_BUG);

	int packet_flags = 0;
	int channel = SYSCH_RELIABLE;
	int transfer_channel = get_transfer_channel();
	if (transfer_channel > 0) {
		channel = SYSCH_MAX + transfer_channel - 1;
	} else {
		switch (get_transfer_mode()) {
			case Multiplayer::TRANSFER_MODE_UNRELIABLE: {
				packet_flags = ENET_PACKET_FLAG_UNSEQUENCED | ENET_PACKET_FLAG_UNRELIABLE_FRAGMENT;
				channel = SYSCH_UNRELIABLE;
			} break;
			case Multiplayer::TRANSFER_MODE_UNRELIABLE_ORDERED: {
				packet_flags = ENET_PACKET_FLAG_UNRELIABLE_FRAGMENT;
				channel = SYSCH_UNRELIABLE;
			} break;
			case Multiplayer::TRANSFER_MODE_RELIABLE: {
				packet_flags = ENET_PACKET_FLAG_RELIABLE;
				channel = SYSCH_RELIABLE;
			} break;
		}
	}

#ifdef DEBUG_ENABLED
	if ((packet_flags & ENET_PACKET_FLAG_UNRELIABLE_FRAGMENT) && p_buffer_size + 8 > ENET_HOST_DEFAULT_MTU) {
		WARN_PRINT_ONCE(vformat("Sending %d bytes unrealiably which is above the MTU (%d), this will result in higher packet loss", p_buffer_size + 8, ENET_HOST_DEFAULT_MTU));
	}
#endif

	ENetPacket *packet = enet_packet_create(nullptr, p_buffer_size + 8, packet_flags);
	encode_uint32(unique_id, &packet->data[0]); // Source ID
	encode_uint32(target_peer, &packet->data[4]); // Dest ID
	memcpy(&packet->data[8], p_buffer, p_buffer_size);

	if (is_server()) {
		if (target_peer == 0) {
			hosts[0]->broadcast(channel, packet);

		} else if (target_peer < 0) {
			// Send to all but one and make copies for sending.
			int exclude = -target_peer;
			for (KeyValue<int, Ref<ENetPacketPeer>> &E : peers) {
				if (E.key == exclude) {
					continue;
				}
				E.value->send(channel, packet);
			}
			_destroy_unused(packet);
		} else {
			peers[target_peer]->send(channel, packet);
		}
		ERR_FAIL_COND_V(!hosts.has(0), ERR_BUG);
		hosts[0]->flush();

	} else if (active_mode == MODE_CLIENT) {
		peers[1]->send(channel, packet); // Send to server for broadcast.
		ERR_FAIL_COND_V(!hosts.has(0), ERR_BUG);
		hosts[0]->flush();

	} else {
		if (target_peer <= 0) {
			int exclude = ABS(target_peer);
			for (KeyValue<int, Ref<ENetPacketPeer>> &E : peers) {
				if (E.key == exclude) {
					continue;
				}
				E.value->send(channel, packet);
				ERR_CONTINUE(!hosts.has(E.key));
				hosts[E.key]->flush();
			}
			_destroy_unused(packet);
		} else {
			peers[target_peer]->send(channel, packet);
			ERR_FAIL_COND_V(!hosts.has(target_peer), ERR_BUG);
			hosts[target_peer]->flush();
		}
	}

	return OK;
}

int ENetMultiplayerPeer::get_max_packet_size() const {
	return 1 << 24; // Anything is good
}

void ENetMultiplayerPeer::_pop_current_packet() {
	if (current_packet.packet) {
		current_packet.packet->referenceCount--;
		_destroy_unused(current_packet.packet);
		current_packet.packet = nullptr;
		current_packet.from = 0;
		current_packet.channel = -1;
	}
}

MultiplayerPeer::ConnectionStatus ENetMultiplayerPeer::get_connection_status() const {
	return connection_status;
}

int ENetMultiplayerPeer::get_unique_id() const {
	ERR_FAIL_COND_V_MSG(!_is_active(), 0, "The multiplayer instance isn't currently active.");
	return unique_id;
}

void ENetMultiplayerPeer::set_refuse_new_connections(bool p_enabled) {
#ifdef GODOT_ENET
	if (_is_active()) {
		for (KeyValue<int, Ref<ENetConnection>> &E : hosts) {
			E.value->refuse_new_connections(p_enabled);
		}
	}
#endif
	MultiplayerPeer::set_refuse_new_connections(p_enabled);
}

void ENetMultiplayerPeer::set_server_relay_enabled(bool p_enabled) {
	ERR_FAIL_COND_MSG(_is_active(), "Server relaying can't be toggled while the multiplayer instance is active.");

	server_relay = p_enabled;
}

bool ENetMultiplayerPeer::is_server_relay_enabled() const {
	return server_relay;
}

Ref<ENetConnection> ENetMultiplayerPeer::get_host() const {
	ERR_FAIL_COND_V(!_is_active(), nullptr);
	ERR_FAIL_COND_V(active_mode == MODE_MESH, nullptr);
	return hosts[0];
}

Ref<ENetPacketPeer> ENetMultiplayerPeer::get_peer(int p_id) const {
	ERR_FAIL_COND_V(!_is_active(), nullptr);
	ERR_FAIL_COND_V(!peers.has(p_id), nullptr);
	ERR_FAIL_COND_V(active_mode == MODE_CLIENT && p_id != 1, nullptr);
	return peers[p_id];
}

void ENetMultiplayerPeer::_destroy_unused(ENetPacket *p_packet) {
	if (p_packet->referenceCount == 0) {
		enet_packet_destroy(p_packet);
	}
}

void ENetMultiplayerPeer::_relay(int p_from, int p_to, enet_uint8 p_channel, ENetPacket *p_packet) {
	if (p_to == 0) {
		// Re-send to everyone but sender :|
		for (KeyValue<int, Ref<ENetPacketPeer>> &E : peers) {
			if (E.key == p_from) {
				continue;
			}

			E.value->send(p_channel, p_packet);
		}
	} else if (p_to < 0) {
		// Re-send to everyone but excluded and sender.
		for (KeyValue<int, Ref<ENetPacketPeer>> &E : peers) {
			if (E.key == p_from || E.key == -p_to) { // Do not resend to self, also do not send to excluded
				continue;
			}

			E.value->send(p_channel, p_packet);
		}
	} else {
		// To someone else, specifically
		ERR_FAIL_COND(!peers.has(p_to));
		ENetPacket *packet = enet_packet_create(p_packet->data, p_packet->dataLength, p_packet->flags);
		peers[p_to]->send(p_channel, packet);
	}
}

void ENetMultiplayerPeer::_notify_peers(int p_id, bool p_connected) {
	if (p_connected) {
		ERR_FAIL_COND(!peers.has(p_id));
		// Someone connected, notify all the peers available.
		Ref<ENetPacketPeer> peer = peers[p_id];
		ENetPacket *packet = enet_packet_create(nullptr, 8, ENET_PACKET_FLAG_RELIABLE);
		encode_uint32(SYSMSG_ADD_PEER, &packet->data[0]);
		encode_uint32(p_id, &packet->data[4]);
		for (KeyValue<int, Ref<ENetPacketPeer>> &E : peers) {
			if (E.key == p_id) {
				continue;
			}
			// Send new peer to existing peer.
			E.value->send(SYSCH_CONFIG, packet);
			// Send existing peer to new peer.
			// This packet will be automatically destroyed by ENet after send.
			ENetPacket *packet2 = enet_packet_create(nullptr, 8, ENET_PACKET_FLAG_RELIABLE);
			encode_uint32(SYSMSG_ADD_PEER, &packet2->data[0]);
			encode_uint32(E.key, &packet2->data[4]);
			peer->send(SYSCH_CONFIG, packet2);
		}
		_destroy_unused(packet);
	} else {
		// Server just received a client disconnect and is in relay mode, notify everyone else.
		ENetPacket *packet = enet_packet_create(nullptr, 8, ENET_PACKET_FLAG_RELIABLE);
		encode_uint32(SYSMSG_REMOVE_PEER, &packet->data[0]);
		encode_uint32(p_id, &packet->data[4]);
		for (KeyValue<int, Ref<ENetPacketPeer>> &E : peers) {
			if (E.key == p_id) {
				continue;
			}
			E.value->send(SYSCH_CONFIG, packet);
		}
		_destroy_unused(packet);
	}
}

void ENetMultiplayerPeer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("create_server", "port", "max_clients", "max_channels", "in_bandwidth", "out_bandwidth"), &ENetMultiplayerPeer::create_server, DEFVAL(32), DEFVAL(0), DEFVAL(0), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("create_client", "address", "port", "channel_count", "in_bandwidth", "out_bandwidth", "local_port"), &ENetMultiplayerPeer::create_client, DEFVAL(0), DEFVAL(0), DEFVAL(0), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("create_mesh", "unique_id"), &ENetMultiplayerPeer::create_mesh);
	ClassDB::bind_method(D_METHOD("add_mesh_peer", "peer_id", "host"), &ENetMultiplayerPeer::add_mesh_peer);
	ClassDB::bind_method(D_METHOD("close_connection", "wait_usec"), &ENetMultiplayerPeer::close_connection, DEFVAL(100));
	ClassDB::bind_method(D_METHOD("set_bind_ip", "ip"), &ENetMultiplayerPeer::set_bind_ip);

	ClassDB::bind_method(D_METHOD("set_server_relay_enabled", "enabled"), &ENetMultiplayerPeer::set_server_relay_enabled);
	ClassDB::bind_method(D_METHOD("is_server_relay_enabled"), &ENetMultiplayerPeer::is_server_relay_enabled);
	ClassDB::bind_method(D_METHOD("get_host"), &ENetMultiplayerPeer::get_host);
	ClassDB::bind_method(D_METHOD("get_peer", "id"), &ENetMultiplayerPeer::get_peer);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "server_relay"), "set_server_relay_enabled", "is_server_relay_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "host", PROPERTY_HINT_RESOURCE_TYPE, "ENetConnection", PROPERTY_USAGE_NONE), "", "get_host");
}

ENetMultiplayerPeer::ENetMultiplayerPeer() {
	bind_ip = IPAddress("*");
}

ENetMultiplayerPeer::~ENetMultiplayerPeer() {
	if (_is_active()) {
		close_connection();
	}
}

// Sets IP for ENet to bind when using create_server or create_client
// if no IP is set, then ENet bind to ENET_HOST_ANY
void ENetMultiplayerPeer::set_bind_ip(const IPAddress &p_ip) {
	ERR_FAIL_COND_MSG(!p_ip.is_valid() && !p_ip.is_wildcard(), vformat("Invalid bind IP address: %s", String(p_ip)));

	bind_ip = p_ip;
}
