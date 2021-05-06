/*************************************************************************/
/*  networked_multiplayer_enet.cpp                                       */
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

#include "networked_multiplayer_enet.h"
#include "core/io/ip.h"
#include "core/io/marshalls.h"
#include "core/os/os.h"

void NetworkedMultiplayerENet::set_transfer_mode(TransferMode p_mode) {
	transfer_mode = p_mode;
}

NetworkedMultiplayerPeer::TransferMode NetworkedMultiplayerENet::get_transfer_mode() const {
	return transfer_mode;
}

void NetworkedMultiplayerENet::set_target_peer(int p_peer) {
	target_peer = p_peer;
}

int NetworkedMultiplayerENet::get_packet_peer() const {
	ERR_FAIL_COND_V_MSG(!active, 1, "The multiplayer instance isn't currently active.");
	ERR_FAIL_COND_V(incoming_packets.size() == 0, 1);

	return incoming_packets.front()->get().from;
}

int NetworkedMultiplayerENet::get_packet_channel() const {
	ERR_FAIL_COND_V_MSG(!active, -1, "The multiplayer instance isn't currently active.");
	ERR_FAIL_COND_V(incoming_packets.size() == 0, -1);

	return incoming_packets.front()->get().channel;
}

int NetworkedMultiplayerENet::get_last_packet_channel() const {
	ERR_FAIL_COND_V_MSG(!active, -1, "The multiplayer instance isn't currently active.");
	ERR_FAIL_COND_V(!current_packet.packet, -1);

	return current_packet.channel;
}

Error NetworkedMultiplayerENet::create_server(int p_port, int p_max_clients, int p_in_bandwidth, int p_out_bandwidth) {
	ERR_FAIL_COND_V_MSG(active, ERR_ALREADY_IN_USE, "The multiplayer instance is already active.");
	ERR_FAIL_COND_V_MSG(p_port < 0 || p_port > 65535, ERR_INVALID_PARAMETER, "The local port number must be between 0 and 65535 (inclusive).");
	ERR_FAIL_COND_V_MSG(p_max_clients < 1 || p_max_clients > 4095, ERR_INVALID_PARAMETER, "The number of clients must be set between 1 and 4095 (inclusive).");
	ERR_FAIL_COND_V_MSG(p_in_bandwidth < 0, ERR_INVALID_PARAMETER, "The incoming bandwidth limit must be greater than or equal to 0 (0 disables the limit).");
	ERR_FAIL_COND_V_MSG(p_out_bandwidth < 0, ERR_INVALID_PARAMETER, "The outgoing bandwidth limit must be greater than or equal to 0 (0 disables the limit).");
	ERR_FAIL_COND_V(dtls_enabled && (dtls_key.is_null() || dtls_cert.is_null()), ERR_INVALID_PARAMETER);

	ENetAddress address;
	memset(&address, 0, sizeof(address));

#ifdef GODOT_ENET
	if (bind_ip.is_wildcard()) {
		address.wildcard = 1;
	} else {
		enet_address_set_ip(&address, bind_ip.get_ipv6(), 16);
	}
#else
	if (bind_ip.is_wildcard()) {
		address.host = 0;
	} else {
		ERR_FAIL_COND_V(!bind_ip.is_ipv4(), ERR_INVALID_PARAMETER);
		address.host = *(uint32_t *)bind_ip.get_ipv4();
	}
#endif
	address.port = p_port;

	host = enet_host_create(&address /* the address to bind the server host to */,
			p_max_clients /* allow up to 32 clients and/or outgoing connections */,
			channel_count /* allow up to channel_count to be used */,
			p_in_bandwidth /* limit incoming bandwidth if > 0 */,
			p_out_bandwidth /* limit outgoing bandwidth if > 0 */);

	ERR_FAIL_COND_V_MSG(!host, ERR_CANT_CREATE, "Couldn't create an ENet multiplayer server.");
#ifdef GODOT_ENET
	if (dtls_enabled) {
		enet_host_dtls_server_setup(host, dtls_key.ptr(), dtls_cert.ptr());
	}
	enet_host_refuse_new_connections(host, refuse_connections);
#endif

	_setup_compressor();
	active = true;
	server = true;
	refuse_connections = false;
	unique_id = 1;
	connection_status = CONNECTION_CONNECTED;
	return OK;
}
Error NetworkedMultiplayerENet::create_client(const String &p_address, int p_port, int p_in_bandwidth, int p_out_bandwidth, int p_local_port) {
	ERR_FAIL_COND_V_MSG(active, ERR_ALREADY_IN_USE, "The multiplayer instance is already active.");
	ERR_FAIL_COND_V_MSG(p_port < 1 || p_port > 65535, ERR_INVALID_PARAMETER, "The remote port number must be between 1 and 65535 (inclusive).");
	ERR_FAIL_COND_V_MSG(p_local_port < 0 || p_local_port > 65535, ERR_INVALID_PARAMETER, "The local port number must be between 0 and 65535 (inclusive).");
	ERR_FAIL_COND_V_MSG(p_in_bandwidth < 0, ERR_INVALID_PARAMETER, "The incoming bandwidth limit must be greater than or equal to 0 (0 disables the limit).");
	ERR_FAIL_COND_V_MSG(p_out_bandwidth < 0, ERR_INVALID_PARAMETER, "The outgoing bandwidth limit must be greater than or equal to 0 (0 disables the limit).");

	ENetAddress c_client;

#ifdef GODOT_ENET
	if (bind_ip.is_wildcard()) {
		c_client.wildcard = 1;
	} else {
		enet_address_set_ip(&c_client, bind_ip.get_ipv6(), 16);
	}
#else
	if (bind_ip.is_wildcard()) {
		c_client.host = 0;
	} else {
		ERR_FAIL_COND_V_MSG(!bind_ip.is_ipv4(), ERR_INVALID_PARAMETER, "Wildcard IP addresses are only permitted in IPv4, not IPv6.");
		c_client.host = *(uint32_t *)bind_ip.get_ipv4();
	}
#endif

	c_client.port = p_local_port;

	host = enet_host_create(&c_client /* create a client host */,
			1 /* only allow 1 outgoing connection */,
			channel_count /* allow up to channel_count to be used */,
			p_in_bandwidth /* limit incoming bandwidth if > 0 */,
			p_out_bandwidth /* limit outgoing bandwidth if > 0 */);

	ERR_FAIL_COND_V_MSG(!host, ERR_CANT_CREATE, "Couldn't create the ENet client host.");
#ifdef GODOT_ENET
	if (dtls_enabled) {
		enet_host_dtls_client_setup(host, dtls_cert.ptr(), dtls_verify, p_address.utf8().get_data());
	}
	enet_host_refuse_new_connections(host, refuse_connections);
#endif

	_setup_compressor();

	IPAddress ip;
	if (p_address.is_valid_ip_address()) {
		ip = p_address;
	} else {
#ifdef GODOT_ENET
		ip = IP::get_singleton()->resolve_hostname(p_address);
#else
		ip = IP::get_singleton()->resolve_hostname(p_address, IP::TYPE_IPV4);
#endif

		ERR_FAIL_COND_V_MSG(!ip.is_valid(), ERR_CANT_RESOLVE, "Couldn't resolve the server IP address or domain name.");
	}

	ENetAddress address;
#ifdef GODOT_ENET
	enet_address_set_ip(&address, ip.get_ipv6(), 16);
#else
	ERR_FAIL_COND_V_MSG(!ip.is_ipv4(), ERR_INVALID_PARAMETER, "Connecting to an IPv6 server isn't supported when using vanilla ENet. Recompile Godot with the bundled ENet library.");
	address.host = *(uint32_t *)ip.get_ipv4();
#endif
	address.port = p_port;

	unique_id = _gen_unique_id();

	// Initiate connection, allocating enough channels
	ENetPeer *peer = enet_host_connect(host, &address, channel_count, unique_id);

	if (peer == nullptr) {
		enet_host_destroy(host);
		ERR_FAIL_COND_V_MSG(!peer, ERR_CANT_CREATE, "Couldn't connect to the ENet multiplayer server.");
	}

	// Technically safe to ignore the peer or anything else.

	connection_status = CONNECTION_CONNECTING;
	active = true;
	server = false;
	refuse_connections = false;

	return OK;
}

void NetworkedMultiplayerENet::poll() {
	ERR_FAIL_COND_MSG(!active, "The multiplayer instance isn't currently active.");

	_pop_current_packet();

	ENetEvent event;
	/* Keep servicing until there are no available events left in queue. */
	while (true) {
		if (!host || !active) { // Might have been disconnected while emitting a notification
			return;
		}

		int ret = enet_host_service(host, &event, 0);

		if (ret < 0) {
			// Error, do something?
			break;
		} else if (ret == 0) {
			break;
		}

		switch (event.type) {
			case ENET_EVENT_TYPE_CONNECT: {
				// Store any relevant client information here.

				if (server && refuse_connections) {
					enet_peer_reset(event.peer);
					break;
				}

				// A client joined with an invalid ID (negative values, 0, and 1 are reserved).
				// Probably trying to exploit us.
				if (server && ((int)event.data < 2 || peer_map.has((int)event.data))) {
					enet_peer_reset(event.peer);
					ERR_CONTINUE(true);
				}

				int *new_id = memnew(int);
				*new_id = event.data;

				if (*new_id == 0) { // Data zero is sent by server (ENet won't let you configure this). Server is always 1.
					*new_id = 1;
				}

				event.peer->data = new_id;

				peer_map[*new_id] = event.peer;

				connection_status = CONNECTION_CONNECTED; // If connecting, this means it connected to something!

				emit_signal("peer_connected", *new_id);

				if (server) {
					// Do not notify other peers when server_relay is disabled.
					if (!server_relay) {
						break;
					}

					// Someone connected, notify all the peers available
					for (Map<int, ENetPeer *>::Element *E = peer_map.front(); E; E = E->next()) {
						if (E->key() == *new_id) {
							continue;
						}
						// Send existing peers to new peer
						ENetPacket *packet = enet_packet_create(nullptr, 8, ENET_PACKET_FLAG_RELIABLE);
						encode_uint32(SYSMSG_ADD_PEER, &packet->data[0]);
						encode_uint32(E->key(), &packet->data[4]);
						enet_peer_send(event.peer, SYSCH_CONFIG, packet);
						// Send the new peer to existing peers
						packet = enet_packet_create(nullptr, 8, ENET_PACKET_FLAG_RELIABLE);
						encode_uint32(SYSMSG_ADD_PEER, &packet->data[0]);
						encode_uint32(*new_id, &packet->data[4]);
						enet_peer_send(E->get(), SYSCH_CONFIG, packet);
					}
				} else {
					emit_signal("connection_succeeded");
				}

			} break;
			case ENET_EVENT_TYPE_DISCONNECT: {
				// Reset the peer's client information.

				int *id = (int *)event.peer->data;

				if (!id) {
					if (!server) {
						emit_signal("connection_failed");
					}
					// Never fully connected.
					break;
				}

				if (!server) {
					// Client just disconnected from server.
					emit_signal("server_disconnected");
					close_connection();
					return;
				} else if (server_relay) {
					// Server just received a client disconnect and is in relay mode, notify everyone else.
					for (Map<int, ENetPeer *>::Element *E = peer_map.front(); E; E = E->next()) {
						if (E->key() == *id) {
							continue;
						}

						ENetPacket *packet = enet_packet_create(nullptr, 8, ENET_PACKET_FLAG_RELIABLE);
						encode_uint32(SYSMSG_REMOVE_PEER, &packet->data[0]);
						encode_uint32(*id, &packet->data[4]);
						enet_peer_send(E->get(), SYSCH_CONFIG, packet);
					}
				}

				emit_signal("peer_disconnected", *id);
				peer_map.erase(*id);
				memdelete(id);
			} break;
			case ENET_EVENT_TYPE_RECEIVE: {
				if (event.channelID == SYSCH_CONFIG) {
					// Some config message
					ERR_CONTINUE(event.packet->dataLength < 8);

					// Only server can send config messages
					ERR_CONTINUE(server);

					int msg = decode_uint32(&event.packet->data[0]);
					int id = decode_uint32(&event.packet->data[4]);

					switch (msg) {
						case SYSMSG_ADD_PEER: {
							peer_map[id] = nullptr;
							emit_signal("peer_connected", id);

						} break;
						case SYSMSG_REMOVE_PEER: {
							peer_map.erase(id);
							emit_signal("peer_disconnected", id);
						} break;
					}

					enet_packet_destroy(event.packet);
				} else if (event.channelID < channel_count) {
					Packet packet;
					packet.packet = event.packet;

					uint32_t *id = (uint32_t *)event.peer->data;

					ERR_CONTINUE(event.packet->dataLength < 8);

					uint32_t source = decode_uint32(&event.packet->data[0]);
					int target = decode_uint32(&event.packet->data[4]);

					packet.from = source;
					packet.channel = event.channelID;

					if (server) {
						// Someone is cheating and trying to fake the source!
						ERR_CONTINUE(source != *id);

						packet.from = *id;

						if (target == 1) {
							// To myself and only myself
							incoming_packets.push_back(packet);
						} else if (!server_relay) {
							// No other destination is allowed when server is not relaying
							continue;
						} else if (target == 0) {
							// Re-send to everyone but sender :|

							incoming_packets.push_back(packet);
							// And make copies for sending
							for (Map<int, ENetPeer *>::Element *E = peer_map.front(); E; E = E->next()) {
								if (uint32_t(E->key()) == source) { // Do not resend to self
									continue;
								}

								ENetPacket *packet2 = enet_packet_create(packet.packet->data, packet.packet->dataLength, packet.packet->flags);

								enet_peer_send(E->get(), event.channelID, packet2);
							}

						} else if (target < 0) {
							// To all but one

							// And make copies for sending
							for (Map<int, ENetPeer *>::Element *E = peer_map.front(); E; E = E->next()) {
								if (uint32_t(E->key()) == source || E->key() == -target) { // Do not resend to self, also do not send to excluded
									continue;
								}

								ENetPacket *packet2 = enet_packet_create(packet.packet->data, packet.packet->dataLength, packet.packet->flags);

								enet_peer_send(E->get(), event.channelID, packet2);
							}

							if (-target != 1) {
								// Server is not excluded
								incoming_packets.push_back(packet);
							} else {
								// Server is excluded, erase packet
								enet_packet_destroy(packet.packet);
							}

						} else {
							// To someone else, specifically
							ERR_CONTINUE(!peer_map.has(target));
							enet_peer_send(peer_map[target], event.channelID, packet.packet);
						}
					} else {
						incoming_packets.push_back(packet);
					}

					// Destroy packet later
				} else {
					ERR_CONTINUE(true);
				}

			} break;
			case ENET_EVENT_TYPE_NONE: {
				// Do nothing
			} break;
		}
	}
}

bool NetworkedMultiplayerENet::is_server() const {
	ERR_FAIL_COND_V_MSG(!active, false, "The multiplayer instance isn't currently active.");

	return server;
}

void NetworkedMultiplayerENet::close_connection(uint32_t wait_usec) {
	ERR_FAIL_COND_MSG(!active, "The multiplayer instance isn't currently active.");

	_pop_current_packet();

	bool peers_disconnected = false;
	for (Map<int, ENetPeer *>::Element *E = peer_map.front(); E; E = E->next()) {
		if (E->get()) {
			enet_peer_disconnect_now(E->get(), unique_id);
			int *id = (int *)(E->get()->data);
			memdelete(id);
			peers_disconnected = true;
		}
	}

	if (peers_disconnected) {
		enet_host_flush(host);

		if (wait_usec > 0) {
			OS::get_singleton()->delay_usec(wait_usec); // Wait for disconnection packets to send
		}
	}

	enet_host_destroy(host);
	active = false;
	incoming_packets.clear();
	peer_map.clear();
	unique_id = 1; // Server is 1
	connection_status = CONNECTION_DISCONNECTED;
}

void NetworkedMultiplayerENet::disconnect_peer(int p_peer, bool now) {
	ERR_FAIL_COND_MSG(!active, "The multiplayer instance isn't currently active.");
	ERR_FAIL_COND_MSG(!is_server(), "Can't disconnect a peer when not acting as a server.");
	ERR_FAIL_COND_MSG(!peer_map.has(p_peer), vformat("Peer ID %d not found in the list of peers.", p_peer));

	if (now) {
		int *id = (int *)peer_map[p_peer]->data;
		enet_peer_disconnect_now(peer_map[p_peer], 0);

		// enet_peer_disconnect_now doesn't generate ENET_EVENT_TYPE_DISCONNECT,
		// notify everyone else, send disconnect signal & remove from peer_map like in poll()
		if (server_relay) {
			for (Map<int, ENetPeer *>::Element *E = peer_map.front(); E; E = E->next()) {
				if (E->key() == p_peer) {
					continue;
				}

				ENetPacket *packet = enet_packet_create(nullptr, 8, ENET_PACKET_FLAG_RELIABLE);
				encode_uint32(SYSMSG_REMOVE_PEER, &packet->data[0]);
				encode_uint32(p_peer, &packet->data[4]);
				enet_peer_send(E->get(), SYSCH_CONFIG, packet);
			}
		}

		if (id) {
			memdelete(id);
		}

		emit_signal("peer_disconnected", p_peer);
		peer_map.erase(p_peer);
	} else {
		enet_peer_disconnect_later(peer_map[p_peer], 0);
	}
}

int NetworkedMultiplayerENet::get_available_packet_count() const {
	return incoming_packets.size();
}

Error NetworkedMultiplayerENet::get_packet(const uint8_t **r_buffer, int &r_buffer_size) {
	ERR_FAIL_COND_V_MSG(incoming_packets.size() == 0, ERR_UNAVAILABLE, "No incoming packets available.");

	_pop_current_packet();

	current_packet = incoming_packets.front()->get();
	incoming_packets.pop_front();

	*r_buffer = (const uint8_t *)(&current_packet.packet->data[8]);
	r_buffer_size = current_packet.packet->dataLength - 8;

	return OK;
}

Error NetworkedMultiplayerENet::put_packet(const uint8_t *p_buffer, int p_buffer_size) {
	ERR_FAIL_COND_V_MSG(!active, ERR_UNCONFIGURED, "The multiplayer instance isn't currently active.");
	ERR_FAIL_COND_V_MSG(connection_status != CONNECTION_CONNECTED, ERR_UNCONFIGURED, "The multiplayer instance isn't currently connected to any server or client.");

	int packet_flags = 0;
	int channel = SYSCH_RELIABLE;

	switch (transfer_mode) {
		case TRANSFER_MODE_UNRELIABLE: {
			if (always_ordered) {
				packet_flags = 0;
			} else {
				packet_flags = ENET_PACKET_FLAG_UNSEQUENCED;
			}
			channel = SYSCH_UNRELIABLE;
		} break;
		case TRANSFER_MODE_UNRELIABLE_ORDERED: {
			packet_flags = 0;
			channel = SYSCH_UNRELIABLE;
		} break;
		case TRANSFER_MODE_RELIABLE: {
			packet_flags = ENET_PACKET_FLAG_RELIABLE;
			channel = SYSCH_RELIABLE;
		} break;
	}

	if (transfer_channel > SYSCH_CONFIG) {
		channel = transfer_channel;
	}

	Map<int, ENetPeer *>::Element *E = nullptr;

	if (target_peer != 0) {
		E = peer_map.find(ABS(target_peer));
		ERR_FAIL_COND_V_MSG(!E, ERR_INVALID_PARAMETER, vformat("Invalid target peer: %d", target_peer));
	}

	ENetPacket *packet = enet_packet_create(nullptr, p_buffer_size + 8, packet_flags);
	encode_uint32(unique_id, &packet->data[0]); // Source ID
	encode_uint32(target_peer, &packet->data[4]); // Dest ID
	memcpy(&packet->data[8], p_buffer, p_buffer_size);

	if (server) {
		if (target_peer == 0) {
			enet_host_broadcast(host, channel, packet);
		} else if (target_peer < 0) {
			// Send to all but one
			// and make copies for sending

			int exclude = -target_peer;

			for (Map<int, ENetPeer *>::Element *F = peer_map.front(); F; F = F->next()) {
				if (F->key() == exclude) { // Exclude packet
					continue;
				}

				ENetPacket *packet2 = enet_packet_create(packet->data, packet->dataLength, packet_flags);

				enet_peer_send(F->get(), channel, packet2);
			}

			enet_packet_destroy(packet); // Original packet no longer needed
		} else {
			enet_peer_send(E->get(), channel, packet);
		}
	} else {
		ERR_FAIL_COND_V(!peer_map.has(1), ERR_BUG);
		enet_peer_send(peer_map[1], channel, packet); // Send to server for broadcast
	}

	enet_host_flush(host);

	return OK;
}

int NetworkedMultiplayerENet::get_max_packet_size() const {
	return 1 << 24; // Anything is good
}

void NetworkedMultiplayerENet::_pop_current_packet() {
	if (current_packet.packet) {
		enet_packet_destroy(current_packet.packet);
		current_packet.packet = nullptr;
		current_packet.from = 0;
		current_packet.channel = -1;
	}
}

NetworkedMultiplayerPeer::ConnectionStatus NetworkedMultiplayerENet::get_connection_status() const {
	return connection_status;
}

uint32_t NetworkedMultiplayerENet::_gen_unique_id() const {
	uint32_t hash = 0;

	while (hash == 0 || hash == 1) {
		hash = hash_djb2_one_32(
				(uint32_t)OS::get_singleton()->get_ticks_usec());
		hash = hash_djb2_one_32(
				(uint32_t)OS::get_singleton()->get_unix_time(), hash);
		hash = hash_djb2_one_32(
				(uint32_t)OS::get_singleton()->get_user_data_dir().hash64(), hash);
		hash = hash_djb2_one_32(
				(uint32_t)((uint64_t)this), hash); // Rely on ASLR heap
		hash = hash_djb2_one_32(
				(uint32_t)((uint64_t)&hash), hash); // Rely on ASLR stack

		hash = hash & 0x7FFFFFFF; // Make it compatible with unsigned, since negative ID is used for exclusion
	}

	return hash;
}

int NetworkedMultiplayerENet::get_unique_id() const {
	ERR_FAIL_COND_V_MSG(!active, 0, "The multiplayer instance isn't currently active.");
	return unique_id;
}

void NetworkedMultiplayerENet::set_refuse_new_connections(bool p_enable) {
	refuse_connections = p_enable;
#ifdef GODOT_ENET
	if (active) {
		enet_host_refuse_new_connections(host, p_enable);
	}
#endif
}

bool NetworkedMultiplayerENet::is_refusing_new_connections() const {
	return refuse_connections;
}

void NetworkedMultiplayerENet::set_compression_mode(CompressionMode p_mode) {
	compression_mode = p_mode;
}

NetworkedMultiplayerENet::CompressionMode NetworkedMultiplayerENet::get_compression_mode() const {
	return compression_mode;
}

size_t NetworkedMultiplayerENet::enet_compress(void *context, const ENetBuffer *inBuffers, size_t inBufferCount, size_t inLimit, enet_uint8 *outData, size_t outLimit) {
	NetworkedMultiplayerENet *enet = (NetworkedMultiplayerENet *)(context);

	if (size_t(enet->src_compressor_mem.size()) < inLimit) {
		enet->src_compressor_mem.resize(inLimit);
	}

	int total = inLimit;
	int ofs = 0;
	while (total) {
		for (size_t i = 0; i < inBufferCount; i++) {
			int to_copy = MIN(total, int(inBuffers[i].dataLength));
			memcpy(&enet->src_compressor_mem.write[ofs], inBuffers[i].data, to_copy);
			ofs += to_copy;
			total -= to_copy;
		}
	}

	Compression::Mode mode;

	switch (enet->compression_mode) {
		case COMPRESS_FASTLZ: {
			mode = Compression::MODE_FASTLZ;
		} break;
		case COMPRESS_ZLIB: {
			mode = Compression::MODE_DEFLATE;
		} break;
		case COMPRESS_ZSTD: {
			mode = Compression::MODE_ZSTD;
		} break;
		default: {
			ERR_FAIL_V_MSG(0, vformat("Invalid ENet compression mode: %d", enet->compression_mode));
		}
	}

	int req_size = Compression::get_max_compressed_buffer_size(ofs, mode);
	if (enet->dst_compressor_mem.size() < req_size) {
		enet->dst_compressor_mem.resize(req_size);
	}
	int ret = Compression::compress(enet->dst_compressor_mem.ptrw(), enet->src_compressor_mem.ptr(), ofs, mode);

	if (ret < 0) {
		return 0;
	}

	if (ret > int(outLimit)) {
		return 0; // Do not bother
	}

	memcpy(outData, enet->dst_compressor_mem.ptr(), ret);

	return ret;
}

size_t NetworkedMultiplayerENet::enet_decompress(void *context, const enet_uint8 *inData, size_t inLimit, enet_uint8 *outData, size_t outLimit) {
	NetworkedMultiplayerENet *enet = (NetworkedMultiplayerENet *)(context);
	int ret = -1;
	switch (enet->compression_mode) {
		case COMPRESS_FASTLZ: {
			ret = Compression::decompress(outData, outLimit, inData, inLimit, Compression::MODE_FASTLZ);
		} break;
		case COMPRESS_ZLIB: {
			ret = Compression::decompress(outData, outLimit, inData, inLimit, Compression::MODE_DEFLATE);
		} break;
		case COMPRESS_ZSTD: {
			ret = Compression::decompress(outData, outLimit, inData, inLimit, Compression::MODE_ZSTD);
		} break;
		default: {
		}
	}
	if (ret < 0) {
		return 0;
	} else {
		return ret;
	}
}

void NetworkedMultiplayerENet::_setup_compressor() {
	switch (compression_mode) {
		case COMPRESS_NONE: {
			enet_host_compress(host, nullptr);
		} break;
		case COMPRESS_RANGE_CODER: {
			enet_host_compress_with_range_coder(host);
		} break;
		case COMPRESS_FASTLZ:
		case COMPRESS_ZLIB:
		case COMPRESS_ZSTD: {
			enet_host_compress(host, &enet_compressor);
		} break;
	}
}

void NetworkedMultiplayerENet::enet_compressor_destroy(void *context) {
	// Nothing to do
}

IPAddress NetworkedMultiplayerENet::get_peer_address(int p_peer_id) const {
	ERR_FAIL_COND_V_MSG(!peer_map.has(p_peer_id), IPAddress(), vformat("Peer ID %d not found in the list of peers.", p_peer_id));
	ERR_FAIL_COND_V_MSG(!is_server() && p_peer_id != 1, IPAddress(), "Can't get the address of peers other than the server (ID -1) when acting as a client.");
	ERR_FAIL_COND_V_MSG(peer_map[p_peer_id] == nullptr, IPAddress(), vformat("Peer ID %d found in the list of peers, but is null.", p_peer_id));

	IPAddress out;
#ifdef GODOT_ENET
	out.set_ipv6((uint8_t *)&(peer_map[p_peer_id]->address.host));
#else
	out.set_ipv4((uint8_t *)&(peer_map[p_peer_id]->address.host));
#endif

	return out;
}

int NetworkedMultiplayerENet::get_peer_port(int p_peer_id) const {
	ERR_FAIL_COND_V_MSG(!peer_map.has(p_peer_id), 0, vformat("Peer ID %d not found in the list of peers.", p_peer_id));
	ERR_FAIL_COND_V_MSG(!is_server() && p_peer_id != 1, 0, "Can't get the address of peers other than the server (ID -1) when acting as a client.");
	ERR_FAIL_COND_V_MSG(peer_map[p_peer_id] == nullptr, 0, vformat("Peer ID %d found in the list of peers, but is null.", p_peer_id));
#ifdef GODOT_ENET
	return peer_map[p_peer_id]->address.port;
#else
	return peer_map[p_peer_id]->address.port;
#endif
}

int NetworkedMultiplayerENet::get_local_port() const {
	ERR_FAIL_COND_V_MSG(!active || !host, 0, "The multiplayer instance isn't currently active.");
	return host->address.port;
}

void NetworkedMultiplayerENet::set_peer_timeout(int p_peer_id, int p_timeout_limit, int p_timeout_min, int p_timeout_max) {
	ERR_FAIL_COND_MSG(!peer_map.has(p_peer_id), vformat("Peer ID %d not found in the list of peers.", p_peer_id));
	ERR_FAIL_COND_MSG(!is_server() && p_peer_id != 1, "Can't change the timeout of peers other then the server when acting as a client.");
	ERR_FAIL_COND_MSG(peer_map[p_peer_id] == nullptr, vformat("Peer ID %d found in the list of peers, but is null.", p_peer_id));
	ERR_FAIL_COND_MSG(p_timeout_limit > p_timeout_min || p_timeout_min > p_timeout_max, "Timeout limit must be less than minimum timeout, which itself must be less then maximum timeout");
	enet_peer_timeout(peer_map[p_peer_id], p_timeout_limit, p_timeout_min, p_timeout_max);
}

void NetworkedMultiplayerENet::set_transfer_channel(int p_channel) {
	ERR_FAIL_COND_MSG(p_channel < -1 || p_channel >= channel_count, vformat("The transfer channel must be set between 0 and %d, inclusive (got %d).", channel_count - 1, p_channel));
	ERR_FAIL_COND_MSG(p_channel == SYSCH_CONFIG, vformat("The channel %d is reserved.", SYSCH_CONFIG));
	transfer_channel = p_channel;
}

int NetworkedMultiplayerENet::get_transfer_channel() const {
	return transfer_channel;
}

void NetworkedMultiplayerENet::set_channel_count(int p_channel) {
	ERR_FAIL_COND_MSG(active, "The channel count can't be set while the multiplayer instance is active.");
	ERR_FAIL_COND_MSG(p_channel < SYSCH_MAX, vformat("The channel count must be greater than or equal to %d to account for reserved channels (got %d).", SYSCH_MAX, p_channel));
	channel_count = p_channel;
}

int NetworkedMultiplayerENet::get_channel_count() const {
	return channel_count;
}

void NetworkedMultiplayerENet::set_always_ordered(bool p_ordered) {
	always_ordered = p_ordered;
}

bool NetworkedMultiplayerENet::is_always_ordered() const {
	return always_ordered;
}

void NetworkedMultiplayerENet::set_server_relay_enabled(bool p_enabled) {
	ERR_FAIL_COND_MSG(active, "Server relaying can't be toggled while the multiplayer instance is active.");

	server_relay = p_enabled;
}

bool NetworkedMultiplayerENet::is_server_relay_enabled() const {
	return server_relay;
}

void NetworkedMultiplayerENet::_bind_methods() {
	ClassDB::bind_method(D_METHOD("create_server", "port", "max_clients", "in_bandwidth", "out_bandwidth"), &NetworkedMultiplayerENet::create_server, DEFVAL(32), DEFVAL(0), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("create_client", "address", "port", "in_bandwidth", "out_bandwidth", "local_port"), &NetworkedMultiplayerENet::create_client, DEFVAL(0), DEFVAL(0), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("close_connection", "wait_usec"), &NetworkedMultiplayerENet::close_connection, DEFVAL(100));
	ClassDB::bind_method(D_METHOD("disconnect_peer", "id", "now"), &NetworkedMultiplayerENet::disconnect_peer, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("set_compression_mode", "mode"), &NetworkedMultiplayerENet::set_compression_mode);
	ClassDB::bind_method(D_METHOD("get_compression_mode"), &NetworkedMultiplayerENet::get_compression_mode);
	ClassDB::bind_method(D_METHOD("set_bind_ip", "ip"), &NetworkedMultiplayerENet::set_bind_ip);
	ClassDB::bind_method(D_METHOD("set_dtls_enabled", "enabled"), &NetworkedMultiplayerENet::set_dtls_enabled);
	ClassDB::bind_method(D_METHOD("is_dtls_enabled"), &NetworkedMultiplayerENet::is_dtls_enabled);
	ClassDB::bind_method(D_METHOD("set_dtls_key", "key"), &NetworkedMultiplayerENet::set_dtls_key);
	ClassDB::bind_method(D_METHOD("set_dtls_certificate", "certificate"), &NetworkedMultiplayerENet::set_dtls_certificate);
	ClassDB::bind_method(D_METHOD("set_dtls_verify_enabled", "enabled"), &NetworkedMultiplayerENet::set_dtls_verify_enabled);
	ClassDB::bind_method(D_METHOD("is_dtls_verify_enabled"), &NetworkedMultiplayerENet::is_dtls_verify_enabled);
	ClassDB::bind_method(D_METHOD("get_peer_address", "id"), &NetworkedMultiplayerENet::get_peer_address);
	ClassDB::bind_method(D_METHOD("get_peer_port", "id"), &NetworkedMultiplayerENet::get_peer_port);
	ClassDB::bind_method(D_METHOD("get_local_port"), &NetworkedMultiplayerENet::get_local_port);
	ClassDB::bind_method(D_METHOD("set_peer_timeout", "id", "timeout_limit", "timeout_min", "timeout_max"), &NetworkedMultiplayerENet::set_peer_timeout);

	ClassDB::bind_method(D_METHOD("get_packet_channel"), &NetworkedMultiplayerENet::get_packet_channel);
	ClassDB::bind_method(D_METHOD("get_last_packet_channel"), &NetworkedMultiplayerENet::get_last_packet_channel);
	ClassDB::bind_method(D_METHOD("set_transfer_channel", "channel"), &NetworkedMultiplayerENet::set_transfer_channel);
	ClassDB::bind_method(D_METHOD("get_transfer_channel"), &NetworkedMultiplayerENet::get_transfer_channel);
	ClassDB::bind_method(D_METHOD("set_channel_count", "channels"), &NetworkedMultiplayerENet::set_channel_count);
	ClassDB::bind_method(D_METHOD("get_channel_count"), &NetworkedMultiplayerENet::get_channel_count);
	ClassDB::bind_method(D_METHOD("set_always_ordered", "ordered"), &NetworkedMultiplayerENet::set_always_ordered);
	ClassDB::bind_method(D_METHOD("is_always_ordered"), &NetworkedMultiplayerENet::is_always_ordered);
	ClassDB::bind_method(D_METHOD("set_server_relay_enabled", "enabled"), &NetworkedMultiplayerENet::set_server_relay_enabled);
	ClassDB::bind_method(D_METHOD("is_server_relay_enabled"), &NetworkedMultiplayerENet::is_server_relay_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "compression_mode", PROPERTY_HINT_ENUM, "None,Range Coder,FastLZ,ZLib,ZStd"), "set_compression_mode", "get_compression_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "transfer_channel"), "set_transfer_channel", "get_transfer_channel");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "channel_count"), "set_channel_count", "get_channel_count");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "always_ordered"), "set_always_ordered", "is_always_ordered");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "server_relay"), "set_server_relay_enabled", "is_server_relay_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "dtls_verify"), "set_dtls_verify_enabled", "is_dtls_verify_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_dtls"), "set_dtls_enabled", "is_dtls_enabled");

	BIND_ENUM_CONSTANT(COMPRESS_NONE);
	BIND_ENUM_CONSTANT(COMPRESS_RANGE_CODER);
	BIND_ENUM_CONSTANT(COMPRESS_FASTLZ);
	BIND_ENUM_CONSTANT(COMPRESS_ZLIB);
	BIND_ENUM_CONSTANT(COMPRESS_ZSTD);
}

NetworkedMultiplayerENet::NetworkedMultiplayerENet() {
	enet_compressor.context = this;
	enet_compressor.compress = enet_compress;
	enet_compressor.decompress = enet_decompress;
	enet_compressor.destroy = enet_compressor_destroy;

	bind_ip = IPAddress("*");
}

NetworkedMultiplayerENet::~NetworkedMultiplayerENet() {
	if (active) {
		close_connection();
	}
}

// Sets IP for ENet to bind when using create_server or create_client
// if no IP is set, then ENet bind to ENET_HOST_ANY
void NetworkedMultiplayerENet::set_bind_ip(const IPAddress &p_ip) {
	ERR_FAIL_COND_MSG(!p_ip.is_valid() && !p_ip.is_wildcard(), vformat("Invalid bind IP address: %s", String(p_ip)));

	bind_ip = p_ip;
}

void NetworkedMultiplayerENet::set_dtls_enabled(bool p_enabled) {
	ERR_FAIL_COND(active);
	dtls_enabled = p_enabled;
}

bool NetworkedMultiplayerENet::is_dtls_enabled() const {
	return dtls_enabled;
}

void NetworkedMultiplayerENet::set_dtls_verify_enabled(bool p_enabled) {
	ERR_FAIL_COND(active);
	dtls_verify = p_enabled;
}

bool NetworkedMultiplayerENet::is_dtls_verify_enabled() const {
	return dtls_verify;
}

void NetworkedMultiplayerENet::set_dtls_key(Ref<CryptoKey> p_key) {
	ERR_FAIL_COND(active);
	dtls_key = p_key;
}

void NetworkedMultiplayerENet::set_dtls_certificate(Ref<X509Certificate> p_cert) {
	ERR_FAIL_COND(active);
	dtls_cert = p_cert;
}
