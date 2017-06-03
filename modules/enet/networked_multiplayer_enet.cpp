/*************************************************************************/
/*  networked_multiplayer_enet.cpp                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "io/marshalls.h"
#include "os/os.h"

void NetworkedMultiplayerENet::set_transfer_mode(TransferMode p_mode) {

	transfer_mode = p_mode;
}

void NetworkedMultiplayerENet::set_target_peer(int p_peer) {

	target_peer = p_peer;
}

int NetworkedMultiplayerENet::get_packet_peer() const {

	ERR_FAIL_COND_V(!active, 1);
	ERR_FAIL_COND_V(incoming_packets.size() == 0, 1);

	return incoming_packets.front()->get().from;
}

Error NetworkedMultiplayerENet::create_server(int p_port, int p_max_clients, int p_in_bandwidth, int p_out_bandwidth) {

	ERR_FAIL_COND_V(active, ERR_ALREADY_IN_USE);

	ENetAddress address;

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
			SYSCH_MAX /* allow up to SYSCH_MAX channels to be used */,
			p_in_bandwidth /* assume any amount of incoming bandwidth */,
			p_out_bandwidth /* assume any amount of outgoing bandwidth */);

	ERR_FAIL_COND_V(!host, ERR_CANT_CREATE);

	_setup_compressor();
	active = true;
	server = true;
	refuse_connections = false;
	unique_id = 1;
	connection_status = CONNECTION_CONNECTED;
	return OK;
}
Error NetworkedMultiplayerENet::create_client(const IP_Address &p_ip, int p_port, int p_in_bandwidth, int p_out_bandwidth) {

	ERR_FAIL_COND_V(active, ERR_ALREADY_IN_USE);

	host = enet_host_create(NULL /* create a client host */,
			1 /* only allow 1 outgoing connection */,
			SYSCH_MAX /* allow up to SYSCH_MAX channels to be used */,
			p_in_bandwidth /* 56K modem with 56 Kbps downstream bandwidth */,
			p_out_bandwidth /* 56K modem with 14 Kbps upstream bandwidth */);

	ERR_FAIL_COND_V(!host, ERR_CANT_CREATE);

	_setup_compressor();

	ENetAddress address;
#ifdef GODOT_ENET
	enet_address_set_ip(&address, p_ip.get_ipv6(), 16);
#else
	ERR_FAIL_COND_V(!p_ip.is_ipv4(), ERR_INVALID_PARAMETER);
	address.host = *(uint32_t *)p_ip.get_ipv4();
#endif
	address.port = p_port;

	//enet_address_set_host (& address, "localhost");
	//address.port = p_port;

	unique_id = _gen_unique_id();

	/* Initiate the connection, allocating the enough channels */
	ENetPeer *peer = enet_host_connect(host, &address, SYSCH_MAX, unique_id);

	if (peer == NULL) {
		enet_host_destroy(host);
		ERR_FAIL_COND_V(!peer, ERR_CANT_CREATE);
	}

	//technically safe to ignore the peer or anything else.

	connection_status = CONNECTION_CONNECTING;
	active = true;
	server = false;
	refuse_connections = false;

	return OK;
}

void NetworkedMultiplayerENet::poll() {

	ERR_FAIL_COND(!active);

	_pop_current_packet();

	ENetEvent event;
	/* Wait up to 1000 milliseconds for an event. */
	while (true) {

		if (!host || !active) //might have been disconnected while emitting a notification
			return;

		int ret = enet_host_service(host, &event, 1);

		if (ret < 0) {
			//error, do something?
			break;
		} else if (ret == 0) {
			break;
		}

		switch (event.type) {
			case ENET_EVENT_TYPE_CONNECT: {
				/* Store any relevant client information here. */

				if (server && refuse_connections) {
					enet_peer_reset(event.peer);
					break;
				}

				int *new_id = memnew(int);
				*new_id = event.data;

				if (*new_id == 0) { //data zero is sent by server (enet won't let you configure this). Server is always 1
					*new_id = 1;
				}

				event.peer->data = new_id;

				peer_map[*new_id] = event.peer;

				connection_status = CONNECTION_CONNECTED; //if connecting, this means it connected t something!

				emit_signal("peer_connected", *new_id);

				if (server) {
					//someone connected, let it know of all the peers available
					for (Map<int, ENetPeer *>::Element *E = peer_map.front(); E; E = E->next()) {

						if (E->key() == *new_id)
							continue;
						//send existing peers to new peer
						ENetPacket *packet = enet_packet_create(NULL, 8, ENET_PACKET_FLAG_RELIABLE);
						encode_uint32(SYSMSG_ADD_PEER, &packet->data[0]);
						encode_uint32(E->key(), &packet->data[4]);
						enet_peer_send(event.peer, SYSCH_CONFIG, packet);
						//send the new peer to existing peers
						packet = enet_packet_create(NULL, 8, ENET_PACKET_FLAG_RELIABLE);
						encode_uint32(SYSMSG_ADD_PEER, &packet->data[0]);
						encode_uint32(*new_id, &packet->data[4]);
						enet_peer_send(E->get(), SYSCH_CONFIG, packet);
					}
				} else {

					emit_signal("connection_succeeded");
				}

			} break;
			case ENET_EVENT_TYPE_DISCONNECT: {

				/* Reset the peer's client information. */

				int *id = (int *)event.peer->data;

				if (!id) {
					if (!server) {
						emit_signal("connection_failed");
					}
				} else {

					if (server) {
						//someone disconnected, let it know to everyone else
						for (Map<int, ENetPeer *>::Element *E = peer_map.front(); E; E = E->next()) {

							if (E->key() == *id)
								continue;
							//send the new peer to existing peers
							ENetPacket *packet = enet_packet_create(NULL, 8, ENET_PACKET_FLAG_RELIABLE);
							encode_uint32(SYSMSG_REMOVE_PEER, &packet->data[0]);
							encode_uint32(*id, &packet->data[4]);
							enet_peer_send(E->get(), SYSCH_CONFIG, packet);
						}
					} else if (!server) {
						emit_signal("server_disconnected");
						close_connection();
						return;
					}

					emit_signal("peer_disconnected", *id);
					peer_map.erase(*id);
					memdelete(id);
				}

			} break;
			case ENET_EVENT_TYPE_RECEIVE: {

				if (event.channelID == SYSCH_CONFIG) {
					//some config message
					ERR_CONTINUE(event.packet->dataLength < 8);

					// Only server can send config messages
					ERR_CONTINUE(server);

					int msg = decode_uint32(&event.packet->data[0]);
					int id = decode_uint32(&event.packet->data[4]);

					switch (msg) {
						case SYSMSG_ADD_PEER: {

							peer_map[id] = NULL;
							emit_signal("peer_connected", id);

						} break;
						case SYSMSG_REMOVE_PEER: {

							peer_map.erase(id);
							emit_signal("peer_disconnected", id);
						} break;
					}

					enet_packet_destroy(event.packet);
				} else if (event.channelID < SYSCH_MAX) {

					Packet packet;
					packet.packet = event.packet;

					uint32_t *id = (uint32_t *)event.peer->data;

					ERR_CONTINUE(event.packet->dataLength < 12)

					uint32_t source = decode_uint32(&event.packet->data[0]);
					int target = decode_uint32(&event.packet->data[4]);
					uint32_t flags = decode_uint32(&event.packet->data[8]);

					packet.from = source;

					if (server) {
						// Someone is cheating and trying to fake the source!
						ERR_CONTINUE(source != *id);

						packet.from = *id;

						if (target == 0) {
							//re-send the everyone but sender :|

							incoming_packets.push_back(packet);
							//and make copies for sending
							for (Map<int, ENetPeer *>::Element *E = peer_map.front(); E; E = E->next()) {

								if (uint32_t(E->key()) == source) //do not resend to self
									continue;

								ENetPacket *packet2 = enet_packet_create(packet.packet->data, packet.packet->dataLength, flags);

								enet_peer_send(E->get(), event.channelID, packet2);
							}

						} else if (target < 0) {
							//to all but one

							//and make copies for sending
							for (Map<int, ENetPeer *>::Element *E = peer_map.front(); E; E = E->next()) {

								if (uint32_t(E->key()) == source || E->key() == -target) //do not resend to self, also do not send to excluded
									continue;

								ENetPacket *packet2 = enet_packet_create(packet.packet->data, packet.packet->dataLength, flags);

								enet_peer_send(E->get(), event.channelID, packet2);
							}

							if (-target != 1) {
								//server is not excluded
								incoming_packets.push_back(packet);
							} else {
								//server is excluded, erase packet
								enet_packet_destroy(packet.packet);
							}

						} else if (target == 1) {
							//to myself and only myself
							incoming_packets.push_back(packet);
						} else {
							//to someone else, specifically
							ERR_CONTINUE(!peer_map.has(target));
							enet_peer_send(peer_map[target], event.channelID, packet.packet);
						}
					} else {

						incoming_packets.push_back(packet);
					}

					//destroy packet later..
				} else {
					ERR_CONTINUE(true);
				}

			} break;
			case ENET_EVENT_TYPE_NONE: {
				//do nothing
			} break;
		}
	}
}

bool NetworkedMultiplayerENet::is_server() const {
	ERR_FAIL_COND_V(!active, false);

	return server;
}

void NetworkedMultiplayerENet::close_connection() {

	if (!active)
		return;

	_pop_current_packet();

	bool peers_disconnected = false;
	for (Map<int, ENetPeer *>::Element *E = peer_map.front(); E; E = E->next()) {
		if (E->get()) {
			enet_peer_disconnect_now(E->get(), unique_id);
			peers_disconnected = true;
		}
	}

	if (peers_disconnected) {
		enet_host_flush(host);
		OS::get_singleton()->delay_usec(100); //wait 100ms for disconnection packets to send
	}

	enet_host_destroy(host);
	active = false;
	incoming_packets.clear();
	unique_id = 1; //server is 1
	connection_status = CONNECTION_DISCONNECTED;
}

int NetworkedMultiplayerENet::get_available_packet_count() const {

	return incoming_packets.size();
}
Error NetworkedMultiplayerENet::get_packet(const uint8_t **r_buffer, int &r_buffer_size) const {

	ERR_FAIL_COND_V(incoming_packets.size() == 0, ERR_UNAVAILABLE);

	_pop_current_packet();

	current_packet = incoming_packets.front()->get();
	incoming_packets.pop_front();

	*r_buffer = (const uint8_t *)(&current_packet.packet->data[12]);
	r_buffer_size = current_packet.packet->dataLength - 12;

	return OK;
}
Error NetworkedMultiplayerENet::put_packet(const uint8_t *p_buffer, int p_buffer_size) {

	ERR_FAIL_COND_V(!active, ERR_UNCONFIGURED);
	ERR_FAIL_COND_V(connection_status != CONNECTION_CONNECTED, ERR_UNCONFIGURED);

	int packet_flags = 0;
	int channel = SYSCH_RELIABLE;

	switch (transfer_mode) {
		case TRANSFER_MODE_UNRELIABLE: {
			packet_flags = ENET_PACKET_FLAG_UNSEQUENCED;
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

	Map<int, ENetPeer *>::Element *E = NULL;

	if (target_peer != 0) {

		E = peer_map.find(ABS(target_peer));
		if (!E) {
			ERR_EXPLAIN("Invalid Target Peer: " + itos(target_peer));
			ERR_FAIL_V(ERR_INVALID_PARAMETER);
		}
	}

	ENetPacket *packet = enet_packet_create(NULL, p_buffer_size + 12, packet_flags);
	encode_uint32(unique_id, &packet->data[0]); //source ID
	encode_uint32(target_peer, &packet->data[4]); //dest ID
	encode_uint32(packet_flags, &packet->data[8]); //dest ID
	copymem(&packet->data[12], p_buffer, p_buffer_size);

	if (server) {

		if (target_peer == 0) {
			enet_host_broadcast(host, channel, packet);
		} else if (target_peer < 0) {
			//send to all but one
			//and make copies for sending

			int exclude = -target_peer;

			for (Map<int, ENetPeer *>::Element *F = peer_map.front(); F; F = F->next()) {

				if (F->key() == exclude) // exclude packet
					continue;

				ENetPacket *packet2 = enet_packet_create(packet->data, packet->dataLength, packet_flags);

				enet_peer_send(F->get(), channel, packet2);
			}

			enet_packet_destroy(packet); //original packet no longer needed
		} else {
			enet_peer_send(E->get(), channel, packet);
		}
	} else {

		ERR_FAIL_COND_V(!peer_map.has(1), ERR_BUG);
		enet_peer_send(peer_map[1], channel, packet); //send to server for broadcast..
	}

	enet_host_flush(host);

	return OK;
}

int NetworkedMultiplayerENet::get_max_packet_size() const {

	return 1 << 24; //anything is good
}

void NetworkedMultiplayerENet::_pop_current_packet() const {

	if (current_packet.packet) {
		enet_packet_destroy(current_packet.packet);
		current_packet.packet = NULL;
		current_packet.from = 0;
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
				(uint32_t)OS::get_singleton()->get_data_dir().hash64(), hash);
		/*
		hash = hash_djb2_one_32(
					(uint32_t)OS::get_singleton()->get_unique_ID().hash64(), hash );
		*/
		hash = hash_djb2_one_32(
				(uint32_t)((uint64_t)this), hash); //rely on aslr heap
		hash = hash_djb2_one_32(
				(uint32_t)((uint64_t)&hash), hash); //rely on aslr stack

		hash = hash & 0x7FFFFFFF; // make it compatible with unsigned, since negatie id is used for exclusion
	}

	return hash;
}

int NetworkedMultiplayerENet::get_unique_id() const {

	ERR_FAIL_COND_V(!active, 0);
	return unique_id;
}

void NetworkedMultiplayerENet::set_refuse_new_connections(bool p_enable) {

	refuse_connections = p_enable;
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
			copymem(&enet->src_compressor_mem[ofs], inBuffers[i].data, to_copy);
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
		default: { ERR_FAIL_V(0); }
	}

	int req_size = Compression::get_max_compressed_buffer_size(ofs, mode);
	if (enet->dst_compressor_mem.size() < req_size) {
		enet->dst_compressor_mem.resize(req_size);
	}
	int ret = Compression::compress(enet->dst_compressor_mem.ptr(), enet->src_compressor_mem.ptr(), ofs, mode);

	if (ret < 0)
		return 0;

	if (ret > int(outLimit))
		return 0; //do not bother

	copymem(outData, enet->dst_compressor_mem.ptr(), ret);

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
		default: {}
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

			enet_host_compress(host, NULL);
		} break;
		case COMPRESS_RANGE_CODER: {
			enet_host_compress_with_range_coder(host);
		} break;
		case COMPRESS_FASTLZ:
		case COMPRESS_ZLIB: {

			enet_host_compress(host, &enet_compressor);
		} break;
	}
}

void NetworkedMultiplayerENet::enet_compressor_destroy(void *context) {

	//do none
}

void NetworkedMultiplayerENet::_bind_methods() {

	ClassDB::bind_method(D_METHOD("create_server", "port", "max_clients", "in_bandwidth", "out_bandwidth"), &NetworkedMultiplayerENet::create_server, DEFVAL(32), DEFVAL(0), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("create_client", "ip", "port", "in_bandwidth", "out_bandwidth"), &NetworkedMultiplayerENet::create_client, DEFVAL(0), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("close_connection"), &NetworkedMultiplayerENet::close_connection);
	ClassDB::bind_method(D_METHOD("set_compression_mode", "mode"), &NetworkedMultiplayerENet::set_compression_mode);
	ClassDB::bind_method(D_METHOD("get_compression_mode"), &NetworkedMultiplayerENet::get_compression_mode);
	ClassDB::bind_method(D_METHOD("set_bind_ip", "ip"), &NetworkedMultiplayerENet::set_bind_ip);

	BIND_CONSTANT(COMPRESS_NONE);
	BIND_CONSTANT(COMPRESS_RANGE_CODER);
	BIND_CONSTANT(COMPRESS_FASTLZ);
	BIND_CONSTANT(COMPRESS_ZLIB);
}

NetworkedMultiplayerENet::NetworkedMultiplayerENet() {

	active = false;
	server = false;
	refuse_connections = false;
	unique_id = 0;
	target_peer = 0;
	current_packet.packet = NULL;
	transfer_mode = TRANSFER_MODE_RELIABLE;
	connection_status = CONNECTION_DISCONNECTED;
	compression_mode = COMPRESS_NONE;
	enet_compressor.context = this;
	enet_compressor.compress = enet_compress;
	enet_compressor.decompress = enet_decompress;
	enet_compressor.destroy = enet_compressor_destroy;

	bind_ip = IP_Address("*");
}

NetworkedMultiplayerENet::~NetworkedMultiplayerENet() {

	close_connection();
}

// sets IP for ENet to bind when using create_server
// if no IP is set, then ENet bind to ENET_HOST_ANY
void NetworkedMultiplayerENet::set_bind_ip(const IP_Address &p_ip) {
	ERR_FAIL_COND(!p_ip.is_valid() && !p_ip.is_wildcard());

	bind_ip = p_ip;
}
