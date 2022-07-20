/*************************************************************************/
/*  networked_multiplayer_custom.h                                       */
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

#ifndef NETWORKED_MULTIPLAYER_CUSTOM_H
#define NETWORKED_MULTIPLAYER_CUSTOM_H

#include "core/io/networked_multiplayer_peer.h"

class NetworkedMultiplayerCustom : public NetworkedMultiplayerPeer {
	GDCLASS(NetworkedMultiplayerCustom, NetworkedMultiplayerPeer);

protected:
	int self_id;
	TransferMode transfer_mode;
	ConnectionStatus connection_status;
	bool refusing_new_connections;
	int target_id;
	int max_packet_size;

	struct Packet {
		PoolVector<uint8_t> data;
		int from;
	};

	List<Packet> incoming_packets;

	Packet current_packet;

	static void _bind_methods();

public:
	NetworkedMultiplayerCustom();
	~NetworkedMultiplayerCustom();

	// PacketPeer.
	Error get_packet(const uint8_t **r_buffer, int &r_buffer_size);
	Error put_packet(const uint8_t *p_buffer, int p_buffer_size);
	int get_available_packet_count() const;
	int get_max_packet_size() const;

	// NetworkedMultiplayerPeer.
	void set_transfer_mode(TransferMode p_mode);
	TransferMode get_transfer_mode() const;
	void set_target_peer(int p_peer);
	int get_packet_peer() const;
	bool is_server() const;
	void poll();
	int get_unique_id() const;
	void set_refuse_new_connections(bool p_enable);
	bool is_refusing_new_connections() const;
	ConnectionStatus get_connection_status() const;

	// Custom methods.
	void initialize(int p_self_id);
	void set_max_packet_size(int p_max_packet_size);
	void set_connection_status(ConnectionStatus p_connection_status);
	void deliver_packet(const PoolByteArray &p_data, int p_from_peer_id);
};

#endif
