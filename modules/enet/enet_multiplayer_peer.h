/*************************************************************************/
/*  enet_multiplayer_peer.h                                              */
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

#ifndef NETWORKED_MULTIPLAYER_ENET_H
#define NETWORKED_MULTIPLAYER_ENET_H

#include "core/crypto/crypto.h"
#include "core/multiplayer/multiplayer_peer.h"

#include "enet_connection.h"
#include <enet/enet.h>

class ENetMultiplayerPeer : public MultiplayerPeer {
	GDCLASS(ENetMultiplayerPeer, MultiplayerPeer);

private:
	enum {
		SYSMSG_ADD_PEER,
		SYSMSG_REMOVE_PEER
	};

	enum {
		SYSCH_CONFIG = 0,
		SYSCH_RELIABLE = 1,
		SYSCH_UNRELIABLE = 2,
		SYSCH_MAX = 3
	};

	enum Mode {
		MODE_NONE,
		MODE_SERVER,
		MODE_CLIENT,
		MODE_MESH,
	};

	Mode active_mode = MODE_NONE;

	uint32_t unique_id = 0;

	int target_peer = 0;

	bool server_relay = true;

	ConnectionStatus connection_status = CONNECTION_DISCONNECTED;

	Map<int, Ref<ENetConnection>> hosts;
	Map<int, Ref<ENetPacketPeer>> peers;

	struct Packet {
		ENetPacket *packet = nullptr;
		int from = 0;
		int channel = 0;
	};

	List<Packet> incoming_packets;

	Packet current_packet;

	void _pop_current_packet();
	bool _parse_server_event(ENetConnection::EventType p_event_type, ENetConnection::Event &p_event);
	bool _parse_client_event(ENetConnection::EventType p_event_type, ENetConnection::Event &p_event);
	bool _parse_mesh_event(ENetConnection::EventType p_event_type, ENetConnection::Event &p_event, int p_peer_id);
	void _relay(int p_from, int p_to, enet_uint8 p_channel, ENetPacket *p_packet);
	void _notify_peers(int p_id, bool p_connected);
	void _destroy_unused(ENetPacket *p_packet);
	_FORCE_INLINE_ bool _is_active() const { return active_mode != MODE_NONE; }

	IPAddress bind_ip;

protected:
	static void _bind_methods();

public:
	virtual void set_target_peer(int p_peer) override;
	virtual int get_packet_peer() const override;

	virtual void poll() override;
	virtual bool is_server() const override;
	// Overridden so we can instrument the DTLSServer when needed.
	virtual void set_refuse_new_connections(bool p_enabled) override;

	virtual ConnectionStatus get_connection_status() const override;

	virtual int get_unique_id() const override;

	virtual int get_max_packet_size() const override;
	virtual int get_available_packet_count() const override;
	virtual Error get_packet(const uint8_t **r_buffer, int &r_buffer_size) override;
	virtual Error put_packet(const uint8_t *p_buffer, int p_buffer_size) override;

	Error create_server(int p_port, int p_max_clients = 32, int p_max_channels = 0, int p_in_bandwidth = 0, int p_out_bandwidth = 0);
	Error create_client(const String &p_address, int p_port, int p_channel_count = 0, int p_in_bandwidth = 0, int p_out_bandwidth = 0, int p_local_port = 0);
	Error create_mesh(int p_id);
	Error add_mesh_peer(int p_id, Ref<ENetConnection> p_host);

	void close_connection(uint32_t wait_usec = 100);

	void disconnect_peer(int p_peer, bool now = false);

	void set_bind_ip(const IPAddress &p_ip);
	void set_server_relay_enabled(bool p_enabled);
	bool is_server_relay_enabled() const;

	Ref<ENetConnection> get_host() const;
	Ref<ENetPacketPeer> get_peer(int p_id) const;

	ENetMultiplayerPeer();
	~ENetMultiplayerPeer();
};

#endif // NETWORKED_MULTIPLAYER_ENET_H
