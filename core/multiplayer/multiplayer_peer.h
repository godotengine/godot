/*************************************************************************/
/*  multiplayer_peer.h                                                   */
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

#ifndef NETWORKED_MULTIPLAYER_PEER_H
#define NETWORKED_MULTIPLAYER_PEER_H

#include "core/io/packet_peer.h"
#include "core/multiplayer/multiplayer.h"

#include "core/object/gdvirtual.gen.inc"
#include "core/object/script_language.h"
#include "core/variant/native_ptr.h"

class MultiplayerPeer : public PacketPeer {
	GDCLASS(MultiplayerPeer, PacketPeer);

protected:
	static void _bind_methods();

private:
	int transfer_channel = 0;
	Multiplayer::TransferMode transfer_mode = Multiplayer::TRANSFER_MODE_RELIABLE;
	bool refuse_connections = false;

public:
	enum {
		TARGET_PEER_BROADCAST = 0,
		TARGET_PEER_SERVER = 1
	};

	enum ConnectionStatus {
		CONNECTION_DISCONNECTED,
		CONNECTION_CONNECTING,
		CONNECTION_CONNECTED,
	};

	virtual void set_transfer_channel(int p_channel);
	virtual int get_transfer_channel() const;
	virtual void set_transfer_mode(Multiplayer::TransferMode p_mode);
	virtual Multiplayer::TransferMode get_transfer_mode() const;
	virtual void set_refuse_new_connections(bool p_enable);
	virtual bool is_refusing_new_connections() const;

	virtual void set_target_peer(int p_peer_id) = 0;

	virtual int get_packet_peer() const = 0;

	virtual bool is_server() const = 0;

	virtual void poll() = 0;

	virtual int get_unique_id() const = 0;

	virtual ConnectionStatus get_connection_status() const = 0;

	uint32_t generate_unique_id() const;

	MultiplayerPeer() {}
};

VARIANT_ENUM_CAST(MultiplayerPeer::ConnectionStatus);

class MultiplayerPeerExtension : public MultiplayerPeer {
	GDCLASS(MultiplayerPeerExtension, MultiplayerPeer);

protected:
	static void _bind_methods();

public:
	/* PacketPeer */
	virtual int get_available_packet_count() const override;
	virtual Error get_packet(const uint8_t **r_buffer, int &r_buffer_size) override; ///< buffer is GONE after next get_packet
	virtual Error put_packet(const uint8_t *p_buffer, int p_buffer_size) override;
	virtual int get_max_packet_size() const override;

	/* MultiplayerPeer */
	virtual void set_transfer_channel(int p_channel) override;
	virtual int get_transfer_channel() const override;
	virtual void set_transfer_mode(Multiplayer::TransferMode p_mode) override;
	virtual Multiplayer::TransferMode get_transfer_mode() const override;
	virtual void set_target_peer(int p_peer_id) override;

	virtual int get_packet_peer() const override;

	virtual bool is_server() const override;

	virtual void poll() override;

	virtual int get_unique_id() const override;

	virtual void set_refuse_new_connections(bool p_enable) override;
	virtual bool is_refusing_new_connections() const override;

	virtual ConnectionStatus get_connection_status() const override;

	/* PacketPeer GDExtension */
	GDVIRTUAL0RC(int, _get_available_packet_count);
	GDVIRTUAL2R(int, _get_packet, GDNativeConstPtr<const uint8_t *>, GDNativePtr<int>);
	GDVIRTUAL2R(int, _put_packet, GDNativeConstPtr<const uint8_t>, int);
	GDVIRTUAL0RC(int, _get_max_packet_size);

	/* MultiplayerPeer GDExtension */
	GDVIRTUAL1(_set_transfer_channel, int);
	GDVIRTUAL0RC(int, _get_transfer_channel);
	GDVIRTUAL1(_set_transfer_mode, int);
	GDVIRTUAL0RC(int, _get_transfer_mode);
	GDVIRTUAL1(_set_target_peer, int);
	GDVIRTUAL0RC(int, _get_packet_peer);
	GDVIRTUAL0RC(bool, _is_server);
	GDVIRTUAL0R(int, _poll);
	GDVIRTUAL0RC(int, _get_unique_id);
	GDVIRTUAL1(_set_refuse_new_connections, bool);
	GDVIRTUAL0RC(bool, _is_refusing_new_connections);
	GDVIRTUAL0RC(int, _get_connection_status);
};

#endif // NETWORKED_MULTIPLAYER_PEER_H
