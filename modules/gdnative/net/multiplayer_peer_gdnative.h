/*************************************************************************/
/*  multiplayer_peer_gdnative.h                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef MULTIPLAYER_PEER_GDNATIVE_H
#define MULTIPLAYER_PEER_GDNATIVE_H

#include "core/io/networked_multiplayer_peer.h"
#include "modules/gdnative/gdnative.h"
#include "modules/gdnative/include/net/godot_net.h"

class MultiplayerPeerGDNative : public NetworkedMultiplayerPeer {
	GDCLASS(MultiplayerPeerGDNative, NetworkedMultiplayerPeer);

protected:
	static void _bind_methods();
	const godot_net_multiplayer_peer *interface;

public:
	MultiplayerPeerGDNative();
	~MultiplayerPeerGDNative();

	/* Sets the interface implementation from GDNative */
	void set_native_multiplayer_peer(const godot_net_multiplayer_peer *p_impl);

	/* Specific to PacketPeer */
	virtual Error get_packet(const uint8_t **r_buffer, int &r_buffer_size) override;
	virtual Error put_packet(const uint8_t *p_buffer, int p_buffer_size) override;
	virtual int get_max_packet_size() const override;
	virtual int get_available_packet_count() const override;

	/* Specific to NetworkedMultiplayerPeer */
	virtual void set_transfer_mode(TransferMode p_mode) override;
	virtual TransferMode get_transfer_mode() const override;
	virtual void set_target_peer(int p_peer_id) override;

	virtual int get_packet_peer() const override;

	virtual bool is_server() const override;

	virtual void poll() override;

	virtual int get_unique_id() const override;

	virtual void set_refuse_new_connections(bool p_enable) override;
	virtual bool is_refusing_new_connections() const override;

	virtual ConnectionStatus get_connection_status() const override;
};

#endif // MULTIPLAYER_PEER_GDNATIVE_H
