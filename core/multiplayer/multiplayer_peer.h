/*************************************************************************/
/*  multiplayer_peer.h                                                   */
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

#ifndef NETWORKED_MULTIPLAYER_PEER_H
#define NETWORKED_MULTIPLAYER_PEER_H

#include "core/io/packet_peer.h"
#include "core/multiplayer/multiplayer.h"

class MultiplayerPeer : public PacketPeer {
	GDCLASS(MultiplayerPeer, PacketPeer);

protected:
	static void _bind_methods();

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

	virtual void set_transfer_channel(int p_channel) = 0;
	virtual int get_transfer_channel() const = 0;
	virtual void set_transfer_mode(Multiplayer::TransferMode p_mode) = 0;
	virtual Multiplayer::TransferMode get_transfer_mode() const = 0;
	virtual void set_target_peer(int p_peer_id) = 0;

	virtual int get_packet_peer() const = 0;

	virtual bool is_server() const = 0;

	virtual void poll() = 0;

	virtual int get_unique_id() const = 0;

	virtual void set_refuse_new_connections(bool p_enable) = 0;
	virtual bool is_refusing_new_connections() const = 0;

	virtual ConnectionStatus get_connection_status() const = 0;
	uint32_t generate_unique_id() const;

	MultiplayerPeer() {}
};

VARIANT_ENUM_CAST(MultiplayerPeer::ConnectionStatus)

#endif // NETWORKED_MULTIPLAYER_PEER_H
