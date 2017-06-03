/*************************************************************************/
/*  networked_multiplayer_peer.h                                         */
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
#ifndef NETWORKED_MULTIPLAYER_PEER_H
#define NETWORKED_MULTIPLAYER_PEER_H

#include "io/packet_peer.h"

class NetworkedMultiplayerPeer : public PacketPeer {

	GDCLASS(NetworkedMultiplayerPeer, PacketPeer);

protected:
	static void _bind_methods();

public:
	enum {
		TARGET_PEER_BROADCAST = 0,
		TARGET_PEER_SERVER = 1
	};
	enum TransferMode {
		TRANSFER_MODE_UNRELIABLE,
		TRANSFER_MODE_UNRELIABLE_ORDERED,
		TRANSFER_MODE_RELIABLE,
	};

	enum ConnectionStatus {
		CONNECTION_DISCONNECTED,
		CONNECTION_CONNECTING,
		CONNECTION_CONNECTED,
	};

	virtual void set_transfer_mode(TransferMode p_mode) = 0;
	virtual void set_target_peer(int p_peer_id) = 0;

	virtual int get_packet_peer() const = 0;

	virtual bool is_server() const = 0;

	virtual void poll() = 0;

	virtual int get_unique_id() const = 0;

	virtual void set_refuse_new_connections(bool p_enable) = 0;
	virtual bool is_refusing_new_connections() const = 0;

	virtual ConnectionStatus get_connection_status() const = 0;

	NetworkedMultiplayerPeer();
};

VARIANT_ENUM_CAST(NetworkedMultiplayerPeer::TransferMode)
VARIANT_ENUM_CAST(NetworkedMultiplayerPeer::ConnectionStatus)

#endif // NetworkedMultiplayerPeer_H
