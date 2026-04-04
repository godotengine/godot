/**************************************************************************/
/*  e_net_multiplayer_peer.hpp                                            */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/multiplayer_peer.hpp>
#include <godot_cpp/classes/ref.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class ENetConnection;
class ENetPacketPeer;
class String;

class ENetMultiplayerPeer : public MultiplayerPeer {
	GDEXTENSION_CLASS(ENetMultiplayerPeer, MultiplayerPeer)

public:
	Error create_server(int32_t p_port, int32_t p_max_clients = 32, int32_t p_max_channels = 0, int32_t p_in_bandwidth = 0, int32_t p_out_bandwidth = 0);
	Error create_client(const String &p_address, int32_t p_port, int32_t p_channel_count = 0, int32_t p_in_bandwidth = 0, int32_t p_out_bandwidth = 0, int32_t p_local_port = 0);
	Error create_mesh(int32_t p_unique_id);
	Error add_mesh_peer(int32_t p_peer_id, const Ref<ENetConnection> &p_host);
	void set_bind_ip(const String &p_ip);
	Ref<ENetConnection> get_host() const;
	Ref<ENetPacketPeer> get_peer(int32_t p_id) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		MultiplayerPeer::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

