/**************************************************************************/
/*  scene_multiplayer.hpp                                                 */
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
#include <godot_cpp/classes/multiplayer_api.hpp>
#include <godot_cpp/classes/multiplayer_peer.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/callable.hpp>
#include <godot_cpp/variant/node_path.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class PackedByteArray;

class SceneMultiplayer : public MultiplayerAPI {
	GDEXTENSION_CLASS(SceneMultiplayer, MultiplayerAPI)

public:
	void set_root_path(const NodePath &p_path);
	NodePath get_root_path() const;
	void clear();
	void disconnect_peer(int32_t p_id);
	PackedInt32Array get_authenticating_peers();
	Error send_auth(int32_t p_id, const PackedByteArray &p_data);
	Error complete_auth(int32_t p_id);
	void set_auth_callback(const Callable &p_callback);
	Callable get_auth_callback() const;
	void set_auth_timeout(double p_timeout);
	double get_auth_timeout() const;
	void set_refuse_new_connections(bool p_refuse);
	bool is_refusing_new_connections() const;
	void set_allow_object_decoding(bool p_enable);
	bool is_object_decoding_allowed() const;
	void set_server_relay_enabled(bool p_enabled);
	bool is_server_relay_enabled() const;
	Error send_bytes(const PackedByteArray &p_bytes, int32_t p_id = 0, MultiplayerPeer::TransferMode p_mode = (MultiplayerPeer::TransferMode)2, int32_t p_channel = 0);
	int32_t get_max_sync_packet_size() const;
	void set_max_sync_packet_size(int32_t p_size);
	int32_t get_max_delta_packet_size() const;
	void set_max_delta_packet_size(int32_t p_size);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		MultiplayerAPI::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

