/**************************************************************************/
/*  multiplayer_peer_extension.hpp                                        */
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
#include <godot_cpp/variant/packed_byte_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class MultiplayerPeerExtension : public MultiplayerPeer {
	GDEXTENSION_CLASS(MultiplayerPeerExtension, MultiplayerPeer)

public:
	virtual Error _get_packet(const uint8_t **r_buffer, int32_t *r_buffer_size);
	virtual Error _put_packet(const uint8_t *p_buffer, int32_t p_buffer_size);
	virtual int32_t _get_available_packet_count() const;
	virtual int32_t _get_max_packet_size() const;
	virtual PackedByteArray _get_packet_script();
	virtual Error _put_packet_script(const PackedByteArray &p_buffer);
	virtual int32_t _get_packet_channel() const;
	virtual MultiplayerPeer::TransferMode _get_packet_mode() const;
	virtual void _set_transfer_channel(int32_t p_channel);
	virtual int32_t _get_transfer_channel() const;
	virtual void _set_transfer_mode(MultiplayerPeer::TransferMode p_mode);
	virtual MultiplayerPeer::TransferMode _get_transfer_mode() const;
	virtual void _set_target_peer(int32_t p_peer);
	virtual int32_t _get_packet_peer() const;
	virtual bool _is_server() const;
	virtual void _poll();
	virtual void _close();
	virtual void _disconnect_peer(int32_t p_peer, bool p_force);
	virtual int32_t _get_unique_id() const;
	virtual void _set_refuse_new_connections(bool p_enable);
	virtual bool _is_refusing_new_connections() const;
	virtual bool _is_server_relay_supported() const;
	virtual MultiplayerPeer::ConnectionStatus _get_connection_status() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		MultiplayerPeer::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_get_packet), decltype(&T::_get_packet)>) {
			BIND_VIRTUAL_METHOD(T, _get_packet, 3099858825);
		}
		if constexpr (!std::is_same_v<decltype(&B::_put_packet), decltype(&T::_put_packet)>) {
			BIND_VIRTUAL_METHOD(T, _put_packet, 3099858825);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_available_packet_count), decltype(&T::_get_available_packet_count)>) {
			BIND_VIRTUAL_METHOD(T, _get_available_packet_count, 3905245786);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_max_packet_size), decltype(&T::_get_max_packet_size)>) {
			BIND_VIRTUAL_METHOD(T, _get_max_packet_size, 3905245786);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_packet_script), decltype(&T::_get_packet_script)>) {
			BIND_VIRTUAL_METHOD(T, _get_packet_script, 2115431945);
		}
		if constexpr (!std::is_same_v<decltype(&B::_put_packet_script), decltype(&T::_put_packet_script)>) {
			BIND_VIRTUAL_METHOD(T, _put_packet_script, 680677267);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_packet_channel), decltype(&T::_get_packet_channel)>) {
			BIND_VIRTUAL_METHOD(T, _get_packet_channel, 3905245786);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_packet_mode), decltype(&T::_get_packet_mode)>) {
			BIND_VIRTUAL_METHOD(T, _get_packet_mode, 3369852622);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_transfer_channel), decltype(&T::_set_transfer_channel)>) {
			BIND_VIRTUAL_METHOD(T, _set_transfer_channel, 1286410249);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_transfer_channel), decltype(&T::_get_transfer_channel)>) {
			BIND_VIRTUAL_METHOD(T, _get_transfer_channel, 3905245786);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_transfer_mode), decltype(&T::_set_transfer_mode)>) {
			BIND_VIRTUAL_METHOD(T, _set_transfer_mode, 950411049);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_transfer_mode), decltype(&T::_get_transfer_mode)>) {
			BIND_VIRTUAL_METHOD(T, _get_transfer_mode, 3369852622);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_target_peer), decltype(&T::_set_target_peer)>) {
			BIND_VIRTUAL_METHOD(T, _set_target_peer, 1286410249);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_packet_peer), decltype(&T::_get_packet_peer)>) {
			BIND_VIRTUAL_METHOD(T, _get_packet_peer, 3905245786);
		}
		if constexpr (!std::is_same_v<decltype(&B::_is_server), decltype(&T::_is_server)>) {
			BIND_VIRTUAL_METHOD(T, _is_server, 36873697);
		}
		if constexpr (!std::is_same_v<decltype(&B::_poll), decltype(&T::_poll)>) {
			BIND_VIRTUAL_METHOD(T, _poll, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_close), decltype(&T::_close)>) {
			BIND_VIRTUAL_METHOD(T, _close, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_disconnect_peer), decltype(&T::_disconnect_peer)>) {
			BIND_VIRTUAL_METHOD(T, _disconnect_peer, 300928843);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_unique_id), decltype(&T::_get_unique_id)>) {
			BIND_VIRTUAL_METHOD(T, _get_unique_id, 3905245786);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_refuse_new_connections), decltype(&T::_set_refuse_new_connections)>) {
			BIND_VIRTUAL_METHOD(T, _set_refuse_new_connections, 2586408642);
		}
		if constexpr (!std::is_same_v<decltype(&B::_is_refusing_new_connections), decltype(&T::_is_refusing_new_connections)>) {
			BIND_VIRTUAL_METHOD(T, _is_refusing_new_connections, 36873697);
		}
		if constexpr (!std::is_same_v<decltype(&B::_is_server_relay_supported), decltype(&T::_is_server_relay_supported)>) {
			BIND_VIRTUAL_METHOD(T, _is_server_relay_supported, 36873697);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_connection_status), decltype(&T::_get_connection_status)>) {
			BIND_VIRTUAL_METHOD(T, _get_connection_status, 2147374275);
		}
	}

public:
};

} // namespace godot

