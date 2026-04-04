/**************************************************************************/
/*  packet_peer_extension.hpp                                             */
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
#include <godot_cpp/classes/packet_peer.hpp>
#include <godot_cpp/classes/ref.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class PacketPeerExtension : public PacketPeer {
	GDEXTENSION_CLASS(PacketPeerExtension, PacketPeer)

public:
	virtual Error _get_packet(const uint8_t **r_buffer, int32_t *r_buffer_size);
	virtual Error _put_packet(const uint8_t *p_buffer, int32_t p_buffer_size);
	virtual int32_t _get_available_packet_count() const;
	virtual int32_t _get_max_packet_size() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		PacketPeer::register_virtuals<T, B>();
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
	}

public:
};

} // namespace godot

