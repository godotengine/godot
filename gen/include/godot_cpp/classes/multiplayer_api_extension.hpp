/**************************************************************************/
/*  multiplayer_api_extension.hpp                                         */
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
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Array;
class MultiplayerPeer;
class Object;
class StringName;
class Variant;

class MultiplayerAPIExtension : public MultiplayerAPI {
	GDEXTENSION_CLASS(MultiplayerAPIExtension, MultiplayerAPI)

public:
	virtual Error _poll();
	virtual void _set_multiplayer_peer(const Ref<MultiplayerPeer> &p_multiplayer_peer);
	virtual Ref<MultiplayerPeer> _get_multiplayer_peer();
	virtual int32_t _get_unique_id() const;
	virtual PackedInt32Array _get_peer_ids() const;
	virtual Error _rpc(int32_t p_peer, Object *p_object, const StringName &p_method, const Array &p_args);
	virtual int32_t _get_remote_sender_id() const;
	virtual Error _object_configuration_add(Object *p_object, const Variant &p_configuration);
	virtual Error _object_configuration_remove(Object *p_object, const Variant &p_configuration);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		MultiplayerAPI::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_poll), decltype(&T::_poll)>) {
			BIND_VIRTUAL_METHOD(T, _poll, 166280745);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_multiplayer_peer), decltype(&T::_set_multiplayer_peer)>) {
			BIND_VIRTUAL_METHOD(T, _set_multiplayer_peer, 3694835298);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_multiplayer_peer), decltype(&T::_get_multiplayer_peer)>) {
			BIND_VIRTUAL_METHOD(T, _get_multiplayer_peer, 3223692825);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_unique_id), decltype(&T::_get_unique_id)>) {
			BIND_VIRTUAL_METHOD(T, _get_unique_id, 3905245786);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_peer_ids), decltype(&T::_get_peer_ids)>) {
			BIND_VIRTUAL_METHOD(T, _get_peer_ids, 1930428628);
		}
		if constexpr (!std::is_same_v<decltype(&B::_rpc), decltype(&T::_rpc)>) {
			BIND_VIRTUAL_METHOD(T, _rpc, 3673574758);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_remote_sender_id), decltype(&T::_get_remote_sender_id)>) {
			BIND_VIRTUAL_METHOD(T, _get_remote_sender_id, 3905245786);
		}
		if constexpr (!std::is_same_v<decltype(&B::_object_configuration_add), decltype(&T::_object_configuration_add)>) {
			BIND_VIRTUAL_METHOD(T, _object_configuration_add, 1171879464);
		}
		if constexpr (!std::is_same_v<decltype(&B::_object_configuration_remove), decltype(&T::_object_configuration_remove)>) {
			BIND_VIRTUAL_METHOD(T, _object_configuration_remove, 1171879464);
		}
	}

public:
};

} // namespace godot

