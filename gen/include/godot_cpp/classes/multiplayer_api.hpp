/**************************************************************************/
/*  multiplayer_api.hpp                                                   */
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
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/string_name.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class MultiplayerPeer;
class Object;
class Variant;

class MultiplayerAPI : public RefCounted {
	GDEXTENSION_CLASS(MultiplayerAPI, RefCounted)

public:
	enum RPCMode {
		RPC_MODE_DISABLED = 0,
		RPC_MODE_ANY_PEER = 1,
		RPC_MODE_AUTHORITY = 2,
	};

	bool has_multiplayer_peer();
	Ref<MultiplayerPeer> get_multiplayer_peer();
	void set_multiplayer_peer(const Ref<MultiplayerPeer> &p_peer);
	int32_t get_unique_id();
	bool is_server();
	int32_t get_remote_sender_id();
	Error poll();
	Error rpc(int32_t p_peer, Object *p_object, const StringName &p_method, const Array &p_arguments = Array());
	Error object_configuration_add(Object *p_object, const Variant &p_configuration);
	Error object_configuration_remove(Object *p_object, const Variant &p_configuration);
	PackedInt32Array get_peers();
	static void set_default_interface(const StringName &p_interface_name);
	static StringName get_default_interface();
	static Ref<MultiplayerAPI> create_default_interface();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(MultiplayerAPI::RPCMode);

