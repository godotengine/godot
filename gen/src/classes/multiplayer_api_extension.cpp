/**************************************************************************/
/*  multiplayer_api_extension.cpp                                         */
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

#include <godot_cpp/classes/multiplayer_api_extension.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/multiplayer_peer.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/variant.hpp>

namespace godot {

Error MultiplayerAPIExtension::_poll() {
	return Error(0);
}

void MultiplayerAPIExtension::_set_multiplayer_peer(const Ref<MultiplayerPeer> &p_multiplayer_peer) {}

Ref<MultiplayerPeer> MultiplayerAPIExtension::_get_multiplayer_peer() {
	return Ref<MultiplayerPeer>();
}

int32_t MultiplayerAPIExtension::_get_unique_id() const {
	return 0;
}

PackedInt32Array MultiplayerAPIExtension::_get_peer_ids() const {
	return PackedInt32Array();
}

Error MultiplayerAPIExtension::_rpc(int32_t p_peer, Object *p_object, const StringName &p_method, const Array &p_args) {
	return Error(0);
}

int32_t MultiplayerAPIExtension::_get_remote_sender_id() const {
	return 0;
}

Error MultiplayerAPIExtension::_object_configuration_add(Object *p_object, const Variant &p_configuration) {
	return Error(0);
}

Error MultiplayerAPIExtension::_object_configuration_remove(Object *p_object, const Variant &p_configuration) {
	return Error(0);
}

} // namespace godot
