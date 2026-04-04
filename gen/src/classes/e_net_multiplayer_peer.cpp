/**************************************************************************/
/*  e_net_multiplayer_peer.cpp                                            */
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

#include <godot_cpp/classes/e_net_multiplayer_peer.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/e_net_connection.hpp>
#include <godot_cpp/classes/e_net_packet_peer.hpp>
#include <godot_cpp/variant/string.hpp>

namespace godot {

Error ENetMultiplayerPeer::create_server(int32_t p_port, int32_t p_max_clients, int32_t p_max_channels, int32_t p_in_bandwidth, int32_t p_out_bandwidth) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetMultiplayerPeer::get_class_static()._native_ptr(), StringName("create_server")._native_ptr(), 2917761309);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_port_encoded;
	PtrToArg<int64_t>::encode(p_port, &p_port_encoded);
	int64_t p_max_clients_encoded;
	PtrToArg<int64_t>::encode(p_max_clients, &p_max_clients_encoded);
	int64_t p_max_channels_encoded;
	PtrToArg<int64_t>::encode(p_max_channels, &p_max_channels_encoded);
	int64_t p_in_bandwidth_encoded;
	PtrToArg<int64_t>::encode(p_in_bandwidth, &p_in_bandwidth_encoded);
	int64_t p_out_bandwidth_encoded;
	PtrToArg<int64_t>::encode(p_out_bandwidth, &p_out_bandwidth_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_port_encoded, &p_max_clients_encoded, &p_max_channels_encoded, &p_in_bandwidth_encoded, &p_out_bandwidth_encoded);
}

Error ENetMultiplayerPeer::create_client(const String &p_address, int32_t p_port, int32_t p_channel_count, int32_t p_in_bandwidth, int32_t p_out_bandwidth, int32_t p_local_port) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetMultiplayerPeer::get_class_static()._native_ptr(), StringName("create_client")._native_ptr(), 2327163476);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_port_encoded;
	PtrToArg<int64_t>::encode(p_port, &p_port_encoded);
	int64_t p_channel_count_encoded;
	PtrToArg<int64_t>::encode(p_channel_count, &p_channel_count_encoded);
	int64_t p_in_bandwidth_encoded;
	PtrToArg<int64_t>::encode(p_in_bandwidth, &p_in_bandwidth_encoded);
	int64_t p_out_bandwidth_encoded;
	PtrToArg<int64_t>::encode(p_out_bandwidth, &p_out_bandwidth_encoded);
	int64_t p_local_port_encoded;
	PtrToArg<int64_t>::encode(p_local_port, &p_local_port_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_address, &p_port_encoded, &p_channel_count_encoded, &p_in_bandwidth_encoded, &p_out_bandwidth_encoded, &p_local_port_encoded);
}

Error ENetMultiplayerPeer::create_mesh(int32_t p_unique_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetMultiplayerPeer::get_class_static()._native_ptr(), StringName("create_mesh")._native_ptr(), 844576869);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_unique_id_encoded;
	PtrToArg<int64_t>::encode(p_unique_id, &p_unique_id_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_unique_id_encoded);
}

Error ENetMultiplayerPeer::add_mesh_peer(int32_t p_peer_id, const Ref<ENetConnection> &p_host) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetMultiplayerPeer::get_class_static()._native_ptr(), StringName("add_mesh_peer")._native_ptr(), 1293458335);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_peer_id_encoded;
	PtrToArg<int64_t>::encode(p_peer_id, &p_peer_id_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_peer_id_encoded, (p_host != nullptr ? &p_host->_owner : nullptr));
}

void ENetMultiplayerPeer::set_bind_ip(const String &p_ip) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetMultiplayerPeer::get_class_static()._native_ptr(), StringName("set_bind_ip")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_ip);
}

Ref<ENetConnection> ENetMultiplayerPeer::get_host() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetMultiplayerPeer::get_class_static()._native_ptr(), StringName("get_host")._native_ptr(), 4103238886);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<ENetConnection>()));
	return Ref<ENetConnection>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<ENetConnection>(_gde_method_bind, _owner));
}

Ref<ENetPacketPeer> ENetMultiplayerPeer::get_peer(int32_t p_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetMultiplayerPeer::get_class_static()._native_ptr(), StringName("get_peer")._native_ptr(), 3793311544);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<ENetPacketPeer>()));
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	return Ref<ENetPacketPeer>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<ENetPacketPeer>(_gde_method_bind, _owner, &p_id_encoded));
}

} // namespace godot
