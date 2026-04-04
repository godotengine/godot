/**************************************************************************/
/*  e_net_packet_peer.cpp                                                 */
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

#include <godot_cpp/classes/e_net_packet_peer.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/variant/packed_byte_array.hpp>

namespace godot {

void ENetPacketPeer::peer_disconnect(int32_t p_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetPacketPeer::get_class_static()._native_ptr(), StringName("peer_disconnect")._native_ptr(), 1995695955);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_data_encoded;
	PtrToArg<int64_t>::encode(p_data, &p_data_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_data_encoded);
}

void ENetPacketPeer::peer_disconnect_later(int32_t p_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetPacketPeer::get_class_static()._native_ptr(), StringName("peer_disconnect_later")._native_ptr(), 1995695955);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_data_encoded;
	PtrToArg<int64_t>::encode(p_data, &p_data_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_data_encoded);
}

void ENetPacketPeer::peer_disconnect_now(int32_t p_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetPacketPeer::get_class_static()._native_ptr(), StringName("peer_disconnect_now")._native_ptr(), 1995695955);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_data_encoded;
	PtrToArg<int64_t>::encode(p_data, &p_data_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_data_encoded);
}

void ENetPacketPeer::ping() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetPacketPeer::get_class_static()._native_ptr(), StringName("ping")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void ENetPacketPeer::ping_interval(int32_t p_ping_interval) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetPacketPeer::get_class_static()._native_ptr(), StringName("ping_interval")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_ping_interval_encoded;
	PtrToArg<int64_t>::encode(p_ping_interval, &p_ping_interval_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_ping_interval_encoded);
}

void ENetPacketPeer::reset() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetPacketPeer::get_class_static()._native_ptr(), StringName("reset")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

Error ENetPacketPeer::send(int32_t p_channel, const PackedByteArray &p_packet, int32_t p_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetPacketPeer::get_class_static()._native_ptr(), StringName("send")._native_ptr(), 120522849);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_channel_encoded;
	PtrToArg<int64_t>::encode(p_channel, &p_channel_encoded);
	int64_t p_flags_encoded;
	PtrToArg<int64_t>::encode(p_flags, &p_flags_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_channel_encoded, &p_packet, &p_flags_encoded);
}

void ENetPacketPeer::throttle_configure(int32_t p_interval, int32_t p_acceleration, int32_t p_deceleration) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetPacketPeer::get_class_static()._native_ptr(), StringName("throttle_configure")._native_ptr(), 1649997291);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_interval_encoded;
	PtrToArg<int64_t>::encode(p_interval, &p_interval_encoded);
	int64_t p_acceleration_encoded;
	PtrToArg<int64_t>::encode(p_acceleration, &p_acceleration_encoded);
	int64_t p_deceleration_encoded;
	PtrToArg<int64_t>::encode(p_deceleration, &p_deceleration_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_interval_encoded, &p_acceleration_encoded, &p_deceleration_encoded);
}

void ENetPacketPeer::set_timeout(int32_t p_timeout, int32_t p_timeout_min, int32_t p_timeout_max) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetPacketPeer::get_class_static()._native_ptr(), StringName("set_timeout")._native_ptr(), 1649997291);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_timeout_encoded;
	PtrToArg<int64_t>::encode(p_timeout, &p_timeout_encoded);
	int64_t p_timeout_min_encoded;
	PtrToArg<int64_t>::encode(p_timeout_min, &p_timeout_min_encoded);
	int64_t p_timeout_max_encoded;
	PtrToArg<int64_t>::encode(p_timeout_max, &p_timeout_max_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_timeout_encoded, &p_timeout_min_encoded, &p_timeout_max_encoded);
}

int32_t ENetPacketPeer::get_packet_flags() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetPacketPeer::get_class_static()._native_ptr(), StringName("get_packet_flags")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

String ENetPacketPeer::get_remote_address() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetPacketPeer::get_class_static()._native_ptr(), StringName("get_remote_address")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

int32_t ENetPacketPeer::get_remote_port() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetPacketPeer::get_class_static()._native_ptr(), StringName("get_remote_port")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

double ENetPacketPeer::get_statistic(ENetPacketPeer::PeerStatistic p_statistic) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetPacketPeer::get_class_static()._native_ptr(), StringName("get_statistic")._native_ptr(), 1642578323);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_statistic_encoded;
	PtrToArg<int64_t>::encode(p_statistic, &p_statistic_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_statistic_encoded);
}

ENetPacketPeer::PeerState ENetPacketPeer::get_state() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetPacketPeer::get_class_static()._native_ptr(), StringName("get_state")._native_ptr(), 711068532);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (ENetPacketPeer::PeerState(0)));
	return (ENetPacketPeer::PeerState)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t ENetPacketPeer::get_channels() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetPacketPeer::get_class_static()._native_ptr(), StringName("get_channels")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool ENetPacketPeer::is_active() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetPacketPeer::get_class_static()._native_ptr(), StringName("is_active")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

} // namespace godot
