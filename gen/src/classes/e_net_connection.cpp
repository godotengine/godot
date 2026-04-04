/**************************************************************************/
/*  e_net_connection.cpp                                                  */
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

#include <godot_cpp/classes/e_net_connection.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/e_net_packet_peer.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/string.hpp>

namespace godot {

Error ENetConnection::create_host_bound(const String &p_bind_address, int32_t p_bind_port, int32_t p_max_peers, int32_t p_max_channels, int32_t p_in_bandwidth, int32_t p_out_bandwidth) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetConnection::get_class_static()._native_ptr(), StringName("create_host_bound")._native_ptr(), 1515002313);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_bind_port_encoded;
	PtrToArg<int64_t>::encode(p_bind_port, &p_bind_port_encoded);
	int64_t p_max_peers_encoded;
	PtrToArg<int64_t>::encode(p_max_peers, &p_max_peers_encoded);
	int64_t p_max_channels_encoded;
	PtrToArg<int64_t>::encode(p_max_channels, &p_max_channels_encoded);
	int64_t p_in_bandwidth_encoded;
	PtrToArg<int64_t>::encode(p_in_bandwidth, &p_in_bandwidth_encoded);
	int64_t p_out_bandwidth_encoded;
	PtrToArg<int64_t>::encode(p_out_bandwidth, &p_out_bandwidth_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_bind_address, &p_bind_port_encoded, &p_max_peers_encoded, &p_max_channels_encoded, &p_in_bandwidth_encoded, &p_out_bandwidth_encoded);
}

Error ENetConnection::create_host(int32_t p_max_peers, int32_t p_max_channels, int32_t p_in_bandwidth, int32_t p_out_bandwidth) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetConnection::get_class_static()._native_ptr(), StringName("create_host")._native_ptr(), 117198950);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_max_peers_encoded;
	PtrToArg<int64_t>::encode(p_max_peers, &p_max_peers_encoded);
	int64_t p_max_channels_encoded;
	PtrToArg<int64_t>::encode(p_max_channels, &p_max_channels_encoded);
	int64_t p_in_bandwidth_encoded;
	PtrToArg<int64_t>::encode(p_in_bandwidth, &p_in_bandwidth_encoded);
	int64_t p_out_bandwidth_encoded;
	PtrToArg<int64_t>::encode(p_out_bandwidth, &p_out_bandwidth_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_max_peers_encoded, &p_max_channels_encoded, &p_in_bandwidth_encoded, &p_out_bandwidth_encoded);
}

void ENetConnection::destroy() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetConnection::get_class_static()._native_ptr(), StringName("destroy")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

Ref<ENetPacketPeer> ENetConnection::connect_to_host(const String &p_address, int32_t p_port, int32_t p_channels, int32_t p_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetConnection::get_class_static()._native_ptr(), StringName("connect_to_host")._native_ptr(), 2171300490);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<ENetPacketPeer>()));
	int64_t p_port_encoded;
	PtrToArg<int64_t>::encode(p_port, &p_port_encoded);
	int64_t p_channels_encoded;
	PtrToArg<int64_t>::encode(p_channels, &p_channels_encoded);
	int64_t p_data_encoded;
	PtrToArg<int64_t>::encode(p_data, &p_data_encoded);
	return Ref<ENetPacketPeer>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<ENetPacketPeer>(_gde_method_bind, _owner, &p_address, &p_port_encoded, &p_channels_encoded, &p_data_encoded));
}

Array ENetConnection::service(int32_t p_timeout) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetConnection::get_class_static()._native_ptr(), StringName("service")._native_ptr(), 2402345344);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Array()));
	int64_t p_timeout_encoded;
	PtrToArg<int64_t>::encode(p_timeout, &p_timeout_encoded);
	return ::godot::internal::_call_native_mb_ret<Array>(_gde_method_bind, _owner, &p_timeout_encoded);
}

void ENetConnection::flush() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetConnection::get_class_static()._native_ptr(), StringName("flush")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void ENetConnection::bandwidth_limit(int32_t p_in_bandwidth, int32_t p_out_bandwidth) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetConnection::get_class_static()._native_ptr(), StringName("bandwidth_limit")._native_ptr(), 2302169788);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_in_bandwidth_encoded;
	PtrToArg<int64_t>::encode(p_in_bandwidth, &p_in_bandwidth_encoded);
	int64_t p_out_bandwidth_encoded;
	PtrToArg<int64_t>::encode(p_out_bandwidth, &p_out_bandwidth_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_in_bandwidth_encoded, &p_out_bandwidth_encoded);
}

void ENetConnection::channel_limit(int32_t p_limit) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetConnection::get_class_static()._native_ptr(), StringName("channel_limit")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_limit_encoded;
	PtrToArg<int64_t>::encode(p_limit, &p_limit_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_limit_encoded);
}

void ENetConnection::broadcast(int32_t p_channel, const PackedByteArray &p_packet, int32_t p_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetConnection::get_class_static()._native_ptr(), StringName("broadcast")._native_ptr(), 2772371345);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_channel_encoded;
	PtrToArg<int64_t>::encode(p_channel, &p_channel_encoded);
	int64_t p_flags_encoded;
	PtrToArg<int64_t>::encode(p_flags, &p_flags_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_channel_encoded, &p_packet, &p_flags_encoded);
}

void ENetConnection::compress(ENetConnection::CompressionMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetConnection::get_class_static()._native_ptr(), StringName("compress")._native_ptr(), 2660215187);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

Error ENetConnection::dtls_server_setup(const Ref<TLSOptions> &p_server_options) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetConnection::get_class_static()._native_ptr(), StringName("dtls_server_setup")._native_ptr(), 1262296096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, (p_server_options != nullptr ? &p_server_options->_owner : nullptr));
}

Error ENetConnection::dtls_client_setup(const String &p_hostname, const Ref<TLSOptions> &p_client_options) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetConnection::get_class_static()._native_ptr(), StringName("dtls_client_setup")._native_ptr(), 1966198364);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_hostname, (p_client_options != nullptr ? &p_client_options->_owner : nullptr));
}

void ENetConnection::refuse_new_connections(bool p_refuse) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetConnection::get_class_static()._native_ptr(), StringName("refuse_new_connections")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_refuse_encoded;
	PtrToArg<bool>::encode(p_refuse, &p_refuse_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_refuse_encoded);
}

double ENetConnection::pop_statistic(ENetConnection::HostStatistic p_statistic) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetConnection::get_class_static()._native_ptr(), StringName("pop_statistic")._native_ptr(), 2166904170);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_statistic_encoded;
	PtrToArg<int64_t>::encode(p_statistic, &p_statistic_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_statistic_encoded);
}

int32_t ENetConnection::get_max_channels() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetConnection::get_class_static()._native_ptr(), StringName("get_max_channels")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t ENetConnection::get_local_port() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetConnection::get_class_static()._native_ptr(), StringName("get_local_port")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

TypedArray<Ref<ENetPacketPeer>> ENetConnection::get_peers() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetConnection::get_class_static()._native_ptr(), StringName("get_peers")._native_ptr(), 2915620761);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Ref<ENetPacketPeer>>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Ref<ENetPacketPeer>>>(_gde_method_bind, _owner);
}

void ENetConnection::socket_send(const String &p_destination_address, int32_t p_destination_port, const PackedByteArray &p_packet) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ENetConnection::get_class_static()._native_ptr(), StringName("socket_send")._native_ptr(), 1100646812);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_destination_port_encoded;
	PtrToArg<int64_t>::encode(p_destination_port, &p_destination_port_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_destination_address, &p_destination_port_encoded, &p_packet);
}

} // namespace godot
