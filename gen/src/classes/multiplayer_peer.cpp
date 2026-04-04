/**************************************************************************/
/*  multiplayer_peer.cpp                                                  */
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

#include <godot_cpp/classes/multiplayer_peer.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void MultiplayerPeer::set_transfer_channel(int32_t p_channel) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerPeer::get_class_static()._native_ptr(), StringName("set_transfer_channel")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_channel_encoded;
	PtrToArg<int64_t>::encode(p_channel, &p_channel_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_channel_encoded);
}

int32_t MultiplayerPeer::get_transfer_channel() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerPeer::get_class_static()._native_ptr(), StringName("get_transfer_channel")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void MultiplayerPeer::set_transfer_mode(MultiplayerPeer::TransferMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerPeer::get_class_static()._native_ptr(), StringName("set_transfer_mode")._native_ptr(), 950411049);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

MultiplayerPeer::TransferMode MultiplayerPeer::get_transfer_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerPeer::get_class_static()._native_ptr(), StringName("get_transfer_mode")._native_ptr(), 3369852622);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (MultiplayerPeer::TransferMode(0)));
	return (MultiplayerPeer::TransferMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void MultiplayerPeer::set_target_peer(int32_t p_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerPeer::get_class_static()._native_ptr(), StringName("set_target_peer")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id_encoded);
}

int32_t MultiplayerPeer::get_packet_peer() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerPeer::get_class_static()._native_ptr(), StringName("get_packet_peer")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t MultiplayerPeer::get_packet_channel() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerPeer::get_class_static()._native_ptr(), StringName("get_packet_channel")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

MultiplayerPeer::TransferMode MultiplayerPeer::get_packet_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerPeer::get_class_static()._native_ptr(), StringName("get_packet_mode")._native_ptr(), 3369852622);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (MultiplayerPeer::TransferMode(0)));
	return (MultiplayerPeer::TransferMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void MultiplayerPeer::poll() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerPeer::get_class_static()._native_ptr(), StringName("poll")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void MultiplayerPeer::close() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerPeer::get_class_static()._native_ptr(), StringName("close")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void MultiplayerPeer::disconnect_peer(int32_t p_peer, bool p_force) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerPeer::get_class_static()._native_ptr(), StringName("disconnect_peer")._native_ptr(), 4023243586);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_peer_encoded;
	PtrToArg<int64_t>::encode(p_peer, &p_peer_encoded);
	int8_t p_force_encoded;
	PtrToArg<bool>::encode(p_force, &p_force_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_peer_encoded, &p_force_encoded);
}

MultiplayerPeer::ConnectionStatus MultiplayerPeer::get_connection_status() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerPeer::get_class_static()._native_ptr(), StringName("get_connection_status")._native_ptr(), 2147374275);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (MultiplayerPeer::ConnectionStatus(0)));
	return (MultiplayerPeer::ConnectionStatus)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t MultiplayerPeer::get_unique_id() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerPeer::get_class_static()._native_ptr(), StringName("get_unique_id")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

uint32_t MultiplayerPeer::generate_unique_id() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerPeer::get_class_static()._native_ptr(), StringName("generate_unique_id")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void MultiplayerPeer::set_refuse_new_connections(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerPeer::get_class_static()._native_ptr(), StringName("set_refuse_new_connections")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool MultiplayerPeer::is_refusing_new_connections() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerPeer::get_class_static()._native_ptr(), StringName("is_refusing_new_connections")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool MultiplayerPeer::is_server_relay_supported() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(MultiplayerPeer::get_class_static()._native_ptr(), StringName("is_server_relay_supported")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

} // namespace godot
