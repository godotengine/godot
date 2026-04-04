/**************************************************************************/
/*  stream_peer_tls.cpp                                                   */
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

#include <godot_cpp/classes/stream_peer_tls.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/variant/string.hpp>

namespace godot {

void StreamPeerTLS::poll() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StreamPeerTLS::get_class_static()._native_ptr(), StringName("poll")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

Error StreamPeerTLS::accept_stream(const Ref<StreamPeer> &p_stream, const Ref<TLSOptions> &p_server_options) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StreamPeerTLS::get_class_static()._native_ptr(), StringName("accept_stream")._native_ptr(), 4292689651);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, (p_stream != nullptr ? &p_stream->_owner : nullptr), (p_server_options != nullptr ? &p_server_options->_owner : nullptr));
}

Error StreamPeerTLS::connect_to_stream(const Ref<StreamPeer> &p_stream, const String &p_common_name, const Ref<TLSOptions> &p_client_options) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StreamPeerTLS::get_class_static()._native_ptr(), StringName("connect_to_stream")._native_ptr(), 57169517);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, (p_stream != nullptr ? &p_stream->_owner : nullptr), &p_common_name, (p_client_options != nullptr ? &p_client_options->_owner : nullptr));
}

StreamPeerTLS::Status StreamPeerTLS::get_status() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StreamPeerTLS::get_class_static()._native_ptr(), StringName("get_status")._native_ptr(), 1128380576);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StreamPeerTLS::Status(0)));
	return (StreamPeerTLS::Status)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Ref<StreamPeer> StreamPeerTLS::get_stream() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StreamPeerTLS::get_class_static()._native_ptr(), StringName("get_stream")._native_ptr(), 2741655269);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<StreamPeer>()));
	return Ref<StreamPeer>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<StreamPeer>(_gde_method_bind, _owner));
}

void StreamPeerTLS::disconnect_from_stream() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StreamPeerTLS::get_class_static()._native_ptr(), StringName("disconnect_from_stream")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

} // namespace godot
