/**************************************************************************/
/*  stream_peer_gzip.cpp                                                  */
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

#include <godot_cpp/classes/stream_peer_gzip.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

Error StreamPeerGZIP::start_compression(bool p_use_deflate, int32_t p_buffer_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StreamPeerGZIP::get_class_static()._native_ptr(), StringName("start_compression")._native_ptr(), 781582770);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int8_t p_use_deflate_encoded;
	PtrToArg<bool>::encode(p_use_deflate, &p_use_deflate_encoded);
	int64_t p_buffer_size_encoded;
	PtrToArg<int64_t>::encode(p_buffer_size, &p_buffer_size_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_use_deflate_encoded, &p_buffer_size_encoded);
}

Error StreamPeerGZIP::start_decompression(bool p_use_deflate, int32_t p_buffer_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StreamPeerGZIP::get_class_static()._native_ptr(), StringName("start_decompression")._native_ptr(), 781582770);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int8_t p_use_deflate_encoded;
	PtrToArg<bool>::encode(p_use_deflate, &p_use_deflate_encoded);
	int64_t p_buffer_size_encoded;
	PtrToArg<int64_t>::encode(p_buffer_size, &p_buffer_size_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_use_deflate_encoded, &p_buffer_size_encoded);
}

Error StreamPeerGZIP::finish() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StreamPeerGZIP::get_class_static()._native_ptr(), StringName("finish")._native_ptr(), 166280745);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void StreamPeerGZIP::clear() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(StreamPeerGZIP::get_class_static()._native_ptr(), StringName("clear")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

} // namespace godot
