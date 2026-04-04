/**************************************************************************/
/*  gltf_buffer_view.cpp                                                  */
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

#include <godot_cpp/classes/gltf_buffer_view.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/gltf_state.hpp>

namespace godot {

PackedByteArray GLTFBufferView::load_buffer_view_data(const Ref<GLTFState> &p_state) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFBufferView::get_class_static()._native_ptr(), StringName("load_buffer_view_data")._native_ptr(), 3945446907);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedByteArray()));
	return ::godot::internal::_call_native_mb_ret<PackedByteArray>(_gde_method_bind, _owner, (p_state != nullptr ? &p_state->_owner : nullptr));
}

Ref<GLTFBufferView> GLTFBufferView::from_dictionary(const Dictionary &p_dictionary) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFBufferView::get_class_static()._native_ptr(), StringName("from_dictionary")._native_ptr(), 2594413512);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<GLTFBufferView>()));
	return Ref<GLTFBufferView>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<GLTFBufferView>(_gde_method_bind, nullptr, &p_dictionary));
}

Dictionary GLTFBufferView::to_dictionary() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFBufferView::get_class_static()._native_ptr(), StringName("to_dictionary")._native_ptr(), 3102165223);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner);
}

int32_t GLTFBufferView::get_buffer() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFBufferView::get_class_static()._native_ptr(), StringName("get_buffer")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void GLTFBufferView::set_buffer(int32_t p_buffer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFBufferView::get_class_static()._native_ptr(), StringName("set_buffer")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_buffer_encoded;
	PtrToArg<int64_t>::encode(p_buffer, &p_buffer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_buffer_encoded);
}

int64_t GLTFBufferView::get_byte_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFBufferView::get_class_static()._native_ptr(), StringName("get_byte_offset")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void GLTFBufferView::set_byte_offset(int64_t p_byte_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFBufferView::get_class_static()._native_ptr(), StringName("set_byte_offset")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_byte_offset_encoded;
	PtrToArg<int64_t>::encode(p_byte_offset, &p_byte_offset_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_byte_offset_encoded);
}

int64_t GLTFBufferView::get_byte_length() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFBufferView::get_class_static()._native_ptr(), StringName("get_byte_length")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void GLTFBufferView::set_byte_length(int64_t p_byte_length) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFBufferView::get_class_static()._native_ptr(), StringName("set_byte_length")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_byte_length_encoded;
	PtrToArg<int64_t>::encode(p_byte_length, &p_byte_length_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_byte_length_encoded);
}

int64_t GLTFBufferView::get_byte_stride() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFBufferView::get_class_static()._native_ptr(), StringName("get_byte_stride")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void GLTFBufferView::set_byte_stride(int64_t p_byte_stride) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFBufferView::get_class_static()._native_ptr(), StringName("set_byte_stride")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_byte_stride_encoded;
	PtrToArg<int64_t>::encode(p_byte_stride, &p_byte_stride_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_byte_stride_encoded);
}

bool GLTFBufferView::get_indices() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFBufferView::get_class_static()._native_ptr(), StringName("get_indices")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void GLTFBufferView::set_indices(bool p_indices) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFBufferView::get_class_static()._native_ptr(), StringName("set_indices")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_indices_encoded;
	PtrToArg<bool>::encode(p_indices, &p_indices_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_indices_encoded);
}

bool GLTFBufferView::get_vertex_attributes() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFBufferView::get_class_static()._native_ptr(), StringName("get_vertex_attributes")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void GLTFBufferView::set_vertex_attributes(bool p_is_attributes) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFBufferView::get_class_static()._native_ptr(), StringName("set_vertex_attributes")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_is_attributes_encoded;
	PtrToArg<bool>::encode(p_is_attributes, &p_is_attributes_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_is_attributes_encoded);
}

} // namespace godot
