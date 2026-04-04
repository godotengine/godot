/**************************************************************************/
/*  rd_vertex_attribute.cpp                                               */
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

#include <godot_cpp/classes/rd_vertex_attribute.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void RDVertexAttribute::set_binding(uint32_t p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDVertexAttribute::get_class_static()._native_ptr(), StringName("set_binding")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

uint32_t RDVertexAttribute::get_binding() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDVertexAttribute::get_class_static()._native_ptr(), StringName("get_binding")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDVertexAttribute::set_location(uint32_t p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDVertexAttribute::get_class_static()._native_ptr(), StringName("set_location")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

uint32_t RDVertexAttribute::get_location() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDVertexAttribute::get_class_static()._native_ptr(), StringName("get_location")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDVertexAttribute::set_offset(uint32_t p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDVertexAttribute::get_class_static()._native_ptr(), StringName("set_offset")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

uint32_t RDVertexAttribute::get_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDVertexAttribute::get_class_static()._native_ptr(), StringName("get_offset")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDVertexAttribute::set_format(RenderingDevice::DataFormat p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDVertexAttribute::get_class_static()._native_ptr(), StringName("set_format")._native_ptr(), 565531219);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

RenderingDevice::DataFormat RDVertexAttribute::get_format() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDVertexAttribute::get_class_static()._native_ptr(), StringName("get_format")._native_ptr(), 2235804183);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingDevice::DataFormat(0)));
	return (RenderingDevice::DataFormat)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDVertexAttribute::set_stride(uint32_t p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDVertexAttribute::get_class_static()._native_ptr(), StringName("set_stride")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

uint32_t RDVertexAttribute::get_stride() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDVertexAttribute::get_class_static()._native_ptr(), StringName("get_stride")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDVertexAttribute::set_frequency(RenderingDevice::VertexFrequency p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDVertexAttribute::get_class_static()._native_ptr(), StringName("set_frequency")._native_ptr(), 522141836);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

RenderingDevice::VertexFrequency RDVertexAttribute::get_frequency() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDVertexAttribute::get_class_static()._native_ptr(), StringName("get_frequency")._native_ptr(), 4154106413);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingDevice::VertexFrequency(0)));
	return (RenderingDevice::VertexFrequency)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
