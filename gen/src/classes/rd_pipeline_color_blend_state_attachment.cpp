/**************************************************************************/
/*  rd_pipeline_color_blend_state_attachment.cpp                          */
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

#include <godot_cpp/classes/rd_pipeline_color_blend_state_attachment.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void RDPipelineColorBlendStateAttachment::set_as_mix() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineColorBlendStateAttachment::get_class_static()._native_ptr(), StringName("set_as_mix")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void RDPipelineColorBlendStateAttachment::set_enable_blend(bool p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineColorBlendStateAttachment::get_class_static()._native_ptr(), StringName("set_enable_blend")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_member_encoded;
	PtrToArg<bool>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

bool RDPipelineColorBlendStateAttachment::get_enable_blend() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineColorBlendStateAttachment::get_class_static()._native_ptr(), StringName("get_enable_blend")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RDPipelineColorBlendStateAttachment::set_src_color_blend_factor(RenderingDevice::BlendFactor p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineColorBlendStateAttachment::get_class_static()._native_ptr(), StringName("set_src_color_blend_factor")._native_ptr(), 2251019273);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

RenderingDevice::BlendFactor RDPipelineColorBlendStateAttachment::get_src_color_blend_factor() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineColorBlendStateAttachment::get_class_static()._native_ptr(), StringName("get_src_color_blend_factor")._native_ptr(), 3691288359);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingDevice::BlendFactor(0)));
	return (RenderingDevice::BlendFactor)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDPipelineColorBlendStateAttachment::set_dst_color_blend_factor(RenderingDevice::BlendFactor p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineColorBlendStateAttachment::get_class_static()._native_ptr(), StringName("set_dst_color_blend_factor")._native_ptr(), 2251019273);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

RenderingDevice::BlendFactor RDPipelineColorBlendStateAttachment::get_dst_color_blend_factor() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineColorBlendStateAttachment::get_class_static()._native_ptr(), StringName("get_dst_color_blend_factor")._native_ptr(), 3691288359);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingDevice::BlendFactor(0)));
	return (RenderingDevice::BlendFactor)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDPipelineColorBlendStateAttachment::set_color_blend_op(RenderingDevice::BlendOperation p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineColorBlendStateAttachment::get_class_static()._native_ptr(), StringName("set_color_blend_op")._native_ptr(), 3073022720);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

RenderingDevice::BlendOperation RDPipelineColorBlendStateAttachment::get_color_blend_op() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineColorBlendStateAttachment::get_class_static()._native_ptr(), StringName("get_color_blend_op")._native_ptr(), 1385093561);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingDevice::BlendOperation(0)));
	return (RenderingDevice::BlendOperation)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDPipelineColorBlendStateAttachment::set_src_alpha_blend_factor(RenderingDevice::BlendFactor p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineColorBlendStateAttachment::get_class_static()._native_ptr(), StringName("set_src_alpha_blend_factor")._native_ptr(), 2251019273);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

RenderingDevice::BlendFactor RDPipelineColorBlendStateAttachment::get_src_alpha_blend_factor() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineColorBlendStateAttachment::get_class_static()._native_ptr(), StringName("get_src_alpha_blend_factor")._native_ptr(), 3691288359);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingDevice::BlendFactor(0)));
	return (RenderingDevice::BlendFactor)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDPipelineColorBlendStateAttachment::set_dst_alpha_blend_factor(RenderingDevice::BlendFactor p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineColorBlendStateAttachment::get_class_static()._native_ptr(), StringName("set_dst_alpha_blend_factor")._native_ptr(), 2251019273);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

RenderingDevice::BlendFactor RDPipelineColorBlendStateAttachment::get_dst_alpha_blend_factor() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineColorBlendStateAttachment::get_class_static()._native_ptr(), StringName("get_dst_alpha_blend_factor")._native_ptr(), 3691288359);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingDevice::BlendFactor(0)));
	return (RenderingDevice::BlendFactor)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDPipelineColorBlendStateAttachment::set_alpha_blend_op(RenderingDevice::BlendOperation p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineColorBlendStateAttachment::get_class_static()._native_ptr(), StringName("set_alpha_blend_op")._native_ptr(), 3073022720);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

RenderingDevice::BlendOperation RDPipelineColorBlendStateAttachment::get_alpha_blend_op() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineColorBlendStateAttachment::get_class_static()._native_ptr(), StringName("get_alpha_blend_op")._native_ptr(), 1385093561);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingDevice::BlendOperation(0)));
	return (RenderingDevice::BlendOperation)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDPipelineColorBlendStateAttachment::set_write_r(bool p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineColorBlendStateAttachment::get_class_static()._native_ptr(), StringName("set_write_r")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_member_encoded;
	PtrToArg<bool>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

bool RDPipelineColorBlendStateAttachment::get_write_r() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineColorBlendStateAttachment::get_class_static()._native_ptr(), StringName("get_write_r")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RDPipelineColorBlendStateAttachment::set_write_g(bool p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineColorBlendStateAttachment::get_class_static()._native_ptr(), StringName("set_write_g")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_member_encoded;
	PtrToArg<bool>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

bool RDPipelineColorBlendStateAttachment::get_write_g() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineColorBlendStateAttachment::get_class_static()._native_ptr(), StringName("get_write_g")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RDPipelineColorBlendStateAttachment::set_write_b(bool p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineColorBlendStateAttachment::get_class_static()._native_ptr(), StringName("set_write_b")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_member_encoded;
	PtrToArg<bool>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

bool RDPipelineColorBlendStateAttachment::get_write_b() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineColorBlendStateAttachment::get_class_static()._native_ptr(), StringName("get_write_b")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RDPipelineColorBlendStateAttachment::set_write_a(bool p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineColorBlendStateAttachment::get_class_static()._native_ptr(), StringName("set_write_a")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_member_encoded;
	PtrToArg<bool>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

bool RDPipelineColorBlendStateAttachment::get_write_a() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineColorBlendStateAttachment::get_class_static()._native_ptr(), StringName("get_write_a")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

} // namespace godot
