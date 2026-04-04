/**************************************************************************/
/*  rd_pipeline_depth_stencil_state.cpp                                   */
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

#include <godot_cpp/classes/rd_pipeline_depth_stencil_state.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void RDPipelineDepthStencilState::set_enable_depth_test(bool p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("set_enable_depth_test")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_member_encoded;
	PtrToArg<bool>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

bool RDPipelineDepthStencilState::get_enable_depth_test() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("get_enable_depth_test")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RDPipelineDepthStencilState::set_enable_depth_write(bool p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("set_enable_depth_write")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_member_encoded;
	PtrToArg<bool>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

bool RDPipelineDepthStencilState::get_enable_depth_write() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("get_enable_depth_write")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RDPipelineDepthStencilState::set_depth_compare_operator(RenderingDevice::CompareOperator p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("set_depth_compare_operator")._native_ptr(), 2573711505);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

RenderingDevice::CompareOperator RDPipelineDepthStencilState::get_depth_compare_operator() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("get_depth_compare_operator")._native_ptr(), 269730778);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingDevice::CompareOperator(0)));
	return (RenderingDevice::CompareOperator)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDPipelineDepthStencilState::set_enable_depth_range(bool p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("set_enable_depth_range")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_member_encoded;
	PtrToArg<bool>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

bool RDPipelineDepthStencilState::get_enable_depth_range() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("get_enable_depth_range")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RDPipelineDepthStencilState::set_depth_range_min(float p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("set_depth_range_min")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_member_encoded;
	PtrToArg<double>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

float RDPipelineDepthStencilState::get_depth_range_min() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("get_depth_range_min")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void RDPipelineDepthStencilState::set_depth_range_max(float p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("set_depth_range_max")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_member_encoded;
	PtrToArg<double>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

float RDPipelineDepthStencilState::get_depth_range_max() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("get_depth_range_max")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void RDPipelineDepthStencilState::set_enable_stencil(bool p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("set_enable_stencil")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_member_encoded;
	PtrToArg<bool>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

bool RDPipelineDepthStencilState::get_enable_stencil() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("get_enable_stencil")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RDPipelineDepthStencilState::set_front_op_fail(RenderingDevice::StencilOperation p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("set_front_op_fail")._native_ptr(), 2092799566);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

RenderingDevice::StencilOperation RDPipelineDepthStencilState::get_front_op_fail() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("get_front_op_fail")._native_ptr(), 1714732389);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingDevice::StencilOperation(0)));
	return (RenderingDevice::StencilOperation)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDPipelineDepthStencilState::set_front_op_pass(RenderingDevice::StencilOperation p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("set_front_op_pass")._native_ptr(), 2092799566);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

RenderingDevice::StencilOperation RDPipelineDepthStencilState::get_front_op_pass() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("get_front_op_pass")._native_ptr(), 1714732389);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingDevice::StencilOperation(0)));
	return (RenderingDevice::StencilOperation)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDPipelineDepthStencilState::set_front_op_depth_fail(RenderingDevice::StencilOperation p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("set_front_op_depth_fail")._native_ptr(), 2092799566);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

RenderingDevice::StencilOperation RDPipelineDepthStencilState::get_front_op_depth_fail() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("get_front_op_depth_fail")._native_ptr(), 1714732389);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingDevice::StencilOperation(0)));
	return (RenderingDevice::StencilOperation)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDPipelineDepthStencilState::set_front_op_compare(RenderingDevice::CompareOperator p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("set_front_op_compare")._native_ptr(), 2573711505);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

RenderingDevice::CompareOperator RDPipelineDepthStencilState::get_front_op_compare() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("get_front_op_compare")._native_ptr(), 269730778);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingDevice::CompareOperator(0)));
	return (RenderingDevice::CompareOperator)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDPipelineDepthStencilState::set_front_op_compare_mask(uint32_t p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("set_front_op_compare_mask")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

uint32_t RDPipelineDepthStencilState::get_front_op_compare_mask() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("get_front_op_compare_mask")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDPipelineDepthStencilState::set_front_op_write_mask(uint32_t p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("set_front_op_write_mask")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

uint32_t RDPipelineDepthStencilState::get_front_op_write_mask() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("get_front_op_write_mask")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDPipelineDepthStencilState::set_front_op_reference(uint32_t p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("set_front_op_reference")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

uint32_t RDPipelineDepthStencilState::get_front_op_reference() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("get_front_op_reference")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDPipelineDepthStencilState::set_back_op_fail(RenderingDevice::StencilOperation p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("set_back_op_fail")._native_ptr(), 2092799566);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

RenderingDevice::StencilOperation RDPipelineDepthStencilState::get_back_op_fail() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("get_back_op_fail")._native_ptr(), 1714732389);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingDevice::StencilOperation(0)));
	return (RenderingDevice::StencilOperation)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDPipelineDepthStencilState::set_back_op_pass(RenderingDevice::StencilOperation p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("set_back_op_pass")._native_ptr(), 2092799566);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

RenderingDevice::StencilOperation RDPipelineDepthStencilState::get_back_op_pass() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("get_back_op_pass")._native_ptr(), 1714732389);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingDevice::StencilOperation(0)));
	return (RenderingDevice::StencilOperation)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDPipelineDepthStencilState::set_back_op_depth_fail(RenderingDevice::StencilOperation p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("set_back_op_depth_fail")._native_ptr(), 2092799566);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

RenderingDevice::StencilOperation RDPipelineDepthStencilState::get_back_op_depth_fail() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("get_back_op_depth_fail")._native_ptr(), 1714732389);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingDevice::StencilOperation(0)));
	return (RenderingDevice::StencilOperation)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDPipelineDepthStencilState::set_back_op_compare(RenderingDevice::CompareOperator p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("set_back_op_compare")._native_ptr(), 2573711505);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

RenderingDevice::CompareOperator RDPipelineDepthStencilState::get_back_op_compare() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("get_back_op_compare")._native_ptr(), 269730778);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingDevice::CompareOperator(0)));
	return (RenderingDevice::CompareOperator)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDPipelineDepthStencilState::set_back_op_compare_mask(uint32_t p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("set_back_op_compare_mask")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

uint32_t RDPipelineDepthStencilState::get_back_op_compare_mask() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("get_back_op_compare_mask")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDPipelineDepthStencilState::set_back_op_write_mask(uint32_t p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("set_back_op_write_mask")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

uint32_t RDPipelineDepthStencilState::get_back_op_write_mask() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("get_back_op_write_mask")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDPipelineDepthStencilState::set_back_op_reference(uint32_t p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("set_back_op_reference")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

uint32_t RDPipelineDepthStencilState::get_back_op_reference() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineDepthStencilState::get_class_static()._native_ptr(), StringName("get_back_op_reference")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
