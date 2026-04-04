/**************************************************************************/
/*  rd_sampler_state.cpp                                                  */
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

#include <godot_cpp/classes/rd_sampler_state.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void RDSamplerState::set_mag_filter(RenderingDevice::SamplerFilter p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDSamplerState::get_class_static()._native_ptr(), StringName("set_mag_filter")._native_ptr(), 1493420382);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

RenderingDevice::SamplerFilter RDSamplerState::get_mag_filter() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDSamplerState::get_class_static()._native_ptr(), StringName("get_mag_filter")._native_ptr(), 2209202801);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingDevice::SamplerFilter(0)));
	return (RenderingDevice::SamplerFilter)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDSamplerState::set_min_filter(RenderingDevice::SamplerFilter p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDSamplerState::get_class_static()._native_ptr(), StringName("set_min_filter")._native_ptr(), 1493420382);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

RenderingDevice::SamplerFilter RDSamplerState::get_min_filter() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDSamplerState::get_class_static()._native_ptr(), StringName("get_min_filter")._native_ptr(), 2209202801);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingDevice::SamplerFilter(0)));
	return (RenderingDevice::SamplerFilter)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDSamplerState::set_mip_filter(RenderingDevice::SamplerFilter p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDSamplerState::get_class_static()._native_ptr(), StringName("set_mip_filter")._native_ptr(), 1493420382);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

RenderingDevice::SamplerFilter RDSamplerState::get_mip_filter() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDSamplerState::get_class_static()._native_ptr(), StringName("get_mip_filter")._native_ptr(), 2209202801);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingDevice::SamplerFilter(0)));
	return (RenderingDevice::SamplerFilter)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDSamplerState::set_repeat_u(RenderingDevice::SamplerRepeatMode p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDSamplerState::get_class_static()._native_ptr(), StringName("set_repeat_u")._native_ptr(), 246127626);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

RenderingDevice::SamplerRepeatMode RDSamplerState::get_repeat_u() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDSamplerState::get_class_static()._native_ptr(), StringName("get_repeat_u")._native_ptr(), 3227895872);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingDevice::SamplerRepeatMode(0)));
	return (RenderingDevice::SamplerRepeatMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDSamplerState::set_repeat_v(RenderingDevice::SamplerRepeatMode p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDSamplerState::get_class_static()._native_ptr(), StringName("set_repeat_v")._native_ptr(), 246127626);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

RenderingDevice::SamplerRepeatMode RDSamplerState::get_repeat_v() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDSamplerState::get_class_static()._native_ptr(), StringName("get_repeat_v")._native_ptr(), 3227895872);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingDevice::SamplerRepeatMode(0)));
	return (RenderingDevice::SamplerRepeatMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDSamplerState::set_repeat_w(RenderingDevice::SamplerRepeatMode p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDSamplerState::get_class_static()._native_ptr(), StringName("set_repeat_w")._native_ptr(), 246127626);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

RenderingDevice::SamplerRepeatMode RDSamplerState::get_repeat_w() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDSamplerState::get_class_static()._native_ptr(), StringName("get_repeat_w")._native_ptr(), 3227895872);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingDevice::SamplerRepeatMode(0)));
	return (RenderingDevice::SamplerRepeatMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDSamplerState::set_lod_bias(float p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDSamplerState::get_class_static()._native_ptr(), StringName("set_lod_bias")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_member_encoded;
	PtrToArg<double>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

float RDSamplerState::get_lod_bias() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDSamplerState::get_class_static()._native_ptr(), StringName("get_lod_bias")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void RDSamplerState::set_use_anisotropy(bool p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDSamplerState::get_class_static()._native_ptr(), StringName("set_use_anisotropy")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_member_encoded;
	PtrToArg<bool>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

bool RDSamplerState::get_use_anisotropy() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDSamplerState::get_class_static()._native_ptr(), StringName("get_use_anisotropy")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RDSamplerState::set_anisotropy_max(float p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDSamplerState::get_class_static()._native_ptr(), StringName("set_anisotropy_max")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_member_encoded;
	PtrToArg<double>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

float RDSamplerState::get_anisotropy_max() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDSamplerState::get_class_static()._native_ptr(), StringName("get_anisotropy_max")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void RDSamplerState::set_enable_compare(bool p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDSamplerState::get_class_static()._native_ptr(), StringName("set_enable_compare")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_member_encoded;
	PtrToArg<bool>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

bool RDSamplerState::get_enable_compare() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDSamplerState::get_class_static()._native_ptr(), StringName("get_enable_compare")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RDSamplerState::set_compare_op(RenderingDevice::CompareOperator p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDSamplerState::get_class_static()._native_ptr(), StringName("set_compare_op")._native_ptr(), 2573711505);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

RenderingDevice::CompareOperator RDSamplerState::get_compare_op() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDSamplerState::get_class_static()._native_ptr(), StringName("get_compare_op")._native_ptr(), 269730778);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingDevice::CompareOperator(0)));
	return (RenderingDevice::CompareOperator)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDSamplerState::set_min_lod(float p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDSamplerState::get_class_static()._native_ptr(), StringName("set_min_lod")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_member_encoded;
	PtrToArg<double>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

float RDSamplerState::get_min_lod() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDSamplerState::get_class_static()._native_ptr(), StringName("get_min_lod")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void RDSamplerState::set_max_lod(float p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDSamplerState::get_class_static()._native_ptr(), StringName("set_max_lod")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_member_encoded;
	PtrToArg<double>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

float RDSamplerState::get_max_lod() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDSamplerState::get_class_static()._native_ptr(), StringName("get_max_lod")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void RDSamplerState::set_border_color(RenderingDevice::SamplerBorderColor p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDSamplerState::get_class_static()._native_ptr(), StringName("set_border_color")._native_ptr(), 1115869595);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

RenderingDevice::SamplerBorderColor RDSamplerState::get_border_color() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDSamplerState::get_class_static()._native_ptr(), StringName("get_border_color")._native_ptr(), 3514246478);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingDevice::SamplerBorderColor(0)));
	return (RenderingDevice::SamplerBorderColor)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDSamplerState::set_unnormalized_uvw(bool p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDSamplerState::get_class_static()._native_ptr(), StringName("set_unnormalized_uvw")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_member_encoded;
	PtrToArg<bool>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

bool RDSamplerState::get_unnormalized_uvw() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDSamplerState::get_class_static()._native_ptr(), StringName("get_unnormalized_uvw")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

} // namespace godot
