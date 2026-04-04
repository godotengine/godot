/**************************************************************************/
/*  rd_pipeline_multisample_state.cpp                                     */
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

#include <godot_cpp/classes/rd_pipeline_multisample_state.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void RDPipelineMultisampleState::set_sample_count(RenderingDevice::TextureSamples p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineMultisampleState::get_class_static()._native_ptr(), StringName("set_sample_count")._native_ptr(), 3774171498);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_member_encoded;
	PtrToArg<int64_t>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

RenderingDevice::TextureSamples RDPipelineMultisampleState::get_sample_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineMultisampleState::get_class_static()._native_ptr(), StringName("get_sample_count")._native_ptr(), 407791724);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingDevice::TextureSamples(0)));
	return (RenderingDevice::TextureSamples)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RDPipelineMultisampleState::set_enable_sample_shading(bool p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineMultisampleState::get_class_static()._native_ptr(), StringName("set_enable_sample_shading")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_member_encoded;
	PtrToArg<bool>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

bool RDPipelineMultisampleState::get_enable_sample_shading() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineMultisampleState::get_class_static()._native_ptr(), StringName("get_enable_sample_shading")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RDPipelineMultisampleState::set_min_sample_shading(float p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineMultisampleState::get_class_static()._native_ptr(), StringName("set_min_sample_shading")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_member_encoded;
	PtrToArg<double>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

float RDPipelineMultisampleState::get_min_sample_shading() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineMultisampleState::get_class_static()._native_ptr(), StringName("get_min_sample_shading")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void RDPipelineMultisampleState::set_enable_alpha_to_coverage(bool p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineMultisampleState::get_class_static()._native_ptr(), StringName("set_enable_alpha_to_coverage")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_member_encoded;
	PtrToArg<bool>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

bool RDPipelineMultisampleState::get_enable_alpha_to_coverage() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineMultisampleState::get_class_static()._native_ptr(), StringName("get_enable_alpha_to_coverage")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RDPipelineMultisampleState::set_enable_alpha_to_one(bool p_member) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineMultisampleState::get_class_static()._native_ptr(), StringName("set_enable_alpha_to_one")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_member_encoded;
	PtrToArg<bool>::encode(p_member, &p_member_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_member_encoded);
}

bool RDPipelineMultisampleState::get_enable_alpha_to_one() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineMultisampleState::get_class_static()._native_ptr(), StringName("get_enable_alpha_to_one")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RDPipelineMultisampleState::set_sample_masks(const TypedArray<int> &p_masks) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineMultisampleState::get_class_static()._native_ptr(), StringName("set_sample_masks")._native_ptr(), 381264803);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_masks);
}

TypedArray<int> RDPipelineMultisampleState::get_sample_masks() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDPipelineMultisampleState::get_class_static()._native_ptr(), StringName("get_sample_masks")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<int>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<int>>(_gde_method_bind, _owner);
}

} // namespace godot
