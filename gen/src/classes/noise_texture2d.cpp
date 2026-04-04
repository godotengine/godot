/**************************************************************************/
/*  noise_texture2d.cpp                                                   */
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

#include <godot_cpp/classes/noise_texture2d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/gradient.hpp>
#include <godot_cpp/classes/noise.hpp>

namespace godot {

void NoiseTexture2D::set_width(int32_t p_width) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NoiseTexture2D::get_class_static()._native_ptr(), StringName("set_width")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_width_encoded;
	PtrToArg<int64_t>::encode(p_width, &p_width_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_width_encoded);
}

void NoiseTexture2D::set_height(int32_t p_height) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NoiseTexture2D::get_class_static()._native_ptr(), StringName("set_height")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_height_encoded;
	PtrToArg<int64_t>::encode(p_height, &p_height_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_height_encoded);
}

void NoiseTexture2D::set_generate_mipmaps(bool p_invert) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NoiseTexture2D::get_class_static()._native_ptr(), StringName("set_generate_mipmaps")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_invert_encoded;
	PtrToArg<bool>::encode(p_invert, &p_invert_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_invert_encoded);
}

bool NoiseTexture2D::is_generating_mipmaps() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NoiseTexture2D::get_class_static()._native_ptr(), StringName("is_generating_mipmaps")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void NoiseTexture2D::set_noise(const Ref<Noise> &p_noise) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NoiseTexture2D::get_class_static()._native_ptr(), StringName("set_noise")._native_ptr(), 4135492439);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_noise != nullptr ? &p_noise->_owner : nullptr));
}

Ref<Noise> NoiseTexture2D::get_noise() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NoiseTexture2D::get_class_static()._native_ptr(), StringName("get_noise")._native_ptr(), 185851837);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Noise>()));
	return Ref<Noise>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Noise>(_gde_method_bind, _owner));
}

void NoiseTexture2D::set_color_ramp(const Ref<Gradient> &p_gradient) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NoiseTexture2D::get_class_static()._native_ptr(), StringName("set_color_ramp")._native_ptr(), 2756054477);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_gradient != nullptr ? &p_gradient->_owner : nullptr));
}

Ref<Gradient> NoiseTexture2D::get_color_ramp() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NoiseTexture2D::get_class_static()._native_ptr(), StringName("get_color_ramp")._native_ptr(), 132272999);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Gradient>()));
	return Ref<Gradient>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Gradient>(_gde_method_bind, _owner));
}

void NoiseTexture2D::set_seamless(bool p_seamless) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NoiseTexture2D::get_class_static()._native_ptr(), StringName("set_seamless")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_seamless_encoded;
	PtrToArg<bool>::encode(p_seamless, &p_seamless_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_seamless_encoded);
}

bool NoiseTexture2D::get_seamless() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NoiseTexture2D::get_class_static()._native_ptr(), StringName("get_seamless")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void NoiseTexture2D::set_invert(bool p_invert) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NoiseTexture2D::get_class_static()._native_ptr(), StringName("set_invert")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_invert_encoded;
	PtrToArg<bool>::encode(p_invert, &p_invert_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_invert_encoded);
}

bool NoiseTexture2D::get_invert() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NoiseTexture2D::get_class_static()._native_ptr(), StringName("get_invert")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void NoiseTexture2D::set_in_3d_space(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NoiseTexture2D::get_class_static()._native_ptr(), StringName("set_in_3d_space")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool NoiseTexture2D::is_in_3d_space() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NoiseTexture2D::get_class_static()._native_ptr(), StringName("is_in_3d_space")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void NoiseTexture2D::set_as_normal_map(bool p_as_normal_map) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NoiseTexture2D::get_class_static()._native_ptr(), StringName("set_as_normal_map")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_as_normal_map_encoded;
	PtrToArg<bool>::encode(p_as_normal_map, &p_as_normal_map_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_as_normal_map_encoded);
}

bool NoiseTexture2D::is_normal_map() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NoiseTexture2D::get_class_static()._native_ptr(), StringName("is_normal_map")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void NoiseTexture2D::set_normalize(bool p_normalize) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NoiseTexture2D::get_class_static()._native_ptr(), StringName("set_normalize")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_normalize_encoded;
	PtrToArg<bool>::encode(p_normalize, &p_normalize_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_normalize_encoded);
}

bool NoiseTexture2D::is_normalized() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NoiseTexture2D::get_class_static()._native_ptr(), StringName("is_normalized")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void NoiseTexture2D::set_seamless_blend_skirt(float p_seamless_blend_skirt) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NoiseTexture2D::get_class_static()._native_ptr(), StringName("set_seamless_blend_skirt")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_seamless_blend_skirt_encoded;
	PtrToArg<double>::encode(p_seamless_blend_skirt, &p_seamless_blend_skirt_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_seamless_blend_skirt_encoded);
}

float NoiseTexture2D::get_seamless_blend_skirt() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NoiseTexture2D::get_class_static()._native_ptr(), StringName("get_seamless_blend_skirt")._native_ptr(), 191475506);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void NoiseTexture2D::set_bump_strength(float p_bump_strength) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NoiseTexture2D::get_class_static()._native_ptr(), StringName("set_bump_strength")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_bump_strength_encoded;
	PtrToArg<double>::encode(p_bump_strength, &p_bump_strength_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_bump_strength_encoded);
}

float NoiseTexture2D::get_bump_strength() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NoiseTexture2D::get_class_static()._native_ptr(), StringName("get_bump_strength")._native_ptr(), 191475506);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

} // namespace godot
