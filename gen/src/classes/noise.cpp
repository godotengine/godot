/**************************************************************************/
/*  noise.cpp                                                             */
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

#include <godot_cpp/classes/noise.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/variant/vector2.hpp>
#include <godot_cpp/variant/vector3.hpp>

namespace godot {

float Noise::get_noise_1d(float p_x) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Noise::get_class_static()._native_ptr(), StringName("get_noise_1d")._native_ptr(), 3919130443);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_x_encoded);
}

float Noise::get_noise_2d(float p_x, float p_y) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Noise::get_class_static()._native_ptr(), StringName("get_noise_2d")._native_ptr(), 2753205203);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	double p_y_encoded;
	PtrToArg<double>::encode(p_y, &p_y_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_x_encoded, &p_y_encoded);
}

float Noise::get_noise_2dv(const Vector2 &p_v) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Noise::get_class_static()._native_ptr(), StringName("get_noise_2dv")._native_ptr(), 2276447920);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_v);
}

float Noise::get_noise_3d(float p_x, float p_y, float p_z) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Noise::get_class_static()._native_ptr(), StringName("get_noise_3d")._native_ptr(), 973811851);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	double p_x_encoded;
	PtrToArg<double>::encode(p_x, &p_x_encoded);
	double p_y_encoded;
	PtrToArg<double>::encode(p_y, &p_y_encoded);
	double p_z_encoded;
	PtrToArg<double>::encode(p_z, &p_z_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_x_encoded, &p_y_encoded, &p_z_encoded);
}

float Noise::get_noise_3dv(const Vector3 &p_v) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Noise::get_class_static()._native_ptr(), StringName("get_noise_3dv")._native_ptr(), 1109078154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_v);
}

Ref<Image> Noise::get_image(int32_t p_width, int32_t p_height, bool p_invert, bool p_in_3d_space, bool p_normalize) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Noise::get_class_static()._native_ptr(), StringName("get_image")._native_ptr(), 3180683109);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Image>()));
	int64_t p_width_encoded;
	PtrToArg<int64_t>::encode(p_width, &p_width_encoded);
	int64_t p_height_encoded;
	PtrToArg<int64_t>::encode(p_height, &p_height_encoded);
	int8_t p_invert_encoded;
	PtrToArg<bool>::encode(p_invert, &p_invert_encoded);
	int8_t p_in_3d_space_encoded;
	PtrToArg<bool>::encode(p_in_3d_space, &p_in_3d_space_encoded);
	int8_t p_normalize_encoded;
	PtrToArg<bool>::encode(p_normalize, &p_normalize_encoded);
	return Ref<Image>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Image>(_gde_method_bind, _owner, &p_width_encoded, &p_height_encoded, &p_invert_encoded, &p_in_3d_space_encoded, &p_normalize_encoded));
}

Ref<Image> Noise::get_seamless_image(int32_t p_width, int32_t p_height, bool p_invert, bool p_in_3d_space, float p_skirt, bool p_normalize) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Noise::get_class_static()._native_ptr(), StringName("get_seamless_image")._native_ptr(), 2770743602);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Image>()));
	int64_t p_width_encoded;
	PtrToArg<int64_t>::encode(p_width, &p_width_encoded);
	int64_t p_height_encoded;
	PtrToArg<int64_t>::encode(p_height, &p_height_encoded);
	int8_t p_invert_encoded;
	PtrToArg<bool>::encode(p_invert, &p_invert_encoded);
	int8_t p_in_3d_space_encoded;
	PtrToArg<bool>::encode(p_in_3d_space, &p_in_3d_space_encoded);
	double p_skirt_encoded;
	PtrToArg<double>::encode(p_skirt, &p_skirt_encoded);
	int8_t p_normalize_encoded;
	PtrToArg<bool>::encode(p_normalize, &p_normalize_encoded);
	return Ref<Image>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Image>(_gde_method_bind, _owner, &p_width_encoded, &p_height_encoded, &p_invert_encoded, &p_in_3d_space_encoded, &p_skirt_encoded, &p_normalize_encoded));
}

TypedArray<Ref<Image>> Noise::get_image_3d(int32_t p_width, int32_t p_height, int32_t p_depth, bool p_invert, bool p_normalize) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Noise::get_class_static()._native_ptr(), StringName("get_image_3d")._native_ptr(), 3977814329);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Ref<Image>>()));
	int64_t p_width_encoded;
	PtrToArg<int64_t>::encode(p_width, &p_width_encoded);
	int64_t p_height_encoded;
	PtrToArg<int64_t>::encode(p_height, &p_height_encoded);
	int64_t p_depth_encoded;
	PtrToArg<int64_t>::encode(p_depth, &p_depth_encoded);
	int8_t p_invert_encoded;
	PtrToArg<bool>::encode(p_invert, &p_invert_encoded);
	int8_t p_normalize_encoded;
	PtrToArg<bool>::encode(p_normalize, &p_normalize_encoded);
	return ::godot::internal::_call_native_mb_ret<TypedArray<Ref<Image>>>(_gde_method_bind, _owner, &p_width_encoded, &p_height_encoded, &p_depth_encoded, &p_invert_encoded, &p_normalize_encoded);
}

TypedArray<Ref<Image>> Noise::get_seamless_image_3d(int32_t p_width, int32_t p_height, int32_t p_depth, bool p_invert, float p_skirt, bool p_normalize) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Noise::get_class_static()._native_ptr(), StringName("get_seamless_image_3d")._native_ptr(), 451006340);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Ref<Image>>()));
	int64_t p_width_encoded;
	PtrToArg<int64_t>::encode(p_width, &p_width_encoded);
	int64_t p_height_encoded;
	PtrToArg<int64_t>::encode(p_height, &p_height_encoded);
	int64_t p_depth_encoded;
	PtrToArg<int64_t>::encode(p_depth, &p_depth_encoded);
	int8_t p_invert_encoded;
	PtrToArg<bool>::encode(p_invert, &p_invert_encoded);
	double p_skirt_encoded;
	PtrToArg<double>::encode(p_skirt, &p_skirt_encoded);
	int8_t p_normalize_encoded;
	PtrToArg<bool>::encode(p_normalize, &p_normalize_encoded);
	return ::godot::internal::_call_native_mb_ret<TypedArray<Ref<Image>>>(_gde_method_bind, _owner, &p_width_encoded, &p_height_encoded, &p_depth_encoded, &p_invert_encoded, &p_skirt_encoded, &p_normalize_encoded);
}

} // namespace godot
