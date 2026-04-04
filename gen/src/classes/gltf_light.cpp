/**************************************************************************/
/*  gltf_light.cpp                                                        */
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

#include <godot_cpp/classes/gltf_light.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/light3d.hpp>
#include <godot_cpp/variant/string_name.hpp>

namespace godot {

Ref<GLTFLight> GLTFLight::from_node(Light3D *p_light_node) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFLight::get_class_static()._native_ptr(), StringName("from_node")._native_ptr(), 3907677874);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<GLTFLight>()));
	return Ref<GLTFLight>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<GLTFLight>(_gde_method_bind, nullptr, (p_light_node != nullptr ? &p_light_node->_owner : nullptr)));
}

Light3D *GLTFLight::to_node() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFLight::get_class_static()._native_ptr(), StringName("to_node")._native_ptr(), 2040811672);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Light3D>(_gde_method_bind, _owner);
}

Ref<GLTFLight> GLTFLight::from_dictionary(const Dictionary &p_dictionary) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFLight::get_class_static()._native_ptr(), StringName("from_dictionary")._native_ptr(), 4057087208);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<GLTFLight>()));
	return Ref<GLTFLight>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<GLTFLight>(_gde_method_bind, nullptr, &p_dictionary));
}

Dictionary GLTFLight::to_dictionary() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFLight::get_class_static()._native_ptr(), StringName("to_dictionary")._native_ptr(), 3102165223);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner);
}

Color GLTFLight::get_color() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFLight::get_class_static()._native_ptr(), StringName("get_color")._native_ptr(), 3200896285);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void GLTFLight::set_color(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFLight::get_class_static()._native_ptr(), StringName("set_color")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

float GLTFLight::get_intensity() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFLight::get_class_static()._native_ptr(), StringName("get_intensity")._native_ptr(), 191475506);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void GLTFLight::set_intensity(float p_intensity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFLight::get_class_static()._native_ptr(), StringName("set_intensity")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_intensity_encoded;
	PtrToArg<double>::encode(p_intensity, &p_intensity_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_intensity_encoded);
}

String GLTFLight::get_light_type() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFLight::get_class_static()._native_ptr(), StringName("get_light_type")._native_ptr(), 2841200299);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void GLTFLight::set_light_type(const String &p_light_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFLight::get_class_static()._native_ptr(), StringName("set_light_type")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light_type);
}

float GLTFLight::get_range() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFLight::get_class_static()._native_ptr(), StringName("get_range")._native_ptr(), 191475506);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void GLTFLight::set_range(float p_range) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFLight::get_class_static()._native_ptr(), StringName("set_range")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_range_encoded;
	PtrToArg<double>::encode(p_range, &p_range_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_range_encoded);
}

float GLTFLight::get_inner_cone_angle() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFLight::get_class_static()._native_ptr(), StringName("get_inner_cone_angle")._native_ptr(), 191475506);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void GLTFLight::set_inner_cone_angle(float p_inner_cone_angle) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFLight::get_class_static()._native_ptr(), StringName("set_inner_cone_angle")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_inner_cone_angle_encoded;
	PtrToArg<double>::encode(p_inner_cone_angle, &p_inner_cone_angle_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_inner_cone_angle_encoded);
}

float GLTFLight::get_outer_cone_angle() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFLight::get_class_static()._native_ptr(), StringName("get_outer_cone_angle")._native_ptr(), 191475506);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void GLTFLight::set_outer_cone_angle(float p_outer_cone_angle) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFLight::get_class_static()._native_ptr(), StringName("set_outer_cone_angle")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_outer_cone_angle_encoded;
	PtrToArg<double>::encode(p_outer_cone_angle, &p_outer_cone_angle_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_outer_cone_angle_encoded);
}

Variant GLTFLight::get_additional_data(const StringName &p_extension_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFLight::get_class_static()._native_ptr(), StringName("get_additional_data")._native_ptr(), 2138907829);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_extension_name);
}

void GLTFLight::set_additional_data(const StringName &p_extension_name, const Variant &p_additional_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFLight::get_class_static()._native_ptr(), StringName("set_additional_data")._native_ptr(), 3776071444);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_extension_name, &p_additional_data);
}

} // namespace godot
