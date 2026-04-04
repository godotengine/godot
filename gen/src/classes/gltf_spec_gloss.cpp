/**************************************************************************/
/*  gltf_spec_gloss.cpp                                                   */
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

#include <godot_cpp/classes/gltf_spec_gloss.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/image.hpp>

namespace godot {

Ref<Image> GLTFSpecGloss::get_diffuse_img() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSpecGloss::get_class_static()._native_ptr(), StringName("get_diffuse_img")._native_ptr(), 564927088);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Image>()));
	return Ref<Image>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Image>(_gde_method_bind, _owner));
}

void GLTFSpecGloss::set_diffuse_img(const Ref<Image> &p_diffuse_img) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSpecGloss::get_class_static()._native_ptr(), StringName("set_diffuse_img")._native_ptr(), 532598488);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_diffuse_img != nullptr ? &p_diffuse_img->_owner : nullptr));
}

Color GLTFSpecGloss::get_diffuse_factor() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSpecGloss::get_class_static()._native_ptr(), StringName("get_diffuse_factor")._native_ptr(), 3200896285);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void GLTFSpecGloss::set_diffuse_factor(const Color &p_diffuse_factor) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSpecGloss::get_class_static()._native_ptr(), StringName("set_diffuse_factor")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_diffuse_factor);
}

float GLTFSpecGloss::get_gloss_factor() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSpecGloss::get_class_static()._native_ptr(), StringName("get_gloss_factor")._native_ptr(), 191475506);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void GLTFSpecGloss::set_gloss_factor(float p_gloss_factor) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSpecGloss::get_class_static()._native_ptr(), StringName("set_gloss_factor")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_gloss_factor_encoded;
	PtrToArg<double>::encode(p_gloss_factor, &p_gloss_factor_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_gloss_factor_encoded);
}

Color GLTFSpecGloss::get_specular_factor() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSpecGloss::get_class_static()._native_ptr(), StringName("get_specular_factor")._native_ptr(), 3200896285);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void GLTFSpecGloss::set_specular_factor(const Color &p_specular_factor) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSpecGloss::get_class_static()._native_ptr(), StringName("set_specular_factor")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_specular_factor);
}

Ref<Image> GLTFSpecGloss::get_spec_gloss_img() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSpecGloss::get_class_static()._native_ptr(), StringName("get_spec_gloss_img")._native_ptr(), 564927088);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Image>()));
	return Ref<Image>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Image>(_gde_method_bind, _owner));
}

void GLTFSpecGloss::set_spec_gloss_img(const Ref<Image> &p_spec_gloss_img) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFSpecGloss::get_class_static()._native_ptr(), StringName("set_spec_gloss_img")._native_ptr(), 532598488);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_spec_gloss_img != nullptr ? &p_spec_gloss_img->_owner : nullptr));
}

} // namespace godot
