/**************************************************************************/
/*  property_tweener.cpp                                                  */
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

#include <godot_cpp/classes/property_tweener.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/variant/callable.hpp>
#include <godot_cpp/variant/variant.hpp>

namespace godot {

Ref<PropertyTweener> PropertyTweener::from(const Variant &p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PropertyTweener::get_class_static()._native_ptr(), StringName("from")._native_ptr(), 4190193059);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<PropertyTweener>()));
	return Ref<PropertyTweener>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<PropertyTweener>(_gde_method_bind, _owner, &p_value));
}

Ref<PropertyTweener> PropertyTweener::from_current() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PropertyTweener::get_class_static()._native_ptr(), StringName("from_current")._native_ptr(), 4279177709);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<PropertyTweener>()));
	return Ref<PropertyTweener>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<PropertyTweener>(_gde_method_bind, _owner));
}

Ref<PropertyTweener> PropertyTweener::as_relative() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PropertyTweener::get_class_static()._native_ptr(), StringName("as_relative")._native_ptr(), 4279177709);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<PropertyTweener>()));
	return Ref<PropertyTweener>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<PropertyTweener>(_gde_method_bind, _owner));
}

Ref<PropertyTweener> PropertyTweener::set_trans(Tween::TransitionType p_trans) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PropertyTweener::get_class_static()._native_ptr(), StringName("set_trans")._native_ptr(), 1899107404);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<PropertyTweener>()));
	int64_t p_trans_encoded;
	PtrToArg<int64_t>::encode(p_trans, &p_trans_encoded);
	return Ref<PropertyTweener>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<PropertyTweener>(_gde_method_bind, _owner, &p_trans_encoded));
}

Ref<PropertyTweener> PropertyTweener::set_ease(Tween::EaseType p_ease) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PropertyTweener::get_class_static()._native_ptr(), StringName("set_ease")._native_ptr(), 1080455622);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<PropertyTweener>()));
	int64_t p_ease_encoded;
	PtrToArg<int64_t>::encode(p_ease, &p_ease_encoded);
	return Ref<PropertyTweener>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<PropertyTweener>(_gde_method_bind, _owner, &p_ease_encoded));
}

Ref<PropertyTweener> PropertyTweener::set_custom_interpolator(const Callable &p_interpolator_method) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PropertyTweener::get_class_static()._native_ptr(), StringName("set_custom_interpolator")._native_ptr(), 3174170268);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<PropertyTweener>()));
	return Ref<PropertyTweener>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<PropertyTweener>(_gde_method_bind, _owner, &p_interpolator_method));
}

Ref<PropertyTweener> PropertyTweener::set_delay(double p_delay) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PropertyTweener::get_class_static()._native_ptr(), StringName("set_delay")._native_ptr(), 2171559331);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<PropertyTweener>()));
	double p_delay_encoded;
	PtrToArg<double>::encode(p_delay, &p_delay_encoded);
	return Ref<PropertyTweener>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<PropertyTweener>(_gde_method_bind, _owner, &p_delay_encoded));
}

} // namespace godot
