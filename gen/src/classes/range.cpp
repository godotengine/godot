/**************************************************************************/
/*  range.cpp                                                             */
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

#include <godot_cpp/classes/range.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/node.hpp>

namespace godot {

double Range::get_value() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Range::get_class_static()._native_ptr(), StringName("get_value")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

double Range::get_min() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Range::get_class_static()._native_ptr(), StringName("get_min")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

double Range::get_max() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Range::get_class_static()._native_ptr(), StringName("get_max")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

double Range::get_step() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Range::get_class_static()._native_ptr(), StringName("get_step")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

double Range::get_page() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Range::get_class_static()._native_ptr(), StringName("get_page")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

double Range::get_as_ratio() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Range::get_class_static()._native_ptr(), StringName("get_as_ratio")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Range::set_value(double p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Range::get_class_static()._native_ptr(), StringName("set_value")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_value_encoded);
}

void Range::set_value_no_signal(double p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Range::get_class_static()._native_ptr(), StringName("set_value_no_signal")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_value_encoded);
}

void Range::set_min(double p_minimum) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Range::get_class_static()._native_ptr(), StringName("set_min")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_minimum_encoded;
	PtrToArg<double>::encode(p_minimum, &p_minimum_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_minimum_encoded);
}

void Range::set_max(double p_maximum) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Range::get_class_static()._native_ptr(), StringName("set_max")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_maximum_encoded;
	PtrToArg<double>::encode(p_maximum, &p_maximum_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_maximum_encoded);
}

void Range::set_step(double p_step) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Range::get_class_static()._native_ptr(), StringName("set_step")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_step_encoded;
	PtrToArg<double>::encode(p_step, &p_step_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_step_encoded);
}

void Range::set_page(double p_pagesize) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Range::get_class_static()._native_ptr(), StringName("set_page")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_pagesize_encoded;
	PtrToArg<double>::encode(p_pagesize, &p_pagesize_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_pagesize_encoded);
}

void Range::set_as_ratio(double p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Range::get_class_static()._native_ptr(), StringName("set_as_ratio")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_value_encoded);
}

void Range::set_use_rounded_values(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Range::get_class_static()._native_ptr(), StringName("set_use_rounded_values")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool Range::is_using_rounded_values() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Range::get_class_static()._native_ptr(), StringName("is_using_rounded_values")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Range::set_exp_ratio(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Range::get_class_static()._native_ptr(), StringName("set_exp_ratio")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool Range::is_ratio_exp() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Range::get_class_static()._native_ptr(), StringName("is_ratio_exp")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Range::set_allow_greater(bool p_allow) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Range::get_class_static()._native_ptr(), StringName("set_allow_greater")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_allow_encoded;
	PtrToArg<bool>::encode(p_allow, &p_allow_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_allow_encoded);
}

bool Range::is_greater_allowed() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Range::get_class_static()._native_ptr(), StringName("is_greater_allowed")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Range::set_allow_lesser(bool p_allow) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Range::get_class_static()._native_ptr(), StringName("set_allow_lesser")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_allow_encoded;
	PtrToArg<bool>::encode(p_allow, &p_allow_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_allow_encoded);
}

bool Range::is_lesser_allowed() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Range::get_class_static()._native_ptr(), StringName("is_lesser_allowed")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Range::share(Node *p_with) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Range::get_class_static()._native_ptr(), StringName("share")._native_ptr(), 1078189570);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_with != nullptr ? &p_with->_owner : nullptr));
}

void Range::unshare() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Range::get_class_static()._native_ptr(), StringName("unshare")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Range::_value_changed(double p_new_value) {}

} // namespace godot
