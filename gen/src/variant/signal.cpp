/**************************************************************************/
/*  signal.cpp                                                            */
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

#include <godot_cpp/variant/signal.hpp>

#include <godot_cpp/core/binder_common.hpp>

#include <godot_cpp/godot.hpp>

#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/callable.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/builtin_ptrcall.hpp>

#include <utility>

namespace godot {

Signal::_MethodBindings Signal::_method_bindings;

void Signal::_init_bindings_constructors_destructor() {
	_method_bindings.from_variant_constructor = ::godot::gdextension_interface::get_variant_to_type_constructor(GDEXTENSION_VARIANT_TYPE_SIGNAL);
	_method_bindings.constructor_0 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_SIGNAL, 0);
	_method_bindings.constructor_1 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_SIGNAL, 1);
	_method_bindings.constructor_2 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_SIGNAL, 2);
	_method_bindings.destructor = ::godot::gdextension_interface::variant_get_ptr_destructor(GDEXTENSION_VARIANT_TYPE_SIGNAL);
}
void Signal::init_bindings() {
	Signal::_init_bindings_constructors_destructor();
	StringName _gde_name;
	_gde_name = StringName("is_null");
	_method_bindings.method_is_null = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_SIGNAL, _gde_name._native_ptr(), 3918633141);
	_gde_name = StringName("get_object");
	_method_bindings.method_get_object = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_SIGNAL, _gde_name._native_ptr(), 4008621732);
	_gde_name = StringName("get_object_id");
	_method_bindings.method_get_object_id = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_SIGNAL, _gde_name._native_ptr(), 3173160232);
	_gde_name = StringName("get_name");
	_method_bindings.method_get_name = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_SIGNAL, _gde_name._native_ptr(), 1825232092);
	_gde_name = StringName("connect");
	_method_bindings.method_connect = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_SIGNAL, _gde_name._native_ptr(), 979702392);
	_gde_name = StringName("disconnect");
	_method_bindings.method_disconnect = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_SIGNAL, _gde_name._native_ptr(), 3470848906);
	_gde_name = StringName("is_connected");
	_method_bindings.method_is_connected = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_SIGNAL, _gde_name._native_ptr(), 4129521963);
	_gde_name = StringName("get_connections");
	_method_bindings.method_get_connections = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_SIGNAL, _gde_name._native_ptr(), 4144163970);
	_gde_name = StringName("has_connections");
	_method_bindings.method_has_connections = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_SIGNAL, _gde_name._native_ptr(), 3918633141);
	_gde_name = StringName("emit");
	_method_bindings.method_emit = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_SIGNAL, _gde_name._native_ptr(), 3286317445);
	_method_bindings.operator_equal_Variant = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_EQUAL, GDEXTENSION_VARIANT_TYPE_SIGNAL, GDEXTENSION_VARIANT_TYPE_NIL);
	_method_bindings.operator_not_equal_Variant = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_NOT_EQUAL, GDEXTENSION_VARIANT_TYPE_SIGNAL, GDEXTENSION_VARIANT_TYPE_NIL);
	_method_bindings.operator_not = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_NOT, GDEXTENSION_VARIANT_TYPE_SIGNAL, GDEXTENSION_VARIANT_TYPE_NIL);
	_method_bindings.operator_equal_Signal = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_EQUAL, GDEXTENSION_VARIANT_TYPE_SIGNAL, GDEXTENSION_VARIANT_TYPE_SIGNAL);
	_method_bindings.operator_not_equal_Signal = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_NOT_EQUAL, GDEXTENSION_VARIANT_TYPE_SIGNAL, GDEXTENSION_VARIANT_TYPE_SIGNAL);
	_method_bindings.operator_in_Dictionary = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_IN, GDEXTENSION_VARIANT_TYPE_SIGNAL, GDEXTENSION_VARIANT_TYPE_DICTIONARY);
	_method_bindings.operator_in_Array = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_IN, GDEXTENSION_VARIANT_TYPE_SIGNAL, GDEXTENSION_VARIANT_TYPE_ARRAY);
}

Signal::Signal(const Variant *p_variant) {
	_method_bindings.from_variant_constructor(&opaque, p_variant->_native_ptr());
}

Signal::Signal() {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_0, &opaque);
}

Signal::Signal(const Signal &p_from) {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_1, &opaque, &p_from);
}

Signal::Signal(Object *p_object, const StringName &p_signal) {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_2, &opaque, (p_object != nullptr ? &p_object->_owner : nullptr), &p_signal);
}

Signal::Signal(Signal &&p_other) {
	std::swap(opaque, p_other.opaque);
}

Signal::~Signal() {
	_method_bindings.destructor(&opaque);
}

bool Signal::is_null() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_null, (GDExtensionTypePtr)&opaque);
}

Object *Signal::get_object() const {
	return ::godot::internal::_call_builtin_method_ptr_ret_obj<Object>(_method_bindings.method_get_object, (GDExtensionTypePtr)&opaque);
}

int64_t Signal::get_object_id() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_get_object_id, (GDExtensionTypePtr)&opaque);
}

StringName Signal::get_name() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<StringName>(_method_bindings.method_get_name, (GDExtensionTypePtr)&opaque);
}

int64_t Signal::connect(const Callable &p_callable, int64_t p_flags) {
	int64_t p_flags_encoded;
	PtrToArg<int64_t>::encode(p_flags, &p_flags_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_connect, (GDExtensionTypePtr)&opaque, &p_callable, &p_flags_encoded);
}

void Signal::disconnect(const Callable &p_callable) {
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_disconnect, (GDExtensionTypePtr)&opaque, &p_callable);
}

bool Signal::is_connected(const Callable &p_callable) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_connected, (GDExtensionTypePtr)&opaque, &p_callable);
}

Array Signal::get_connections() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<Array>(_method_bindings.method_get_connections, (GDExtensionTypePtr)&opaque);
}

bool Signal::has_connections() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_has_connections, (GDExtensionTypePtr)&opaque);
}

bool Signal::operator==(const Variant &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_equal_Variant, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool Signal::operator!=(const Variant &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_not_equal_Variant, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool Signal::operator!() const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_not, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr) nullptr);
}

bool Signal::operator==(const Signal &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_equal_Signal, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool Signal::operator!=(const Signal &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_not_equal_Signal, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

Signal &Signal::operator=(const Signal &p_other) {
	_method_bindings.destructor(&opaque);
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_1, &opaque, &p_other);
	return *this;
}

Signal &Signal::operator=(Signal &&p_other) {
	std::swap(opaque, p_other.opaque);
	return *this;
}

} //namespace godot
