/**************************************************************************/
/*  callable.cpp                                                          */
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

#include <godot_cpp/variant/callable.hpp>

#include <godot_cpp/core/binder_common.hpp>

#include <godot_cpp/godot.hpp>

#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/builtin_ptrcall.hpp>

#include <utility>

namespace godot {

Callable::_MethodBindings Callable::_method_bindings;

void Callable::_init_bindings_constructors_destructor() {
	_method_bindings.from_variant_constructor = ::godot::gdextension_interface::get_variant_to_type_constructor(GDEXTENSION_VARIANT_TYPE_CALLABLE);
	_method_bindings.constructor_0 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_CALLABLE, 0);
	_method_bindings.constructor_1 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_CALLABLE, 1);
	_method_bindings.constructor_2 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_CALLABLE, 2);
	_method_bindings.destructor = ::godot::gdextension_interface::variant_get_ptr_destructor(GDEXTENSION_VARIANT_TYPE_CALLABLE);
}
void Callable::init_bindings() {
	Callable::_init_bindings_constructors_destructor();
	StringName _gde_name;
	_gde_name = StringName("create");
	_method_bindings.method_create = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_CALLABLE, _gde_name._native_ptr(), 1709381114);
	_gde_name = StringName("callv");
	_method_bindings.method_callv = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_CALLABLE, _gde_name._native_ptr(), 413578926);
	_gde_name = StringName("is_null");
	_method_bindings.method_is_null = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_CALLABLE, _gde_name._native_ptr(), 3918633141);
	_gde_name = StringName("is_custom");
	_method_bindings.method_is_custom = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_CALLABLE, _gde_name._native_ptr(), 3918633141);
	_gde_name = StringName("is_standard");
	_method_bindings.method_is_standard = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_CALLABLE, _gde_name._native_ptr(), 3918633141);
	_gde_name = StringName("is_valid");
	_method_bindings.method_is_valid = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_CALLABLE, _gde_name._native_ptr(), 3918633141);
	_gde_name = StringName("get_object");
	_method_bindings.method_get_object = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_CALLABLE, _gde_name._native_ptr(), 4008621732);
	_gde_name = StringName("get_object_id");
	_method_bindings.method_get_object_id = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_CALLABLE, _gde_name._native_ptr(), 3173160232);
	_gde_name = StringName("get_method");
	_method_bindings.method_get_method = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_CALLABLE, _gde_name._native_ptr(), 1825232092);
	_gde_name = StringName("get_argument_count");
	_method_bindings.method_get_argument_count = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_CALLABLE, _gde_name._native_ptr(), 3173160232);
	_gde_name = StringName("get_bound_arguments_count");
	_method_bindings.method_get_bound_arguments_count = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_CALLABLE, _gde_name._native_ptr(), 3173160232);
	_gde_name = StringName("get_bound_arguments");
	_method_bindings.method_get_bound_arguments = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_CALLABLE, _gde_name._native_ptr(), 4144163970);
	_gde_name = StringName("get_unbound_arguments_count");
	_method_bindings.method_get_unbound_arguments_count = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_CALLABLE, _gde_name._native_ptr(), 3173160232);
	_gde_name = StringName("hash");
	_method_bindings.method_hash = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_CALLABLE, _gde_name._native_ptr(), 3173160232);
	_gde_name = StringName("bindv");
	_method_bindings.method_bindv = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_CALLABLE, _gde_name._native_ptr(), 3564560322);
	_gde_name = StringName("unbind");
	_method_bindings.method_unbind = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_CALLABLE, _gde_name._native_ptr(), 755001590);
	_gde_name = StringName("call");
	_method_bindings.method_call = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_CALLABLE, _gde_name._native_ptr(), 3643564216);
	_gde_name = StringName("call_deferred");
	_method_bindings.method_call_deferred = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_CALLABLE, _gde_name._native_ptr(), 3286317445);
	_gde_name = StringName("rpc");
	_method_bindings.method_rpc = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_CALLABLE, _gde_name._native_ptr(), 3286317445);
	_gde_name = StringName("rpc_id");
	_method_bindings.method_rpc_id = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_CALLABLE, _gde_name._native_ptr(), 2270047679);
	_gde_name = StringName("bind");
	_method_bindings.method_bind = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_CALLABLE, _gde_name._native_ptr(), 3224143119);
	_method_bindings.operator_equal_Variant = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_EQUAL, GDEXTENSION_VARIANT_TYPE_CALLABLE, GDEXTENSION_VARIANT_TYPE_NIL);
	_method_bindings.operator_not_equal_Variant = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_NOT_EQUAL, GDEXTENSION_VARIANT_TYPE_CALLABLE, GDEXTENSION_VARIANT_TYPE_NIL);
	_method_bindings.operator_not = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_NOT, GDEXTENSION_VARIANT_TYPE_CALLABLE, GDEXTENSION_VARIANT_TYPE_NIL);
	_method_bindings.operator_equal_Callable = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_EQUAL, GDEXTENSION_VARIANT_TYPE_CALLABLE, GDEXTENSION_VARIANT_TYPE_CALLABLE);
	_method_bindings.operator_not_equal_Callable = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_NOT_EQUAL, GDEXTENSION_VARIANT_TYPE_CALLABLE, GDEXTENSION_VARIANT_TYPE_CALLABLE);
	_method_bindings.operator_in_Dictionary = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_IN, GDEXTENSION_VARIANT_TYPE_CALLABLE, GDEXTENSION_VARIANT_TYPE_DICTIONARY);
	_method_bindings.operator_in_Array = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_IN, GDEXTENSION_VARIANT_TYPE_CALLABLE, GDEXTENSION_VARIANT_TYPE_ARRAY);
}

Callable::Callable(const Variant *p_variant) {
	_method_bindings.from_variant_constructor(&opaque, p_variant->_native_ptr());
}

Callable::Callable() {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_0, &opaque);
}

Callable::Callable(const Callable &p_from) {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_1, &opaque, &p_from);
}

Callable::Callable(Object *p_object, const StringName &p_method) {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_2, &opaque, (p_object != nullptr ? &p_object->_owner : nullptr), &p_method);
}

Callable::Callable(Callable &&p_other) {
	std::swap(opaque, p_other.opaque);
}

Callable::~Callable() {
	_method_bindings.destructor(&opaque);
}

Callable Callable::create(const Variant &p_variant, const StringName &p_method) {
	return ::godot::internal::_call_builtin_method_ptr_ret<Callable>(_method_bindings.method_create, nullptr, &p_variant, &p_method);
}

Variant Callable::callv(const Array &p_arguments) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<Variant>(_method_bindings.method_callv, (GDExtensionTypePtr)&opaque, &p_arguments);
}

bool Callable::is_null() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_null, (GDExtensionTypePtr)&opaque);
}

bool Callable::is_custom() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_custom, (GDExtensionTypePtr)&opaque);
}

bool Callable::is_standard() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_standard, (GDExtensionTypePtr)&opaque);
}

bool Callable::is_valid() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_valid, (GDExtensionTypePtr)&opaque);
}

Object *Callable::get_object() const {
	return ::godot::internal::_call_builtin_method_ptr_ret_obj<Object>(_method_bindings.method_get_object, (GDExtensionTypePtr)&opaque);
}

int64_t Callable::get_object_id() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_get_object_id, (GDExtensionTypePtr)&opaque);
}

StringName Callable::get_method() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<StringName>(_method_bindings.method_get_method, (GDExtensionTypePtr)&opaque);
}

int64_t Callable::get_argument_count() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_get_argument_count, (GDExtensionTypePtr)&opaque);
}

int64_t Callable::get_bound_arguments_count() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_get_bound_arguments_count, (GDExtensionTypePtr)&opaque);
}

Array Callable::get_bound_arguments() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<Array>(_method_bindings.method_get_bound_arguments, (GDExtensionTypePtr)&opaque);
}

int64_t Callable::get_unbound_arguments_count() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_get_unbound_arguments_count, (GDExtensionTypePtr)&opaque);
}

int64_t Callable::hash() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_hash, (GDExtensionTypePtr)&opaque);
}

Callable Callable::bindv(const Array &p_arguments) {
	return ::godot::internal::_call_builtin_method_ptr_ret<Callable>(_method_bindings.method_bindv, (GDExtensionTypePtr)&opaque, &p_arguments);
}

Callable Callable::unbind(int64_t p_argcount) const {
	int64_t p_argcount_encoded;
	PtrToArg<int64_t>::encode(p_argcount, &p_argcount_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<Callable>(_method_bindings.method_unbind, (GDExtensionTypePtr)&opaque, &p_argcount_encoded);
}

bool Callable::operator==(const Variant &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_equal_Variant, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool Callable::operator!=(const Variant &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_not_equal_Variant, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool Callable::operator!() const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_not, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr) nullptr);
}

bool Callable::operator==(const Callable &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_equal_Callable, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool Callable::operator!=(const Callable &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_not_equal_Callable, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

Callable &Callable::operator=(const Callable &p_other) {
	_method_bindings.destructor(&opaque);
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_1, &opaque, &p_other);
	return *this;
}

Callable &Callable::operator=(Callable &&p_other) {
	std::swap(opaque, p_other.opaque);
	return *this;
}

} //namespace godot
