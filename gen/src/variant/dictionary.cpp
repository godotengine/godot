/**************************************************************************/
/*  dictionary.cpp                                                        */
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

#include <godot_cpp/variant/dictionary.hpp>

#include <godot_cpp/core/binder_common.hpp>

#include <godot_cpp/godot.hpp>

#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/builtin_ptrcall.hpp>

#include <utility>

namespace godot {

Dictionary::_MethodBindings Dictionary::_method_bindings;

void Dictionary::_init_bindings_constructors_destructor() {
	_method_bindings.from_variant_constructor = ::godot::gdextension_interface::get_variant_to_type_constructor(GDEXTENSION_VARIANT_TYPE_DICTIONARY);
	_method_bindings.constructor_0 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_DICTIONARY, 0);
	_method_bindings.constructor_1 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_DICTIONARY, 1);
	_method_bindings.constructor_2 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_DICTIONARY, 2);
	_method_bindings.destructor = ::godot::gdextension_interface::variant_get_ptr_destructor(GDEXTENSION_VARIANT_TYPE_DICTIONARY);
}
void Dictionary::init_bindings() {
	Dictionary::_init_bindings_constructors_destructor();
	StringName _gde_name;
	_gde_name = StringName("size");
	_method_bindings.method_size = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 3173160232);
	_gde_name = StringName("is_empty");
	_method_bindings.method_is_empty = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 3918633141);
	_gde_name = StringName("clear");
	_method_bindings.method_clear = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 3218959716);
	_gde_name = StringName("assign");
	_method_bindings.method_assign = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 3642266950);
	_gde_name = StringName("sort");
	_method_bindings.method_sort = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 3218959716);
	_gde_name = StringName("merge");
	_method_bindings.method_merge = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 2079548978);
	_gde_name = StringName("merged");
	_method_bindings.method_merged = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 2271165639);
	_gde_name = StringName("has");
	_method_bindings.method_has = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 3680194679);
	_gde_name = StringName("has_all");
	_method_bindings.method_has_all = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 2988181878);
	_gde_name = StringName("find_key");
	_method_bindings.method_find_key = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 1988825835);
	_gde_name = StringName("erase");
	_method_bindings.method_erase = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 1776646889);
	_gde_name = StringName("hash");
	_method_bindings.method_hash = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 3173160232);
	_gde_name = StringName("keys");
	_method_bindings.method_keys = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 4144163970);
	_gde_name = StringName("values");
	_method_bindings.method_values = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 4144163970);
	_gde_name = StringName("duplicate");
	_method_bindings.method_duplicate = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 830099069);
	_gde_name = StringName("duplicate_deep");
	_method_bindings.method_duplicate_deep = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 2160600714);
	_gde_name = StringName("get");
	_method_bindings.method_get = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 2205440559);
	_gde_name = StringName("get_or_add");
	_method_bindings.method_get_or_add = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 1052551076);
	_gde_name = StringName("set");
	_method_bindings.method_set = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 2175348267);
	_gde_name = StringName("is_typed");
	_method_bindings.method_is_typed = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 3918633141);
	_gde_name = StringName("is_typed_key");
	_method_bindings.method_is_typed_key = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 3918633141);
	_gde_name = StringName("is_typed_value");
	_method_bindings.method_is_typed_value = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 3918633141);
	_gde_name = StringName("is_same_typed");
	_method_bindings.method_is_same_typed = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 3471775634);
	_gde_name = StringName("is_same_typed_key");
	_method_bindings.method_is_same_typed_key = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 3471775634);
	_gde_name = StringName("is_same_typed_value");
	_method_bindings.method_is_same_typed_value = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 3471775634);
	_gde_name = StringName("get_typed_key_builtin");
	_method_bindings.method_get_typed_key_builtin = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 3173160232);
	_gde_name = StringName("get_typed_value_builtin");
	_method_bindings.method_get_typed_value_builtin = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 3173160232);
	_gde_name = StringName("get_typed_key_class_name");
	_method_bindings.method_get_typed_key_class_name = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 1825232092);
	_gde_name = StringName("get_typed_value_class_name");
	_method_bindings.method_get_typed_value_class_name = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 1825232092);
	_gde_name = StringName("get_typed_key_script");
	_method_bindings.method_get_typed_key_script = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 1460142086);
	_gde_name = StringName("get_typed_value_script");
	_method_bindings.method_get_typed_value_script = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 1460142086);
	_gde_name = StringName("make_read_only");
	_method_bindings.method_make_read_only = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 3218959716);
	_gde_name = StringName("is_read_only");
	_method_bindings.method_is_read_only = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 3918633141);
	_gde_name = StringName("recursive_equal");
	_method_bindings.method_recursive_equal = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_DICTIONARY, _gde_name._native_ptr(), 1404404751);
	_method_bindings.indexed_setter = ::godot::gdextension_interface::variant_get_ptr_indexed_setter(GDEXTENSION_VARIANT_TYPE_DICTIONARY);
	_method_bindings.indexed_getter = ::godot::gdextension_interface::variant_get_ptr_indexed_getter(GDEXTENSION_VARIANT_TYPE_DICTIONARY);
	_method_bindings.keyed_setter = ::godot::gdextension_interface::variant_get_ptr_keyed_setter(GDEXTENSION_VARIANT_TYPE_DICTIONARY);
	_method_bindings.keyed_getter = ::godot::gdextension_interface::variant_get_ptr_keyed_getter(GDEXTENSION_VARIANT_TYPE_DICTIONARY);
	_method_bindings.keyed_checker = ::godot::gdextension_interface::variant_get_ptr_keyed_checker(GDEXTENSION_VARIANT_TYPE_DICTIONARY);
	_method_bindings.operator_equal_Variant = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_EQUAL, GDEXTENSION_VARIANT_TYPE_DICTIONARY, GDEXTENSION_VARIANT_TYPE_NIL);
	_method_bindings.operator_not_equal_Variant = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_NOT_EQUAL, GDEXTENSION_VARIANT_TYPE_DICTIONARY, GDEXTENSION_VARIANT_TYPE_NIL);
	_method_bindings.operator_not = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_NOT, GDEXTENSION_VARIANT_TYPE_DICTIONARY, GDEXTENSION_VARIANT_TYPE_NIL);
	_method_bindings.operator_equal_Dictionary = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_EQUAL, GDEXTENSION_VARIANT_TYPE_DICTIONARY, GDEXTENSION_VARIANT_TYPE_DICTIONARY);
	_method_bindings.operator_not_equal_Dictionary = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_NOT_EQUAL, GDEXTENSION_VARIANT_TYPE_DICTIONARY, GDEXTENSION_VARIANT_TYPE_DICTIONARY);
	_method_bindings.operator_in_Dictionary = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_IN, GDEXTENSION_VARIANT_TYPE_DICTIONARY, GDEXTENSION_VARIANT_TYPE_DICTIONARY);
	_method_bindings.operator_in_Array = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_IN, GDEXTENSION_VARIANT_TYPE_DICTIONARY, GDEXTENSION_VARIANT_TYPE_ARRAY);
}

Dictionary::Dictionary(const Variant *p_variant) {
	_method_bindings.from_variant_constructor(&opaque, p_variant->_native_ptr());
}

Dictionary::Dictionary() {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_0, &opaque);
}

Dictionary::Dictionary(const Dictionary &p_from) {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_1, &opaque, &p_from);
}

Dictionary::Dictionary(const Dictionary &p_base, int64_t p_key_type, const StringName &p_key_class_name, const Variant &p_key_script, int64_t p_value_type, const StringName &p_value_class_name, const Variant &p_value_script) {
	int64_t p_key_type_encoded;
	PtrToArg<int64_t>::encode(p_key_type, &p_key_type_encoded);
	int64_t p_value_type_encoded;
	PtrToArg<int64_t>::encode(p_value_type, &p_value_type_encoded);
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_2, &opaque, &p_base, &p_key_type_encoded, &p_key_class_name, &p_key_script, &p_value_type_encoded, &p_value_class_name, &p_value_script);
}

Dictionary::Dictionary(Dictionary &&p_other) {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_1, &opaque, &p_other);
}

Dictionary::~Dictionary() {
	_method_bindings.destructor(&opaque);
}

int64_t Dictionary::size() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_size, (GDExtensionTypePtr)&opaque);
}

bool Dictionary::is_empty() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_empty, (GDExtensionTypePtr)&opaque);
}

void Dictionary::clear() {
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_clear, (GDExtensionTypePtr)&opaque);
}

void Dictionary::assign(const Dictionary &p_dictionary) {
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_assign, (GDExtensionTypePtr)&opaque, &p_dictionary);
}

void Dictionary::sort() {
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_sort, (GDExtensionTypePtr)&opaque);
}

void Dictionary::merge(const Dictionary &p_dictionary, bool p_overwrite) {
	int8_t p_overwrite_encoded;
	PtrToArg<bool>::encode(p_overwrite, &p_overwrite_encoded);
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_merge, (GDExtensionTypePtr)&opaque, &p_dictionary, &p_overwrite_encoded);
}

Dictionary Dictionary::merged(const Dictionary &p_dictionary, bool p_overwrite) const {
	int8_t p_overwrite_encoded;
	PtrToArg<bool>::encode(p_overwrite, &p_overwrite_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<Dictionary>(_method_bindings.method_merged, (GDExtensionTypePtr)&opaque, &p_dictionary, &p_overwrite_encoded);
}

bool Dictionary::has(const Variant &p_key) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_has, (GDExtensionTypePtr)&opaque, &p_key);
}

bool Dictionary::has_all(const Array &p_keys) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_has_all, (GDExtensionTypePtr)&opaque, &p_keys);
}

Variant Dictionary::find_key(const Variant &p_value) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<Variant>(_method_bindings.method_find_key, (GDExtensionTypePtr)&opaque, &p_value);
}

bool Dictionary::erase(const Variant &p_key) {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_erase, (GDExtensionTypePtr)&opaque, &p_key);
}

int64_t Dictionary::hash() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_hash, (GDExtensionTypePtr)&opaque);
}

Array Dictionary::keys() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<Array>(_method_bindings.method_keys, (GDExtensionTypePtr)&opaque);
}

Array Dictionary::values() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<Array>(_method_bindings.method_values, (GDExtensionTypePtr)&opaque);
}

Dictionary Dictionary::duplicate(bool p_deep) const {
	int8_t p_deep_encoded;
	PtrToArg<bool>::encode(p_deep, &p_deep_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<Dictionary>(_method_bindings.method_duplicate, (GDExtensionTypePtr)&opaque, &p_deep_encoded);
}

Dictionary Dictionary::duplicate_deep(int64_t p_deep_subresources_mode) const {
	int64_t p_deep_subresources_mode_encoded;
	PtrToArg<int64_t>::encode(p_deep_subresources_mode, &p_deep_subresources_mode_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<Dictionary>(_method_bindings.method_duplicate_deep, (GDExtensionTypePtr)&opaque, &p_deep_subresources_mode_encoded);
}

Variant Dictionary::get(const Variant &p_key, const Variant &p_default) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<Variant>(_method_bindings.method_get, (GDExtensionTypePtr)&opaque, &p_key, &p_default);
}

Variant Dictionary::get_or_add(const Variant &p_key, const Variant &p_default) {
	return ::godot::internal::_call_builtin_method_ptr_ret<Variant>(_method_bindings.method_get_or_add, (GDExtensionTypePtr)&opaque, &p_key, &p_default);
}

bool Dictionary::set(const Variant &p_key, const Variant &p_value) {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_set, (GDExtensionTypePtr)&opaque, &p_key, &p_value);
}

bool Dictionary::is_typed() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_typed, (GDExtensionTypePtr)&opaque);
}

bool Dictionary::is_typed_key() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_typed_key, (GDExtensionTypePtr)&opaque);
}

bool Dictionary::is_typed_value() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_typed_value, (GDExtensionTypePtr)&opaque);
}

bool Dictionary::is_same_typed(const Dictionary &p_dictionary) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_same_typed, (GDExtensionTypePtr)&opaque, &p_dictionary);
}

bool Dictionary::is_same_typed_key(const Dictionary &p_dictionary) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_same_typed_key, (GDExtensionTypePtr)&opaque, &p_dictionary);
}

bool Dictionary::is_same_typed_value(const Dictionary &p_dictionary) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_same_typed_value, (GDExtensionTypePtr)&opaque, &p_dictionary);
}

int64_t Dictionary::get_typed_key_builtin() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_get_typed_key_builtin, (GDExtensionTypePtr)&opaque);
}

int64_t Dictionary::get_typed_value_builtin() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_get_typed_value_builtin, (GDExtensionTypePtr)&opaque);
}

StringName Dictionary::get_typed_key_class_name() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<StringName>(_method_bindings.method_get_typed_key_class_name, (GDExtensionTypePtr)&opaque);
}

StringName Dictionary::get_typed_value_class_name() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<StringName>(_method_bindings.method_get_typed_value_class_name, (GDExtensionTypePtr)&opaque);
}

Variant Dictionary::get_typed_key_script() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<Variant>(_method_bindings.method_get_typed_key_script, (GDExtensionTypePtr)&opaque);
}

Variant Dictionary::get_typed_value_script() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<Variant>(_method_bindings.method_get_typed_value_script, (GDExtensionTypePtr)&opaque);
}

void Dictionary::make_read_only() {
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_make_read_only, (GDExtensionTypePtr)&opaque);
}

bool Dictionary::is_read_only() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_read_only, (GDExtensionTypePtr)&opaque);
}

bool Dictionary::recursive_equal(const Dictionary &p_dictionary, int64_t p_recursion_count) const {
	int64_t p_recursion_count_encoded;
	PtrToArg<int64_t>::encode(p_recursion_count, &p_recursion_count_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_recursive_equal, (GDExtensionTypePtr)&opaque, &p_dictionary, &p_recursion_count_encoded);
}

bool Dictionary::operator==(const Variant &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_equal_Variant, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool Dictionary::operator!=(const Variant &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_not_equal_Variant, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool Dictionary::operator!() const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_not, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr) nullptr);
}

bool Dictionary::operator==(const Dictionary &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_equal_Dictionary, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool Dictionary::operator!=(const Dictionary &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_not_equal_Dictionary, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

Dictionary &Dictionary::operator=(const Dictionary &p_other) {
	_method_bindings.destructor(&opaque);
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_1, &opaque, &p_other);
	return *this;
}

Dictionary &Dictionary::operator=(Dictionary &&p_other) {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_1, &opaque, &p_other);
	return *this;
}

} //namespace godot
