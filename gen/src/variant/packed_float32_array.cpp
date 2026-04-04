/**************************************************************************/
/*  packed_float32_array.cpp                                              */
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

#include <godot_cpp/variant/packed_float32_array.hpp>

#include <godot_cpp/core/binder_common.hpp>

#include <godot_cpp/godot.hpp>

#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/builtin_ptrcall.hpp>

#include <utility>

namespace godot {

PackedFloat32Array::_MethodBindings PackedFloat32Array::_method_bindings;

void PackedFloat32Array::_init_bindings_constructors_destructor() {
	_method_bindings.from_variant_constructor = ::godot::gdextension_interface::get_variant_to_type_constructor(GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY);
	_method_bindings.constructor_0 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, 0);
	_method_bindings.constructor_1 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, 1);
	_method_bindings.constructor_2 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, 2);
	_method_bindings.destructor = ::godot::gdextension_interface::variant_get_ptr_destructor(GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY);
}
void PackedFloat32Array::init_bindings() {
	PackedFloat32Array::_init_bindings_constructors_destructor();
	StringName _gde_name;
	_gde_name = StringName("get");
	_method_bindings.method_get = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, _gde_name._native_ptr(), 1401583798);
	_gde_name = StringName("set");
	_method_bindings.method_set = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, _gde_name._native_ptr(), 1113000516);
	_gde_name = StringName("size");
	_method_bindings.method_size = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, _gde_name._native_ptr(), 3173160232);
	_gde_name = StringName("is_empty");
	_method_bindings.method_is_empty = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, _gde_name._native_ptr(), 3918633141);
	_gde_name = StringName("push_back");
	_method_bindings.method_push_back = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, _gde_name._native_ptr(), 4094791666);
	_gde_name = StringName("append");
	_method_bindings.method_append = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, _gde_name._native_ptr(), 4094791666);
	_gde_name = StringName("append_array");
	_method_bindings.method_append_array = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, _gde_name._native_ptr(), 2981316639);
	_gde_name = StringName("remove_at");
	_method_bindings.method_remove_at = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, _gde_name._native_ptr(), 2823966027);
	_gde_name = StringName("insert");
	_method_bindings.method_insert = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, _gde_name._native_ptr(), 1379903876);
	_gde_name = StringName("fill");
	_method_bindings.method_fill = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, _gde_name._native_ptr(), 833936903);
	_gde_name = StringName("resize");
	_method_bindings.method_resize = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, _gde_name._native_ptr(), 848867239);
	_gde_name = StringName("clear");
	_method_bindings.method_clear = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, _gde_name._native_ptr(), 3218959716);
	_gde_name = StringName("has");
	_method_bindings.method_has = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, _gde_name._native_ptr(), 1296369134);
	_gde_name = StringName("reverse");
	_method_bindings.method_reverse = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, _gde_name._native_ptr(), 3218959716);
	_gde_name = StringName("slice");
	_method_bindings.method_slice = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, _gde_name._native_ptr(), 1418229160);
	_gde_name = StringName("to_byte_array");
	_method_bindings.method_to_byte_array = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, _gde_name._native_ptr(), 247621236);
	_gde_name = StringName("sort");
	_method_bindings.method_sort = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, _gde_name._native_ptr(), 3218959716);
	_gde_name = StringName("bsearch");
	_method_bindings.method_bsearch = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, _gde_name._native_ptr(), 1175118842);
	_gde_name = StringName("duplicate");
	_method_bindings.method_duplicate = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, _gde_name._native_ptr(), 3575107827);
	_gde_name = StringName("find");
	_method_bindings.method_find = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, _gde_name._native_ptr(), 1343150241);
	_gde_name = StringName("rfind");
	_method_bindings.method_rfind = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, _gde_name._native_ptr(), 1343150241);
	_gde_name = StringName("count");
	_method_bindings.method_count = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, _gde_name._native_ptr(), 2859915090);
	_gde_name = StringName("erase");
	_method_bindings.method_erase = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, _gde_name._native_ptr(), 4094791666);
	_method_bindings.indexed_setter = ::godot::gdextension_interface::variant_get_ptr_indexed_setter(GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY);
	_method_bindings.indexed_getter = ::godot::gdextension_interface::variant_get_ptr_indexed_getter(GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY);
	_method_bindings.operator_equal_Variant = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_EQUAL, GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, GDEXTENSION_VARIANT_TYPE_NIL);
	_method_bindings.operator_not_equal_Variant = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_NOT_EQUAL, GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, GDEXTENSION_VARIANT_TYPE_NIL);
	_method_bindings.operator_not = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_NOT, GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, GDEXTENSION_VARIANT_TYPE_NIL);
	_method_bindings.operator_in_Dictionary = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_IN, GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, GDEXTENSION_VARIANT_TYPE_DICTIONARY);
	_method_bindings.operator_in_Array = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_IN, GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, GDEXTENSION_VARIANT_TYPE_ARRAY);
	_method_bindings.operator_equal_PackedFloat32Array = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_EQUAL, GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY);
	_method_bindings.operator_not_equal_PackedFloat32Array = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_NOT_EQUAL, GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY);
	_method_bindings.operator_add_PackedFloat32Array = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_ADD, GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY, GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY);
}

PackedFloat32Array::PackedFloat32Array(const Variant *p_variant) {
	_method_bindings.from_variant_constructor(&opaque, p_variant->_native_ptr());
}

PackedFloat32Array::PackedFloat32Array() {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_0, &opaque);
}

PackedFloat32Array::PackedFloat32Array(const PackedFloat32Array &p_from) {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_1, &opaque, &p_from);
}

PackedFloat32Array::PackedFloat32Array(const Array &p_from) {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_2, &opaque, &p_from);
}

PackedFloat32Array::PackedFloat32Array(PackedFloat32Array &&p_other) {
	std::swap(opaque, p_other.opaque);
}

PackedFloat32Array::~PackedFloat32Array() {
	_method_bindings.destructor(&opaque);
}

double PackedFloat32Array::get(int64_t p_index) const {
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<double>(_method_bindings.method_get, (GDExtensionTypePtr)&opaque, &p_index_encoded);
}

void PackedFloat32Array::set(int64_t p_index, double p_value) {
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_set, (GDExtensionTypePtr)&opaque, &p_index_encoded, &p_value_encoded);
}

int64_t PackedFloat32Array::size() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_size, (GDExtensionTypePtr)&opaque);
}

bool PackedFloat32Array::is_empty() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_empty, (GDExtensionTypePtr)&opaque);
}

bool PackedFloat32Array::push_back(double p_value) {
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_push_back, (GDExtensionTypePtr)&opaque, &p_value_encoded);
}

bool PackedFloat32Array::append(double p_value) {
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_append, (GDExtensionTypePtr)&opaque, &p_value_encoded);
}

void PackedFloat32Array::append_array(const PackedFloat32Array &p_array) {
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_append_array, (GDExtensionTypePtr)&opaque, &p_array);
}

void PackedFloat32Array::remove_at(int64_t p_index) {
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_remove_at, (GDExtensionTypePtr)&opaque, &p_index_encoded);
}

int64_t PackedFloat32Array::insert(int64_t p_at_index, double p_value) {
	int64_t p_at_index_encoded;
	PtrToArg<int64_t>::encode(p_at_index, &p_at_index_encoded);
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_insert, (GDExtensionTypePtr)&opaque, &p_at_index_encoded, &p_value_encoded);
}

void PackedFloat32Array::fill(double p_value) {
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_fill, (GDExtensionTypePtr)&opaque, &p_value_encoded);
}

int64_t PackedFloat32Array::resize(int64_t p_new_size) {
	int64_t p_new_size_encoded;
	PtrToArg<int64_t>::encode(p_new_size, &p_new_size_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_resize, (GDExtensionTypePtr)&opaque, &p_new_size_encoded);
}

void PackedFloat32Array::clear() {
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_clear, (GDExtensionTypePtr)&opaque);
}

bool PackedFloat32Array::has(double p_value) const {
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_has, (GDExtensionTypePtr)&opaque, &p_value_encoded);
}

void PackedFloat32Array::reverse() {
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_reverse, (GDExtensionTypePtr)&opaque);
}

PackedFloat32Array PackedFloat32Array::slice(int64_t p_begin, int64_t p_end) const {
	int64_t p_begin_encoded;
	PtrToArg<int64_t>::encode(p_begin, &p_begin_encoded);
	int64_t p_end_encoded;
	PtrToArg<int64_t>::encode(p_end, &p_end_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<PackedFloat32Array>(_method_bindings.method_slice, (GDExtensionTypePtr)&opaque, &p_begin_encoded, &p_end_encoded);
}

PackedByteArray PackedFloat32Array::to_byte_array() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<PackedByteArray>(_method_bindings.method_to_byte_array, (GDExtensionTypePtr)&opaque);
}

void PackedFloat32Array::sort() {
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_sort, (GDExtensionTypePtr)&opaque);
}

int64_t PackedFloat32Array::bsearch(double p_value, bool p_before) const {
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	int8_t p_before_encoded;
	PtrToArg<bool>::encode(p_before, &p_before_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_bsearch, (GDExtensionTypePtr)&opaque, &p_value_encoded, &p_before_encoded);
}

PackedFloat32Array PackedFloat32Array::duplicate() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<PackedFloat32Array>(_method_bindings.method_duplicate, (GDExtensionTypePtr)&opaque);
}

int64_t PackedFloat32Array::find(double p_value, int64_t p_from) const {
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	int64_t p_from_encoded;
	PtrToArg<int64_t>::encode(p_from, &p_from_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_find, (GDExtensionTypePtr)&opaque, &p_value_encoded, &p_from_encoded);
}

int64_t PackedFloat32Array::rfind(double p_value, int64_t p_from) const {
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	int64_t p_from_encoded;
	PtrToArg<int64_t>::encode(p_from, &p_from_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_rfind, (GDExtensionTypePtr)&opaque, &p_value_encoded, &p_from_encoded);
}

int64_t PackedFloat32Array::count(double p_value) const {
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_count, (GDExtensionTypePtr)&opaque, &p_value_encoded);
}

bool PackedFloat32Array::erase(double p_value) {
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_erase, (GDExtensionTypePtr)&opaque, &p_value_encoded);
}

bool PackedFloat32Array::operator==(const Variant &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_equal_Variant, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool PackedFloat32Array::operator!=(const Variant &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_not_equal_Variant, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool PackedFloat32Array::operator!() const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_not, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr) nullptr);
}

bool PackedFloat32Array::operator==(const PackedFloat32Array &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_equal_PackedFloat32Array, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool PackedFloat32Array::operator!=(const PackedFloat32Array &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_not_equal_PackedFloat32Array, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

PackedFloat32Array PackedFloat32Array::operator+(const PackedFloat32Array &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<PackedFloat32Array>(_method_bindings.operator_add_PackedFloat32Array, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

PackedFloat32Array &PackedFloat32Array::operator=(const PackedFloat32Array &p_other) {
	_method_bindings.destructor(&opaque);
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_1, &opaque, &p_other);
	return *this;
}

PackedFloat32Array &PackedFloat32Array::operator=(PackedFloat32Array &&p_other) {
	std::swap(opaque, p_other.opaque);
	return *this;
}

} //namespace godot
