/**************************************************************************/
/*  array.cpp                                                             */
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

#include <godot_cpp/variant/array.hpp>

#include <godot_cpp/core/binder_common.hpp>

#include <godot_cpp/godot.hpp>

#include <godot_cpp/variant/callable.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/packed_color_array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_float64_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/packed_int64_array.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/packed_vector4_array.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/builtin_ptrcall.hpp>

#include <utility>

namespace godot {

Array::_MethodBindings Array::_method_bindings;

void Array::_init_bindings_constructors_destructor() {
	_method_bindings.from_variant_constructor = ::godot::gdextension_interface::get_variant_to_type_constructor(GDEXTENSION_VARIANT_TYPE_ARRAY);
	_method_bindings.constructor_0 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_ARRAY, 0);
	_method_bindings.constructor_1 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_ARRAY, 1);
	_method_bindings.constructor_2 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_ARRAY, 2);
	_method_bindings.constructor_3 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_ARRAY, 3);
	_method_bindings.constructor_4 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_ARRAY, 4);
	_method_bindings.constructor_5 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_ARRAY, 5);
	_method_bindings.constructor_6 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_ARRAY, 6);
	_method_bindings.constructor_7 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_ARRAY, 7);
	_method_bindings.constructor_8 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_ARRAY, 8);
	_method_bindings.constructor_9 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_ARRAY, 9);
	_method_bindings.constructor_10 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_ARRAY, 10);
	_method_bindings.constructor_11 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_ARRAY, 11);
	_method_bindings.constructor_12 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_ARRAY, 12);
	_method_bindings.destructor = ::godot::gdextension_interface::variant_get_ptr_destructor(GDEXTENSION_VARIANT_TYPE_ARRAY);
}
void Array::init_bindings() {
	Array::_init_bindings_constructors_destructor();
	StringName _gde_name;
	_gde_name = StringName("size");
	_method_bindings.method_size = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 3173160232);
	_gde_name = StringName("is_empty");
	_method_bindings.method_is_empty = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 3918633141);
	_gde_name = StringName("clear");
	_method_bindings.method_clear = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 3218959716);
	_gde_name = StringName("hash");
	_method_bindings.method_hash = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 3173160232);
	_gde_name = StringName("assign");
	_method_bindings.method_assign = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 2307260970);
	_gde_name = StringName("get");
	_method_bindings.method_get = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 708700221);
	_gde_name = StringName("set");
	_method_bindings.method_set = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 3798478031);
	_gde_name = StringName("push_back");
	_method_bindings.method_push_back = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 3316032543);
	_gde_name = StringName("push_front");
	_method_bindings.method_push_front = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 3316032543);
	_gde_name = StringName("append");
	_method_bindings.method_append = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 3316032543);
	_gde_name = StringName("append_array");
	_method_bindings.method_append_array = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 2307260970);
	_gde_name = StringName("resize");
	_method_bindings.method_resize = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 848867239);
	_gde_name = StringName("insert");
	_method_bindings.method_insert = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 3176316662);
	_gde_name = StringName("remove_at");
	_method_bindings.method_remove_at = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 2823966027);
	_gde_name = StringName("fill");
	_method_bindings.method_fill = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 3316032543);
	_gde_name = StringName("erase");
	_method_bindings.method_erase = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 3316032543);
	_gde_name = StringName("front");
	_method_bindings.method_front = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 1460142086);
	_gde_name = StringName("back");
	_method_bindings.method_back = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 1460142086);
	_gde_name = StringName("pick_random");
	_method_bindings.method_pick_random = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 1460142086);
	_gde_name = StringName("find");
	_method_bindings.method_find = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 2336346817);
	_gde_name = StringName("find_custom");
	_method_bindings.method_find_custom = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 2145562546);
	_gde_name = StringName("rfind");
	_method_bindings.method_rfind = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 2336346817);
	_gde_name = StringName("rfind_custom");
	_method_bindings.method_rfind_custom = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 2145562546);
	_gde_name = StringName("count");
	_method_bindings.method_count = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 1481661226);
	_gde_name = StringName("has");
	_method_bindings.method_has = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 3680194679);
	_gde_name = StringName("pop_back");
	_method_bindings.method_pop_back = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 1321915136);
	_gde_name = StringName("pop_front");
	_method_bindings.method_pop_front = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 1321915136);
	_gde_name = StringName("pop_at");
	_method_bindings.method_pop_at = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 3518259424);
	_gde_name = StringName("sort");
	_method_bindings.method_sort = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 3218959716);
	_gde_name = StringName("sort_custom");
	_method_bindings.method_sort_custom = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 3470848906);
	_gde_name = StringName("shuffle");
	_method_bindings.method_shuffle = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 3218959716);
	_gde_name = StringName("bsearch");
	_method_bindings.method_bsearch = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 3372222236);
	_gde_name = StringName("bsearch_custom");
	_method_bindings.method_bsearch_custom = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 161317131);
	_gde_name = StringName("reverse");
	_method_bindings.method_reverse = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 3218959716);
	_gde_name = StringName("duplicate");
	_method_bindings.method_duplicate = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 636440122);
	_gde_name = StringName("duplicate_deep");
	_method_bindings.method_duplicate_deep = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 1949240801);
	_gde_name = StringName("slice");
	_method_bindings.method_slice = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 1393718243);
	_gde_name = StringName("filter");
	_method_bindings.method_filter = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 4075186556);
	_gde_name = StringName("map");
	_method_bindings.method_map = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 4075186556);
	_gde_name = StringName("reduce");
	_method_bindings.method_reduce = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 4272450342);
	_gde_name = StringName("any");
	_method_bindings.method_any = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 4129521963);
	_gde_name = StringName("all");
	_method_bindings.method_all = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 4129521963);
	_gde_name = StringName("max");
	_method_bindings.method_max = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 1460142086);
	_gde_name = StringName("min");
	_method_bindings.method_min = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 1460142086);
	_gde_name = StringName("is_typed");
	_method_bindings.method_is_typed = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 3918633141);
	_gde_name = StringName("is_same_typed");
	_method_bindings.method_is_same_typed = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 2988181878);
	_gde_name = StringName("get_typed_builtin");
	_method_bindings.method_get_typed_builtin = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 3173160232);
	_gde_name = StringName("get_typed_class_name");
	_method_bindings.method_get_typed_class_name = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 1825232092);
	_gde_name = StringName("get_typed_script");
	_method_bindings.method_get_typed_script = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 1460142086);
	_gde_name = StringName("make_read_only");
	_method_bindings.method_make_read_only = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 3218959716);
	_gde_name = StringName("is_read_only");
	_method_bindings.method_is_read_only = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_ARRAY, _gde_name._native_ptr(), 3918633141);
	_method_bindings.indexed_setter = ::godot::gdextension_interface::variant_get_ptr_indexed_setter(GDEXTENSION_VARIANT_TYPE_ARRAY);
	_method_bindings.indexed_getter = ::godot::gdextension_interface::variant_get_ptr_indexed_getter(GDEXTENSION_VARIANT_TYPE_ARRAY);
	_method_bindings.operator_equal_Variant = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_EQUAL, GDEXTENSION_VARIANT_TYPE_ARRAY, GDEXTENSION_VARIANT_TYPE_NIL);
	_method_bindings.operator_not_equal_Variant = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_NOT_EQUAL, GDEXTENSION_VARIANT_TYPE_ARRAY, GDEXTENSION_VARIANT_TYPE_NIL);
	_method_bindings.operator_not = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_NOT, GDEXTENSION_VARIANT_TYPE_ARRAY, GDEXTENSION_VARIANT_TYPE_NIL);
	_method_bindings.operator_in_Dictionary = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_IN, GDEXTENSION_VARIANT_TYPE_ARRAY, GDEXTENSION_VARIANT_TYPE_DICTIONARY);
	_method_bindings.operator_equal_Array = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_EQUAL, GDEXTENSION_VARIANT_TYPE_ARRAY, GDEXTENSION_VARIANT_TYPE_ARRAY);
	_method_bindings.operator_not_equal_Array = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_NOT_EQUAL, GDEXTENSION_VARIANT_TYPE_ARRAY, GDEXTENSION_VARIANT_TYPE_ARRAY);
	_method_bindings.operator_less_Array = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_LESS, GDEXTENSION_VARIANT_TYPE_ARRAY, GDEXTENSION_VARIANT_TYPE_ARRAY);
	_method_bindings.operator_less_equal_Array = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_LESS_EQUAL, GDEXTENSION_VARIANT_TYPE_ARRAY, GDEXTENSION_VARIANT_TYPE_ARRAY);
	_method_bindings.operator_greater_Array = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_GREATER, GDEXTENSION_VARIANT_TYPE_ARRAY, GDEXTENSION_VARIANT_TYPE_ARRAY);
	_method_bindings.operator_greater_equal_Array = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_GREATER_EQUAL, GDEXTENSION_VARIANT_TYPE_ARRAY, GDEXTENSION_VARIANT_TYPE_ARRAY);
	_method_bindings.operator_add_Array = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_ADD, GDEXTENSION_VARIANT_TYPE_ARRAY, GDEXTENSION_VARIANT_TYPE_ARRAY);
	_method_bindings.operator_in_Array = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_IN, GDEXTENSION_VARIANT_TYPE_ARRAY, GDEXTENSION_VARIANT_TYPE_ARRAY);
}

Array::Array(const Variant *p_variant) {
	_method_bindings.from_variant_constructor(&opaque, p_variant->_native_ptr());
}

Array::Array() {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_0, &opaque);
}

Array::Array(const Array &p_from) {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_1, &opaque, &p_from);
}

Array::Array(const Array &p_base, int64_t p_type, const StringName &p_class_name, const Variant &p_script) {
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_2, &opaque, &p_base, &p_type_encoded, &p_class_name, &p_script);
}

Array::Array(const PackedByteArray &p_from) {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_3, &opaque, &p_from);
}

Array::Array(const PackedInt32Array &p_from) {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_4, &opaque, &p_from);
}

Array::Array(const PackedInt64Array &p_from) {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_5, &opaque, &p_from);
}

Array::Array(const PackedFloat32Array &p_from) {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_6, &opaque, &p_from);
}

Array::Array(const PackedFloat64Array &p_from) {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_7, &opaque, &p_from);
}

Array::Array(const PackedStringArray &p_from) {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_8, &opaque, &p_from);
}

Array::Array(const PackedVector2Array &p_from) {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_9, &opaque, &p_from);
}

Array::Array(const PackedVector3Array &p_from) {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_10, &opaque, &p_from);
}

Array::Array(const PackedColorArray &p_from) {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_11, &opaque, &p_from);
}

Array::Array(const PackedVector4Array &p_from) {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_12, &opaque, &p_from);
}

Array::Array(Array &&p_other) {
	std::swap(opaque, p_other.opaque);
}

Array::~Array() {
	_method_bindings.destructor(&opaque);
}

int64_t Array::size() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_size, (GDExtensionTypePtr)&opaque);
}

bool Array::is_empty() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_empty, (GDExtensionTypePtr)&opaque);
}

void Array::clear() {
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_clear, (GDExtensionTypePtr)&opaque);
}

int64_t Array::hash() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_hash, (GDExtensionTypePtr)&opaque);
}

void Array::assign(const Array &p_array) {
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_assign, (GDExtensionTypePtr)&opaque, &p_array);
}

Variant Array::get(int64_t p_index) const {
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<Variant>(_method_bindings.method_get, (GDExtensionTypePtr)&opaque, &p_index_encoded);
}

void Array::set(int64_t p_index, const Variant &p_value) {
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_set, (GDExtensionTypePtr)&opaque, &p_index_encoded, &p_value);
}

void Array::push_back(const Variant &p_value) {
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_push_back, (GDExtensionTypePtr)&opaque, &p_value);
}

void Array::push_front(const Variant &p_value) {
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_push_front, (GDExtensionTypePtr)&opaque, &p_value);
}

void Array::append(const Variant &p_value) {
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_append, (GDExtensionTypePtr)&opaque, &p_value);
}

void Array::append_array(const Array &p_array) {
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_append_array, (GDExtensionTypePtr)&opaque, &p_array);
}

int64_t Array::resize(int64_t p_size) {
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_resize, (GDExtensionTypePtr)&opaque, &p_size_encoded);
}

int64_t Array::insert(int64_t p_position, const Variant &p_value) {
	int64_t p_position_encoded;
	PtrToArg<int64_t>::encode(p_position, &p_position_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_insert, (GDExtensionTypePtr)&opaque, &p_position_encoded, &p_value);
}

void Array::remove_at(int64_t p_position) {
	int64_t p_position_encoded;
	PtrToArg<int64_t>::encode(p_position, &p_position_encoded);
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_remove_at, (GDExtensionTypePtr)&opaque, &p_position_encoded);
}

void Array::fill(const Variant &p_value) {
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_fill, (GDExtensionTypePtr)&opaque, &p_value);
}

void Array::erase(const Variant &p_value) {
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_erase, (GDExtensionTypePtr)&opaque, &p_value);
}

Variant Array::front() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<Variant>(_method_bindings.method_front, (GDExtensionTypePtr)&opaque);
}

Variant Array::back() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<Variant>(_method_bindings.method_back, (GDExtensionTypePtr)&opaque);
}

Variant Array::pick_random() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<Variant>(_method_bindings.method_pick_random, (GDExtensionTypePtr)&opaque);
}

int64_t Array::find(const Variant &p_what, int64_t p_from) const {
	int64_t p_from_encoded;
	PtrToArg<int64_t>::encode(p_from, &p_from_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_find, (GDExtensionTypePtr)&opaque, &p_what, &p_from_encoded);
}

int64_t Array::find_custom(const Callable &p_method, int64_t p_from) const {
	int64_t p_from_encoded;
	PtrToArg<int64_t>::encode(p_from, &p_from_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_find_custom, (GDExtensionTypePtr)&opaque, &p_method, &p_from_encoded);
}

int64_t Array::rfind(const Variant &p_what, int64_t p_from) const {
	int64_t p_from_encoded;
	PtrToArg<int64_t>::encode(p_from, &p_from_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_rfind, (GDExtensionTypePtr)&opaque, &p_what, &p_from_encoded);
}

int64_t Array::rfind_custom(const Callable &p_method, int64_t p_from) const {
	int64_t p_from_encoded;
	PtrToArg<int64_t>::encode(p_from, &p_from_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_rfind_custom, (GDExtensionTypePtr)&opaque, &p_method, &p_from_encoded);
}

int64_t Array::count(const Variant &p_value) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_count, (GDExtensionTypePtr)&opaque, &p_value);
}

bool Array::has(const Variant &p_value) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_has, (GDExtensionTypePtr)&opaque, &p_value);
}

Variant Array::pop_back() {
	return ::godot::internal::_call_builtin_method_ptr_ret<Variant>(_method_bindings.method_pop_back, (GDExtensionTypePtr)&opaque);
}

Variant Array::pop_front() {
	return ::godot::internal::_call_builtin_method_ptr_ret<Variant>(_method_bindings.method_pop_front, (GDExtensionTypePtr)&opaque);
}

Variant Array::pop_at(int64_t p_position) {
	int64_t p_position_encoded;
	PtrToArg<int64_t>::encode(p_position, &p_position_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<Variant>(_method_bindings.method_pop_at, (GDExtensionTypePtr)&opaque, &p_position_encoded);
}

void Array::sort() {
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_sort, (GDExtensionTypePtr)&opaque);
}

void Array::sort_custom(const Callable &p_func) {
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_sort_custom, (GDExtensionTypePtr)&opaque, &p_func);
}

void Array::shuffle() {
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_shuffle, (GDExtensionTypePtr)&opaque);
}

int64_t Array::bsearch(const Variant &p_value, bool p_before) const {
	int8_t p_before_encoded;
	PtrToArg<bool>::encode(p_before, &p_before_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_bsearch, (GDExtensionTypePtr)&opaque, &p_value, &p_before_encoded);
}

int64_t Array::bsearch_custom(const Variant &p_value, const Callable &p_func, bool p_before) const {
	int8_t p_before_encoded;
	PtrToArg<bool>::encode(p_before, &p_before_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_bsearch_custom, (GDExtensionTypePtr)&opaque, &p_value, &p_func, &p_before_encoded);
}

void Array::reverse() {
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_reverse, (GDExtensionTypePtr)&opaque);
}

Array Array::duplicate(bool p_deep) const {
	int8_t p_deep_encoded;
	PtrToArg<bool>::encode(p_deep, &p_deep_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<Array>(_method_bindings.method_duplicate, (GDExtensionTypePtr)&opaque, &p_deep_encoded);
}

Array Array::duplicate_deep(int64_t p_deep_subresources_mode) const {
	int64_t p_deep_subresources_mode_encoded;
	PtrToArg<int64_t>::encode(p_deep_subresources_mode, &p_deep_subresources_mode_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<Array>(_method_bindings.method_duplicate_deep, (GDExtensionTypePtr)&opaque, &p_deep_subresources_mode_encoded);
}

Array Array::slice(int64_t p_begin, int64_t p_end, int64_t p_step, bool p_deep) const {
	int64_t p_begin_encoded;
	PtrToArg<int64_t>::encode(p_begin, &p_begin_encoded);
	int64_t p_end_encoded;
	PtrToArg<int64_t>::encode(p_end, &p_end_encoded);
	int64_t p_step_encoded;
	PtrToArg<int64_t>::encode(p_step, &p_step_encoded);
	int8_t p_deep_encoded;
	PtrToArg<bool>::encode(p_deep, &p_deep_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<Array>(_method_bindings.method_slice, (GDExtensionTypePtr)&opaque, &p_begin_encoded, &p_end_encoded, &p_step_encoded, &p_deep_encoded);
}

Array Array::filter(const Callable &p_method) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<Array>(_method_bindings.method_filter, (GDExtensionTypePtr)&opaque, &p_method);
}

Array Array::map(const Callable &p_method) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<Array>(_method_bindings.method_map, (GDExtensionTypePtr)&opaque, &p_method);
}

Variant Array::reduce(const Callable &p_method, const Variant &p_accum) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<Variant>(_method_bindings.method_reduce, (GDExtensionTypePtr)&opaque, &p_method, &p_accum);
}

bool Array::any(const Callable &p_method) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_any, (GDExtensionTypePtr)&opaque, &p_method);
}

bool Array::all(const Callable &p_method) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_all, (GDExtensionTypePtr)&opaque, &p_method);
}

Variant Array::max() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<Variant>(_method_bindings.method_max, (GDExtensionTypePtr)&opaque);
}

Variant Array::min() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<Variant>(_method_bindings.method_min, (GDExtensionTypePtr)&opaque);
}

bool Array::is_typed() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_typed, (GDExtensionTypePtr)&opaque);
}

bool Array::is_same_typed(const Array &p_array) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_same_typed, (GDExtensionTypePtr)&opaque, &p_array);
}

int64_t Array::get_typed_builtin() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_get_typed_builtin, (GDExtensionTypePtr)&opaque);
}

StringName Array::get_typed_class_name() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<StringName>(_method_bindings.method_get_typed_class_name, (GDExtensionTypePtr)&opaque);
}

Variant Array::get_typed_script() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<Variant>(_method_bindings.method_get_typed_script, (GDExtensionTypePtr)&opaque);
}

void Array::make_read_only() {
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_make_read_only, (GDExtensionTypePtr)&opaque);
}

bool Array::is_read_only() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_read_only, (GDExtensionTypePtr)&opaque);
}

bool Array::operator==(const Variant &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_equal_Variant, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool Array::operator!=(const Variant &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_not_equal_Variant, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool Array::operator!() const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_not, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr) nullptr);
}

bool Array::operator==(const Array &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_equal_Array, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool Array::operator!=(const Array &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_not_equal_Array, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool Array::operator<(const Array &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_less_Array, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool Array::operator<=(const Array &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_less_equal_Array, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool Array::operator>(const Array &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_greater_Array, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool Array::operator>=(const Array &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_greater_equal_Array, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

Array Array::operator+(const Array &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<Array>(_method_bindings.operator_add_Array, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

Array &Array::operator=(const Array &p_other) {
	_method_bindings.destructor(&opaque);
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_1, &opaque, &p_other);
	return *this;
}

Array &Array::operator=(Array &&p_other) {
	std::swap(opaque, p_other.opaque);
	return *this;
}

} //namespace godot
