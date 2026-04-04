/**************************************************************************/
/*  packed_byte_array.cpp                                                 */
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

#include <godot_cpp/variant/packed_byte_array.hpp>

#include <godot_cpp/core/binder_common.hpp>

#include <godot_cpp/godot.hpp>

#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_color_array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_float64_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/packed_int64_array.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/packed_vector4_array.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/builtin_ptrcall.hpp>

#include <utility>

namespace godot {

PackedByteArray::_MethodBindings PackedByteArray::_method_bindings;

void PackedByteArray::_init_bindings_constructors_destructor() {
	_method_bindings.from_variant_constructor = ::godot::gdextension_interface::get_variant_to_type_constructor(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY);
	_method_bindings.constructor_0 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, 0);
	_method_bindings.constructor_1 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, 1);
	_method_bindings.constructor_2 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, 2);
	_method_bindings.destructor = ::godot::gdextension_interface::variant_get_ptr_destructor(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY);
}
void PackedByteArray::init_bindings() {
	PackedByteArray::_init_bindings_constructors_destructor();
	StringName _gde_name;
	_gde_name = StringName("get");
	_method_bindings.method_get = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 4103005248);
	_gde_name = StringName("set");
	_method_bindings.method_set = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 3638975848);
	_gde_name = StringName("size");
	_method_bindings.method_size = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 3173160232);
	_gde_name = StringName("is_empty");
	_method_bindings.method_is_empty = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 3918633141);
	_gde_name = StringName("push_back");
	_method_bindings.method_push_back = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 694024632);
	_gde_name = StringName("append");
	_method_bindings.method_append = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 694024632);
	_gde_name = StringName("append_array");
	_method_bindings.method_append_array = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 791097111);
	_gde_name = StringName("remove_at");
	_method_bindings.method_remove_at = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 2823966027);
	_gde_name = StringName("insert");
	_method_bindings.method_insert = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 1487112728);
	_gde_name = StringName("fill");
	_method_bindings.method_fill = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 2823966027);
	_gde_name = StringName("resize");
	_method_bindings.method_resize = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 848867239);
	_gde_name = StringName("clear");
	_method_bindings.method_clear = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 3218959716);
	_gde_name = StringName("has");
	_method_bindings.method_has = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 931488181);
	_gde_name = StringName("reverse");
	_method_bindings.method_reverse = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 3218959716);
	_gde_name = StringName("slice");
	_method_bindings.method_slice = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 2278869132);
	_gde_name = StringName("sort");
	_method_bindings.method_sort = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 3218959716);
	_gde_name = StringName("bsearch");
	_method_bindings.method_bsearch = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 954237325);
	_gde_name = StringName("duplicate");
	_method_bindings.method_duplicate = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 247621236);
	_gde_name = StringName("find");
	_method_bindings.method_find = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 2984303840);
	_gde_name = StringName("rfind");
	_method_bindings.method_rfind = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 2984303840);
	_gde_name = StringName("count");
	_method_bindings.method_count = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 4103005248);
	_gde_name = StringName("erase");
	_method_bindings.method_erase = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 694024632);
	_gde_name = StringName("get_string_from_ascii");
	_method_bindings.method_get_string_from_ascii = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 3942272618);
	_gde_name = StringName("get_string_from_utf8");
	_method_bindings.method_get_string_from_utf8 = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 3942272618);
	_gde_name = StringName("get_string_from_utf16");
	_method_bindings.method_get_string_from_utf16 = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 3942272618);
	_gde_name = StringName("get_string_from_utf32");
	_method_bindings.method_get_string_from_utf32 = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 3942272618);
	_gde_name = StringName("get_string_from_wchar");
	_method_bindings.method_get_string_from_wchar = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 3942272618);
	_gde_name = StringName("get_string_from_multibyte_char");
	_method_bindings.method_get_string_from_multibyte_char = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 3134094431);
	_gde_name = StringName("hex_encode");
	_method_bindings.method_hex_encode = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 3942272618);
	_gde_name = StringName("compress");
	_method_bindings.method_compress = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 1845905913);
	_gde_name = StringName("decompress");
	_method_bindings.method_decompress = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 2278869132);
	_gde_name = StringName("decompress_dynamic");
	_method_bindings.method_decompress_dynamic = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 2278869132);
	_gde_name = StringName("decode_u8");
	_method_bindings.method_decode_u8 = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 4103005248);
	_gde_name = StringName("decode_s8");
	_method_bindings.method_decode_s8 = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 4103005248);
	_gde_name = StringName("decode_u16");
	_method_bindings.method_decode_u16 = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 4103005248);
	_gde_name = StringName("decode_s16");
	_method_bindings.method_decode_s16 = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 4103005248);
	_gde_name = StringName("decode_u32");
	_method_bindings.method_decode_u32 = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 4103005248);
	_gde_name = StringName("decode_s32");
	_method_bindings.method_decode_s32 = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 4103005248);
	_gde_name = StringName("decode_u64");
	_method_bindings.method_decode_u64 = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 4103005248);
	_gde_name = StringName("decode_s64");
	_method_bindings.method_decode_s64 = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 4103005248);
	_gde_name = StringName("decode_half");
	_method_bindings.method_decode_half = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 1401583798);
	_gde_name = StringName("decode_float");
	_method_bindings.method_decode_float = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 1401583798);
	_gde_name = StringName("decode_double");
	_method_bindings.method_decode_double = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 1401583798);
	_gde_name = StringName("has_encoded_var");
	_method_bindings.method_has_encoded_var = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 2914632957);
	_gde_name = StringName("decode_var");
	_method_bindings.method_decode_var = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 1740420038);
	_gde_name = StringName("decode_var_size");
	_method_bindings.method_decode_var_size = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 954237325);
	_gde_name = StringName("to_int32_array");
	_method_bindings.method_to_int32_array = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 3158844420);
	_gde_name = StringName("to_int64_array");
	_method_bindings.method_to_int64_array = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 1961294120);
	_gde_name = StringName("to_float32_array");
	_method_bindings.method_to_float32_array = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 3575107827);
	_gde_name = StringName("to_float64_array");
	_method_bindings.method_to_float64_array = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 1627308337);
	_gde_name = StringName("to_vector2_array");
	_method_bindings.method_to_vector2_array = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 1660374357);
	_gde_name = StringName("to_vector3_array");
	_method_bindings.method_to_vector3_array = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 4171207452);
	_gde_name = StringName("to_vector4_array");
	_method_bindings.method_to_vector4_array = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 146203628);
	_gde_name = StringName("to_color_array");
	_method_bindings.method_to_color_array = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 3072026941);
	_gde_name = StringName("bswap16");
	_method_bindings.method_bswap16 = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 3638975848);
	_gde_name = StringName("bswap32");
	_method_bindings.method_bswap32 = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 3638975848);
	_gde_name = StringName("bswap64");
	_method_bindings.method_bswap64 = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 3638975848);
	_gde_name = StringName("encode_u8");
	_method_bindings.method_encode_u8 = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 3638975848);
	_gde_name = StringName("encode_s8");
	_method_bindings.method_encode_s8 = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 3638975848);
	_gde_name = StringName("encode_u16");
	_method_bindings.method_encode_u16 = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 3638975848);
	_gde_name = StringName("encode_s16");
	_method_bindings.method_encode_s16 = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 3638975848);
	_gde_name = StringName("encode_u32");
	_method_bindings.method_encode_u32 = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 3638975848);
	_gde_name = StringName("encode_s32");
	_method_bindings.method_encode_s32 = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 3638975848);
	_gde_name = StringName("encode_u64");
	_method_bindings.method_encode_u64 = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 3638975848);
	_gde_name = StringName("encode_s64");
	_method_bindings.method_encode_s64 = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 3638975848);
	_gde_name = StringName("encode_half");
	_method_bindings.method_encode_half = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 1113000516);
	_gde_name = StringName("encode_float");
	_method_bindings.method_encode_float = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 1113000516);
	_gde_name = StringName("encode_double");
	_method_bindings.method_encode_double = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 1113000516);
	_gde_name = StringName("encode_var");
	_method_bindings.method_encode_var = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, _gde_name._native_ptr(), 2604460497);
	_method_bindings.indexed_setter = ::godot::gdextension_interface::variant_get_ptr_indexed_setter(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY);
	_method_bindings.indexed_getter = ::godot::gdextension_interface::variant_get_ptr_indexed_getter(GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY);
	_method_bindings.operator_equal_Variant = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_EQUAL, GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, GDEXTENSION_VARIANT_TYPE_NIL);
	_method_bindings.operator_not_equal_Variant = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_NOT_EQUAL, GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, GDEXTENSION_VARIANT_TYPE_NIL);
	_method_bindings.operator_not = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_NOT, GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, GDEXTENSION_VARIANT_TYPE_NIL);
	_method_bindings.operator_in_Dictionary = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_IN, GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, GDEXTENSION_VARIANT_TYPE_DICTIONARY);
	_method_bindings.operator_in_Array = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_IN, GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, GDEXTENSION_VARIANT_TYPE_ARRAY);
	_method_bindings.operator_equal_PackedByteArray = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_EQUAL, GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY);
	_method_bindings.operator_not_equal_PackedByteArray = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_NOT_EQUAL, GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY);
	_method_bindings.operator_add_PackedByteArray = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_ADD, GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY, GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY);
}

PackedByteArray::PackedByteArray(const Variant *p_variant) {
	_method_bindings.from_variant_constructor(&opaque, p_variant->_native_ptr());
}

PackedByteArray::PackedByteArray() {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_0, &opaque);
}

PackedByteArray::PackedByteArray(const PackedByteArray &p_from) {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_1, &opaque, &p_from);
}

PackedByteArray::PackedByteArray(const Array &p_from) {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_2, &opaque, &p_from);
}

PackedByteArray::PackedByteArray(PackedByteArray &&p_other) {
	std::swap(opaque, p_other.opaque);
}

PackedByteArray::~PackedByteArray() {
	_method_bindings.destructor(&opaque);
}

int64_t PackedByteArray::get(int64_t p_index) const {
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_get, (GDExtensionTypePtr)&opaque, &p_index_encoded);
}

void PackedByteArray::set(int64_t p_index, int64_t p_value) {
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	int64_t p_value_encoded;
	PtrToArg<int64_t>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_set, (GDExtensionTypePtr)&opaque, &p_index_encoded, &p_value_encoded);
}

int64_t PackedByteArray::size() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_size, (GDExtensionTypePtr)&opaque);
}

bool PackedByteArray::is_empty() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_empty, (GDExtensionTypePtr)&opaque);
}

bool PackedByteArray::push_back(int64_t p_value) {
	int64_t p_value_encoded;
	PtrToArg<int64_t>::encode(p_value, &p_value_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_push_back, (GDExtensionTypePtr)&opaque, &p_value_encoded);
}

bool PackedByteArray::append(int64_t p_value) {
	int64_t p_value_encoded;
	PtrToArg<int64_t>::encode(p_value, &p_value_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_append, (GDExtensionTypePtr)&opaque, &p_value_encoded);
}

void PackedByteArray::append_array(const PackedByteArray &p_array) {
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_append_array, (GDExtensionTypePtr)&opaque, &p_array);
}

void PackedByteArray::remove_at(int64_t p_index) {
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_remove_at, (GDExtensionTypePtr)&opaque, &p_index_encoded);
}

int64_t PackedByteArray::insert(int64_t p_at_index, int64_t p_value) {
	int64_t p_at_index_encoded;
	PtrToArg<int64_t>::encode(p_at_index, &p_at_index_encoded);
	int64_t p_value_encoded;
	PtrToArg<int64_t>::encode(p_value, &p_value_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_insert, (GDExtensionTypePtr)&opaque, &p_at_index_encoded, &p_value_encoded);
}

void PackedByteArray::fill(int64_t p_value) {
	int64_t p_value_encoded;
	PtrToArg<int64_t>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_fill, (GDExtensionTypePtr)&opaque, &p_value_encoded);
}

int64_t PackedByteArray::resize(int64_t p_new_size) {
	int64_t p_new_size_encoded;
	PtrToArg<int64_t>::encode(p_new_size, &p_new_size_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_resize, (GDExtensionTypePtr)&opaque, &p_new_size_encoded);
}

void PackedByteArray::clear() {
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_clear, (GDExtensionTypePtr)&opaque);
}

bool PackedByteArray::has(int64_t p_value) const {
	int64_t p_value_encoded;
	PtrToArg<int64_t>::encode(p_value, &p_value_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_has, (GDExtensionTypePtr)&opaque, &p_value_encoded);
}

void PackedByteArray::reverse() {
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_reverse, (GDExtensionTypePtr)&opaque);
}

PackedByteArray PackedByteArray::slice(int64_t p_begin, int64_t p_end) const {
	int64_t p_begin_encoded;
	PtrToArg<int64_t>::encode(p_begin, &p_begin_encoded);
	int64_t p_end_encoded;
	PtrToArg<int64_t>::encode(p_end, &p_end_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<PackedByteArray>(_method_bindings.method_slice, (GDExtensionTypePtr)&opaque, &p_begin_encoded, &p_end_encoded);
}

void PackedByteArray::sort() {
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_sort, (GDExtensionTypePtr)&opaque);
}

int64_t PackedByteArray::bsearch(int64_t p_value, bool p_before) const {
	int64_t p_value_encoded;
	PtrToArg<int64_t>::encode(p_value, &p_value_encoded);
	int8_t p_before_encoded;
	PtrToArg<bool>::encode(p_before, &p_before_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_bsearch, (GDExtensionTypePtr)&opaque, &p_value_encoded, &p_before_encoded);
}

PackedByteArray PackedByteArray::duplicate() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<PackedByteArray>(_method_bindings.method_duplicate, (GDExtensionTypePtr)&opaque);
}

int64_t PackedByteArray::find(int64_t p_value, int64_t p_from) const {
	int64_t p_value_encoded;
	PtrToArg<int64_t>::encode(p_value, &p_value_encoded);
	int64_t p_from_encoded;
	PtrToArg<int64_t>::encode(p_from, &p_from_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_find, (GDExtensionTypePtr)&opaque, &p_value_encoded, &p_from_encoded);
}

int64_t PackedByteArray::rfind(int64_t p_value, int64_t p_from) const {
	int64_t p_value_encoded;
	PtrToArg<int64_t>::encode(p_value, &p_value_encoded);
	int64_t p_from_encoded;
	PtrToArg<int64_t>::encode(p_from, &p_from_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_rfind, (GDExtensionTypePtr)&opaque, &p_value_encoded, &p_from_encoded);
}

int64_t PackedByteArray::count(int64_t p_value) const {
	int64_t p_value_encoded;
	PtrToArg<int64_t>::encode(p_value, &p_value_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_count, (GDExtensionTypePtr)&opaque, &p_value_encoded);
}

bool PackedByteArray::erase(int64_t p_value) {
	int64_t p_value_encoded;
	PtrToArg<int64_t>::encode(p_value, &p_value_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_erase, (GDExtensionTypePtr)&opaque, &p_value_encoded);
}

String PackedByteArray::get_string_from_ascii() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_get_string_from_ascii, (GDExtensionTypePtr)&opaque);
}

String PackedByteArray::get_string_from_utf8() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_get_string_from_utf8, (GDExtensionTypePtr)&opaque);
}

String PackedByteArray::get_string_from_utf16() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_get_string_from_utf16, (GDExtensionTypePtr)&opaque);
}

String PackedByteArray::get_string_from_utf32() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_get_string_from_utf32, (GDExtensionTypePtr)&opaque);
}

String PackedByteArray::get_string_from_wchar() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_get_string_from_wchar, (GDExtensionTypePtr)&opaque);
}

String PackedByteArray::get_string_from_multibyte_char(const String &p_encoding) const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_get_string_from_multibyte_char, (GDExtensionTypePtr)&opaque, &p_encoding);
}

String PackedByteArray::hex_encode() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<String>(_method_bindings.method_hex_encode, (GDExtensionTypePtr)&opaque);
}

PackedByteArray PackedByteArray::compress(int64_t p_compression_mode) const {
	int64_t p_compression_mode_encoded;
	PtrToArg<int64_t>::encode(p_compression_mode, &p_compression_mode_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<PackedByteArray>(_method_bindings.method_compress, (GDExtensionTypePtr)&opaque, &p_compression_mode_encoded);
}

PackedByteArray PackedByteArray::decompress(int64_t p_buffer_size, int64_t p_compression_mode) const {
	int64_t p_buffer_size_encoded;
	PtrToArg<int64_t>::encode(p_buffer_size, &p_buffer_size_encoded);
	int64_t p_compression_mode_encoded;
	PtrToArg<int64_t>::encode(p_compression_mode, &p_compression_mode_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<PackedByteArray>(_method_bindings.method_decompress, (GDExtensionTypePtr)&opaque, &p_buffer_size_encoded, &p_compression_mode_encoded);
}

PackedByteArray PackedByteArray::decompress_dynamic(int64_t p_max_output_size, int64_t p_compression_mode) const {
	int64_t p_max_output_size_encoded;
	PtrToArg<int64_t>::encode(p_max_output_size, &p_max_output_size_encoded);
	int64_t p_compression_mode_encoded;
	PtrToArg<int64_t>::encode(p_compression_mode, &p_compression_mode_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<PackedByteArray>(_method_bindings.method_decompress_dynamic, (GDExtensionTypePtr)&opaque, &p_max_output_size_encoded, &p_compression_mode_encoded);
}

int64_t PackedByteArray::decode_u8(int64_t p_byte_offset) const {
	int64_t p_byte_offset_encoded;
	PtrToArg<int64_t>::encode(p_byte_offset, &p_byte_offset_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_decode_u8, (GDExtensionTypePtr)&opaque, &p_byte_offset_encoded);
}

int64_t PackedByteArray::decode_s8(int64_t p_byte_offset) const {
	int64_t p_byte_offset_encoded;
	PtrToArg<int64_t>::encode(p_byte_offset, &p_byte_offset_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_decode_s8, (GDExtensionTypePtr)&opaque, &p_byte_offset_encoded);
}

int64_t PackedByteArray::decode_u16(int64_t p_byte_offset) const {
	int64_t p_byte_offset_encoded;
	PtrToArg<int64_t>::encode(p_byte_offset, &p_byte_offset_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_decode_u16, (GDExtensionTypePtr)&opaque, &p_byte_offset_encoded);
}

int64_t PackedByteArray::decode_s16(int64_t p_byte_offset) const {
	int64_t p_byte_offset_encoded;
	PtrToArg<int64_t>::encode(p_byte_offset, &p_byte_offset_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_decode_s16, (GDExtensionTypePtr)&opaque, &p_byte_offset_encoded);
}

int64_t PackedByteArray::decode_u32(int64_t p_byte_offset) const {
	int64_t p_byte_offset_encoded;
	PtrToArg<int64_t>::encode(p_byte_offset, &p_byte_offset_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_decode_u32, (GDExtensionTypePtr)&opaque, &p_byte_offset_encoded);
}

int64_t PackedByteArray::decode_s32(int64_t p_byte_offset) const {
	int64_t p_byte_offset_encoded;
	PtrToArg<int64_t>::encode(p_byte_offset, &p_byte_offset_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_decode_s32, (GDExtensionTypePtr)&opaque, &p_byte_offset_encoded);
}

int64_t PackedByteArray::decode_u64(int64_t p_byte_offset) const {
	int64_t p_byte_offset_encoded;
	PtrToArg<int64_t>::encode(p_byte_offset, &p_byte_offset_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_decode_u64, (GDExtensionTypePtr)&opaque, &p_byte_offset_encoded);
}

int64_t PackedByteArray::decode_s64(int64_t p_byte_offset) const {
	int64_t p_byte_offset_encoded;
	PtrToArg<int64_t>::encode(p_byte_offset, &p_byte_offset_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_decode_s64, (GDExtensionTypePtr)&opaque, &p_byte_offset_encoded);
}

double PackedByteArray::decode_half(int64_t p_byte_offset) const {
	int64_t p_byte_offset_encoded;
	PtrToArg<int64_t>::encode(p_byte_offset, &p_byte_offset_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<double>(_method_bindings.method_decode_half, (GDExtensionTypePtr)&opaque, &p_byte_offset_encoded);
}

double PackedByteArray::decode_float(int64_t p_byte_offset) const {
	int64_t p_byte_offset_encoded;
	PtrToArg<int64_t>::encode(p_byte_offset, &p_byte_offset_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<double>(_method_bindings.method_decode_float, (GDExtensionTypePtr)&opaque, &p_byte_offset_encoded);
}

double PackedByteArray::decode_double(int64_t p_byte_offset) const {
	int64_t p_byte_offset_encoded;
	PtrToArg<int64_t>::encode(p_byte_offset, &p_byte_offset_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<double>(_method_bindings.method_decode_double, (GDExtensionTypePtr)&opaque, &p_byte_offset_encoded);
}

bool PackedByteArray::has_encoded_var(int64_t p_byte_offset, bool p_allow_objects) const {
	int64_t p_byte_offset_encoded;
	PtrToArg<int64_t>::encode(p_byte_offset, &p_byte_offset_encoded);
	int8_t p_allow_objects_encoded;
	PtrToArg<bool>::encode(p_allow_objects, &p_allow_objects_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_has_encoded_var, (GDExtensionTypePtr)&opaque, &p_byte_offset_encoded, &p_allow_objects_encoded);
}

Variant PackedByteArray::decode_var(int64_t p_byte_offset, bool p_allow_objects) const {
	int64_t p_byte_offset_encoded;
	PtrToArg<int64_t>::encode(p_byte_offset, &p_byte_offset_encoded);
	int8_t p_allow_objects_encoded;
	PtrToArg<bool>::encode(p_allow_objects, &p_allow_objects_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<Variant>(_method_bindings.method_decode_var, (GDExtensionTypePtr)&opaque, &p_byte_offset_encoded, &p_allow_objects_encoded);
}

int64_t PackedByteArray::decode_var_size(int64_t p_byte_offset, bool p_allow_objects) const {
	int64_t p_byte_offset_encoded;
	PtrToArg<int64_t>::encode(p_byte_offset, &p_byte_offset_encoded);
	int8_t p_allow_objects_encoded;
	PtrToArg<bool>::encode(p_allow_objects, &p_allow_objects_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_decode_var_size, (GDExtensionTypePtr)&opaque, &p_byte_offset_encoded, &p_allow_objects_encoded);
}

PackedInt32Array PackedByteArray::to_int32_array() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<PackedInt32Array>(_method_bindings.method_to_int32_array, (GDExtensionTypePtr)&opaque);
}

PackedInt64Array PackedByteArray::to_int64_array() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<PackedInt64Array>(_method_bindings.method_to_int64_array, (GDExtensionTypePtr)&opaque);
}

PackedFloat32Array PackedByteArray::to_float32_array() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<PackedFloat32Array>(_method_bindings.method_to_float32_array, (GDExtensionTypePtr)&opaque);
}

PackedFloat64Array PackedByteArray::to_float64_array() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<PackedFloat64Array>(_method_bindings.method_to_float64_array, (GDExtensionTypePtr)&opaque);
}

PackedVector2Array PackedByteArray::to_vector2_array() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<PackedVector2Array>(_method_bindings.method_to_vector2_array, (GDExtensionTypePtr)&opaque);
}

PackedVector3Array PackedByteArray::to_vector3_array() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<PackedVector3Array>(_method_bindings.method_to_vector3_array, (GDExtensionTypePtr)&opaque);
}

PackedVector4Array PackedByteArray::to_vector4_array() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<PackedVector4Array>(_method_bindings.method_to_vector4_array, (GDExtensionTypePtr)&opaque);
}

PackedColorArray PackedByteArray::to_color_array() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<PackedColorArray>(_method_bindings.method_to_color_array, (GDExtensionTypePtr)&opaque);
}

void PackedByteArray::bswap16(int64_t p_offset, int64_t p_count) {
	int64_t p_offset_encoded;
	PtrToArg<int64_t>::encode(p_offset, &p_offset_encoded);
	int64_t p_count_encoded;
	PtrToArg<int64_t>::encode(p_count, &p_count_encoded);
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_bswap16, (GDExtensionTypePtr)&opaque, &p_offset_encoded, &p_count_encoded);
}

void PackedByteArray::bswap32(int64_t p_offset, int64_t p_count) {
	int64_t p_offset_encoded;
	PtrToArg<int64_t>::encode(p_offset, &p_offset_encoded);
	int64_t p_count_encoded;
	PtrToArg<int64_t>::encode(p_count, &p_count_encoded);
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_bswap32, (GDExtensionTypePtr)&opaque, &p_offset_encoded, &p_count_encoded);
}

void PackedByteArray::bswap64(int64_t p_offset, int64_t p_count) {
	int64_t p_offset_encoded;
	PtrToArg<int64_t>::encode(p_offset, &p_offset_encoded);
	int64_t p_count_encoded;
	PtrToArg<int64_t>::encode(p_count, &p_count_encoded);
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_bswap64, (GDExtensionTypePtr)&opaque, &p_offset_encoded, &p_count_encoded);
}

void PackedByteArray::encode_u8(int64_t p_byte_offset, int64_t p_value) {
	int64_t p_byte_offset_encoded;
	PtrToArg<int64_t>::encode(p_byte_offset, &p_byte_offset_encoded);
	int64_t p_value_encoded;
	PtrToArg<int64_t>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_encode_u8, (GDExtensionTypePtr)&opaque, &p_byte_offset_encoded, &p_value_encoded);
}

void PackedByteArray::encode_s8(int64_t p_byte_offset, int64_t p_value) {
	int64_t p_byte_offset_encoded;
	PtrToArg<int64_t>::encode(p_byte_offset, &p_byte_offset_encoded);
	int64_t p_value_encoded;
	PtrToArg<int64_t>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_encode_s8, (GDExtensionTypePtr)&opaque, &p_byte_offset_encoded, &p_value_encoded);
}

void PackedByteArray::encode_u16(int64_t p_byte_offset, int64_t p_value) {
	int64_t p_byte_offset_encoded;
	PtrToArg<int64_t>::encode(p_byte_offset, &p_byte_offset_encoded);
	int64_t p_value_encoded;
	PtrToArg<int64_t>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_encode_u16, (GDExtensionTypePtr)&opaque, &p_byte_offset_encoded, &p_value_encoded);
}

void PackedByteArray::encode_s16(int64_t p_byte_offset, int64_t p_value) {
	int64_t p_byte_offset_encoded;
	PtrToArg<int64_t>::encode(p_byte_offset, &p_byte_offset_encoded);
	int64_t p_value_encoded;
	PtrToArg<int64_t>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_encode_s16, (GDExtensionTypePtr)&opaque, &p_byte_offset_encoded, &p_value_encoded);
}

void PackedByteArray::encode_u32(int64_t p_byte_offset, int64_t p_value) {
	int64_t p_byte_offset_encoded;
	PtrToArg<int64_t>::encode(p_byte_offset, &p_byte_offset_encoded);
	int64_t p_value_encoded;
	PtrToArg<int64_t>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_encode_u32, (GDExtensionTypePtr)&opaque, &p_byte_offset_encoded, &p_value_encoded);
}

void PackedByteArray::encode_s32(int64_t p_byte_offset, int64_t p_value) {
	int64_t p_byte_offset_encoded;
	PtrToArg<int64_t>::encode(p_byte_offset, &p_byte_offset_encoded);
	int64_t p_value_encoded;
	PtrToArg<int64_t>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_encode_s32, (GDExtensionTypePtr)&opaque, &p_byte_offset_encoded, &p_value_encoded);
}

void PackedByteArray::encode_u64(int64_t p_byte_offset, int64_t p_value) {
	int64_t p_byte_offset_encoded;
	PtrToArg<int64_t>::encode(p_byte_offset, &p_byte_offset_encoded);
	int64_t p_value_encoded;
	PtrToArg<int64_t>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_encode_u64, (GDExtensionTypePtr)&opaque, &p_byte_offset_encoded, &p_value_encoded);
}

void PackedByteArray::encode_s64(int64_t p_byte_offset, int64_t p_value) {
	int64_t p_byte_offset_encoded;
	PtrToArg<int64_t>::encode(p_byte_offset, &p_byte_offset_encoded);
	int64_t p_value_encoded;
	PtrToArg<int64_t>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_encode_s64, (GDExtensionTypePtr)&opaque, &p_byte_offset_encoded, &p_value_encoded);
}

void PackedByteArray::encode_half(int64_t p_byte_offset, double p_value) {
	int64_t p_byte_offset_encoded;
	PtrToArg<int64_t>::encode(p_byte_offset, &p_byte_offset_encoded);
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_encode_half, (GDExtensionTypePtr)&opaque, &p_byte_offset_encoded, &p_value_encoded);
}

void PackedByteArray::encode_float(int64_t p_byte_offset, double p_value) {
	int64_t p_byte_offset_encoded;
	PtrToArg<int64_t>::encode(p_byte_offset, &p_byte_offset_encoded);
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_encode_float, (GDExtensionTypePtr)&opaque, &p_byte_offset_encoded, &p_value_encoded);
}

void PackedByteArray::encode_double(int64_t p_byte_offset, double p_value) {
	int64_t p_byte_offset_encoded;
	PtrToArg<int64_t>::encode(p_byte_offset, &p_byte_offset_encoded);
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_builtin_method_ptr_no_ret(_method_bindings.method_encode_double, (GDExtensionTypePtr)&opaque, &p_byte_offset_encoded, &p_value_encoded);
}

int64_t PackedByteArray::encode_var(int64_t p_byte_offset, const Variant &p_value, bool p_allow_objects) {
	int64_t p_byte_offset_encoded;
	PtrToArg<int64_t>::encode(p_byte_offset, &p_byte_offset_encoded);
	int8_t p_allow_objects_encoded;
	PtrToArg<bool>::encode(p_allow_objects, &p_allow_objects_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_encode_var, (GDExtensionTypePtr)&opaque, &p_byte_offset_encoded, &p_value, &p_allow_objects_encoded);
}

bool PackedByteArray::operator==(const Variant &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_equal_Variant, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool PackedByteArray::operator!=(const Variant &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_not_equal_Variant, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool PackedByteArray::operator!() const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_not, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr) nullptr);
}

bool PackedByteArray::operator==(const PackedByteArray &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_equal_PackedByteArray, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool PackedByteArray::operator!=(const PackedByteArray &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_not_equal_PackedByteArray, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

PackedByteArray PackedByteArray::operator+(const PackedByteArray &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<PackedByteArray>(_method_bindings.operator_add_PackedByteArray, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

PackedByteArray &PackedByteArray::operator=(const PackedByteArray &p_other) {
	_method_bindings.destructor(&opaque);
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_1, &opaque, &p_other);
	return *this;
}

PackedByteArray &PackedByteArray::operator=(PackedByteArray &&p_other) {
	std::swap(opaque, p_other.opaque);
	return *this;
}

} //namespace godot
