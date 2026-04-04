/**************************************************************************/
/*  reg_ex.cpp                                                            */
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

#include <godot_cpp/classes/reg_ex.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/reg_ex_match.hpp>

namespace godot {

Ref<RegEx> RegEx::create_from_string(const String &p_pattern, bool p_show_error) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RegEx::get_class_static()._native_ptr(), StringName("create_from_string")._native_ptr(), 4249111514);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<RegEx>()));
	int8_t p_show_error_encoded;
	PtrToArg<bool>::encode(p_show_error, &p_show_error_encoded);
	return Ref<RegEx>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<RegEx>(_gde_method_bind, nullptr, &p_pattern, &p_show_error_encoded));
}

void RegEx::clear() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RegEx::get_class_static()._native_ptr(), StringName("clear")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

Error RegEx::compile(const String &p_pattern, bool p_show_error) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RegEx::get_class_static()._native_ptr(), StringName("compile")._native_ptr(), 3565188097);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int8_t p_show_error_encoded;
	PtrToArg<bool>::encode(p_show_error, &p_show_error_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_pattern, &p_show_error_encoded);
}

Ref<RegExMatch> RegEx::search(const String &p_subject, int32_t p_offset, int32_t p_end) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RegEx::get_class_static()._native_ptr(), StringName("search")._native_ptr(), 3365977994);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<RegExMatch>()));
	int64_t p_offset_encoded;
	PtrToArg<int64_t>::encode(p_offset, &p_offset_encoded);
	int64_t p_end_encoded;
	PtrToArg<int64_t>::encode(p_end, &p_end_encoded);
	return Ref<RegExMatch>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<RegExMatch>(_gde_method_bind, _owner, &p_subject, &p_offset_encoded, &p_end_encoded));
}

TypedArray<Ref<RegExMatch>> RegEx::search_all(const String &p_subject, int32_t p_offset, int32_t p_end) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RegEx::get_class_static()._native_ptr(), StringName("search_all")._native_ptr(), 849021363);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Ref<RegExMatch>>()));
	int64_t p_offset_encoded;
	PtrToArg<int64_t>::encode(p_offset, &p_offset_encoded);
	int64_t p_end_encoded;
	PtrToArg<int64_t>::encode(p_end, &p_end_encoded);
	return ::godot::internal::_call_native_mb_ret<TypedArray<Ref<RegExMatch>>>(_gde_method_bind, _owner, &p_subject, &p_offset_encoded, &p_end_encoded);
}

String RegEx::sub(const String &p_subject, const String &p_replacement, bool p_all, int32_t p_offset, int32_t p_end) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RegEx::get_class_static()._native_ptr(), StringName("sub")._native_ptr(), 54019702);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int8_t p_all_encoded;
	PtrToArg<bool>::encode(p_all, &p_all_encoded);
	int64_t p_offset_encoded;
	PtrToArg<int64_t>::encode(p_offset, &p_offset_encoded);
	int64_t p_end_encoded;
	PtrToArg<int64_t>::encode(p_end, &p_end_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_subject, &p_replacement, &p_all_encoded, &p_offset_encoded, &p_end_encoded);
}

bool RegEx::is_valid() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RegEx::get_class_static()._native_ptr(), StringName("is_valid")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

String RegEx::get_pattern() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RegEx::get_class_static()._native_ptr(), StringName("get_pattern")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

int32_t RegEx::get_group_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RegEx::get_class_static()._native_ptr(), StringName("get_group_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

PackedStringArray RegEx::get_names() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RegEx::get_class_static()._native_ptr(), StringName("get_names")._native_ptr(), 1139954409);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner);
}

} // namespace godot
