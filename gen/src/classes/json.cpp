/**************************************************************************/
/*  json.cpp                                                              */
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

#include <godot_cpp/classes/json.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

String JSON::stringify(const Variant &p_data, const String &p_indent, bool p_sort_keys, bool p_full_precision) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(JSON::get_class_static()._native_ptr(), StringName("stringify")._native_ptr(), 462733549);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int8_t p_sort_keys_encoded;
	PtrToArg<bool>::encode(p_sort_keys, &p_sort_keys_encoded);
	int8_t p_full_precision_encoded;
	PtrToArg<bool>::encode(p_full_precision, &p_full_precision_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, nullptr, &p_data, &p_indent, &p_sort_keys_encoded, &p_full_precision_encoded);
}

Variant JSON::parse_string(const String &p_json_string) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(JSON::get_class_static()._native_ptr(), StringName("parse_string")._native_ptr(), 309047738);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, nullptr, &p_json_string);
}

Error JSON::parse(const String &p_json_text, bool p_keep_text) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(JSON::get_class_static()._native_ptr(), StringName("parse")._native_ptr(), 885841341);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int8_t p_keep_text_encoded;
	PtrToArg<bool>::encode(p_keep_text, &p_keep_text_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_json_text, &p_keep_text_encoded);
}

Variant JSON::get_data() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(JSON::get_class_static()._native_ptr(), StringName("get_data")._native_ptr(), 1214101251);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner);
}

void JSON::set_data(const Variant &p_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(JSON::get_class_static()._native_ptr(), StringName("set_data")._native_ptr(), 1114965689);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_data);
}

String JSON::get_parsed_text() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(JSON::get_class_static()._native_ptr(), StringName("get_parsed_text")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

int32_t JSON::get_error_line() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(JSON::get_class_static()._native_ptr(), StringName("get_error_line")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

String JSON::get_error_message() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(JSON::get_class_static()._native_ptr(), StringName("get_error_message")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

Variant JSON::from_native(const Variant &p_variant, bool p_full_objects) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(JSON::get_class_static()._native_ptr(), StringName("from_native")._native_ptr(), 2963479484);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	int8_t p_full_objects_encoded;
	PtrToArg<bool>::encode(p_full_objects, &p_full_objects_encoded);
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, nullptr, &p_variant, &p_full_objects_encoded);
}

Variant JSON::to_native(const Variant &p_json, bool p_allow_objects) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(JSON::get_class_static()._native_ptr(), StringName("to_native")._native_ptr(), 2963479484);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	int8_t p_allow_objects_encoded;
	PtrToArg<bool>::encode(p_allow_objects, &p_allow_objects_encoded);
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, nullptr, &p_json, &p_allow_objects_encoded);
}

} // namespace godot
