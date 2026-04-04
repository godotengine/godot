/**************************************************************************/
/*  crypto_key.cpp                                                        */
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

#include <godot_cpp/classes/crypto_key.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

Error CryptoKey::save(const String &p_path, bool p_public_only) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CryptoKey::get_class_static()._native_ptr(), StringName("save")._native_ptr(), 885841341);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int8_t p_public_only_encoded;
	PtrToArg<bool>::encode(p_public_only, &p_public_only_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path, &p_public_only_encoded);
}

Error CryptoKey::load(const String &p_path, bool p_public_only) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CryptoKey::get_class_static()._native_ptr(), StringName("load")._native_ptr(), 885841341);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int8_t p_public_only_encoded;
	PtrToArg<bool>::encode(p_public_only, &p_public_only_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path, &p_public_only_encoded);
}

bool CryptoKey::is_public_only() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CryptoKey::get_class_static()._native_ptr(), StringName("is_public_only")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

String CryptoKey::save_to_string(bool p_public_only) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CryptoKey::get_class_static()._native_ptr(), StringName("save_to_string")._native_ptr(), 32795936);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int8_t p_public_only_encoded;
	PtrToArg<bool>::encode(p_public_only, &p_public_only_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_public_only_encoded);
}

Error CryptoKey::load_from_string(const String &p_string_key, bool p_public_only) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CryptoKey::get_class_static()._native_ptr(), StringName("load_from_string")._native_ptr(), 885841341);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int8_t p_public_only_encoded;
	PtrToArg<bool>::encode(p_public_only, &p_public_only_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_string_key, &p_public_only_encoded);
}

} // namespace godot
