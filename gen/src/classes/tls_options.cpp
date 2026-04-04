/**************************************************************************/
/*  tls_options.cpp                                                       */
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

#include <godot_cpp/classes/tls_options.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/crypto_key.hpp>

namespace godot {

Ref<TLSOptions> TLSOptions::client(const Ref<X509Certificate> &p_trusted_chain, const String &p_common_name_override) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TLSOptions::get_class_static()._native_ptr(), StringName("client")._native_ptr(), 3565000357);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<TLSOptions>()));
	return Ref<TLSOptions>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<TLSOptions>(_gde_method_bind, nullptr, (p_trusted_chain != nullptr ? &p_trusted_chain->_owner : nullptr), &p_common_name_override));
}

Ref<TLSOptions> TLSOptions::client_unsafe(const Ref<X509Certificate> &p_trusted_chain) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TLSOptions::get_class_static()._native_ptr(), StringName("client_unsafe")._native_ptr(), 2090251749);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<TLSOptions>()));
	return Ref<TLSOptions>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<TLSOptions>(_gde_method_bind, nullptr, (p_trusted_chain != nullptr ? &p_trusted_chain->_owner : nullptr)));
}

Ref<TLSOptions> TLSOptions::server(const Ref<CryptoKey> &p_key, const Ref<X509Certificate> &p_certificate) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TLSOptions::get_class_static()._native_ptr(), StringName("server")._native_ptr(), 36969539);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<TLSOptions>()));
	return Ref<TLSOptions>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<TLSOptions>(_gde_method_bind, nullptr, (p_key != nullptr ? &p_key->_owner : nullptr), (p_certificate != nullptr ? &p_certificate->_owner : nullptr)));
}

bool TLSOptions::is_server() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TLSOptions::get_class_static()._native_ptr(), StringName("is_server")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool TLSOptions::is_unsafe_client() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TLSOptions::get_class_static()._native_ptr(), StringName("is_unsafe_client")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

String TLSOptions::get_common_name_override() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TLSOptions::get_class_static()._native_ptr(), StringName("get_common_name_override")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

Ref<X509Certificate> TLSOptions::get_trusted_ca_chain() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TLSOptions::get_class_static()._native_ptr(), StringName("get_trusted_ca_chain")._native_ptr(), 1120709175);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<X509Certificate>()));
	return Ref<X509Certificate>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<X509Certificate>(_gde_method_bind, _owner));
}

Ref<CryptoKey> TLSOptions::get_private_key() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TLSOptions::get_class_static()._native_ptr(), StringName("get_private_key")._native_ptr(), 2119971811);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<CryptoKey>()));
	return Ref<CryptoKey>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<CryptoKey>(_gde_method_bind, _owner));
}

Ref<X509Certificate> TLSOptions::get_own_certificate() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TLSOptions::get_class_static()._native_ptr(), StringName("get_own_certificate")._native_ptr(), 1120709175);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<X509Certificate>()));
	return Ref<X509Certificate>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<X509Certificate>(_gde_method_bind, _owner));
}

} // namespace godot
