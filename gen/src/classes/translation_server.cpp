/**************************************************************************/
/*  translation_server.cpp                                                */
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

#include <godot_cpp/classes/translation_server.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/translation.hpp>
#include <godot_cpp/classes/translation_domain.hpp>

namespace godot {

TranslationServer *TranslationServer::singleton = nullptr;

TranslationServer *TranslationServer::get_singleton() {
	if (unlikely(singleton == nullptr)) {
		GDExtensionObjectPtr singleton_obj = ::godot::gdextension_interface::global_get_singleton(TranslationServer::get_class_static()._native_ptr());
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton_obj, nullptr);
#endif // DEBUG_ENABLED
		singleton = reinterpret_cast<TranslationServer *>(::godot::gdextension_interface::object_get_instance_binding(singleton_obj, ::godot::gdextension_interface::token, &TranslationServer::_gde_binding_callbacks));
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton, nullptr);
#endif // DEBUG_ENABLED
		if (likely(singleton)) {
			ClassDB::_register_engine_singleton(TranslationServer::get_class_static(), singleton);
		}
	}
	return singleton;
}

TranslationServer::~TranslationServer() {
	if (singleton == this) {
		ClassDB::_unregister_engine_singleton(TranslationServer::get_class_static());
		singleton = nullptr;
	}
}

void TranslationServer::set_locale(const String &p_locale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("set_locale")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_locale);
}

String TranslationServer::get_locale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("get_locale")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String TranslationServer::get_tool_locale() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("get_tool_locale")._native_ptr(), 2841200299);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

int32_t TranslationServer::compare_locales(const String &p_locale_a, const String &p_locale_b) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("compare_locales")._native_ptr(), 2878152881);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_locale_a, &p_locale_b);
}

String TranslationServer::standardize_locale(const String &p_locale, bool p_add_defaults) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("standardize_locale")._native_ptr(), 4216441673);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int8_t p_add_defaults_encoded;
	PtrToArg<bool>::encode(p_add_defaults, &p_add_defaults_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_locale, &p_add_defaults_encoded);
}

PackedStringArray TranslationServer::get_all_languages() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("get_all_languages")._native_ptr(), 1139954409);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner);
}

String TranslationServer::get_language_name(const String &p_language) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("get_language_name")._native_ptr(), 3135753539);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_language);
}

PackedStringArray TranslationServer::get_all_scripts() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("get_all_scripts")._native_ptr(), 1139954409);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner);
}

String TranslationServer::get_script_name(const String &p_script) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("get_script_name")._native_ptr(), 3135753539);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_script);
}

PackedStringArray TranslationServer::get_all_countries() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("get_all_countries")._native_ptr(), 1139954409);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner);
}

String TranslationServer::get_country_name(const String &p_country) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("get_country_name")._native_ptr(), 3135753539);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_country);
}

String TranslationServer::get_locale_name(const String &p_locale) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("get_locale_name")._native_ptr(), 3135753539);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_locale);
}

String TranslationServer::get_plural_rules(const String &p_locale) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("get_plural_rules")._native_ptr(), 3135753539);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_locale);
}

StringName TranslationServer::translate(const StringName &p_message, const StringName &p_context) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("translate")._native_ptr(), 1829228469);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner, &p_message, &p_context);
}

StringName TranslationServer::translate_plural(const StringName &p_message, const StringName &p_plural_message, int32_t p_n, const StringName &p_context) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("translate_plural")._native_ptr(), 229954002);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	int64_t p_n_encoded;
	PtrToArg<int64_t>::encode(p_n, &p_n_encoded);
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner, &p_message, &p_plural_message, &p_n_encoded, &p_context);
}

void TranslationServer::add_translation(const Ref<Translation> &p_translation) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("add_translation")._native_ptr(), 1466479800);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_translation != nullptr ? &p_translation->_owner : nullptr));
}

void TranslationServer::remove_translation(const Ref<Translation> &p_translation) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("remove_translation")._native_ptr(), 1466479800);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_translation != nullptr ? &p_translation->_owner : nullptr));
}

Ref<Translation> TranslationServer::get_translation_object(const String &p_locale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("get_translation_object")._native_ptr(), 2065240175);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Translation>()));
	return Ref<Translation>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Translation>(_gde_method_bind, _owner, &p_locale));
}

TypedArray<Ref<Translation>> TranslationServer::get_translations() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("get_translations")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Ref<Translation>>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Ref<Translation>>>(_gde_method_bind, _owner);
}

TypedArray<Ref<Translation>> TranslationServer::find_translations(const String &p_locale, bool p_exact) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("find_translations")._native_ptr(), 2109650934);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Ref<Translation>>()));
	int8_t p_exact_encoded;
	PtrToArg<bool>::encode(p_exact, &p_exact_encoded);
	return ::godot::internal::_call_native_mb_ret<TypedArray<Ref<Translation>>>(_gde_method_bind, _owner, &p_locale, &p_exact_encoded);
}

bool TranslationServer::has_translation_for_locale(const String &p_locale, bool p_exact) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("has_translation_for_locale")._native_ptr(), 2034713381);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int8_t p_exact_encoded;
	PtrToArg<bool>::encode(p_exact, &p_exact_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_locale, &p_exact_encoded);
}

bool TranslationServer::has_translation(const Ref<Translation> &p_translation) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("has_translation")._native_ptr(), 2696976312);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, (p_translation != nullptr ? &p_translation->_owner : nullptr));
}

bool TranslationServer::has_domain(const StringName &p_domain) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("has_domain")._native_ptr(), 2619796661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_domain);
}

Ref<TranslationDomain> TranslationServer::get_or_add_domain(const StringName &p_domain) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("get_or_add_domain")._native_ptr(), 397200075);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<TranslationDomain>()));
	return Ref<TranslationDomain>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<TranslationDomain>(_gde_method_bind, _owner, &p_domain));
}

void TranslationServer::remove_domain(const StringName &p_domain) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("remove_domain")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_domain);
}

void TranslationServer::clear() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("clear")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

PackedStringArray TranslationServer::get_loaded_locales() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("get_loaded_locales")._native_ptr(), 1139954409);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner);
}

String TranslationServer::format_number(const String &p_number, const String &p_locale) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("format_number")._native_ptr(), 315676799);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_number, &p_locale);
}

String TranslationServer::get_percent_sign(const String &p_locale) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("get_percent_sign")._native_ptr(), 3135753539);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_locale);
}

String TranslationServer::parse_number(const String &p_number, const String &p_locale) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("parse_number")._native_ptr(), 315676799);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_number, &p_locale);
}

bool TranslationServer::is_pseudolocalization_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("is_pseudolocalization_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TranslationServer::set_pseudolocalization_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("set_pseudolocalization_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

void TranslationServer::reload_pseudolocalization() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("reload_pseudolocalization")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

StringName TranslationServer::pseudolocalize(const StringName &p_message) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationServer::get_class_static()._native_ptr(), StringName("pseudolocalize")._native_ptr(), 1965194235);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner, &p_message);
}

} // namespace godot
