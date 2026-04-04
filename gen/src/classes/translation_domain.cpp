/**************************************************************************/
/*  translation_domain.cpp                                                */
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

#include <godot_cpp/classes/translation_domain.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/translation.hpp>

namespace godot {

Ref<Translation> TranslationDomain::get_translation_object(const String &p_locale) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationDomain::get_class_static()._native_ptr(), StringName("get_translation_object")._native_ptr(), 606768082);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Translation>()));
	return Ref<Translation>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Translation>(_gde_method_bind, _owner, &p_locale));
}

void TranslationDomain::add_translation(const Ref<Translation> &p_translation) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationDomain::get_class_static()._native_ptr(), StringName("add_translation")._native_ptr(), 1466479800);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_translation != nullptr ? &p_translation->_owner : nullptr));
}

void TranslationDomain::remove_translation(const Ref<Translation> &p_translation) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationDomain::get_class_static()._native_ptr(), StringName("remove_translation")._native_ptr(), 1466479800);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_translation != nullptr ? &p_translation->_owner : nullptr));
}

void TranslationDomain::clear() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationDomain::get_class_static()._native_ptr(), StringName("clear")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

TypedArray<Ref<Translation>> TranslationDomain::get_translations() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationDomain::get_class_static()._native_ptr(), StringName("get_translations")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Ref<Translation>>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Ref<Translation>>>(_gde_method_bind, _owner);
}

bool TranslationDomain::has_translation_for_locale(const String &p_locale, bool p_exact) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationDomain::get_class_static()._native_ptr(), StringName("has_translation_for_locale")._native_ptr(), 2034713381);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int8_t p_exact_encoded;
	PtrToArg<bool>::encode(p_exact, &p_exact_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_locale, &p_exact_encoded);
}

bool TranslationDomain::has_translation(const Ref<Translation> &p_translation) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationDomain::get_class_static()._native_ptr(), StringName("has_translation")._native_ptr(), 2696976312);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, (p_translation != nullptr ? &p_translation->_owner : nullptr));
}

TypedArray<Ref<Translation>> TranslationDomain::find_translations(const String &p_locale, bool p_exact) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationDomain::get_class_static()._native_ptr(), StringName("find_translations")._native_ptr(), 2109650934);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Ref<Translation>>()));
	int8_t p_exact_encoded;
	PtrToArg<bool>::encode(p_exact, &p_exact_encoded);
	return ::godot::internal::_call_native_mb_ret<TypedArray<Ref<Translation>>>(_gde_method_bind, _owner, &p_locale, &p_exact_encoded);
}

StringName TranslationDomain::translate(const StringName &p_message, const StringName &p_context) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationDomain::get_class_static()._native_ptr(), StringName("translate")._native_ptr(), 1829228469);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner, &p_message, &p_context);
}

StringName TranslationDomain::translate_plural(const StringName &p_message, const StringName &p_message_plural, int32_t p_n, const StringName &p_context) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationDomain::get_class_static()._native_ptr(), StringName("translate_plural")._native_ptr(), 229954002);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	int64_t p_n_encoded;
	PtrToArg<int64_t>::encode(p_n, &p_n_encoded);
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner, &p_message, &p_message_plural, &p_n_encoded, &p_context);
}

String TranslationDomain::get_locale_override() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationDomain::get_class_static()._native_ptr(), StringName("get_locale_override")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void TranslationDomain::set_locale_override(const String &p_locale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationDomain::get_class_static()._native_ptr(), StringName("set_locale_override")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_locale);
}

bool TranslationDomain::is_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationDomain::get_class_static()._native_ptr(), StringName("is_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TranslationDomain::set_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationDomain::get_class_static()._native_ptr(), StringName("set_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TranslationDomain::is_pseudolocalization_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationDomain::get_class_static()._native_ptr(), StringName("is_pseudolocalization_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TranslationDomain::set_pseudolocalization_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationDomain::get_class_static()._native_ptr(), StringName("set_pseudolocalization_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TranslationDomain::is_pseudolocalization_accents_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationDomain::get_class_static()._native_ptr(), StringName("is_pseudolocalization_accents_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TranslationDomain::set_pseudolocalization_accents_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationDomain::get_class_static()._native_ptr(), StringName("set_pseudolocalization_accents_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TranslationDomain::is_pseudolocalization_double_vowels_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationDomain::get_class_static()._native_ptr(), StringName("is_pseudolocalization_double_vowels_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TranslationDomain::set_pseudolocalization_double_vowels_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationDomain::get_class_static()._native_ptr(), StringName("set_pseudolocalization_double_vowels_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TranslationDomain::is_pseudolocalization_fake_bidi_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationDomain::get_class_static()._native_ptr(), StringName("is_pseudolocalization_fake_bidi_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TranslationDomain::set_pseudolocalization_fake_bidi_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationDomain::get_class_static()._native_ptr(), StringName("set_pseudolocalization_fake_bidi_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TranslationDomain::is_pseudolocalization_override_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationDomain::get_class_static()._native_ptr(), StringName("is_pseudolocalization_override_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TranslationDomain::set_pseudolocalization_override_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationDomain::get_class_static()._native_ptr(), StringName("set_pseudolocalization_override_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool TranslationDomain::is_pseudolocalization_skip_placeholders_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationDomain::get_class_static()._native_ptr(), StringName("is_pseudolocalization_skip_placeholders_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void TranslationDomain::set_pseudolocalization_skip_placeholders_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationDomain::get_class_static()._native_ptr(), StringName("set_pseudolocalization_skip_placeholders_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

float TranslationDomain::get_pseudolocalization_expansion_ratio() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationDomain::get_class_static()._native_ptr(), StringName("get_pseudolocalization_expansion_ratio")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void TranslationDomain::set_pseudolocalization_expansion_ratio(float p_ratio) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationDomain::get_class_static()._native_ptr(), StringName("set_pseudolocalization_expansion_ratio")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_ratio_encoded;
	PtrToArg<double>::encode(p_ratio, &p_ratio_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_ratio_encoded);
}

String TranslationDomain::get_pseudolocalization_prefix() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationDomain::get_class_static()._native_ptr(), StringName("get_pseudolocalization_prefix")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void TranslationDomain::set_pseudolocalization_prefix(const String &p_prefix) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationDomain::get_class_static()._native_ptr(), StringName("set_pseudolocalization_prefix")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_prefix);
}

String TranslationDomain::get_pseudolocalization_suffix() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationDomain::get_class_static()._native_ptr(), StringName("get_pseudolocalization_suffix")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void TranslationDomain::set_pseudolocalization_suffix(const String &p_suffix) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationDomain::get_class_static()._native_ptr(), StringName("set_pseudolocalization_suffix")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_suffix);
}

StringName TranslationDomain::pseudolocalize(const StringName &p_message) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TranslationDomain::get_class_static()._native_ptr(), StringName("pseudolocalize")._native_ptr(), 1965194235);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner, &p_message);
}

} // namespace godot
