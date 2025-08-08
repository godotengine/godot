/**************************************************************************/
/*  translation_server.h                                                  */
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

#pragma once

#include "core/string/translation.h"
#include "core/string/translation_domain.h"

class TranslationServer : public Object {
	GDCLASS(TranslationServer, Object);

	String locale = "en";
	String fallback;

	Ref<TranslationDomain> main_domain;
	Ref<TranslationDomain> editor_domain;
	Ref<TranslationDomain> property_domain;
	Ref<TranslationDomain> doc_domain;
	HashMap<StringName, Ref<TranslationDomain>> custom_domains;

	mutable HashMap<String, int> locale_compare_cache;

	static inline TranslationServer *singleton = nullptr;

	Vector<String> _get_csv_line(const uint8_t *p_data, int64_t p_size, int64_t p_start, int64_t &r_end) const;
	String _strip_diacritics(const String &p_string) const;
	void _diacritics_map_add(const String &p_from, char32_t p_to);
	bool _match_code(const String &p_key, const String &p_val, const String &p_str) const;

	static void _bind_methods();

#ifndef DISABLE_DEPRECATED
	String _standardize_locale_bind_compat_98972(const String &p_locale) const;
	static void _bind_compatibility_methods();
#endif

	struct LocaleScriptInfo {
		String name;
		String script;
		String default_country;
		HashSet<String> supported_countries;
	};
	static Vector<LocaleScriptInfo> locale_script_info;

public:
	struct Locale {
		String language;
		String script;
		String country;
		String variant;

		bool operator==(const Locale &p_locale) const {
			return (p_locale.language == language) &&
					(p_locale.script == script) &&
					(p_locale.country == country) &&
					(p_locale.variant == variant);
		}

		explicit operator String() const;

		Locale() {}
		Locale(const String &p_locale, bool p_add_defaults);
	};

private:
	static HashMap<String, String> language_map;
	static HashMap<String, String> language_map_a3_to_a1;
	static HashMap<String, String> script_map;
	static HashMap<String, String> locale_rename_map;
	static HashMap<String, String> country_name_map;
	static HashMap<String, String> country_name_map_a3_to_a1;
	static HashMap<String, String> country_rename_map;
	static HashMap<String, String> variant_map;
	static HashMap<char32_t, char32_t> diacritics_map;

	HashMap<String, String> language_map_custom;
	HashMap<String, String> country_name_map_custom;

	void init_locale_info();

public:
	_FORCE_INLINE_ static TranslationServer *get_singleton() { return singleton; }

	Ref<TranslationDomain> get_main_domain() const { return main_domain; }
	Ref<TranslationDomain> get_editor_domain() const { return editor_domain; }

	void set_locale(const String &p_locale);
	String get_locale() const;
	void set_fallback_locale(const String &p_locale);
	String get_fallback_locale() const;
	Ref<Translation> get_translation_object(const String &p_locale);

	Vector<String> get_all_languages() const;
	String get_language_name(const String &p_language) const;

	Vector<String> get_all_scripts() const;
	String get_script_name(const String &p_script) const;

	Vector<String> get_all_countries() const;
	String get_country_name(const String &p_country) const;

	static bool is_language_code(const String &p_code);
	static bool is_script_code(const String &p_code);
	static bool is_country_code(const String &p_code);

	bool is_language_code_free(const String &p_code) const;
	bool is_country_code_free(const String &p_code) const;

	Vector<String> find_language(const String &p_str) const;
	Vector<String> find_script(const String &p_str) const;
	Vector<String> find_country(const String &p_str) const;

	String get_locale_name(const String &p_locale) const;

	PackedStringArray get_loaded_locales() const;

	void add_translation(const Ref<Translation> &p_translation);
	void remove_translation(const Ref<Translation> &p_translation);

	StringName translate(const StringName &p_message, const StringName &p_context = "") const;
	StringName translate_plural(const StringName &p_message, const StringName &p_message_plural, int p_n, const StringName &p_context = "") const;

	StringName pseudolocalize(const StringName &p_message) const;

	bool is_pseudolocalization_enabled() const;
	void set_pseudolocalization_enabled(bool p_enabled);
	void reload_pseudolocalization();

	String standardize_locale(const String &p_locale, bool p_add_defaults = false) const;

	int compare_locales(const String &p_locale_a, const String &p_locale_b) const;

	String get_tool_locale();
	StringName tool_translate(const StringName &p_message, const StringName &p_context = "") const;
	StringName tool_translate_plural(const StringName &p_message, const StringName &p_message_plural, int p_n, const StringName &p_context = "") const;
	StringName property_translate(const StringName &p_message, const StringName &p_context = "") const;
	StringName doc_translate(const StringName &p_message, const StringName &p_context = "") const;
	StringName doc_translate_plural(const StringName &p_message, const StringName &p_message_plural, int p_n, const StringName &p_context = "") const;

	void set_custom_language_codes(const Dictionary &p_dict);
	Dictionary get_custom_language_codes() const;

	void set_custom_country_codes(const Dictionary &p_dict);
	Dictionary get_custom_country_codes() const;

	bool has_domain(const StringName &p_domain) const;
	Ref<TranslationDomain> get_or_add_domain(const StringName &p_domain);
	void remove_domain(const StringName &p_domain);

	void setup();

	void clear();

	void load_translations();

#ifdef TOOLS_ENABLED
	virtual void get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const override;
#endif // TOOLS_ENABLED

	TranslationServer();
};
