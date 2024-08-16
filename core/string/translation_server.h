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

#ifndef TRANSLATION_SERVER_H
#define TRANSLATION_SERVER_H

#include "core/string/translation.h"

class TranslationServer : public Object {
	GDCLASS(TranslationServer, Object);

	String locale = "en";
	String fallback;

	HashSet<Ref<Translation>> translations;
	Ref<Translation> tool_translation;
	Ref<Translation> property_translation;
	Ref<Translation> doc_translation;
	Ref<Translation> extractable_translation;

	bool enabled = true;

	bool pseudolocalization_enabled = false;
	bool pseudolocalization_accents_enabled = false;
	bool pseudolocalization_double_vowels_enabled = false;
	bool pseudolocalization_fake_bidi_enabled = false;
	bool pseudolocalization_override_enabled = false;
	bool pseudolocalization_skip_placeholders_enabled = false;
	float expansion_ratio = 0.0;
	String pseudolocalization_prefix;
	String pseudolocalization_suffix;

	StringName tool_pseudolocalize(const StringName &p_message) const;
	String get_override_string(String &p_message) const;
	String double_vowels(String &p_message) const;
	String replace_with_accented_string(String &p_message) const;
	String wrap_with_fakebidi_characters(String &p_message) const;
	String add_padding(const String &p_message, int p_length) const;
	const char32_t *get_accented_version(char32_t p_character) const;
	bool is_placeholder(String &p_message, int p_index) const;

	static TranslationServer *singleton;
	bool _load_translations(const String &p_from);
	String _standardize_locale(const String &p_locale, bool p_add_defaults) const;

	StringName _get_message_from_translations(const StringName &p_message, const StringName &p_context, const String &p_locale, bool plural, const String &p_message_plural = "", int p_n = 0) const;

	static void _bind_methods();

#ifndef DISABLE_DEPRECATED
	static void _bind_compatibility_methods();
#endif

	struct LocaleScriptInfo {
		String name;
		String script;
		String default_country;
		HashSet<String> supported_countries;
	};
	static Vector<LocaleScriptInfo> locale_script_info;

	static HashMap<String, String> language_map;
	static HashMap<String, String> script_map;
	static HashMap<String, String> locale_rename_map;
	static HashMap<String, String> country_name_map;
	static HashMap<String, String> country_rename_map;
	static HashMap<String, String> variant_map;

	void init_locale_info();

public:
	_FORCE_INLINE_ static TranslationServer *get_singleton() { return singleton; }

	void set_enabled(bool p_enabled) { enabled = p_enabled; }
	_FORCE_INLINE_ bool is_enabled() const { return enabled; }

	void set_locale(const String &p_locale);
	String get_locale() const;
	Ref<Translation> get_translation_object(const String &p_locale);

	Vector<String> get_all_languages() const;
	String get_language_name(const String &p_language) const;

	Vector<String> get_all_scripts() const;
	String get_script_name(const String &p_script) const;

	Vector<String> get_all_countries() const;
	String get_country_name(const String &p_country) const;

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

	String standardize_locale(const String &p_locale) const;

	int compare_locales(const String &p_locale_a, const String &p_locale_b) const;

	String get_tool_locale();
	void set_tool_translation(const Ref<Translation> &p_translation);
	Ref<Translation> get_tool_translation() const;
	StringName tool_translate(const StringName &p_message, const StringName &p_context = "") const;
	StringName tool_translate_plural(const StringName &p_message, const StringName &p_message_plural, int p_n, const StringName &p_context = "") const;
	void set_property_translation(const Ref<Translation> &p_translation);
	StringName property_translate(const StringName &p_message, const StringName &p_context = "") const;
	void set_doc_translation(const Ref<Translation> &p_translation);
	StringName doc_translate(const StringName &p_message, const StringName &p_context = "") const;
	StringName doc_translate_plural(const StringName &p_message, const StringName &p_message_plural, int p_n, const StringName &p_context = "") const;
	void set_extractable_translation(const Ref<Translation> &p_translation);
	StringName extractable_translate(const StringName &p_message, const StringName &p_context = "") const;
	StringName extractable_translate_plural(const StringName &p_message, const StringName &p_message_plural, int p_n, const StringName &p_context = "") const;

	void setup();

	void clear();

	void load_translations();

#ifdef TOOLS_ENABLED
	virtual void get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const override;
#endif // TOOLS_ENABLED

	TranslationServer();
};

#endif // TRANSLATION_SERVER_H
