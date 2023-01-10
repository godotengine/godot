/**************************************************************************/
/*  translation.h                                                         */
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

#ifndef TRANSLATION_H
#define TRANSLATION_H

#include "core/resource.h"

class Translation : public Resource {
	GDCLASS(Translation, Resource);
	OBJ_SAVE_TYPE(Translation);
	RES_BASE_EXTENSION("translation");

	String locale;
	Map<StringName, StringName> translation_map;

	PoolVector<String> _get_message_list() const;

	PoolVector<String> _get_messages() const;
	void _set_messages(const PoolVector<String> &p_messages);

protected:
	static void _bind_methods();

public:
	void set_locale(const String &p_locale);
	_FORCE_INLINE_ String get_locale() const { return locale; }

	void add_message(const StringName &p_src_text, const StringName &p_xlated_text);
	virtual StringName get_message(const StringName &p_src_text) const; //overridable for other implementations
	void erase_message(const StringName &p_src_text);

	void get_message_list(List<StringName> *r_messages) const;
	int get_message_count() const;

	// Not exposed to scripting. For easy usage of `ContextTranslation`.
	virtual void add_context_message(const StringName &p_src_text, const StringName &p_xlated_text, const StringName &p_context);
	virtual StringName get_context_message(const StringName &p_src_text, const StringName &p_context) const;

	Translation();
};

class ContextTranslation : public Translation {
	GDCLASS(ContextTranslation, Translation);

	Map<StringName, Map<StringName, StringName>> context_translation_map;

public:
	virtual void add_context_message(const StringName &p_src_text, const StringName &p_xlated_text, const StringName &p_context);
	virtual StringName get_context_message(const StringName &p_src_text, const StringName &p_context) const;
};

class TranslationServer : public Object {
	GDCLASS(TranslationServer, Object);

	String locale;
	String fallback;

	Set<Ref<Translation>> translations;
	Ref<Translation> tool_translation;
	Ref<Translation> doc_translation;

	bool enabled;

	static TranslationServer *singleton;
	bool _load_translations(const String &p_from);

	static void _bind_methods();

	struct LocaleScriptInfo {
		String name;
		String script;
		String default_country;
		Set<String> supported_countries;
	};
	static Vector<LocaleScriptInfo> locale_script_info;

	static Map<String, String> language_map;
	static Map<String, String> script_map;
	static Map<String, String> locale_rename_map;
	static Map<String, String> country_name_map;
	static Map<String, String> country_rename_map;
	static Map<String, String> variant_map;

	void init_locale_info();

public:
	_FORCE_INLINE_ static TranslationServer *get_singleton() { return singleton; }

	void set_enabled(bool p_enabled) { enabled = p_enabled; }
	_FORCE_INLINE_ bool is_enabled() const { return enabled; }

	void set_locale(const String &p_locale);
	String get_locale() const;

	int compare_locales(const String &p_locale_a, const String &p_locale_b) const;
	String standardize_locale(const String &p_locale) const;

	Vector<String> get_all_languages() const;
	String get_language_name(const String &p_language) const;

	Vector<String> get_all_scripts() const;
	String get_script_name(const String &p_script) const;

	Vector<String> get_all_countries() const;
	String get_country_name(const String &p_country) const;

	String get_locale_name(const String &p_locale) const;

	Array get_loaded_locales() const;

	void add_translation(const Ref<Translation> &p_translation);
	void remove_translation(const Ref<Translation> &p_translation);

	StringName translate(const StringName &p_message) const;

	void set_tool_translation(const Ref<Translation> &p_translation);
	StringName tool_translate(const StringName &p_message, const StringName &p_context) const;
	void set_doc_translation(const Ref<Translation> &p_translation);
	StringName doc_translate(const StringName &p_message) const;

	void setup();

	void clear();

	void load_translations();

	TranslationServer();
};

#endif // TRANSLATION_H
