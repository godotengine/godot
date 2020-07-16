/*************************************************************************/
/*  translation.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef TRANSLATION_H
#define TRANSLATION_H

//#define DEBUG_TRANSLATION

#include "core/math/expression.h"
#include "core/resource.h"

class Translation : public Resource {
	GDCLASS(Translation, Resource);
	OBJ_SAVE_TYPE(Translation);
	RES_BASE_EXTENSION("translation");

	String locale = "en";
	int plural_forms = 0; // 0 means no "Plural-Forms" is given in the PO header file. The min for all languages is 1.
	String plural_rule;

	// TLDR: Maps context to a list of source strings and translated strings. In PO terms, maps msgctxt to a list of msgid and msgstr.
	// The first key corresponds to context, and the second key (of the contained HashMap) corresponds to source string.
	// The value Vector<StringName> in the second map stores the translated strings. Index 0, 1, 2 matches msgstr[0], msgstr[1], msgstr[2]... in the case of plurals.
	// Otherwise index 0 mathes to msgstr in a singular translation.
	// Strings without context have "" as first key.
	HashMap<StringName, HashMap<StringName, Vector<StringName>>> translation_map;

	Vector<String> _get_message_list() const;

	Dictionary _get_messages() const;
	void _set_messages(const Dictionary &p_messages);

	int _get_plural_index(int p_n) const;
	int _get_plural_index(const String &p_plural_rule, const Vector<String> &p_input_name, const Array &p_input_value, Ref<Expression> &r_expr) const;

protected:
	static void _bind_methods();

public:
	void set_locale(const String &p_locale);
	_FORCE_INLINE_ String get_locale() const { return locale; }
	void set_plural_rule(const String &p_plural_rule);

	void add_message(const StringName &p_src_text, const StringName &p_xlated_text, const StringName &p_context = "");
	void add_plural_message(const StringName &p_src_text, const Vector<String> &p_plural_texts, const StringName &p_context = "");
	virtual StringName get_message(const StringName &p_src_text, const StringName &p_context = "") const; //overridable for other implementations
	virtual StringName get_plural_message(const StringName &p_src_text, const StringName &p_plural_text, int p_n, const StringName &p_context = "") const;
	void erase_message(const StringName &p_src_text, const StringName &p_context = "");

	void get_message_list(List<StringName> *r_messages) const;
	int get_message_count() const;

	int get_plural_forms() const;
	String get_plural_rule() const;

#ifdef DEBUG_TRANSLATION
	void print_translation_map();
#endif

	Translation() {}
};

class TranslationServer : public Object {
	GDCLASS(TranslationServer, Object);

	String locale = "en";
	String fallback;

	Set<Ref<Translation>> translations;
	Ref<Translation> tool_translation;
	Ref<Translation> doc_translation;

	Map<String, String> locale_name_map;

	bool enabled = true;

	static TranslationServer *singleton;
	bool _load_translations(const String &p_from);

	StringName _get_message_from_translations(const StringName &p_message, const StringName &p_context, const String &p_locale, const String &p_message_plural = "", int p_n = -1) const;

	static void _bind_methods();

public:
	_FORCE_INLINE_ static TranslationServer *get_singleton() { return singleton; }

	void set_enabled(bool p_enabled) { enabled = p_enabled; }
	_FORCE_INLINE_ bool is_enabled() const { return enabled; }

	void set_locale(const String &p_locale);
	String get_locale() const;

	String get_locale_name(const String &p_locale) const;

	Array get_loaded_locales() const;

	void add_translation(const Ref<Translation> &p_translation);
	void remove_translation(const Ref<Translation> &p_translation);

	StringName translate(const StringName &p_message, const StringName &p_context = "") const;
	StringName translate_plural(const StringName &p_message, const StringName &p_message_plural, int p_n, const StringName &p_context = "") const;

	static Vector<String> get_all_locales();
	static Vector<String> get_all_locale_names();
	static bool is_locale_valid(const String &p_locale);
	static String standardize_locale(const String &p_locale);
	static String get_language_code(const String &p_locale);

	void set_tool_translation(const Ref<Translation> &p_translation);
	StringName tool_translate(const StringName &p_message, const StringName &p_context = "") const;
	StringName tool_translate_plural(const StringName &p_message, const StringName &p_message_plural, int p_n, const StringName &p_context = "") const;
	void set_doc_translation(const Ref<Translation> &p_translation);
	StringName doc_translate(const StringName &p_message, const StringName &p_context = "") const;
	StringName doc_translate_plural(const StringName &p_message, const StringName &p_message_plural, int p_n, const StringName &p_context = "") const;

	void setup();

	void clear();

	void load_translations();

	TranslationServer();
};

#endif // TRANSLATION_H
