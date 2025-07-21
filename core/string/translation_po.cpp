/**************************************************************************/
/*  translation_po.cpp                                                    */
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

#include "translation_po.h"

#include "core/string/plural_rules.h"

#ifdef DEBUG_TRANSLATION_PO
#include "core/io/file_access.h"

void TranslationPO::print_translation_map() {
	Error err;
	Ref<FileAccess> file = FileAccess::open("translation_map_print_test.txt", FileAccess::WRITE, &err);
	if (err != OK) {
		ERR_PRINT("Failed to open translation_map_print_test.txt");
		return;
	}

	file->store_line("NPlural : " + String::num_int64(get_plural_forms()));
	file->store_line("Plural rule : " + get_plural_rule());
	file->store_line("");

	List<StringName> context_l;
	translation_map.get_key_list(&context_l);
	for (const StringName &ctx : context_l) {
		file->store_line(" ===== Context: " + String::utf8(String(ctx).utf8()) + " ===== ");
		const HashMap<StringName, Vector<StringName>> &inner_map = translation_map[ctx];

		List<StringName> id_l;
		inner_map.get_key_list(&id_l);
		for (const StringName &id : id_l) {
			file->store_line("msgid: " + String::utf8(String(id).utf8()));
			for (int i = 0; i < inner_map[id].size(); i++) {
				file->store_line("msgstr[" + String::num_int64(i) + "]: " + String::utf8(String(inner_map[id][i]).utf8()));
			}
			file->store_line("");
		}
	}
}
#endif

Dictionary TranslationPO::_get_messages() const {
	// Return translation_map as a Dictionary.

	Dictionary d;

	for (const KeyValue<StringName, HashMap<StringName, Vector<StringName>>> &E : translation_map) {
		Dictionary d2;

		for (const KeyValue<StringName, Vector<StringName>> &E2 : E.value) {
			d2[E2.key] = E2.value;
		}

		d[E.key] = d2;
	}

	return d;
}

void TranslationPO::_set_messages(const Dictionary &p_messages) {
	// Construct translation_map from a Dictionary.

	for (const KeyValue<Variant, Variant> &kv : p_messages) {
		const Dictionary &id_str_map = kv.value;

		HashMap<StringName, Vector<StringName>> temp_map;
		for (const KeyValue<Variant, Variant> &kv_id : id_str_map) {
			StringName id = kv_id.key;
			temp_map[id] = kv_id.value;
		}

		translation_map[kv.key] = temp_map;
	}
}

Vector<String> TranslationPO::get_translated_message_list() const {
	Vector<String> msgs;
	for (const KeyValue<StringName, HashMap<StringName, Vector<StringName>>> &E : translation_map) {
		if (E.key != StringName()) {
			continue;
		}

		for (const KeyValue<StringName, Vector<StringName>> &E2 : E.value) {
			for (const StringName &E3 : E2.value) {
				msgs.push_back(E3);
			}
		}
	}

	return msgs;
}

Vector<String> TranslationPO::_get_message_list() const {
	// Return all keys in translation_map.

	List<StringName> msgs;
	get_message_list(&msgs);

	Vector<String> v;
	for (const StringName &E : msgs) {
		v.push_back(E);
	}

	return v;
}

void TranslationPO::set_plural_rule(const String &p_plural_rule) {
	if (plural_rules) {
		memdelete(plural_rules);
	}
	plural_rules = PluralRules::parse(p_plural_rule);
}

void TranslationPO::add_message(const StringName &p_src_text, const StringName &p_xlated_text, const StringName &p_context) {
	HashMap<StringName, Vector<StringName>> &map_id_str = translation_map[p_context];

	if (map_id_str.has(p_src_text)) {
		WARN_PRINT(vformat("Double translations for \"%s\" under the same context \"%s\" for locale \"%s\".\nThere should only be one unique translation for a given string under the same context.", String(p_src_text), String(p_context), get_locale()));
		map_id_str[p_src_text].set(0, p_xlated_text);
	} else {
		map_id_str[p_src_text].push_back(p_xlated_text);
	}
}

void TranslationPO::add_plural_message(const StringName &p_src_text, const Vector<String> &p_plural_xlated_texts, const StringName &p_context) {
	ERR_FAIL_NULL_MSG(plural_rules, "Plural rules are not set. Please call set_plural_rule() before calling add_plural_message().");
	ERR_FAIL_COND_MSG(p_plural_xlated_texts.size() != plural_rules->get_nplurals(), vformat("Trying to add plural texts that don't match the required number of plural forms for locale \"%s\".", get_locale()));

	HashMap<StringName, Vector<StringName>> &map_id_str = translation_map[p_context];

	if (map_id_str.has(p_src_text)) {
		WARN_PRINT(vformat("Double translations for \"%s\" under the same context \"%s\" for locale %s.\nThere should only be one unique translation for a given string under the same context.", p_src_text, p_context, get_locale()));
		map_id_str[p_src_text].clear();
	}

	for (int i = 0; i < p_plural_xlated_texts.size(); i++) {
		map_id_str[p_src_text].push_back(p_plural_xlated_texts[i]);
	}
}

int TranslationPO::get_plural_forms() const {
	return plural_rules ? plural_rules->get_nplurals() : 0;
}

String TranslationPO::get_plural_rule() const {
	return plural_rules ? plural_rules->get_plural() : String();
}

StringName TranslationPO::get_message(const StringName &p_src_text, const StringName &p_context) const {
	if (!translation_map.has(p_context) || !translation_map[p_context].has(p_src_text)) {
		return StringName();
	}
	ERR_FAIL_COND_V_MSG(translation_map[p_context][p_src_text].is_empty(), StringName(), vformat("Source text \"%s\" is registered but doesn't have a translation. Please report this bug.", String(p_src_text)));

	return translation_map[p_context][p_src_text][0];
}

StringName TranslationPO::get_plural_message(const StringName &p_src_text, const StringName &p_plural_text, int p_n, const StringName &p_context) const {
	ERR_FAIL_COND_V_MSG(p_n < 0, StringName(), "N passed into translation to get a plural message should not be negative. For negative numbers, use singular translation please. Search \"gettext PO Plural Forms\" online for the documentation on translating negative numbers.");
	ERR_FAIL_NULL_V_MSG(plural_rules, StringName(), "Plural rules are not set. Please call set_plural_rule() before calling get_plural_message().");

	if (!translation_map.has(p_context) || !translation_map[p_context].has(p_src_text)) {
		return StringName();
	}
	ERR_FAIL_COND_V_MSG(translation_map[p_context][p_src_text].is_empty(), StringName(), vformat("Source text \"%s\" is registered but doesn't have a translation. Please report this bug.", String(p_src_text)));

	int plural_index = plural_rules->evaluate(p_n);
	ERR_FAIL_COND_V_MSG(plural_index < 0 || translation_map[p_context][p_src_text].size() < plural_index + 1, StringName(), "Plural index returned or number of plural translations is not valid. Please report this bug.");

	return translation_map[p_context][p_src_text][plural_index];
}

void TranslationPO::erase_message(const StringName &p_src_text, const StringName &p_context) {
	if (!translation_map.has(p_context)) {
		return;
	}

	translation_map[p_context].erase(p_src_text);
}

void TranslationPO::get_message_list(List<StringName> *r_messages) const {
	// OptimizedTranslation uses this function to get the list of msgid.
	// Return all the keys of translation_map under "" context.

	for (const KeyValue<StringName, HashMap<StringName, Vector<StringName>>> &E : translation_map) {
		if (E.key != StringName()) {
			continue;
		}

		for (const KeyValue<StringName, Vector<StringName>> &E2 : E.value) {
			r_messages->push_back(E2.key);
		}
	}
}

int TranslationPO::get_message_count() const {
	int count = 0;

	for (const KeyValue<StringName, HashMap<StringName, Vector<StringName>>> &E : translation_map) {
		count += E.value.size();
	}

	return count;
}

void TranslationPO::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_plural_forms"), &TranslationPO::get_plural_forms);
	ClassDB::bind_method(D_METHOD("get_plural_rule"), &TranslationPO::get_plural_rule);
}

TranslationPO::~TranslationPO() {
	if (plural_rules) {
		memdelete(plural_rules);
		plural_rules = nullptr;
	}
}
