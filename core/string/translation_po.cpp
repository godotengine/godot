/*************************************************************************/
/*  translation_po.cpp                                                   */
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

#include "translation_po.h"

#include "core/os/file_access.h"

#ifdef DEBUG_TRANSLATION_PO
void TranslationPO::print_translation_map() {
	Error err;
	FileAccess *file = FileAccess::open("translation_map_print_test.txt", FileAccess::WRITE, &err);
	if (err != OK) {
		ERR_PRINT("Failed to open translation_map_print_test.txt");
		return;
	}

	file->store_line("NPlural : " + String::num_int64(this->get_plural_forms()));
	file->store_line("Plural rule : " + this->get_plural_rule());
	file->store_line("");

	List<StringName> context_l;
	translation_map.get_key_list(&context_l);
	for (auto E = context_l.front(); E; E = E->next()) {
		StringName ctx = E->get();
		file->store_line(" ===== Context: " + String::utf8(String(ctx).utf8()) + " ===== ");
		const HashMap<StringName, Vector<StringName>> &inner_map = translation_map[ctx];

		List<StringName> id_l;
		inner_map.get_key_list(&id_l);
		for (auto E2 = id_l.front(); E2; E2 = E2->next()) {
			StringName id = E2->get();
			file->store_line("msgid: " + String::utf8(String(id).utf8()));
			for (int i = 0; i < inner_map[id].size(); i++) {
				file->store_line("msgstr[" + String::num_int64(i) + "]: " + String::utf8(String(inner_map[id][i]).utf8()));
			}
			file->store_line("");
		}
	}
	file->close();
}
#endif

Dictionary TranslationPO::_get_messages() const {
	// Return translation_map as a Dictionary.

	Dictionary d;

	List<StringName> context_l;
	translation_map.get_key_list(&context_l);
	for (auto E = context_l.front(); E; E = E->next()) {
		StringName ctx = E->get();
		const HashMap<StringName, Vector<StringName>> &id_str_map = translation_map[ctx];

		Dictionary d2;
		List<StringName> id_l;
		id_str_map.get_key_list(&id_l);
		// Save list of id and strs associated with a context in a temporary dictionary.
		for (auto E2 = id_l.front(); E2; E2 = E2->next()) {
			StringName id = E2->get();
			d2[id] = id_str_map[id];
		}

		d[ctx] = d2;
	}

	return d;
}

void TranslationPO::_set_messages(const Dictionary &p_messages) {
	// Construct translation_map from a Dictionary.

	List<Variant> context_l;
	p_messages.get_key_list(&context_l);
	for (auto E = context_l.front(); E; E = E->next()) {
		StringName ctx = E->get();
		const Dictionary &id_str_map = p_messages[ctx];

		HashMap<StringName, Vector<StringName>> temp_map;
		List<Variant> id_l;
		id_str_map.get_key_list(&id_l);
		for (auto E2 = id_l.front(); E2; E2 = E2->next()) {
			StringName id = E2->get();
			temp_map[id] = id_str_map[id];
		}

		translation_map[ctx] = temp_map;
	}
}

Vector<String> TranslationPO::_get_message_list() const {
	// Return all keys in translation_map.

	List<StringName> msgs;
	get_message_list(&msgs);

	Vector<String> v;
	for (auto E = msgs.front(); E; E = E->next()) {
		v.push_back(E->get());
	}

	return v;
}

int TranslationPO::_get_plural_index(int p_n) const {
	// Get a number between [0;number of plural forms).

	input_val.clear();
	input_val.push_back(p_n);

	Variant result;
	for (int i = 0; i < equi_tests.size(); i++) {
		Error err = expr->parse(equi_tests[i], input_name);
		ERR_FAIL_COND_V_MSG(err != OK, 0, "Cannot parse expression. Error: " + expr->get_error_text());

		result = expr->execute(input_val);
		ERR_FAIL_COND_V_MSG(expr->has_execute_failed(), 0, "Cannot evaluate expression.");

		// Last expression. Variant result will either map to a bool or an integer, in both cases returning it will give the correct plural index.
		if (i + 1 == equi_tests.size()) {
			return result;
		}

		if (bool(result)) {
			return i;
		}
	}

	ERR_FAIL_V_MSG(0, "Unexpected. Function should have returned. Please report this bug.");
}

void TranslationPO::_cache_plural_tests(const String &p_plural_rule) {
	// Some examples of p_plural_rule passed in can have the form:
	// "n==0 ? 0 : n==1 ? 1 : n==2 ? 2 : n%100>=3 && n%100<=10 ? 3 : n%100>=11 && n%100<=99 ? 4 : 5" (Arabic)
	// "n >= 2" (French) // When evaluating the last, esp careful with this one.
	// "n != 1" (English)
	int first_ques_mark = p_plural_rule.find("?");
	if (first_ques_mark == -1) {
		equi_tests.push_back(p_plural_rule.strip_edges());
		return;
	}

	String equi_test = p_plural_rule.substr(0, first_ques_mark).strip_edges();
	equi_tests.push_back(equi_test);

	String after_colon = p_plural_rule.substr(p_plural_rule.find(":") + 1, p_plural_rule.length());
	_cache_plural_tests(after_colon);
}

void TranslationPO::set_plural_rule(const String &p_plural_rule) {
	// Set plural_forms and plural_rule.
	// p_plural_rule passed in has the form "Plural-Forms: nplurals=2; plural=(n >= 2);".

	int first_semi_col = p_plural_rule.find(";");
	plural_forms = p_plural_rule.substr(p_plural_rule.find("=") + 1, first_semi_col - (p_plural_rule.find("=") + 1)).to_int();

	int expression_start = p_plural_rule.find("=", first_semi_col) + 1;
	int second_semi_col = p_plural_rule.rfind(";");
	plural_rule = p_plural_rule.substr(expression_start, second_semi_col - expression_start);

	// Setup the cache to make evaluating plural rule faster later on.
	plural_rule = plural_rule.replacen("(", "");
	plural_rule = plural_rule.replacen(")", "");
	_cache_plural_tests(plural_rule);
	expr.instance();
	input_name.push_back("n");
}

void TranslationPO::add_message(const StringName &p_src_text, const StringName &p_xlated_text, const StringName &p_context) {
	HashMap<StringName, Vector<StringName>> &map_id_str = translation_map[p_context];

	if (map_id_str.has(p_src_text)) {
		WARN_PRINT("Double translations for \"" + String(p_src_text) + "\" under the same context \"" + String(p_context) + "\" for locale \"" + get_locale() + "\".\nThere should only be one unique translation for a given string under the same context.");
		map_id_str[p_src_text].set(0, p_xlated_text);
	} else {
		map_id_str[p_src_text].push_back(p_xlated_text);
	}
}

void TranslationPO::add_plural_message(const StringName &p_src_text, const Vector<String> &p_plural_xlated_texts, const StringName &p_context) {
	ERR_FAIL_COND_MSG(p_plural_xlated_texts.size() != plural_forms, "Trying to add plural texts that don't match the required number of plural forms for locale \"" + get_locale() + "\"");

	HashMap<StringName, Vector<StringName>> &map_id_str = translation_map[p_context];

	if (map_id_str.has(p_src_text)) {
		WARN_PRINT("Double translations for \"" + p_src_text + "\" under the same context \"" + p_context + "\" for locale " + get_locale() + ".\nThere should only be one unique translation for a given string under the same context.");
		map_id_str[p_src_text].clear();
	}

	for (int i = 0; i < p_plural_xlated_texts.size(); i++) {
		map_id_str[p_src_text].push_back(p_plural_xlated_texts[i]);
	}
}

int TranslationPO::get_plural_forms() const {
	return plural_forms;
}

String TranslationPO::get_plural_rule() const {
	return plural_rule;
}

StringName TranslationPO::get_message(const StringName &p_src_text, const StringName &p_context) const {
	if (!translation_map.has(p_context) || !translation_map[p_context].has(p_src_text)) {
		return StringName();
	}
	ERR_FAIL_COND_V_MSG(translation_map[p_context][p_src_text].empty(), StringName(), "Source text \"" + String(p_src_text) + "\" is registered but doesn't have a translation. Please report this bug.");

	return translation_map[p_context][p_src_text][0];
}

StringName TranslationPO::get_plural_message(const StringName &p_src_text, const StringName &p_plural_text, int p_n, const StringName &p_context) const {
	ERR_FAIL_COND_V_MSG(p_n < 0, StringName(), "N passed into translation to get a plural message should not be negative. For negative numbers, use singular translation please. Search \"gettext PO Plural Forms\" online for the documentation on translating negative numbers.");

	// If the query is the same as last time, return the cached result.
	if (p_n == last_plural_n && p_context == last_plural_context && p_src_text == last_plural_key) {
		return translation_map[p_context][p_src_text][last_plural_mapped_index];
	}

	if (!translation_map.has(p_context) || !translation_map[p_context].has(p_src_text)) {
		return StringName();
	}
	ERR_FAIL_COND_V_MSG(translation_map[p_context][p_src_text].empty(), StringName(), "Source text \"" + String(p_src_text) + "\" is registered but doesn't have a translation. Please report this bug.");

	if (translation_map[p_context][p_src_text].size() == 1) {
		WARN_PRINT("Source string \"" + String(p_src_text) + "\" doesn't have plural translations. Use singular translation API for such as tr(), TTR() to translate \"" + String(p_src_text) + "\"");
		return translation_map[p_context][p_src_text][0];
	}

	int plural_index = _get_plural_index(p_n);
	ERR_FAIL_COND_V_MSG(plural_index < 0 || translation_map[p_context][p_src_text].size() < plural_index + 1, StringName(), "Plural index returned or number of plural translations is not valid. Please report this bug.");

	// Cache result so that if the next entry is the same, we can return directly.
	// _get_plural_index(p_n) can get very costly, especially when evaluating long plural-rule (Arabic)
	last_plural_key = p_src_text;
	last_plural_context = p_context;
	last_plural_n = p_n;
	last_plural_mapped_index = plural_index;

	return translation_map[p_context][p_src_text][plural_index];
}

void TranslationPO::erase_message(const StringName &p_src_text, const StringName &p_context) {
	if (!translation_map.has(p_context)) {
		return;
	}

	translation_map[p_context].erase(p_src_text);
}

void TranslationPO::get_message_list(List<StringName> *r_messages) const {
	// PHashTranslation uses this function to get the list of msgid.
	// Return all the keys of translation_map under "" context.

	List<StringName> context_l;
	translation_map.get_key_list(&context_l);

	for (auto E = context_l.front(); E; E = E->next()) {
		if (String(E->get()) != "") {
			continue;
		}

		List<StringName> msgid_l;
		translation_map[E->get()].get_key_list(&msgid_l);

		for (auto E2 = msgid_l.front(); E2; E2 = E2->next()) {
			r_messages->push_back(E2->get());
		}
	}
}

int TranslationPO::get_message_count() const {
	List<StringName> context_l;
	translation_map.get_key_list(&context_l);

	int count = 0;
	for (auto E = context_l.front(); E; E = E->next()) {
		count += translation_map[E->get()].size();
	}
	return count;
}

void TranslationPO::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_plural_forms"), &TranslationPO::get_plural_forms);
	ClassDB::bind_method(D_METHOD("get_plural_rule"), &TranslationPO::get_plural_rule);
}
