/*************************************************************************/
/*  translation.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "translation.h"

#include "core/config/project_settings.h"
#include "core/io/resource_loader.h"
#include "core/os/os.h"
#include "core/string/locales.h"

#ifdef TOOLS_ENABLED
#include "main/main.h"
#endif

Dictionary Translation::_get_messages() const {
	Dictionary d;
	for (const KeyValue<StringName, StringName> &E : translation_map) {
		d[E.key] = E.value;
	}
	return d;
}

Vector<String> Translation::_get_message_list() const {
	Vector<String> msgs;
	msgs.resize(translation_map.size());
	int idx = 0;
	for (const KeyValue<StringName, StringName> &E : translation_map) {
		msgs.set(idx, E.key);
		idx += 1;
	}

	return msgs;
}

void Translation::_set_messages(const Dictionary &p_messages) {
	List<Variant> keys;
	p_messages.get_key_list(&keys);
	for (const Variant &E : keys) {
		translation_map[E] = p_messages[E];
	}
}

void Translation::set_locale(const String &p_locale) {
	locale = TranslationServer::get_singleton()->standardize_locale(p_locale);

	if (OS::get_singleton()->get_main_loop() && TranslationServer::get_singleton()->get_loaded_locales().has(this)) {
		OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_TRANSLATION_CHANGED);
	}
}

void Translation::add_message(const StringName &p_src_text, const StringName &p_xlated_text, const StringName &p_context) {
	translation_map[p_src_text] = p_xlated_text;
}

void Translation::add_plural_message(const StringName &p_src_text, const Vector<String> &p_plural_xlated_texts, const StringName &p_context) {
	WARN_PRINT("Translation class doesn't handle plural messages. Calling add_plural_message() on a Translation instance is probably a mistake. \nUse a derived Translation class that handles plurals, such as TranslationPO class");
	ERR_FAIL_COND_MSG(p_plural_xlated_texts.is_empty(), "Parameter vector p_plural_xlated_texts passed in is empty.");
	translation_map[p_src_text] = p_plural_xlated_texts[0];
}

StringName Translation::get_message(const StringName &p_src_text, const StringName &p_context) const {
	StringName ret;
	if (GDVIRTUAL_CALL(_get_message, p_src_text, p_context, ret)) {
		return ret;
	}

	if (p_context != StringName()) {
		WARN_PRINT("Translation class doesn't handle context. Using context in get_message() on a Translation instance is probably a mistake. \nUse a derived Translation class that handles context, such as TranslationPO class");
	}

	const Map<StringName, StringName>::Element *E = translation_map.find(p_src_text);
	if (!E) {
		return StringName();
	}

	return E->get();
}

StringName Translation::get_plural_message(const StringName &p_src_text, const StringName &p_plural_text, int p_n, const StringName &p_context) const {
	StringName ret;
	if (GDVIRTUAL_CALL(_get_plural_message, p_src_text, p_plural_text, p_n, p_context, ret)) {
		return ret;
	}

	WARN_PRINT("Translation class doesn't handle plural messages. Calling get_plural_message() on a Translation instance is probably a mistake. \nUse a derived Translation class that handles plurals, such as TranslationPO class");
	return get_message(p_src_text);
}

void Translation::erase_message(const StringName &p_src_text, const StringName &p_context) {
	if (p_context != StringName()) {
		WARN_PRINT("Translation class doesn't handle context. Using context in erase_message() on a Translation instance is probably a mistake. \nUse a derived Translation class that handles context, such as TranslationPO class");
	}

	translation_map.erase(p_src_text);
}

void Translation::get_message_list(List<StringName> *r_messages) const {
	for (const KeyValue<StringName, StringName> &E : translation_map) {
		r_messages->push_back(E.key);
	}
}

int Translation::get_message_count() const {
	return translation_map.size();
}

void Translation::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_locale", "locale"), &Translation::set_locale);
	ClassDB::bind_method(D_METHOD("get_locale"), &Translation::get_locale);
	ClassDB::bind_method(D_METHOD("add_message", "src_message", "xlated_message", "context"), &Translation::add_message, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("add_plural_message", "src_message", "xlated_messages", "context"), &Translation::add_plural_message, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("get_message", "src_message", "context"), &Translation::get_message, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("get_plural_message", "src_message", "src_plural_message", "n", "context"), &Translation::get_plural_message, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("erase_message", "src_message", "context"), &Translation::erase_message, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("get_message_list"), &Translation::_get_message_list);
	ClassDB::bind_method(D_METHOD("get_message_count"), &Translation::get_message_count);
	ClassDB::bind_method(D_METHOD("_set_messages"), &Translation::_set_messages);
	ClassDB::bind_method(D_METHOD("_get_messages"), &Translation::_get_messages);

	GDVIRTUAL_BIND(_get_plural_message, "src_message", "src_plural_message", "n", "context");
	GDVIRTUAL_BIND(_get_message, "src_message", "context");

	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "messages", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL), "_set_messages", "_get_messages");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "locale"), "set_locale", "get_locale");
}

///////////////////////////////////////////////

struct _character_accent_pair {
	const char32_t character;
	const char32_t *accented_character;
};

static _character_accent_pair _character_to_accented[] = {
	{ 'A', U"Å" },
	{ 'B', U"ß" },
	{ 'C', U"Ç" },
	{ 'D', U"Ð" },
	{ 'E', U"É" },
	{ 'F', U"F́" },
	{ 'G', U"Ĝ" },
	{ 'H', U"Ĥ" },
	{ 'I', U"Ĩ" },
	{ 'J', U"Ĵ" },
	{ 'K', U"ĸ" },
	{ 'L', U"Ł" },
	{ 'M', U"Ḿ" },
	{ 'N', U"й" },
	{ 'O', U"Ö" },
	{ 'P', U"Ṕ" },
	{ 'Q', U"Q́" },
	{ 'R', U"Ř" },
	{ 'S', U"Ŝ" },
	{ 'T', U"Ŧ" },
	{ 'U', U"Ũ" },
	{ 'V', U"Ṽ" },
	{ 'W', U"Ŵ" },
	{ 'X', U"X́" },
	{ 'Y', U"Ÿ" },
	{ 'Z', U"Ž" },
	{ 'a', U"á" },
	{ 'b', U"ḅ" },
	{ 'c', U"ć" },
	{ 'd', U"d́" },
	{ 'e', U"é" },
	{ 'f', U"f́" },
	{ 'g', U"ǵ" },
	{ 'h', U"h̀" },
	{ 'i', U"í" },
	{ 'j', U"ǰ" },
	{ 'k', U"ḱ" },
	{ 'l', U"ł" },
	{ 'm', U"m̀" },
	{ 'n', U"ή" },
	{ 'o', U"ô" },
	{ 'p', U"ṕ" },
	{ 'q', U"q́" },
	{ 'r', U"ŕ" },
	{ 's', U"š" },
	{ 't', U"ŧ" },
	{ 'u', U"ü" },
	{ 'v', U"ṽ" },
	{ 'w', U"ŵ" },
	{ 'x', U"x́" },
	{ 'y', U"ý" },
	{ 'z', U"ź" },
};

static _FORCE_INLINE_ bool is_upper_case(char32_t c) {
	return (c >= 'A' && c <= 'Z');
}

static _FORCE_INLINE_ bool is_lower_case(char32_t c) {
	return (c >= 'a' && c <= 'z');
}

Vector<TranslationServer::LocaleScriptInfo> TranslationServer::locale_script_info;

Map<String, String> TranslationServer::language_map;
Map<String, String> TranslationServer::script_map;
Map<String, String> TranslationServer::locale_rename_map;
Map<String, String> TranslationServer::country_name_map;
Map<String, String> TranslationServer::variant_map;
Map<String, String> TranslationServer::country_rename_map;

void TranslationServer::init_locale_info() {
	// Init locale info.
	language_map.clear();
	int idx = 0;
	while (language_list[idx][0] != nullptr) {
		language_map[language_list[idx][0]] = String::utf8(language_list[idx][1]);
		idx++;
	}

	// Init locale-script map.
	locale_script_info.clear();
	idx = 0;
	while (locale_scripts[idx][0] != nullptr) {
		LocaleScriptInfo info;
		info.name = locale_scripts[idx][0];
		info.script = locale_scripts[idx][1];
		info.default_country = locale_scripts[idx][2];
		Vector<String> supported_countries = String(locale_scripts[idx][3]).split(",", false);
		for (int i = 0; i < supported_countries.size(); i++) {
			info.supported_countries.insert(supported_countries[i]);
		}
		locale_script_info.push_back(info);
		idx++;
	}

	// Init supported script list.
	script_map.clear();
	idx = 0;
	while (script_list[idx][0] != nullptr) {
		script_map[script_list[idx][1]] = String::utf8(script_list[idx][0]);
		idx++;
	}

	// Init regional variant map.
	variant_map.clear();
	idx = 0;
	while (locale_variants[idx][0] != nullptr) {
		variant_map[locale_variants[idx][0]] = locale_variants[idx][1];
		idx++;
	}

	// Init locale renames.
	locale_rename_map.clear();
	idx = 0;
	while (locale_renames[idx][0] != nullptr) {
		if (!String(locale_renames[idx][1]).is_empty()) {
			locale_rename_map[locale_renames[idx][0]] = locale_renames[idx][1];
		}
		idx++;
	}

	// Init country names.
	country_name_map.clear();
	idx = 0;
	while (country_names[idx][0] != nullptr) {
		country_name_map[String(country_names[idx][0])] = String::utf8(country_names[idx][1]);
		idx++;
	}

	// Init country renames.
	country_rename_map.clear();
	idx = 0;
	while (country_renames[idx][0] != nullptr) {
		if (!String(country_renames[idx][1]).is_empty()) {
			country_rename_map[country_renames[idx][0]] = country_renames[idx][1];
		}
		idx++;
	}
}

String TranslationServer::standardize_locale(const String &p_locale) const {
	// Replaces '-' with '_' for macOS style locales.
	String univ_locale = p_locale.replace("-", "_");

	// Extract locale elements.
	String lang, script, country, variant;
	Vector<String> locale_elements = univ_locale.get_slice("@", 0).split("_");
	lang = locale_elements[0];
	if (locale_elements.size() >= 2) {
		if (locale_elements[1].length() == 4 && is_upper_case(locale_elements[1][0]) && is_lower_case(locale_elements[1][1]) && is_lower_case(locale_elements[1][2]) && is_lower_case(locale_elements[1][3])) {
			script = locale_elements[1];
		}
		if (locale_elements[1].length() == 2 && is_upper_case(locale_elements[1][0]) && is_upper_case(locale_elements[1][1])) {
			country = locale_elements[1];
		}
	}
	if (locale_elements.size() >= 3) {
		if (locale_elements[2].length() == 2 && is_upper_case(locale_elements[2][0]) && is_upper_case(locale_elements[2][1])) {
			country = locale_elements[2];
		} else if (variant_map.has(locale_elements[2].to_lower()) && variant_map[locale_elements[2].to_lower()] == lang) {
			variant = locale_elements[2].to_lower();
		}
	}
	if (locale_elements.size() >= 4) {
		if (variant_map.has(locale_elements[3].to_lower()) && variant_map[locale_elements[3].to_lower()] == lang) {
			variant = locale_elements[3].to_lower();
		}
	}

	// Try extract script and variant from the extra part.
	Vector<String> script_extra = univ_locale.get_slice("@", 1).split(";");
	for (int i = 0; i < script_extra.size(); i++) {
		if (script_extra[i].to_lower() == "cyrillic") {
			script = "Cyrl";
			break;
		} else if (script_extra[i].to_lower() == "latin") {
			script = "Latn";
			break;
		} else if (script_extra[i].to_lower() == "devanagari") {
			script = "Deva";
			break;
		} else if (variant_map.has(script_extra[i].to_lower()) && variant_map[script_extra[i].to_lower()] == lang) {
			variant = script_extra[i].to_lower();
		}
	}

	// Handles known non-ISO language names used e.g. on Windows.
	if (locale_rename_map.has(lang)) {
		lang = locale_rename_map[lang];
	}

	// Handle country renames.
	if (country_rename_map.has(country)) {
		country = country_rename_map[country];
	}

	// Remove unsupported script codes.
	if (!script_map.has(script)) {
		script = "";
	}

	// Add script code base on language and country codes for some ambiguous cases.
	if (script.is_empty()) {
		for (int i = 0; i < locale_script_info.size(); i++) {
			const LocaleScriptInfo &info = locale_script_info[i];
			if (info.name == lang) {
				if (country.is_empty() || info.supported_countries.has(country)) {
					script = info.script;
					break;
				}
			}
		}
	}
	if (!script.is_empty() && country.is_empty()) {
		// Add conntry code based on script for some ambiguous cases.
		for (int i = 0; i < locale_script_info.size(); i++) {
			const LocaleScriptInfo &info = locale_script_info[i];
			if (info.name == lang && info.script == script) {
				country = info.default_country;
				break;
			}
		}
	}

	// Combine results.
	String locale = lang;
	if (!script.is_empty()) {
		locale = locale + "_" + script;
	}
	if (!country.is_empty()) {
		locale = locale + "_" + country;
	}
	if (!variant.is_empty()) {
		locale = locale + "_" + variant;
	}
	return locale;
}

int TranslationServer::compare_locales(const String &p_locale_a, const String &p_locale_b) const {
	String locale_a = standardize_locale(p_locale_a);
	String locale_b = standardize_locale(p_locale_b);

	if (locale_a == locale_b) {
		// Exact match.
		return 10;
	}

	Vector<String> locale_a_elements = locale_a.split("_");
	Vector<String> locale_b_elements = locale_b.split("_");
	if (locale_a_elements[0] == locale_b_elements[0]) {
		// Matching language, both locales have extra parts.
		// Return number of matching elements.
		int matching_elements = 1;
		for (int i = 1; i < locale_a_elements.size(); i++) {
			for (int j = 1; j < locale_b_elements.size(); j++) {
				if (locale_a_elements[i] == locale_b_elements[j]) {
					matching_elements++;
				}
			}
		}
		return matching_elements;
	} else {
		// No match.
		return 0;
	}
}

String TranslationServer::get_locale_name(const String &p_locale) const {
	String locale = standardize_locale(p_locale);

	String lang, script, country;
	Vector<String> locale_elements = locale.split("_");
	lang = locale_elements[0];
	if (locale_elements.size() >= 2) {
		if (locale_elements[1].length() == 4 && is_upper_case(locale_elements[1][0]) && is_lower_case(locale_elements[1][1]) && is_lower_case(locale_elements[1][2]) && is_lower_case(locale_elements[1][3])) {
			script = locale_elements[1];
		}
		if (locale_elements[1].length() == 2 && is_upper_case(locale_elements[1][0]) && is_upper_case(locale_elements[1][1])) {
			country = locale_elements[1];
		}
	}
	if (locale_elements.size() >= 3) {
		if (locale_elements[2].length() == 2 && is_upper_case(locale_elements[2][0]) && is_upper_case(locale_elements[2][1])) {
			country = locale_elements[2];
		}
	}

	String name = language_map[lang];
	if (!script.is_empty()) {
		name = name + " (" + script_map[script] + ")";
	}
	if (!country.is_empty()) {
		name = name + ", " + country_name_map[country];
	}
	return name;
}

Vector<String> TranslationServer::get_all_languages() const {
	Vector<String> languages;

	for (const Map<String, String>::Element *E = language_map.front(); E; E = E->next()) {
		languages.push_back(E->key());
	}

	return languages;
}

String TranslationServer::get_language_name(const String &p_language) const {
	return language_map[p_language];
}

Vector<String> TranslationServer::get_all_scripts() const {
	Vector<String> scripts;

	for (const Map<String, String>::Element *E = script_map.front(); E; E = E->next()) {
		scripts.push_back(E->key());
	}

	return scripts;
}

String TranslationServer::get_script_name(const String &p_script) const {
	return script_map[p_script];
}

Vector<String> TranslationServer::get_all_countries() const {
	Vector<String> countries;

	for (const Map<String, String>::Element *E = country_name_map.front(); E; E = E->next()) {
		countries.push_back(E->key());
	}

	return countries;
}

String TranslationServer::get_country_name(const String &p_country) const {
	return country_name_map[p_country];
}

void TranslationServer::set_locale(const String &p_locale) {
	locale = standardize_locale(p_locale);

	if (OS::get_singleton()->get_main_loop()) {
		OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_TRANSLATION_CHANGED);
	}

	ResourceLoader::reload_translation_remaps();
}

String TranslationServer::get_locale() const {
	return locale;
}

Array TranslationServer::get_loaded_locales() const {
	Array locales;
	for (const Set<Ref<Translation>>::Element *E = translations.front(); E; E = E->next()) {
		const Ref<Translation> &t = E->get();
		ERR_FAIL_COND_V(t.is_null(), Array());
		String l = t->get_locale();

		locales.push_back(l);
	}

	return locales;
}

void TranslationServer::add_translation(const Ref<Translation> &p_translation) {
	translations.insert(p_translation);
}

void TranslationServer::remove_translation(const Ref<Translation> &p_translation) {
	translations.erase(p_translation);
}

Ref<Translation> TranslationServer::get_translation_object(const String &p_locale) {
	Ref<Translation> res;
	int best_score = 0;

	for (const Set<Ref<Translation>>::Element *E = translations.front(); E; E = E->next()) {
		const Ref<Translation> &t = E->get();
		ERR_FAIL_COND_V(t.is_null(), nullptr);
		String l = t->get_locale();

		int score = compare_locales(p_locale, l);
		if (score > 0 && score >= best_score) {
			res = t;
			best_score = score;
			if (score == 10) {
				break; // Exact match, skip the rest.
			}
		}
	}
	return res;
}

void TranslationServer::clear() {
	translations.clear();
}

StringName TranslationServer::translate(const StringName &p_message, const StringName &p_context) const {
	// Match given message against the translation catalog for the project locale.

	if (!enabled) {
		return p_message;
	}

	StringName res = _get_message_from_translations(p_message, p_context, locale, false);

	if (!res && fallback.length() >= 2) {
		res = _get_message_from_translations(p_message, p_context, fallback, false);
	}

	if (!res) {
		return pseudolocalization_enabled ? pseudolocalize(p_message) : p_message;
	}

	return pseudolocalization_enabled ? pseudolocalize(res) : res;
}

StringName TranslationServer::translate_plural(const StringName &p_message, const StringName &p_message_plural, int p_n, const StringName &p_context) const {
	if (!enabled) {
		if (p_n == 1) {
			return p_message;
		}
		return p_message_plural;
	}

	StringName res = _get_message_from_translations(p_message, p_context, locale, true, p_message_plural, p_n);

	if (!res && fallback.length() >= 2) {
		res = _get_message_from_translations(p_message, p_context, fallback, true, p_message_plural, p_n);
	}

	if (!res) {
		if (p_n == 1) {
			return p_message;
		}
		return p_message_plural;
	}

	return res;
}

StringName TranslationServer::_get_message_from_translations(const StringName &p_message, const StringName &p_context, const String &p_locale, bool plural, const String &p_message_plural, int p_n) const {
	StringName res;
	int best_score = 0;

	for (const Set<Ref<Translation>>::Element *E = translations.front(); E; E = E->next()) {
		const Ref<Translation> &t = E->get();
		ERR_FAIL_COND_V(t.is_null(), p_message);
		String l = t->get_locale();

		int score = compare_locales(p_locale, l);
		if (score > 0 && score >= best_score) {
			StringName r;
			if (!plural) {
				r = t->get_message(p_message, p_context);
			} else {
				r = t->get_plural_message(p_message, p_message_plural, p_n, p_context);
			}
			if (!r) {
				continue;
			}
			res = r;
			best_score = score;
			if (score == 10) {
				break; // Exact match, skip the rest.
			}
		}
	}

	return res;
}

TranslationServer *TranslationServer::singleton = nullptr;

bool TranslationServer::_load_translations(const String &p_from) {
	if (ProjectSettings::get_singleton()->has_setting(p_from)) {
		Vector<String> translations = ProjectSettings::get_singleton()->get(p_from);

		int tcount = translations.size();

		if (tcount) {
			const String *r = translations.ptr();

			for (int i = 0; i < tcount; i++) {
				Ref<Translation> tr = ResourceLoader::load(r[i]);
				if (tr.is_valid()) {
					add_translation(tr);
				}
			}
		}
		return true;
	}

	return false;
}

void TranslationServer::setup() {
	String test = GLOBAL_DEF("internationalization/locale/test", "");
	test = test.strip_edges();
	if (!test.is_empty()) {
		set_locale(test);
	} else {
		set_locale(OS::get_singleton()->get_locale());
	}

	fallback = GLOBAL_DEF("internationalization/locale/fallback", "en");
	pseudolocalization_enabled = GLOBAL_DEF("internationalization/pseudolocalization/use_pseudolocalization", false);
	pseudolocalization_accents_enabled = GLOBAL_DEF("internationalization/pseudolocalization/replace_with_accents", true);
	pseudolocalization_double_vowels_enabled = GLOBAL_DEF("internationalization/pseudolocalization/double_vowels", false);
	pseudolocalization_fake_bidi_enabled = GLOBAL_DEF("internationalization/pseudolocalization/fake_bidi", false);
	pseudolocalization_override_enabled = GLOBAL_DEF("internationalization/pseudolocalization/override", false);
	expansion_ratio = GLOBAL_DEF("internationalization/pseudolocalization/expansion_ratio", 0.0);
	pseudolocalization_prefix = GLOBAL_DEF("internationalization/pseudolocalization/prefix", "[");
	pseudolocalization_suffix = GLOBAL_DEF("internationalization/pseudolocalization/suffix", "]");
	pseudolocalization_skip_placeholders_enabled = GLOBAL_DEF("internationalization/pseudolocalization/skip_placeholders", true);

#ifdef TOOLS_ENABLED
	ProjectSettings::get_singleton()->set_custom_property_info("internationalization/locale/fallback", PropertyInfo(Variant::STRING, "internationalization/locale/fallback", PROPERTY_HINT_LOCALE_ID, ""));
#endif
}

void TranslationServer::set_tool_translation(const Ref<Translation> &p_translation) {
	tool_translation = p_translation;
}

Ref<Translation> TranslationServer::get_tool_translation() const {
	return tool_translation;
}

String TranslationServer::get_tool_locale() {
#ifdef TOOLS_ENABLED
	if (TranslationServer::get_singleton()->get_tool_translation().is_valid() && (Engine::get_singleton()->is_editor_hint() || Main::is_project_manager())) {
		return tool_translation->get_locale();
	} else {
#else
	{
#endif
		return get_locale();
	}
}

StringName TranslationServer::tool_translate(const StringName &p_message, const StringName &p_context) const {
	if (tool_translation.is_valid()) {
		StringName r = tool_translation->get_message(p_message, p_context);
		if (r) {
			return editor_pseudolocalization ? tool_pseudolocalize(r) : r;
		}
	}
	return editor_pseudolocalization ? tool_pseudolocalize(p_message) : p_message;
}

StringName TranslationServer::tool_translate_plural(const StringName &p_message, const StringName &p_message_plural, int p_n, const StringName &p_context) const {
	if (tool_translation.is_valid()) {
		StringName r = tool_translation->get_plural_message(p_message, p_message_plural, p_n, p_context);
		if (r) {
			return r;
		}
	}

	if (p_n == 1) {
		return p_message;
	}
	return p_message_plural;
}

void TranslationServer::set_doc_translation(const Ref<Translation> &p_translation) {
	doc_translation = p_translation;
}

StringName TranslationServer::doc_translate(const StringName &p_message, const StringName &p_context) const {
	if (doc_translation.is_valid()) {
		StringName r = doc_translation->get_message(p_message, p_context);
		if (r) {
			return r;
		}
	}
	return p_message;
}

StringName TranslationServer::doc_translate_plural(const StringName &p_message, const StringName &p_message_plural, int p_n, const StringName &p_context) const {
	if (doc_translation.is_valid()) {
		StringName r = doc_translation->get_plural_message(p_message, p_message_plural, p_n, p_context);
		if (r) {
			return r;
		}
	}

	if (p_n == 1) {
		return p_message;
	}
	return p_message_plural;
}

bool TranslationServer::is_pseudolocalization_enabled() const {
	return pseudolocalization_enabled;
}

void TranslationServer::set_pseudolocalization_enabled(bool p_enabled) {
	pseudolocalization_enabled = p_enabled;

	if (OS::get_singleton()->get_main_loop()) {
		OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_TRANSLATION_CHANGED);
	}
	ResourceLoader::reload_translation_remaps();
}

void TranslationServer::set_editor_pseudolocalization(bool p_enabled) {
	editor_pseudolocalization = p_enabled;
}

void TranslationServer::reload_pseudolocalization() {
	pseudolocalization_accents_enabled = GLOBAL_GET("internationalization/pseudolocalization/replace_with_accents");
	pseudolocalization_double_vowels_enabled = GLOBAL_GET("internationalization/pseudolocalization/double_vowels");
	pseudolocalization_fake_bidi_enabled = GLOBAL_GET("internationalization/pseudolocalization/fake_bidi");
	pseudolocalization_override_enabled = GLOBAL_GET("internationalization/pseudolocalization/override");
	expansion_ratio = GLOBAL_GET("internationalization/pseudolocalization/expansion_ratio");
	pseudolocalization_prefix = GLOBAL_GET("internationalization/pseudolocalization/prefix");
	pseudolocalization_suffix = GLOBAL_GET("internationalization/pseudolocalization/suffix");
	pseudolocalization_skip_placeholders_enabled = GLOBAL_GET("internationalization/pseudolocalization/skip_placeholders");

	if (OS::get_singleton()->get_main_loop()) {
		OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_TRANSLATION_CHANGED);
	}
	ResourceLoader::reload_translation_remaps();
}

StringName TranslationServer::pseudolocalize(const StringName &p_message) const {
	String message = p_message;
	int length = message.length();
	if (pseudolocalization_override_enabled) {
		message = get_override_string(message);
	}

	if (pseudolocalization_double_vowels_enabled) {
		message = double_vowels(message);
	}

	if (pseudolocalization_accents_enabled) {
		message = replace_with_accented_string(message);
	}

	if (pseudolocalization_fake_bidi_enabled) {
		message = wrap_with_fakebidi_characters(message);
	}

	StringName res = add_padding(message, length);
	return res;
}

StringName TranslationServer::tool_pseudolocalize(const StringName &p_message) const {
	String message = p_message;
	message = double_vowels(message);
	message = replace_with_accented_string(message);
	StringName res = "[!!! " + message + " !!!]";
	return res;
}

String TranslationServer::get_override_string(String &p_message) const {
	String res;
	for (int i = 0; i < p_message.size(); i++) {
		if (pseudolocalization_skip_placeholders_enabled && is_placeholder(p_message, i)) {
			res += p_message[i];
			res += p_message[i + 1];
			i++;
			continue;
		}
		res += '*';
	}
	return res;
}

String TranslationServer::double_vowels(String &p_message) const {
	String res;
	for (int i = 0; i < p_message.size(); i++) {
		if (pseudolocalization_skip_placeholders_enabled && is_placeholder(p_message, i)) {
			res += p_message[i];
			res += p_message[i + 1];
			i++;
			continue;
		}
		res += p_message[i];
		if (p_message[i] == 'a' || p_message[i] == 'e' || p_message[i] == 'i' || p_message[i] == 'o' || p_message[i] == 'u' ||
				p_message[i] == 'A' || p_message[i] == 'E' || p_message[i] == 'I' || p_message[i] == 'O' || p_message[i] == 'U') {
			res += p_message[i];
		}
	}
	return res;
};

String TranslationServer::replace_with_accented_string(String &p_message) const {
	String res;
	for (int i = 0; i < p_message.size(); i++) {
		if (pseudolocalization_skip_placeholders_enabled && is_placeholder(p_message, i)) {
			res += p_message[i];
			res += p_message[i + 1];
			i++;
			continue;
		}
		const char32_t *accented = get_accented_version(p_message[i]);
		if (accented) {
			res += accented;
		} else {
			res += p_message[i];
		}
	}
	return res;
}

String TranslationServer::wrap_with_fakebidi_characters(String &p_message) const {
	String res;
	char32_t fakebidiprefix = U'\u202e';
	char32_t fakebidisuffix = U'\u202c';
	res += fakebidiprefix;
	// The fake bidi unicode gets popped at every newline so pushing it back at every newline.
	for (int i = 0; i < p_message.size(); i++) {
		if (p_message[i] == '\n') {
			res += fakebidisuffix;
			res += p_message[i];
			res += fakebidiprefix;
		} else if (pseudolocalization_skip_placeholders_enabled && is_placeholder(p_message, i)) {
			res += fakebidisuffix;
			res += p_message[i];
			res += p_message[i + 1];
			res += fakebidiprefix;
			i++;
		} else {
			res += p_message[i];
		}
	}
	res += fakebidisuffix;
	return res;
}

String TranslationServer::add_padding(String &p_message, int p_length) const {
	String res;
	String prefix = pseudolocalization_prefix;
	String suffix;
	for (int i = 0; i < p_length * expansion_ratio / 2; i++) {
		prefix += "_";
		suffix += "_";
	}
	suffix += pseudolocalization_suffix;
	res += prefix;
	res += p_message;
	res += suffix;
	return res;
}

const char32_t *TranslationServer::get_accented_version(char32_t p_character) const {
	if (!((p_character >= 'a' && p_character <= 'z') || (p_character >= 'A' && p_character <= 'Z'))) {
		return nullptr;
	}

	for (unsigned int i = 0; i < sizeof(_character_to_accented) / sizeof(_character_to_accented[0]); i++) {
		if (_character_to_accented[i].character == p_character) {
			return _character_to_accented[i].accented_character;
		}
	}

	return nullptr;
}

bool TranslationServer::is_placeholder(String &p_message, int p_index) const {
	return p_message[p_index] == '%' && p_index < p_message.size() - 1 &&
			(p_message[p_index + 1] == 's' || p_message[p_index + 1] == 'c' || p_message[p_index + 1] == 'd' ||
					p_message[p_index + 1] == 'o' || p_message[p_index + 1] == 'x' || p_message[p_index + 1] == 'X' || p_message[p_index + 1] == 'f');
}

void TranslationServer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_locale", "locale"), &TranslationServer::set_locale);
	ClassDB::bind_method(D_METHOD("get_locale"), &TranslationServer::get_locale);

	ClassDB::bind_method(D_METHOD("compare_locales", "locale_a", "locale_b"), &TranslationServer::compare_locales);
	ClassDB::bind_method(D_METHOD("standardize_locale", "locale"), &TranslationServer::standardize_locale);

	ClassDB::bind_method(D_METHOD("get_all_languages"), &TranslationServer::get_all_languages);
	ClassDB::bind_method(D_METHOD("get_language_name", "language"), &TranslationServer::get_language_name);

	ClassDB::bind_method(D_METHOD("get_all_scripts"), &TranslationServer::get_all_scripts);
	ClassDB::bind_method(D_METHOD("get_script_name", "script"), &TranslationServer::get_script_name);

	ClassDB::bind_method(D_METHOD("get_all_countries"), &TranslationServer::get_all_countries);
	ClassDB::bind_method(D_METHOD("get_country_name", "country"), &TranslationServer::get_country_name);

	ClassDB::bind_method(D_METHOD("get_locale_name", "locale"), &TranslationServer::get_locale_name);

	ClassDB::bind_method(D_METHOD("translate", "message", "context"), &TranslationServer::translate, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("translate_plural", "message", "plural_message", "n", "context"), &TranslationServer::translate_plural, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("add_translation", "translation"), &TranslationServer::add_translation);
	ClassDB::bind_method(D_METHOD("remove_translation", "translation"), &TranslationServer::remove_translation);
	ClassDB::bind_method(D_METHOD("get_translation_object", "locale"), &TranslationServer::get_translation_object);

	ClassDB::bind_method(D_METHOD("clear"), &TranslationServer::clear);

	ClassDB::bind_method(D_METHOD("get_loaded_locales"), &TranslationServer::get_loaded_locales);

	ClassDB::bind_method(D_METHOD("is_pseudolocalization_enabled"), &TranslationServer::is_pseudolocalization_enabled);
	ClassDB::bind_method(D_METHOD("set_pseudolocalization_enabled", "enabled"), &TranslationServer::set_pseudolocalization_enabled);
	ClassDB::bind_method(D_METHOD("reload_pseudolocalization"), &TranslationServer::reload_pseudolocalization);
	ClassDB::bind_method(D_METHOD("pseudolocalize", "message"), &TranslationServer::pseudolocalize);
	ADD_PROPERTY(PropertyInfo(Variant::Type::BOOL, "pseudolocalization_enabled"), "set_pseudolocalization_enabled", "is_pseudolocalization_enabled");
}

void TranslationServer::load_translations() {
	String locale = get_locale();
	_load_translations("internationalization/locale/translations"); //all
	_load_translations("internationalization/locale/translations_" + locale.substr(0, 2));

	if (locale.substr(0, 2) != locale) {
		_load_translations("internationalization/locale/translations_" + locale);
	}
}

TranslationServer::TranslationServer() {
	singleton = this;
	init_locale_info();
}
