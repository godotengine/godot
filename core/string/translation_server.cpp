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

#include "translation_server.h"
#include "translation_server.compat.inc"

#include "core/config/project_settings.h"
#include "core/io/resource_loader.h"
#include "core/os/os.h"
#include "core/string/locales.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_settings.h"
#include "main/main.h"
#endif

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

Vector<TranslationServer::LocaleScriptInfo> TranslationServer::locale_script_info;

HashMap<String, String> TranslationServer::language_map;
HashMap<String, String> TranslationServer::script_map;
HashMap<String, String> TranslationServer::locale_rename_map;
HashMap<String, String> TranslationServer::country_name_map;
HashMap<String, String> TranslationServer::variant_map;
HashMap<String, String> TranslationServer::country_rename_map;

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

void TranslationServer::init_num_systems() {
	{
		NumSystemData ar;
		ar.lang.insert(StringName("ar")); // Arabic
		ar.lang.insert(StringName("ar_AE"));
		ar.lang.insert(StringName("ar_BH"));
		ar.lang.insert(StringName("ar_DJ"));
		ar.lang.insert(StringName("ar_EG"));
		ar.lang.insert(StringName("ar_ER"));
		ar.lang.insert(StringName("ar_IL"));
		ar.lang.insert(StringName("ar_IQ"));
		ar.lang.insert(StringName("ar_JO"));
		ar.lang.insert(StringName("ar_KM"));
		ar.lang.insert(StringName("ar_KW"));
		ar.lang.insert(StringName("ar_LB"));
		ar.lang.insert(StringName("ar_MR"));
		ar.lang.insert(StringName("ar_OM"));
		ar.lang.insert(StringName("ar_PS"));
		ar.lang.insert(StringName("ar_QA"));
		ar.lang.insert(StringName("ar_SA"));
		ar.lang.insert(StringName("ar_SD"));
		ar.lang.insert(StringName("ar_SO"));
		ar.lang.insert(StringName("ar_SS"));
		ar.lang.insert(StringName("ar_SY"));
		ar.lang.insert(StringName("ar_TD"));
		ar.lang.insert(StringName("ar_YE"));
		ar.lang.insert(StringName("ckb")); // Central Kurdish
		ar.lang.insert(StringName("ckb_IQ"));
		ar.lang.insert(StringName("ckb_IR"));
		ar.lang.insert(StringName("sd")); // Sindhi
		ar.lang.insert(StringName("sd_PK"));
		ar.lang.insert(StringName("sd_Arab"));
		ar.lang.insert(StringName("sd_Arab_PK"));
		ar.digits = U"٠١٢٣٤٥٦٧٨٩٫";
		ar.percent_sign = U"٪";
		ar.exp = U"اس";
		num_systems.push_back(ar);
	}

	// Persian and Urdu numerals.
	{
		NumSystemData pr;
		pr.lang.insert(StringName("fa")); // Persian
		pr.lang.insert(StringName("fa_AF"));
		pr.lang.insert(StringName("fa_IR"));
		pr.lang.insert(StringName("ks")); // Kashmiri
		pr.lang.insert(StringName("ks_IN"));
		pr.lang.insert(StringName("ks_Arab"));
		pr.lang.insert(StringName("ks_Arab_IN"));
		pr.lang.insert(StringName("lrc")); // Northern Luri
		pr.lang.insert(StringName("lrc_IQ"));
		pr.lang.insert(StringName("lrc_IR"));
		pr.lang.insert(StringName("mzn")); // Mazanderani
		pr.lang.insert(StringName("mzn_IR"));
		pr.lang.insert(StringName("pa_PK")); // Panjabi
		pr.lang.insert(StringName("pa_Arab"));
		pr.lang.insert(StringName("pa_Arab_PK"));
		pr.lang.insert(StringName("ps")); // Pushto
		pr.lang.insert(StringName("ps_AF"));
		pr.lang.insert(StringName("ps_PK"));
		pr.lang.insert(StringName("ur_IN")); // Urdu
		pr.lang.insert(StringName("uz_AF")); // Uzbek
		pr.lang.insert(StringName("uz_Arab"));
		pr.lang.insert(StringName("uz_Arab_AF"));
		pr.digits = U"۰۱۲۳۴۵۶۷۸۹٫";
		pr.percent_sign = U"٪";
		pr.exp = U"اس";
		num_systems.push_back(pr);
	}

	// Bengali numerals.
	{
		NumSystemData bn;
		bn.lang.insert(StringName("as")); // Assamese
		bn.lang.insert(StringName("as_IN"));
		bn.lang.insert(StringName("bn")); // Bengali
		bn.lang.insert(StringName("bn_BD"));
		bn.lang.insert(StringName("bn_IN"));
		bn.lang.insert(StringName("mni")); // Manipuri
		bn.lang.insert(StringName("mni_IN"));
		bn.lang.insert(StringName("mni_Beng"));
		bn.lang.insert(StringName("mni_Beng_IN"));
		bn.digits = U"০১২৩৪৫৬৭৮৯.";
		bn.percent_sign = U"%";
		bn.exp = U"e";
		num_systems.push_back(bn);
	}

	// Devanagari numerals.
	{
		NumSystemData mr;
		mr.lang.insert(StringName("mr")); // Marathi
		mr.lang.insert(StringName("mr_IN"));
		mr.lang.insert(StringName("ne")); // Nepali
		mr.lang.insert(StringName("ne_IN"));
		mr.lang.insert(StringName("ne_NP"));
		mr.lang.insert(StringName("sa")); // Sanskrit
		mr.lang.insert(StringName("sa_IN"));
		mr.digits = U"०१२३४५६७८९.";
		mr.percent_sign = U"%";
		mr.exp = U"e";
		num_systems.push_back(mr);
	}

	// Dzongkha numerals.
	{
		NumSystemData dz;
		dz.lang.insert(StringName("dz")); // Dzongkha
		dz.lang.insert(StringName("dz_BT"));
		dz.digits = U"༠༡༢༣༤༥༦༧༨༩.";
		dz.percent_sign = U"%";
		dz.exp = U"e";
		num_systems.push_back(dz);
	}

	// Santali numerals.
	{
		NumSystemData sat;
		sat.lang.insert(StringName("sat")); // Santali
		sat.lang.insert(StringName("sat_IN"));
		sat.lang.insert(StringName("sat_Olck"));
		sat.lang.insert(StringName("sat_Olck_IN"));
		sat.digits = U"᱐᱑᱒᱓᱔᱕᱖᱗᱘᱙.";
		sat.percent_sign = U"%";
		sat.exp = U"e";
		num_systems.push_back(sat);
	}

	// Burmese numerals.
	{
		NumSystemData my;
		my.lang.insert(StringName("my")); // Burmese
		my.lang.insert(StringName("my_MM"));
		my.digits = U"၀၁၂၃၄၅၆၇၈၉.";
		my.percent_sign = U"%";
		my.exp = U"e";
		num_systems.push_back(my);
	}

	// Chakma numerals.
	{
		NumSystemData ccp;
		ccp.lang.insert(StringName("ccp")); // Chakma
		ccp.lang.insert(StringName("ccp_BD"));
		ccp.lang.insert(StringName("ccp_IN"));
		ccp.digits = U"𑄶𑄷𑄸𑄹𑄺𑄻𑄼𑄽𑄾𑄿.";
		ccp.percent_sign = U"%";
		ccp.exp = U"e";
		num_systems.push_back(ccp);
	}

	// Adlam numerals.
	{
		NumSystemData ff;
		ff.lang.insert(StringName("ff")); // Fulah
		ff.lang.insert(StringName("ff_Adlm_BF"));
		ff.lang.insert(StringName("ff_Adlm_CM"));
		ff.lang.insert(StringName("ff_Adlm_GH"));
		ff.lang.insert(StringName("ff_Adlm_GM"));
		ff.lang.insert(StringName("ff_Adlm_GN"));
		ff.lang.insert(StringName("ff_Adlm_GW"));
		ff.lang.insert(StringName("ff_Adlm_LR"));
		ff.lang.insert(StringName("ff_Adlm_MR"));
		ff.lang.insert(StringName("ff_Adlm_NE"));
		ff.lang.insert(StringName("ff_Adlm_NG"));
		ff.lang.insert(StringName("ff_Adlm_SL"));
		ff.lang.insert(StringName("ff_Adlm_SN"));
		ff.digits = U"𞥐𞥑𞥒𞥓𞥔𞥕𞥖𞥗𞥘𞥙.";
		ff.percent_sign = U"%";
		ff.exp = U"e";
		num_systems.push_back(ff);
	}

	// Pseudolocalization.
	{
		NumSystemData ps;
		ps.digits = U"ѻїԇӡһѵδґβя'";
		ps.percent_sign = U"ϖ";
		ps.exp = U"Σ";
		num_systems.push_back(ps);
	}
}

String TranslationServer::standardize_locale(const String &p_locale) const {
	return _standardize_locale(p_locale, false);
}

String TranslationServer::_standardize_locale(const String &p_locale, bool p_add_defaults) const {
	// Replaces '-' with '_' for macOS style locales.
	String univ_locale = p_locale.replace("-", "_");

	// Extract locale elements.
	String lang_name, script_name, country_name, variant_name;
	Vector<String> locale_elements = univ_locale.get_slice("@", 0).split("_");
	lang_name = locale_elements[0];
	if (locale_elements.size() >= 2) {
		if (locale_elements[1].length() == 4 && is_ascii_upper_case(locale_elements[1][0]) && is_ascii_lower_case(locale_elements[1][1]) && is_ascii_lower_case(locale_elements[1][2]) && is_ascii_lower_case(locale_elements[1][3])) {
			script_name = locale_elements[1];
		}
		if (locale_elements[1].length() == 2 && is_ascii_upper_case(locale_elements[1][0]) && is_ascii_upper_case(locale_elements[1][1])) {
			country_name = locale_elements[1];
		}
	}
	if (locale_elements.size() >= 3) {
		if (locale_elements[2].length() == 2 && is_ascii_upper_case(locale_elements[2][0]) && is_ascii_upper_case(locale_elements[2][1])) {
			country_name = locale_elements[2];
		} else if (variant_map.has(locale_elements[2].to_lower()) && variant_map[locale_elements[2].to_lower()] == lang_name) {
			variant_name = locale_elements[2].to_lower();
		}
	}
	if (locale_elements.size() >= 4) {
		if (variant_map.has(locale_elements[3].to_lower()) && variant_map[locale_elements[3].to_lower()] == lang_name) {
			variant_name = locale_elements[3].to_lower();
		}
	}

	// Try extract script and variant from the extra part.
	Vector<String> script_extra = univ_locale.get_slice("@", 1).split(";");
	for (int i = 0; i < script_extra.size(); i++) {
		if (script_extra[i].to_lower() == "cyrillic") {
			script_name = "Cyrl";
			break;
		} else if (script_extra[i].to_lower() == "latin") {
			script_name = "Latn";
			break;
		} else if (script_extra[i].to_lower() == "devanagari") {
			script_name = "Deva";
			break;
		} else if (variant_map.has(script_extra[i].to_lower()) && variant_map[script_extra[i].to_lower()] == lang_name) {
			variant_name = script_extra[i].to_lower();
		}
	}

	// Handles known non-ISO language names used e.g. on Windows.
	if (locale_rename_map.has(lang_name)) {
		lang_name = locale_rename_map[lang_name];
	}

	// Handle country renames.
	if (country_rename_map.has(country_name)) {
		country_name = country_rename_map[country_name];
	}

	// Remove unsupported script codes.
	if (!script_map.has(script_name)) {
		script_name = "";
	}

	// Add script code base on language and country codes for some ambiguous cases.
	if (p_add_defaults) {
		if (script_name.is_empty()) {
			for (int i = 0; i < locale_script_info.size(); i++) {
				const LocaleScriptInfo &info = locale_script_info[i];
				if (info.name == lang_name) {
					if (country_name.is_empty() || info.supported_countries.has(country_name)) {
						script_name = info.script;
						break;
					}
				}
			}
		}
		if (!script_name.is_empty() && country_name.is_empty()) {
			// Add conntry code based on script for some ambiguous cases.
			for (int i = 0; i < locale_script_info.size(); i++) {
				const LocaleScriptInfo &info = locale_script_info[i];
				if (info.name == lang_name && info.script == script_name) {
					country_name = info.default_country;
					break;
				}
			}
		}
	}

	// Combine results.
	String out = lang_name;
	if (!script_name.is_empty()) {
		out = out + "_" + script_name;
	}
	if (!country_name.is_empty()) {
		out = out + "_" + country_name;
	}
	if (!variant_name.is_empty()) {
		out = out + "_" + variant_name;
	}
	return out;
}

int TranslationServer::compare_locales(const String &p_locale_a, const String &p_locale_b) const {
	String locale_a = _standardize_locale(p_locale_a, true);
	String locale_b = _standardize_locale(p_locale_b, true);

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
	String lang_name, script_name, country_name;
	Vector<String> locale_elements = standardize_locale(p_locale).split("_");
	lang_name = locale_elements[0];
	if (locale_elements.size() >= 2) {
		if (locale_elements[1].length() == 4 && is_ascii_upper_case(locale_elements[1][0]) && is_ascii_lower_case(locale_elements[1][1]) && is_ascii_lower_case(locale_elements[1][2]) && is_ascii_lower_case(locale_elements[1][3])) {
			script_name = locale_elements[1];
		}
		if (locale_elements[1].length() == 2 && is_ascii_upper_case(locale_elements[1][0]) && is_ascii_upper_case(locale_elements[1][1])) {
			country_name = locale_elements[1];
		}
	}
	if (locale_elements.size() >= 3) {
		if (locale_elements[2].length() == 2 && is_ascii_upper_case(locale_elements[2][0]) && is_ascii_upper_case(locale_elements[2][1])) {
			country_name = locale_elements[2];
		}
	}

	String name = language_map[lang_name];
	if (!script_name.is_empty()) {
		name = name + " (" + script_map[script_name] + ")";
	}
	if (!country_name.is_empty()) {
		name = name + ", " + country_name_map[country_name];
	}
	return name;
}

Vector<String> TranslationServer::get_all_languages() const {
	Vector<String> languages;

	for (const KeyValue<String, String> &E : language_map) {
		languages.push_back(E.key);
	}

	return languages;
}

String TranslationServer::get_language_name(const String &p_language) const {
	return language_map[p_language];
}

Vector<String> TranslationServer::get_all_scripts() const {
	Vector<String> scripts;

	for (const KeyValue<String, String> &E : script_map) {
		scripts.push_back(E.key);
	}

	return scripts;
}

String TranslationServer::get_script_name(const String &p_script) const {
	return script_map[p_script];
}

Vector<String> TranslationServer::get_all_countries() const {
	Vector<String> countries;

	for (const KeyValue<String, String> &E : country_name_map) {
		countries.push_back(E.key);
	}

	return countries;
}

String TranslationServer::get_country_name(const String &p_country) const {
	return country_name_map[p_country];
}

void TranslationServer::set_locale(const String &p_locale) {
	String new_locale = standardize_locale(p_locale);
	if (locale == new_locale) {
		return;
	}

	locale = new_locale;
	ResourceLoader::reload_translation_remaps();

	if (OS::get_singleton()->get_main_loop()) {
		OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_TRANSLATION_CHANGED);
	}
}

String TranslationServer::get_locale() const {
	return locale;
}

PackedStringArray TranslationServer::get_loaded_locales() const {
	PackedStringArray locales;
	for (const Ref<Translation> &E : translations) {
		const Ref<Translation> &t = E;
		ERR_FAIL_COND_V(t.is_null(), PackedStringArray());
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

	for (const Ref<Translation> &E : translations) {
		const Ref<Translation> &t = E;
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

	if (p_message == StringName()) {
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

	if (p_message == StringName() && p_message_plural == StringName()) {
		return p_message;
	}

	StringName res = _get_message_from_translations(p_message, p_context, locale, true, p_message_plural, p_n);

	if (!res && fallback.length() >= 2) {
		res = _get_message_from_translations(p_message, p_context, fallback, true, p_message_plural, p_n);
	}

	if (!res) {
		if (p_n == 1) {
			return pseudolocalization_enabled ? pseudolocalize(p_message) : p_message;
		}
		return pseudolocalization_enabled ? pseudolocalize(p_message_plural) : p_message_plural;
	}

	return pseudolocalization_enabled ? pseudolocalize(res) : res;
}

StringName TranslationServer::_get_message_from_translations(const StringName &p_message, const StringName &p_context, const String &p_locale, bool plural, const String &p_message_plural, int p_n) const {
	StringName res;
	int best_score = 0;

	for (const Ref<Translation> &E : translations) {
		const Ref<Translation> &t = E;
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
		const Vector<String> &translation_names = GLOBAL_GET(p_from);

		int tcount = translation_names.size();

		if (tcount) {
			const String *r = translation_names.ptr();

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
	pseudolocalization_numbers_enabled = GLOBAL_DEF("internationalization/pseudolocalization/numbers", true);
	expansion_ratio = GLOBAL_DEF("internationalization/pseudolocalization/expansion_ratio", 0.0);
	pseudolocalization_prefix = GLOBAL_DEF("internationalization/pseudolocalization/prefix", "[");
	pseudolocalization_suffix = GLOBAL_DEF("internationalization/pseudolocalization/suffix", "]");
	pseudolocalization_skip_placeholders_enabled = GLOBAL_DEF("internationalization/pseudolocalization/skip_placeholders", true);

#ifdef TOOLS_ENABLED
	ProjectSettings::get_singleton()->set_custom_property_info(PropertyInfo(Variant::STRING, "internationalization/locale/fallback", PROPERTY_HINT_LOCALE_ID, ""));
#endif
}

String TranslationServer::format_number(const String &p_number, const String &p_language) const {
	StringName lang = p_language.is_empty() ? get_locale() : p_language;
	String res = p_number;
	if (pseudolocalization_enabled && pseudolocalization_numbers_enabled) {
		res.replace("e", num_systems[num_systems.size() - 1].exp);
		res.replace("E", num_systems[num_systems.size() - 1].exp);
		char32_t *data = res.ptrw();
		for (int j = 0; j < res.length(); j++) {
			if (data[j] >= 0x30 && data[j] <= 0x39) {
				data[j] = num_systems[num_systems.size() - 1].digits[data[j] - 0x30];
			} else if (data[j] == '.' || data[j] == ',') {
				data[j] = num_systems[num_systems.size() - 1].digits[10];
			}
		}
		return pseudolocalization_prefix + res + pseudolocalization_suffix;
	} else {
		for (int i = 0; i < num_systems.size() - 1; i++) {
			if (num_systems[i].lang.has(lang)) {
				if (num_systems[i].digits.is_empty()) {
					return p_number;
				}
				res.replace("e", num_systems[i].exp);
				res.replace("E", num_systems[i].exp);
				char32_t *data = res.ptrw();
				for (int j = 0; j < res.length(); j++) {
					if (data[j] >= 0x30 && data[j] <= 0x39) {
						data[j] = num_systems[i].digits[data[j] - 0x30];
					} else if (data[j] == '.' || data[j] == ',') {
						data[j] = num_systems[i].digits[10];
					}
				}
				break;
			}
		}
	}
	return res;
}

String TranslationServer::tool_format_number(const String &p_number) const {
	StringName lang = get_tool_locale();
	String res = p_number;
	if (ed_pseudolocalization_enabled && ed_pseudolocalization_numbers_enabled) {
		res.replace("e", num_systems[num_systems.size() - 1].exp);
		res.replace("E", num_systems[num_systems.size() - 1].exp);
		char32_t *data = res.ptrw();
		for (int j = 0; j < res.length(); j++) {
			if (data[j] >= 0x30 && data[j] <= 0x39) {
				data[j] = num_systems[num_systems.size() - 1].digits[data[j] - 0x30];
			} else if (data[j] == '.' || data[j] == ',') {
				data[j] = num_systems[num_systems.size() - 1].digits[10];
			}
		}
		return ed_pseudolocalization_prefix + res + ed_pseudolocalization_suffix;
	} else {
		for (int i = 0; i < num_systems.size() - 1; i++) {
			if (num_systems[i].lang.has(lang)) {
				if (num_systems[i].digits.is_empty()) {
					return p_number;
				}
				res.replace("e", num_systems[i].exp);
				res.replace("E", num_systems[i].exp);
				char32_t *data = res.ptrw();
				for (int j = 0; j < res.length(); j++) {
					if (data[j] >= 0x30 && data[j] <= 0x39) {
						data[j] = num_systems[i].digits[data[j] - 0x30];
					} else if (data[j] == '.' || data[j] == ',') {
						data[j] = num_systems[i].digits[10];
					}
				}
				break;
			}
		}
	}
	return res;
}

String TranslationServer::get_percent_sign(const String &p_language) const {
	if (pseudolocalization_enabled && pseudolocalization_numbers_enabled) {
		return pseudolocalization_prefix + num_systems[num_systems.size() - 1].percent_sign + pseudolocalization_suffix;
	} else {
		const StringName lang = (p_language.is_empty()) ? get_locale() : p_language;

		for (int i = 0; i < num_systems.size(); i++) {
			if (num_systems[i].lang.has(lang)) {
				if (num_systems[i].percent_sign.is_empty()) {
					return "%";
				}
				return num_systems[i].percent_sign;
			}
		}
		return "%";
	}
}

String TranslationServer::get_tool_percent_sign() const {
	if (ed_pseudolocalization_enabled && ed_pseudolocalization_numbers_enabled) {
		return ed_pseudolocalization_prefix + num_systems[num_systems.size() - 1].percent_sign + ed_pseudolocalization_suffix;
	} else {
		const StringName lang = get_tool_locale();

		for (int i = 0; i < num_systems.size(); i++) {
			if (num_systems[i].lang.has(lang)) {
				if (num_systems[i].percent_sign.is_empty()) {
					return "%";
				}
				return num_systems[i].percent_sign;
			}
		}
		return "%";
	}
}

String TranslationServer::parse_number(const String &p_string, const String &p_language) const {
	String res = p_string;
	if (pseudolocalization_enabled && pseudolocalization_numbers_enabled) {
		res = res.trim_prefix(pseudolocalization_prefix).trim_suffix(pseudolocalization_suffix);
		res.replace(num_systems[num_systems.size() - 1].exp, "e");
		char32_t *data = res.ptrw();
		for (int j = 0; j < res.length(); j++) {
			if (data[j] == num_systems[num_systems.size() - 1].digits[10]) {
				data[j] = '.';
			} else {
				for (int k = 0; k < 10; k++) {
					if (data[j] == num_systems[num_systems.size() - 1].digits[k]) {
						data[j] = 0x30 + k;
					}
				}
			}
		}
	} else {
		StringName lang = p_language.is_empty() ? get_locale() : p_language;
		for (int i = 0; i < num_systems.size() - 1; i++) {
			if (num_systems[i].lang.has(lang)) {
				if (num_systems[i].digits.is_empty()) {
					return p_string;
				}
				res.replace(num_systems[i].exp, "e");
				char32_t *data = res.ptrw();
				for (int j = 0; j < res.length(); j++) {
					if (data[j] == num_systems[i].digits[10]) {
						data[j] = '.';
					} else {
						for (int k = 0; k < 10; k++) {
							if (data[j] == num_systems[i].digits[k]) {
								data[j] = 0x30 + k;
							}
						}
					}
				}
				break;
			}
		}
	}
	return res;
}

String TranslationServer::tool_parse_number(const String &p_string) const {
	StringName lang = get_tool_locale();

	String res = p_string;
	if (ed_pseudolocalization_enabled && ed_pseudolocalization_numbers_enabled) {
		res = res.trim_prefix(ed_pseudolocalization_prefix).trim_suffix(ed_pseudolocalization_suffix);
		res.replace(num_systems[num_systems.size() - 1].exp, "e");
		char32_t *data = res.ptrw();
		for (int j = 0; j < res.length(); j++) {
			if (data[j] == num_systems[num_systems.size() - 1].digits[10]) {
				data[j] = '.';
			} else {
				for (int k = 0; k < 10; k++) {
					if (data[j] == num_systems[num_systems.size() - 1].digits[k]) {
						data[j] = 0x30 + k;
					}
				}
			}
		}
	} else {
		for (int i = 0; i < num_systems.size() - 1; i++) {
			if (num_systems[i].lang.has(lang)) {
				if (num_systems[i].digits.is_empty()) {
					return p_string;
				}
				res.replace(num_systems[i].exp, "e");
				char32_t *data = res.ptrw();
				for (int j = 0; j < res.length(); j++) {
					if (data[j] == num_systems[i].digits[10]) {
						data[j] = '.';
					} else {
						for (int k = 0; k < 10; k++) {
							if (data[j] == num_systems[i].digits[k]) {
								data[j] = 0x30 + k;
							}
						}
					}
				}
				break;
			}
		}
	}
	return res;
}

void TranslationServer::set_tool_translation(const Ref<Translation> &p_translation) {
	tool_translation = p_translation;
}

Ref<Translation> TranslationServer::get_tool_translation() const {
	return tool_translation;
}

String TranslationServer::get_tool_locale() const {
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint() || Engine::get_singleton()->is_project_manager_hint()) {
		if (TranslationServer::get_singleton()->get_tool_translation().is_valid()) {
			return tool_translation->get_locale();
		} else {
			return "en";
		}
	} else {
#else
	{
#endif
		// Look for best matching loaded translation.
		String best_locale = "en";
		int best_score = 0;

		for (const Ref<Translation> &E : translations) {
			const Ref<Translation> &t = E;
			ERR_FAIL_COND_V(t.is_null(), best_locale);
			String l = t->get_locale();

			int score = compare_locales(locale, l);
			if (score > 0 && score >= best_score) {
				best_locale = l;
				best_score = score;
				if (score == 10) {
					break; // Exact match, skip the rest.
				}
			}
		}
		return best_locale;
	}
}

StringName TranslationServer::tool_translate(const StringName &p_message, const StringName &p_context) const {
	if (p_message == StringName()) {
		return p_message;
	}
	if (tool_translation.is_valid()) {
		StringName r = tool_translation->get_message(p_message, p_context);
		if (r) {
			return ed_pseudolocalization_enabled ? tool_pseudolocalize(r) : r;
		}
	}
	return ed_pseudolocalization_enabled ? tool_pseudolocalize(p_message) : p_message;
}

StringName TranslationServer::tool_translate_plural(const StringName &p_message, const StringName &p_message_plural, int p_n, const StringName &p_context) const {
	if (p_message == StringName() && p_message_plural == StringName()) {
		return p_message;
	}
	if (tool_translation.is_valid()) {
		StringName r = tool_translation->get_plural_message(p_message, p_message_plural, p_n, p_context);
		if (r) {
			return ed_pseudolocalization_enabled ? tool_pseudolocalize(r) : r;
		}
	}

	if (p_n == 1) {
		return ed_pseudolocalization_enabled ? tool_pseudolocalize(p_message) : p_message;
	}
	return ed_pseudolocalization_enabled ? tool_pseudolocalize(p_message_plural) : p_message_plural;
}

void TranslationServer::set_property_translation(const Ref<Translation> &p_translation) {
	property_translation = p_translation;
}

StringName TranslationServer::property_translate(const StringName &p_message, const StringName &p_context) const {
	if (property_translation.is_valid()) {
		StringName r = property_translation->get_message(p_message, p_context);
		if (r) {
			return r;
		}
	}
	return p_message;
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

void TranslationServer::set_extractable_translation(const Ref<Translation> &p_translation) {
	extractable_translation = p_translation;
}

StringName TranslationServer::extractable_translate(const StringName &p_message, const StringName &p_context) const {
	if (extractable_translation.is_valid()) {
		StringName r = extractable_translation->get_message(p_message, p_context);
		if (r) {
			return r;
		}
	}
	return p_message;
}

StringName TranslationServer::extractable_translate_plural(const StringName &p_message, const StringName &p_message_plural, int p_n, const StringName &p_context) const {
	if (extractable_translation.is_valid()) {
		StringName r = extractable_translation->get_plural_message(p_message, p_message_plural, p_n, p_context);
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

	ResourceLoader::reload_translation_remaps();

	if (OS::get_singleton()->get_main_loop()) {
		OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_TRANSLATION_CHANGED);
	}
}

#ifdef TOOLS_ENABLED
void TranslationServer::reload_editor_pseudolocalization() {
	ed_pseudolocalization_enabled = EDITOR_GET("interface/debug/pseudolocalization/use_pseudolocalization");
	ed_pseudolocalization_accents_enabled = EDITOR_GET("interface/debug/pseudolocalization/replace_with_accents");
	ed_pseudolocalization_double_vowels_enabled = EDITOR_GET("interface/debug/pseudolocalization/double_vowels");
	ed_pseudolocalization_fake_bidi_enabled = EDITOR_GET("interface/debug/pseudolocalization/fake_bidi");
	ed_pseudolocalization_override_enabled = EDITOR_GET("interface/debug/pseudolocalization/override");
	ed_pseudolocalization_numbers_enabled = EDITOR_GET("interface/debug/pseudolocalization/numbers");
	ed_expansion_ratio = EDITOR_GET("interface/debug/pseudolocalization/expansion_ratio");
	ed_pseudolocalization_prefix = EDITOR_GET("interface/debug/pseudolocalization/prefix");
	ed_pseudolocalization_suffix = EDITOR_GET("interface/debug/pseudolocalization/suffix");
	ed_pseudolocalization_skip_placeholders_enabled = EDITOR_GET("interface/debug/pseudolocalization/skip_placeholders");

	if (OS::get_singleton()->get_main_loop()) {
		OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_TRANSLATION_CHANGED);
	}
}
#endif

void TranslationServer::reload_pseudolocalization() {
	pseudolocalization_accents_enabled = GLOBAL_GET("internationalization/pseudolocalization/replace_with_accents");
	pseudolocalization_double_vowels_enabled = GLOBAL_GET("internationalization/pseudolocalization/double_vowels");
	pseudolocalization_fake_bidi_enabled = GLOBAL_GET("internationalization/pseudolocalization/fake_bidi");
	pseudolocalization_override_enabled = GLOBAL_GET("internationalization/pseudolocalization/override");
	pseudolocalization_numbers_enabled = GLOBAL_GET("internationalization/pseudolocalization/numbers");
	expansion_ratio = GLOBAL_GET("internationalization/pseudolocalization/expansion_ratio");
	pseudolocalization_prefix = GLOBAL_GET("internationalization/pseudolocalization/prefix");
	pseudolocalization_suffix = GLOBAL_GET("internationalization/pseudolocalization/suffix");
	pseudolocalization_skip_placeholders_enabled = GLOBAL_GET("internationalization/pseudolocalization/skip_placeholders");

	ResourceLoader::reload_translation_remaps();

	if (OS::get_singleton()->get_main_loop()) {
		OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_TRANSLATION_CHANGED);
	}
}

StringName TranslationServer::pseudolocalize(const StringName &p_message) const {
	String message = p_message;
	int length = message.length();
	if (pseudolocalization_override_enabled) {
		message = get_override_string(message, false);
	}

	if (pseudolocalization_double_vowels_enabled) {
		message = double_vowels(message, false);
	}

	if (pseudolocalization_accents_enabled) {
		message = replace_with_accented_string(message, false);
	}

	if (pseudolocalization_fake_bidi_enabled) {
		message = wrap_with_fakebidi_characters(message, false);
	}

	StringName res = add_padding(message, length, false);
	return res;
}

StringName TranslationServer::tool_pseudolocalize(const StringName &p_message) const {
	String message = p_message;
	int length = message.length();
	if (ed_pseudolocalization_override_enabled) {
		message = get_override_string(message, true);
	}

	if (ed_pseudolocalization_double_vowels_enabled) {
		message = double_vowels(message, true);
	}

	if (ed_pseudolocalization_accents_enabled) {
		message = replace_with_accented_string(message, true);
	}

	if (ed_pseudolocalization_fake_bidi_enabled) {
		message = wrap_with_fakebidi_characters(message, true);
	}

	StringName res = add_padding(message, length, true);
	return res;
}

String TranslationServer::get_override_string(String &p_message, bool p_tool) const {
	String res;
	bool skip = p_tool ? ed_pseudolocalization_skip_placeholders_enabled : pseudolocalization_skip_placeholders_enabled;
	for (int i = 0; i < p_message.length(); i++) {
		if (skip && is_placeholder(p_message, i)) {
			res += p_message[i];
			res += p_message[i + 1];
			i++;
			continue;
		}
		res += '*';
	}
	return res;
}

String TranslationServer::double_vowels(String &p_message, bool p_tool) const {
	String res;
	bool skip = p_tool ? ed_pseudolocalization_skip_placeholders_enabled : pseudolocalization_skip_placeholders_enabled;
	for (int i = 0; i < p_message.length(); i++) {
		if (skip && is_placeholder(p_message, i)) {
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

String TranslationServer::replace_with_accented_string(String &p_message, bool p_tool) const {
	String res;
	bool skip = p_tool ? ed_pseudolocalization_skip_placeholders_enabled : pseudolocalization_skip_placeholders_enabled;
	for (int i = 0; i < p_message.length(); i++) {
		if (skip && is_placeholder(p_message, i)) {
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

String TranslationServer::wrap_with_fakebidi_characters(String &p_message, bool p_tool) const {
	String res;
	char32_t fakebidiprefix = U'\u202e';
	char32_t fakebidisuffix = U'\u202c';
	res += fakebidiprefix;
	bool skip = p_tool ? ed_pseudolocalization_skip_placeholders_enabled : pseudolocalization_skip_placeholders_enabled;
	// The fake bidi unicode gets popped at every newline so pushing it back at every newline.
	for (int i = 0; i < p_message.length(); i++) {
		if (p_message[i] == '\n') {
			res += fakebidisuffix;
			res += p_message[i];
			res += fakebidiprefix;
		} else if (skip && is_placeholder(p_message, i)) {
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

String TranslationServer::add_padding(const String &p_message, int p_length, bool p_tool) const {
	String underscores = String("_").repeat(p_length * (p_tool ? ed_expansion_ratio : expansion_ratio) / 2);
	String prefix = (p_tool ? ed_pseudolocalization_prefix : pseudolocalization_prefix) + underscores;
	String suffix = underscores + (p_tool ? ed_pseudolocalization_suffix : pseudolocalization_suffix);

	return prefix + p_message + suffix;
}

const char32_t *TranslationServer::get_accented_version(char32_t p_character) const {
	if (!is_ascii_alphabet_char(p_character)) {
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
	return p_index < p_message.length() - 1 && p_message[p_index] == '%' &&
			(p_message[p_index + 1] == 's' || p_message[p_index + 1] == 'c' || p_message[p_index + 1] == 'd' ||
					p_message[p_index + 1] == 'o' || p_message[p_index + 1] == 'x' || p_message[p_index + 1] == 'X' || p_message[p_index + 1] == 'f');
}

#ifdef TOOLS_ENABLED
void TranslationServer::get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const {
	const String pf = p_function;
	if (p_idx == 0) {
		HashMap<String, String> *target_hash_map = nullptr;
		if (pf == "get_language_name") {
			target_hash_map = &language_map;
		} else if (pf == "get_script_name") {
			target_hash_map = &script_map;
		} else if (pf == "get_country_name") {
			target_hash_map = &country_name_map;
		}

		if (target_hash_map) {
			for (const KeyValue<String, String> &E : *target_hash_map) {
				r_options->push_back(E.key.quote());
			}
		}
	}
	Object::get_argument_options(p_function, p_idx, r_options);
}
#endif // TOOLS_ENABLED

void TranslationServer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_locale", "locale"), &TranslationServer::set_locale);
	ClassDB::bind_method(D_METHOD("get_locale"), &TranslationServer::get_locale);
	ClassDB::bind_method(D_METHOD("get_tool_locale"), &TranslationServer::get_tool_locale);

	ClassDB::bind_method(D_METHOD("compare_locales", "locale_a", "locale_b"), &TranslationServer::compare_locales);
	ClassDB::bind_method(D_METHOD("standardize_locale", "locale"), &TranslationServer::standardize_locale);

	ClassDB::bind_method(D_METHOD("get_all_languages"), &TranslationServer::get_all_languages);
	ClassDB::bind_method(D_METHOD("get_language_name", "language"), &TranslationServer::get_language_name);

	ClassDB::bind_method(D_METHOD("get_all_scripts"), &TranslationServer::get_all_scripts);
	ClassDB::bind_method(D_METHOD("get_script_name", "script"), &TranslationServer::get_script_name);

	ClassDB::bind_method(D_METHOD("get_all_countries"), &TranslationServer::get_all_countries);
	ClassDB::bind_method(D_METHOD("get_country_name", "country"), &TranslationServer::get_country_name);

	ClassDB::bind_method(D_METHOD("get_locale_name", "locale"), &TranslationServer::get_locale_name);

	ClassDB::bind_method(D_METHOD("translate", "message", "context"), &TranslationServer::translate, DEFVAL(StringName()));
	ClassDB::bind_method(D_METHOD("translate_plural", "message", "plural_message", "n", "context"), &TranslationServer::translate_plural, DEFVAL(StringName()));

	ClassDB::bind_method(D_METHOD("add_translation", "translation"), &TranslationServer::add_translation);
	ClassDB::bind_method(D_METHOD("remove_translation", "translation"), &TranslationServer::remove_translation);
	ClassDB::bind_method(D_METHOD("get_translation_object", "locale"), &TranslationServer::get_translation_object);

	ClassDB::bind_method(D_METHOD("clear"), &TranslationServer::clear);

	ClassDB::bind_method(D_METHOD("format_number", "number", "language"), &TranslationServer::format_number, DEFVAL(String()));
	ClassDB::bind_method(D_METHOD("get_percent_sign", "language"), &TranslationServer::get_percent_sign, DEFVAL(String()));
	ClassDB::bind_method(D_METHOD("parse_number", "number", "language"), &TranslationServer::parse_number, DEFVAL(String()));

	ClassDB::bind_method(D_METHOD("get_loaded_locales"), &TranslationServer::get_loaded_locales);

	ClassDB::bind_method(D_METHOD("is_pseudolocalization_enabled"), &TranslationServer::is_pseudolocalization_enabled);
	ClassDB::bind_method(D_METHOD("set_pseudolocalization_enabled", "enabled"), &TranslationServer::set_pseudolocalization_enabled);
	ClassDB::bind_method(D_METHOD("reload_pseudolocalization"), &TranslationServer::reload_pseudolocalization);
	ClassDB::bind_method(D_METHOD("pseudolocalize", "message"), &TranslationServer::pseudolocalize);
	ADD_PROPERTY(PropertyInfo(Variant::Type::BOOL, "pseudolocalization_enabled"), "set_pseudolocalization_enabled", "is_pseudolocalization_enabled");
}

void TranslationServer::load_translations() {
	_load_translations("internationalization/locale/translations"); //all
	_load_translations("internationalization/locale/translations_" + locale.substr(0, 2));

	if (locale.substr(0, 2) != locale) {
		_load_translations("internationalization/locale/translations_" + locale);
	}
}

TranslationServer::TranslationServer() {
	singleton = this;
	init_locale_info();
	init_num_systems();
}
