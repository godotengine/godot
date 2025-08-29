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
#include "core/io/compression.h"
#include "core/io/resource_loader.h"
#include "core/os/os.h"
#include "core/string/locale_remaps.h"

#include "core/string/country_names.gen.h"
#include "core/string/language_names.gen.h"
#include "core/string/script_names.gen.h"

Vector<TranslationServer::LocaleScriptInfo> TranslationServer::locale_script_info;

HashMap<String, String> TranslationServer::language_map;
HashMap<String, String> TranslationServer::language_map_a3_to_a1;
HashMap<String, String> TranslationServer::script_map;
HashMap<String, String> TranslationServer::locale_rename_map;
HashMap<String, String> TranslationServer::country_name_map;
HashMap<String, String> TranslationServer::country_name_map_a3_to_a1;
HashMap<String, String> TranslationServer::variant_map;
HashMap<String, String> TranslationServer::country_rename_map;
HashMap<char32_t, char32_t> TranslationServer::diacritics_map;

Vector<String> TranslationServer::_get_csv_line(const uint8_t *p_data, int64_t p_size, int64_t p_start, int64_t &r_end) const {
	String line;

	// CSV can support entries with line breaks as long as they are enclosed
	// in double quotes. So our "line" might be more than a single line in the
	// text file.
	int64_t pos = p_start;
	int64_t prev_pos = p_start;
	int qc = 0;
	do {
		if (pos == p_size) {
			break;
		}
		while (pos < p_size) {
			if (p_data[pos] == '\n') {
				break;
			}
			pos++;
		}
		line += String::utf8((const char *)&p_data[prev_pos], pos - prev_pos).replace("\r", "") + "\n";
		prev_pos = pos + 1;
		qc = 0;
		for (int i = 0; i < line.length(); i++) {
			if (line[i] == '"') {
				qc++;
			}
		}
	} while (qc % 2);

	r_end = pos + 1;

	// Remove the extraneous newline we've added above.
	line = line.substr(0, line.length() - 1);

	Vector<String> strings;

	bool in_quote = false;
	String current;
	for (int i = 0; i < line.length(); i++) {
		char32_t c = line[i];
		// A delimiter ends the current entry, unless it's in a quoted string.
		if (!in_quote && c == ',') {
			strings.push_back(current);
			current = String();
		} else if (c == '"') {
			// Doubled quotes are escapes for intentional quotes in the string.
			if (line[i + 1] == '"' && in_quote) {
				current += '"';
				i++;
			} else {
				in_quote = !in_quote;
			}
		} else {
			current += c;
		}
	}

	strings.push_back(current);

	return strings;
}

void TranslationServer::_diacritics_map_add(const String &p_from, char32_t p_to) {
	for (int i = 0; i < p_from.size(); i++) {
		diacritics_map[p_from[i]] = p_to;
	}
}

void TranslationServer::init_locale_info() {
	// Init diacritics list for name matching.
	_diacritics_map_add(U"àáâãäåāăąǎǟǡǻȁȃȧḁẚạảấầẩẫậắằẳẵặ", U'a');
	_diacritics_map_add(U"ǣǽ", U'æ');
	_diacritics_map_add(U"ḃḅḇ", U'b');
	_diacritics_map_add(U"çćĉċčḉ", U'c');
	_diacritics_map_add(U"ďḋḍḏḑḓ", U'd');
	_diacritics_map_add(U"èéêëēĕėęěȇȩḕḗḙḛḝẹẻẽếềểễệ", U'e');
	_diacritics_map_add(U"ḟ", U'f');
	_diacritics_map_add(U"ĝğġģǧǵḡ", U'g');
	_diacritics_map_add(U"ĥȟḣḥḧḩḫẖ", U'h');
	_diacritics_map_add(U"ìíîïĩīĭįıǐȉȋḭḯỉị", U'i');
	_diacritics_map_add(U"ĵ", U'j');
	_diacritics_map_add(U"ķĸǩḱḳḵ", U'k');
	_diacritics_map_add(U"ĺļľŀḷḹḻḽ", U'l');
	_diacritics_map_add(U"ḿṁṃ", U'm');
	_diacritics_map_add(U"ñńņňŉǹṅṇṉṋ", U'n');
	_diacritics_map_add(U"òóôõöōŏőơǒǫǭȍȏȫȭȯȱṍṏṑṓọỏốồổỗộớờởỡợ", U'o');
	_diacritics_map_add(U"ṗṕ", U'p');
	_diacritics_map_add(U"ŕŗřȑȓṙṛṝṟ", U'r');
	_diacritics_map_add(U"śŝşšſșṡṣṥṧṩẛẜẝ", U's');
	_diacritics_map_add(U"ţťțṫṭṯṱẗ", U't');
	_diacritics_map_add(U"ùúûüũūŭůűųưǔǖǘǚǜȕȗṳṵṷṹṻụủứừửữự", U'u');
	_diacritics_map_add(U"ṽṿ", U'v');
	_diacritics_map_add(U"ŵẁẃẅẇẉẘ", U'w');
	_diacritics_map_add(U"ẋẍ", U'x');
	_diacritics_map_add(U"ýÿŷẏẙỳỵỷỹỿ", U'y');
	_diacritics_map_add(U"źżžẑẓẕ", U'z');

	// Init locale info.
	language_map.clear();
	language_map_a3_to_a1.clear();
	{
		Vector<uint8_t> lang_data;
		lang_data.resize(_lang_data_uncompressed_size);
		int64_t size = Compression::decompress(lang_data.ptrw(), _lang_data_uncompressed_size, _lang_data_compressed, _lang_data_compressed_size, Compression::MODE_DEFLATE);
		if (size > 0) {
			int64_t pos = 0;
			while (pos < lang_data.size()) {
				Vector<String> lang_rec = _get_csv_line(lang_data.ptr(), lang_data.size(), pos, pos);
				if (lang_rec.size() == 3) {
					const String &a1_code = lang_rec[0];
					const String &a3_code = lang_rec[1];
					const String &name = lang_rec[2];
					if (!a3_code.is_empty() || !a1_code.is_empty()) {
						if (!a3_code.is_empty() && !a1_code.is_empty()) {
							language_map_a3_to_a1[a3_code] = a1_code;
						}
						if (!a1_code.is_empty()) {
							language_map[a1_code] = name;
						} else {
							language_map[a3_code] = name;
						}
					}
				}
			}
		}
	}

	// Init locale-script map.
	locale_script_info.clear();
	int idx = 0;
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
	{
		Vector<uint8_t> script_data;
		script_data.resize(_script_data_uncompressed_size);
		int64_t size = Compression::decompress(script_data.ptrw(), _script_data_uncompressed_size, _script_data_compressed, _script_data_compressed_size, Compression::MODE_DEFLATE);
		if (size > 0) {
			int64_t pos = 0;
			while (pos < script_data.size()) {
				Vector<String> script_rec = _get_csv_line(script_data.ptr(), script_data.size(), pos, pos);
				if (script_rec.size() == 2) {
					const String &code = script_rec[0];
					const String &name = script_rec[1];
					if (!code.is_empty()) {
						script_map[code] = name;
					}
				}
			}
		}
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
	country_name_map_a3_to_a1.clear();
	{
		Vector<uint8_t> country_data;
		country_data.resize(_country_data_uncompressed_size);
		int64_t size = Compression::decompress(country_data.ptrw(), _country_data_uncompressed_size, _country_data_compressed, _country_data_compressed_size, Compression::MODE_DEFLATE);
		if (size > 0) {
			int64_t pos = 0;
			while (pos < country_data.size()) {
				Vector<String> country_rec = _get_csv_line(country_data.ptr(), country_data.size(), pos, pos);
				if (country_rec.size() == 3) {
					const String &a1_code = country_rec[0];
					const String &a3_code = country_rec[1];
					const String &name = country_rec[2];
					if (!a3_code.is_empty() || !a1_code.is_empty()) {
						if (!a3_code.is_empty() && !a1_code.is_empty()) {
							country_name_map_a3_to_a1[a3_code] = a1_code;
						}
						if (!a1_code.is_empty()) {
							country_name_map[a1_code] = name;
						} else {
							country_name_map[a3_code] = name;
						}
					}
				}
			}
		}
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

TranslationServer::Locale::operator String() const {
	String out = language;
	if (!script.is_empty()) {
		out = out + "_" + script;
	}
	if (!country.is_empty()) {
		out = out + "_" + country;
	}
	if (!variant.is_empty()) {
		out = out + "_" + variant;
	}
	return out;
}

TranslationServer::Locale::Locale(const String &p_locale, bool p_add_defaults) {
	const TranslationServer *server = TranslationServer::get_singleton();

	// Replaces '-' with '_' for macOS style locales.
	String univ_locale = p_locale.replace_char('-', '_');

	// Extract locale elements.
	Vector<String> locale_elements = univ_locale.get_slicec('@', 0).split("_");
	language = locale_elements[0];
	const String *lang_a3 = language_map_a3_to_a1.getptr(language);
	if (lang_a3) {
		language = *lang_a3;
	}
	if (locale_elements.size() >= 2) {
		if (is_script_code(locale_elements[1])) {
			script = locale_elements[1];
		}
		if (is_country_code(locale_elements[1])) {
			country = locale_elements[1];
		}
	}
	if (!country.is_empty()) {
		const String *country_a3 = country_name_map_a3_to_a1.getptr(country);
		if (country_a3) {
			country = *country_a3;
		}
	}
	if (locale_elements.size() >= 3) {
		if (is_country_code(locale_elements[2])) {
			country = locale_elements[2];
		} else if (server->variant_map.has(locale_elements[2].to_lower()) && server->variant_map[locale_elements[2].to_lower()] == language) {
			variant = locale_elements[2].to_lower();
		}
	}
	if (locale_elements.size() >= 4) {
		if (server->variant_map.has(locale_elements[3].to_lower()) && server->variant_map[locale_elements[3].to_lower()] == language) {
			variant = locale_elements[3].to_lower();
		}
	}

	// Try extract script and variant from the extra part.
	Vector<String> script_extra = univ_locale.get_slicec('@', 1).split(";");
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
		} else if (server->variant_map.has(script_extra[i].to_lower()) && server->variant_map[script_extra[i].to_lower()] == language) {
			variant = script_extra[i].to_lower();
		}
	}

	// Handles known non-ISO language names used e.g. on Windows.
	if (server->locale_rename_map.has(language)) {
		language = server->locale_rename_map[language];
	}

	// Handle country renames.
	if (server->country_rename_map.has(country)) {
		country = server->country_rename_map[country];
	}

	// Remove unsupported script codes.
	if (!server->script_map.has(script)) {
		script = "";
	}

	// Add script code base on language and country codes for some ambiguous cases.
	if (p_add_defaults) {
		if (script.is_empty()) {
			for (int i = 0; i < server->locale_script_info.size(); i++) {
				const LocaleScriptInfo &info = server->locale_script_info[i];
				if (info.name == language) {
					if (country.is_empty() || info.supported_countries.has(country)) {
						script = info.script;
						break;
					}
				}
			}
		}
		if (!script.is_empty() && country.is_empty()) {
			// Add conntry code based on script for some ambiguous cases.
			for (int i = 0; i < server->locale_script_info.size(); i++) {
				const LocaleScriptInfo &info = server->locale_script_info[i];
				if (info.name == language && info.script == script) {
					country = info.default_country;
					break;
				}
			}
		}
	}
}

String TranslationServer::standardize_locale(const String &p_locale, bool p_add_defaults) const {
	return Locale(p_locale, p_add_defaults).operator String();
}

int TranslationServer::compare_locales(const String &p_locale_a, const String &p_locale_b) const {
	if (p_locale_a == p_locale_b) {
		// Exact match.
		return 10;
	}

	const String cache_key = p_locale_a + "|" + p_locale_b;
	const int *cached_result = locale_compare_cache.getptr(cache_key);
	if (cached_result) {
		return *cached_result;
	}

	Locale locale_a = Locale(p_locale_a, true);
	Locale locale_b = Locale(p_locale_b, true);

	if (locale_a == locale_b) {
		// Exact match.
		locale_compare_cache.insert(cache_key, 10);
		return 10;
	}

	if (locale_a.language != locale_b.language) {
		// No match.
		locale_compare_cache.insert(cache_key, 0);
		return 0;
	}

	// Matching language, both locales have extra parts. Compare the
	// remaining elements. If both elements are non-empty, check the
	// match to increase or decrease the score. If either element or
	// both are empty, leave the score as is.
	int score = 5;
	if (!locale_a.script.is_empty() && !locale_b.script.is_empty()) {
		if (locale_a.script == locale_b.script) {
			score++;
		} else {
			score--;
		}
	}
	if (!locale_a.country.is_empty() && !locale_b.country.is_empty()) {
		if (locale_a.country == locale_b.country) {
			score++;
		} else {
			score--;
		}
	}
	if (!locale_a.variant.is_empty() && !locale_b.variant.is_empty()) {
		if (locale_a.variant == locale_b.variant) {
			score++;
		} else {
			score--;
		}
	}

	locale_compare_cache.insert(cache_key, score);
	return score;
}

bool TranslationServer::is_language_code(const String &p_code) {
	// xx or xxx
	if ((p_code.length() == 2 && is_ascii_lower_case(p_code[0]) && is_ascii_lower_case(p_code[1]))) {
		return true; // Alpha-2 code.
	} else if (p_code.length() == 3 && is_ascii_lower_case(p_code[0]) && is_ascii_lower_case(p_code[1]) && is_ascii_lower_case(p_code[2])) {
		return true; // Alpha-3 code.
	} else {
		return false;
	}
}

bool TranslationServer::is_script_code(const String &p_code) {
	// Xxxx
	if (p_code.length() == 4 && is_ascii_upper_case(p_code[0]) && is_ascii_lower_case(p_code[1]) && is_ascii_lower_case(p_code[2]) && is_ascii_lower_case(p_code[3])) {
		return true;
	} else {
		return false;
	}
}

bool TranslationServer::is_country_code(const String &p_code) {
	// XX or XXX or NNN
	if ((p_code.length() == 2 && is_ascii_upper_case(p_code[0]) && is_ascii_upper_case(p_code[1]))) {
		return true; // Alpha-2 code.
	} else if (p_code.length() == 3 && is_ascii_upper_case(p_code[0]) && is_ascii_upper_case(p_code[1]) && is_ascii_upper_case(p_code[2])) {
		return true; // Alpha-3 code.
	} else if (p_code.length() == 3 && is_digit(p_code[0]) && is_digit(p_code[1]) && is_digit(p_code[2])) {
		return true; // UN M49 area code.
	} else {
		return false;
	}
}

String TranslationServer::get_locale_name(const String &p_locale) const {
	String lang_name, script_name, country_name;
	Vector<String> locale_elements = standardize_locale(p_locale).split("_");
	lang_name = locale_elements[0];
	if (locale_elements.size() >= 2) {
		if (is_script_code(locale_elements[1])) {
			script_name = locale_elements[1];
		}
		if (is_country_code(locale_elements[1])) {
			country_name = locale_elements[1];
		}
	}
	if (locale_elements.size() >= 3) {
		if (is_country_code(locale_elements[2])) {
			country_name = locale_elements[2];
		}
	}

	String name = get_language_name(lang_name);
	if (!script_name.is_empty()) {
		name = name + " (" + get_script_name(script_name) + ")";
	}
	if (!country_name.is_empty()) {
		name = name + ", " + get_country_name(country_name);
	}
	return name;
}

bool TranslationServer::is_language_code_free(const String &p_code) const {
	return !language_map.has(p_code) && !language_map_a3_to_a1.has(p_code);
}

bool TranslationServer::is_country_code_free(const String &p_code) const {
	return !country_name_map.has(p_code) && !country_rename_map.has(p_code) && !country_name_map_a3_to_a1.has(p_code);
}

void TranslationServer::set_custom_language_codes(const Dictionary &p_dict) {
	language_map_custom.clear();
	for (const Variant *key = p_dict.next(nullptr); key; key = p_dict.next(key)) {
		const String &key_s = *key;
		if (!is_language_code(key_s)) {
			WARN_PRINT(vformat("Invalid language code format: '%s'", key_s));
			continue;
		}
		if (!is_language_code_free(key_s)) {
			WARN_PRINT(vformat("Language code '%s' is already defined.", key_s));
			continue;
		}
		const String &value_s = p_dict[*key];
		language_map_custom[key_s] = value_s;
	}
}

Dictionary TranslationServer::get_custom_language_codes() const {
	Dictionary out;
	for (const KeyValue<String, String> &E : language_map_custom) {
		out[E.key] = E.value;
	}
	return out;
}

void TranslationServer::set_custom_country_codes(const Dictionary &p_dict) {
	country_name_map_custom.clear();
	for (const Variant *key = p_dict.next(nullptr); key; key = p_dict.next(key)) {
		const String &key_s = *key;
		if (!is_country_code(key_s)) {
			WARN_PRINT(vformat("Invalid country code format: '%s'", key_s));
			continue;
		}
		if (!is_country_code_free(key_s)) {
			WARN_PRINT(vformat("Country code '%s' is already defined.", key_s));
			continue;
		}
		const String &value_s = p_dict[*key];
		country_name_map_custom[key_s] = value_s;
	}
}

Dictionary TranslationServer::get_custom_country_codes() const {
	Dictionary out;
	for (const KeyValue<String, String> &E : country_name_map_custom) {
		out[E.key] = E.value;
	}
	return out;
}

String TranslationServer::_strip_diacritics(const String &p_string) const {
	String result;
	for (int i = 0; i < p_string.length(); i++) {
		if (diacritics_map.has(p_string[i])) {
			result += diacritics_map[p_string[i]];
		} else {
			result += p_string[i];
		}
	}
	return result;
}

bool TranslationServer::_match_code(const String &p_key, const String &p_val, const String &p_str) const {
	if (p_key.to_lower().begins_with(p_str)) {
		return true;
	}
	String val = _strip_diacritics(p_val.to_lower());
	if (val.begins_with(p_str)) {
		return true;
	}
	return false;
}

Vector<String> TranslationServer::get_all_languages() const {
	Vector<String> languages;

	for (const KeyValue<String, String> &E : language_map) {
		languages.push_back(E.key);
	}
	for (const KeyValue<String, String> &E : language_map_custom) {
		languages.push_back(E.key);
	}

	return languages;
}

Vector<String> TranslationServer::find_language(const String &p_str) const {
	Vector<String> languages;
	if (p_str.is_empty()) {
		return languages;
	}

	const String &str = _strip_diacritics(p_str.to_lower());
	for (const KeyValue<String, String> &E : language_map) {
		if (_match_code(E.key, E.value, str)) {
			languages.push_back(E.key);
		}
	}
	for (const KeyValue<String, String> &E : language_map_custom) {
		if (_match_code(E.key, E.value, str)) {
			languages.push_back(E.key);
		}
	}

	return languages;
}

String TranslationServer::get_language_name(const String &p_language) const {
	String language = p_language;
	const String *custom_lang = language_map_custom.getptr(language);
	if (custom_lang) {
		return *custom_lang;
	}

	const String *a3_to_a1_lang = language_map_a3_to_a1.getptr(language);
	if (a3_to_a1_lang) {
		language = *a3_to_a1_lang;
	}
	const String *lang = language_map.getptr(language);
	if (lang) {
		return *lang;
	} else {
		return language;
	}
}

Vector<String> TranslationServer::get_all_scripts() const {
	Vector<String> scripts;

	for (const KeyValue<String, String> &E : script_map) {
		scripts.push_back(E.key);
	}

	return scripts;
}

Vector<String> TranslationServer::find_script(const String &p_str) const {
	Vector<String> scripts;
	if (p_str.is_empty()) {
		return scripts;
	}

	const String &str = _strip_diacritics(p_str.to_lower());
	for (const KeyValue<String, String> &E : script_map) {
		if (_match_code(E.key, E.value, str)) {
			scripts.push_back(E.key);
		}
	}

	return scripts;
}

String TranslationServer::get_script_name(const String &p_script) const {
	const String *script_code = script_map.getptr(p_script);
	if (script_code) {
		return *script_code;
	} else {
		return p_script;
	}
}

Vector<String> TranslationServer::get_all_countries() const {
	Vector<String> countries;

	for (const KeyValue<String, String> &E : country_name_map) {
		countries.push_back(E.key);
	}
	for (const KeyValue<String, String> &E : country_name_map_custom) {
		countries.push_back(E.key);
	}

	return countries;
}

Vector<String> TranslationServer::find_country(const String &p_str) const {
	Vector<String> countries;
	if (p_str.is_empty()) {
		return countries;
	}

	const String &str = _strip_diacritics(p_str.to_lower());
	for (const KeyValue<String, String> &E : country_name_map) {
		if (_match_code(E.key, E.value, str)) {
			countries.push_back(E.key);
		}
	}
	for (const KeyValue<String, String> &E : country_name_map_custom) {
		if (_match_code(E.key, E.value, str)) {
			countries.push_back(E.key);
		}
	}

	return countries;
}

String TranslationServer::get_country_name(const String &p_country) const {
	String country = p_country;
	const String *custom_country = country_name_map_custom.getptr(country);
	if (custom_country) {
		return *custom_country;
	}
	const String *a3_to_a1_cnt = country_name_map_a3_to_a1.getptr(country);
	if (a3_to_a1_cnt) {
		country = *a3_to_a1_cnt;
	}
	const String *cnt = country_name_map.getptr(country);
	if (cnt) {
		return *cnt;
	} else {
		return country;
	}
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

void TranslationServer::set_fallback_locale(const String &p_locale) {
	fallback = p_locale;
}

String TranslationServer::get_fallback_locale() const {
	return fallback;
}

PackedStringArray TranslationServer::get_loaded_locales() const {
	return main_domain->get_loaded_locales();
}

void TranslationServer::add_translation(const Ref<Translation> &p_translation) {
	main_domain->add_translation(p_translation);
}

void TranslationServer::remove_translation(const Ref<Translation> &p_translation) {
	main_domain->remove_translation(p_translation);
}

Ref<Translation> TranslationServer::get_translation_object(const String &p_locale) {
	return main_domain->get_translation_object(p_locale);
}

void TranslationServer::clear() {
	main_domain->clear();
}

StringName TranslationServer::translate(const StringName &p_message, const StringName &p_context) const {
	return main_domain->translate(p_message, p_context);
}

StringName TranslationServer::translate_plural(const StringName &p_message, const StringName &p_message_plural, int p_n, const StringName &p_context) const {
	return main_domain->translate_plural(p_message, p_message_plural, p_n, p_context);
}

bool TranslationServer::has_domain(const StringName &p_domain) const {
	if (p_domain == StringName()) {
		return true;
	}
	return custom_domains.has(p_domain);
}

Ref<TranslationDomain> TranslationServer::get_or_add_domain(const StringName &p_domain) {
	if (p_domain == StringName()) {
		return main_domain;
	}
	const Ref<TranslationDomain> *domain = custom_domains.getptr(p_domain);
	if (domain) {
		if (domain->is_valid()) {
			return *domain;
		}
		ERR_PRINT("Bug (please report): Found invalid translation domain.");
	}
	Ref<TranslationDomain> new_domain = memnew(TranslationDomain);
	custom_domains[p_domain] = new_domain;
	return new_domain;
}

void TranslationServer::remove_domain(const StringName &p_domain) {
	ERR_FAIL_COND_MSG(p_domain == StringName(), "Cannot remove main translation domain.");
	custom_domains.erase(p_domain);
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
	main_domain->set_pseudolocalization_enabled(GLOBAL_DEF("internationalization/pseudolocalization/use_pseudolocalization", false));
	main_domain->set_pseudolocalization_accents_enabled(GLOBAL_DEF("internationalization/pseudolocalization/replace_with_accents", true));
	main_domain->set_pseudolocalization_double_vowels_enabled(GLOBAL_DEF("internationalization/pseudolocalization/double_vowels", false));
	main_domain->set_pseudolocalization_fake_bidi_enabled(GLOBAL_DEF("internationalization/pseudolocalization/fake_bidi", false));
	main_domain->set_pseudolocalization_override_enabled(GLOBAL_DEF("internationalization/pseudolocalization/override", false));
	main_domain->set_pseudolocalization_expansion_ratio(GLOBAL_DEF("internationalization/pseudolocalization/expansion_ratio", 0.0));
	main_domain->set_pseudolocalization_prefix(GLOBAL_DEF("internationalization/pseudolocalization/prefix", "["));
	main_domain->set_pseudolocalization_suffix(GLOBAL_DEF("internationalization/pseudolocalization/suffix", "]"));
	main_domain->set_pseudolocalization_skip_placeholders_enabled(GLOBAL_DEF("internationalization/pseudolocalization/skip_placeholders", true));

#ifdef TOOLS_ENABLED
	ProjectSettings::get_singleton()->set_custom_property_info(PropertyInfo(Variant::STRING, "internationalization/locale/test", PROPERTY_HINT_LOCALE_ID, ""));
	ProjectSettings::get_singleton()->set_custom_property_info(PropertyInfo(Variant::STRING, "internationalization/locale/fallback", PROPERTY_HINT_LOCALE_ID, ""));
#endif
}

String TranslationServer::get_tool_locale() {
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint() || Engine::get_singleton()->is_project_manager_hint()) {
		const PackedStringArray &locales = editor_domain->get_loaded_locales();
		if (locales.has(locale)) {
			return locale;
		}
		return "en";
	} else {
#else
	{
#endif
		// Look for best matching loaded translation.
		Ref<Translation> t = main_domain->get_translation_object(locale);
		if (t.is_null()) {
			return fallback;
		}
		return t->get_locale();
	}
}

StringName TranslationServer::tool_translate(const StringName &p_message, const StringName &p_context) const {
	return editor_domain->translate(p_message, p_context);
}

StringName TranslationServer::tool_translate_plural(const StringName &p_message, const StringName &p_message_plural, int p_n, const StringName &p_context) const {
	return editor_domain->translate_plural(p_message, p_message_plural, p_n, p_context);
}

StringName TranslationServer::property_translate(const StringName &p_message, const StringName &p_context) const {
	return property_domain->translate(p_message, p_context);
}

StringName TranslationServer::doc_translate(const StringName &p_message, const StringName &p_context) const {
	return doc_domain->translate(p_message, p_context);
}

StringName TranslationServer::doc_translate_plural(const StringName &p_message, const StringName &p_message_plural, int p_n, const StringName &p_context) const {
	return doc_domain->translate_plural(p_message, p_message_plural, p_n, p_context);
}

bool TranslationServer::is_pseudolocalization_enabled() const {
	return main_domain->is_pseudolocalization_enabled();
}

void TranslationServer::set_pseudolocalization_enabled(bool p_enabled) {
	main_domain->set_pseudolocalization_enabled(p_enabled);

	ResourceLoader::reload_translation_remaps();

	if (OS::get_singleton()->get_main_loop()) {
		OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_TRANSLATION_CHANGED);
	}
}

void TranslationServer::reload_pseudolocalization() {
	main_domain->set_pseudolocalization_accents_enabled(GLOBAL_GET("internationalization/pseudolocalization/replace_with_accents"));
	main_domain->set_pseudolocalization_double_vowels_enabled(GLOBAL_GET("internationalization/pseudolocalization/double_vowels"));
	main_domain->set_pseudolocalization_fake_bidi_enabled(GLOBAL_GET("internationalization/pseudolocalization/fake_bidi"));
	main_domain->set_pseudolocalization_override_enabled(GLOBAL_GET("internationalization/pseudolocalization/override"));
	main_domain->set_pseudolocalization_expansion_ratio(GLOBAL_GET("internationalization/pseudolocalization/expansion_ratio"));
	main_domain->set_pseudolocalization_prefix(GLOBAL_GET("internationalization/pseudolocalization/prefix"));
	main_domain->set_pseudolocalization_suffix(GLOBAL_GET("internationalization/pseudolocalization/suffix"));
	main_domain->set_pseudolocalization_skip_placeholders_enabled(GLOBAL_GET("internationalization/pseudolocalization/skip_placeholders"));

	ResourceLoader::reload_translation_remaps();

	if (OS::get_singleton()->get_main_loop()) {
		OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_TRANSLATION_CHANGED);
	}
}

StringName TranslationServer::pseudolocalize(const StringName &p_message) const {
	return main_domain->pseudolocalize(p_message);
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
	ClassDB::bind_method(D_METHOD("standardize_locale", "locale", "add_defaults"), &TranslationServer::standardize_locale, DEFVAL(false));

	ClassDB::bind_method(D_METHOD("get_all_languages"), &TranslationServer::get_all_languages);
	ClassDB::bind_method(D_METHOD("find_language", "str"), &TranslationServer::find_language);
	ClassDB::bind_method(D_METHOD("get_language_name", "language"), &TranslationServer::get_language_name);

	ClassDB::bind_method(D_METHOD("get_all_scripts"), &TranslationServer::get_all_scripts);
	ClassDB::bind_method(D_METHOD("find_script", "str"), &TranslationServer::find_script);
	ClassDB::bind_method(D_METHOD("get_script_name", "script"), &TranslationServer::get_script_name);

	ClassDB::bind_method(D_METHOD("get_all_countries"), &TranslationServer::get_all_countries);
	ClassDB::bind_method(D_METHOD("find_country", "str"), &TranslationServer::find_country);
	ClassDB::bind_method(D_METHOD("get_country_name", "country"), &TranslationServer::get_country_name);

	ClassDB::bind_method(D_METHOD("get_locale_name", "locale"), &TranslationServer::get_locale_name);

	ClassDB::bind_method(D_METHOD("translate", "message", "context"), &TranslationServer::translate, DEFVAL(StringName()));
	ClassDB::bind_method(D_METHOD("translate_plural", "message", "plural_message", "n", "context"), &TranslationServer::translate_plural, DEFVAL(StringName()));

	ClassDB::bind_method(D_METHOD("add_translation", "translation"), &TranslationServer::add_translation);
	ClassDB::bind_method(D_METHOD("remove_translation", "translation"), &TranslationServer::remove_translation);
	ClassDB::bind_method(D_METHOD("get_translation_object", "locale"), &TranslationServer::get_translation_object);

	ClassDB::bind_method(D_METHOD("has_domain", "domain"), &TranslationServer::has_domain);
	ClassDB::bind_method(D_METHOD("get_or_add_domain", "domain"), &TranslationServer::get_or_add_domain);
	ClassDB::bind_method(D_METHOD("remove_domain", "domain"), &TranslationServer::remove_domain);

	ClassDB::bind_method(D_METHOD("clear"), &TranslationServer::clear);

	ClassDB::bind_method(D_METHOD("get_loaded_locales"), &TranslationServer::get_loaded_locales);

	ClassDB::bind_method(D_METHOD("is_pseudolocalization_enabled"), &TranslationServer::is_pseudolocalization_enabled);
	ClassDB::bind_method(D_METHOD("set_pseudolocalization_enabled", "enabled"), &TranslationServer::set_pseudolocalization_enabled);
	ClassDB::bind_method(D_METHOD("reload_pseudolocalization"), &TranslationServer::reload_pseudolocalization);
	ClassDB::bind_method(D_METHOD("pseudolocalize", "message"), &TranslationServer::pseudolocalize);
	ADD_PROPERTY(PropertyInfo(Variant::Type::BOOL, "pseudolocalization_enabled"), "set_pseudolocalization_enabled", "is_pseudolocalization_enabled");
}

void TranslationServer::load_translations() {
	const String prop = "internationalization/locale/translations";
	if (!ProjectSettings::get_singleton()->has_setting(prop)) {
		return;
	}
	const Vector<String> &translations = GLOBAL_GET(prop);
	for (const String &path : translations) {
		Ref<Translation> tr = ResourceLoader::load(path);
		if (tr.is_valid()) {
			add_translation(tr);
		}
	}
}

TranslationServer::TranslationServer() {
	singleton = this;
	main_domain.instantiate();
	editor_domain = get_or_add_domain("godot.editor");
	property_domain = get_or_add_domain("godot.properties");
	doc_domain = get_or_add_domain("godot.documentation");
	init_locale_info();
}
