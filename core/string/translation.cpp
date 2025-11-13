/**************************************************************************/
/*  translation.cpp                                                       */
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

#include "translation.h"

#include "core/os/thread.h"
#include "core/string/plural_rules.h"
#include "core/string/translation_server.h"

#define HINTS_STORAGE_KEY 0

static void _check_for_incompatibility(const String &p_msgctxt, const String &p_msgid) {
	// Gettext PO and MO files use an empty untranslated string without context
	// to store metadata.
	if (p_msgctxt.is_empty() && p_msgid.is_empty()) {
		WARN_PRINT("Both context and the untranslated string are empty. This may cause issues with the translation system and external tools.");
	}

	// The EOT character (0x04) is used as a separator between context and
	// untranslated string in the MO file format. This convention is also used
	// by `get_message_list()`.
	//
	// It's unusual to have this character in the context or untranslated
	// string. But it doesn't do any harm as long as you are aware of this when
	// using the relevant APIs and tools.
	if (p_msgctxt.contains_char(0x04)) {
		WARN_PRINT(vformat("Found EOT character (0x04) within context '%s'. This may cause issues with the translation system and external tools.", p_msgctxt));
	}
	if (p_msgid.contains_char(0x04)) {
		WARN_PRINT(vformat("Found EOT character (0x04) within untranslated string '%s'. This may cause issues with the translation system and external tools.", p_msgid));
	}
}

static Variant _write_message_key(const Translation::MessageKey &p_key) {
	if (p_key.msgctxt.is_empty()) {
		return p_key.msgid;
	}
	const Array &arr = { p_key.msgctxt, p_key.msgid };
	return arr;
}

static bool _read_message_key(const Variant &p_raw, Translation::MessageKey &r_key) {
	switch (p_raw.get_type()) {
		case Variant::STRING_NAME: {
			r_key.msgctxt = StringName();
			r_key.msgid = p_raw;
			return true;
		} break;

		case Variant::ARRAY: {
			const Array &arr = p_raw;
			r_key.msgctxt = arr[0];
			r_key.msgid = arr[1];
			return true;
		} break;

		default: {
			WARN_PRINT(vformat("Unknown message key type: %s.", Variant::get_type_name(p_raw.get_type())));
		} break;
	}
	return false;
}

static Vector<StringName> _read_message_value(const Variant &p_raw) {
	switch (p_raw.get_type()) {
		case Variant::STRING_NAME: {
			return { p_raw };
		} break;

		case Variant::ARRAY: {
			const Array &arr = p_raw;

			Vector<StringName> msgstrs;
			msgstrs.resize(arr.size());
			for (int i = 0; i < arr.size(); i++) {
				msgstrs.write[i] = arr[i];
			}
			return msgstrs;
		} break;

		default: {
			WARN_PRINT(vformat("Unknown message value type: %s.", Variant::get_type_name(p_raw.get_type())));
		} break;
	}
	return {};
}

Dictionary Translation::_get_messages() const {
	Dictionary d;

	// Messages.
	for (const KeyValue<MessageKey, Vector<StringName>> &E : translation_map) {
		const Variant message_key = _write_message_key(E.key);

		Variant message_value;
		switch (E.value.size()) {
			// Should not happen, but just in case.
			case 0: {
				continue;
			} break;

			// Compact and compatible with older versions.
			case 1: {
				message_value = E.value[0];
			} break;

			default: {
				Array arr;
				arr.resize(E.value.size());
				for (int i = 0; i < E.value.size(); i++) {
					arr[i] = E.value[i];
				}
				message_value = arr;
			} break;
		}

		d[message_key] = message_value;
	}

	// Optional hints.
	Dictionary hints_dict;
	for (int type = 0; type < HINT_MAX; type++) {
		const HashMap<MessageKey, String, MessageKey> *map = hint_maps[type];
		if (map == nullptr || map->is_empty()) {
			continue;
		}

		for (const KeyValue<MessageKey, String> &E : *map) {
			if (E.value.is_empty()) {
				continue; // Should not happen, but just in case.
			}

			const Variant message_key = _write_message_key(E.key);
			Dictionary hints = hints_dict.get_or_add(message_key, Dictionary());
			hints[type] = E.value;
			hints_dict[message_key] = hints;
		}
	}
	d[HINTS_STORAGE_KEY] = hints_dict;

	return d;
}

void Translation::_set_messages(const Dictionary &p_messages) {
	translation_map.clear();
	for (int type = 0; type < HINT_MAX; type++) {
		HashMap<MessageKey, String, MessageKey> *map = hint_maps[type];
		if (map) {
			map->clear();
		}
	}

	Dictionary hints_dict;
	for (const KeyValue<Variant, Variant> &kv : p_messages) {
		// Hints are added after all messages are loaded.
		if (kv.key.get_type() == Variant::INT && kv.key.operator int() == HINTS_STORAGE_KEY) {
			hints_dict = kv.value;
			continue;
		}

		MessageKey key;
		if (!_read_message_key(kv.key, key)) {
			continue;
		}
		_check_for_incompatibility(key.msgctxt, key.msgid);

		const Vector<StringName> msgstrs = _read_message_value(kv.value);
		ERR_CONTINUE_MSG(msgstrs.is_empty(), vformat("No translated strings for untranslated string '%s' with context '%s'.", key.msgid, key.msgctxt));

		translation_map[key] = msgstrs;
	}

	for (const KeyValue<Variant, Variant> &kv : hints_dict) {
		MessageKey key;
		if (!_read_message_key(kv.key, key)) {
			continue;
		}
		if (!translation_map.has(key)) {
			continue;
		}

		const Dictionary hints = kv.value;
		for (int i = 0; i < HINT_MAX; i++) {
			const Variant *v = hints.getptr(i);
			if (v == nullptr || v->get_type() != Variant::STRING) {
				continue;
			}
			const String hint_value = *v;
			if (hint_value.is_empty()) {
				continue;
			}

			if (hint_maps[i] == nullptr) {
				hint_maps[i] = memnew((HashMap<MessageKey, String, MessageKey>));
			}
			HashMap<MessageKey, String, MessageKey> &map = *hint_maps[i];
			map[key] = hint_value;
		}
	}
}

Vector<String> Translation::_get_message_list() const {
	List<StringName> msgstrs;
	get_message_list(&msgstrs);

	Vector<String> keys;
	keys.resize(msgstrs.size());
	int idx = 0;
	for (const StringName &msgstr : msgstrs) {
		keys.write[idx++] = msgstr;
	}
	return keys;
}

Vector<String> Translation::get_translated_message_list() const {
	Vector<String> msgstrs;
	for (const KeyValue<MessageKey, Vector<StringName>> &E : translation_map) {
		for (const StringName &msgstr : E.value) {
			msgstrs.push_back(msgstr);
		}
	}
	return msgstrs;
}

void Translation::set_locale(const String &p_locale) {
	locale = TranslationServer::get_singleton()->standardize_locale(p_locale);

	if (plural_rules_cache && plural_rules_override.is_empty()) {
		memdelete(plural_rules_cache);
		plural_rules_cache = nullptr;
	}
}

void Translation::add_message(const StringName &p_src_text, const StringName &p_xlated_text, const StringName &p_context) {
	_check_for_incompatibility(p_context, p_src_text);
	translation_map[{ p_context, p_src_text }] = { p_xlated_text };
}

void Translation::add_plural_message(const StringName &p_src_text, const Vector<String> &p_plural_xlated_texts, const StringName &p_context) {
	ERR_FAIL_COND_MSG(p_plural_xlated_texts.is_empty(), "Parameter vector p_plural_xlated_texts passed in is empty.");

	Vector<StringName> msgstrs;
	msgstrs.resize(p_plural_xlated_texts.size());
	for (int i = 0; i < p_plural_xlated_texts.size(); i++) {
		msgstrs.write[i] = p_plural_xlated_texts[i];
	}

	_check_for_incompatibility(p_context, p_src_text);
	translation_map[{ p_context, p_src_text }] = msgstrs;
}

StringName Translation::get_message(const StringName &p_src_text, const StringName &p_context) const {
	StringName ret;
	if (GDVIRTUAL_CALL(_get_message, p_src_text, p_context, ret)) {
		return ret;
	}

	const Vector<StringName> *msgstrs = translation_map.getptr({ p_context, p_src_text });
	if (msgstrs == nullptr) {
		return StringName();
	}

	DEV_ASSERT(!msgstrs->is_empty()); // Should be prevented when adding messages.
	return msgstrs->get(0);
}

StringName Translation::get_plural_message(const StringName &p_src_text, const StringName &p_plural_text, int p_n, const StringName &p_context) const {
	StringName ret;
	if (GDVIRTUAL_CALL(_get_plural_message, p_src_text, p_plural_text, p_n, p_context, ret)) {
		return ret;
	}

	ERR_FAIL_COND_V_MSG(p_n < 0, StringName(), "N passed into translation to get a plural message should not be negative. For negative numbers, use singular translation please. Search \"gettext PO Plural Forms\" online for details on translating negative numbers.");

	const Vector<StringName> *msgstrs = translation_map.getptr({ p_context, p_src_text });
	if (msgstrs == nullptr) {
		return StringName();
	}

	const int index = _get_plural_rules()->evaluate(p_n);
	ERR_FAIL_INDEX_V_MSG(index, msgstrs->size(), StringName(), "Plural index returned or number of plural translations is not valid.");
	return msgstrs->get(index);
}

void Translation::erase_message(const StringName &p_src_text, const StringName &p_context) {
	const MessageKey key = { p_context, p_src_text };
	translation_map.erase(key);

	for (int type = 0; type < HINT_MAX; type++) {
		HashMap<MessageKey, String, MessageKey> *map = hint_maps[type];
		if (map) {
			map->erase(key);
		}
	}
}

void Translation::get_message_list(List<StringName> *r_messages) const {
	for (const KeyValue<MessageKey, Vector<StringName>> &E : translation_map) {
		if (E.key.msgctxt.is_empty()) {
			r_messages->push_back(E.key.msgid);
		} else {
			// Separated by the EOT character. Compatible with the MO file format.
			r_messages->push_back(vformat("%s\x04%s", E.key.msgctxt, E.key.msgid));
		}
	}
}

int Translation::get_message_count() const {
	return translation_map.size();
}

PluralRules *Translation::_get_plural_rules() const {
	if (plural_rules_cache) {
		return plural_rules_cache;
	}

	if (!plural_rules_override.is_empty()) {
		plural_rules_cache = PluralRules::parse(plural_rules_override);
	}

	if (!plural_rules_cache) {
		// Locale's default plural rules.
		const String &default_rule = TranslationServer::get_singleton()->get_plural_rules(locale);
		if (!default_rule.is_empty()) {
			plural_rules_cache = PluralRules::parse(default_rule);
		}

		// Use English plural rules as a fallback.
		if (!plural_rules_cache) {
			plural_rules_cache = PluralRules::parse("nplurals=2; plural=(n != 1);");
		}
	}

	DEV_ASSERT(plural_rules_cache != nullptr);
	return plural_rules_cache;
}

void Translation::set_plural_rules_override(const String &p_rules) {
	plural_rules_override = p_rules;
	if (plural_rules_cache) {
		memdelete(plural_rules_cache);
		plural_rules_cache = nullptr;
	}
}

String Translation::get_plural_rules_override() const {
	return plural_rules_override;
}

int Translation::get_nplurals() const {
	return _get_plural_rules()->get_nplurals();
}

String Translation::get_hint(const StringName &p_src_text, const StringName &p_context, HintType p_type) const {
	ERR_FAIL_INDEX_V(p_type, HINT_MAX, String());
	const HashMap<MessageKey, String, MessageKey> *map = hint_maps[p_type];
	if (map == nullptr) {
		return String();
	}
	const String *value = map->getptr({ p_context, p_src_text });
	if (value == nullptr) {
		return String();
	}
	return *value; // This should be non-empty technically.
}

void Translation::set_hint(const StringName &p_src_text, const StringName &p_context, HintType p_type, const String &p_value) {
	ERR_FAIL_INDEX(p_type, HINT_MAX);
	const MessageKey key = { p_context, p_src_text };

	// Empty value removes the hint entry.
	if (p_value.is_empty()) {
		HashMap<MessageKey, String, MessageKey> *map = hint_maps[p_type];
		if (map) {
			map->erase(key);
		}
		return;
	}

	// There should always be a corresponding translation for the hint.
	ERR_FAIL_COND_MSG(!translation_map.has(key), vformat("Can't set hint for '%s' with context '%s' as such a message does not exist.", p_src_text, p_context));

	if (hint_maps[p_type] == nullptr) {
		hint_maps[p_type] = memnew((HashMap<MessageKey, String, MessageKey>));
	}
	HashMap<MessageKey, String, MessageKey> &map = *hint_maps[p_type];
	map[key] = p_value;
}

void Translation::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_locale", "locale"), &Translation::set_locale);
	ClassDB::bind_method(D_METHOD("get_locale"), &Translation::get_locale);
	ClassDB::bind_method(D_METHOD("add_message", "src_message", "xlated_message", "context"), &Translation::add_message, DEFVAL(StringName()));
	ClassDB::bind_method(D_METHOD("add_plural_message", "src_message", "xlated_messages", "context"), &Translation::add_plural_message, DEFVAL(StringName()));
	ClassDB::bind_method(D_METHOD("get_message", "src_message", "context"), &Translation::get_message, DEFVAL(StringName()));
	ClassDB::bind_method(D_METHOD("get_plural_message", "src_message", "src_plural_message", "n", "context"), &Translation::get_plural_message, DEFVAL(StringName()));
	ClassDB::bind_method(D_METHOD("erase_message", "src_message", "context"), &Translation::erase_message, DEFVAL(StringName()));
	ClassDB::bind_method(D_METHOD("get_message_list"), &Translation::_get_message_list);
	ClassDB::bind_method(D_METHOD("get_translated_message_list"), &Translation::get_translated_message_list);
	ClassDB::bind_method(D_METHOD("get_message_count"), &Translation::get_message_count);
	ClassDB::bind_method(D_METHOD("_set_messages", "messages"), &Translation::_set_messages);
	ClassDB::bind_method(D_METHOD("_get_messages"), &Translation::_get_messages);
	ClassDB::bind_method(D_METHOD("set_plural_rules_override", "rules"), &Translation::set_plural_rules_override);
	ClassDB::bind_method(D_METHOD("get_plural_rules_override"), &Translation::get_plural_rules_override);

	ClassDB::bind_method(D_METHOD("get_hint", "src_message", "context", "type"), &Translation::get_hint);
	ClassDB::bind_method(D_METHOD("set_hint", "src_message", "context", "type", "value"), &Translation::set_hint);

	GDVIRTUAL_BIND(_get_plural_message, "src_message", "src_plural_message", "n", "context");
	GDVIRTUAL_BIND(_get_message, "src_message", "context");

	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "messages", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL), "_set_messages", "_get_messages");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "locale", PROPERTY_HINT_LOCALE_ID), "set_locale", "get_locale");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "plural_rules_override"), "set_plural_rules_override", "get_plural_rules_override");

	BIND_ENUM_CONSTANT(HINT_PLURAL);
	BIND_ENUM_CONSTANT(HINT_COMMENTS);
	BIND_ENUM_CONSTANT(HINT_LOCATIONS);
	BIND_ENUM_CONSTANT(HINT_MAX);
}

Translation::~Translation() {
	if (plural_rules_cache) {
		memdelete(plural_rules_cache);
		plural_rules_cache = nullptr;
	}
	for (int i = 0; i < HINT_MAX; i++) {
		if (hint_maps[i]) {
			memdelete(hint_maps[i]);
			hint_maps[i] = nullptr;
		}
	}
}
