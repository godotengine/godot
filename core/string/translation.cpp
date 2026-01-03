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

void _check_for_incompatibility(const String &p_msgctxt, const String &p_msgid) {
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

Dictionary Translation::_get_messages() const {
	Dictionary d;
	for (const KeyValue<MessageKey, Vector<StringName>> &E : translation_map) {
		const Array &storage_key = { E.key.msgctxt, E.key.msgid };

		Array storage_value;
		storage_value.resize(E.value.size());
		for (int i = 0; i < E.value.size(); i++) {
			storage_value[i] = E.value[i];
		}
		d[storage_key] = storage_value;
	}
	return d;
}

void Translation::_set_messages(const Dictionary &p_messages) {
	translation_map.clear();

	for (const KeyValue<Variant, Variant> &kv : p_messages) {
		switch (kv.key.get_type()) {
			// Old version, no context or plural support.
			case Variant::STRING:
			case Variant::STRING_NAME: {
				const MessageKey msg_key = { StringName(), kv.key };
				_check_for_incompatibility(msg_key.msgctxt, msg_key.msgid);
				translation_map[msg_key] = { kv.value };
			} break;

			// Current version.
			case Variant::ARRAY: {
				const Array &storage_key = kv.key;
				const MessageKey msg_key = { storage_key[0], storage_key[1] };

				const Array &storage_value = kv.value;
				ERR_CONTINUE_MSG(storage_value.is_empty(), vformat("No translated strings for untranslated string '%s' with context '%s'.", msg_key.msgid, msg_key.msgctxt));

				Vector<StringName> msgstrs;
				msgstrs.resize(storage_value.size());
				for (int i = 0; i < storage_value.size(); i++) {
					msgstrs.write[i] = storage_value[i];
				}

				_check_for_incompatibility(msg_key.msgctxt, msg_key.msgid);
				translation_map[msg_key] = msgstrs;
			} break;

			default: {
				WARN_PRINT(vformat("Invalid key type in messages dictionary: %s.", Variant::get_type_name(kv.key.get_type())));
				continue;
			}
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
	translation_map.erase({ p_context, p_src_text });
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

	GDVIRTUAL_BIND(_get_plural_message, "src_message", "src_plural_message", "n", "context");
	GDVIRTUAL_BIND(_get_message, "src_message", "context");

	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "messages", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL), "_set_messages", "_get_messages");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "locale", PROPERTY_HINT_LOCALE_ID), "set_locale", "get_locale");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "plural_rules_override"), "set_plural_rules_override", "get_plural_rules_override");
}

Translation::~Translation() {
	if (plural_rules_cache) {
		memdelete(plural_rules_cache);
		plural_rules_cache = nullptr;
	}
}
