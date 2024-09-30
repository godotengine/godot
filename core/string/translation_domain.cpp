/**************************************************************************/
/*  translation_domain.cpp                                                */
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

#include "translation_domain.h"

#include "core/string/translation.h"
#include "core/string/translation_server.h"

StringName TranslationDomain::get_message_from_translations(const String &p_locale, const StringName &p_message, const StringName &p_context) const {
	StringName res;
	int best_score = 0;

	for (const Ref<Translation> &E : translations) {
		ERR_CONTINUE(E.is_null());
		int score = TranslationServer::get_singleton()->compare_locales(p_locale, E->get_locale());
		if (score > 0 && score >= best_score) {
			const StringName r = E->get_message(p_message, p_context);
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

StringName TranslationDomain::get_message_from_translations(const String &p_locale, const StringName &p_message, const StringName &p_message_plural, int p_n, const StringName &p_context) const {
	StringName res;
	int best_score = 0;

	for (const Ref<Translation> &E : translations) {
		ERR_CONTINUE(E.is_null());
		int score = TranslationServer::get_singleton()->compare_locales(p_locale, E->get_locale());
		if (score > 0 && score >= best_score) {
			const StringName r = E->get_plural_message(p_message, p_message_plural, p_n, p_context);
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

PackedStringArray TranslationDomain::get_loaded_locales() const {
	PackedStringArray locales;
	for (const Ref<Translation> &E : translations) {
		ERR_CONTINUE(E.is_null());
		locales.push_back(E->get_locale());
	}
	return locales;
}

Ref<Translation> TranslationDomain::get_translation_object(const String &p_locale) const {
	Ref<Translation> res;
	int best_score = 0;

	for (const Ref<Translation> &E : translations) {
		ERR_CONTINUE(E.is_null());

		int score = TranslationServer::get_singleton()->compare_locales(p_locale, E->get_locale());
		if (score > 0 && score >= best_score) {
			res = E;
			best_score = score;
			if (score == 10) {
				break; // Exact match, skip the rest.
			}
		}
	}
	return res;
}

void TranslationDomain::add_translation(const Ref<Translation> &p_translation) {
	translations.insert(p_translation);
}

void TranslationDomain::remove_translation(const Ref<Translation> &p_translation) {
	translations.erase(p_translation);
}

void TranslationDomain::clear() {
	translations.clear();
}

StringName TranslationDomain::translate(const StringName &p_message, const StringName &p_context) const {
	const String &locale = TranslationServer::get_singleton()->get_locale();
	StringName res = get_message_from_translations(locale, p_message, p_context);

	const String &fallback = TranslationServer::get_singleton()->get_fallback_locale();
	if (!res && fallback.length() >= 2) {
		res = get_message_from_translations(fallback, p_message, p_context);
	}

	if (!res) {
		return p_message;
	}
	return res;
}

StringName TranslationDomain::translate_plural(const StringName &p_message, const StringName &p_message_plural, int p_n, const StringName &p_context) const {
	const String &locale = TranslationServer::get_singleton()->get_locale();
	StringName res = get_message_from_translations(locale, p_message, p_message_plural, p_n, p_context);

	const String &fallback = TranslationServer::get_singleton()->get_fallback_locale();
	if (!res && fallback.length() >= 2) {
		res = get_message_from_translations(fallback, p_message, p_message_plural, p_n, p_context);
	}

	if (!res) {
		if (p_n == 1) {
			return p_message;
		}
		return p_message_plural;
	}
	return res;
}

void TranslationDomain::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_translation_object", "locale"), &TranslationDomain::get_translation_object);
	ClassDB::bind_method(D_METHOD("add_translation", "translation"), &TranslationDomain::add_translation);
	ClassDB::bind_method(D_METHOD("remove_translation", "translation"), &TranslationDomain::remove_translation);
	ClassDB::bind_method(D_METHOD("clear"), &TranslationDomain::clear);
	ClassDB::bind_method(D_METHOD("translate", "message", "context"), &TranslationDomain::translate, DEFVAL(StringName()));
	ClassDB::bind_method(D_METHOD("translate_plural", "message", "message_plural", "n", "context"), &TranslationDomain::translate_plural, DEFVAL(StringName()));
}

TranslationDomain::TranslationDomain() {
}
