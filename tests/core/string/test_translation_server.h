/**************************************************************************/
/*  test_translation_server.h                                             */
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

#pragma once

#include "core/string/translation_server.h"

#include "tests/test_macros.h"

namespace TestTranslationServer {
TEST_CASE("[TranslationServer] Translation operations") {
	Ref<TranslationDomain> td = TranslationServer::get_singleton()->get_or_add_domain("godot.test");
	CHECK(td->get_translations().is_empty());

	Ref<Translation> t1 = memnew(Translation);
	t1->set_locale("uk"); // Ukrainian.
	t1->add_message("Good Morning", String(U"Добрий ранок"));
	td->add_translation(t1);
	CHECK(td->get_translations().size() == 1);
	CHECK(td->has_translation_for_locale("uk", true));
	CHECK(td->has_translation_for_locale("uk", false));
	CHECK_FALSE(td->has_translation_for_locale("uk_UA", true));
	CHECK(td->has_translation_for_locale("uk_UA", false));
	CHECK(td->find_translations("uk", false).size() == 1);
	CHECK(td->find_translations("uk", true).size() == 1);
	CHECK(td->find_translations("uk_UA", false).size() == 1);
	CHECK(td->find_translations("uk_UA", true).size() == 0);

	Ref<Translation> t2 = memnew(Translation);
	t2->set_locale("uk_UA"); // Ukrainian in Ukraine.
	t2->add_message("Hello Godot", String(U"Привіт, Годо."));
	td->add_translation(t2);
	CHECK(td->get_translations().size() == 2);
	CHECK(td->has_translation_for_locale("uk", true));
	CHECK(td->has_translation_for_locale("uk", false));
	CHECK(td->has_translation_for_locale("uk_UA", true));
	CHECK(td->has_translation_for_locale("uk_UA", false));
	CHECK(td->find_translations("uk", false).size() == 2);
	CHECK(td->find_translations("uk", true).size() == 1);
	CHECK(td->find_translations("uk_UA", false).size() == 2);
	CHECK(td->find_translations("uk_UA", true).size() == 1);

	td->set_locale_override("uk");
	CHECK(td->translate("Good Morning", StringName()) == String::utf8("Добрий ранок"));

	td->remove_translation(t1);
	CHECK(td->get_translations().size() == 1);
	CHECK_FALSE(td->has_translation_for_locale("uk", true));
	CHECK(td->has_translation_for_locale("uk", false));
	CHECK(td->has_translation_for_locale("uk_UA", true));
	CHECK(td->has_translation_for_locale("uk_UA", false));
	CHECK(td->find_translations("uk", true).size() == 0);
	CHECK(td->find_translations("uk", false).size() == 1);
	CHECK(td->find_translations("uk_UA", true).size() == 1);
	CHECK(td->find_translations("uk_UA", false).size() == 1);

	// If no suitable Translation object has been found - the original message should be returned.
	CHECK(td->translate("Good Morning", StringName()) == "Good Morning");

	TranslationServer::get_singleton()->remove_domain("godot.test");
}

TEST_CASE("[TranslationServer] Locale operations") {
	TranslationServer *ts = TranslationServer::get_singleton();

	// Language variant test; we supplied the variant of Español and the result should be the same string.
	String loc = "es_Hani_ES_tradnl";
	String res = ts->standardize_locale(loc);

	CHECK(res == loc);

	// No such variant in variant_map; should return everything except the variant.
	loc = "es_Hani_ES_missing";
	res = ts->standardize_locale(loc);

	CHECK(res == "es_Hani_ES");

	// Non-ISO language name check (Windows issue).
	loc = "iw_Hani_IL";
	res = ts->standardize_locale(loc);

	CHECK(res == "he_Hani_IL");

	// Country rename check.
	loc = "uk_Hani_UK";
	res = ts->standardize_locale(loc);

	CHECK(res == "uk_Hani_GB");

	// Supplying a script name that is not in the list.
	loc = "de_Wrong_DE";
	res = ts->standardize_locale(loc);

	CHECK(res == "de_DE");

	// No added defaults.
	loc = "es_ES";
	res = ts->standardize_locale(loc, true);

	CHECK(res == "es_ES");

	// Add default script.
	loc = "az_AZ";
	res = ts->standardize_locale(loc, true);

	CHECK(res == "az_Latn_AZ");

	// Add default country.
	loc = "pa_Arab";
	res = ts->standardize_locale(loc, true);

	CHECK(res == "pa_Arab_PK");

	// Add default script and country.
	loc = "zh";
	res = ts->standardize_locale(loc, true);

	CHECK(res == "zh_Hans_CN");

	// Explicitly don't add defaults.
	loc = "zh";
	res = ts->standardize_locale(loc, false);

	CHECK(res == "zh");
}

TEST_CASE("[TranslationServer] Comparing locales") {
	TranslationServer *ts = TranslationServer::get_singleton();

	String locale_a = "es";
	String locale_b = "es";

	// Exact match check.
	int res = ts->compare_locales(locale_a, locale_b);

	CHECK(res == 10);

	locale_a = "sr-Latn-CS";
	locale_b = "sr-Latn-RS";

	// Script matches (+1) but country doesn't (-1).
	res = ts->compare_locales(locale_a, locale_b);

	CHECK(res == 5);

	locale_a = "uz-Cyrl-UZ";
	locale_b = "uz-Latn-UZ";

	// Country matches (+1) but script doesn't (-1).
	res = ts->compare_locales(locale_a, locale_b);

	CHECK(res == 5);

	locale_a = "aa-Latn-ER";
	locale_b = "aa-Latn-ER-saaho";

	// Script and country match (+2) with variant on one locale (+0).
	res = ts->compare_locales(locale_a, locale_b);

	CHECK(res == 7);

	locale_a = "uz-Cyrl-UZ";
	locale_b = "uz-Latn-KG";

	// Both script and country mismatched (-2).
	res = ts->compare_locales(locale_a, locale_b);

	CHECK(res == 3);

	locale_a = "es-ES";
	locale_b = "es-AR";

	// Mismatched country (-1).
	res = ts->compare_locales(locale_a, locale_b);

	CHECK(res == 4);

	locale_a = "es";
	locale_b = "es-AR";

	// No country for one locale (+0).
	res = ts->compare_locales(locale_a, locale_b);

	CHECK(res == 5);

	locale_a = "es-EC";
	locale_b = "fr-LU";

	// No match.
	res = ts->compare_locales(locale_a, locale_b);

	CHECK(res == 0);

	locale_a = "zh-HK";
	locale_b = "zh";

	// In full standardization, zh-HK becomes zh_Hant_HK and zh becomes
	// zh_Hans_CN. Both script and country mismatch (-2).
	res = ts->compare_locales(locale_a, locale_b);

	CHECK(res == 3);

	locale_a = "zh-CN";
	locale_b = "zh";

	// In full standardization, zh and zh-CN both become zh_Hans_CN for an
	// exact match.
	res = ts->compare_locales(locale_a, locale_b);

	CHECK(res == 10);
}

TEST_CASE("[TranslationServer] Language, country, and locale names") {
	TranslationServer *ts = TranslationServer::get_singleton();
	CHECK(ts->get_language_name("fr") == "French");
	CHECK(ts->get_country_name("FR") == "France");
	CHECK(ts->get_locale_name("fr") == "French");
	CHECK(ts->get_locale_name("fr_FR") == "French (France)");
	CHECK(ts->get_native_locale_name("fr") == U"français");
	CHECK(ts->get_native_locale_name("fr_FR") == U"français (France)");
}
} // namespace TestTranslationServer
