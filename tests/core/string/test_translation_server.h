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

#ifndef TEST_TRANSLATION_SERVER_H
#define TEST_TRANSLATION_SERVER_H

#include "core/string/translation_server.h"

#include "tests/test_macros.h"

namespace TestTranslationServer {
TEST_CASE("[TranslationServer] Translation operations") {
	Ref<Translation> t = memnew(Translation);
	t->set_locale("uk");
	t->add_message("Good Morning", String::utf8("Добрий ранок"));

	TranslationServer *ts = TranslationServer::get_singleton();

	int l_count_before = ts->get_loaded_locales().size();
	ts->add_translation(t);
	int l_count_after = ts->get_loaded_locales().size();
	// Newly created Translation object should be added to the list, so the counter should increase, too.
	CHECK(l_count_after > l_count_before);

	Ref<Translation> trans = ts->get_translation_object("uk");
	CHECK(trans.is_valid());

	ts->set_locale("uk");
	CHECK(ts->translate("Good Morning") == String::utf8("Добрий ранок"));

	ts->remove_translation(t);
	trans = ts->get_translation_object("uk");
	CHECK(trans.is_null());
	// If no suitable Translation object has been found - the original message should be returned.
	CHECK(ts->translate("Good Morning") == "Good Morning");
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

	// Two elements from locales match.
	res = ts->compare_locales(locale_a, locale_b);

	CHECK(res == 2);

	locale_a = "uz-Cyrl-UZ";
	locale_b = "uz-Latn-UZ";

	// Two elements match, but they are not sequentual.
	res = ts->compare_locales(locale_a, locale_b);

	CHECK(res == 2);

	locale_a = "es-EC";
	locale_b = "fr-LU";

	// No match.
	res = ts->compare_locales(locale_a, locale_b);

	CHECK(res == 0);
}
} // namespace TestTranslationServer

#endif // TEST_TRANSLATION_SERVER_H
