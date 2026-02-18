/**************************************************************************/
/*  test_translation.h                                                    */
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

#include "core/string/optimized_translation.h"
#include "core/string/plural_rules.h"
#include "core/string/translation.h"
#include "core/string/translation_server.h"

#ifdef TOOLS_ENABLED
#include "editor/import/resource_importer_csv_translation.h"
#endif

#include "tests/test_macros.h"
#include "tests/test_utils.h"

namespace TestTranslation {

TEST_CASE("[Translation] Messages") {
	Ref<Translation> translation;
	translation.instantiate();
	translation->set_locale("fr");
	translation->add_message("Hello", "Bonjour");
	CHECK(translation->get_message("Hello") == "Bonjour");

	translation->erase_message("Hello");
	// The message no longer exists, so it returns an empty string instead.
	CHECK(translation->get_message("Hello") == "");

	List<StringName> messages;
	translation->get_message_list(&messages);
	CHECK(translation->get_message_count() == 0);
	CHECK(messages.size() == 0);

	translation->add_message("Hello2", "Bonjour2");
	translation->add_message("Hello3", "Bonjour3");
	messages.clear();
	translation->get_message_list(&messages);
	CHECK(translation->get_message_count() == 2);
	CHECK(messages.size() == 2);
	// Messages are stored in a Map, don't assume ordering.
	CHECK(messages.find("Hello2"));
	CHECK(messages.find("Hello3"));
}

TEST_CASE("[Translation] Messages with context") {
	Ref<Translation> translation;
	translation.instantiate();
	translation->set_locale("fr");
	translation->add_message("Hello", "Bonjour");
	translation->add_message("Hello", "Salut", "friendly");
	CHECK(translation->get_message("Hello") == "Bonjour");
	CHECK(translation->get_message("Hello", "friendly") == "Salut");
	CHECK(translation->get_message("Hello", "nonexistent_context") == "");

	// Only remove the message for the default context, not the "friendly" context.
	translation->erase_message("Hello");
	// The message no longer exists, so it returns an empty string instead.
	CHECK(translation->get_message("Hello") == "");
	CHECK(translation->get_message("Hello", "friendly") == "Salut");
	CHECK(translation->get_message("Hello", "nonexistent_context") == "");

	List<StringName> messages;
	translation->get_message_list(&messages);

	CHECK(translation->get_message_count() == 1);
	CHECK(messages.size() == 1);

	translation->add_message("Hello2", "Bonjour2");
	translation->add_message("Hello2", "Salut2", "friendly");
	translation->add_message("Hello3", "Bonjour3");
	messages.clear();
	translation->get_message_list(&messages);

	CHECK(translation->get_message_count() == 4);
	CHECK(messages.size() == 4);
	// Messages are stored in a Map, don't assume ordering.
	CHECK(messages.find("Hello2"));
	CHECK(messages.find("Hello3"));
	// Context and untranslated string are separated by EOT.
	CHECK(messages.find("friendly\x04Hello2"));
}

TEST_CASE("[Translation] Plural messages") {
	{
		Ref<Translation> translation;
		translation.instantiate();
		translation->set_locale("fr");
		CHECK(translation->get_nplurals() == 2);
	}

	{
		Ref<Translation> translation;
		translation.instantiate();
		translation->set_locale("invalid");
		CHECK(translation->get_nplurals() == 2);
	}

	{
		Ref<Translation> translation;
		translation.instantiate();
		translation->set_plural_rules_override("Plural-Forms: nplurals=2; plural=(n >= 2);");
		CHECK(translation->get_nplurals() == 2);

		PackedStringArray plurals;
		plurals.push_back("Il y a %d pomme");
		plurals.push_back("Il y a %d pommes");
		translation->add_plural_message("There are %d apples", plurals);
		ERR_PRINT_OFF;
		// This is invalid, as the number passed to `get_plural_message()` may not be negative.
		CHECK(vformat(translation->get_plural_message("There are %d apples", "", -1), -1) == "");
		ERR_PRINT_ON;
		CHECK(vformat(translation->get_plural_message("There are %d apples", "", 0), 0) == "Il y a 0 pomme");
		CHECK(vformat(translation->get_plural_message("There are %d apples", "", 1), 1) == "Il y a 1 pomme");
		CHECK(vformat(translation->get_plural_message("There are %d apples", "", 2), 2) == "Il y a 2 pommes");
	}
}

TEST_CASE("[Translation] Plural rules parsing") {
	ERR_PRINT_OFF;
	{
		CHECK(PluralRules::parse("") == nullptr);

		CHECK(PluralRules::parse("plurals=(n != 1);") == nullptr);
		CHECK(PluralRules::parse("nplurals; plurals=(n != 1);") == nullptr);
		CHECK(PluralRules::parse("nplurals=; plurals=(n != 1);") == nullptr);
		CHECK(PluralRules::parse("nplurals=0; plurals=(n != 1);") == nullptr);
		CHECK(PluralRules::parse("nplurals=-1; plurals=(n != 1);") == nullptr);

		CHECK(PluralRules::parse("nplurals=2;") == nullptr);
		CHECK(PluralRules::parse("nplurals=2; plurals;") == nullptr);
		CHECK(PluralRules::parse("nplurals=2; plurals=;") == nullptr);
	}
	ERR_PRINT_ON;

	{
		PluralRules *pr = PluralRules::parse("nplurals=3; plural=(n==0 ? 0 : n==1 ? 1 : 2);");
		REQUIRE(pr != nullptr);

		CHECK(pr->get_nplurals() == 3);
		CHECK(pr->get_plural() == "(n==0 ? 0 : n==1 ? 1 : 2)");

		CHECK(pr->evaluate(0) == 0);
		CHECK(pr->evaluate(1) == 1);
		CHECK(pr->evaluate(2) == 2);
		CHECK(pr->evaluate(3) == 2);

		memdelete(pr);
	}

	{
		PluralRules *pr = PluralRules::parse("nplurals=1; plural=0;");
		REQUIRE(pr != nullptr);

		CHECK(pr->get_nplurals() == 1);
		CHECK(pr->get_plural() == "0");

		CHECK(pr->evaluate(0) == 0);
		CHECK(pr->evaluate(1) == 0);
		CHECK(pr->evaluate(2) == 0);
		CHECK(pr->evaluate(3) == 0);

		memdelete(pr);
	}
}

#ifdef TOOLS_ENABLED
TEST_CASE("[OptimizedTranslation] Generate from Translation and read messages") {
	Ref<Translation> translation = memnew(Translation);
	translation->set_locale("fr");
	translation->add_message("Hello", "Bonjour");
	translation->add_message("Hello2", "Bonjour2");
	translation->add_message("Hello3", "Bonjour3");

	Ref<OptimizedTranslation> optimized_translation = memnew(OptimizedTranslation);
	optimized_translation->generate(translation);
	CHECK(optimized_translation->get_message("Hello") == "Bonjour");
	CHECK(optimized_translation->get_message("Hello2") == "Bonjour2");
	CHECK(optimized_translation->get_message("Hello3") == "Bonjour3");
	CHECK(optimized_translation->get_message("DoesNotExist") == "");

	List<StringName> messages;
	// `get_message_list()` can't return the list of messages stored in an OptimizedTranslation.
	optimized_translation->get_message_list(&messages);
	CHECK(optimized_translation->get_message_count() == 0);
	CHECK(messages.size() == 0);
}

TEST_CASE("[OptimizedTranslation] Generates translatoins and checks get_translated_message_list") {
	Ref<Translation> translation = memnew(Translation);
	translation->set_locale("fr");
	translation->add_message("Hello", "Bonjour");
	translation->add_message("Hello2", "Bonjour2");
	translation->add_message("Hello3", "Bonjour3");

	Ref<OptimizedTranslation> optimized_translation = memnew(OptimizedTranslation);
	optimized_translation->generate(translation);
	

	CHECK(optimized_translation->get_translated_message_list().size() == 3);
	CHECK(optimized_translation->get_translated_message_list().find("Bonjour") != -1);
	CHECK(optimized_translation->get_translated_message_list().find("Bonjour2") != -1);
	CHECK(optimized_translation->get_translated_message_list().find("Bonjour3") != -1);
	CHECK(optimized_translation->get_translated_message_list().find("NotInList") == -1);
}

TEST_CASE("[TranslationCSV] CSV import") {
	Ref<ResourceImporterCSVTranslation> import_csv_translation = memnew(ResourceImporterCSVTranslation);

	HashMap<StringName, Variant> options;
	options["compress"] = false;
	options["delimiter"] = 0;

	List<String> gen_files;

	Error result = import_csv_translation->import(0, TestUtils::get_data_path("translations.csv"),
			"", options, nullptr, &gen_files);
	CHECK(result == OK);
	CHECK(gen_files.size() == 4);

	Ref<TranslationDomain> td = TranslationServer::get_singleton()->get_or_add_domain("godot.test");
	for (const String &file : gen_files) {
		Ref<Translation> translation = ResourceLoader::load(file);
		CHECK(translation.is_valid());
		td->add_translation(translation);
	}

	td->set_locale_override("en");

	CHECK(td->translate("GOOD_MORNING", StringName()) == "Good Morning");
	CHECK(td->translate("GOOD_EVENING", StringName()) == "Good Evening");

	td->set_locale_override("de");

	CHECK(td->translate("GOOD_MORNING", StringName()) == "Guten Morgen");
	CHECK(td->translate("GOOD_EVENING", StringName()) == "Good Evening"); // Left blank in CSV, should source from 'en'.

	td->set_locale_override("ja");

	CHECK(td->translate("GOOD_MORNING", StringName()) == String::utf8("おはよう"));
	CHECK(td->translate("GOOD_EVENING", StringName()) == String::utf8("こんばんは"));

	/* FIXME: This passes, but triggers a chain reaction that makes test_viewport
	 * and test_text_edit explode in a billion glittery Unicode particles.
	td->set_locale_override("fa");

	CHECK(td->translate("GOOD_MORNING", String()) == String::utf8("صبح بخیر"));
	CHECK(td->translate("GOOD_EVENING", String()) == String::utf8("عصر بخیر"));
	*/

	TranslationServer::get_singleton()->remove_domain("godot.test");
}
#endif // TOOLS_ENABLED

} // namespace TestTranslation
