/**************************************************************************/
/*  test_csv_translation_importer.h                                       */
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

#ifndef TEST_CSV_TRANSLATION_IMPORTER_H
#define TEST_CSV_TRANSLATION_IMPORTER_H

#include "core/string/translation.h"
#include "editor/import/resource_importer_csv_translation.h"
#include "tests/test_macros.h"
#include "tests/test_utils.h"

namespace TestCSVTranslationImporter {
TEST_CASE("[ResourceImporterCSVTranslation] CSV import generates valid translation files") {
	Ref<ResourceImporterCSVTranslation> import_csv_translation = memnew(ResourceImporterCSVTranslation);

	HashMap<StringName, Variant> options;
	options["compress"] = false;
	options["delimiter"] = 0;

	List<String> gen_files;

	Error result = import_csv_translation->import(TestUtils::get_data_path("translations/translations.csv"),
			"", options, nullptr, &gen_files);
	CHECK(result == OK);
	CHECK(gen_files.size() == 4);

	TranslationServer *ts = TranslationServer::get_singleton();
	ts->clear();

	for (const String &file : gen_files) {
		Ref<Translation> translation = ResourceLoader::load(file);
		CHECK(translation.is_valid());
		ts->add_translation(translation);
	}

	ts->set_locale("en");

	// `tr` can be called on any Object, we reuse TranslationServer for convenience.
	CHECK(ts->tr("GOOD_MORNING") == "Good Morning");
	CHECK(ts->tr("GOOD_EVENING") == "Good Evening");

	ts->set_locale("de");

	CHECK(ts->tr("GOOD_MORNING") == "Guten Morgen");
	CHECK(ts->tr("GOOD_EVENING") == "Good Evening"); // Left blank in CSV, should source from 'en'.

	ts->set_locale("ja");

	CHECK(ts->tr("GOOD_MORNING") == String::utf8("おはよう"));
	CHECK(ts->tr("GOOD_EVENING") == String::utf8("こんばんは"));
}

TEST_CASE("[ResourceImporterCSVTranslation] CSV import with badly formatted header row will successfully import with no data") {
	Ref<ResourceImporterCSVTranslation> import_csv_translation = memnew(ResourceImporterCSVTranslation);

	HashMap<StringName, Variant> options;
	options["compress"] = false;
	options["delimiter"] = 0;

	List<String> gen_files;

	ERR_PRINT_OFF;
	Error result = import_csv_translation->import(TestUtils::get_data_path("translations/translations_with_semicolons.csv"),
			"", options, nullptr, &gen_files);
	ERR_PRINT_ON;

	CHECK(result == OK);
	CHECK(gen_files.size() == 0);
}

TEST_CASE("[ResourceImporterCSVTranslation] CSV import will skip a translated string if the line is badly formed") {
	Ref<ResourceImporterCSVTranslation> import_csv_translation = memnew(ResourceImporterCSVTranslation);

	HashMap<StringName, Variant> options;
	options["compress"] = false;
	options["delimiter"] = 0;

	List<String> gen_files;

	ERR_PRINT_OFF;
	Error result = import_csv_translation->import(TestUtils::get_data_path("translations/translations_with_malformed_line.csv"),
			"", options, nullptr, &gen_files);
	ERR_PRINT_ON;

	CHECK(result == OK);
	CHECK(gen_files.size() == 2);

	TranslationServer *ts = TranslationServer::get_singleton();
	ts->clear();

	for (const String &file : gen_files) {
		Ref<Translation> translation = ResourceLoader::load(file);
		CHECK(translation.is_valid());
		ts->add_translation(translation);
	}

	ts->set_locale("es");

	// `tr` can be called on any Object, we reuse TranslationServer for convenience.
	CHECK_EQ(ts->tr("GOOD_MORNING"), "buenos dias");
	CHECK_EQ(ts->tr("GOOD_EVENING"), "GOOD_EVENING"); // Returns the given string if no translation is found
}
} //namespace TestCSVTranslationImporter

#endif // TEST_CSV_TRANSLATION_IMPORTER_H