/**************************************************************************/
/*  test_editor_translation.h                                             */
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

#ifndef TEST_EDITOR_TRANSLATION_H
#define TEST_EDITOR_TRANSLATION_H

#ifdef TOOLS_ENABLED

#include "tests/test_macros.h"

#include "core/version.h"
#include "editor/editor_settings.h"

namespace EditorTranslationTests {

static String build_url(String &p_language) {
	String version = (p_language == "en") ? VERSION_DOCS_BRANCH : vformat("%s.x", VERSION_MAJOR);
	return vformat(VERSION_DOCS_URL "%s/%s/", p_language, version);
}

TEST_CASE("[Editor][EditorSettings][Translation] Docs URL changes based on editor language settings") {
	String english = "en";
	String translated = "fr"; // French
	String untranslated = "gl"; // Galician

	SUBCASE("Editor language is English, online docs language is auto.") {
		EditorSettings::get_singleton()->set_setting("interface/editor/editor_language", english);
		EditorSettings::get_singleton()->set_setting("interface/editor/online_docs_language", "auto");
		CHECK(EditorSettings::get_singleton()->get_online_docs_url() == build_url(english));
	}

	SUBCASE("Editor language is English, online docs language is overridden.") {
		EditorSettings::get_singleton()->set_setting("interface/editor/editor_language", english);
		EditorSettings::get_singleton()->set_setting("interface/editor/online_docs_language", translated);
		CHECK(EditorSettings::get_singleton()->get_online_docs_url() == build_url(translated));
	}

	SUBCASE("Editor language has translated online docs, online docs language is auto.") {
		EditorSettings::get_singleton()->set_setting("interface/editor/editor_language", translated);
		EditorSettings::get_singleton()->set_setting("interface/editor/online_docs_language", "auto");
		CHECK(EditorSettings::get_singleton()->get_online_docs_url() == build_url(translated));
	}

	SUBCASE("Editor language has translated online docs, online docs language is overridden.") {
		EditorSettings::get_singleton()->set_setting("interface/editor/editor_language", translated);
		EditorSettings::get_singleton()->set_setting("interface/editor/online_docs_language", english);
		CHECK(EditorSettings::get_singleton()->get_online_docs_url() == build_url(english));
	}

	SUBCASE("Editor language has untranslated online docs, online docs language is auto.") {
		EditorSettings::get_singleton()->set_setting("interface/editor/editor_language", untranslated);
		EditorSettings::get_singleton()->set_setting("interface/editor/online_docs_language", "auto");
		CHECK(EditorSettings::get_singleton()->get_online_docs_url() == build_url(english));
	}

	SUBCASE("Editor language has untranslated online docs, online docs language is overridden.") {
		EditorSettings::get_singleton()->set_setting("interface/editor/editor_language", untranslated);
		EditorSettings::get_singleton()->set_setting("interface/editor/online_docs_language", translated);
		CHECK(EditorSettings::get_singleton()->get_online_docs_url() == build_url(translated));
	}
}

} // namespace EditorTranslationTests

#endif // TOOLS_ENABLED

#endif // TEST_EDITOR_TRANSLATION_H
