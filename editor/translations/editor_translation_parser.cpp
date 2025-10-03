/**************************************************************************/
/*  editor_translation_parser.cpp                                         */
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

#include "editor_translation_parser.h"

#include "core/error/error_macros.h"
#include "core/object/script_language.h"
#include "core/templates/hash_set.h"
#include "editor/translations/editor_translation.h"

EditorTranslationParser *EditorTranslationParser::singleton = nullptr;

Error EditorTranslationParserPlugin::parse_file(const String &p_path, Vector<Vector<String>> *r_translations) {
	TypedArray<PackedStringArray> ret;

	if (GDVIRTUAL_CALL(_parse_file, p_path, ret)) {
		// Copy over entries directly.
		for (const PackedStringArray translation : ret) {
			r_translations->push_back(translation);
		}

		return OK;
	}

#ifndef DISABLE_DEPRECATED
	TypedArray<String> ids;
	TypedArray<Array> ids_ctx_plural;

	if (GDVIRTUAL_CALL(_parse_file_bind_compat_99297, p_path, ids, ids_ctx_plural)) {
		// Add user's extracted translatable messages.
		for (int i = 0; i < ids.size(); i++) {
			r_translations->push_back({ ids[i] });
		}

		// Add user's collected translatable messages with context or plurals.
		for (int i = 0; i < ids_ctx_plural.size(); i++) {
			Array arr = ids_ctx_plural[i];
			ERR_FAIL_COND_V_MSG(arr.size() != 3, ERR_INVALID_DATA, R"*(Array entries written into "msgids_context_plural" in "_parse_file()" method should have the form ["message", "context", "plural message"].)*");

			r_translations->push_back({ arr[0], arr[1], arr[2] });
		}
		return OK;
	}
#endif // DISABLE_DEPRECATED

	ERR_PRINT(R"*(Custom translation parser plugin's "_parse_file()" method is undefined.)*");
	return ERR_UNAVAILABLE;
}

void EditorTranslationParserPlugin::get_recognized_extensions(List<String> *r_extensions) const {
	Vector<String> extensions;
	if (GDVIRTUAL_CALL(_get_recognized_extensions, extensions)) {
		for (int i = 0; i < extensions.size(); i++) {
			r_extensions->push_back(extensions[i]);
		}
	} else {
		ERR_PRINT(R"*(Custom translation parser plugin's "_get_recognized_extensions()" method is undefined.)*");
	}
}

PackedStringArray EditorTranslationParserPlugin::get_all_recognized_extensions_bind() {
	List<String> extensions;
	EditorTranslationParser::get_singleton()->get_recognized_extensions(&extensions);

	PackedStringArray arr;
	for (const String &extension : extensions) {
		arr.push_back(extension);
	}

	return arr;
}

Ref<EditorTranslationParserPlugin> EditorTranslationParserPlugin::get_parser_bind(const String &p_extension, bool p_standard_only) {
	return EditorTranslationParser::get_singleton()->get_parser(p_extension, p_standard_only);
}

TypedArray<PackedStringArray> EditorTranslationParserPlugin::get_builtin_strings_bind() {
	const Vector<Vector<String>> messages = get_extractable_message_list();

	TypedArray<PackedStringArray> arr;
	arr.resize(messages.size());
	for (int i = 0; i < messages.size(); i++) {
		arr[i] = messages[i];
	}

	return arr;
}

TypedArray<PackedStringArray> EditorTranslationParserPlugin::parse_file_bind(const String &p_path) {
	Vector<Vector<String>> translations;
	parse_file(p_path, &translations);

	TypedArray<PackedStringArray> arr;
	arr.resize(translations.size());
	for (int i = 0; i < translations.size(); i++) {
		arr[i] = translations[i];
	}

	return arr;
}

PackedStringArray EditorTranslationParserPlugin::get_recognized_extensions_bind() const {
	List<String> extensions;
	get_recognized_extensions(&extensions);

	PackedStringArray arr;
	for (const String &extension : extensions) {
		arr.push_back(extension);
	}

	return arr;
}

void EditorTranslationParserPlugin::_bind_methods() {
	GDVIRTUAL_BIND(_parse_file, "path");
	GDVIRTUAL_BIND(_get_recognized_extensions);

#ifndef DISABLE_DEPRECATED
	GDVIRTUAL_BIND_COMPAT(_parse_file_bind_compat_99297, "path", "msgids", "msgids_context_plural");
#endif

	ClassDB::bind_static_method("EditorTranslationParserPlugin", D_METHOD("get_all_recognized_extensions"), &EditorTranslationParserPlugin::get_all_recognized_extensions_bind);
	ClassDB::bind_static_method("EditorTranslationParserPlugin", D_METHOD("get_parser", "extension", "standard_only"), &EditorTranslationParserPlugin::get_parser_bind, DEFVAL(false));
	ClassDB::bind_static_method("EditorTranslationParserPlugin", D_METHOD("get_builtin_strings"), &EditorTranslationParserPlugin::get_builtin_strings_bind);

	ClassDB::bind_method(D_METHOD("parse_file", "path"), &EditorTranslationParserPlugin::parse_file_bind);
	ClassDB::bind_method(D_METHOD("get_recognized_extensions"), &EditorTranslationParserPlugin::get_recognized_extensions_bind);
}

/////////////////////////

void EditorTranslationParser::get_recognized_extensions(List<String> *r_extensions) const {
	HashSet<String> extensions;
	List<String> temp;
	for (int i = 0; i < standard_parsers.size(); i++) {
		standard_parsers[i]->get_recognized_extensions(&temp);
	}
	for (int i = 0; i < custom_parsers.size(); i++) {
		custom_parsers[i]->get_recognized_extensions(&temp);
	}
	// Remove duplicates.
	for (const String &E : temp) {
		extensions.insert(E);
	}
	for (const String &E : extensions) {
		r_extensions->push_back(E);
	}
}

bool EditorTranslationParser::can_parse(const String &p_extension) const {
	List<String> extensions;
	get_recognized_extensions(&extensions);
	for (const String &extension : extensions) {
		if (p_extension == extension) {
			return true;
		}
	}
	return false;
}

Ref<EditorTranslationParserPlugin> EditorTranslationParser::get_parser(const String &p_extension, bool p_standard_only) const {
	if (!p_standard_only) {
		// Consider user-defined parsers first.
		for (int i = 0; i < custom_parsers.size(); i++) {
			List<String> extensions;
			custom_parsers[i]->get_recognized_extensions(&extensions);
			for (const String &extension : extensions) {
				if (extension == p_extension) {
					return custom_parsers[i];
				}
			}
		}
	}

	for (int i = 0; i < standard_parsers.size(); i++) {
		List<String> extensions;
		standard_parsers[i]->get_recognized_extensions(&extensions);
		for (const String &extension : extensions) {
			if (extension == p_extension) {
				return standard_parsers[i];
			}
		}
	}

	WARN_PRINT(vformat(R"(No translation parser available for "%s" extension.)", p_extension));

	return nullptr;
}

void EditorTranslationParser::add_parser(const Ref<EditorTranslationParserPlugin> &p_parser, ParserType p_type) {
	ERR_FAIL_COND_MSG(standard_parsers.has(p_parser) || custom_parsers.has(p_parser), "Сannot add an already registered translation parser.");

	switch (p_type) {
		case STANDARD:
			standard_parsers.push_back(p_parser);
			break;
		case CUSTOM:
			custom_parsers.push_back(p_parser);
			break;
	}
}

void EditorTranslationParser::remove_parser(const Ref<EditorTranslationParserPlugin> &p_parser, ParserType p_type) {
	ERR_FAIL_COND_MSG(!standard_parsers.has(p_parser) && !custom_parsers.has(p_parser), "Сannot remove an unregistered translation parser.");

	switch (p_type) {
		case STANDARD:
			ERR_FAIL_COND_MSG(custom_parsers.has(p_parser), "Trying to remove a custom parser as a standard one."); // Only possible if there is a bug.
			standard_parsers.erase(p_parser);
			break;
		case CUSTOM:
			ERR_FAIL_COND_MSG(standard_parsers.has(p_parser), "Сannot remove a standard translation parser."); // The user may try to do this.
			custom_parsers.erase(p_parser);
			break;
	}
}

void EditorTranslationParser::clean_parsers() {
	standard_parsers.clear();
	custom_parsers.clear();
}

EditorTranslationParser *EditorTranslationParser::get_singleton() {
	if (!singleton) {
		singleton = memnew(EditorTranslationParser);
	}
	return singleton;
}

EditorTranslationParser::~EditorTranslationParser() {
	memdelete(singleton);
	singleton = nullptr;
}
