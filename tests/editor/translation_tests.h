/**************************************************************************/
/*  translation_tests.h                                                   */
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

#include "core/io/compression.h"
#include "core/io/file_access_memory.h"
#include "core/io/translation_loader_po.h"
#include "core/string/translation.h"
#include "editor/translations/doc_translations.gen.h"
#include "editor/translations/editor_translations.gen.h"
#include "editor/translations/extractable_translations.gen.h"
#include "editor/translations/property_translations.gen.h"

#include "tests/test_macros.h"

namespace TestEditorTranslation {

Vector<int> _util_sprintf_extract_arg_types(const StringName &p_str) {
	static const String ZERO("0");
	static const String SPACE(" ");
	static const String MINUS("-");
	static const String PLUS("+");

	Vector<int> value_types;
	char32_t *self = (char32_t *)p_str.get_data();
	int selected_index = -1;
	int pending_index = 0;
	bool in_decimals = false;
	int value_index = 0;
	bool in_format = false;

	for (; *self; self++) {
		const char32_t c = *self;

		if (in_format) { // We have % - let's see what else we get.
			switch (c) {
				case '%': { // Replace %% with %
					in_format = false;
					break;
				}
				case 'd': // Integer (signed)
				case 'o': // Octal
				case 'x': // Hexadecimal (lowercase)
				case 'X': { // Hexadecimal (uppercase)
					int64_t index = (selected_index >= 0 ? selected_index : value_index);
					if (index >= value_types.size()) {
						for (int i = value_types.size(); i < index + 1; i++) {
							value_types.push_back(-1);
						}
					}
					value_types.write[index] = Variant::Type::INT;
					if (selected_index == -1) {
						++value_index;
					}
					in_format = false;
					break;
				}
				case 'f': { // Float
					int64_t index = (selected_index >= 0 ? selected_index : value_index);
					if (index >= value_types.size()) {
						for (int i = value_types.size(); i < index + 1; i++) {
							value_types.push_back(-1);
						}
					}
					value_types.write[index] = Variant::Type::FLOAT;
					if (selected_index == -1) {
						++value_index;
					}
					in_format = false;
					break;
				}
				case 'v': { // Vector2/3/4/2i/3i/4i
					int64_t index = (selected_index >= 0 ? selected_index : value_index);
					if (index >= value_types.size()) {
						for (int i = value_types.size(); i < index + 1; i++) {
							value_types.push_back(-1);
						}
					}
					value_types.write[index] = Variant::Type::VECTOR2;
					if (selected_index == -1) {
						++value_index;
					}
					in_format = false;
					break;
				}
				case 'c':
				case 's': { // String
					int64_t index = (selected_index >= 0 ? selected_index : value_index);
					if (index >= value_types.size()) {
						for (int i = value_types.size(); i < index + 1; i++) {
							value_types.push_back(-1);
						}
					}
					value_types.write[index] = Variant::Type::STRING;
					if (selected_index == -1) {
						++value_index;
					}
					in_format = false;
					break;
				}
				case '-': // Left justify
				case '+': // Show + if positive.
				case 'u': { // Treat as unsigned (for int/hex).
					break;
				}
				case '0':
				case '1':
				case '2':
				case '3':
				case '4':
				case '5':
				case '6':
				case '7':
				case '8':
				case '9': {
					int n = c - '0';
					if (!in_decimals) {
						if (c != '0' || pending_index != 0) {
							pending_index *= 10;
							pending_index += n;
						}
					}
					break;
				}
				case '$': {
					if (pending_index > 0) {
						selected_index = pending_index - 1;
					}
					pending_index = 0;
					break;
				}
				case '.': { // Float/Vector separator.
					in_decimals = true;
					break;
				}
				case '*': { // Dynamic width, based on value.
					int64_t index = (selected_index >= 0 ? selected_index : value_index);
					if (index >= value_types.size()) {
						for (int i = value_types.size(); i < index + 1; i++) {
							value_types.push_back(-1);
						}
					}
					value_types.write[index] = Variant::Type::NIL;
					if (selected_index == -1) {
						++value_index;
					}
					break;
				}

				default: {
					in_format = false;
					break;
				}
			}
		} else { // Not in format string.
			if (c == '%') {
				in_format = true;
				selected_index = -1;
				pending_index = 0;
				in_decimals = false;
			}
		}
	}

	return value_types;
}

bool _check_string_formats(const Ref<Translation> &p_translation, String &r_errors) {
	HashMap<StringName, Vector<StringName>> map = p_translation->get_translated_message_map();
	bool ok = true;
	for (const KeyValue<StringName, Vector<StringName>> &E : map) {
		const Vector<int> fmt = _util_sprintf_extract_arg_types(E.key);
		for (const StringName &s : E.value) {
			const Vector<int> sfmt = _util_sprintf_extract_arg_types(s);
			if (fmt != sfmt) {
				r_errors += vformat("[%s] Translated string format mismatch:\n.     %s\n.     %s\n", p_translation->get_locale(), String(E.key).replace("\n", "\\n"), String(s).replace("\n", "\\n"));
				ok = false;
			}
		}
	}
	return ok;
}

Ref<Translation> _load_ed_tr(const EditorTranslationList *p_etl) {
	LocalVector<uint8_t> data;
	data.resize_uninitialized(p_etl->uncomp_size);
	const int64_t ret = Compression::decompress(data.ptr(), p_etl->uncomp_size, p_etl->data, p_etl->comp_size, Compression::MODE_DEFLATE);
	if (ret == -1) {
		return Ref<Translation>();
	}

	Ref<FileAccessMemory> fa;
	fa.instantiate();
	fa->open_custom(data.ptr(), data.size());

	Ref<Translation> tr = TranslationLoaderPO::load_translation(fa);
	if (tr.is_valid()) {
		tr->set_locale(p_etl->lang);
	}
	return tr;
}

TEST_CASE("[EditorTranslation] Editor translations string format") {
	for (const EditorTranslationList *etl = _editor_translations; etl->data; etl++) {
		if (String(etl->lang) == "source") {
			continue;
		}
		Ref<Translation> tr = _load_ed_tr(etl);
		if (tr.is_valid()) {
			String errors;
			CHECK(_check_string_formats(tr, errors));
			if (!errors.is_empty()) {
				print_line(errors);
			}
		}
	}
}

TEST_CASE("[EditorTranslation] Editor extractable translations string format") {
	for (const EditorTranslationList *etl = _extractable_translations; etl->data; etl++) {
		if (String(etl->lang) == "source") {
			continue;
		}
		Ref<Translation> tr = _load_ed_tr(etl);
		if (tr.is_valid()) {
			String errors;
			CHECK(_check_string_formats(tr, errors));
			if (!errors.is_empty()) {
				print_line(errors);
			}
		}
	}
}

TEST_CASE("[EditorTranslation] Editor property translations string format") {
	for (const EditorTranslationList *etl = _property_translations; etl->data; etl++) {
		if (String(etl->lang) == "source") {
			continue;
		}
		Ref<Translation> tr = _load_ed_tr(etl);
		if (tr.is_valid()) {
			String errors;
			CHECK(_check_string_formats(tr, errors));
			if (!errors.is_empty()) {
				print_line(errors);
			}
		}
	}
}

} // namespace TestEditorTranslation
