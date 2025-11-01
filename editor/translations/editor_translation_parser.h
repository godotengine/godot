/**************************************************************************/
/*  editor_translation_parser.h                                           */
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

#include "core/object/gdvirtual.gen.inc"
#include "core/object/ref_counted.h"
#include "core/variant/typed_array.h"

class EditorTranslationParserPlugin : public RefCounted {
	GDCLASS(EditorTranslationParserPlugin, RefCounted);

	// These static methods were added since this class is exposed, but the singleton `EditorTranslationParser` is not.
	// The `add_parser()` and `remove_parser()` methods are exposed through `EditorPlugin`, but there's no point
	// in cluttering it with more unrelated methods.
	static PackedStringArray get_all_recognized_extensions_bind();
	static Ref<EditorTranslationParserPlugin> get_parser_bind(const String &p_extension);
	// It's also convenient to expose the global function `get_extractable_message_list()` here.
	static TypedArray<PackedStringArray> get_builtin_strings_bind();

	TypedArray<PackedStringArray> parse_file_bind(const String &p_path);
	PackedStringArray get_recognized_extensions_bind() const;
	Ref<EditorTranslationParserPlugin> get_previous_parser_bind(const String &p_extension) const;

protected:
	static void _bind_methods();

	GDVIRTUAL1R(TypedArray<PackedStringArray>, _parse_file, String)
	GDVIRTUAL0RC(PackedStringArray, _get_recognized_extensions)

#ifndef DISABLE_DEPRECATED
	GDVIRTUAL3_COMPAT(_parse_file_bind_compat_99297, _parse_file, String, TypedArray<String>, TypedArray<Array>)
#endif

public:
	virtual Error parse_file(const String &p_path, Vector<Vector<String>> *r_translations);
	virtual void get_recognized_extensions(List<String> *r_extensions) const;
};

class EditorTranslationParser {
	struct ParserData {
		Ref<EditorTranslationParserPlugin> parser;
		bool is_standard = false;
	};

	static EditorTranslationParser *singleton;

	Vector<ParserData> parsers;

	int _find_parser(const Ref<EditorTranslationParserPlugin> &p_parser) const;

public:
	static EditorTranslationParser *get_singleton();

	void get_recognized_extensions(List<String> *r_extensions) const;
	bool can_parse(const String &p_extension) const;
	Ref<EditorTranslationParserPlugin> get_parser(const String &p_extension) const;
	Ref<EditorTranslationParserPlugin> get_previous_parser(const Ref<EditorTranslationParserPlugin> &p_parser, const String &p_extension) const;
	void add_parser(const Ref<EditorTranslationParserPlugin> &p_parser, bool p_is_standard);
	void remove_parser(const Ref<EditorTranslationParserPlugin> &p_parser, bool p_is_standard);
	void clean_parsers();

	~EditorTranslationParser();
};
