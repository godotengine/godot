/**************************************************************************/
/*  script_builtin_scanner.h                                              */
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

#ifndef SCRIPT_BUILTIN_SCANNER_H
#define SCRIPT_BUILTIN_SCANNER_H

#include "script_search_replace.h"

#include "core/variant/typed_array.h"
#include "script_default_scanner.h"

class ScriptBuiltinScanner : public ScriptDefaultScanner {
private:
	Vector<Ref<Script>> _parse_scene(String p_file_path) const;
	Ref<Script> _parse_scene(String p_file_path, String p_display_path) const;

public:
	virtual PackedStringArray get_supported_extensions() const { return { "tscn" }; }
	virtual TypedArray<Dictionary> scan(Ref<FileAccess> p_file, String p_display_path, String p_pattern, bool p_match_case, bool p_whole_words, Size2i p_range = Size2i(1, -1));
	virtual bool replace(Ref<FileAccess> p_file, String p_file_path, String p_display_path, TypedArray<Dictionary> p_locations, bool p_match_case, bool p_whole_words, String p_search_text, String p_new_text, Size2i p_range = Size2i(1, -1));
};

#endif // SCRIPT_BUILTIN_SCANNER_H
