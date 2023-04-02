/**************************************************************************/
/*  script_default_scanner.h                                              */
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

#ifndef SCRIPT_DEFAULT_SCANNER_H
#define SCRIPT_DEFAULT_SCANNER_H

#include "script_search_replace.h"

#include "core/variant/typed_array.h"

class ScriptDefaultScanner : public ScriptScanner {
protected:
	Vector<char> line_buffer;
	int line_buffer_idx = 0;

	// Same as get_line, but preserves line ending characters.
	String _get_line(String p_source_code);
	bool _find_next(const String &p_line, String p_pattern, int p_from, bool p_match_case, bool p_whole_words, int &out_begin, int &out_end);

	bool _scan_line(String p_line, int p_line_number, String p_path, String p_display_path, String p_pattern, bool p_match_case, bool p_whole_words, Size2i p_range, TypedArray<Dictionary> &p_matches);
	bool _replace(String p_source_code, String p_file_path, String p_display_path, TypedArray<Dictionary> p_locations, bool p_match_case, bool p_whole_words, String p_search_text, String p_new_text, Size2i p_range, String &result);

public:
	virtual TypedArray<Dictionary> scan(Ref<FileAccess> p_file, String p_display_path, String p_pattern, bool p_match_case, bool p_whole_words, Size2i p_range = Size2i(1, -1));
	virtual bool replace(Ref<FileAccess> p_file, String p_file_path, String p_display_path, TypedArray<Dictionary> p_locations, bool p_match_case, bool p_whole_words, String p_search_text, String p_new_text, Size2i p_range = Size2i(1, -1));
};

#endif // SCRIPT_DEFAULT_SCANNER_H
