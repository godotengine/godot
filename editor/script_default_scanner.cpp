/**************************************************************************/
/*  script_default_scanner.cpp                                            */
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

#include "script_default_scanner.h"

TypedArray<Dictionary> ScriptDefaultScanner::scan(Ref<FileAccess> p_file, String p_display_path, String p_pattern, bool p_match_case, bool p_whole_words, Size2i p_range) {
	TypedArray<Dictionary> matches;

	// Line number starts at 1.
	for (int line_number = 1; !p_file->eof_reached(); line_number++) {
		String line = p_file->get_line();
		if (!_scan_line(line, line_number, p_file->get_path(), p_display_path, p_pattern, p_match_case, p_whole_words, p_range, matches)) {
			break;
		}
	}

	return matches;
}

bool ScriptDefaultScanner::replace(Ref<FileAccess> p_file, String p_file_path, String p_display_path, TypedArray<Dictionary> p_locations, bool p_match_case, bool p_whole_words, String p_search_text, String p_new_text, Size2i p_range) {
	String buffer;
	_replace(p_file->get_as_text(), p_file_path, p_display_path, p_locations, p_match_case, p_whole_words, p_search_text, p_new_text, p_range, buffer);

	// Now the modified contents are in the buffer, rewrite the file with our changes.
	Error err = p_file->reopen(p_file->get_path(), FileAccess::WRITE);
	ERR_FAIL_COND_V_MSG(err != OK, false, vformat("Cannot create file in path '%s'.", p_display_path));

	p_file->store_string(buffer);
	return true;
}

bool ScriptDefaultScanner::_scan_line(String p_line, int p_line_number, String p_path, String p_display_path, String p_pattern, bool p_match_case, bool p_whole_words, Size2i p_range, TypedArray<Dictionary> &p_matches) {
	int begin = 0, end = 0;
	if (p_line_number < p_range.x) {
		return true;
	} else if (p_range.y > 0 && p_line_number > p_range.y) {
		return false;
	}

	while (_find_next(p_line, p_pattern, end, p_match_case, p_whole_words, begin, end)) {
		ScriptSearchReplace::ScanMatch match = ScriptSearchReplace::ScanMatch{ p_path, p_display_path, p_line_number - p_range.x + 1, begin, end, p_line };
		p_matches.append(match.to_dict());
	}

	return true;
}

bool ScriptDefaultScanner::_replace(String p_source_code, String p_file_path, String p_display_path, TypedArray<Dictionary> p_locations, bool p_match_case, bool p_whole_words, String p_search_text, String p_new_text, Size2i p_range, String &p_result) {
	int current_line = 1;

	line_buffer_idx = 0;
	String line = _get_line(p_source_code);

	int offset = 0;

	for (int i = 0; i < p_locations.size(); ++i) {
		ScriptSearchReplace::ScanLocation location = ScriptSearchReplace::ScanLocation::from_dict(p_locations[i]);
		int repl_line_number = location.line_number + (p_range.x - 1); // "- 1" because p_range is in terms of "visual lines", which starts at 1, but here we need it to start at 0.

		while (current_line < repl_line_number) {
			p_result += line;
			line = _get_line(p_source_code);
			++current_line;
			offset = 0;
		}

		int repl_begin = location.begin + offset;
		int repl_end = location.end + offset;

		int _;
		if (!_find_next(line, p_search_text, repl_begin, p_match_case, p_whole_words, _, _)) {
			// Make sure the replacement is still valid in case the file was tampered with.
			print_verbose(vformat("Occurrence no longer matches, replace will be ignored in %s: line %d, col %d", p_display_path, repl_line_number, repl_begin));
			continue;
		}

		line = line.left(repl_begin) + p_new_text + line.substr(repl_end);
		// Keep an offset in case there are successive replaces in the same line.
		offset += p_new_text.length() - (repl_end - repl_begin);
	}

	p_result += line;

	while (line_buffer_idx < p_source_code.size() - 1) {
		p_result += _get_line(p_source_code);
	}

	return current_line > 1 || p_result != p_source_code;
}

bool ScriptDefaultScanner::_find_next(const String &p_line, String p_pattern, int from, bool p_match_case, bool p_whole_words, int &out_begin, int &out_end) {
	int end = from;

	while (true) {
		int begin = p_match_case ? p_line.find(p_pattern, end) : p_line.findn(p_pattern, end);

		if (begin == -1) {
			return false;
		}

		end = begin + p_pattern.length();
		out_begin = begin;
		out_end = end;

		if (p_whole_words) {
			if (begin > 0 && (is_ascii_identifier_char(p_line[begin - 1]))) {
				continue;
			}
			if (end < p_line.size() && (is_ascii_identifier_char(p_line[end]))) {
				continue;
			}
		}

		return true;
	}
}

String ScriptDefaultScanner::_get_line(String source_code) {
	line_buffer.clear();
	char32_t c = source_code[line_buffer_idx++];

	while (line_buffer_idx < source_code.size() - 1) {
		if (c == '\n') {
			line_buffer.push_back(c);
			line_buffer.push_back(0);
			return String::utf8(line_buffer.ptr());

		} else if (c == '\0') {
			line_buffer.push_back(c);
			return String::utf8(line_buffer.ptr());

		} else if (c != '\r') {
			line_buffer.push_back(c);
		}

		c = source_code[line_buffer_idx++];
	}

	line_buffer.push_back(0);
	return String::utf8(line_buffer.ptr());
}
