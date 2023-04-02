/**************************************************************************/
/*  script_search_replace.cpp                                             */
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

#include "script_search_replace.h"

#include "script_builtin_scanner.h"
#include "script_default_scanner.h"

ScriptSearchReplace *ScriptSearchReplace::singleton = nullptr;

static Ref<ScriptBuiltinScanner> builtin_scanner;
static Ref<ScriptDefaultScanner> default_scanner;

ScriptSearchReplace *ScriptSearchReplace::get_singleton() {
	if (singleton == nullptr) {
		memnew(ScriptSearchReplace);
	}
	return singleton;
}

TypedArray<Dictionary> ScriptSearchReplace::scan(String p_file_path, String p_display_path, String p_pattern, bool p_match_case, bool p_whole_words) const {
	Ref<FileAccess> file = FileAccess::open(p_file_path, FileAccess::READ);
	ERR_FAIL_COND_V_MSG(file.is_null(), {}, "Error opening file: '" + p_file_path + "'.");

	String extension = file->get_path().get_extension().to_lower();
	Ref<ScriptScanner> scanner = get_suitable_scanner(extension);
	if (scanner.is_valid()) {
		return scanner->scan(file, p_display_path, p_pattern, p_match_case, p_whole_words);
	}

	return {};
}

bool ScriptSearchReplace::replace(String p_file_path, String p_display_path, TypedArray<Dictionary> p_locations, bool p_match_case, bool p_whole_words, String p_search_text, String p_new_text) const {
	Ref<FileAccess> file = FileAccess::open(p_file_path, FileAccess::READ);
	ERR_FAIL_COND_V_MSG(file.is_null(), false, "Error opening file: '" + p_file_path + "'.");

	String extension = file->get_path().get_extension().to_lower();
	Ref<ScriptScanner> scanner = get_suitable_scanner(extension);
	if (scanner.is_valid()) {
		return scanner->replace(file, p_file_path, p_display_path, p_locations, p_match_case, p_whole_words, p_search_text, p_new_text);
	}

	return false;
}

void ScriptSearchReplace::add_scanner(Ref<ScriptScanner> p_scanner, bool p_at_front) {
	ERR_FAIL_COND(p_scanner.is_null());
	ERR_FAIL_COND(scanners_count >= MAX_SCANNERS);

	if (p_at_front) {
		for (int i = scanners_count; i > 0; i--) {
			scanners[i] = scanners[i - 1];
		}
		scanners[0] = p_scanner;
		scanners_count++;
	} else {
		scanners[scanners_count++] = p_scanner;
	}
}

void ScriptSearchReplace::remove_scanner(Ref<ScriptScanner> p_scanner) {
	ERR_FAIL_COND(p_scanner.is_null());

	// Find scanner
	int i = 0;
	for (; i < scanners_count; ++i) {
		if (scanners[i] == p_scanner) {
			break;
		}
	}

	ERR_FAIL_COND(i >= scanners_count); // Not found

	// Shift next scanners up
	for (; i < scanners_count - 1; ++i) {
		scanners[i] = scanners[i + 1];
	}
	scanners[scanners_count - 1].unref();
	--scanners_count;
}

Ref<ScriptScanner> ScriptSearchReplace::get_suitable_scanner(String p_extension) const {
	for (int i = 0; i < scanners_count; i++) {
		Ref<ScriptScanner> scanner = scanners[i];
		PackedStringArray supported_extensions = scanner->get_supported_extensions();
		if (supported_extensions.has(p_extension)) {
			return scanner;
			break;
		}
	}

	return default_scanner;
}

Dictionary ScriptSearchReplace::assemble_scan_match(String p_file_path, String p_display_path, int p_line_number, int p_begin, int p_end, String p_line) {
	return ScanMatch{ p_file_path, p_display_path, p_line_number, p_begin, p_end, p_line }.to_dict();
}

Dictionary ScriptSearchReplace::ScanMatch::to_dict() {
	Dictionary ret;
	ret["file_path"] = file_path;
	ret["display_text"] = display_text;
	ret["line_number"] = line_number;
	ret["begin"] = begin;
	ret["end"] = end;
	ret["line"] = line;
	return ret;
}

ScriptSearchReplace::ScanMatch ScriptSearchReplace::ScanMatch::from_dict(Dictionary p_dict) {
	return ScriptSearchReplace::ScanMatch{
		p_dict["file_path"],
		p_dict["display_text"],
		p_dict["line_number"],
		p_dict["begin"],
		p_dict["end"],
		p_dict["line"],
	};
}

Dictionary ScriptSearchReplace::assemble_scan_location(int p_line_number, int p_begin, int p_end) {
	return ScanLocation{ p_line_number, p_begin, p_end }.to_dict();
}

Dictionary ScriptSearchReplace::ScanLocation::to_dict() {
	Dictionary ret;
	ret["line_number"] = line_number;
	ret["begin"] = begin;
	ret["end"] = end;
	return ret;
}

ScriptSearchReplace::ScanLocation ScriptSearchReplace::ScanLocation::from_dict(Dictionary p_dict) {
	return ScriptSearchReplace::ScanLocation{
		p_dict["line_number"],
		p_dict["begin"],
		p_dict["end"],
	};
}

PackedStringArray ScriptScanner::get_supported_extensions() const {
	PackedStringArray ret;
	if (GDVIRTUAL_CALL(_get_supported_extensions, ret)) {
		return ret;
	}
	return {};
}

TypedArray<Dictionary> ScriptScanner::scan(Ref<FileAccess> p_file, String p_display_path, String p_pattern, bool p_match_case, bool p_whole_words, Size2i p_range) {
	TypedArray<Dictionary> ret;
	GDVIRTUAL_CALL(_scan, p_file, p_display_path, p_pattern, p_match_case, p_whole_words, p_range, ret);
	return ret;
}

bool ScriptScanner::replace(Ref<FileAccess> p_file, String p_file_path, String p_display_path, TypedArray<Dictionary> p_locations, bool p_match_case, bool p_whole_words, String p_search_text, String p_new_text, Size2i p_range) {
	bool success = false;
	GDVIRTUAL_CALL(_replace, p_file, p_file_path, p_display_path, p_locations, p_match_case, p_whole_words, p_search_text, p_new_text, p_range, success);
	return success;
}

void ScriptScanner::_bind_methods() {
	GDVIRTUAL_BIND(_get_supported_extensions);
	GDVIRTUAL_BIND(_scan, "file", "display_path", "pattern", "match_case", "whole_words", "range");
	GDVIRTUAL_BIND(_replace, "file", "file_path", "display_path", "locations", "match_case", "whole_words", "search_text", "new_text", "range");
}

void ScriptSearchReplace::_bind_methods() {
	ClassDB::bind_static_method("ScriptSearchReplace", D_METHOD("get_singleton"), &ScriptSearchReplace::get_singleton);

	ClassDB::bind_method(D_METHOD("add_scanner", "scanner", "at_front"), &ScriptSearchReplace::add_scanner, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("remove_scanner", "scanner"), &ScriptSearchReplace::remove_scanner);
	ClassDB::bind_method(D_METHOD("get_suitable_scanner", "extension"), &ScriptSearchReplace::get_suitable_scanner);

	ClassDB::bind_method(D_METHOD("assemble_scan_match", "file_path", "display_text", "line_number", "begin", "end", "line"), &ScriptSearchReplace::assemble_scan_match);
	ClassDB::bind_method(D_METHOD("assemble_scan_location", "line_number", "begin", "end"), &ScriptSearchReplace::assemble_scan_location);

	ClassDB::bind_method(D_METHOD("scan", "file_path", "display_path", "pattern", "match_case", "whole_words"), &ScriptSearchReplace::scan);
	ClassDB::bind_method(D_METHOD("replace", "file_path", "display_path", "locations", "match_case", "whole_words", "search_text", "new_text"), &ScriptSearchReplace::replace);
}

ScriptSearchReplace::ScriptSearchReplace() {
	singleton = this;

	builtin_scanner.instantiate();
	singleton->add_scanner(builtin_scanner);

	::default_scanner.instantiate();
	singleton->set_default_scanner(::default_scanner);
}

ScriptSearchReplace::~ScriptSearchReplace() {
	for (int i = 0; i < singleton->scanners_count; i++) {
		singleton->scanners[i].unref();
	}
	singleton->default_scanner.unref();

	singleton = nullptr;
}
