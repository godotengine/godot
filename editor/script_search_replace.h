/**************************************************************************/
/*  script_search_replace.h                                               */
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

#ifndef SCRIPT_SEARCH_REPLACE_H
#define SCRIPT_SEARCH_REPLACE_H

#include "core/io/file_access.h"
#include "core/object/gdvirtual.gen.inc"
#include "core/object/object.h"
#include "core/object/script_language.h"
#include "core/variant/typed_array.h"

class ScriptScanner : public RefCounted {
	GDCLASS(ScriptScanner, RefCounted);

protected:
	static void _bind_methods();

	GDVIRTUAL0RC(String, _get_name);
	GDVIRTUAL0RC(PackedStringArray, _get_supported_extensions);
	GDVIRTUAL6RC(TypedArray<Dictionary>, _scan, Ref<FileAccess>, String, String, bool, bool, Size2i);
	GDVIRTUAL9RC(bool, _replace, Ref<FileAccess>, String, String, TypedArray<Dictionary>, bool, bool, String, String, Size2i);

public:
	virtual PackedStringArray get_supported_extensions() const;
	virtual TypedArray<Dictionary> scan(Ref<FileAccess> p_file, String p_display_path, String p_pattern, bool p_match_case, bool p_whole_words, Size2i p_range = Size2i(1, -1));
	virtual bool replace(Ref<FileAccess> p_file, String p_file_path, String p_display_path, TypedArray<Dictionary> p_locations, bool p_match_case, bool p_whole_words, String p_search_text, String p_new_text, Size2i p_range = Size2i(1, -1));

	virtual ~ScriptScanner() {}
};

class ScriptSearchReplace : public Object {
	GDCLASS(ScriptSearchReplace, Object);

	static ScriptSearchReplace *singleton;

	enum {
		MAX_SCANNERS = 64
	};

private:
	Ref<ScriptScanner> default_scanner = nullptr;
	Ref<ScriptScanner> scanners[MAX_SCANNERS];
	int scanners_count = 0;

	void set_default_scanner(Ref<ScriptScanner> p_scanner) { default_scanner = p_scanner; }

protected:
	static void _bind_methods();

public:
	struct ScanMatch {
		String file_path; // Allows us to specify different paths for builtin scripts
		String display_text;
		int line_number;
		int begin;
		int end;
		String line;

		Dictionary to_dict();
		static ScanMatch from_dict(Dictionary p_dict);
	};
	struct ScanLocation {
		int line_number = 0;
		int begin = 0;
		int end = 0;

		Dictionary to_dict();
		static ScanLocation from_dict(Dictionary p_dict);
	};

	static ScriptSearchReplace *get_singleton();

	void add_scanner(Ref<ScriptScanner> p_scanner, bool p_at_front = false);
	void remove_scanner(Ref<ScriptScanner> p_scanner);
	Ref<ScriptScanner> get_default_scanner() const { return default_scanner; }
	Ref<ScriptScanner> get_suitable_scanner(String p_extension) const;

	// Type Helpers
	Dictionary assemble_scan_match(String p_file_path, String p_display_path, int p_line_number, int p_begin, int p_end, String p_line);
	Dictionary assemble_scan_location(int p_line_number, int p_begin, int p_end);

	TypedArray<Dictionary> scan(String p_file_path, String p_display_path, String p_pattern, bool p_match_case, bool p_whole_words) const;
	bool replace(String p_file_path, String p_display_path, TypedArray<Dictionary> p_locations, bool p_match_case, bool p_whole_words, String p_search_text, String p_new_text) const;

	ScriptSearchReplace();
	~ScriptSearchReplace();
};

#endif // SCRIPT_SEARCH_REPLACE_H
