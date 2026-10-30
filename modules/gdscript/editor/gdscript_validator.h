/**************************************************************************/
/*  gdscript_validator.h                                                  */
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

#ifdef TOOLS_ENABLED

#include "core/object/editor_language.h"
#include "core/object/ref_counted.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"
#include "core/templates/list.h"

class GDScriptValidator : public RefCounted {
	GDCLASS(GDScriptValidator, RefCounted);

	List<EditorLanguage::Warning> warnings;
	List<EditorLanguage::ScriptError> errors;
	HashMap<String, List<EditorLanguage::ScriptError>> depended_errors;
	HashSet<int> safe_lines;

	String code;
	String path;

protected:
	static void _bind_methods();

public:
	bool validate(const String &p_code, const String &p_path = String());
	const String &get_validated_code() { return code; }
	const String &get_validated_path() { return path; }

	bool has_errors() const;

	int get_error_count() const;
	String get_error(int p_idx) const;
	int get_error_start_line(int p_idx) const;
	int get_error_end_line(int p_idx) const;
	int get_error_start_column(int p_idx) const;
	int get_error_end_column(int p_idx) const;

	bool has_dependency_errors() const;
	PackedStringArray get_dependencies_with_errors() const;
	int get_dependency_error_count(const String &p_dependency) const;
	String get_dependency_error(const String &p_dependency, int p_idx) const;
	int get_dependency_error_start_line(const String &p_dependency, int p_idx) const;
	int get_dependency_error_end_line(const String &p_dependency, int p_idx) const;
	int get_dependency_error_start_column(const String &p_dependency, int p_idx) const;
	int get_dependency_error_end_column(const String &p_dependency, int p_idx) const;

	bool has_warnings() const;
	int get_warning_count() const;
	String get_warning(int p_idx) const;
	String get_warning_string_code(int p_idx) const;
	int get_warning_start_line(int p_idx) const;
	int get_warning_end_line(int p_idx) const;
	int get_warning_start_column(int p_idx) const;
	int get_warning_end_column(int p_idx) const;

	PackedInt32Array get_safe_lines() const;

	GDScriptValidator() {}
	~GDScriptValidator() {}
};

#endif // TOOLS_ENABLED
