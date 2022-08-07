/**************************************************************************/
/*  gdscript_validator.cpp                                                */
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

#include "gdscript_validator.h"

#include "../gdscript.h"

#include "core/object/class_db.h"

void GDScriptValidator::_bind_methods() {
	ClassDB::bind_method(D_METHOD("validate", "code", "path"), &GDScriptValidator::validate, DEFVAL(String()));
	ClassDB::bind_method("get_validated_code", &GDScriptValidator::get_validated_code);
	ClassDB::bind_method("get_validated_path", &GDScriptValidator::get_validated_path);

	ClassDB::bind_method("has_errors", &GDScriptValidator::has_errors);

	ClassDB::bind_method("get_error_count", &GDScriptValidator::get_error_count);
	ClassDB::bind_method(D_METHOD("get_error", "idx"), &GDScriptValidator::get_error);
	ClassDB::bind_method(D_METHOD("get_error_start_line", "idx"), &GDScriptValidator::get_error_start_line);
	ClassDB::bind_method(D_METHOD("get_error_end_line", "idx"), &GDScriptValidator::get_error_end_line);
	ClassDB::bind_method(D_METHOD("get_error_start_column", "idx"), &GDScriptValidator::get_error_start_column);
	ClassDB::bind_method(D_METHOD("get_error_end_column", "idx"), &GDScriptValidator::get_error_end_column);

	ClassDB::bind_method("has_dependency_errors", &GDScriptValidator::has_dependency_errors);
	ClassDB::bind_method("get_dependencies_with_errors", &GDScriptValidator::get_dependencies_with_errors);
	ClassDB::bind_method(D_METHOD("get_dependency_error_count", "dependency"), &GDScriptValidator::get_dependency_error_count);
	ClassDB::bind_method(D_METHOD("get_dependency_error", "dependency", "idx"), &GDScriptValidator::get_dependency_error);
	ClassDB::bind_method(D_METHOD("get_dependency_error_start_line", "dependency", "idx"), &GDScriptValidator::get_dependency_error_start_line);
	ClassDB::bind_method(D_METHOD("get_dependency_error_end_line", "dependency", "idx"), &GDScriptValidator::get_dependency_error_end_line);
	ClassDB::bind_method(D_METHOD("get_dependency_error_start_column", "dependency", "idx"), &GDScriptValidator::get_dependency_error_start_column);
	ClassDB::bind_method(D_METHOD("get_dependency_error_end_column", "dependency", "idx"), &GDScriptValidator::get_dependency_error_end_column);

	ClassDB::bind_method("has_warnings", &GDScriptValidator::has_warnings);
	ClassDB::bind_method("get_warning_count", &GDScriptValidator::get_warning_count);
	ClassDB::bind_method(D_METHOD("get_warning", "idx"), &GDScriptValidator::get_warning);
	ClassDB::bind_method(D_METHOD("get_warning_string_code", "idx"), &GDScriptValidator::get_warning_string_code);
	ClassDB::bind_method(D_METHOD("get_warning_start_line", "idx"), &GDScriptValidator::get_warning_start_line);
	ClassDB::bind_method(D_METHOD("get_warning_end_line", "idx"), &GDScriptValidator::get_warning_end_line);
	ClassDB::bind_method(D_METHOD("get_warning_start_column", "idx"), &GDScriptValidator::get_warning_start_column);
	ClassDB::bind_method(D_METHOD("get_warning_end_column", "idx"), &GDScriptValidator::get_warning_end_column);

	ClassDB::bind_method("get_safe_lines", &GDScriptValidator::get_safe_lines);
}

bool GDScriptValidator::validate(const String &p_code, const String &p_path) {
	warnings.clear();
	errors.clear();
	depended_errors.clear();
	safe_lines.clear();
	List<String> fnc;

	code = p_code;
	path = p_path;
	return GDScriptLanguage::get_singleton()->get_editor_language()->validate(p_code, p_path, &errors, &warnings, &fnc, &safe_lines);
}

bool GDScriptValidator::has_errors() const {
	ERR_FAIL_COND_V_MSG(code.is_empty(), false, "No source code provided.");
	return !errors.is_empty() || !depended_errors.is_empty();
}

int GDScriptValidator::get_error_count() const {
	ERR_FAIL_COND_V_MSG(code.is_empty(), false, "No source code provided.");
	return errors.size();
}

String GDScriptValidator::get_error(int p_idx) const {
	ERR_FAIL_COND_V_MSG(code.is_empty(), String(), "No source code provided.");
	ERR_FAIL_INDEX_V(p_idx, errors.size(), String());
	return errors.get(p_idx).message;
}

int GDScriptValidator::get_error_start_line(int p_idx) const {
	ERR_FAIL_COND_V_MSG(code.is_empty(), -1, "No source code provided.");
	ERR_FAIL_INDEX_V(p_idx, errors.size(), -1);
	return errors.get(p_idx).start_line;
}

int GDScriptValidator::get_error_end_line(int p_idx) const {
	ERR_FAIL_COND_V_MSG(code.is_empty(), -1, "No source code provided.");
	ERR_FAIL_INDEX_V(p_idx, errors.size(), -1);
	return errors.get(p_idx).end_line;
}

int GDScriptValidator::get_error_start_column(int p_idx) const {
	ERR_FAIL_COND_V_MSG(code.is_empty(), -1, "No source code provided.");
	ERR_FAIL_INDEX_V(p_idx, errors.size(), -1);
	return errors.get(p_idx).start_column;
}

int GDScriptValidator::get_error_end_column(int p_idx) const {
	ERR_FAIL_COND_V_MSG(code.is_empty(), -1, "No source code provided.");
	ERR_FAIL_INDEX_V(p_idx, errors.size(), -1);
	return errors.get(p_idx).end_column;
}

bool GDScriptValidator::has_dependency_errors() const {
	ERR_FAIL_COND_V_MSG(code.is_empty(), false, "No source code provided.");
	return !depended_errors.is_empty();
}

PackedStringArray GDScriptValidator::get_dependencies_with_errors() const {
	ERR_FAIL_COND_V_MSG(code.is_empty(), PackedStringArray(), "No source code provided.");
	PackedStringArray ret;
	for (const KeyValue<String, List<EditorLanguage::ScriptError>> &pair : depended_errors) {
		ret.append(pair.key);
	}
	return ret;
}

int GDScriptValidator::get_dependency_error_count(const String &p_dependency) const {
	ERR_FAIL_COND_V_MSG(code.is_empty(), 0, "No source code provided.");
	ERR_FAIL_COND_V(!depended_errors.has(p_dependency), 0);
	return depended_errors.get(p_dependency).size();
}

String GDScriptValidator::get_dependency_error(const String &p_dependency, int p_idx) const {
	ERR_FAIL_COND_V_MSG(code.is_empty(), String(), "No source code provided.");
	ERR_FAIL_COND_V(!depended_errors.has(p_dependency), String());
	ERR_FAIL_INDEX_V(p_idx, depended_errors.get(p_dependency).size(), String());
	return depended_errors.get(p_dependency).get(p_idx).message;
}

int GDScriptValidator::get_dependency_error_start_line(const String &p_dependency, int p_idx) const {
	ERR_FAIL_COND_V_MSG(code.is_empty(), -1, "No source code provided.");
	ERR_FAIL_COND_V(!depended_errors.has(p_dependency), -1);
	ERR_FAIL_INDEX_V(p_idx, depended_errors.get(p_dependency).size(), -1);
	return depended_errors.get(p_dependency).get(p_idx).start_line;
}

int GDScriptValidator::get_dependency_error_end_line(const String &p_dependency, int p_idx) const {
	ERR_FAIL_COND_V_MSG(code.is_empty(), -1, "No source code provided.");
	ERR_FAIL_COND_V(!depended_errors.has(p_dependency), -1);
	ERR_FAIL_INDEX_V(p_idx, depended_errors.get(p_dependency).size(), -1);
	return depended_errors.get(p_dependency).get(p_idx).end_line;
}

int GDScriptValidator::get_dependency_error_start_column(const String &p_dependency, int p_idx) const {
	ERR_FAIL_COND_V_MSG(code.is_empty(), -1, "No source code provided.");
	ERR_FAIL_COND_V(!depended_errors.has(p_dependency), -1);
	ERR_FAIL_INDEX_V(p_idx, depended_errors.get(p_dependency).size(), -1);
	return depended_errors.get(p_dependency).get(p_idx).start_column;
}

int GDScriptValidator::get_dependency_error_end_column(const String &p_dependency, int p_idx) const {
	ERR_FAIL_COND_V_MSG(code.is_empty(), -1, "No source code provided.");
	ERR_FAIL_COND_V(!depended_errors.has(p_dependency), -1);
	ERR_FAIL_INDEX_V(p_idx, depended_errors.get(p_dependency).size(), -1);
	return depended_errors.get(p_dependency).get(p_idx).end_column;
}

bool GDScriptValidator::has_warnings() const {
	ERR_FAIL_COND_V_MSG(code.is_empty(), false, "No source code provided.");
	return !warnings.is_empty();
}

int GDScriptValidator::get_warning_count() const {
	ERR_FAIL_COND_V_MSG(code.is_empty(), 0, "No source code provided.");
	return warnings.size();
}

String GDScriptValidator::get_warning(int p_idx) const {
	ERR_FAIL_COND_V_MSG(code.is_empty(), String(), "No source code provided.");
	ERR_FAIL_INDEX_V(p_idx, warnings.size(), String());
	return warnings.get(p_idx).message;
}

String GDScriptValidator::get_warning_string_code(int p_idx) const {
	ERR_FAIL_COND_V_MSG(code.is_empty(), String(), "No source code provided.");
	ERR_FAIL_INDEX_V(p_idx, warnings.size(), String());
	return warnings.get(p_idx).string_code;
}

int GDScriptValidator::get_warning_start_line(int p_idx) const {
	ERR_FAIL_COND_V_MSG(code.is_empty(), -1, "No source code provided.");
	ERR_FAIL_INDEX_V(p_idx, warnings.size(), -1);
	return warnings.get(p_idx).start_line;
}

int GDScriptValidator::get_warning_end_line(int p_idx) const {
	ERR_FAIL_COND_V_MSG(code.is_empty(), -1, "No source code provided.");
	ERR_FAIL_INDEX_V(p_idx, warnings.size(), -1);
	return warnings.get(p_idx).end_line;
}

int GDScriptValidator::get_warning_start_column(int p_idx) const {
	ERR_FAIL_COND_V_MSG(code.is_empty(), -1, "No source code provided.");
	ERR_FAIL_INDEX_V(p_idx, warnings.size(), -1);
	return warnings.get(p_idx).start_column;
}

int GDScriptValidator::get_warning_end_column(int p_idx) const {
	ERR_FAIL_COND_V_MSG(code.is_empty(), -1, "No source code provided.");
	ERR_FAIL_INDEX_V(p_idx, warnings.size(), -1);
	return warnings.get(p_idx).end_column;
}

PackedInt32Array GDScriptValidator::get_safe_lines() const {
	PackedInt32Array slines;
	for (int line : safe_lines) {
		slines.append(line);
	}
	slines.sort();
	return slines;
}
