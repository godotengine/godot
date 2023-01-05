/**************************************************************************/
/*  project_converter_3_to_4.h                                            */
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

#ifndef PROJECT_CONVERTER_3_TO_4_H
#define PROJECT_CONVERTER_3_TO_4_H

#include "core/io/file_access.h"
#include "core/object/ref_counted.h"
#include "core/string/ustring.h"
#include "core/templates/local_vector.h"

class RegEx;

class ProjectConverter3To4 {
public:
	class RegExContainer;

private:
	uint64_t maximum_file_size;
	uint64_t maximum_line_length;

	void rename_colors(Vector<String> &lines, const RegExContainer &reg_container);
	Vector<String> check_for_rename_colors(Vector<String> &lines, const RegExContainer &reg_container);

	void rename_classes(Vector<String> &lines, const RegExContainer &reg_container);
	Vector<String> check_for_rename_classes(Vector<String> &lines, const RegExContainer &reg_container);

	void rename_gdscript_functions(Vector<String> &lines, const RegExContainer &reg_container, bool builtin);
	Vector<String> check_for_rename_gdscript_functions(Vector<String> &lines, const RegExContainer &reg_container, bool builtin);
	void process_gdscript_line(String &line, const RegExContainer &reg_container, bool builtin);

	void rename_csharp_functions(Vector<String> &lines, const RegExContainer &reg_container);
	Vector<String> check_for_rename_csharp_functions(Vector<String> &lines, const RegExContainer &reg_container);
	void process_csharp_line(String &line, const RegExContainer &reg_container);

	void rename_csharp_attributes(Vector<String> &lines, const RegExContainer &reg_container);
	Vector<String> check_for_rename_csharp_attributes(Vector<String> &lines, const RegExContainer &reg_container);

	void rename_gdscript_keywords(Vector<String> &lines, const RegExContainer &reg_container);
	Vector<String> check_for_rename_gdscript_keywords(Vector<String> &lines, const RegExContainer &reg_container);

	void custom_rename(Vector<String> &lines, String from, String to);
	Vector<String> check_for_custom_rename(Vector<String> &lines, String from, String to);

	void rename_common(const char *array[][2], LocalVector<RegEx *> &cached_regexes, Vector<String> &lines);
	Vector<String> check_for_rename_common(const char *array[][2], LocalVector<RegEx *> &cached_regexes, Vector<String> &lines);

	Vector<String> check_for_files();

	Vector<String> parse_arguments(const String &line);
	int get_end_parenthesis(const String &line) const;
	String connect_arguments(const Vector<String> &line, int from, int to = -1) const;
	String get_starting_space(const String &line) const;
	String get_object_of_execution(const String &line) const;

	String line_formatter(int current_line, String from, String to, String line);
	String simple_line_formatter(int current_line, String old_line, String line);
	String collect_string_from_vector(Vector<String> &vector);

	bool test_single_array(const char *array[][2], bool ignore_second_check = false);
	bool test_conversion_gdscript_builtin(String name, String expected, void (ProjectConverter3To4::*func)(Vector<String> &, const RegExContainer &, bool), String what, const RegExContainer &reg_container, bool builtin);
	bool test_conversion_with_regex(String name, String expected, void (ProjectConverter3To4::*func)(Vector<String> &, const RegExContainer &), String what, const RegExContainer &reg_container);
	bool test_conversion_basic(String name, String expected, const char *array[][2], LocalVector<RegEx *> &regex_cache, String what);
	bool test_array_names();
	bool test_conversion(RegExContainer &reg_container);

public:
	ProjectConverter3To4(int, int);
	int validate_conversion();
	int convert();
};

#endif // PROJECT_CONVERTER_3_TO_4_H
