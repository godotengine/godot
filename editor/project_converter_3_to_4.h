/*************************************************************************/
/*  project_converter_3_to_4.h                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef PROJECT_CONVERTER_3_TO_4_H
#define PROJECT_CONVERTER_3_TO_4_H

#include "core/core_bind.h"
#include "core/io/file_access.h"
#include "core/object/ref_counted.h"
#include "core/string/ustring.h"

class ProjectConverter3To4 {
	void rename_enums(String &file_content);
	Vector<String> check_for_rename_enums(Vector<String> &file_content);

	void rename_classes(String &file_content);
	Vector<String> check_for_rename_classes(Vector<String> &file_content);

	void rename_gdscript_functions(String &file_content);
	Vector<String> check_for_rename_gdscript_functions(Vector<String> &file_content);

	void rename_csharp_functions(String &file_content);
	Vector<String> check_for_rename_csharp_functions(Vector<String> &file_content);

	void rename_gdscript_keywords(String &file_content);
	Vector<String> check_for_rename_gdscript_keywords(Vector<String> &file_content);

	void custom_rename(String &file_content, String from, String to);
	Vector<String> check_for_custom_rename(Vector<String> &file_content, String from, String to);

	void rename_common(const char *array[][2], String &file_content);
	Vector<String> check_for_rename_common(const char *array[][2], Vector<String> &file_content);

	Vector<String> check_for_files();

	Vector<String> parse_arguments(const String &line);
	int get_end_parenthess(const String &line) const;
	String connect_arguments(const Vector<String> &line, int from, int to = -1) const;
	String get_starting_space(const String &line) const;
	String get_object_of_execution(const String &line) const;

	String line_formatter(int current_line, String from, String to, String line);
	String simple_line_formatter(int current_line, String old_line, String line);

	bool test_single_array(const char *array[][2], bool ignore_second_check = false);
	bool test_conversion_single_additional(String name, String expected, void (ProjectConverter3To4::*func)(String &), String what);
	bool test_conversion_single_normal(String name, String expected, const char *array[][2], String what);
	bool test_array_names();
	bool test_conversion();

public:
	int validate_conversion();
	int convert();
};

#endif // PROJECT_CONVERTER_3_TO_4_H
