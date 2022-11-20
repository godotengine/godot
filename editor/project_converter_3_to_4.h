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

#include "core/io/file_access.h"
#include "core/object/ref_counted.h"
#include "core/string/ustring.h"
#include "core/templates/local_vector.h"

class RegEx;

class ProjectConverter3To4 {
public:
	class RegExContainer;
	struct SourceExcluded {
		enum Type {
			ONE_LINE_STRING,
			MULTI_LINE_STRING,
			COMMENT,
			RESOURCE_PATH,
			SIGNAL_NAME,
			PROJECT_SETTING,
			NODE_NAME,
			NODE_PATH,
		};

		Type type;
		int line_number;
		String content;
		String source;
	};
	struct SourceCode {
		int line_number;
		String content;
	};

private:
	uint64_t maximum_file_size;
	uint64_t maximum_line_length;

	SourceExcluded::Type find_excluded_type(const String &literal, const String &line, int start);

	void process_gdscript(Vector<SourceCode> &lines, RegExContainer &reg_container);
	void process_scene(Vector<SourceCode> &lines, RegExContainer &reg_container);
	void process_resource(Vector<SourceCode> &lines, RegExContainer &reg_container);
	void process_excluded(Vector<SourceExcluded> &excluded);

	Vector<SourceCode> parse_scene(const Vector<SourceCode> &lines, Vector<Vector<SourceCode>> &scripts, Vector<SourceExcluded> &excluded);
	Vector<SourceCode> parse_gdscript(const Vector<SourceCode> &lines, Vector<SourceExcluded> &excluded);
	String parse_gdscript_node_name(const String &line, int &column);

	Vector<SourceCode> restore_excluded(const Vector<SourceCode> &lines, const Vector<SourceExcluded> &excluded);
	Vector<SourceCode> restore_embedded_scripts(const Vector<SourceCode> &lines, const Vector<Vector<SourceCode>> &scripts);
	void restore_source_excluded(String &line, int &current, const Vector<SourceExcluded> &excluded);
	String exclude_from_scene(const SourceCode &source, Vector<SourceExcluded> &excluded);
	String exclude_scene_attribute(const int &line_number, const String &line, const String &name, SourceExcluded::Type ex_type, Vector<SourceExcluded> &excluded);

	Vector<String> validate_conversion_gdscript(Vector<SourceCode> &lines, RegExContainer &reg_container);
	Vector<String> validate_conversion_scene(Vector<SourceCode> &lines, RegExContainer &reg_container);
	Vector<String> validate_conversion_resource(Vector<SourceCode> &lines, RegExContainer &reg_container);
	Vector<String> validate_conversion_excluded(const Vector<SourceExcluded> &excluded);

	void rename_colors(Vector<SourceCode> &lines, const RegExContainer &reg_container);
	Vector<String> check_for_rename_colors(Vector<SourceCode> &lines, const RegExContainer &reg_container);

	void rename_classes(Vector<SourceCode> &lines, const RegExContainer &reg_container);
	Vector<String> check_for_rename_classes(Vector<SourceCode> &lines, const RegExContainer &reg_container);

	void rename_gdscript_functions(Vector<SourceCode> &lines, const RegExContainer &reg_container, bool builtin);
	Vector<String> check_for_rename_gdscript_functions(Vector<SourceCode> &lines, const RegExContainer &reg_container, bool builtin);
	void process_gdscript_line(String &line, const RegExContainer &reg_container, bool builtin);

	void rename_csharp_functions(Vector<SourceCode> &lines, const RegExContainer &reg_container);
	Vector<String> check_for_rename_csharp_functions(Vector<SourceCode> &lines, const RegExContainer &reg_container);
	void process_csharp_line(String &line, const RegExContainer &reg_container);

	void rename_csharp_attributes(Vector<SourceCode> &lines, const RegExContainer &reg_container);
	Vector<String> check_for_rename_csharp_attributes(Vector<SourceCode> &lines, const RegExContainer &reg_container);

	void rename_gdscript_keywords(Vector<SourceCode> &lines, const RegExContainer &reg_container);
	Vector<String> check_for_rename_gdscript_keywords(Vector<SourceCode> &lines, const RegExContainer &reg_container);

	void custom_rename(Vector<SourceCode> &lines, String from, String to);
	Vector<String> check_for_custom_rename(Vector<SourceCode> &lines, String from, String to);

	void rename_common(const char *array[][2], LocalVector<RegEx *> &cached_regexes, Vector<SourceCode> &lines);
	Vector<String> check_for_rename_common(const char *array[][2], LocalVector<RegEx *> &cached_regexes, Vector<SourceCode> &lines);

	Vector<String> check_for_files();
	Vector<SourceCode> split_name(const String &name);

	Vector<String> parse_arguments(const String &line);
	int get_end_parenthesis(const String &line) const;
	String connect_arguments(const Vector<String> &line, int from, int to = -1) const;
	String get_starting_space(const String &line) const;
	String get_object_of_execution(const String &line) const;

	String line_formatter(int current_line, String from, String to, String line);
	String simple_line_formatter(int current_line, String old_line, String line);
	String collect_string_from_vector(Vector<SourceCode> &vector);

	bool test_single_array(const char *array[][2], bool ignore_second_check = false);
	bool test_conversion_gdscript_builtin(String name, String expected, String what, const RegExContainer &reg_container, bool builtin);
	bool test_conversion_csharp_functions(String name, String expected, String what, const RegExContainer &reg_container);
	bool test_conversion_csharp_attributes(String name, String expected, String what, const RegExContainer &reg_container);
	bool test_conversion_classes(String name, String expected, String what, const RegExContainer &reg_container);
	bool test_conversion_gdscript_keywords(String name, String expected, String what, const RegExContainer &reg_container);
	bool test_conversion_colors(String name, String expected, String what, const RegExContainer &reg_container);
	bool test_conversion_basic(String name, String expected, const char *array[][2], LocalVector<RegEx *> &regex_cache, String what);
	bool test_array_names();
	bool test_conversion(RegExContainer &reg_container);

public:
	ProjectConverter3To4(int, int);
	int validate_conversion();
	int convert();
};

#endif // PROJECT_CONVERTER_3_TO_4_H
