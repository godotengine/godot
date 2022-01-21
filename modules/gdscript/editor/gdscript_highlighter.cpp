/*************************************************************************/
/*  gdscript_highlighter.cpp                                             */
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

#include "gdscript_highlighter.h"
#include "../gdscript.h"
#include "../gdscript_tokenizer.h"
#include "editor/editor_settings.h"

static bool _is_char(char32_t c) {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

static bool _is_hex_symbol(char32_t c) {
	return ((c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'));
}

static bool _is_bin_symbol(char32_t c) {
	return (c == '0' || c == '1');
}

Dictionary GDScriptSyntaxHighlighter::_get_line_syntax_highlighting_impl(int p_line) {
	Dictionary color_map;

	Type next_type = NONE;
	Type current_type = NONE;
	Type previous_type = NONE;

	String previous_text = "";
	int previous_column = 0;

	bool prev_is_char = false;
	bool prev_is_number = false;
	bool in_keyword = false;
	bool in_word = false;
	bool in_function_name = false;
	bool in_lambda = false;
	bool in_variable_declaration = false;
	bool in_signal_declaration = false;
	bool in_function_args = false;
	bool in_member_variable = false;
	bool in_node_path = false;
	bool in_annotation = false;
	bool is_hex_notation = false;
	bool is_bin_notation = false;
	bool expect_type = false;
	Color keyword_color;
	Color color;

	color_region_cache[p_line] = -1;
	int in_region = -1;
	if (p_line != 0) {
		int prev_region_line = p_line - 1;
		while (prev_region_line > 0 && !color_region_cache.has(prev_region_line)) {
			prev_region_line--;
		}
		for (int i = prev_region_line; i < p_line - 1; i++) {
			get_line_syntax_highlighting(i);
		}
		if (!color_region_cache.has(p_line - 1)) {
			get_line_syntax_highlighting(p_line - 1);
		}
		in_region = color_region_cache[p_line - 1];
	}

	const String &str = text_edit->get_line(p_line);
	const int line_length = str.length();
	Color prev_color;

	if (in_region != -1 && str.length() == 0) {
		color_region_cache[p_line] = in_region;
	}
	for (int j = 0; j < str.length(); j++) {
		Dictionary highlighter_info;

		color = font_color;
		bool is_char = !is_symbol(str[j]);
		bool is_a_symbol = is_symbol(str[j]);
		bool is_number = (str[j] >= '0' && str[j] <= '9');

		/* color regions */
		if (is_a_symbol || in_region != -1) {
			int from = j;

			if (in_region == -1) {
				for (; from < line_length; from++) {
					if (str[from] == '\\') {
						from++;
						continue;
					}
					break;
				}
			}

			if (from != line_length) {
				/* check if we are in entering a region */
				if (in_region == -1) {
					for (int c = 0; c < color_regions.size(); c++) {
						/* check there is enough room */
						int chars_left = line_length - from;
						int start_key_length = color_regions[c].start_key.length();
						int end_key_length = color_regions[c].end_key.length();
						if (chars_left < start_key_length) {
							continue;
						}

						/* search the line */
						bool match = true;
						const char32_t *start_key = color_regions[c].start_key.get_data();
						for (int k = 0; k < start_key_length; k++) {
							if (start_key[k] != str[from + k]) {
								match = false;
								break;
							}
						}
						if (!match) {
							continue;
						}
						in_region = c;
						from += start_key_length;

						/* check if it's the whole line */
						if (end_key_length == 0 || color_regions[c].line_only || from + end_key_length > line_length) {
							if (from + end_key_length > line_length) {
								// If it's key length and there is a '\', dont skip to highlight esc chars.
								if (str.find("\\", from) >= 0) {
									break;
								}
							}
							prev_color = color_regions[in_region].color;
							highlighter_info["color"] = color_regions[c].color;
							color_map[j] = highlighter_info;

							j = line_length;
							if (!color_regions[c].line_only) {
								color_region_cache[p_line] = c;
							}
						}
						break;
					}

					if (j == line_length) {
						continue;
					}
				}

				/* if we are in one find the end key */
				if (in_region != -1) {
					Color region_color = color_regions[in_region].color;
					if (in_node_path && (color_regions[in_region].start_key == "\"" || color_regions[in_region].start_key == "\'")) {
						region_color = node_path_color;
					}

					prev_color = region_color;
					highlighter_info["color"] = region_color;
					color_map[j] = highlighter_info;

					/* search the line */
					int region_end_index = -1;
					int end_key_length = color_regions[in_region].end_key.length();
					const char32_t *end_key = color_regions[in_region].end_key.get_data();
					for (; from < line_length; from++) {
						if (line_length - from < end_key_length) {
							// Don't break if '\' to highlight esc chars.
							if (str.find("\\", from) < 0) {
								break;
							}
						}

						if (!is_symbol(str[from])) {
							continue;
						}

						if (str[from] == '\\') {
							Dictionary escape_char_highlighter_info;
							escape_char_highlighter_info["color"] = symbol_color;
							color_map[from] = escape_char_highlighter_info;

							from++;

							Dictionary region_continue_highlighter_info;
							prev_color = region_color;
							region_continue_highlighter_info["color"] = region_color;
							color_map[from + 1] = region_continue_highlighter_info;
							continue;
						}

						region_end_index = from;
						for (int k = 0; k < end_key_length; k++) {
							if (end_key[k] != str[from + k]) {
								region_end_index = -1;
								break;
							}
						}

						if (region_end_index != -1) {
							break;
						}
					}

					previous_type = REGION;
					previous_text = "";
					previous_column = j;
					j = from + (end_key_length - 1);
					if (region_end_index == -1) {
						color_region_cache[p_line] = in_region;
					}

					in_region = -1;
					prev_is_char = false;
					prev_is_number = false;
					continue;
				}
			}
		}

		// allow ABCDEF in hex notation
		if (is_hex_notation && (_is_hex_symbol(str[j]) || is_number)) {
			is_number = true;
		} else {
			is_hex_notation = false;
		}

		// disallow anything not a 0 or 1
		if (is_bin_notation && (_is_bin_symbol(str[j]))) {
			is_number = true;
		} else if (is_bin_notation) {
			is_bin_notation = false;
			is_number = false;
		} else {
			is_bin_notation = false;
		}

		// check for dot or underscore or 'x' for hex notation in floating point number or 'e' for scientific notation
		if ((str[j] == '.' || str[j] == 'x' || str[j] == 'b' || str[j] == '_' || str[j] == 'e') && !in_word && prev_is_number && !is_number) {
			is_number = true;
			is_a_symbol = false;
			is_char = false;

			if (str[j] == 'x' && str[j - 1] == '0') {
				is_hex_notation = true;
			} else if (str[j] == 'b' && str[j - 1] == '0') {
				is_bin_notation = true;
			}
		}

		if (!in_word && _is_char(str[j]) && !is_number) {
			in_word = true;
		}

		if ((in_keyword || in_word) && !is_hex_notation) {
			is_number = false;
		}

		if (is_a_symbol && str[j] != '.' && in_word) {
			in_word = false;
		}

		if (!is_char) {
			in_keyword = false;
		}

		if (!in_keyword && is_char && !prev_is_char) {
			int to = j;
			while (to < str.length() && !is_symbol(str[to])) {
				to++;
			}

			String word = str.substr(j, to - j);
			Color col = Color();
			if (keywords.has(word)) {
				col = keywords[word];
			} else if (member_keywords.has(word)) {
				col = member_keywords[word];
			}

			if (col != Color()) {
				for (int k = j - 1; k >= 0; k--) {
					if (str[k] == '.') {
						col = Color(); // keyword & member indexing not allowed
						break;
					} else if (str[k] > 32) {
						break;
					}
				}
				if (col != Color()) {
					in_keyword = true;
					keyword_color = col;
				}
			}
		}

		if (!in_function_name && in_word && !in_keyword) {
			if (previous_text == GDScriptTokenizer::get_token_name(GDScriptTokenizer::Token::SIGNAL)) {
				in_signal_declaration = true;
			} else {
				int k = j;
				while (k < str.length() && !is_symbol(str[k]) && str[k] != '\t' && str[k] != ' ') {
					k++;
				}

				// check for space between name and bracket
				while (k < str.length() && (str[k] == '\t' || str[k] == ' ')) {
					k++;
				}

				if (str[k] == '(') {
					in_function_name = true;
				} else if (previous_text == GDScriptTokenizer::get_token_name(GDScriptTokenizer::Token::VAR)) {
					in_variable_declaration = true;
				}

				// Check for lambda.
				if (in_function_name && previous_text == GDScriptTokenizer::get_token_name(GDScriptTokenizer::Token::FUNC)) {
					k = j - 1;
					while (k > 0 && (str[k] == '\t' || str[k] == ' ')) {
						k--;
					}

					if (str[k] == ':') {
						in_lambda = true;
					}
				}
			}
		}

		if (!in_function_name && !in_member_variable && !in_keyword && !is_number && in_word) {
			int k = j;
			while (k > 0 && !is_symbol(str[k]) && str[k] != '\t' && str[k] != ' ') {
				k--;
			}

			if (str[k] == '.') {
				in_member_variable = true;
			}
		}

		if (is_a_symbol) {
			if (in_function_name) {
				in_function_args = true;
			}

			if (in_function_args && str[j] == ')') {
				in_function_args = false;
			}

			if (expect_type && (prev_is_char || str[j] == '=')) {
				expect_type = false;
			}

			if (j > 0 && str[j] == '>' && str[j - 1] == '-') {
				expect_type = true;
			}

			if (in_variable_declaration || in_function_args) {
				int k = j;
				// Skip space
				while (k < str.length() && (str[k] == '\t' || str[k] == ' ')) {
					k++;
				}

				if (str[k] == ':') {
					// has type hint
					expect_type = true;
				}
			}

			in_variable_declaration = false;
			in_signal_declaration = false;
			in_function_name = false;
			in_lambda = false;
			in_member_variable = false;
		}

		if (!in_node_path && in_region == -1 && str[j] == '$') {
			in_node_path = true;
		} else if (in_region != -1 || (is_a_symbol && str[j] != '/')) {
			in_node_path = false;
		}

		if (!in_annotation && in_region == -1 && str[j] == '@') {
			in_annotation = true;
		} else if (in_region != -1 || is_a_symbol) {
			in_annotation = false;
		}

		if (in_node_path) {
			next_type = NODE_PATH;
			color = node_path_color;
		} else if (in_annotation) {
			next_type = ANNOTATION;
			color = annotation_color;
		} else if (in_keyword) {
			next_type = KEYWORD;
			color = keyword_color;
		} else if (in_member_variable) {
			next_type = MEMBER;
			color = member_color;
		} else if (in_signal_declaration) {
			next_type = SIGNAL;

			color = member_color;
		} else if (in_function_name) {
			next_type = FUNCTION;

			if (!in_lambda && previous_text == GDScriptTokenizer::get_token_name(GDScriptTokenizer::Token::FUNC)) {
				color = function_definition_color;
			} else {
				color = function_color;
			}
		} else if (is_a_symbol) {
			next_type = SYMBOL;
			color = symbol_color;
		} else if (is_number) {
			next_type = NUMBER;
			color = number_color;
		} else if (expect_type) {
			next_type = TYPE;
			color = type_color;
		} else {
			next_type = IDENTIFIER;
		}

		if (next_type != current_type) {
			if (current_type == NONE) {
				current_type = next_type;
			} else {
				previous_type = current_type;
				current_type = next_type;

				// no need to store regions...
				if (previous_type == REGION) {
					previous_text = "";
					previous_column = j;
				} else {
					String text = str.substr(previous_column, j - previous_column).strip_edges();
					previous_column = j;

					// ignore if just whitespace
					if (!text.is_empty()) {
						previous_text = text;
					}
				}
			}
		}

		prev_is_char = is_char;
		prev_is_number = is_number;

		if (color != prev_color) {
			prev_color = color;
			highlighter_info["color"] = color;
			color_map[j] = highlighter_info;
		}
	}
	return color_map;
}

String GDScriptSyntaxHighlighter::_get_name() const {
	return "GDScript";
}

Array GDScriptSyntaxHighlighter::_get_supported_languages() const {
	Array languages;
	languages.push_back("GDScript");
	return languages;
}

void GDScriptSyntaxHighlighter::_update_cache() {
	keywords.clear();
	member_keywords.clear();
	color_regions.clear();
	color_region_cache.clear();

	font_color = text_edit->get_theme_color(SNAME("font_color"));
	symbol_color = EDITOR_GET("text_editor/theme/highlighting/symbol_color");
	function_color = EDITOR_GET("text_editor/theme/highlighting/function_color");
	number_color = EDITOR_GET("text_editor/theme/highlighting/number_color");
	member_color = EDITOR_GET("text_editor/theme/highlighting/member_variable_color");

	/* Engine types. */
	const Color types_color = EDITOR_GET("text_editor/theme/highlighting/engine_type_color");
	List<StringName> types;
	ClassDB::get_class_list(&types);
	for (const StringName &E : types) {
		keywords[E] = types_color;
	}

	/* User types. */
	const Color usertype_color = EDITOR_GET("text_editor/theme/highlighting/user_type_color");
	List<StringName> global_classes;
	ScriptServer::get_global_class_list(&global_classes);
	for (const StringName &E : global_classes) {
		keywords[E] = usertype_color;
	}

	/* Autoloads. */
	OrderedHashMap<StringName, ProjectSettings::AutoloadInfo> autoloads = ProjectSettings::get_singleton()->get_autoload_list();
	for (OrderedHashMap<StringName, ProjectSettings::AutoloadInfo>::Element E = autoloads.front(); E; E = E.next()) {
		const ProjectSettings::AutoloadInfo &info = E.value();
		if (info.is_singleton) {
			keywords[info.name] = usertype_color;
		}
	}

	const GDScriptLanguage *gdscript = GDScriptLanguage::get_singleton();

	/* Core types. */
	const Color basetype_color = EDITOR_GET("text_editor/theme/highlighting/base_type_color");
	List<String> core_types;
	gdscript->get_core_type_words(&core_types);
	for (const String &E : core_types) {
		keywords[StringName(E)] = basetype_color;
	}

	/* Reserved words. */
	const Color keyword_color = EDITOR_GET("text_editor/theme/highlighting/keyword_color");
	const Color control_flow_keyword_color = EDITOR_GET("text_editor/theme/highlighting/control_flow_keyword_color");
	List<String> keyword_list;
	gdscript->get_reserved_words(&keyword_list);
	for (const String &E : keyword_list) {
		if (gdscript->is_control_flow_keyword(E)) {
			keywords[StringName(E)] = control_flow_keyword_color;
		} else {
			keywords[StringName(E)] = keyword_color;
		}
	}

	/* Comments */
	const Color comment_color = EDITOR_GET("text_editor/theme/highlighting/comment_color");
	List<String> comments;
	gdscript->get_comment_delimiters(&comments);
	for (const String &comment : comments) {
		String beg = comment.get_slice(" ", 0);
		String end = comment.get_slice_count(" ") > 1 ? comment.get_slice(" ", 1) : String();
		add_color_region(beg, end, comment_color, end.is_empty());
	}

	/* Strings */
	const Color string_color = EDITOR_GET("text_editor/theme/highlighting/string_color");
	List<String> strings;
	gdscript->get_string_delimiters(&strings);
	for (const String &string : strings) {
		String beg = string.get_slice(" ", 0);
		String end = string.get_slice_count(" ") > 1 ? string.get_slice(" ", 1) : String();
		add_color_region(beg, end, string_color, end.is_empty());
	}

	const Ref<Script> script = _get_edited_resource();
	if (script.is_valid()) {
		/* Member types. */
		const Color member_variable_color = EDITOR_GET("text_editor/theme/highlighting/member_variable_color");
		StringName instance_base = script->get_instance_base_type();
		if (instance_base != StringName()) {
			List<PropertyInfo> plist;
			ClassDB::get_property_list(instance_base, &plist);
			for (const PropertyInfo &E : plist) {
				String name = E.name;
				if (E.usage & PROPERTY_USAGE_CATEGORY || E.usage & PROPERTY_USAGE_GROUP || E.usage & PROPERTY_USAGE_SUBGROUP) {
					continue;
				}
				if (name.find("/") != -1) {
					continue;
				}
				member_keywords[name] = member_variable_color;
			}

			List<String> clist;
			ClassDB::get_integer_constant_list(instance_base, &clist);
			for (const String &E : clist) {
				member_keywords[E] = member_variable_color;
			}
		}
	}

	const String text_edit_color_theme = EditorSettings::get_singleton()->get("text_editor/theme/color_theme");
	const bool godot_2_theme = text_edit_color_theme == "Godot 2";

	if (godot_2_theme || EditorSettings::get_singleton()->is_dark_theme()) {
		function_definition_color = Color(0.4, 0.9, 1.0);
		node_path_color = Color(0.39, 0.76, 0.35);
		annotation_color = Color(1.0, 0.7, 0.45);
	} else {
		function_definition_color = Color(0.0, 0.65, 0.73);
		node_path_color = Color(0.32, 0.55, 0.29);
		annotation_color = Color(0.8, 0.5, 0.25);
	}

	EDITOR_DEF("text_editor/theme/highlighting/gdscript/function_definition_color", function_definition_color);
	EDITOR_DEF("text_editor/theme/highlighting/gdscript/node_path_color", node_path_color);
	EDITOR_DEF("text_editor/theme/highlighting/gdscript/annotation_color", annotation_color);
	if (text_edit_color_theme == "Default" || godot_2_theme) {
		EditorSettings::get_singleton()->set_initial_value(
				"text_editor/theme/highlighting/gdscript/function_definition_color",
				function_definition_color,
				true);
		EditorSettings::get_singleton()->set_initial_value(
				"text_editor/theme/highlighting/gdscript/node_path_color",
				node_path_color,
				true);
		EditorSettings::get_singleton()->set_initial_value(
				"text_editor/theme/highlighting/gdscript/annotation_color",
				annotation_color,
				true);
	}

	function_definition_color = EDITOR_GET("text_editor/theme/highlighting/gdscript/function_definition_color");
	node_path_color = EDITOR_GET("text_editor/theme/highlighting/gdscript/node_path_color");
	annotation_color = EDITOR_GET("text_editor/theme/highlighting/gdscript/annotation_color");
	type_color = EDITOR_GET("text_editor/theme/highlighting/base_type_color");
}

void GDScriptSyntaxHighlighter::add_color_region(const String &p_start_key, const String &p_end_key, const Color &p_color, bool p_line_only) {
	for (int i = 0; i < p_start_key.length(); i++) {
		ERR_FAIL_COND_MSG(!is_symbol(p_start_key[i]), "color regions must start with a symbol");
	}

	if (p_end_key.length() > 0) {
		for (int i = 0; i < p_end_key.length(); i++) {
			ERR_FAIL_COND_MSG(!is_symbol(p_end_key[i]), "color regions must end with a symbol");
		}
	}

	int at = 0;
	for (int i = 0; i < color_regions.size(); i++) {
		ERR_FAIL_COND_MSG(color_regions[i].start_key == p_start_key, "color region with start key '" + p_start_key + "' already exists.");
		if (p_start_key.length() < color_regions[i].start_key.length()) {
			at++;
		}
	}

	ColorRegion color_region;
	color_region.color = p_color;
	color_region.start_key = p_start_key;
	color_region.end_key = p_end_key;
	color_region.line_only = p_line_only;
	color_regions.insert(at, color_region);
	clear_highlighting_cache();
}

Ref<EditorSyntaxHighlighter> GDScriptSyntaxHighlighter::_create() const {
	Ref<GDScriptSyntaxHighlighter> syntax_highlighter;
	syntax_highlighter.instantiate();
	return syntax_highlighter;
}
