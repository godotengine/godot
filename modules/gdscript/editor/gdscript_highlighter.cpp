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
#include "../gdscript_tokenizer.h"
#include "editor/editor_settings.h"
#include "scene/gui/text_edit.h"

inline bool _is_symbol(CharType c) {
	return is_symbol(c);
}

static bool _is_text_char(CharType c) {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
}

static bool _is_char(CharType c) {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

static bool _is_number(CharType c) {
	return (c >= '0' && c <= '9');
}

static bool _is_hex_symbol(CharType c) {
	return ((c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'));
}

static bool _is_bin_symbol(CharType c) {
	return (c == '0' || c == '1');
}

Map<int, TextEdit::HighlighterInfo> GDScriptSyntaxHighlighter::_get_line_syntax_highlighting(int p_line) {
	Map<int, TextEdit::HighlighterInfo> color_map;

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
	bool in_variable_declaration = false;
	bool in_function_args = false;
	bool in_member_variable = false;
	bool in_node_path = false;
	bool is_hex_notation = false;
	bool is_bin_notation = false;
	bool expect_type = false;
	Color keyword_color;
	Color color;

	int in_region = text_editor->_is_line_in_region(p_line);
	int deregion = 0;

	const Map<int, TextEdit::Text::ColorRegionInfo> cri_map = text_editor->_get_line_color_region_info(p_line);
	const String &str = text_editor->get_line(p_line);
	Color prev_color;
	for (int j = 0; j < str.length(); j++) {
		TextEdit::HighlighterInfo highlighter_info;

		if (deregion > 0) {
			deregion--;
			if (deregion == 0) {
				in_region = -1;
			}
		}

		if (deregion != 0) {
			if (color != prev_color) {
				prev_color = color;
				highlighter_info.color = color;
				color_map[j] = highlighter_info;
			}
			continue;
		}

		color = font_color;

		bool is_char = _is_text_char(str[j]);
		bool is_symbol = _is_symbol(str[j]);
		bool is_number = _is_number(str[j]);

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
			is_symbol = false;
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

		if (is_symbol && str[j] != '.' && in_word) {
			in_word = false;
		}

		if (is_symbol && cri_map.has(j)) {
			const TextEdit::Text::ColorRegionInfo &cri = cri_map[j];

			if (in_region == -1) {
				if (!cri.end) {
					in_region = cri.region;
				}
			} else {
				TextEdit::ColorRegion cr = text_editor->_get_color_region(cri.region);
				if (in_region == cri.region && !cr.line_only) { //ignore otherwise
					if (cri.end || cr.eq) {
						deregion = cr.eq ? cr.begin_key.length() : cr.end_key.length();
					}
				}
			}
		}

		if (!is_char) {
			in_keyword = false;
		}

		if (in_region == -1 && !in_keyword && is_char && !prev_is_char) {
			int to = j;
			while (to < str.length() && _is_text_char(str[to])) {
				to++;
			}

			String word = str.substr(j, to - j);
			Color col = Color();
			if (text_editor->has_keyword_color(word)) {
				col = text_editor->get_keyword_color(word);
			} else if (text_editor->has_member_color(word)) {
				col = text_editor->get_member_color(word);
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
			int k = j;
			while (k < str.length() && !_is_symbol(str[k]) && str[k] != '\t' && str[k] != ' ') {
				k++;
			}

			// check for space between name and bracket
			while (k < str.length() && (str[k] == '\t' || str[k] == ' ')) {
				k++;
			}

			if (str[k] == '(') {
				in_function_name = true;
			} else if (previous_text == GDScriptTokenizer::get_token_name(GDScriptTokenizer::TK_PR_VAR)) {
				in_variable_declaration = true;
			}
		}

		if (!in_function_name && !in_member_variable && !in_keyword && !is_number && in_word) {
			int k = j;
			while (k > 0 && !_is_symbol(str[k]) && str[k] != '\t' && str[k] != ' ') {
				k--;
			}

			if (str[k] == '.') {
				in_member_variable = true;
			}
		}

		if (is_symbol) {
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
			in_function_name = false;
			in_member_variable = false;
		}

		if (!in_node_path && in_region == -1 && str[j] == '$') {
			in_node_path = true;
		} else if (in_region != -1 || (is_symbol && str[j] != '/')) {
			in_node_path = false;
		}

		if (in_region >= 0) {
			next_type = REGION;
			color = text_editor->_get_color_region(in_region).color;
		} else if (in_node_path) {
			next_type = NODE_PATH;
			color = node_path_color;
		} else if (in_keyword) {
			next_type = KEYWORD;
			color = keyword_color;
		} else if (in_member_variable) {
			next_type = MEMBER;
			color = member_color;
		} else if (in_function_name) {
			next_type = FUNCTION;

			if (previous_text == GDScriptTokenizer::get_token_name(GDScriptTokenizer::TK_PR_FUNCTION)) {
				color = function_definition_color;
			} else {
				color = function_color;
			}
		} else if (is_symbol) {
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
					if (text != "") {
						previous_text = text;
					}
				}
			}
		}

		prev_is_char = is_char;
		prev_is_number = is_number;

		if (color != prev_color) {
			prev_color = color;
			highlighter_info.color = color;
			color_map[j] = highlighter_info;
		}
	}
	return color_map;
}

String GDScriptSyntaxHighlighter::get_name() const {
	return "GDScript";
}

List<String> GDScriptSyntaxHighlighter::get_supported_languages() {
	List<String> languages;
	languages.push_back("GDScript");
	return languages;
}

void GDScriptSyntaxHighlighter::_update_cache() {
	font_color = text_editor->get_color("font_color");
	symbol_color = text_editor->get_color("symbol_color");
	function_color = text_editor->get_color("function_color");
	number_color = text_editor->get_color("number_color");
	member_color = text_editor->get_color("member_variable_color");

	const String text_editor_color_theme = EditorSettings::get_singleton()->get("text_editor/theme/color_theme");
	const bool default_theme = text_editor_color_theme == "Default";

	if (default_theme || EditorSettings::get_singleton()->is_dark_theme()) {
		function_definition_color = Color(0.4, 0.9, 1.0);
		node_path_color = Color(0.39, 0.76, 0.35);
	} else {
		function_definition_color = Color(0.0, 0.65, 0.73);
		node_path_color = Color(0.32, 0.55, 0.29);
	}

	EDITOR_DEF("text_editor/highlighting/gdscript/function_definition_color", function_definition_color);
	EDITOR_DEF("text_editor/highlighting/gdscript/node_path_color", node_path_color);
	if (text_editor_color_theme == "Adaptive" || default_theme) {
		EditorSettings::get_singleton()->set_initial_value(
				"text_editor/highlighting/gdscript/function_definition_color",
				function_definition_color,
				true);
		EditorSettings::get_singleton()->set_initial_value(
				"text_editor/highlighting/gdscript/node_path_color",
				node_path_color,
				true);
	}

	function_definition_color = EDITOR_GET("text_editor/highlighting/gdscript/function_definition_color");
	node_path_color = EDITOR_GET("text_editor/highlighting/gdscript/node_path_color");
	type_color = EDITOR_GET("text_editor/highlighting/base_type_color");
}

SyntaxHighlighter *GDScriptSyntaxHighlighter::create() {
	return memnew(GDScriptSyntaxHighlighter);
}
