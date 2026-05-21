/**************************************************************************/
/*  test_highlighter.h                                                    */
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

#if defined(TOOLS_ENABLED) && !defined(ADVANCED_GUI_DISABLED)

#include "../editor/gdscript_highlighter.h"

#include "editor/settings/editor_settings.h"
#include "scene/gui/text_edit.h"
#include "tests/test_macros.h"

namespace GDScriptTests {
namespace Highlighter {

static const Color FONT_COLOR = Color(0.91, 0.92, 0.93);
static const Color SYMBOL_COLOR = Color(0.08, 0.70, 0.83);
static const Color FUNCTION_COLOR = Color(0.18, 0.52, 0.96);
static const Color GLOBAL_FUNCTION_COLOR = Color(0.64, 0.49, 0.96);
static const Color FUNCTION_DEFINITION_COLOR = Color(0.94, 0.58, 0.17);
static const Color NUMBER_COLOR = Color(0.14, 0.76, 0.45);
static const Color MEMBER_COLOR = Color(0.96, 0.38, 0.45);
static const Color STRING_COLOR = Color(0.95, 0.77, 0.24);
static const Color PLACEHOLDER_COLOR = Color(0.99, 0.46, 0.19);
static const Color NODE_PATH_COLOR = Color(0.30, 0.73, 0.56);
static const Color NODE_REFERENCE_COLOR = Color(0.31, 0.63, 0.96);
static const Color ANNOTATION_COLOR = Color(0.83, 0.48, 0.92);
static const Color STRING_NAME_COLOR = Color(0.96, 0.61, 0.76);
static const Color KEYWORD_COLOR = Color(0.91, 0.31, 0.56);
static const Color CONTROL_FLOW_KEYWORD_COLOR = Color(0.74, 0.36, 0.89);
static const Color TYPE_COLOR = Color(0.42, 0.70, 0.98);
static const Color ENGINE_TYPE_COLOR = Color(0.63, 0.83, 0.98);
static const Color USER_TYPE_COLOR = Color(0.58, 0.83, 0.66);
static const Color COMMENT_COLOR = Color(0.50, 0.55, 0.61);
static const Color DOC_COMMENT_COLOR = Color(0.56, 0.66, 0.78);
static const Color CODE_REGION_COLOR = Color(0.45, 0.61, 0.73);
static const Color COMMENT_MARKER_CRITICAL_COLOR = Color(0.95, 0.22, 0.22);
static const Color COMMENT_MARKER_WARNING_COLOR = Color(0.95, 0.63, 0.16);
static const Color COMMENT_MARKER_NOTICE_COLOR = Color(0.35, 0.68, 0.96);

static void setup_highlighter_settings() {
	EditorSettings *settings = EditorSettings::get_singleton();
	REQUIRE(settings != nullptr);

	settings->set_setting("text_editor/theme/highlighting/symbol_color", SYMBOL_COLOR);
	settings->set_setting("text_editor/theme/highlighting/function_color", FUNCTION_COLOR);
	settings->set_setting("text_editor/theme/highlighting/number_color", NUMBER_COLOR);
	settings->set_setting("text_editor/theme/highlighting/member_variable_color", MEMBER_COLOR);
	settings->set_setting("text_editor/theme/highlighting/engine_type_color", ENGINE_TYPE_COLOR);
	settings->set_setting("text_editor/theme/highlighting/user_type_color", USER_TYPE_COLOR);
	settings->set_setting("text_editor/theme/highlighting/base_type_color", TYPE_COLOR);
	settings->set_setting("text_editor/theme/highlighting/keyword_color", KEYWORD_COLOR);
	settings->set_setting("text_editor/theme/highlighting/control_flow_keyword_color", CONTROL_FLOW_KEYWORD_COLOR);
	settings->set_setting("text_editor/theme/highlighting/comment_color", COMMENT_COLOR);
	settings->set_setting("text_editor/theme/highlighting/doc_comment_color", DOC_COMMENT_COLOR);
	settings->set_setting("text_editor/theme/highlighting/folded_code_region_color", CODE_REGION_COLOR);
	settings->set_setting("text_editor/theme/highlighting/string_color", STRING_COLOR);
	settings->set_setting("text_editor/theme/highlighting/string_placeholder_color", PLACEHOLDER_COLOR);
	settings->set_setting("text_editor/theme/highlighting/gdscript/function_definition_color", FUNCTION_DEFINITION_COLOR);
	settings->set_setting("text_editor/theme/highlighting/gdscript/global_function_color", GLOBAL_FUNCTION_COLOR);
	settings->set_setting("text_editor/theme/highlighting/gdscript/node_path_color", NODE_PATH_COLOR);
	settings->set_setting("text_editor/theme/highlighting/gdscript/node_reference_color", NODE_REFERENCE_COLOR);
	settings->set_setting("text_editor/theme/highlighting/gdscript/annotation_color", ANNOTATION_COLOR);
	settings->set_setting("text_editor/theme/highlighting/gdscript/string_name_color", STRING_NAME_COLOR);
	settings->set_setting("text_editor/theme/highlighting/comment_markers/critical_color", COMMENT_MARKER_CRITICAL_COLOR);
	settings->set_setting("text_editor/theme/highlighting/comment_markers/warning_color", COMMENT_MARKER_WARNING_COLOR);
	settings->set_setting("text_editor/theme/highlighting/comment_markers/notice_color", COMMENT_MARKER_NOTICE_COLOR);
	settings->set_setting("text_editor/theme/highlighting/comment_markers/critical_list", "ALERT,FIXME");
	settings->set_setting("text_editor/theme/highlighting/comment_markers/warning_list", "TODO,WARN");
	settings->set_setting("text_editor/theme/highlighting/comment_markers/notice_list", "NOTE,INFO");
}

struct HighlighterFixture {
	HighlighterFixture() {
		setup_highlighter_settings();
	}
};

struct HighlightedSource {
	TextEdit *text_edit = nullptr;
	Ref<GDScriptSyntaxHighlighter> highlighter;

	HighlightedSource(const String &p_source) {
		text_edit = memnew(TextEdit);
		text_edit->add_theme_color_override("font_color", FONT_COLOR);
		text_edit->set_text(p_source);

		highlighter.instantiate();
		text_edit->set_syntax_highlighter(highlighter);
	}

	~HighlightedSource() {
		if (highlighter.is_valid()) {
			highlighter->set_text_edit(nullptr);
		}
		memdelete(text_edit);
	}
};

static Color get_color_at(Dictionary p_line_colors, int p_column) {
	Array columns = p_line_colors.keys();
	columns.sort();

	Color color;
	for (int i = 0; i < columns.size(); i++) {
		const int column = columns[i];
		if (column > p_column) {
			break;
		}

		Dictionary entry = p_line_colors[column];
		if (entry.has("color")) {
			color = entry["color"];
		}
	}

	return color;
}

static void check_color_at(HighlightedSource &p_source, int p_line, int p_column, const Color &p_color) {
	CHECK(get_color_at(p_source.highlighter->get_line_syntax_highlighting(p_line), p_column) == p_color);
}

static int find_column(const String &p_line, const String &p_text, int p_from = 0) {
	const int column = p_line.find(p_text, p_from);
	REQUIRE(column >= 0);
	return column;
}

static void check_token_color(HighlightedSource &p_source, int p_line, const String &p_line_text, const String &p_token,
		const Color &p_color, int p_from = 0) {
	const int start_column = find_column(p_line_text, p_token, p_from);
	const int end_column = start_column + p_token.length();

	for (int column = start_column; column < end_column; column++) {
		check_color_at(p_source, p_line, column, p_color);
	}
}

} // namespace Highlighter

TEST_SUITE("[Modules][GDScript][SyntaxHighlighter]") {
TEST_CASE("[Editor] reports GDScript metadata") {
	Ref<GDScriptSyntaxHighlighter> highlighter;
	highlighter.instantiate();

	CHECK(highlighter->_get_name() == "GDScript");

	const PackedStringArray languages = highlighter->_get_supported_languages();
	REQUIRE(languages.size() == 1);
	CHECK(languages[0] == "GDScript");
}

TEST_CASE("[Editor] highlights declarations, types, calls, and literals") {
	using namespace Highlighter;

	HighlighterFixture fixture;
	const String code = "extends Node2D\n"
						"@export var health: int = -42\n"
						"const SPAWN_NAME := &\"Player%d\" % 2\n"
						"signal hit(amount: float)\n"
						"func _ready() -> void:\n"
						"\tvar path: NodePath = ^\"Root/Child\"\n"
						"\t$Sprite2D.visible = true\n"
						"\t%HealthBar.value = clampf(health, 0.0, 100.0)\n"
						"\tprint(\"hp={health}\\n\")\n"
						"\tself.position.x += TAU\n";
	HighlightedSource highlighted(code);

	check_token_color(highlighted, 0, "extends Node2D", "extends", KEYWORD_COLOR);
	check_token_color(highlighted, 0, "extends Node2D", "Node2D", ENGINE_TYPE_COLOR);

	check_token_color(highlighted, 1, "@export var health: int = -42", "@export", ANNOTATION_COLOR);
	check_token_color(highlighted, 1, "@export var health: int = -42", "var", KEYWORD_COLOR);
	check_token_color(highlighted, 1, "@export var health: int = -42", "int", TYPE_COLOR);
	check_token_color(highlighted, 1, "@export var health: int = -42", "-42", NUMBER_COLOR);

	check_token_color(highlighted, 2, "const SPAWN_NAME := &\"Player%d\" % 2", "const", KEYWORD_COLOR);
	check_token_color(highlighted, 2, "const SPAWN_NAME := &\"Player%d\" % 2", "&\"Player", STRING_NAME_COLOR);
	check_token_color(highlighted, 2, "const SPAWN_NAME := &\"Player%d\" % 2", "%d", PLACEHOLDER_COLOR);
	check_token_color(highlighted, 2, "const SPAWN_NAME := &\"Player%d\" % 2", "2", NUMBER_COLOR);

	check_token_color(highlighted, 3, "signal hit(amount: float)", "signal", KEYWORD_COLOR);
	check_token_color(highlighted, 3, "signal hit(amount: float)", "hit", MEMBER_COLOR);
	check_token_color(highlighted, 3, "signal hit(amount: float)", "float", TYPE_COLOR);

	check_token_color(highlighted, 4, "func _ready() -> void:", "func", KEYWORD_COLOR);
	check_token_color(highlighted, 4, "func _ready() -> void:", "_ready", FUNCTION_DEFINITION_COLOR);
	check_token_color(highlighted, 4, "func _ready() -> void:", "void", TYPE_COLOR);

	check_token_color(highlighted, 5, "\tvar path: NodePath = ^\"Root/Child\"", "NodePath", TYPE_COLOR);
	check_token_color(highlighted, 5, "\tvar path: NodePath = ^\"Root/Child\"", "^\"Root/Child\"", NODE_PATH_COLOR);

	check_token_color(highlighted, 6, "\t$Sprite2D.visible = true", "$Sprite2D", NODE_REFERENCE_COLOR);
	check_token_color(highlighted, 6, "\t$Sprite2D.visible = true", "visible", MEMBER_COLOR);
	check_token_color(highlighted, 6, "\t$Sprite2D.visible = true", "true", KEYWORD_COLOR);

	check_token_color(highlighted, 7, "\t%HealthBar.value = clampf(health, 0.0, 100.0)", "%HealthBar",
			NODE_REFERENCE_COLOR);
	check_token_color(highlighted, 7, "\t%HealthBar.value = clampf(health, 0.0, 100.0)", "value", MEMBER_COLOR);
	check_token_color(highlighted, 7, "\t%HealthBar.value = clampf(health, 0.0, 100.0)", "clampf", GLOBAL_FUNCTION_COLOR);
	check_token_color(highlighted, 7, "\t%HealthBar.value = clampf(health, 0.0, 100.0)", "0.0", NUMBER_COLOR);
	check_token_color(highlighted, 7, "\t%HealthBar.value = clampf(health, 0.0, 100.0)", "100.0", NUMBER_COLOR);

	check_token_color(highlighted, 8, "\tprint(\"hp={health}\\n\")", "print", GLOBAL_FUNCTION_COLOR);
	check_token_color(highlighted, 8, "\tprint(\"hp={health}\\n\")", "\"hp=", STRING_COLOR);
	check_token_color(highlighted, 8, "\tprint(\"hp={health}\\n\")", "{health}", PLACEHOLDER_COLOR);
	check_token_color(highlighted, 8, "\tprint(\"hp={health}\\n\")", "\\n", SYMBOL_COLOR);

	check_token_color(highlighted, 9, "\tself.position.x += TAU", "self", KEYWORD_COLOR);
	check_token_color(highlighted, 9, "\tself.position.x += TAU", "position", MEMBER_COLOR);
	check_token_color(highlighted, 9, "\tself.position.x += TAU", "x", MEMBER_COLOR);
	check_token_color(highlighted, 9, "\tself.position.x += TAU", "TAU", KEYWORD_COLOR);
}

TEST_CASE("[Editor] keeps comments, markers, and multiline strings distinct") {
	using namespace Highlighter;

	HighlighterFixture fixture;
	const String code = "#region Gameplay\n"
						"var text = \"\"\"First\n"
						"TODO inside string\n"
						"Third\"\"\"\n"
						"# TODO: schedule follow-up\n"
						"## NOTE: exported docs\n"
						"pass #region not a fold marker\n"
						"#endregion\n"
						"var done = true\n";
	HighlightedSource highlighted(code);

	check_token_color(highlighted, 0, "#region Gameplay", "#region", CODE_REGION_COLOR);
	check_token_color(highlighted, 1, "var text = \"\"\"First", "\"\"\"First", STRING_COLOR);
	check_token_color(highlighted, 2, "TODO inside string", "TODO", STRING_COLOR);
	check_token_color(highlighted, 3, "Third\"\"\"", "Third\"\"\"", STRING_COLOR);

	check_token_color(highlighted, 4, "# TODO: schedule follow-up", "# ", COMMENT_COLOR);
	check_token_color(highlighted, 4, "# TODO: schedule follow-up", "TODO", COMMENT_MARKER_WARNING_COLOR);
	check_token_color(highlighted, 4, "# TODO: schedule follow-up", ": schedule", COMMENT_COLOR);

	check_token_color(highlighted, 5, "## NOTE: exported docs", "## ", DOC_COMMENT_COLOR);
	check_token_color(highlighted, 5, "## NOTE: exported docs", "NOTE", COMMENT_MARKER_NOTICE_COLOR);
	check_token_color(highlighted, 5, "## NOTE: exported docs", ": exported", DOC_COMMENT_COLOR);

	check_token_color(highlighted, 6, "pass #region not a fold marker", "pass", CONTROL_FLOW_KEYWORD_COLOR);
	check_token_color(highlighted, 6, "pass #region not a fold marker", "#region", COMMENT_COLOR);

	check_token_color(highlighted, 7, "#endregion", "#endregion", CODE_REGION_COLOR);
	check_token_color(highlighted, 8, "var done = true", "var", KEYWORD_COLOR);
	check_token_color(highlighted, 8, "var done = true", "true", KEYWORD_COLOR);
}

TEST_CASE("[Editor] separates raw strings, escape sequences, and prefixed literals") {
	using namespace Highlighter;

	HighlighterFixture fixture;
	const String code = "var escaped = \"line\\t%04d\" % 7\n"
						"var raw = r\"line\\t{placeholder}\"\n"
						"var refs = value && &\"name\" != ^\"path\"\n";
	HighlightedSource highlighted(code);

	check_token_color(highlighted, 0, "var escaped = \"line\\t%04d\" % 7", "\"line", STRING_COLOR);
	check_token_color(highlighted, 0, "var escaped = \"line\\t%04d\" % 7", "\\t", SYMBOL_COLOR);
	check_token_color(highlighted, 0, "var escaped = \"line\\t%04d\" % 7", "%04d", PLACEHOLDER_COLOR);
	check_token_color(highlighted, 0, "var escaped = \"line\\t%04d\" % 7", "7", NUMBER_COLOR);

	check_token_color(highlighted, 1, "var raw = r\"line\\t{placeholder}\"", "r\"line", STRING_COLOR);
	check_token_color(highlighted, 1, "var raw = r\"line\\t{placeholder}\"", "\\t", STRING_COLOR);

	check_token_color(highlighted, 2, "var refs = value && &\"name\" != ^\"path\"", "&&", SYMBOL_COLOR);
	check_token_color(highlighted, 2, "var refs = value && &\"name\" != ^\"path\"", "&\"name\"", STRING_NAME_COLOR);
	check_token_color(highlighted, 2, "var refs = value && &\"name\" != ^\"path\"", "^\"path\"", NODE_PATH_COLOR);
}

TEST_CASE("[Editor] distinguishes global functions from identifiers and methods") {
	using namespace Highlighter;

	HighlighterFixture fixture;
	const String code = "func demo():\n"
						"\tprint\n"
						"\tprint()\n"
						"\tobject.print()\n"
						"\tobject.if\n";
	HighlightedSource highlighted(code);

	check_token_color(highlighted, 0, "func demo():", "demo", FUNCTION_DEFINITION_COLOR);
	check_token_color(highlighted, 1, "\tprint", "print", FONT_COLOR);
	check_token_color(highlighted, 2, "\tprint()", "print", GLOBAL_FUNCTION_COLOR);
	check_token_color(highlighted, 3, "\tobject.print()", "print", FUNCTION_COLOR);
	check_token_color(highlighted, 4, "\tobject.if", "if", MEMBER_COLOR);
}
}
} // namespace GDScriptTests

#endif // defined(TOOLS_ENABLED) && !defined(ADVANCED_GUI_DISABLED)
