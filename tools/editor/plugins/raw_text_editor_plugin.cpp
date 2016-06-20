/*************************************************************************/
/*  raw_text_editor_plugin.cpp                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#include "raw_text_editor_plugin.h"
#include "tools/editor/editor_settings.h"

#include "spatial_editor_plugin.h"
#include "io/resource_loader.h"
#include "io/resource_saver.h"
#include "os/keyboard.h"
#include "tools/editor/editor_node.h"
#include "tools/editor/property_editor.h"
#include "os/os.h"

Ref<RawText>& RawTextSourceEditor::get_edited_text() {
	return text;
}

void RawTextSourceEditor::set_edited_text(const Ref<RawText>& p_text) {

	text=p_text;
	language = text->get_path().extension();

	_load_theme_settings();

	get_text_edit()->set_text(text->get_text());
	get_text_edit()->call("cursor_set_blink_enabled", true);

	_line_col_changed();
}

void RawTextSourceEditor::set_highlight_language(const String& lang) {

	language = lang;
	_load_theme_settings();

}

HighlightConfig RawTextSourceEditor::get_highlight(const String& extension) {
	HighlightConfig hl;
	hl.filled = false;

	if (extension == "json") { // json
		hl.filled = true;

		hl.string_delimiters.push_back("\" \"");

		hl.multi_line_comment_beg = "/*";
		hl.multi_line_comment_beg = "*/";

		String keywords [] = {
			"null", "true", "false"
		};
		for (size_t i=0; i < sizeof (keywords) / sizeof (String); i++) {
			hl.keywords.push_back(keywords[i]);
		}
	}
	else if (extension == "xml" || extension == "html" || extension == "htm" || extension == "xhtml") { // xml
		hl.filled = true;
		hl.string_delimiters.push_back("\" \"");
	}
	else if (extension == "yml" || extension == "yaml") { // yaml
		hl.filled = true;

		hl.line_comment = "#";
		hl.string_delimiters.push_back("\" \"");
		hl.string_delimiters.push_back("\"\"\" \"\"\"");

		hl.multi_line_comment_beg = "/*";
		hl.multi_line_comment_beg = "*/";
		String keywords [] = {
			"null", "true", "false"
		};
		for (size_t i=0; i < sizeof (keywords) / sizeof (String); i++) {
			hl.keywords.push_back(keywords[i]);
		}
	}
	else if (extension == "cfg" || extension == "toml" || extension == "ini" ) { // toml
		hl.filled = true;
		hl.string_delimiters.push_back("\" \"");
		hl.string_delimiters.push_back("\"\"\" \"\"\"");
		hl.string_delimiters.push_back("' '");

		hl.line_comment = "#";
		hl.multi_line_comment_beg = ";";
		hl.multi_line_comment_beg = "";
		String keywords [] = {
			"null", "true", "false"
		};
		for (size_t i=0; i < sizeof (keywords) / sizeof (String); i++) {
			hl.keywords.push_back(keywords[i]);
		}
	}
	else if (extension == "c++" || extension == "cxx" || extension == "cpp" || extension == "h" || extension == "hpp") { // C++
		hl.filled = true;
		hl.string_delimiters.push_back("R\"( )\"");
		hl.string_delimiters.push_back("\" \"");
		hl.string_delimiters.push_back("' '");

		hl.line_comment = "//";
		hl.multi_line_comment_beg = "/*";
		hl.multi_line_comment_beg = "*/";

		String keywords [] = {
			"alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor",
			"bool", "break", "case", "catch", "char", "char16_t", "char32_t",
			"class", "compl", "concept", "const", "constexpr", "const_cast",
			"continue", "decltype", "default", "delete", "do", "double", "else",
			"dynamic_cast", "enum", "explicit", "export", "extern", "false", "for",
			"float","friend", "goto", "if", "inline", "int", "long", "mutable",
			"namespace", "new", "noexcept", "not", "not_eq", "nullptr", "operator",
			"or", "or_eq", "private", "protected", "public", "register",
			"reinterpret_cast", "requires", "return", "short", "signed", "sizeof",
			"static", "static_assert", "static_cast", "struct", "switch", "xor_eq",
			"template", "this", "thread_local", "throw", "true", "try", "typedef",
			"typeid", "typename", "union", "unsigned", "using", "virtual", "void",
			"volatile", "wchar_t", "while", "xor"
		};
		for (size_t i=0; i < sizeof (keywords) / sizeof (String); i++) {
			hl.keywords.push_back(keywords[i]);
		}
	}
	else if( extension == "cs") { // C#

		hl.filled = true;

		hl.string_delimiters.push_back("\" \"");
		hl.string_delimiters.push_back("' '");

		hl.line_comment = "//";
		hl.multi_line_comment_beg = "/*";
		hl.multi_line_comment_beg = "*/";

		String keywords [] = {
			"abstract", "as", "base", "bool", "break", "byte", "case", "catch",
			"char", "checked", "class", "const", "continue", "decimal", "default",
			"delegate", "do", "double", "else","enum", "event", "explicit", "extern",
			"false", "finally", "fixed", "float", "for", "foreach", "goto", "if",
			"implicit", "in", "int", "interface", "internal", "is", "lock", "long",
			"namespace", "new", "null", "object", "operator", "out", "override",
			"params", "private", "protected", "public", "readonly", "ref", "return",
			"sbyte", "sealed", "short", "sizeof", "stackalloc", "static", "string",
			"struct", "switch", "this", "throw", "true", "try", "typeof", "uint",
			"ulong", "unchecked", "unsafe", "ushort", "using", "virtual", "void",
			"volatile", "while", "add", "alias", "ascending", "async", "await",
			"descending", "dynamic", "from", "get", "global", "group", "into",
			"join", "let", "orderby", "remove", "select", "set", "value", "var",
			"where", "yield"
		};
		for (size_t i=0; i < sizeof (keywords) / sizeof (String); i++) {
			hl.keywords.push_back(keywords[i]);
		}

	}
	else if( extension == "js") { // Javascript

		hl.filled = true;

		hl.string_delimiters.push_back("\" \"");
		hl.string_delimiters.push_back("' '");
		hl.string_delimiters.push_back("$` `");

		hl.line_comment = "//";
		hl.multi_line_comment_beg = "/*";
		hl.multi_line_comment_beg = "*/";

		String keywords [] = {
			"do", "if", "in", "for", "let", "new", "try", "var", "case", "else",
			"enum", "eval", "null", "this", "true", "void", "with", "await", "break",
			"catch", "class", "const", "false", "super", "throw", "while", "yield",
			"delete", "export", "import", "public", "return", "static", "switch",
			"typeof", "default", "extends", "finally", "package", "private",
			"continue", "debugger", "function", "arguments", "interface", "protected",
			"implements", "instanceof"
		};
		for (size_t i=0; i < sizeof (keywords) / sizeof (String); i++) {
			hl.keywords.push_back(keywords[i]);
		}
	}
	else if( extension == "css") { // CSS

		hl.filled = true;

		hl.string_delimiters.push_back("\" \"");
		hl.multi_line_comment_beg = "/*";
		hl.multi_line_comment_beg = "*/";

		String keywords [] = {
			"accelerator", "azimuth", "background", "background-attachment",
			"background-color", "background-image", "background-position",
			"background-position-x", "background-position-y", "background-repeat",
			"behavior", "border", "border-bottom", "border-bottom-color", "border-bottom-style",
			"border-bottom-width", "border-collapse", "border-color", "border-left",
			"border-left-color", "border-left-style", "border-left-width", "border-right",
			"border-right-color", "border-right-style", "border-right-width",
			"border-spacing", "border-style", "border-top", "border-top-color",
			"border-top-style", "border-top-width", "border-width", "bottom",
			"caption-side", "clear", "clip", "color", "content", "counter-increment",
			"counter-reset", "cue", "cue-after", "cue-before", "cursor", "direction",
			"display", "elevation", "empty-cells", "filter", "float", "font", "font-family",
			"font-size", "font-size-adjust", "font-stretch", "font-style", "font-variant",
			"font-weight", "height", "ime-mode", "include-source", "layer-background-color",
			"layer-background-image", "layout-flow", "layout-grid", "layout-grid-char",
			"layout-grid-char-spacing", "layout-grid-line", "layout-grid-mode",
			"layout-grid-type", "left", "letter-spacing", "line-break", "line-height",
			"list-style", "list-style-image", "list-style-position", "list-style-type",
			"margin", "margin-bottom", "margin-left", "margin-right", "margin-top",
			"marker-offset", "marks", "max-height", "max-width", "min-height",
			"min-width", "-moz-binding", "-moz-border-radius",
			"-moz-border-radius-topleft", "-moz-border-radius-topright",
			"-moz-border-radius-bottomright", "-moz-border-radius-bottomleft",
			"-moz-border-top-colors", "-moz-border-right-colors",
			"-moz-border-bottom-colors", "-moz-border-left-colors",
			"-moz-opacity", "-moz-outline", "-moz-outline-color",
			"-moz-outline-style", "-moz-outline-width", "-moz-user-focus",
			"-moz-user-input", "-moz-user-modify", "-moz-user-select", "orphans",
			"outline", "outline-color", "outline-style", "outline-width", "overflow",
			"overflow-X", "overflow-Y", "padding", "padding-bottom", "padding-left",
			"padding-right", "padding-top", "page", "page-break-after", "page-break-before",
			"page-break-inside", "pause", "pause-after", "pause-before", "pitch",
			"pitch-range", "play-during", "position", "quotes", "-replace", "richness",
			"right", "ruby-align", "ruby-overhang", "ruby-position", "-set-link-source",
			"size", "speak", "speak-header", "speak-numeral", "speak-punctuation",
			"speech-rate", "stress", "scrollbar-arrow-color", "scrollbar-base-color",
			"scrollbar-dark-shadow-color", "scrollbar-face-color", "scrollbar-highlight-color",
			"scrollbar-shadow-color", "scrollbar-3d-light-color", "scrollbar-track-color",
			"table-layout", "text-align", "text-align-last", "text-decoration",
			"text-indent", "text-justify", "text-overflow", "text-shadow",
			"text-transform", "text-autospace", "text-kashida-space", "text-underline-position",
			"top", "unicode-bidi", "-use-link-source", "vertical-align", "visibility",
			"voice-family", "volume", "white-space", "widows", "width", "word-break",
			"word-spacing", "word-wrap", "writing-mode", "z-index", "zoom",
			"rgb", "rgba", "hsl", "hsla", "#", "~", "transparent"
		};
		for (size_t i=0; i < sizeof (keywords) / sizeof (String); i++) {
			hl.keywords.push_back(keywords[i]);
		}
	}
	else if (extension == "py") { // python
		hl.filled = true;

		hl.string_delimiters.push_back("\" \"");
		hl.string_delimiters.push_back("' '");
		hl.string_delimiters.push_back("\"\"\" \"\"\"");

		hl.line_comment = "#";

		String pykeywords [] = {
			"and", "del", "from", "not", "while", "as", "elif", "global", "or",
			"with", "assert", "else", "if", "pass", "yield", "break", "except",
			"import", "print", "class", "exec", "in", "raise", "continue",
			"finally", "is", "return", "def", "for", "lambda", "try"
		};
		for (size_t i=0; i < sizeof (pykeywords) / sizeof (String); i++) {
			hl.keywords.push_back(pykeywords[i]);
		}
	}
	else if (extension == "sh" || extension == "bash") { // Bash shell
		hl.filled = true;

		hl.string_delimiters.push_back("\" \"");
		hl.string_delimiters.push_back("' '");

		hl.line_comment = "#";

		String pykeywords [] = {
			".", ":", "[", "alias", "bg", "bind", "break", "builtin", "caller",
			"cd", "ls", "mkdir", "command", "compgen", "complete", "compopt",
			"dirs", "disown", "echo", "enable", "eval", "exec", "exit", "export",
			"false", "fc", "fg", "getopts", "hash", "help", "history", "jobs",
			"kill", "let", "local", "logout", "mapfile", "popd", "printf", "pushd",
			"pwd", "read", "readarray", "readonly", "return", "set", "shift",
			"shopt", "source", "suspend", "test", "times", "trap", "true", "type",
			"typeset", "ulimit", "umask", "unalias", "unset", "wait", "if", "then",
			"else", "elif", "fi", "case", "esac", "for", "select", "while", "until",
			"do", "done", "in", "function", "time", "{", "}", "!", "[[", "]]",
			"coproc", "continue", "declare"
		};
		for (size_t i=0; i < sizeof (pykeywords) / sizeof (String); i++) {
			hl.keywords.push_back(pykeywords[i]);
		}
	}
	return hl;
}

void RawTextSourceEditor::_load_theme_settings() {

	get_text_edit()->clear_colors();

	Color font_color = EDITOR_DEF("text_editor/text_color",Color(0,0,0));
	Color keyword_color= EDITOR_DEF("text_editor/keyword_color",Color(0.5,0.0,0.2));
	Color comment_color = EDITOR_DEF("text_editor/comment_color",Color::hex(0x797e7eff));
	Color string_color = EDITOR_DEF("text_editor/string_color",Color::hex(0x6b6f00ff));

	get_text_edit()->add_color_override("font_color", font_color);
	get_text_edit()->set_custom_bg_color(EDITOR_DEF("text_editor/background_color",Color(0,0,0,0)));
	get_text_edit()->add_color_override("line_number_color",EDITOR_DEF("text_editor/line_number_color",Color(0,0,0)));
	get_text_edit()->add_color_override("caret_color",EDITOR_DEF("text_editor/caret_color",Color(0,0,0)));
	get_text_edit()->add_color_override("font_selected_color",EDITOR_DEF("text_editor/text_selected_color",Color(1,1,1)));
	get_text_edit()->add_color_override("selection_color",EDITOR_DEF("text_editor/selection_color",Color(0.2,0.2,1)));
	get_text_edit()->add_color_override("current_line_color",EDITOR_DEF("text_editor/current_line_color",Color(0.3,0.5,0.8,0.15)));
	get_text_edit()->add_color_override("search_result_color",EDITOR_DEF("text_editor/search_result_color",Color(0.05,0.25,0.05,1)));
	get_text_edit()->add_color_override("search_result_border_color",EDITOR_DEF("text_editor/search_result_border_color",Color(0.1,0.45,0.1,1)));
	get_text_edit()->add_color_override("mark_color", EDITOR_DEF("text_editor/mark_color", Color(1.0,0.4,0.4,0.4)));
	get_text_edit()->add_color_override("word_highlighted_color",EDITOR_DEF("text_editor/word_highlighted_color",Color(0.8,0.9,0.9,0.15)));

	HighlightConfig hl = get_highlight(language);
	if (hl.filled) {

		// colorize code
		get_text_edit()->add_color_override("brace_mismatch_color",EDITOR_DEF("text_editor/brace_mismatch_color",Color(1,0.2,0.2)));
		get_text_edit()->add_color_override("number_color",EDITOR_DEF("text_editor/number_color",Color(0.9,0.6,0.0,2)));
		get_text_edit()->add_color_override("function_color",EDITOR_DEF("text_editor/function_color",Color(0.4,0.6,0.8)));
		get_text_edit()->add_color_override("member_variable_color",EDITOR_DEF("text_editor/member_variable_color",Color(0.9,0.3,0.3)));
		get_text_edit()->add_color_override("breakpoint_color", EDITOR_DEF("text_editor/breakpoint_color", Color(0.8,0.8,0.4,0.2)));

		// colorize strings
		for (List<String>::Element *E=hl.string_delimiters.front();E;E=E->next()) {
			String string = E->get();
			String beg = string.get_slice(" ",0);
			String end = string.get_slice_count(" ")>1?string.get_slice(" ",1):String();
			get_text_edit()->add_color_region(beg,end,string_color,end=="");
		}

		// colorize comments
		if (hl.multi_line_comment_beg.length())
			get_text_edit()->add_color_region(hl.multi_line_comment_beg, hl.multi_line_comment_end,comment_color,false);
		if (hl.line_comment.length())
			get_text_edit()->add_color_region(hl.line_comment,"",comment_color,false);

		// colorize keywords
		for(List<String>::Element *E=hl.keywords.front();E;E=E->next()) {
			get_text_edit()->add_keyword_color(E->get(),keyword_color);
		}

		// colorize symbols
		Color symbol_color= EDITOR_DEF("text_editor/symbol_color",Color::hex(0x005291ff));
		get_text_edit()->set_symbol_color(symbol_color);
	}
	else {
		get_text_edit()->add_color_override("brace_mismatch_color", font_color);
		get_text_edit()->add_color_override("number_color", font_color);
		get_text_edit()->add_color_override("function_color", font_color);
		get_text_edit()->add_color_override("member_variable_color", font_color);
		get_text_edit()->add_color_override("breakpoint_color",  font_color);
		get_text_edit()->set_symbol_color(font_color);
	}

}

void RawTextSourceEditor::_validate_script() {

}


void RawTextSourceEditor::_bind_methods() {

}

RawTextSourceEditor::RawTextSourceEditor() {
}

void RawTextEditor::_menu_option(int p_option) {

	switch(p_option) {
		case EDIT_UNDO: {
			text_editor->get_text_edit()->undo();
		} break;
		case EDIT_REDO: {
			text_editor->get_text_edit()->redo();

		} break;
		case EDIT_CUT: {

			text_editor->get_text_edit()->cut();
		} break;
		case EDIT_COPY: {
			text_editor->get_text_edit()->copy();

		} break;
		case EDIT_PASTE: {
			text_editor->get_text_edit()->paste();

		} break;
		case EDIT_SELECT_ALL: {

			text_editor->get_text_edit()->select_all();

		} break;
		case SEARCH_FIND: {

			text_editor->get_find_replace_bar()->popup_search();
		} break;
		case SEARCH_FIND_NEXT: {

			text_editor->get_find_replace_bar()->search_next();
		} break;
		case SEARCH_FIND_PREV: {

			text_editor->get_find_replace_bar()->search_prev();
		} break;
		case SEARCH_REPLACE: {

			text_editor->get_find_replace_bar()->popup_replace();
		} break;
		case SEARCH_GOTO_LINE: {
			goto_line_dialog->popup_find_line(text_editor->get_text_edit());
		} break;

	}
}

void RawTextEditor::_notification(int p_what) {

	if (p_what==NOTIFICATION_ENTER_TREE) {

		close->set_normal_texture( get_icon("Close","EditorIcons"));
		close->set_hover_texture( get_icon("CloseHover","EditorIcons"));
		close->set_pressed_texture( get_icon("Close","EditorIcons"));
		close->connect("pressed",this,"_close_callback");

	}
	if (p_what==NOTIFICATION_DRAW) {

		RID ci = get_canvas_item();
		Ref<StyleBox> style = get_stylebox("panel","Panel");
		style->draw( ci, Rect2( Point2(), get_size() ) );

	}

}

Dictionary RawTextEditor::get_state() const {
	return Dictionary();
}

void RawTextEditor::set_state(const Dictionary& p_state) {

}

void RawTextEditor::clear() {
	text_editor->get_text_edit()->set_text("");
}

void RawTextEditor::_text_changed() {
	text_editor->_validate_script();
	apply_text();
}

void RawTextEditor::_highlight_selected(int lang) {
	String langs = "";

	switch (lang) {
		case HighlightConfig::LANG_CPP:
			langs = "cpp";
			break;
		case HighlightConfig::LANG_CSHARP:
			langs = "cs";
			break;
		case HighlightConfig::LANG_CSS:
			langs = "css";
			break;
		case HighlightConfig::LANG_JAVASCRIPT:
			langs = "js";
			break;
		case HighlightConfig::LANG_JSON:
			langs = "json";
			break;
		case HighlightConfig::LANG_PYTHON:
			langs = "py";
			break;
		case HighlightConfig::LANG_XML:
			langs = "xml";
			break;
		case HighlightConfig::LANG_SHELL:
			langs = "sh";
			break;
		case HighlightConfig::LANG_TOML:
			langs = "toml";
			break;
		case HighlightConfig::LANG_YAML:
			langs = "yaml";
			break;
		case HighlightConfig::LANG_NONE:
		default:
			langs = "txt";
			break;
	}

	text_editor->set_highlight_language(langs);
}

void RawTextEditor::_editor_settings_changed() {

	text_editor->get_text_edit()->set_auto_brace_completion(EditorSettings::get_singleton()->get("text_editor/auto_brace_complete"));
	text_editor->get_text_edit()->set_scroll_pass_end_of_file(EditorSettings::get_singleton()->get("text_editor/scroll_past_end_of_file"));
	text_editor->get_text_edit()->set_tab_size(EditorSettings::get_singleton()->get("text_editor/tab_size"));
	text_editor->get_text_edit()->set_draw_tabs(EditorSettings::get_singleton()->get("text_editor/draw_tabs"));
	text_editor->get_text_edit()->set_show_line_numbers(EditorSettings::get_singleton()->get("text_editor/show_line_numbers"));
	text_editor->get_text_edit()->set_syntax_coloring(EditorSettings::get_singleton()->get("text_editor/syntax_highlighting"));
	text_editor->get_text_edit()->set_highlight_all_occurrences(EditorSettings::get_singleton()->get("text_editor/highlight_all_occurrences"));
	text_editor->get_text_edit()->cursor_set_blink_enabled(EditorSettings::get_singleton()->get("text_editor/caret_blink"));
	text_editor->get_text_edit()->cursor_set_blink_speed(EditorSettings::get_singleton()->get("text_editor/caret_blink_speed"));
	text_editor->get_text_edit()->add_constant_override("line_spacing", EditorSettings::get_singleton()->get("text_editor/line_spacing"));

}

void RawTextEditor::_update_lang_option() {
	String extension = text->get_path().extension();

	if (extension == "json")
		lang_option->select(HighlightConfig::LANG_JSON);
	else if (extension == "xml" || extension == "html" || extension == "htm" || extension == "xhtml")
		lang_option->select(HighlightConfig::LANG_XML);
	else if (extension == "yml" || extension == "yaml")
		lang_option->select(HighlightConfig::LANG_YAML);
	else if (extension == "sh" || extension == "bash")
		lang_option->select(HighlightConfig::LANG_SHELL);
	else if (extension == "cfg" || extension == "toml" || extension == "ini" )
		lang_option->select(HighlightConfig::LANG_TOML);
	else if (extension == "c++" || extension == "cxx" || extension == "cpp" || extension == "h" || extension == "hpp")
		lang_option->select(HighlightConfig::LANG_CPP);
	else if (extension == "py")
		lang_option->select(HighlightConfig::LANG_PYTHON);
	else if(extension == "js")
		lang_option->select(HighlightConfig::LANG_JAVASCRIPT);
	else if(extension == "cs")
		lang_option->select(HighlightConfig::LANG_CSHARP);
	else if(extension == "css")
		lang_option->select(HighlightConfig::LANG_CSS);
	else
		lang_option->select(HighlightConfig::LANG_NONE);

}

void RawTextEditor::_update_title() {

	String title = text->get_path();
	if (title.length() == 0)
		title = "Untitled";
	title_label->set_text(title);
}

void RawTextEditor::_bind_methods() {

	ObjectTypeDB::bind_method("_editor_settings_changed",&RawTextEditor::_editor_settings_changed);
	ObjectTypeDB::bind_method("_menu_option",&RawTextEditor::_menu_option);
	ObjectTypeDB::bind_method("_close_callback",&RawTextEditor::_close_callback);
	ObjectTypeDB::bind_method("_text_changed",&RawTextEditor::_text_changed);
	ObjectTypeDB::bind_method(_MD("_highlight_selected","lang"),&RawTextEditor::_highlight_selected);
	ObjectTypeDB::bind_method("apply_text",&RawTextEditor::apply_text);
}

void RawTextEditor::ensure_select_current() {

}

void RawTextEditor::edit(const Ref<RawText>& p_text) {

	if (p_text.is_null())
		return;

	text=p_text;
	text_editor->set_edited_text(text);

	_update_title();
	_update_lang_option();
}

void RawTextEditor::save_external_data() {

	if (text.is_null())
		return;
	apply_text();

	if (text->get_path()!="" && text->get_path().find("local://")==-1 &&text->get_path().find("::")==-1) {
		//external text, save it
		ResourceSaver::save(text->get_path(),text);
	}
}

void RawTextEditor::apply_text()  {

	if (text.is_valid()) {
		text->set_text(text_editor->get_text_edit()->get_text());
		text->set_edited(true);
	}
}

void RawTextEditor::_close_callback() {

	hide();
}


RawTextEditor::RawTextEditor() {

	VBoxContainer* container = memnew(VBoxContainer);
	add_child(container);
	container->set_area_as_parent_rect();
	container->set_begin(Point2(0,0));

	Control * menuspacer = memnew(Control);
	menuspacer->set_custom_minimum_size(Size2(0,5));
	container->add_child(menuspacer);

	HBoxContainer * menubar = memnew(HBoxContainer);
	container->add_child(menubar);
	menubar->set_h_size_flags(SIZE_EXPAND_FILL);

	edit_menu = memnew( MenuButton );
	menubar->add_child(edit_menu);
	edit_menu->set_pos(Point2(5,-1));
	edit_menu->set_text(TTR("Edit"));
	edit_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/undo", TTR("Undo"), KEY_MASK_CMD|KEY_Z), EDIT_UNDO);
	edit_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/redo", TTR("Redo"), KEY_MASK_CMD|KEY_MASK_SHIFT|KEY_Z), EDIT_REDO);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/cut", TTR("Cut"), KEY_MASK_CMD|KEY_X), EDIT_CUT);
	edit_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/copy", TTR("Copy"), KEY_MASK_CMD|KEY_C), EDIT_COPY);
	edit_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/paste", TTR("Paste"), KEY_MASK_CMD|KEY_V), EDIT_PASTE);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/select_all", TTR("Select All"), KEY_MASK_CMD|KEY_A), EDIT_SELECT_ALL);
	edit_menu->get_popup()->connect("item_pressed", this,"_menu_option");


	search_menu = memnew( MenuButton );
	menubar->add_child(search_menu);
	search_menu->set_pos(Point2(38,-1));
	search_menu->set_text(TTR("Search"));
	search_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/find", TTR("Find.."), KEY_MASK_CMD|KEY_F), SEARCH_FIND);
	search_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/find_next", TTR("Find Next"), KEY_F3), SEARCH_FIND_NEXT);
	search_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/find_previous", TTR("Find Previous"), KEY_MASK_SHIFT|KEY_F3), SEARCH_FIND_PREV);
	search_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/replace", TTR("Replace.."), KEY_MASK_CMD|KEY_R), SEARCH_REPLACE);
	search_menu->get_popup()->add_separator();
//	search_menu->get_popup()->add_item("Locate Symbol..",SEARCH_LOCATE_SYMBOL,KEY_MASK_CMD|KEY_K);
	search_menu->get_popup()->add_shortcut(ED_SHORTCUT("script_editor/goto_line", TTR("Goto Line.."), KEY_MASK_CMD|KEY_G), SEARCH_GOTO_LINE);
	search_menu->get_popup()->connect("item_pressed", this,"_menu_option");

	title_label = memnew( Label );
	title_label->set_h_size_flags(SIZE_EXPAND_FILL);
	title_label->set_align(Label::ALIGN_CENTER);
	menubar->add_child(title_label);

	lang_option = memnew( OptionButton );
	menubar->add_child(lang_option);
	lang_option->set_pos(Point2(0,10));
	lang_option->add_item(TTR("Plain Text"), HighlightConfig::LANG_NONE);
	lang_option->add_item(TTR("C/C++"), HighlightConfig::LANG_CPP);
	lang_option->add_item(TTR("C#"), HighlightConfig::LANG_CSHARP);
	lang_option->add_item(TTR("CSS"), HighlightConfig::LANG_CSS);
	lang_option->add_item(TTR("JavaScript"), HighlightConfig::LANG_JAVASCRIPT);
	lang_option->add_item(TTR("Json"), HighlightConfig::LANG_JSON);
	lang_option->add_item(TTR("Python"), HighlightConfig::LANG_PYTHON);
	lang_option->add_item(TTR("Shell"),  HighlightConfig::LANG_SHELL);
	lang_option->add_item(TTR("TOML, INI"),  HighlightConfig::LANG_TOML);
	lang_option->add_item(TTR("XML"), HighlightConfig::LANG_XML);
	lang_option->add_item(TTR("YAML"), HighlightConfig::LANG_YAML);
	lang_option->select(HighlightConfig::LANG_NONE);
	lang_option->connect("item_selected", this, "_highlight_selected");

	close = memnew( TextureButton );
	menubar->add_child(close);

	erase_tab_confirm = memnew( ConfirmationDialog );
	add_child(erase_tab_confirm);
	erase_tab_confirm->connect("confirmed", this,"_close_current_tab");

	goto_line_dialog = memnew(GotoLineDialog);
	add_child(goto_line_dialog);

	text_editor = memnew( RawTextSourceEditor );
	container->add_child(text_editor);
	text_editor->set_h_size_flags(SIZE_EXPAND_FILL);
	text_editor->set_v_size_flags(SIZE_EXPAND_FILL);
	text_editor->get_text_edit()->connect("text_changed", this,"_text_changed");

	EditorSettings::get_singleton()->connect("settings_changed",this,"_editor_settings_changed");
	_editor_settings_changed();
}


void RawTextEditorPlugin::edit(Object *p_object) {

	if (!p_object->cast_to<RawText>())
		return;

	text_editor->edit(p_object->cast_to<RawText>());
}

bool RawTextEditorPlugin::handles(Object *p_object) const {

	return p_object->cast_to<RawText>() != NULL;
}

void RawTextEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		text_editor->show();
	} else {
		text_editor->apply_text();
	}
}

void RawTextEditorPlugin::selected_notify() {

	text_editor->ensure_select_current();
}

Dictionary RawTextEditorPlugin::get_state() const {

	return text_editor->get_state();
}

void RawTextEditorPlugin::set_state(const Dictionary& p_state) {

	text_editor->set_state(p_state);
}
void RawTextEditorPlugin::clear() {

	text_editor->clear();
}

void RawTextEditorPlugin::save_external_data() {

	text_editor->save_external_data();
}

void RawTextEditorPlugin::apply_changes() {

	text_editor->apply_text();

}

RawTextEditorPlugin::RawTextEditorPlugin(EditorNode *p_node, bool p_2d) {

	editor=p_node;
	text_editor = memnew( RawTextEditor );
	if (p_2d)
		add_control_to_container(CONTAINER_CANVAS_EDITOR_BOTTOM,text_editor);
	else
		add_control_to_container(CONTAINER_SPATIAL_EDITOR_BOTTOM,text_editor);
	text_editor->hide();

}


RawTextEditorPlugin::~RawTextEditorPlugin() {

}
