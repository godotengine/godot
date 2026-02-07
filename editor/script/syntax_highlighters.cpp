/**************************************************************************/
/*  syntax_highlighters.cpp                                               */
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

#include "syntax_highlighters.h"

#include "core/config/project_settings.h"
#include "core/object/script_language.h"
#include "editor/settings/editor_settings.h"

String EditorSyntaxHighlighter::_get_name() const {
	String ret = "Unnamed";
	GDVIRTUAL_CALL(_get_name, ret);
	return ret;
}

PackedStringArray EditorSyntaxHighlighter::_get_supported_languages() const {
	PackedStringArray ret;
	GDVIRTUAL_CALL(_get_supported_languages, ret);
	return ret;
}

Ref<EditorSyntaxHighlighter> EditorSyntaxHighlighter::_create() const {
	Ref<EditorSyntaxHighlighter> syntax_highlighter;
	if (GDVIRTUAL_IS_OVERRIDDEN(_create)) {
		GDVIRTUAL_CALL(_create, syntax_highlighter);
	} else {
		syntax_highlighter.instantiate();
		if (get_script_instance()) {
			syntax_highlighter->set_script(get_script_instance()->get_script());
		}
	}
	return syntax_highlighter;
}

void EditorSyntaxHighlighter::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_get_edited_resource"), &EditorSyntaxHighlighter::_get_edited_resource);

	GDVIRTUAL_BIND(_get_name)
	GDVIRTUAL_BIND(_get_supported_languages)
	GDVIRTUAL_BIND(_create)
}

////

void EditorStandardSyntaxHighlighter::_update_cache() {
	highlighter->set_text_edit(text_edit);
	highlighter->clear_keyword_colors();
	highlighter->clear_member_keyword_colors();
	highlighter->clear_color_regions();

	highlighter->set_symbol_color(EDITOR_GET("text_editor/theme/highlighting/symbol_color"));
	highlighter->set_function_color(EDITOR_GET("text_editor/theme/highlighting/function_color"));
	highlighter->set_number_color(EDITOR_GET("text_editor/theme/highlighting/number_color"));
	highlighter->set_member_variable_color(EDITOR_GET("text_editor/theme/highlighting/member_variable_color"));

	/* Engine types. */
	const Color type_color = EDITOR_GET("text_editor/theme/highlighting/engine_type_color");
	LocalVector<StringName> types;
	ClassDB::get_class_list(types);
	for (const StringName &type : types) {
		highlighter->add_keyword_color(type, type_color);
	}

	/* User types. */
	const Color usertype_color = EDITOR_GET("text_editor/theme/highlighting/user_type_color");
	LocalVector<StringName> global_classes;
	ScriptServer::get_global_class_list(global_classes);
	for (const StringName &class_name : global_classes) {
		highlighter->add_keyword_color(class_name, usertype_color);
	}

	/* Autoloads. */
	HashMap<StringName, ProjectSettings::AutoloadInfo> autoloads(ProjectSettings::get_singleton()->get_autoload_list());
	for (const KeyValue<StringName, ProjectSettings::AutoloadInfo> &E : autoloads) {
		const ProjectSettings::AutoloadInfo &info = E.value;
		if (info.is_singleton) {
			highlighter->add_keyword_color(info.name, usertype_color);
		}
	}

	const ScriptLanguage *scr_lang = script_language;
	StringName instance_base;

	if (scr_lang == nullptr) {
		const Ref<Script> scr = _get_edited_resource();
		if (scr.is_valid()) {
			scr_lang = scr->get_language();
			instance_base = scr->get_instance_base_type();
		}
	}

	if (scr_lang != nullptr) {
		/* Core types. */
		const Color basetype_color = EDITOR_GET("text_editor/theme/highlighting/base_type_color");
		List<String> core_types;
		scr_lang->get_core_type_words(&core_types);
		for (const String &E : core_types) {
			highlighter->add_keyword_color(E, basetype_color);
		}

		/* Reserved words. */
		const Color keyword_color = EDITOR_GET("text_editor/theme/highlighting/keyword_color");
		const Color control_flow_keyword_color = EDITOR_GET("text_editor/theme/highlighting/control_flow_keyword_color");
		for (const String &keyword : scr_lang->get_reserved_words()) {
			if (scr_lang->is_control_flow_keyword(keyword)) {
				highlighter->add_keyword_color(keyword, control_flow_keyword_color);
			} else {
				highlighter->add_keyword_color(keyword, keyword_color);
			}
		}

		/* Member types. */
		const Color member_variable_color = EDITOR_GET("text_editor/theme/highlighting/member_variable_color");
		if (instance_base != StringName()) {
			List<PropertyInfo> plist;
			ClassDB::get_property_list(instance_base, &plist);
			for (const PropertyInfo &E : plist) {
				String prop_name = E.name;
				if (E.usage & PROPERTY_USAGE_CATEGORY || E.usage & PROPERTY_USAGE_GROUP || E.usage & PROPERTY_USAGE_SUBGROUP) {
					continue;
				}
				if (prop_name.contains_char('/')) {
					continue;
				}
				highlighter->add_member_keyword_color(prop_name, member_variable_color);
			}

			List<String> clist;
			ClassDB::get_integer_constant_list(instance_base, &clist);
			for (const String &E : clist) {
				highlighter->add_member_keyword_color(E, member_variable_color);
			}
		}

		/* Comments */
		const Color comment_color = EDITOR_GET("text_editor/theme/highlighting/comment_color");
		for (const String &comment : scr_lang->get_comment_delimiters()) {
			String beg = comment.get_slicec(' ', 0);
			String end = comment.get_slice_count(" ") > 1 ? comment.get_slicec(' ', 1) : String();
			highlighter->add_color_region(beg, end, comment_color, end.is_empty());
		}

		/* Doc comments */
		const Color doc_comment_color = EDITOR_GET("text_editor/theme/highlighting/doc_comment_color");
		for (const String &doc_comment : scr_lang->get_doc_comment_delimiters()) {
			String beg = doc_comment.get_slicec(' ', 0);
			String end = doc_comment.get_slice_count(" ") > 1 ? doc_comment.get_slicec(' ', 1) : String();
			highlighter->add_color_region(beg, end, doc_comment_color, end.is_empty());
		}

		/* Strings */
		const Color string_color = EDITOR_GET("text_editor/theme/highlighting/string_color");
		for (const String &string : scr_lang->get_string_delimiters()) {
			String beg = string.get_slicec(' ', 0);
			String end = string.get_slice_count(" ") > 1 ? string.get_slicec(' ', 1) : String();
			highlighter->add_color_region(beg, end, string_color, end.is_empty());
		}
	}
}

Ref<EditorSyntaxHighlighter> EditorStandardSyntaxHighlighter::_create() const {
	Ref<EditorStandardSyntaxHighlighter> syntax_highlighter;
	syntax_highlighter.instantiate();
	return syntax_highlighter;
}

////

Ref<EditorSyntaxHighlighter> EditorPlainTextSyntaxHighlighter::_create() const {
	Ref<EditorPlainTextSyntaxHighlighter> syntax_highlighter;
	syntax_highlighter.instantiate();
	return syntax_highlighter;
}

////

void EditorJSONSyntaxHighlighter::_update_cache() {
	highlighter->set_text_edit(text_edit);
	highlighter->clear_keyword_colors();
	highlighter->clear_member_keyword_colors();
	highlighter->clear_color_regions();

	highlighter->set_symbol_color(EDITOR_GET("text_editor/theme/highlighting/symbol_color"));
	highlighter->set_number_color(EDITOR_GET("text_editor/theme/highlighting/number_color"));

	const Color string_color = EDITOR_GET("text_editor/theme/highlighting/string_color");
	highlighter->add_color_region("\"", "\"", string_color);
}

Ref<EditorSyntaxHighlighter> EditorJSONSyntaxHighlighter::_create() const {
	Ref<EditorJSONSyntaxHighlighter> syntax_highlighter;
	syntax_highlighter.instantiate();
	return syntax_highlighter;
}

////

void EditorMarkdownSyntaxHighlighter::_update_cache() {
	highlighter->set_text_edit(text_edit);
	highlighter->clear_keyword_colors();
	highlighter->clear_member_keyword_colors();
	highlighter->clear_color_regions();

	// Disable automatic symbolic highlights, as these don't make sense for prose.
	highlighter->set_symbol_color(EDITOR_GET("text_editor/theme/highlighting/text_color"));
	highlighter->set_number_color(EDITOR_GET("text_editor/theme/highlighting/text_color"));
	highlighter->set_member_variable_color(EDITOR_GET("text_editor/theme/highlighting/text_color"));
	highlighter->set_function_color(EDITOR_GET("text_editor/theme/highlighting/text_color"));

	// Headings (any level).
	const Color function_color = EDITOR_GET("text_editor/theme/highlighting/function_color");
	highlighter->add_color_region("#", "", function_color);

	// Bold.
	highlighter->add_color_region("**", "**", function_color);
	// `__bold__` syntax is not supported as color regions must begin with a symbol,
	// not a character that is valid in an identifier.

	// Code (both inline code and triple-backticks code blocks).
	const Color code_color = EDITOR_GET("text_editor/theme/highlighting/engine_type_color");
	highlighter->add_color_region("`", "`", code_color);

	// Link (both references and inline links with URLs). The URL is not highlighted.
	const Color link_color = EDITOR_GET("text_editor/theme/highlighting/keyword_color");
	highlighter->add_color_region("[", "]", link_color);

	// Quote.
	const Color quote_color = EDITOR_GET("text_editor/theme/highlighting/string_color");
	highlighter->add_color_region(">", "", quote_color, true);

	// HTML comment, which is also supported in Markdown.
	const Color comment_color = EDITOR_GET("text_editor/theme/highlighting/comment_color");
	highlighter->add_color_region("<!--", "-->", comment_color);
}

Ref<EditorSyntaxHighlighter> EditorMarkdownSyntaxHighlighter::_create() const {
	Ref<EditorMarkdownSyntaxHighlighter> syntax_highlighter;
	syntax_highlighter.instantiate();
	return syntax_highlighter;
}

///

void EditorConfigFileSyntaxHighlighter::_update_cache() {
	highlighter->set_text_edit(text_edit);
	highlighter->clear_keyword_colors();
	highlighter->clear_member_keyword_colors();
	highlighter->clear_color_regions();

	highlighter->set_symbol_color(EDITOR_GET("text_editor/theme/highlighting/symbol_color"));
	highlighter->set_number_color(EDITOR_GET("text_editor/theme/highlighting/number_color"));
	// Assume that all function-style syntax is for types such as `Vector2()` and `PackedStringArray()`.
	highlighter->set_function_color(EDITOR_GET("text_editor/theme/highlighting/base_type_color"));

	// Disable member variable highlighting as it's not relevant for ConfigFile.
	highlighter->set_member_variable_color(EDITOR_GET("text_editor/theme/highlighting/text_color"));

	const Color string_color = EDITOR_GET("text_editor/theme/highlighting/string_color");
	highlighter->add_color_region("\"", "\"", string_color);

	// FIXME: Sections in ConfigFile must be at the beginning of a line. Otherwise, it can be an array within a line.
	const Color function_color = EDITOR_GET("text_editor/theme/highlighting/function_color");
	highlighter->add_color_region("[", "]", function_color);

	const Color keyword_color = EDITOR_GET("text_editor/theme/highlighting/keyword_color");
	highlighter->add_keyword_color("true", keyword_color);
	highlighter->add_keyword_color("false", keyword_color);
	highlighter->add_keyword_color("null", keyword_color);
	highlighter->add_keyword_color("ExtResource", keyword_color);
	highlighter->add_keyword_color("SubResource", keyword_color);

	const Color comment_color = EDITOR_GET("text_editor/theme/highlighting/comment_color");
	highlighter->add_color_region(";", "", comment_color);
}

Ref<EditorSyntaxHighlighter> EditorConfigFileSyntaxHighlighter::_create() const {
	Ref<EditorConfigFileSyntaxHighlighter> syntax_highlighter;
	syntax_highlighter.instantiate();
	return syntax_highlighter;
}
