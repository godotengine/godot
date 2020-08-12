/*************************************************************************/
/*  gdscript_translation_parser_plugin.h                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef GDSCRIPT_TRANSLATION_PARSER_PLUGIN_H
#define GDSCRIPT_TRANSLATION_PARSER_PLUGIN_H

#include "core/set.h"
#include "editor/editor_translation_parser.h"
#include "modules/gdscript/gdscript_parser.h"
#include "modules/regex/regex.h"

class GDScriptEditorTranslationParserPlugin : public EditorTranslationParserPlugin {
	GDCLASS(GDScriptEditorTranslationParserPlugin, EditorTranslationParserPlugin);

	Vector<String> *ids;
	Vector<Vector<String>> *ids_ctx_plural;

	// List of patterns used for extracting translation strings.
	StringName tr_func = "tr";
	StringName trn_func = "tr_n";
	Set<StringName> assignment_patterns;
	Set<StringName> first_arg_patterns;
	Set<StringName> second_arg_patterns;
	// FileDialog patterns.
	StringName fd_add_filter = "add_filter";
	StringName fd_set_filter = "set_filters";
	StringName fd_filters = "filters";

	void _traverse_class(const GDScriptParser::ClassNode *p_class);
	void _traverse_function(const GDScriptParser::FunctionNode *p_func);
	void _traverse_block(const GDScriptParser::SuiteNode *p_suite);

	void _read_variable(const GDScriptParser::VariableNode *p_var);
	void _assess_expression(GDScriptParser::ExpressionNode *p_expression);
	void _assess_assignment(GDScriptParser::AssignmentNode *p_assignment);
	void _extract_from_call(GDScriptParser::CallNode *p_call);
	void _extract_fd_literals(GDScriptParser::ExpressionNode *p_expression);

public:
	virtual Error parse_file(const String &p_path, Vector<String> *r_ids, Vector<Vector<String>> *r_ids_ctx_plural) override;
	virtual void get_recognized_extensions(List<String> *r_extensions) const override;

	GDScriptEditorTranslationParserPlugin();
};

#endif // GDSCRIPT_TRANSLATION_PARSER_PLUGIN_H
