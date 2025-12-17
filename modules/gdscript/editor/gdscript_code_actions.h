/**************************************************************************/
/*  gdscript_code_actions.h                                               */
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

#include "modules/gdscript/gdscript_parser.h"

namespace GDScriptCodeActions {

ScriptLanguage::CodeActionOperation add_script_annotation(const String &p_annotation, const String &p_file_path) {
	ScriptLanguage::CodeActionOperation op;
	ScriptLanguage::TextEdit te;

	te.start_line = 1;
	te.start_column = 1;
	te.end_line = 1;
	te.end_column = 1;

	te.new_text = vformat("%s\n", p_annotation);
	op.description = vformat("Add \"%s\"", p_annotation);

	ScriptLanguage::DocumentEditOperation de;
	de.edits.append(te);
	de.file_path = p_file_path;
	op.document_edits.append(de);

	return op;
}

ScriptLanguage::CodeActionOperation add_member_annotation(const String &p_annotation, const GDScriptParser::Node *p_node, const String &p_file_path) {
	ScriptLanguage::CodeActionOperation op;
	ScriptLanguage::TextEdit te;

	te.start_line = p_node->start_line;
	te.start_column = p_node->start_column;
	te.end_line = p_node->start_line;
	te.end_column = p_node->start_column;

	te.new_text = vformat("%s ", p_annotation);
	op.description = vformat("Add \"%s\"", p_annotation);

	ScriptLanguage::DocumentEditOperation de;
	de.edits.append(te);
	de.file_path = p_file_path;
	op.document_edits.append(de);

	return op;
}

ScriptLanguage::CodeActionOperation remove_underscore_prefix_from_identifier(const GDScriptParser::IdentifierNode *p_identifier_node, const String &p_type_description, const String &p_file_path) {
	ScriptLanguage::CodeActionOperation op;
	ScriptLanguage::TextEdit te;

	te.start_line = p_identifier_node->start_line;
	te.start_column = p_identifier_node->start_column;
	te.end_line = p_identifier_node->start_line;
	te.end_column = p_identifier_node->start_column + 1;

	te.new_text = "";
	op.description = vformat("Remove underscore from %s", p_type_description);

	ScriptLanguage::DocumentEditOperation de;
	de.edits.append(te);
	de.file_path = p_file_path;
	op.document_edits.append(de);

	return op;
}

ScriptLanguage::CodeActionOperation remove_statement(const GDScriptParser::Node *p_node, const String &p_type_description, const String &p_file_path) {
	ScriptLanguage::CodeActionOperation op;
	ScriptLanguage::TextEdit te;

	te.start_line = p_node->start_line;
	te.start_column = 1;
	te.end_line = p_node->end_line + 1;
	te.end_column = 1;

	te.new_text = "";
	op.description = vformat("Remove %s", p_type_description);

	ScriptLanguage::DocumentEditOperation de;
	de.edits.append(te);
	de.file_path = p_file_path;
	op.document_edits.append(de);

	return op;
}

ScriptLanguage::CodeActionOperation add_underscore_prefix_to_identifier(const GDScriptParser::IdentifierNode *p_identifier_node, const String &p_type_description, const String &p_file_path) {
	ScriptLanguage::CodeActionOperation op;
	ScriptLanguage::TextEdit te;

	te.start_line = p_identifier_node->start_line;
	te.start_column = p_identifier_node->start_column;
	te.end_line = p_identifier_node->start_line;
	te.end_column = p_identifier_node->start_column;

	te.new_text = "_";
	op.description = vformat("Add underscore to %s", p_type_description);

	ScriptLanguage::DocumentEditOperation de;
	de.edits.append(te);
	de.file_path = p_file_path;
	op.document_edits.append(de);

	return op;
}

ScriptLanguage::CodeActionOperation make_type_declaration_explicit(const GDScriptParser::AssignableNode *p_assignable_node, const String &p_file_path) {
	ScriptLanguage::CodeActionOperation op;

	ScriptLanguage::TextEdit te;
	te.start_line = p_assignable_node->identifier->end_line;
	te.start_column = p_assignable_node->identifier->end_column;
	te.end_line = p_assignable_node->initializer->start_line;
	te.end_column = p_assignable_node->initializer->start_column;
	te.new_text = vformat(": %s = ", p_assignable_node->initializer->type_constraint.to_string());

	op.description = vformat("Make type \"%s\" explicit", p_assignable_node->initializer->type_constraint.to_string());

	ScriptLanguage::DocumentEditOperation de;
	de.edits.append(te);
	de.file_path = p_file_path;
	op.document_edits.append(de);

	return op;
}

ScriptLanguage::CodeActionOperation add_type_specifier_for_identifier(const GDScriptParser::IdentifierNode *p_identifier_node, const String &p_file_path) {
	ScriptLanguage::CodeActionOperation op;
	ScriptLanguage::TextEdit te;

	te.start_line = p_identifier_node->end_line;
	te.start_column = p_identifier_node->end_column;
	te.end_line = p_identifier_node->end_line;
	te.end_column = p_identifier_node->end_column;

	te.new_text = vformat(": %s", p_identifier_node->type_constraint.to_string());
	op.description = vformat("Add type specifier \"%s\"", p_identifier_node->type_constraint.to_string());

	ScriptLanguage::DocumentEditOperation de;
	de.edits.append(te);
	de.file_path = p_file_path;
	op.document_edits.append(de);

	return op;
}

ScriptLanguage::CodeActionOperation add_type_specifier_for_assignable(const GDScriptParser::AssignableNode *p_assignable_node, const String &p_file_path) {
	ScriptLanguage::CodeActionOperation op;
	ScriptLanguage::TextEdit te;

	te.start_line = p_assignable_node->identifier->end_line;
	te.start_column = p_assignable_node->identifier->end_column;
	te.end_line = p_assignable_node->identifier->end_line;
	te.end_column = p_assignable_node->identifier->end_column;

	te.new_text = vformat(": %s", p_assignable_node->initializer->type_constraint.to_string());
	op.description = vformat("Add type specifier \"%s\"", p_assignable_node->initializer->type_constraint.to_string());

	ScriptLanguage::DocumentEditOperation de;
	de.edits.append(te);
	de.file_path = p_file_path;
	op.document_edits.append(de);

	return op;
}

ScriptLanguage::CodeActionOperation add_await(const GDScriptParser::ExpressionNode *p_expression_node, const String &p_file_path) {
	ScriptLanguage::CodeActionOperation op;
	ScriptLanguage::TextEdit te;

	te.start_line = p_expression_node->start_line;
	te.start_column = p_expression_node->start_column;
	te.end_line = p_expression_node->start_line;
	te.end_column = p_expression_node->start_column;
	te.new_text = "await ";

	op.description = "Add \"await\"";

	ScriptLanguage::DocumentEditOperation de;
	de.edits.append(te);
	de.file_path = p_file_path;
	op.document_edits.append(de);

	return op;
}

ScriptLanguage::CodeActionOperation remove_await(const GDScriptParser::AwaitNode *p_await_node, const String &p_file_path) {
	ScriptLanguage::CodeActionOperation op;
	ScriptLanguage::TextEdit te;

	te.start_line = p_await_node->start_line;
	te.start_column = p_await_node->start_column;
	te.end_line = p_await_node->end_line;
	te.end_column = p_await_node->to_await->start_column;
	te.new_text = "";

	op.description = "Remove \"await\"";

	ScriptLanguage::DocumentEditOperation de;
	de.edits.append(te);
	de.file_path = p_file_path;
	op.document_edits.append(de);

	return op;
}

ScriptLanguage::CodeActionOperation call_method_from_type(const GDScriptParser::CallNode *p_call_node, const String &p_caller_type, const String &p_file_path) {
	ScriptLanguage::CodeActionOperation op;
	ScriptLanguage::TextEdit te;

	te.start_line = p_call_node->callee->start_line;
	te.start_column = p_call_node->callee->start_column;
	te.end_line = p_call_node->callee->end_line;
	te.end_column = p_call_node->callee->end_column;
	te.new_text = vformat("%s.%s", p_caller_type, p_call_node->function_name);

	op.description = vformat("Call from type \"%s\"", p_caller_type);

	ScriptLanguage::DocumentEditOperation de;
	de.edits.append(te);
	de.file_path = p_file_path;
	op.document_edits.append(de);

	return op;
}

} // namespace GDScriptCodeActions
