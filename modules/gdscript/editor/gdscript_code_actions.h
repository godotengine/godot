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
extern EditorLanguage::CodeActionOperation add_script_annotation(const String &p_annotation, const String &p_file_path);
extern EditorLanguage::CodeActionOperation add_member_annotation(const String &p_annotation, const GDScriptParser::Node *p_node, const String &p_file_path);
extern EditorLanguage::CodeActionOperation remove_underscore_prefix_from_identifier(const GDScriptParser::IdentifierNode *p_identifier_node, const String &p_type_description, const String &p_file_path);
extern EditorLanguage::CodeActionOperation remove_statement(const GDScriptParser::Node *p_node, const String &p_type_description, const String &p_file_path);
extern EditorLanguage::CodeActionOperation add_underscore_prefix_to_identifier(const GDScriptParser::IdentifierNode *p_identifier_node, const String &p_type_description, const String &p_file_path);
extern EditorLanguage::CodeActionOperation make_type_declaration_explicit(const GDScriptParser::AssignableNode *p_assignable_node, const String &p_file_path);
extern EditorLanguage::CodeActionOperation add_type_specifier_for_identifier(const GDScriptParser::IdentifierNode *p_identifier_node, const String &p_file_path);
extern EditorLanguage::CodeActionOperation add_type_specifier_for_assignable(const GDScriptParser::AssignableNode *p_assignable_node, const String &p_file_path);
extern EditorLanguage::CodeActionOperation add_await(const GDScriptParser::ExpressionNode *p_expression_node, const String &p_file_path);
extern EditorLanguage::CodeActionOperation remove_await(const GDScriptParser::AwaitNode *p_await_node, const String &p_file_path);
extern EditorLanguage::CodeActionOperation call_method_from_type(const GDScriptParser::CallNode *p_call_node, const String &p_caller_type, const String &p_file_path);
extern EditorLanguage::CodeActionOperation ignore_warning(GDScriptWarning::Code p_code, const String &p_source, const String &p_script_path, int p_start_line);

extern EditorLanguage::CodeActionGroup make_code_action_ranges_zero_based(const EditorLanguage::CodeActionGroup &p_group, const Vector<String> &p_lines);

extern Vector<EditorLanguage::CodeActionOperation> get_code_actions_for_warning(const GDScriptParser::Node *p_source, GDScriptWarning::Code p_code, const String &p_script_path);
} // namespace GDScriptCodeActions
