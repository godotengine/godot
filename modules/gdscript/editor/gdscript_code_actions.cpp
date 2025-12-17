/**************************************************************************/
/*  gdscript_code_actions.cpp                                             */
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

#include "modules/gdscript/editor/gdscript_code_actions.h"

#include "core/object/editor_language.h"
#include "editor/settings/editor_settings.h"

EditorLanguage::CodeActionOperation GDScriptCodeActions::add_script_annotation(const String &p_annotation, const String &p_file_path) {
	EditorLanguage::CodeActionOperation op;
	ScriptLanguage::TextEdit te;

	te.start_line = 1;
	te.start_column = 1;
	te.end_line = 1;
	te.end_column = 1;

	te.new_text = vformat("%s\n", p_annotation);
	op.description = vformat(TTR("Add \"%s\""), p_annotation);

	EditorLanguage::DocumentEditOperation de;
	de.edits.append(te);
	de.file_path = p_file_path;
	op.document_edits.append(de);

	return op;
}

EditorLanguage::CodeActionOperation GDScriptCodeActions::add_member_annotation(const String &p_annotation, const GDScriptParser::Node *p_node, const String &p_file_path) {
	EditorLanguage::CodeActionOperation op;
	ScriptLanguage::TextEdit te;

	te.start_line = p_node->start_line;
	te.start_column = p_node->start_column;
	te.end_line = p_node->start_line;
	te.end_column = p_node->start_column;

	te.new_text = vformat("%s ", p_annotation);
	op.description = vformat(TTR("Add \"%s\""), p_annotation);

	EditorLanguage::DocumentEditOperation de;
	de.edits.append(te);
	de.file_path = p_file_path;
	op.document_edits.append(de);

	return op;
}

EditorLanguage::CodeActionOperation GDScriptCodeActions::remove_underscore_prefix_from_identifier(const GDScriptParser::IdentifierNode *p_identifier_node, const String &p_type_description, const String &p_file_path) {
	EditorLanguage::CodeActionOperation op;
	ScriptLanguage::TextEdit te;

	te.start_line = p_identifier_node->start_line;
	te.start_column = p_identifier_node->start_column;
	te.end_line = p_identifier_node->start_line;
	te.end_column = p_identifier_node->start_column + 1;

	te.new_text = "";
	op.description = vformat(TTR("Remove underscore from %s"), p_type_description);

	EditorLanguage::DocumentEditOperation de;
	de.edits.append(te);
	de.file_path = p_file_path;
	op.document_edits.append(de);

	return op;
}

EditorLanguage::CodeActionOperation GDScriptCodeActions::remove_statement(const GDScriptParser::Node *p_node, const String &p_type_description, const String &p_file_path) {
	EditorLanguage::CodeActionOperation op;
	ScriptLanguage::TextEdit te;

	te.start_line = p_node->start_line;
	te.start_column = 1;
	te.end_line = p_node->end_line + 1;
	te.end_column = 1;

	te.new_text = "";
	op.description = vformat(TTR("Remove %s"), p_type_description);

	EditorLanguage::DocumentEditOperation de;
	de.edits.append(te);
	de.file_path = p_file_path;
	op.document_edits.append(de);

	return op;
}

EditorLanguage::CodeActionOperation GDScriptCodeActions::add_underscore_prefix_to_identifier(const GDScriptParser::IdentifierNode *p_identifier_node, const String &p_type_description, const String &p_file_path) {
	EditorLanguage::CodeActionOperation op;
	ScriptLanguage::TextEdit te;

	te.start_line = p_identifier_node->start_line;
	te.start_column = p_identifier_node->start_column;
	te.end_line = p_identifier_node->start_line;
	te.end_column = p_identifier_node->start_column;

	te.new_text = "_";
	op.description = vformat(TTR("Add underscore to %s"), p_type_description);

	EditorLanguage::DocumentEditOperation de;
	de.edits.append(te);
	de.file_path = p_file_path;
	op.document_edits.append(de);

	return op;
}

EditorLanguage::CodeActionOperation GDScriptCodeActions::make_type_declaration_explicit(const GDScriptParser::AssignableNode *p_assignable_node, const String &p_file_path) {
	EditorLanguage::CodeActionOperation op;

	ScriptLanguage::TextEdit te;
	te.start_line = p_assignable_node->identifier->end_line;
	te.start_column = p_assignable_node->identifier->end_column;
	te.end_line = p_assignable_node->initializer->start_line;
	te.end_column = p_assignable_node->initializer->start_column;
	te.new_text = vformat(": %s = ", p_assignable_node->initializer->type_constraint.to_string());

	op.description = vformat(TTR("Make type \"%s\" explicit"), p_assignable_node->initializer->type_constraint.to_string());

	EditorLanguage::DocumentEditOperation de;
	de.edits.append(te);
	de.file_path = p_file_path;
	op.document_edits.append(de);

	return op;
}

EditorLanguage::CodeActionOperation GDScriptCodeActions::add_type_specifier_for_identifier(const GDScriptParser::IdentifierNode *p_identifier_node, const String &p_file_path) {
	EditorLanguage::CodeActionOperation op;
	ScriptLanguage::TextEdit te;

	te.start_line = p_identifier_node->end_line;
	te.start_column = p_identifier_node->end_column;
	te.end_line = p_identifier_node->end_line;
	te.end_column = p_identifier_node->end_column;

	te.new_text = vformat(": %s", p_identifier_node->type_constraint.to_string());
	op.description = vformat(TTR("Add type specifier \"%s\""), p_identifier_node->type_constraint.to_string());

	EditorLanguage::DocumentEditOperation de;
	de.edits.append(te);
	de.file_path = p_file_path;
	op.document_edits.append(de);

	return op;
}

EditorLanguage::CodeActionOperation GDScriptCodeActions::add_type_specifier_for_assignable(const GDScriptParser::AssignableNode *p_assignable_node, const String &p_file_path) {
	EditorLanguage::CodeActionOperation op;
	ScriptLanguage::TextEdit te;

	te.start_line = p_assignable_node->identifier->end_line;
	te.start_column = p_assignable_node->identifier->end_column;
	te.end_line = p_assignable_node->identifier->end_line;
	te.end_column = p_assignable_node->identifier->end_column;

	te.new_text = vformat(": %s", p_assignable_node->initializer->type_constraint.to_string());
	op.description = vformat(TTR("Add type specifier \"%s\""), p_assignable_node->initializer->type_constraint.to_string());

	EditorLanguage::DocumentEditOperation de;
	de.edits.append(te);
	de.file_path = p_file_path;
	op.document_edits.append(de);

	return op;
}

EditorLanguage::CodeActionOperation GDScriptCodeActions::add_await(const GDScriptParser::ExpressionNode *p_expression_node, const String &p_file_path) {
	EditorLanguage::CodeActionOperation op;
	ScriptLanguage::TextEdit te;

	te.start_line = p_expression_node->start_line;
	te.start_column = p_expression_node->start_column;
	te.end_line = p_expression_node->start_line;
	te.end_column = p_expression_node->start_column;
	te.new_text = "await ";

	op.description = TTR("Add \"await\"");

	EditorLanguage::DocumentEditOperation de;
	de.edits.append(te);
	de.file_path = p_file_path;
	op.document_edits.append(de);

	return op;
}

EditorLanguage::CodeActionOperation GDScriptCodeActions::remove_await(const GDScriptParser::AwaitNode *p_await_node, const String &p_file_path) {
	EditorLanguage::CodeActionOperation op;
	ScriptLanguage::TextEdit te;

	te.start_line = p_await_node->start_line;
	te.start_column = p_await_node->start_column;
	te.end_line = p_await_node->end_line;
	te.end_column = p_await_node->to_await->start_column;
	te.new_text = "";

	op.description = TTR("Remove \"await\"");

	EditorLanguage::DocumentEditOperation de;
	de.edits.append(te);
	de.file_path = p_file_path;
	op.document_edits.append(de);

	return op;
}

EditorLanguage::CodeActionOperation GDScriptCodeActions::call_method_from_type(const GDScriptParser::CallNode *p_call_node, const String &p_caller_type, const String &p_file_path) {
	EditorLanguage::CodeActionOperation op;
	ScriptLanguage::TextEdit te;

	te.start_line = p_call_node->callee->start_line;
	te.start_column = p_call_node->callee->start_column;
	te.end_line = p_call_node->callee->end_line;
	te.end_column = p_call_node->callee->end_column;
	te.new_text = vformat("%s.%s", p_caller_type, p_call_node->function_name);

	op.description = vformat(TTR("Call from type \"%s\""), p_caller_type);

	EditorLanguage::DocumentEditOperation de;
	de.edits.append(te);
	de.file_path = p_file_path;
	op.document_edits.append(de);

	return op;
}

EditorLanguage::CodeActionOperation GDScriptCodeActions::ignore_warning(GDScriptWarning::Code p_code, const String &p_source, const String &p_script_path, int p_start_line) {
	// Create the ignore code action.
	EditorLanguage::CodeActionOperation op;
	ScriptLanguage::TextEdit te;

	String warning_name = GDScriptWarning::get_name_from_code(p_code);

	// Find the existing leading whitespace/indentation used for this line of
	// code, and copy it for after the newline.
	PackedStringArray lines = p_source.split("\n");
	String line = lines[p_start_line - 1];
	String leading_whitespace;
	for (int i = 0; i < line.length() - 1; i++) {
		if (line[i] == '\t' || line[i] == ' ') {
			leading_whitespace += line[i];
		} else {
			break;
		}
	}

	te.start_line = p_start_line;
	te.start_column = 1;
	te.end_line = p_start_line;
	te.end_column = 1;

	const String code = warning_name.to_lower();

	String quote_style = "\"";
	if (EditorSettings::get_singleton() && _EDITOR_GET("text_editor/completion/use_single_quotes")) {
		quote_style = "'";
	}

	te.new_text = vformat("%s@warning_ignore(%s)\n", leading_whitespace, code.quote(quote_style));

	// Determine if there's an existing @warning_ignore here; if so, rather than
	// inserting a new line and a new @warning_ignore, just add this warning to its list.
	// (Logic mostly copied from ScriptTextEditor::_warning_clicked.)
	if (p_start_line - 2 >= 0) {
		String line_before = lines[p_start_line - 2];
		if (line_before.strip_edges().begins_with("@warning_ignore(")) {
			const int closing_bracket_idx = line_before.find_char(')');
			const String text_to_insert = ", " + code.quote(quote_style);

			te.new_text = text_to_insert;
			te.start_line = p_start_line - 1;
			te.start_column = closing_bracket_idx + 1;
			te.end_line = p_start_line - 1;
			te.end_column = closing_bracket_idx + 1;
		}
	}

	op.description = vformat(TTR("Ignore \"%s\""), warning_name);

	EditorLanguage::DocumentEditOperation de;
	de.file_path = p_script_path;
	de.edits.append(te);

	op.document_edits.append(de);
	return op;
}

EditorLanguage::CodeActionGroup GDScriptCodeActions::make_code_action_ranges_zero_based(const EditorLanguage::CodeActionGroup &p_group, const Vector<String> &p_lines) {
	EditorLanguage::CodeActionGroup new_group;
	new_group.title = p_group.title;
	new_group.actions = p_group.actions.duplicate();

	for (EditorLanguage::CodeActionOperation &code_action_op : new_group.actions) {
		for (EditorLanguage::DocumentEditOperation &doc_edit_op : code_action_op.document_edits) {
			for (ScriptLanguage::TextEdit &text_edit_op : doc_edit_op.edits) {
				text_edit_op.start_line -= 1;
				text_edit_op.start_column -= 1;
				text_edit_op.end_line -= 1;
				text_edit_op.end_column -= 1;
			}
		}
	}
	return new_group;
}

Vector<EditorLanguage::CodeActionOperation> GDScriptCodeActions::get_code_actions_for_warning(const GDScriptParser::Node *p_source, GDScriptWarning::Code p_code, const String &p_script_path) {
	Vector<EditorLanguage::CodeActionOperation> actions;

	if (!p_source) {
		return actions;
	}

	switch (p_code) {
		case GDScriptWarning::MISSING_TOOL: {
			actions.append(add_script_annotation("@tool", p_script_path));
			break;
		}
		case GDScriptWarning::GET_NODE_DEFAULT_WITHOUT_ONREADY: {
			actions.append(add_member_annotation("@onready", p_source, p_script_path));
			break;
		}
		case GDScriptWarning::UNUSED_PRIVATE_CLASS_VARIABLE: {
			if (p_source->type != GDScriptParser::Node::IDENTIFIER) {
				break;
			}
			const GDScriptParser::IdentifierNode *identifier = static_cast<const GDScriptParser::IdentifierNode *>(p_source);
			actions.append(remove_underscore_prefix_from_identifier(identifier, "class variable name", p_script_path));
			actions.append(remove_statement(identifier, "class variable declaration", p_script_path));

			break;
		}
		case GDScriptWarning::UNUSED_PARAMETER: {
			if (p_source->type != GDScriptParser::Node::IDENTIFIER) {
				break;
			}
			const GDScriptParser::IdentifierNode *identifier = static_cast<const GDScriptParser::IdentifierNode *>(p_source);
			actions.append(add_underscore_prefix_to_identifier(identifier, "parameter name", p_script_path));
			break;
		}
		case GDScriptWarning::UNUSED_VARIABLE: {
			const GDScriptParser::IdentifierNode *identifier = nullptr;

			if (p_source->type == GDScriptParser::Node::VARIABLE) {
				identifier = static_cast<const GDScriptParser::VariableNode *>(p_source)->identifier;
			} else if (p_source->type == GDScriptParser::Node::IDENTIFIER) {
				identifier = static_cast<const GDScriptParser::IdentifierNode *>(p_source);
			} else {
				break;
			}

			actions.append(add_underscore_prefix_to_identifier(identifier, "variable name", p_script_path));
			actions.append(remove_statement(identifier, "variable declaration", p_script_path));

			break;
		}
		case GDScriptWarning::UNUSED_LOCAL_CONSTANT: {
			if (p_source->type != GDScriptParser::Node::CONSTANT) {
				break;
			}
			const GDScriptParser::IdentifierNode *identifier = static_cast<const GDScriptParser::ConstantNode *>(p_source)->identifier;

			actions.append(add_underscore_prefix_to_identifier(identifier, "constant name", p_script_path));
			actions.append(remove_statement(identifier, "constant declaration", p_script_path));

			break;
		}
		case GDScriptWarning::INFERRED_DECLARATION: {
			if (p_source->type == GDScriptParser::Node::VARIABLE || p_source->type == GDScriptParser::Node::CONSTANT) {
				const GDScriptParser::AssignableNode *assignable = static_cast<const GDScriptParser::AssignableNode *>(p_source);
				if (assignable->initializer) {
					actions.append(make_type_declaration_explicit(assignable, p_script_path));
				}
			} else if (p_source->type == GDScriptParser::Node::IDENTIFIER) {
				const GDScriptParser::IdentifierNode *identifier = static_cast<const GDScriptParser::IdentifierNode *>(p_source);
				actions.append(add_type_specifier_for_identifier(identifier, p_script_path));
			}
			break;
		}
		case GDScriptWarning::UNTYPED_DECLARATION: {
			if (p_source->type != GDScriptParser::Node::VARIABLE) {
				break;
			}
			const GDScriptParser::AssignableNode *assignable = static_cast<const GDScriptParser::AssignableNode *>(p_source);

			if (assignable->identifier && assignable->initializer) {
				GDScriptParser::DataType::Kind kind = assignable->initializer->type_constraint.kind;
				if (kind != GDScriptParser::DataType::Kind::RESOLVING && kind != GDScriptParser::DataType::Kind::UNRESOLVED) {
					actions.append(add_type_specifier_for_assignable(assignable, p_script_path));
				}
			}
			break;
		}
		case GDScriptWarning::ASSERT_ALWAYS_TRUE: {
			actions.append(remove_statement(p_source, "assert statement", p_script_path));
			break;
		}
		case GDScriptWarning::REDUNDANT_AWAIT: {
			if (p_source->type != GDScriptParser::Node::AWAIT) {
				break;
			}
			const GDScriptParser::AwaitNode *await = static_cast<const GDScriptParser::AwaitNode *>(p_source);
			actions.append(remove_await(await, p_script_path));
			break;
		}
		case GDScriptWarning::MISSING_AWAIT: {
			if (p_source->type != GDScriptParser::Node::CALL) {
				break;
			}
			const GDScriptParser::CallNode *call = static_cast<const GDScriptParser::CallNode *>(p_source);
			actions.append(add_await(call->callee, p_script_path));
			break;
		}
		default:
			break;
	}
	return actions;
}
