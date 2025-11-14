/**************************************************************************/
/*  gdscript_reconstructor.h                                              */
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

#ifndef GDSCRIPT_RECONSTRUCTOR_H
#define GDSCRIPT_RECONSTRUCTOR_H

#include "gdscript_parser.h"

class GDScriptReconstructor {
	struct Output {
		int indent = 0;
		String *text = nullptr;

		Output(int p_indent, String *p_text) {
			indent = p_indent;
			text = p_text;
		}
	};

	struct Data {
		mutable bool output_tab_required = true;
		mutable bool var_declaration = false;

		// We can optionally emphasize a node in the output, in order to highlight a line to the user.
		const GDScriptParser::Node *emphasis_node = nullptr;
	} data;

	bool debug_output_node(const GDScriptParser::Node *p_node, Output out) const;
	bool debug_output_operator(const GDScriptParser::OperatorNode *p_node, Output out) const;
	bool debug_output_control_flow(const GDScriptParser::ControlFlowNode *p_node, Output out) const;

	bool output_node(const GDScriptParser::Node *p_node, Output out, bool p_apply_parent_brackets = false) const;
	bool output_class(const GDScriptParser::ClassNode *p_node, Output out) const;
	bool output_function(const GDScriptParser::FunctionNode *p_node, Output out) const;
	bool output_block(const GDScriptParser::BlockNode *p_node, Output out) const;
	bool output_control_flow(const GDScriptParser::ControlFlowNode *p_node, Output out) const;
	bool output_operator(const GDScriptParser::OperatorNode *p_node, Output out, bool p_apply_parent_brackets = false) const;
	void output_indent(const Output &p_out, bool p_force = false) const;

	static String _variant_data_to_string(const Variant &p_val, bool p_style = true);
	static bool helper_data_type_to_string(const GDScriptParser::DataType &p_dt, String &r_text) {
		if (p_dt.has_type) {
			r_text += draw_operator(" : ") + draw_constant(p_dt.to_string());
			return true;
		}
		return false;
	}

	void apply_emphasis(const GDScriptParser::Node *p_node, String &r_text) const;

public:
	static const char *OPStrings[];

	bool output_branch(const GDScriptParser::Node *p_branch, String &r_text, const GDScriptParser::Node *p_emphasis_node = nullptr);
	bool output(GDScriptParser &r_parser, String &r_text);
	bool debug_output(GDScriptParser &r_parser, String &r_text);

	static String draw_keyword(String p_string) { return TerminalColor::draw(TerminalColor::CYAN, p_string); }
	static String draw_operator(String p_string) { return TerminalColor::draw(TerminalColor::GREEN, p_string); }
	static String draw_identifier(String p_string) { return TerminalColor::draw(TerminalColor::WHITE, p_string); }
	static String draw_constant(String p_string) { return TerminalColor::draw(TerminalColor::YELLOW, p_string); }

	static String draw_bold(String p_string) { return TerminalColor::draw(TerminalColor::BOLD, p_string); }
	static String draw_node_info(String p_string) { return TerminalColor::draw(TerminalColor::YELLOW, p_string); }
	static String draw_location(String p_string) { return TerminalColor::draw(TerminalColor::WHITE, p_string); }
	static String draw_highlight(String p_string) { return TerminalColor::draw(TerminalColor::CYAN, p_string); }

	static String draw_attention(String p_string) { return TerminalColor::draw_combined(TerminalColor::BOLD, TerminalColor::UNDERLINE, p_string); }

	static String variant_to_string(const Variant &p_val, bool p_style = true);
};

#endif // GDSCRIPT_RECONSTRUCTOR_H
