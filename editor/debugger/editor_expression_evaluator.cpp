/**************************************************************************/
/*  editor_expression_evaluator.cpp                                       */
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

#include "editor_expression_evaluator.h"

#include "core/debugger/debugger_marshalls.h"
#include "core/io/marshalls.h"
#include "core/object/callable_method_pointer.h"
#include "core/os/os.h"
#include "editor/debugger/script_editor_debugger.h"
#include "editor/editor_log.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"

void EditorExpressionEvaluator::evaluate() {
	String expression = expression_input->get_text();

	if (expression.is_empty()) {
		return;
	}

	if (editor_debugger->is_session_active()) {
		EditorNode::get_log()->add_message("Evaluate Expression: " + expression, EditorLog::MSG_TYPE_EDITOR);

		Array expr_data;
		expr_data.push_back(expression);
		expr_data.push_back(editor_debugger->get_stack_script_frame());

		Array expr_msg;
		expr_msg.push_back("evaluate");
		expr_msg.push_back(expr_data);
		editor_debugger->get_peer()->put_message(expr_msg);
	}
}

void EditorExpressionEvaluator::clear() {
	inspector->clear_stack_variables();
}

void EditorExpressionEvaluator::set_expression_evaluator(ScriptEditorDebugger *p_editor_debugger) {
	editor_debugger = p_editor_debugger;
}

void EditorExpressionEvaluator::add_value(const Array &p_array) {
	inspector->add_stack_variable(p_array);
	inspector->set_h_scroll(0);
	inspector->set_v_scroll((int)(inspector->get_rect().get_size().y));
}

void EditorExpressionEvaluator::_remote_object_selected(ObjectID p_id) {
	editor_debugger->emit_signal(SNAME("remote_object_requested"), p_id);
}

void EditorExpressionEvaluator::_on_expression_input_submitted(String p_expression) {
	evaluate();
}

EditorExpressionEvaluator::EditorExpressionEvaluator() {
	set_h_size_flags(SIZE_EXPAND_FILL);

	HBoxContainer *hb = memnew(HBoxContainer);
	add_child(hb);

	expression_input = memnew(LineEdit);
	expression_input->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	expression_input->set_placeholder(TTR("Expression to evaluate"));
	expression_input->set_clear_button_enabled(true);
	expression_input->connect("text_submitted", callable_mp(this, &EditorExpressionEvaluator::_on_expression_input_submitted));
	hb->add_child(expression_input);

	evaluate_btn = memnew(Button);
	evaluate_btn->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
	evaluate_btn->set_text(TTR("Evaluate"));
	evaluate_btn->connect("pressed", callable_mp(this, &EditorExpressionEvaluator::evaluate));
	hb->add_child(evaluate_btn);

	clear_btn = memnew(Button);
	clear_btn->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
	clear_btn->set_text(TTR("Clear"));
	clear_btn->connect("pressed", callable_mp(this, &EditorExpressionEvaluator::clear));
	hb->add_child(clear_btn);

	inspector = memnew(EditorDebuggerInspector);
	inspector->set_h_size_flags(SIZE_EXPAND_FILL);
	inspector->set_v_size_flags(SIZE_EXPAND_FILL);
	inspector->set_property_name_style(EditorPropertyNameProcessor::STYLE_RAW);
	inspector->set_read_only(true);
	inspector->connect("object_selected", callable_mp(this, &EditorExpressionEvaluator::_remote_object_selected));
	inspector->set_use_filter(true);
	add_child(inspector);
}
