/**************************************************************************/
/*  expression_evaluator_editor_plugin.cpp                                */
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

#include "expression_evaluator_editor_plugin.h"

#include "editor/debugger/editor_debugger_inspector.h"
#include "editor/debugger/script_editor_debugger.h"
#include "scene/gui/button.h"
#include "scene/gui/check_box.h"

void EditorExpressionEvaluator::set_can_evaluate(bool p_enabled) {
	expression_input->set_editable(p_enabled);
	evaluate_btn->set_disabled(!p_enabled);

	if (p_enabled) {
		expression_input->set_placeholder(TTR("Expression to evaluate"));
	} else {
		expression_input->set_placeholder(TTR("Expressions can only be evaluated when a running project is paused in the debugger."));
	}
}

void EditorExpressionEvaluator::on_start() {
	if (clear_on_run_checkbox->is_pressed()) {
		inspector->clear_stack_variables();
	}
}

void EditorExpressionEvaluator::add_value(const Array &p_array) {
	inspector->add_stack_variable(p_array, 0);
	inspector->set_v_scroll(0);
	inspector->set_h_scroll(0);
}

void EditorExpressionEvaluator::_evaluate() {
	const String &expression = expression_input->get_text();
	if (expression.is_empty()) {
		return;
	}

	if (!session->is_active()) {
		return;
	}

	Array expr_data;
	expr_data.push_back(expression);
	// expr_data.push_back(editor_debugger->get_stack_script_frame());
	expr_data.push_back(0);
	session->send_message("evaluate", expr_data);

	expression_input->clear();
}

void EditorExpressionEvaluator::_clear() {
	inspector->clear_stack_variables();
}

void EditorExpressionEvaluator::_remote_object_selected(ObjectID p_id) {
	// editor_debugger->emit_signal(SNAME("remote_object_requested"), p_id);
}

void EditorExpressionEvaluator::_on_expression_input_changed(const String &p_expression) {
	evaluate_btn->set_disabled(p_expression.is_empty());
}

EditorExpressionEvaluator::EditorExpressionEvaluator(const Ref<EditorDebuggerSession> &p_session) {
	set_name(TTR("Evaluate"));
	session = p_session;

	HBoxContainer *hb = memnew(HBoxContainer);
	add_child(hb);

	expression_input = memnew(LineEdit);
	expression_input->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	expression_input->set_clear_button_enabled(true);
	expression_input->connect("text_submitted", callable_mp(this, &EditorExpressionEvaluator::_evaluate).unbind(1));
	expression_input->connect("text_changed", callable_mp(this, &EditorExpressionEvaluator::_on_expression_input_changed));
	hb->add_child(expression_input);

	clear_on_run_checkbox = memnew(CheckBox);
	clear_on_run_checkbox->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
	clear_on_run_checkbox->set_text(TTR("Clear on Run"));
	clear_on_run_checkbox->set_pressed(true);
	hb->add_child(clear_on_run_checkbox);

	evaluate_btn = memnew(Button);
	evaluate_btn->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
	evaluate_btn->set_text(TTR("Evaluate"));
	evaluate_btn->connect(SceneStringName(pressed), callable_mp(this, &EditorExpressionEvaluator::_evaluate));
	hb->add_child(evaluate_btn);

	clear_btn = memnew(Button);
	clear_btn->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
	clear_btn->set_text(TTR("Clear"));
	clear_btn->connect(SceneStringName(pressed), callable_mp(this, &EditorExpressionEvaluator::_clear));
	hb->add_child(clear_btn);

	inspector = memnew(EditorDebuggerInspector);
	inspector->set_v_size_flags(SIZE_EXPAND_FILL);
	inspector->set_property_name_style(EditorPropertyNameProcessor::STYLE_RAW);
	inspector->set_read_only(true);
	inspector->connect("object_selected", callable_mp(this, &EditorExpressionEvaluator::_remote_object_selected));
	inspector->set_use_filter(true);
	add_child(inspector);

	set_can_evaluate(false);
}

void ExpressionEvaluatorDebugger::setup_session(int p_idx) {
	Ref<EditorDebuggerSession> session = get_session(p_idx);

	EditorExpressionEvaluator *evaluator = memnew(EditorExpressionEvaluator(session));
	session->add_session_tab(evaluator);
	session->connect("started", callable_mp(evaluator, &EditorExpressionEvaluator::on_start));
	session->connect("breaked", callable_mp(evaluator, &EditorExpressionEvaluator::set_can_evaluate));
	session->connect("stopped", callable_mp(evaluator, &EditorExpressionEvaluator::set_can_evaluate).bind(false));
	session->connect("continued", callable_mp(evaluator, &EditorExpressionEvaluator::set_can_evaluate).bind(false));

	evaluators[p_idx] = evaluator;
}

bool ExpressionEvaluatorDebugger::capture(const String &p_message, const Array &p_data, int p_session) {
	EditorExpressionEvaluator *evaluator = evaluators[p_session];
	evaluator->add_value(p_data);
	return true;
}

bool ExpressionEvaluatorDebugger::has_capture(const String &p_capture) const {
	return p_capture == "evaluation_return";
}

void ExpressionEvaluatorEditorPlugin::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			add_debugger_plugin(plugin);
		} break;
		case NOTIFICATION_EXIT_TREE: {
			remove_debugger_plugin(plugin);
		}
	}
}

ExpressionEvaluatorEditorPlugin::ExpressionEvaluatorEditorPlugin() {
	plugin.instantiate();
}
