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

#include "editor/debugger/editor_debugger_inspector.h"
#include "editor/debugger/script_editor_debugger.h"
#include "editor/editor_string_names.h"
#include "scene/gui/button.h"
#include "scene/gui/check_box.h"
#include "scene/gui/line_edit.h"

void EditorExpressionEvaluator::on_start() {
	expression_input->set_editable(false);
	evaluate_btn->set_disabled(true);

	if (clear_on_run_checkbox->is_pressed()) {
		inspector->clear_stack_variables();
	}
}

void EditorExpressionEvaluator::set_editor_debugger(ScriptEditorDebugger *p_editor_debugger) {
	editor_debugger = p_editor_debugger;
}

void EditorExpressionEvaluator::add_value(const Array &p_array) {
	inspector->add_stack_variable(p_array, 0);
	inspector->set_v_scroll(0);
	inspector->set_h_scroll(0);
}

void EditorExpressionEvaluator::_line_edit_gui_input(const Ref<InputEvent> &p_event) {
	if (!expression_input->is_editing()) {
		return;
	}

	const Ref<InputEventKey> k = p_event;
	if (k.is_null() || !k->is_pressed()) {
		return;
	}

	if (k->is_action_pressed("ui_up", true)) {
		int size = expression_history.size();
		if (expression_index < size - 1) {
			expression_index++;

			const String &history_text = expression_history[expression_history.size() - expression_index - 1];
			expression_input->set_text(history_text);
			expression_input->set_caret_column(history_text.size());
		}
		accept_event();
	} else if (k->is_action_pressed("ui_down", true)) {
		if (expression_index > 0) {
			expression_index--;

			const String &history_text = expression_history[expression_history.size() - expression_index - 1];
			expression_input->set_text(history_text);
			expression_input->set_caret_column(history_text.size());
		} else if (expression_index == 0) {
			expression_index = -1;
			expression_input->clear();
		}
		accept_event();
	}
}

void EditorExpressionEvaluator::_evaluate() {
	const String &expression = expression_input->get_text();
	if (expression.is_empty()) {
		return;
	}

	if (!editor_debugger->is_session_active()) {
		return;
	}

	expression_history.erase(expression);
	expression_history.push_back(expression);
	expression_index = -1;

	editor_debugger->request_remote_evaluate(expression, editor_debugger->get_stack_script_frame());

	expression_input->clear();
}

void EditorExpressionEvaluator::_clear() {
	inspector->clear_stack_variables();
}

void EditorExpressionEvaluator::_remote_object_selected(ObjectID p_id) {
	editor_debugger->emit_signal(SNAME("remote_objects_requested"), Array{ p_id });
}

void EditorExpressionEvaluator::_on_expression_input_changed(const String &p_expression) {
	evaluate_btn->set_disabled(p_expression.is_empty());
	expression_index = -1;
}

void EditorExpressionEvaluator::_on_debugger_breaked(bool p_breaked, bool p_can_debug) {
	expression_input->set_editable(p_breaked);
	evaluate_btn->set_disabled(!p_breaked);
}

void EditorExpressionEvaluator::_on_debugger_clear_execution(Ref<Script> p_stack_script) {
	expression_input->set_editable(false);
	evaluate_btn->set_disabled(true);
}

void EditorExpressionEvaluator::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			EditorDebuggerNode::get_singleton()->connect("breaked", callable_mp(this, &EditorExpressionEvaluator::_on_debugger_breaked));
			EditorDebuggerNode::get_singleton()->connect("clear_execution", callable_mp(this, &EditorExpressionEvaluator::_on_debugger_clear_execution));
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			expression_input->add_theme_font_override(SceneStringName(font), get_theme_font(SNAME("expression"), EditorStringName(EditorFonts)));
			expression_input->add_theme_font_size_override(SceneStringName(font_size), get_theme_font_size(SNAME("expression_size"), EditorStringName(EditorFonts)));
		} break;
	}
}

EditorExpressionEvaluator::EditorExpressionEvaluator() {
	set_h_size_flags(SIZE_EXPAND_FILL);

	HBoxContainer *hb = memnew(HBoxContainer);
	add_child(hb);

	expression_input = memnew(LineEdit);
	expression_input->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	expression_input->set_placeholder(TTRC("Expression to evaluate (Up/Down: Navigate history)"));
	expression_input->set_accessibility_name(TTRC("Expression to evaluate"));
	expression_input->set_clear_button_enabled(true);
	expression_input->set_keep_editing_on_text_submit(true);
	expression_input->connect(SceneStringName(gui_input), callable_mp(this, &EditorExpressionEvaluator::_line_edit_gui_input));
	expression_input->connect(SceneStringName(text_submitted), callable_mp(this, &EditorExpressionEvaluator::_evaluate).unbind(1));
	expression_input->connect(SceneStringName(text_changed), callable_mp(this, &EditorExpressionEvaluator::_on_expression_input_changed));
	hb->add_child(expression_input);

	clear_on_run_checkbox = memnew(CheckBox);
	clear_on_run_checkbox->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
	clear_on_run_checkbox->set_text(TTRC("Clear on Run"));
	clear_on_run_checkbox->set_pressed(true);
	hb->add_child(clear_on_run_checkbox);

	evaluate_btn = memnew(Button);
	evaluate_btn->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
	evaluate_btn->set_text(TTRC("Evaluate"));
	evaluate_btn->connect(SceneStringName(pressed), callable_mp(this, &EditorExpressionEvaluator::_evaluate));
	hb->add_child(evaluate_btn);

	clear_btn = memnew(Button);
	clear_btn->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
	clear_btn->set_text(TTRC("Clear"));
	clear_btn->connect(SceneStringName(pressed), callable_mp(this, &EditorExpressionEvaluator::_clear));
	hb->add_child(clear_btn);

	inspector = memnew(EditorDebuggerInspector);
	inspector->set_v_size_flags(SIZE_EXPAND_FILL);
	inspector->set_property_name_style(EditorPropertyNameProcessor::STYLE_RAW);
	inspector->set_read_only(true);
	inspector->connect("object_selected", callable_mp(this, &EditorExpressionEvaluator::_remote_object_selected));
	inspector->set_use_filter(true);
	add_child(inspector);

	expression_input->set_editable(false);
	evaluate_btn->set_disabled(true);
}
