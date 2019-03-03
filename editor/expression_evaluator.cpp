/*************************************************************************/
/*  expression_evaluator.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "expression_evaluator.h"

#include "core/math/expression.h"
#include "core/os/input.h"
#include "core/os/keyboard.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "scene/gui/button.h"
#include "scene/gui/control.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/rich_text_label.h"

class HistoryLineEdit : public Control {

	GDCLASS(HistoryLineEdit, Control);

	struct History {
		Vector<String> cache;
		int index;
	};

	History history;
	LineEdit *line;

	_ALWAYS_INLINE_ void _text_entered(const String &p_text) {
		emit_signal("text_entered", p_text);
	}

protected:
	static void _bind_methods();
	void _input(const Ref<InputEvent> &p_event);

public:
	_ALWAYS_INLINE_ String get_text() const {
		return line->get_text();
	}

	_ALWAYS_INLINE_ void set_text(const String &p_text) {
		line->set_text(p_text);
	}

	_ALWAYS_INLINE_ void push_history(const String &p_text) {
		const int sz = history.cache.size();

		if (sz < 2 || history.cache[sz - 2] != p_text) {
			history.cache.set(sz - 1, p_text);
			history.cache.push_back("");
		}

		if (p_text != history.cache[history.index]) {
			history.index = history.cache.size() - 1;
		} else if (history.index < history.cache.size() - 1) {
			++history.index;
		}
	}

	_ALWAYS_INLINE_ void clear_history() {
		history.cache.clear();
		history.cache.push_back("");
		history.index = 0;
	}

	HistoryLineEdit();
};

void HistoryLineEdit::_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventKey> k = p_event;

	if (k.is_valid() && k->is_pressed() && line->has_focus()) {
		if (k->get_scancode() == KEY_UP) {
			if (history.index > 0) {
				--history.index;
			}
			set_text(history.cache[history.index]);
			line->set_cursor_position(get_text().size() - 1);
			accept_event();
		} else if (k->get_scancode() == KEY_DOWN) {
			if (history.index < history.cache.size() - 1) {
				++history.index;
			}
			set_text(history.cache[history.index]);
			line->set_cursor_position(get_text().size() - 1);
			accept_event();
		}
	}
}

void HistoryLineEdit::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_input", "event"), &HistoryLineEdit::_input);
	ClassDB::bind_method(D_METHOD("_text_entered", "text"), &HistoryLineEdit::_text_entered);

	ADD_SIGNAL(MethodInfo("text_entered", PropertyInfo("text")));
}

HistoryLineEdit::HistoryLineEdit() {
	history.cache.push_back("");
	history.index = 0;
	set_process_input(true);

	Ref<Font> mono_font = EditorNode::get_singleton()->get_gui_base()->get_font("source", "EditorFonts");

	set_h_size_flags(SIZE_EXPAND_FILL);

	line = memnew(LineEdit);
	add_child(line);
	line->set_anchors_and_margins_preset(PRESET_WIDE);
	line->set_context_menu_enabled(false);
	line->connect("text_entered", this, "_text_entered");
	line->add_font_override("font", mono_font);
}

/////////////////////////////////////////////////////////////////////

enum MenuItem {
	MENU_CLEAR_HISTORY,
	MENU_CLEAR_LOG
};

void ExpressionEvaluator::_expression_entered(const String &p_text) {
	_print_expression(p_text);
}

void ExpressionEvaluator::_print_pressed() {
	_print_expression(expression_line->get_text());
}

void ExpressionEvaluator::_watch_pressed() {
	emit_signal("add_watch", expression_line->get_text());
	expression_line->set_text("");
}

void ExpressionEvaluator::_log_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid() && mb->is_pressed()) {
		if (mb->get_button_index() == BUTTON_RIGHT) {
			Point2 pos = Input::get_singleton()->get_mouse_position();

			menu->set_global_position(pos);
			menu->set_scale(get_global_transform().get_scale());
			menu->popup();
		}
	}
}

void ExpressionEvaluator::_menu_item_pressed(int p_id) {
	const MenuItem item = (MenuItem)p_id;

	switch (item) {
		case MENU_CLEAR_HISTORY: {
			expression_line->clear_history();
		} break;

		case MENU_CLEAR_LOG: {
			log_label->clear();
		} break;
	}
}

void ExpressionEvaluator::_print_expression(const String &p_text) {
	if (p_text.empty())
		return;

	expression_line->push_history(p_text);
	expression_line->set_text("");

	log_label->add_text(p_text + ":\t");

	if (expression->parse(p_text) != Error::OK) {
		set_result(expression->get_error_text(), true);
		return;
	}

	Object obj;
	Variant result = expression->execute(Array(), &obj, false);

	if (!expression->has_execute_failed()) {
		set_result(result.get_construct_string(), false);
		return;
	}

	emit_signal("evaluate", p_text, expression->get_error_text());
}

void ExpressionEvaluator::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_expression_entered", "text"), &ExpressionEvaluator::_expression_entered);
	ClassDB::bind_method(D_METHOD("_print_pressed"), &ExpressionEvaluator::_print_pressed);
	ClassDB::bind_method(D_METHOD("_watch_pressed"), &ExpressionEvaluator::_watch_pressed);
	ClassDB::bind_method(D_METHOD("_log_input"), &ExpressionEvaluator::_log_input);
	ClassDB::bind_method(D_METHOD("_menu_item_pressed"), &ExpressionEvaluator::_menu_item_pressed);

	ADD_SIGNAL(MethodInfo("evaluate", PropertyInfo("expression"), PropertyInfo("error_text")));
	ADD_SIGNAL(MethodInfo("add_watch", PropertyInfo("expression")));
}

void ExpressionEvaluator::set_result(const String &p_result, bool p_error) {
	if (p_error) {
		log_label->push_color(get_color("error_color", "Editor"));
	}

	log_label->add_text(p_result);
	log_label->add_newline();

	if (p_error) {
		log_label->pop();
	}
}

ExpressionEvaluator::ExpressionEvaluator() {
	expression = memnew(Expression);

	Ref<Font> mono_font = EditorNode::get_singleton()->get_gui_base()->get_font("source", "EditorFonts");

	set_name("Expressions");
	set_focus_mode(FOCUS_NONE);

	log_label = memnew(RichTextLabel);
	log_label->set_focus_mode(FOCUS_NONE);
	log_label->set_h_size_flags(SIZE_EXPAND_FILL);
	log_label->set_v_size_flags(SIZE_EXPAND_FILL);
	log_label->push_font(mono_font);
	log_label->set_scroll_follow(true);
	log_label->connect("gui_input", this, "_log_input");
	add_child(log_label);

	HBoxContainer *hb = memnew(HBoxContainer);
	add_child(hb);

	expression_line = memnew(HistoryLineEdit);
	hb->add_child(expression_line);
	expression_line->connect("text_entered", this, "_expression_entered");

	Button *print_btn = memnew(Button);
	print_btn->set_focus_mode(FOCUS_NONE);
	print_btn->set_text("Print");
	print_btn->connect("pressed", this, "_print_pressed");
	hb->add_child(print_btn);

	Button *watch_btn = memnew(Button);
	watch_btn->set_focus_mode(FOCUS_NONE);
	watch_btn->set_text("Watch");
	watch_btn->connect("pressed", this, "_watch_pressed");
	hb->add_child(watch_btn);

	menu = memnew(PopupMenu);
	add_child(menu);
	menu->add_item("Clear History", (int)MENU_CLEAR_HISTORY);
	menu->add_item("Clear Log", (int)MENU_CLEAR_LOG);
	menu->connect("id_pressed", this, "_menu_item_pressed");
}

ExpressionEvaluator::~ExpressionEvaluator() {
	memdelete(expression);
}
