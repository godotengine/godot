/*************************************************************************/
/*  expression_evaluator.h                                               */
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

#ifndef EXPRESSION_EVALUATOR_H
#define EXPRESSION_EVALUATOR_H

#include "core/vector.h"
#include "scene/gui/box_container.h"

class RichTextLabel;
class LineEdit;
class PopupMenu;
class HistoryLineEdit;

class ExpressionEvaluator : public VBoxContainer {

	GDCLASS(ExpressionEvaluator, VBoxContainer);

	RichTextLabel *log_label;
	HistoryLineEdit *expression_line;
	PopupMenu *menu;

	Expression *expression;

	void _expression_entered(const String &p_text);
	void _print_pressed();
	void _watch_pressed();
	void _log_input(const Ref<InputEvent> &p_event);
	void _menu_item_pressed(int p_id);

	void _print_expression(const String &p_text);

protected:
	static void _bind_methods();

public:
	void set_result(const String &p_result, bool p_error);

	ExpressionEvaluator();
	~ExpressionEvaluator();
};

#endif // EXPRESSION_EVALUATOR_H
