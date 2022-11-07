/**************************************************************************/
/*  editor_expression_evaluator.h                                         */
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

#ifndef EDITOR_EXPRESSION_EVALUATOR_H
#define EDITOR_EXPRESSION_EVALUATOR_H

#include "editor/debugger/editor_debugger_inspector.h"
#include "editor/debugger/script_editor_debugger.h"
#include "scene/debugger/scene_debugger.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/split_container.h"
#include "scene/gui/tree.h"

class EditorExpressionEvaluator : public VBoxContainer {
	GDCLASS(EditorExpressionEvaluator, VBoxContainer)

private:
	Ref<RemoteDebuggerPeer> peer;

	LineEdit *expression_input;
	Button *evaluate_btn;
	Button *clear_btn;

	EditorDebuggerInspector *inspector;

	void _remote_object_selected(ObjectID p_id);
	void _on_expression_input_submitted(String p_expression);

protected:
	ScriptEditorDebugger *editor_debugger;

public:
	void evaluate();
	void clear();
	void set_expression_evaluator(ScriptEditorDebugger *p_editor_debugger);
	void add_value(const Array &p_array);

	EditorExpressionEvaluator();
};

#endif // EDITOR_EXPRESSION_EVALUATOR_H
