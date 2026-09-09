/**************************************************************************/
/*  line_2d_editor_plugin.cpp                                             */
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

#include "line_2d_editor_plugin.h"

#include "core/object/callable_mp.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/scene/canvas_item_editor_plugin.h"

Node2D *Line2DEditor::_get_node() const {
	return node;
}

void Line2DEditor::_set_node(Node *p_line) {
	CanvasItem *ci_editor_control = CanvasItemEditor::get_singleton()->get_viewport_control();

	if (node) {
		node->disconnect(SceneStringName(draw), callable_mp(ci_editor_control, &CanvasItem::queue_redraw));
	}

	node = Object::cast_to<Line2D>(p_line);
	if (node) {
		// Update the canvas overlay.
		node->connect(SceneStringName(draw), callable_mp(ci_editor_control, &CanvasItem::queue_redraw));
	}
}

bool Line2DEditor::_is_line() const {
	return true;
}

Variant Line2DEditor::_get_polygon(int p_idx) const {
	return _get_node()->get("points");
}

void Line2DEditor::_set_polygon(int p_idx, const Variant &p_polygon) const {
	_get_node()->set("points", p_polygon);
}

void Line2DEditor::_action_set_polygon(int p_idx, const Variant &p_previous, const Variant &p_polygon) {
	Node2D *_node = _get_node();
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->add_do_method(_node, "set_points", p_polygon);
	undo_redo->add_undo_method(_node, "set_points", p_previous);
}

Line2DEditorPlugin::Line2DEditorPlugin() :
		AbstractPolygon2DEditorPlugin(memnew(Line2DEditor), "Line2D") {
}
