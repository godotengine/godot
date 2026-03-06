/**************************************************************************/
/*  scene_paint_3d_editor_plugin.cpp                                      */
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

#include "scene_paint_3d_editor_plugin.h"

#include "editor/editor_node.h"

void ScenePaint3DEditor::_can_handle(bool p_is_node_3d, bool p_edit) {
}

void ScenePaint3DEditor::_edit(Object *p_object) {
}

ScenePaint3DEditor::ScenePaint3DEditor() {
}

void ScenePaint3DEditorPlugin::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
		} break;
	}
}

void ScenePaint3DEditorPlugin::edit(Object *p_object) {
	scene_paint_3d_editor->_edit(p_object);
}

bool ScenePaint3DEditorPlugin::handles(Object *p_object) const {
	return is_node_3d = bool(Object::cast_to<Node3D>(p_object));
}

void ScenePaint3DEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		scene_paint_3d_editor->_can_handle(is_node_3d, false);
	} else {
		scene_paint_3d_editor->_can_handle(is_node_3d, true);
	}
}

ScenePaint3DEditorPlugin::ScenePaint3DEditorPlugin() {
	scene_paint_3d_editor = memnew(ScenePaint3DEditor);
	EditorNode::get_singleton()->get_gui_base()->add_child(scene_paint_3d_editor);
	make_visible(false);
}
