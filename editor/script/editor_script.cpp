/**************************************************************************/
/*  editor_script.cpp                                                     */
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

#include "editor_script.h"

#include "editor/editor_interface.h"
#include "editor/editor_node.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/scene/editor_scene_tabs.h"
#include "scene/main/node.h"
#include "scene/resources/packed_scene.h"

#ifndef DISABLE_DEPRECATED
void EditorScript::add_root_node(Node *p_node) {
	WARN_DEPRECATED_MSG("EditorScript::add_root_node is deprecated. Use EditorInterface::add_root_node instead.");
	EditorInterface::get_singleton()->add_root_node(p_node);
}

Node *EditorScript::get_scene() const {
	WARN_DEPRECATED_MSG("EditorScript::get_scene is deprecated. Use EditorInterface::get_edited_scene_root instead.");
	if (!EditorNode::get_singleton()) {
		EditorNode::add_io_error("EditorScript::get_scene: " + TTR("Write your logic in the _run() method."));
		return nullptr;
	}

	return EditorNode::get_singleton()->get_edited_scene();
}

EditorInterface *EditorScript::get_editor_interface() const {
	WARN_DEPRECATED_MSG("EditorInterface is a global singleton and can be accessed directly by its name.");
	return EditorInterface::get_singleton();
}
#endif // DISABLE_DEPRECATED

void EditorScript::run() {
	GDVIRTUAL_CALL(_run);
}

void EditorScript::_bind_methods() {
#ifndef DISABLE_DEPRECATED
	ClassDB::bind_method(D_METHOD("add_root_node", "node"), &EditorScript::add_root_node);

	ClassDB::bind_method(D_METHOD("get_scene"), &EditorScript::get_scene);

	ClassDB::bind_method(D_METHOD("get_editor_interface"), &EditorScript::get_editor_interface);
#endif // DISABLE_DEPRECATED

	GDVIRTUAL_BIND(_run);
}
