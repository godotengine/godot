/*************************************************************************/
/*  editor_run_script.cpp                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "editor_run_script.h"

#include "editor_node.h"

void EditorScript::add_root_node(Node *p_node) {

	if (!editor) {
		EditorNode::add_io_error("EditorScript::add_root_node: " + TTR("Write your logic in the _run() method."));
		return;
	}

	if (editor->get_edited_scene()) {
		EditorNode::add_io_error("EditorScript::add_root_node: " + TTR("There is an edited scene already."));
		return;
	}

	//editor->set_edited_scene(p_node);
}

Node *EditorScript::get_scene() {

	if (!editor) {
		EditorNode::add_io_error("EditorScript::get_scene: " + TTR("Write your logic in the _run() method."));
		return NULL;
	}

	return editor->get_edited_scene();
}

void EditorScript::_run() {

	Ref<Script> s = get_script();
	ERR_FAIL_COND(!s.is_valid());
	if (!get_script_instance()) {
		EditorNode::add_io_error(TTR("Couldn't instance script:") + "\n " + s->get_path() + "\n" + TTR("Did you forget the 'tool' keyword?"));
		return;
	}

	Variant::CallError ce;
	ce.error = Variant::CallError::CALL_OK;
	get_script_instance()->call("_run", NULL, 0, ce);
	if (ce.error != Variant::CallError::CALL_OK) {

		EditorNode::add_io_error(TTR("Couldn't run script:") + "\n " + s->get_path() + "\n" + TTR("Did you forget the '_run' method?"));
	}
}

void EditorScript::set_editor(EditorNode *p_editor) {

	editor = p_editor;
}

void EditorScript::_bind_methods() {

	ClassDB::bind_method(D_METHOD("add_root_node", "node"), &EditorScript::add_root_node);
	ClassDB::bind_method(D_METHOD("get_scene"), &EditorScript::get_scene);
	BIND_VMETHOD(MethodInfo("_run"));
}

EditorScript::EditorScript() {

	editor = NULL;
}
