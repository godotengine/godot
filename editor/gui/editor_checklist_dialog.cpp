/**************************************************************************/
/*  editor_checklist_dialog.cpp                                           */
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

#include "editor_checklist_dialog.h"

#include "editor/themes/editor_scale.h"
#include "scene/gui/tree.h"

void EditorChecklistDialog::_visibility_changed() {
	if (!is_visible()) {
		tree = nullptr;

		int child_count = get_child_count(false);

		while (--child_count > -1) {
			get_child(child_count, false)->queue_free();
		}
	}
}

void EditorChecklistDialog::reload(Tree *p_tree, const String &p_title, const String &p_label_text) {
	set_title(p_title);

	VBoxContainer *vb = memnew(VBoxContainer);
	add_child(vb);
	vb->add_margin_child(p_label_text, p_tree, true);
	tree = p_tree;
}

Vector<TreeItem*> EditorChecklistDialog::get_all_checked() const {
	Vector<TreeItem *> result;

	if (!tree || !tree->get_root()) {
		return result;
	}

	Array stack;
	stack.push_back(tree->get_root());

	// Iterative DFS preorder traversal
	while (!stack.is_empty()) {
		TreeItem *item = Object::cast_to<TreeItem>(stack.pop_back());

		if (item->is_checked(0)) {
			result.append(item);
		}

		TypedArray<TreeItem> children = item->get_children();

		for (int i = children.size() - 1; i > -1; --i) {
			stack.push_back(item->get_child(i));
		}
	}

	return result;
}

EditorChecklistDialog::EditorChecklistDialog() {
	set_size(Size2(0, 400) * EDSCALE);

	callable_mp((Object *)this, &Object::connect).bind("visibility_changed", callable_mp(this, &EditorChecklistDialog::_visibility_changed), Object::CONNECT_DEFERRED).call_deferred();
}
