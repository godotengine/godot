/**************************************************************************/
/*  remove_missing_dialog.cpp                                             */
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

#include "remove_missing_dialog.h"

#include "editor/themes/editor_scale.h"
#include "scene/gui/box_container.h"
#include "scene/gui/label.h"
#include "scene/gui/tree.h"

void RemoveMissingDialog::_on_item_edited() {
	bool has_checked = false;
	for (TreeItem *item = tree->get_root()->get_next_visible(); item != nullptr; item = item->get_next_visible()) {
		if (item->is_checked(0)) {
			has_checked = true;
			break;
		}
	}
	get_ok_button()->set_disabled(!has_checked);
}

void RemoveMissingDialog::_bind_methods() {
	ADD_SIGNAL(MethodInfo("remove_missing_projects", PropertyInfo(Variant::PACKED_STRING_ARRAY, "paths")));
}

void RemoveMissingDialog::ok_pressed() {
	Vector<String> paths;
	for (TreeItem *item = tree->get_root()->get_next_visible(); item != nullptr; item = item->get_next_visible()) {
		if (item->is_checked(0)) {
			paths.push_back(item->get_text(0));
		}
	}
	hide();
	emit_signal("remove_missing_projects", paths);
}

void RemoveMissingDialog::show_dialog(const Vector<String> &p_paths) {
	tree->clear();

	TreeItem *root = tree->create_item();

	for (const String &path : p_paths) {
		TreeItem *item = tree->create_item(root);
		item->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
		item->set_editable(0, true);
		item->set_checked(0, true);
		item->set_text(0, path);
	}

	popup_centered_clamped(Size2(480, 260) * EDSCALE);
}

RemoveMissingDialog::RemoveMissingDialog() {
	set_title(TTR("Remove Missing"));
	set_hide_on_ok(false);

	VBoxContainer *vb = memnew(VBoxContainer);
	add_child(vb);

	vb->add_child(memnew(Label(TTR("The following paths will be removed from the project list."))));

	tree = memnew(Tree);
	tree->set_custom_minimum_size(Size2(0, 100) * EDSCALE);
	tree->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	tree->set_h_scroll_enabled(false);
	tree->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	tree->set_hide_root(true);
	tree->set_select_mode(Tree::SELECT_ROW);
	tree->connect("item_edited", callable_mp(this, &RemoveMissingDialog::_on_item_edited));
	vb->add_child(tree);
}
