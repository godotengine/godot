/**************************************************************************/
/*  editor_verify_dialog.cpp                                              */
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

#include "editor_verify_dialog.h"

#include "core/config/project_settings.h"
#include "editor/export/project_export.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/tree.h"

bool EditorVerifyDialog::reload(int p_export_type, const Callable &p_export_all_callable, const Callable &p_export_callable, const Callable &p_export_pck_zip_callable) {
	phrase = GLOBAL_GET("application/config/export_phrase");
	Dictionary checklists = GLOBAL_GET("application/config/export_checklists");

	if (phrase.is_empty() && checklists.is_empty()) {
		return false;
	}

	items_count = 0;
	evaluation_bit = (!checklists.is_empty() << 1) | !phrase.is_empty();

	checklists_label->hide();
	checklists_tree->hide();

	phrase_label->hide();
	phrase_line_edit->hide();

	Button *ok_bttn = get_ok_button();

	ok_bttn->set_disabled(true);

	if (ok_bttn->is_connected("pressed", p_export_all_callable)) {
		ok_bttn->disconnect("pressed", p_export_all_callable);
	} else if (ok_bttn->is_connected("pressed", p_export_callable)) {
		ok_bttn->disconnect("pressed", p_export_callable);
	} else if (ok_bttn->is_connected("pressed", p_export_pck_zip_callable)) {
		ok_bttn->disconnect("pressed", p_export_pck_zip_callable);
	}

	switch (p_export_type) {
		case ProjectExportDialog::EXPORT_ALL:
			ok_bttn->connect("pressed", p_export_all_callable);
			break;
		case ProjectExportDialog::EXPORT:
			ok_bttn->connect("pressed", p_export_callable);
			break;
		case ProjectExportDialog::EXPORT_PCK_ZIP:
			ok_bttn->connect("pressed", p_export_pck_zip_callable);
			break;
	}

	switch (evaluation_bit) {
		case 0b0001:
			evaluation_bit = 0b0010;
			set_size(Size2(0, 0));
			ok_bttn->set_tooltip_text(TTR("Enter the phrase defined in the Project Settings to proceed."));
			_show_phrase();
			break;
		case 0b0010:
			evaluation_bit = 0b0001;
			set_size(Size2(0, 400) * EDSCALE);
			ok_bttn->set_tooltip_text(TTR("Check all items defined in the Project Settings to proceed."));
			_show_checklists(checklists);
			break;
		case 0b0011:
			evaluation_bit = 0;
			set_size(Size2(0, 500) * EDSCALE);
			ok_bttn->set_tooltip_text(TTR("Check all items and enter the phrase defined in the Project Settings to proceed."));
			_show_phrase();
			_show_checklists(checklists);
			break;
	}

	return true;
}

void EditorVerifyDialog::_show_phrase() {
	phrase_label->set_text(vformat(TTR("Enter the phrase \"%s\" below:"), phrase));

	phrase_line_edit->set_text("");
	callable_mp((Control *)phrase_line_edit, &Control::grab_focus).call_deferred();

	phrase_label->show();
	phrase_line_edit->show();
}

void EditorVerifyDialog::_show_checklists(const Dictionary &p_checklists) {
	checklists_tree->clear();

	checklists_tree->create_item()->set_text(0, "Items");

	Array titles = p_checklists.keys();

	for (int i = 0; i < titles.size(); ++i) {
		String checklist_title = static_cast<String>(titles[i]);

		TreeItem *root = checklists_tree->create_item();
		root->set_text(0, checklist_title);

		PackedStringArray items = static_cast<PackedStringArray>(p_checklists[checklist_title]);

		items_count += items.size();

		for (const String &item : items) {
			TreeItem *new_item = checklists_tree->create_item(root);
			new_item->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
			new_item->set_text(0, item);
			new_item->set_editable(0, true);
		}
	}

	checklists_label->show();
	checklists_tree->show();
}

void EditorVerifyDialog::_item_edited() {
	if (checklists_tree->get_edited()->is_checked(0)) {
		if (--items_count == 0) {
			evaluation_bit |= 0b0010;
		}
	} else {
		++items_count;
		evaluation_bit &= 0b1101;
	}

	_evaluate();
}

void EditorVerifyDialog::_validate_phrase(const String &p_text) {
	if (p_text == phrase) {
		evaluation_bit |= 0b0001;
	} else {
		evaluation_bit &= 0b1110;
	}

	_evaluate();
}

void EditorVerifyDialog::_evaluate() {
	if (evaluation_bit == 0b0011) {
		get_ok_button()->set_disabled(false);
	} else {
		get_ok_button()->set_disabled(true);
	}
}

EditorVerifyDialog::EditorVerifyDialog() {
	set_title(TTR("Verify before exporting"));

	VBoxContainer *vb = memnew(VBoxContainer);
	add_child(vb);

	checklists_tree = memnew(Tree);
	checklists_tree->connect("item_edited", callable_mp(this, &EditorVerifyDialog::_item_edited));
	vb->add_margin_child("Check the items below:", checklists_tree, true);
	checklists_label = Object::cast_to<Label>(vb->get_child(0));

	phrase_line_edit = memnew(LineEdit(TTR("Case-sensitive")));
	phrase_line_edit->connect("text_changed", callable_mp(this, &EditorVerifyDialog::_validate_phrase));
	vb->add_margin_child("", phrase_line_edit);
	phrase_label = Object::cast_to<Label>(vb->get_child(2));
}
