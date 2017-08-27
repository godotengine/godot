/*************************************************************************/
/*  editor_about.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#include "editor_about.h"

#include "authors.gen.h"
#include "license.gen.h"
#include "version.h"
#include "version_hash.gen.h"

void EditorAbout::_license_tree_selected() {

	TreeItem *selected = _tpl_tree->get_selected();
	_tpl_text->select(0, 0, 0, 0);
	_tpl_text->set_text(selected->get_metadata(0));
}

void EditorAbout::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_license_tree_selected"), &EditorAbout::_license_tree_selected);
}

TextureRect *EditorAbout::get_logo() const {

	return _logo;
}

EditorAbout::EditorAbout() {

	set_title(TTR("Thanks from the Godot community!"));
	get_ok()->set_text(TTR("Thanks!"));
	set_hide_on_ok(true);
	set_resizable(true);

	VBoxContainer *vbc = memnew(VBoxContainer);
	HBoxContainer *hbc = memnew(HBoxContainer);
	hbc->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	hbc->set_alignment(BoxContainer::ALIGN_CENTER);
	hbc->add_constant_override("separation", 30 * EDSCALE);
	add_child(vbc);
	vbc->add_child(hbc);

	_logo = memnew(TextureRect);
	hbc->add_child(_logo);

	String hash = String(VERSION_HASH);
	if (hash.length() != 0)
		hash = "." + hash.left(7);

	Label *about_text = memnew(Label);
	about_text->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
	about_text->set_text(VERSION_FULL_NAME + hash + String::utf8("\n\u00A9 2007-2017 Juan Linietsky, Ariel Manzur.\n\u00A9 2014-2017 ") +
						 TTR("Godot Engine contributors") + "\n");
	hbc->add_child(about_text);

	TabContainer *tc = memnew(TabContainer);
	tc->set_custom_minimum_size(Size2(630, 240) * EDSCALE);
	tc->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	vbc->add_child(tc);

	ScrollContainer *dev_base = memnew(ScrollContainer);
	dev_base->set_name(TTR("Authors"));
	dev_base->set_v_size_flags(Control::SIZE_EXPAND);
	tc->add_child(dev_base);

	VBoxContainer *dev_vbc = memnew(VBoxContainer);
	dev_vbc->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	dev_base->add_child(dev_vbc);

	List<String> dev_sections;
	dev_sections.push_back(TTR("Project Founders"));
	dev_sections.push_back(TTR("Lead Developer"));
	dev_sections.push_back(TTR("Project Manager"));
	dev_sections.push_back(TTR("Developers"));

	const char **dev_src[] = { dev_founders, dev_lead, dev_manager, dev_names };

	for (int i = 0; i < dev_sections.size(); i++) {

		Label *lbl = memnew(Label);
		lbl->set_text(dev_sections[i]);
		dev_vbc->add_child(lbl);

		ItemList *il = memnew(ItemList);
		il->set_max_columns(16);
		il->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		il->set_fixed_column_width(230 * EDSCALE);
		il->set_auto_height(true);
		const char **dev_names_ptr = dev_src[i];
		while (*dev_names_ptr)
			il->add_item(String::utf8(*dev_names_ptr++), NULL, false);
		dev_vbc->add_child(il);

		HSeparator *hs = memnew(HSeparator);
		hs->set_modulate(Color(0, 0, 0, 0));
		dev_vbc->add_child(hs);
	}

	TextEdit *license = memnew(TextEdit);
	license->set_name(TTR("License"));
	license->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	license->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	license->set_wrap(true);
	license->set_readonly(true);
	license->set_text(String::utf8(about_license));
	tc->add_child(license);

	VBoxContainer *license_thirdparty = memnew(VBoxContainer);
	license_thirdparty->set_name(TTR("Thirdparty License"));
	license_thirdparty->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	tc->add_child(license_thirdparty);

	Label *tpl_label = memnew(Label);
	tpl_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	tpl_label->set_autowrap(true);
	tpl_label->set_text(TTR("Godot Engine relies on a number of thirdparty free and open source libraries, all compatible with the terms of its MIT license. The following is an exhaustive list of all such thirdparty components with their respective copyright statements and license terms."));
	license_thirdparty->add_child(tpl_label);

	HSplitContainer *tpl_hbc = memnew(HSplitContainer);
	tpl_hbc->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	tpl_hbc->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	tpl_hbc->set_split_offset(240 * EDSCALE);
	license_thirdparty->add_child(tpl_hbc);

	_tpl_tree = memnew(Tree);
	_tpl_tree->set_hide_root(true);
	TreeItem *root = _tpl_tree->create_item();
	TreeItem *tpl_ti_all = _tpl_tree->create_item(root);
	tpl_ti_all->set_text(0, TTR("All Components"));
	TreeItem *tpl_ti_tp = _tpl_tree->create_item(root);
	tpl_ti_tp->set_text(0, TTR("Components"));
	tpl_ti_tp->set_selectable(0, false);
	TreeItem *tpl_ti_lc = _tpl_tree->create_item(root);
	tpl_ti_lc->set_text(0, TTR("Licenses"));
	tpl_ti_lc->set_selectable(0, false);
	int read_idx = 0;
	String long_text = "";
	for (int i = 0; i < THIRDPARTY_COUNT; i++) {

		TreeItem *ti = _tpl_tree->create_item(tpl_ti_tp);
		String thirdparty = String(about_thirdparty[i]);
		ti->set_text(0, thirdparty);
		String text = thirdparty + "\n";
		long_text += "- " + thirdparty + "\n\n";
		for (int j = 0; j < about_tp_copyright_count[i]; j++) {

			text += "\n    Files:\n        " + String(about_tp_file[read_idx]).replace("\n", "\n        ") + "\n";
			String copyright = String::utf8("    \u00A9 ") + String::utf8(about_tp_copyright[read_idx]).replace("\n", String::utf8("\n    \u00A9 "));
			text += copyright;
			long_text += copyright;
			String license = "\n    License: " + String(about_tp_license[read_idx]) + "\n";
			text += license;
			long_text += license + "\n";
			read_idx++;
		}
		ti->set_metadata(0, text);
	}
	for (int i = 0; i < LICENSE_COUNT; i++) {

		TreeItem *ti = _tpl_tree->create_item(tpl_ti_lc);
		String licensename = String(about_license_name[i]);
		ti->set_text(0, licensename);
		long_text += "- " + licensename + "\n\n";
		String licensebody = String(about_license_body[i]);
		ti->set_metadata(0, licensebody);
		long_text += "    " + licensebody.replace("\n", "\n    ") + "\n\n";
	}
	tpl_ti_all->set_metadata(0, long_text);
	tpl_hbc->add_child(_tpl_tree);

	_tpl_text = memnew(TextEdit);
	_tpl_text->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	_tpl_text->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	_tpl_text->set_wrap(true);
	_tpl_text->set_readonly(true);
	tpl_hbc->add_child(_tpl_text);

	_tpl_tree->connect("item_selected", this, "_license_tree_selected");
	tpl_ti_all->select(0);
	_tpl_text->set_text(tpl_ti_all->get_metadata(0));
}

EditorAbout::~EditorAbout() {}
