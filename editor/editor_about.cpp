/**************************************************************************/
/*  editor_about.cpp                                                      */
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

#include "editor_about.h"

#include "core/authors.gen.h"
#include "core/donors.gen.h"
#include "core/license.gen.h"
#include "core/version.h"
#include "editor_node.h"

// The metadata key used to store and retrieve the version text to copy to the clipboard.
const String EditorAbout::META_TEXT_TO_COPY = "text_to_copy";

void EditorAbout::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			Ref<Font> font = get_font("source", "EditorFonts");
			_tpl_text->add_font_override("normal_font", font);
			_tpl_text->add_constant_override("line_separation", 6 * EDSCALE);
			_license_text->add_font_override("normal_font", font);
			_license_text->add_constant_override("line_separation", 6 * EDSCALE);
			_logo->set_texture(get_icon("Logo", "EditorIcons"));
		} break;
	}
}

void EditorAbout::_license_tree_selected() {
	TreeItem *selected = _tpl_tree->get_selected();
	_tpl_text->scroll_to_line(0);
	_tpl_text->set_text(selected->get_metadata(0));
}

void EditorAbout::_version_button_pressed() {
	OS::get_singleton()->set_clipboard(version_btn->get_meta(META_TEXT_TO_COPY));
}

void EditorAbout::_bind_methods() {
	ClassDB::bind_method("_version_button_pressed", &EditorAbout::_version_button_pressed);
	ClassDB::bind_method(D_METHOD("_license_tree_selected"), &EditorAbout::_license_tree_selected);
}

TextureRect *EditorAbout::get_logo() const {
	return _logo;
}

ScrollContainer *EditorAbout::_populate_list(const String &p_name, const List<String> &p_sections, const char *const *const p_src[], const int p_flag_single_column) {
	ScrollContainer *sc = memnew(ScrollContainer);
	sc->set_name(p_name);
	sc->set_v_size_flags(Control::SIZE_EXPAND);

	VBoxContainer *vbc = memnew(VBoxContainer);
	vbc->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	sc->add_child(vbc);

	for (int i = 0; i < p_sections.size(); i++) {
		bool single_column = p_flag_single_column & 1 << i;
		const char *const *names_ptr = p_src[i];
		if (*names_ptr) {
			Label *lbl = memnew(Label);
			lbl->set_text(p_sections[i]);
			vbc->add_child(lbl);

			ItemList *il = memnew(ItemList);
			il->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			il->set_same_column_width(true);
			il->set_auto_height(true);
			il->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
			il->add_constant_override("hseparation", 16 * EDSCALE);
			while (*names_ptr) {
				il->add_item(String::utf8(*names_ptr++), nullptr, false);
			}
			il->set_max_columns(il->get_item_count() < 4 || single_column ? 1 : 16);
			vbc->add_child(il);

			HSeparator *hs = memnew(HSeparator);
			hs->set_modulate(Color(0, 0, 0, 0));
			vbc->add_child(hs);
		}
	}

	return sc;
}

EditorAbout::EditorAbout() {
	set_title(TTR("Thanks from the Godot community!"));
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

	VBoxContainer *version_info_vbc = memnew(VBoxContainer);

	// Add a dummy control node for spacing.
	Control *v_spacer = memnew(Control);
	version_info_vbc->add_child(v_spacer);

	version_btn = memnew(LinkButton);
	String hash = String(VERSION_HASH);
	if (hash.length() != 0) {
		hash = " " + vformat("[%s]", hash.left(9));
	}
	version_btn->set_text(VERSION_FULL_NAME + hash);
	// Set the text to copy in metadata as it slightly differs from the button's text.
	version_btn->set_meta(META_TEXT_TO_COPY, "v" VERSION_FULL_BUILD + hash);
	version_btn->set_underline_mode(LinkButton::UNDERLINE_MODE_ON_HOVER);
	version_btn->set_tooltip(TTR("Click to copy."));
	version_btn->connect("pressed", this, "_version_button_pressed");
	version_info_vbc->add_child(version_btn);

	Label *about_text = memnew(Label);
	about_text->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
	about_text->set_text(
			String::utf8("\xc2\xa9 2014-present ") + TTR("Godot Engine contributors") + "." +
			String::utf8("\n\xc2\xa9 2007-2014 Juan Linietsky, Ariel Manzur.\n"));
	version_info_vbc->add_child(about_text);

	hbc->add_child(version_info_vbc);

	TabContainer *tc = memnew(TabContainer);
	tc->set_custom_minimum_size(Size2(400, 200) * EDSCALE);
	tc->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	vbc->add_child(tc);

	// Authors

	List<String> dev_sections;
	dev_sections.push_back(TTR("Project Founders"));
	dev_sections.push_back(TTR("Lead Developer"));
	// TRANSLATORS: This refers to a job title.
	dev_sections.push_back(TTR("Project Manager", "Job Title"));
	dev_sections.push_back(TTR("Developers"));
	const char *const *dev_src[] = {
		AUTHORS_FOUNDERS,
		AUTHORS_LEAD_DEVELOPERS,
		AUTHORS_PROJECT_MANAGERS,
		AUTHORS_DEVELOPERS,
	};
	tc->add_child(_populate_list(TTR("Authors"), dev_sections, dev_src, 1));

	// Donors

	List<String> donor_sections;
	donor_sections.push_back(TTR("Patrons"));
	donor_sections.push_back(TTR("Platinum Sponsors"));
	donor_sections.push_back(TTR("Gold Sponsors"));
	donor_sections.push_back(TTR("Silver Sponsors"));
	donor_sections.push_back(TTR("Diamond Members"));
	donor_sections.push_back(TTR("Titanium Members"));
	donor_sections.push_back(TTR("Platinum Members"));
	donor_sections.push_back(TTR("Gold Members"));
	const char *const *donor_src[] = {
		DONORS_PATRONS,
		DONORS_SPONSORS_PLATINUM,
		DONORS_SPONSORS_GOLD,
		DONORS_SPONSORS_SILVER,
		DONORS_MEMBERS_DIAMOND,
		DONORS_MEMBERS_TITANIUM,
		DONORS_MEMBERS_PLATINUM,
		DONORS_MEMBERS_GOLD,
	};
	tc->add_child(_populate_list(TTR("Donors"), donor_sections, donor_src, 3));

	// License

	_license_text = memnew(RichTextLabel);
	_license_text->set_name(TTR("License"));
	_license_text->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	_license_text->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	_license_text->set_text(String::utf8(GODOT_LICENSE_TEXT));
	tc->add_child(_license_text);

	// Thirdparty License

	VBoxContainer *license_thirdparty = memnew(VBoxContainer);
	license_thirdparty->set_name(TTR("Third-party Licenses"));
	license_thirdparty->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	tc->add_child(license_thirdparty);

	Label *tpl_label = memnew(Label);
	tpl_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	tpl_label->set_autowrap(true);
	tpl_label->set_text(TTR("Godot Engine relies on a number of third-party free and open source libraries, all compatible with the terms of its MIT license. The following is an exhaustive list of all such third-party components with their respective copyright statements and license terms."));
	tpl_label->set_size(Size2(630, 1) * EDSCALE);
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
	String long_text = "";
	for (int component_index = 0; component_index < COPYRIGHT_INFO_COUNT; component_index++) {
		const ComponentCopyright &component = COPYRIGHT_INFO[component_index];
		TreeItem *ti = _tpl_tree->create_item(tpl_ti_tp);
		String component_name = component.name;
		ti->set_text(0, component_name);
		String text = component_name + "\n";
		long_text += "- " + component_name + "\n";
		for (int part_index = 0; part_index < component.part_count; part_index++) {
			const ComponentCopyrightPart &part = component.parts[part_index];
			text += "\n    Files:";
			for (int file_num = 0; file_num < part.file_count; file_num++) {
				text += "\n        " + String(part.files[file_num]);
			}
			String copyright;
			for (int copyright_index = 0; copyright_index < part.copyright_count; copyright_index++) {
				copyright += String::utf8("\n    \xc2\xa9 ") + String::utf8(part.copyright_statements[copyright_index]);
			}
			text += copyright;
			long_text += copyright;
			String license = "\n    License: " + String(part.license) + "\n";
			text += license;
			long_text += license + "\n";
		}
		ti->set_metadata(0, text);
	}
	for (int i = 0; i < LICENSE_COUNT; i++) {
		TreeItem *ti = _tpl_tree->create_item(tpl_ti_lc);
		String licensename = String(LICENSE_NAMES[i]);
		ti->set_text(0, licensename);
		long_text += "- " + licensename + "\n\n";
		String licensebody = String(LICENSE_BODIES[i]);
		ti->set_metadata(0, licensebody);
		long_text += "    " + licensebody.replace("\n", "\n    ") + "\n\n";
	}
	tpl_ti_all->set_metadata(0, long_text);
	tpl_hbc->add_child(_tpl_tree);

	_tpl_text = memnew(RichTextLabel);
	_tpl_text->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	_tpl_text->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	tpl_hbc->add_child(_tpl_text);

	_tpl_tree->connect("item_selected", this, "_license_tree_selected");
	tpl_ti_all->select(0);
	_tpl_text->set_text(tpl_ti_all->get_metadata(0));
}

EditorAbout::~EditorAbout() {}
