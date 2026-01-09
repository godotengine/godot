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
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/gui/credits_roll.h"
#include "editor/gui/editor_toaster.h"
#include "editor/gui/editor_version_button.h"
#include "editor/run/editor_run_bar.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/item_list.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/separator.h"
#include "scene/gui/split_container.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/texture_rect.h"
#include "scene/gui/tree.h"
#include "scene/resources/style_box.h"

void EditorAbout::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_TRANSLATION_CHANGED: {
			_about_text_label->set_text(
					String(U"© 2014-present ") + TTR("Godot Engine contributors") + ".\n" +
					String(U"© 2007-2014 Juan Linietsky, Ariel Manzur.\n"));

			_project_manager_label->set_text(TTR("Project Manager", "Job Title"));

			for (ItemList *il : name_lists) {
				for (int i = 0; i < il->get_item_count(); i++) {
					const Variant val = il->get_item_metadata(i);
					if (val.get_type() == Variant::STRING) {
						il->set_item_tooltip(i, val.operator String() + "\n\n" + TTR("Double-click to open in browser."));
					}
				}
			}
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			const Ref<Font> font = get_theme_font(SNAME("source"), EditorStringName(EditorFonts));
			const int font_size = get_theme_font_size(SNAME("source_size"), EditorStringName(EditorFonts));

			_tpl_text->begin_bulk_theme_override();
			_tpl_text->add_theme_font_override("normal_font", font);
			_tpl_text->add_theme_font_size_override("normal_font_size", font_size);
			_tpl_text->add_theme_constant_override(SceneStringName(line_separation), 4 * EDSCALE);
			_tpl_text->end_bulk_theme_override();

			license_text_label->begin_bulk_theme_override();
			license_text_label->add_theme_font_override("normal_font", font);
			license_text_label->add_theme_font_size_override("normal_font_size", font_size);
			license_text_label->add_theme_constant_override(SceneStringName(line_separation), 4 * EDSCALE);
			license_text_label->end_bulk_theme_override();

			_logo->set_texture(get_editor_theme_icon(SNAME("Logo")));

			for (ItemList *il : name_lists) {
				for (int i = 0; i < il->get_item_count(); i++) {
					if (il->get_item_metadata(i)) {
						il->set_item_icon(i, get_theme_icon(SNAME("ExternalLink"), EditorStringName(EditorIcons)));
						il->set_item_icon_modulate(i, get_theme_color(SNAME("font_disabled_color"), EditorStringName(Editor)));
					}
				}
			}
		} break;
	}
}

void EditorAbout::_license_tree_selected() {
	TreeItem *selected = _tpl_tree->get_selected();
	_tpl_text->scroll_to_line(0);
	_tpl_text->set_text(selected->get_metadata(0));
}

void EditorAbout::_credits_visibility_changed() {
	if (!credits_roll->is_visible()) {
		credits_roll->queue_free();
		credits_roll = nullptr;

		show();
	}
}

void EditorAbout::_item_activated(int p_idx, ItemList *p_il) {
	const Variant val = p_il->get_item_metadata(p_idx);
	if (val.get_type() == Variant::STRING) {
		OS::get_singleton()->shell_open(val);
	} else {
		// Easter egg! :D
		if (EditorRunBar::get_singleton()->is_playing()) {
			// Don't allow if the game is running, as it will look weird if it's embedded.
			EditorToaster::get_singleton()->popup_str(TTR("No distractions for this, close that game first."));
			return;
		}

		if (!credits_roll) {
			credits_roll = memnew(CreditsRoll);
			credits_roll->connect("visibility_changed", callable_mp(this, &EditorAbout::_credits_visibility_changed));
			get_tree()->get_root()->add_child(credits_roll);
		}
		credits_roll->roll_credits();
		hide();
	}
}

void EditorAbout::_item_list_resized(ItemList *p_il) {
	p_il->set_fixed_column_width(p_il->get_size().x / 3.0 - 16 * EDSCALE * 2.5); // Weird. Should be 3.0 and that's it?.
}

Label *EditorAbout::_create_section(Control *p_parent, const String &p_name, const char *const *p_src, BitField<SectionFlags> p_flags) {
	Label *lbl = memnew(Label(p_name));
	lbl->set_theme_type_variation("HeaderSmall");
	lbl->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	p_parent->add_child(lbl);

	ItemList *il = memnew(ItemList);
	il->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	il->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	il->set_same_column_width(true);
	il->set_auto_height(true);
	il->set_max_columns(p_flags.has_flag(FLAG_SINGLE_COLUMN) ? 1 : 16);
	il->add_theme_constant_override("h_separation", 16 * EDSCALE);

	// Don't allow the Easter egg in the Project Manager.
	if (p_flags.has_flag(FLAG_ALLOW_WEBSITE) || (p_flags.has_flag(FLAG_EASTER_EGG) && EditorNode::get_singleton())) {
		Ref<StyleBoxEmpty> empty_stylebox = memnew(StyleBoxEmpty);
		il->add_theme_style_override("focus", empty_stylebox);
		il->add_theme_style_override("selected", empty_stylebox);

		il->connect("item_activated", callable_mp(this, &EditorAbout::_item_activated).bind(il));
	} else {
		il->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
		il->set_focus_mode(Control::FOCUS_NONE);
	}

	const char *const *names_ptr = p_src;
	if (p_flags.has_flag(FLAG_ALLOW_WEBSITE)) {
		il->connect(SceneStringName(resized), callable_mp(this, &EditorAbout::_item_list_resized).bind(il));
		il->connect(SceneStringName(focus_exited), callable_mp(il, &ItemList::deselect_all));

		while (*names_ptr) {
			const String name = String::utf8(*names_ptr++);
			const String identifier = name.get_slicec('<', 0);
			const String website = name.get_slice_count("<") == 1 ? "" : name.get_slicec('<', 1).trim_suffix(">");

			il->add_item(identifier, nullptr, !website.is_empty());

			if (website.is_empty()) {
				il->set_item_tooltip_enabled(-1, false);
			} else {
				il->set_item_metadata(-1, website);
			}

			if (!*names_ptr && name.contains(" anonymous ")) {
				il->set_item_disabled(-1, true);
			}
		}
	} else {
		while (*names_ptr) {
			il->add_item(String::utf8(*names_ptr++), nullptr, false);
			il->set_item_tooltip_enabled(-1, false);
		}
	}

	name_lists.append(il);

	p_parent->add_child(il);

	HSeparator *hs = memnew(HSeparator);
	hs->set_modulate(Color(0, 0, 0, 0));
	p_parent->add_child(hs);

	return lbl;
}

EditorAbout::EditorAbout() {
	set_title(TTRC("Thanks from the Godot community!"));
	set_hide_on_ok(true);

	VBoxContainer *vbc = memnew(VBoxContainer);
	add_child(vbc);

	HBoxContainer *hbc = memnew(HBoxContainer);
	hbc->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	hbc->set_alignment(BoxContainer::ALIGNMENT_CENTER);
	hbc->add_theme_constant_override("separation", 30 * EDSCALE);
	vbc->add_child(hbc);

	_logo = memnew(TextureRect);
	_logo->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
	hbc->add_child(_logo);

	VBoxContainer *version_info_vbc = memnew(VBoxContainer);

	// Add a dummy control node for spacing.
	Control *v_spacer = memnew(Control);
	version_info_vbc->add_child(v_spacer);

	version_info_vbc->add_child(memnew(EditorVersionButton(EditorVersionButton::FORMAT_WITH_NAME_AND_BUILD)));

	_about_text_label = memnew(Label);
	_about_text_label->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	_about_text_label->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
	_about_text_label->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	version_info_vbc->add_child(_about_text_label);

	hbc->add_child(version_info_vbc);

	TabContainer *tc = memnew(TabContainer);
	tc->set_tab_alignment(TabBar::ALIGNMENT_CENTER);
	tc->set_custom_minimum_size(Size2(400, 200) * EDSCALE);
	tc->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	tc->set_theme_type_variation("TabContainerOdd");
	vbc->add_child(tc);

	{
		ScrollContainer *sc = memnew(ScrollContainer);
		sc->set_name(TTRC("Authors"));
		sc->set_v_size_flags(Control::SIZE_EXPAND);
		tc->add_child(sc);

		VBoxContainer *vb = memnew(VBoxContainer);
		vb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		sc->add_child(vb);

		_create_section(vb, TTRC("Project Founders"), AUTHORS_FOUNDERS, FLAG_SINGLE_COLUMN);
		_create_section(vb, TTRC("Lead Developer"), AUTHORS_LEAD_DEVELOPERS);
		// The section title will be updated in NOTIFICATION_TRANSLATION_CHANGED.
		_project_manager_label = _create_section(vb, "", AUTHORS_PROJECT_MANAGERS, FLAG_EASTER_EGG);
		_project_manager_label->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
		_create_section(vb, TTRC("Developers"), AUTHORS_DEVELOPERS);
	}

	{
		ScrollContainer *sc = memnew(ScrollContainer);
		sc->set_name(TTRC("Donors"));
		sc->set_v_size_flags(Control::SIZE_EXPAND);
		tc->add_child(sc);

		VBoxContainer *vb = memnew(VBoxContainer);
		vb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		sc->add_child(vb);

		_create_section(vb, TTRC("Patrons"), DONORS_PATRONS, FLAG_ALLOW_WEBSITE | FLAG_SINGLE_COLUMN);
		_create_section(vb, TTRC("Platinum Sponsors"), DONORS_SPONSORS_PLATINUM, FLAG_ALLOW_WEBSITE);
		_create_section(vb, TTRC("Gold Sponsors"), DONORS_SPONSORS_GOLD, FLAG_ALLOW_WEBSITE);
		_create_section(vb, TTRC("Silver Sponsors"), DONORS_SPONSORS_SILVER, FLAG_ALLOW_WEBSITE);
		_create_section(vb, TTRC("Diamond Members"), DONORS_MEMBERS_DIAMOND, FLAG_ALLOW_WEBSITE);
		_create_section(vb, TTRC("Titanium Members"), DONORS_MEMBERS_TITANIUM, FLAG_ALLOW_WEBSITE);
		_create_section(vb, TTRC("Platinum Members"), DONORS_MEMBERS_PLATINUM, FLAG_ALLOW_WEBSITE);
		_create_section(vb, TTRC("Gold Members"), DONORS_MEMBERS_GOLD, FLAG_ALLOW_WEBSITE);
	}

	// License.

	license_text_label = memnew(RichTextLabel);
	license_text_label->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	license_text_label->set_threaded(true);
	license_text_label->set_name(TTRC("License"));
	license_text_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	license_text_label->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	license_text_label->set_text(String::utf8(GODOT_LICENSE_TEXT));
	tc->add_child(license_text_label);

	// Thirdparty License.

	VBoxContainer *license_thirdparty = memnew(VBoxContainer);
	license_thirdparty->set_name(TTRC("Third-party Licenses"));
	license_thirdparty->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	tc->add_child(license_thirdparty);

	Label *tpl_label = memnew(Label(TTRC("Godot Engine relies on a number of third-party free and open source libraries, all compatible with the terms of its MIT license. The following is an exhaustive list of all such third-party components with their respective copyright statements and license terms.")));
	tpl_label->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	tpl_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	tpl_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	tpl_label->set_size(Size2(630, 1) * EDSCALE);
	license_thirdparty->add_child(tpl_label);

	HSplitContainer *tpl_hbc = memnew(HSplitContainer);
	tpl_hbc->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	tpl_hbc->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	tpl_hbc->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	tpl_hbc->set_split_offset(240 * EDSCALE);
	license_thirdparty->add_child(tpl_hbc);

	_tpl_tree = memnew(Tree);
	_tpl_tree->set_hide_root(true);
	TreeItem *root = _tpl_tree->create_item();
	TreeItem *tpl_ti_all = _tpl_tree->create_item(root);
	tpl_ti_all->set_text(0, TTRC("All Components"));
	tpl_ti_all->set_auto_translate_mode(0, AUTO_TRANSLATE_MODE_ALWAYS);
	TreeItem *tpl_ti_tp = _tpl_tree->create_item(root);
	tpl_ti_tp->set_text(0, TTRC("Components"));
	tpl_ti_tp->set_auto_translate_mode(0, AUTO_TRANSLATE_MODE_ALWAYS);
	tpl_ti_tp->set_selectable(0, false);
	TreeItem *tpl_ti_lc = _tpl_tree->create_item(root);
	tpl_ti_lc->set_text(0, TTRC("Licenses"));
	tpl_ti_lc->set_auto_translate_mode(0, AUTO_TRANSLATE_MODE_ALWAYS);
	tpl_ti_lc->set_selectable(0, false);
	String long_text = "";
	for (int component_index = 0; component_index < COPYRIGHT_INFO_COUNT; component_index++) {
		const ComponentCopyright &component = COPYRIGHT_INFO[component_index];
		TreeItem *ti = _tpl_tree->create_item(tpl_ti_tp);
		String component_name = String::utf8(component.name);
		ti->set_text(0, component_name);
		String text = component_name + "\n";
		long_text += "- " + component_name + "\n";
		for (int part_index = 0; part_index < component.part_count; part_index++) {
			const ComponentCopyrightPart &part = component.parts[part_index];
			text += "\n    Files:";
			for (int file_num = 0; file_num < part.file_count; file_num++) {
				text += "\n        " + String::utf8(part.files[file_num]);
			}
			String copyright;
			for (int copyright_index = 0; copyright_index < part.copyright_count; copyright_index++) {
				copyright += String::utf8("\n    \xc2\xa9 ") + String::utf8(part.copyright_statements[copyright_index]);
			}
			text += copyright;
			long_text += copyright;
			String license = "\n    License: " + String::utf8(part.license) + "\n";
			text += license;
			long_text += license + "\n";
		}
		ti->set_metadata(0, text);
	}
	for (int i = 0; i < LICENSE_COUNT; i++) {
		TreeItem *ti = _tpl_tree->create_item(tpl_ti_lc);
		String licensename = String::utf8(LICENSE_NAMES[i]);
		ti->set_text(0, licensename);
		long_text += "- " + licensename + "\n\n";
		String licensebody = String::utf8(LICENSE_BODIES[i]);
		ti->set_metadata(0, licensebody);
		long_text += "    " + licensebody.replace("\n", "\n    ") + "\n\n";
	}
	tpl_ti_all->set_metadata(0, long_text);
	tpl_hbc->add_child(_tpl_tree);

	_tpl_text = memnew(RichTextLabel);
	_tpl_text->set_threaded(true);
	_tpl_text->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	_tpl_text->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	tpl_hbc->add_child(_tpl_text);

	_tpl_tree->connect(SceneStringName(item_selected), callable_mp(this, &EditorAbout::_license_tree_selected));
	tpl_ti_all->select(0);
	_tpl_text->set_text(tpl_ti_all->get_metadata(0));
}
