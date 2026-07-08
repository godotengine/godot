/**************************************************************************/
/*  editor_layouts_dialog.cpp                                             */
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

#include "editor_layouts_dialog.h"

#include "core/io/config_file.h"
#include "core/object/callable_mp.h"
#include "editor/gui/editor_validation_panel.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/item_list.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/margin_container.h"

void EditorLayoutsDialog::_validate_name() {
	const String layout_name = name->get_text().strip_edges();

	if (!layout_names->is_anything_selected()) {
		String error;

		if (layout_name.is_empty()) {
			error = TTRC("Layout name can't be empty.");
		} else if (layout_name.contains_char('/') || layout_name.contains_char('\\')) {
			error = TTRC("Layout name contains invalid characters: \"/\" or \"\\\".");
		}

		if (error != "") {
			validation->set_message(0, error, EditorValidationPanel::MSG_ERROR);
			return;
		}
	}

	if (layout_names->is_anything_selected()) {
		validation->set_message(0, TTRC("Selected layout will be overridden."), EditorValidationPanel::MSG_OK);
		return;
	}

	bool name_in_use = false;
	for (int i = 0; i < layout_names->get_item_count(); i++) {
		if (layout_names->get_item_metadata(i) == layout_name) {
			name_in_use = true;
			break;
		}
	}
	validation->set_message(0, name_in_use ? TTRC("Layout already exists and will be overridden.") : TTRC("Layout name is valid."), EditorValidationPanel::MSG_OK);
}

void EditorLayoutsDialog::_deselect_layout_names() {
	// The deselect method does not emit any signal, therefore we need update the validation state as well.
	layout_names->deselect_all();
	validation->update();
}

void EditorLayoutsDialog::_multi_selected() {
	get_ok_button()->set_disabled(!layout_names->is_anything_selected());
}

void EditorLayoutsDialog::_item_activated() {
	if (layout_names->is_anything_selected()) {
		for (const int item : layout_names->get_selected_items()) {
			emit_signal(SNAME("name_confirmed"), layout_names->get_item_metadata(item));
		}

		hide();
	}
}

void EditorLayoutsDialog::_bind_methods() {
	ADD_SIGNAL(MethodInfo("name_confirmed", PropertyInfo(Variant::STRING, "name")));
}

void EditorLayoutsDialog::ok_pressed() {
	if (layout_names->is_anything_selected()) {
		for (const int item : layout_names->get_selected_items()) {
			emit_signal(SNAME("name_confirmed"), layout_names->get_item_metadata(item));
		}
	} else if (name->is_visible() && !name->get_text().strip_edges().is_empty()) {
		emit_signal(SNAME("name_confirmed"), name->get_text().strip_edges());
	}
}

void EditorLayoutsDialog::_post_popup() {
	ConfirmationDialog::_post_popup();
	layout_names->clear();
	name->clear();

	if (save_mode) {
		layout_names->add_item(TTR("Default"));
		layout_names->set_item_metadata(0, "Default");
		name->grab_focus();
	} else {
		layout_names->grab_focus(true);
		get_ok_button()->set_disabled(true);
	}

	Ref<ConfigFile> config;
	config.instantiate();
	Error err = config->load(EditorSettings::get_singleton()->get_editor_layouts_config());
	if (err == OK) {
		Vector<String> layouts = config->get_sections();

		if (!save_mode && layouts.has("Default")) {
			layout_names->add_item(TTR("Default (Restore)"));
			layout_names->set_item_metadata(0, "Default");
		}

		for (const String &E : layouts) {
			if (E != "Default" && !E.contains_char('/')) {
				layout_names->add_item(E);
				layout_names->set_item_metadata(-1, E);
			}
		}
	}
}

EditorLayoutsDialog::EditorLayoutsDialog() {
	makevb = memnew(VBoxContainer);
	add_child(makevb);

	validation = memnew(EditorValidationPanel);
	validation->add_line(0);
	validation->set_update_callback(callable_mp(this, &EditorLayoutsDialog::_validate_name));
	validation->set_accept_button(get_ok_button());
	validation->set_v_size_flags(0);

	layout_names = memnew(ItemList);
	layout_names->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	layout_names->set_allow_rmb_select(true);
	layout_names->set_scroll_hint_mode(ItemList::SCROLL_HINT_MODE_BOTH);
	layout_names->connect(SceneStringName(item_selected), callable_mp(validation, &EditorValidationPanel::update).unbind(1));
	layout_names->connect("multi_selected", callable_mp(this, &EditorLayoutsDialog::_multi_selected).unbind(2)); // For deletion mode.
	layout_names->connect("item_activated", callable_mp(this, &EditorLayoutsDialog::_item_activated).unbind(1));

	MarginContainer *mc = makevb->add_margin_child(TTRC("Select Existing Layout:"), layout_names);
	mc->set_custom_minimum_size(Size2(300 * EDSCALE, 50 * EDSCALE));
	mc->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	mc->set_theme_type_variation("NoBorderHorizontalWindow");

	name = memnew(LineEdit);
	makevb->add_child(name);
	name->set_placeholder(TTRC("Or enter new layout name."));
	name->set_accessibility_name(TTRC("New layout name"));
	register_text_enter(name);
	name->connect(SceneStringName(focus_entered), callable_mp(this, &EditorLayoutsDialog::_deselect_layout_names));
	name->connect(SceneStringName(text_changed), callable_mp(validation, &EditorValidationPanel::update).unbind(1));

	makevb->add_child(validation);

	set_save_mode_enabled(save_mode);
}

void EditorLayoutsDialog::set_save_mode_enabled(bool p_enabled) {
	save_mode = p_enabled;

	set_title(p_enabled ? TTRC("Save Layout") : TTRC("Delete Layout"));
	set_ok_button_text(p_enabled ? TTRC("Save") : TTRC("Delete"));

	layout_names->set_select_mode(p_enabled ? ItemList::SELECT_SINGLE : ItemList::SELECT_MULTI);
	name->set_visible(p_enabled);
	validation->set_visible(p_enabled);
}
