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
#include "editor/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/item_list.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/margin_container.h"

void EditorLayoutsDialog::_line_gui_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventKey> k = p_event;

	if (k.is_valid()) {
		if (k->is_action_pressed(SNAME("ui_text_submit"), false, true)) {
			if (get_hide_on_ok()) {
				hide();
			}
			ok_pressed();
			set_input_as_handled();
		} else if (k->is_action_pressed(SNAME("ui_cancel"), false, true)) {
			hide();
			set_input_as_handled();
		}
	}
}

void EditorLayoutsDialog::_update_ok_disable_state() {
	if (layout_names->is_anything_selected()) {
		get_ok_button()->set_disabled(false);
	} else {
		get_ok_button()->set_disabled(!name->is_visible() || name->get_text().strip_edges().is_empty());
	}
}

void EditorLayoutsDialog::_deselect_layout_names() {
	// The deselect method does not emit any signal, therefore we need update the disable state as well.
	layout_names->deselect_all();
	_update_ok_disable_state();
}

void EditorLayoutsDialog::_bind_methods() {
	ADD_SIGNAL(MethodInfo("name_confirmed", PropertyInfo(Variant::STRING, "name")));
}

void EditorLayoutsDialog::ok_pressed() {
	if (layout_names->is_anything_selected()) {
		Vector<int> const selected_items = layout_names->get_selected_items();
		for (int i = 0; i < selected_items.size(); ++i) {
			emit_signal(SNAME("name_confirmed"), layout_names->get_item_text(selected_items[i]));
		}
	} else if (name->is_visible() && !name->get_text().strip_edges().is_empty()) {
		emit_signal(SNAME("name_confirmed"), name->get_text().strip_edges());
	}
}

void EditorLayoutsDialog::_post_popup() {
	ConfirmationDialog::_post_popup();
	layout_names->clear();
	name->clear();

	Ref<ConfigFile> config;
	config.instantiate();
	Error err = config->load(EditorSettings::get_singleton()->get_editor_layouts_config());
	if (err != OK) {
		return;
	}

	List<String> layouts;
	config.ptr()->get_sections(&layouts);

	for (const String &E : layouts) {
		layout_names->add_item(E);
	}
	if (name->is_visible()) {
		name->grab_focus();
	} else {
		layout_names->grab_focus();
	}
}

EditorLayoutsDialog::EditorLayoutsDialog() {
	makevb = memnew(VBoxContainer);
	add_child(makevb);

	layout_names = memnew(ItemList);
	layout_names->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	layout_names->set_auto_height(true);
	layout_names->set_custom_minimum_size(Size2(300 * EDSCALE, 50 * EDSCALE));
	layout_names->set_visible(true);
	layout_names->set_offset(SIDE_TOP, 5);
	layout_names->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	layout_names->set_select_mode(ItemList::SELECT_MULTI);
	layout_names->set_allow_rmb_select(true);
	layout_names->connect("multi_selected", callable_mp(this, &EditorLayoutsDialog::_update_ok_disable_state).unbind(2));
	MarginContainer *mc = makevb->add_margin_child(TTR("Select existing layout:"), layout_names);
	mc->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	name = memnew(LineEdit);
	makevb->add_child(name);
	name->set_placeholder(TTR("Or enter new layout name"));
	name->set_offset(SIDE_TOP, 5);
	name->set_anchor_and_offset(SIDE_LEFT, Control::ANCHOR_BEGIN, 5);
	name->set_anchor_and_offset(SIDE_RIGHT, Control::ANCHOR_END, -5);
	name->connect(SceneStringName(gui_input), callable_mp(this, &EditorLayoutsDialog::_line_gui_input));
	name->connect(SceneStringName(focus_entered), callable_mp(this, &EditorLayoutsDialog::_deselect_layout_names));
	name->connect(SceneStringName(text_changed), callable_mp(this, &EditorLayoutsDialog::_update_ok_disable_state).unbind(1));
}

void EditorLayoutsDialog::set_name_line_enabled(bool p_enabled) {
	name->set_visible(p_enabled);
}
