/**************************************************************************/
/*  openxr_select_interaction_profile_dialog.cpp                          */
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

#include "openxr_select_interaction_profile_dialog.h"

void OpenXRSelectInteractionProfileDialog::_bind_methods() {
	ADD_SIGNAL(MethodInfo("interaction_profile_selected", PropertyInfo(Variant::STRING, "interaction_profile")));
}

void OpenXRSelectInteractionProfileDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			scroll->add_theme_style_override("panel", get_theme_stylebox(SNAME("panel"), SNAME("Tree")));
		} break;
	}
}

void OpenXRSelectInteractionProfileDialog::_on_select_interaction_profile(const String p_interaction_profile) {
	if (selected_interaction_profile != "") {
		NodePath button_path = ip_buttons[selected_interaction_profile];
		Button *button = Object::cast_to<Button>(get_node(button_path));
		if (button != nullptr) {
			button->set_flat(true);
		}
	}

	selected_interaction_profile = p_interaction_profile;

	if (selected_interaction_profile != "") {
		NodePath button_path = ip_buttons[selected_interaction_profile];
		Button *button = Object::cast_to<Button>(get_node(button_path));
		if (button != nullptr) {
			button->set_flat(false);
		}
	}
}

void OpenXRSelectInteractionProfileDialog::open(PackedStringArray p_do_not_include) {
	int available_count = 0;

	// out with the old...
	while (main_vb->get_child_count() > 0) {
		memdelete(main_vb->get_child(0));
	}

	selected_interaction_profile = "";
	ip_buttons.clear();

	// in with the new
	PackedStringArray interaction_profiles = OpenXRInteractionProfileMetadata::get_singleton()->get_interaction_profile_paths();
	for (int i = 0; i < interaction_profiles.size(); i++) {
		String path = interaction_profiles[i];
		if (!p_do_not_include.has(path)) {
			Button *ip_button = memnew(Button);
			ip_button->set_flat(true);
			ip_button->set_text(OpenXRInteractionProfileMetadata::get_singleton()->get_profile(path)->display_name);
			ip_button->connect("pressed", callable_mp(this, &OpenXRSelectInteractionProfileDialog::_on_select_interaction_profile).bind(path));
			main_vb->add_child(ip_button);

			ip_buttons[path] = ip_button->get_path();
			available_count++;
		}
	}

	if (available_count == 0) {
		// give warning that we have all profiles selected

	} else {
		// TODO maybe if we only have one, auto select it?

		popup_centered();
	}
}

void OpenXRSelectInteractionProfileDialog::ok_pressed() {
	if (selected_interaction_profile == "") {
		return;
	}

	emit_signal("interaction_profile_selected", selected_interaction_profile);

	hide();
}

OpenXRSelectInteractionProfileDialog::OpenXRSelectInteractionProfileDialog() {
	set_title("Select an interaction profile");

	scroll = memnew(ScrollContainer);
	scroll->set_custom_minimum_size(Size2(600.0, 400.0));
	add_child(scroll);

	main_vb = memnew(VBoxContainer);
	// main_vb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	scroll->add_child(main_vb);
}
