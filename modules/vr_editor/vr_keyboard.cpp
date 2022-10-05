/**************************************************************************/
/*  vr_keyboard.cpp                                                       */
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

#include "vr_keyboard.h"

#include "core/os/os.h"
#include "scene/gui/box_container.h"
#include "scene/gui/center_container.h"
#include "scene/gui/color_rect.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/separator.h"

void VRKeyboard::_on_key_down(String p_scan_code_text, int p_unicode) {
	ERR_FAIL_NULL(shift_button);

	Key keycode = find_keycode(p_scan_code_text);

	Ref<InputEventKey> input;
	input.instantiate();
	input->set_keycode(keycode);
	input->set_physical_keycode(keycode);
	input->set_unicode(p_unicode == 0 ? char32_t(keycode) : char32_t(p_unicode));
	input->set_pressed(true);
	input->set_shift_pressed(shift_button->is_pressed());

	// We sent this to our parent viewport
	// get_viewport()->push_input(input);

	Input::get_singleton()->parse_input_event(input);
}

void VRKeyboard::_on_key_up(String p_scan_code_text, int p_unicode) {
	ERR_FAIL_NULL(shift_button);

	Key keycode = find_keycode(p_scan_code_text);

	Ref<InputEventKey> input;
	input.instantiate();
	input->set_keycode(keycode);
	input->set_physical_keycode(keycode);
	input->set_unicode(p_unicode == 0 ? char32_t(keycode) : char32_t(p_unicode));
	input->set_pressed(false);
	input->set_shift_pressed(shift_button->is_pressed());

	// We sent this to our parent viewport
	// get_viewport()->push_input(input);

	Input::get_singleton()->parse_input_event(input);
}

void VRKeyboard::_toggle_shift() {
	// Our shift really works like a caps lock seeing we don't support multitouch well enough,
	// could probably change that by using touch buttons...
	// and then add caps in here as well.. TODO I guess
	ERR_FAIL_NULL(shift_button);

	for (int i = 0; i < update_case_on_shift.size(); i++) {
		String text = update_case_on_shift[i]->get_text();

		if (shift_button->is_pressed()) {
			text = text.to_upper();
		} else {
			text = text.to_lower();
		}
		update_case_on_shift[i]->set_text(text);
	}
}

Control *VRKeyboard::_create_keyboard_row(const char *p_keys) {
	HBoxContainer *hbox = memnew(HBoxContainer);

	int len = strlen(p_keys);
	for (int i = 0; i < len; i++) {
		Button *button = memnew(Button);

		if (memcmp(p_keys + i, "&BSP", 4) == 0) {
			button->set_name("key_BKSP");
			button->set_custom_minimum_size(Size2(50, 20));
			button->set_text("BKSP");

			button->connect(SNAME("button_down"), callable_mp(this, &VRKeyboard::_on_key_down).bind("BackSpace", 0));
			button->connect(SNAME("button_up"), callable_mp(this, &VRKeyboard::_on_key_up).bind("BackSpace", 0));

			i += 3;
		} else if (memcmp(p_keys + i, "&RET", 4) == 0) {
			button->set_name("key_Enter");
			button->set_custom_minimum_size(Size2(50, 20));
			button->set_text("Enter");

			button->connect(SNAME("button_down"), callable_mp(this, &VRKeyboard::_on_key_down).bind("Enter", 0));
			button->connect(SNAME("button_up"), callable_mp(this, &VRKeyboard::_on_key_up).bind("Enter", 0));

			i += 3;
		} else if (memcmp(p_keys + i, "&SFT", 4) == 0) {
			shift_button = button;

			button->set_name("key_Shift");
			button->set_custom_minimum_size(Size2(70, 20));
			button->set_text("Shift");
			button->set_toggle_mode(true);

			button->connect(SNAME("pressed"), callable_mp(this, &VRKeyboard::_toggle_shift));

			i += 3;
		} else if (memcmp(p_keys + i, "&SPC", 4) == 0) {
			shift_button = button;

			button->set_name("key_Space");
			button->set_custom_minimum_size(Size2(250, 20));
			button->set_text("");
			button->set_toggle_mode(true);

			button->connect(SNAME("button_down"), callable_mp(this, &VRKeyboard::_on_key_down).bind("Space", 0));
			button->connect(SNAME("button_up"), callable_mp(this, &VRKeyboard::_on_key_up).bind("Space", 0));

			i += 3;
		} else if (memcmp(p_keys + i, "&TAB", 4) == 0) {
			button->set_name("key_Tab");
			button->set_custom_minimum_size(Size2(50, 20));
			button->set_text("TAB");

			button->connect(SNAME("button_down"), callable_mp(this, &VRKeyboard::_on_key_down).bind("Tab", 0));
			button->connect(SNAME("button_up"), callable_mp(this, &VRKeyboard::_on_key_up).bind("Tab", 0));

			i += 3;
		} else {
			// Button for a single normal character
			String text;
			text += char32_t(p_keys[i]);

			button->set_name("key_" + text);
			button->set_custom_minimum_size(Size2(30, 20));
			button->set_text(text.to_lower());

			button->connect(SNAME("button_down"), callable_mp(this, &VRKeyboard::_on_key_down).bind(text, p_keys[i]));
			button->connect(SNAME("button_up"), callable_mp(this, &VRKeyboard::_on_key_up).bind(text, p_keys[i]));

			if (p_keys[i] >= 'a' && p_keys[i] <= 'z') {
				update_case_on_shift.push_back(button);
			}
		}

		hbox->add_child(button);
	}

	CenterContainer *row = memnew(CenterContainer);
	row->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	row->add_child(hbox);

	return row;
}

VRKeyboard::VRKeyboard() :
		VRWindow(Size2i(800, 250)) {
	// We want a transparent background
	set_transparent_background(true);

	// We want our keyboard flat, not upright
	mesh_instance->set_rotation(Vector3(-0.5 * Math_PI, 0.0, 0.0));

	// Create our theme base
	theme = create_custom_theme();
	Control *theme_base = memnew(Control);
	subviewport->add_child(theme_base);
	theme_base->set_theme(theme);

	// Add a panel container for a background
	PanelContainer *panel = memnew(PanelContainer);
	panel->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	panel->add_theme_style_override("panel", panel->get_theme_stylebox(SNAME("panel"), SNAME("EditorStyles")));
	theme_base->add_child(panel);

	// Generate our keyboard
	VBoxContainer *main_vbox = memnew(VBoxContainer);
	main_vbox->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);

	HSeparator *separator = memnew(HSeparator);
	main_vbox->add_child(separator);

	main_vbox->add_child(_create_keyboard_row("`1234567890-=\\&BSP"));
	main_vbox->add_child(_create_keyboard_row("&TABQWERTYUIOP[]"));
	main_vbox->add_child(_create_keyboard_row("ASDFGHJKL;'&RET"));
	main_vbox->add_child(_create_keyboard_row("&SFTZXCVBNM,./"));
	main_vbox->add_child(_create_keyboard_row("&SPC"));
	panel->add_child(main_vbox);
}

VRKeyboard::~VRKeyboard() {
}
