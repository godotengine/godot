/*************************************************************************/
/*  action_map_editor.cpp                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "action_map_editor.h"
#include "core/input/input_map.h"
#include "core/os/keyboard.h"
#include "editor/editor_scale.h"
#include "scene/gui/center_container.h"

/////////////////////////////////////////

// Maps to 2*axis if value is neg, or 2*axis+1 if value is pos.
static const char *_joy_axis_descriptions[(size_t)JoyAxis::MAX * 2] = {
	TTRC("Left Stick Left, Joystick 0 Left"),
	TTRC("Left Stick Right, Joystick 0 Right"),
	TTRC("Left Stick Up, Joystick 0 Up"),
	TTRC("Left Stick Down, Joystick 0 Down"),
	TTRC("Right Stick Left, Joystick 1 Left"),
	TTRC("Right Stick Right, Joystick 1 Right"),
	TTRC("Right Stick Up, Joystick 1 Up"),
	TTRC("Right Stick Down, Joystick 1 Down"),
	TTRC("Joystick 2 Left"),
	TTRC("Left Trigger, Sony L2, Xbox LT, Joystick 2 Right"),
	TTRC("Joystick 2 Up"),
	TTRC("Right Trigger, Sony R2, Xbox RT, Joystick 2 Down"),
	TTRC("Joystick 3 Left"),
	TTRC("Joystick 3 Right"),
	TTRC("Joystick 3 Up"),
	TTRC("Joystick 3 Down"),
	TTRC("Joystick 4 Left"),
	TTRC("Joystick 4 Right"),
	TTRC("Joystick 4 Up"),
	TTRC("Joystick 4 Down"),
};

String InputEventConfigurationDialog::get_event_text(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND_V_MSG(p_event.is_null(), String(), "Provided event is not a valid instance of InputEvent");

	// Joypad motion events will display slightly differently than what the event->as_text() provides. See #43660.
	Ref<InputEventJoypadMotion> jpmotion = p_event;
	if (jpmotion.is_valid()) {
		String desc = TTR("Unknown Joypad Axis");
		if (jpmotion->get_axis() < JoyAxis::MAX) {
			desc = RTR(_joy_axis_descriptions[2 * (size_t)jpmotion->get_axis() + (jpmotion->get_axis_value() < 0 ? 0 : 1)]);
		}

		return vformat("Joypad Axis %s %s (%s)", itos((int64_t)jpmotion->get_axis()), jpmotion->get_axis_value() < 0 ? "-" : "+", desc);
	} else {
		return p_event->as_text();
	}
}

void InputEventConfigurationDialog::_set_event(const Ref<InputEvent> &p_event) {
	if (p_event.is_valid()) {
		event = p_event;

		// Update Label
		event_as_text->set_text(get_event_text(event));

		Ref<InputEventKey> k = p_event;
		Ref<InputEventMouseButton> mb = p_event;
		Ref<InputEventJoypadButton> joyb = p_event;
		Ref<InputEventJoypadMotion> joym = p_event;
		Ref<InputEventWithModifiers> mod = p_event;

		// Update option values and visibility
		bool show_mods = false;
		bool show_device = false;
		bool show_phys_key = false;

		if (mod.is_valid()) {
			show_mods = true;
			mod_checkboxes[MOD_ALT]->set_pressed(mod->is_alt_pressed());
			mod_checkboxes[MOD_SHIFT]->set_pressed(mod->is_shift_pressed());
			mod_checkboxes[MOD_COMMAND]->set_pressed(mod->is_command_pressed());
			mod_checkboxes[MOD_CTRL]->set_pressed(mod->is_ctrl_pressed());
			mod_checkboxes[MOD_META]->set_pressed(mod->is_meta_pressed());

			store_command_checkbox->set_pressed(mod->is_storing_command());
		}

		if (k.is_valid()) {
			show_phys_key = true;
			physical_key_checkbox->set_pressed(k->get_physical_keycode() != Key::NONE && k->get_keycode() == Key::NONE);

		} else if (joyb.is_valid() || joym.is_valid() || mb.is_valid()) {
			show_device = true;
			_set_current_device(event->get_device());
		}

		mod_container->set_visible(show_mods);
		device_container->set_visible(show_device);
		physical_key_checkbox->set_visible(show_phys_key);
		additional_options_container->show();

		// Update selected item in input list.
		if (k.is_valid() || joyb.is_valid() || joym.is_valid() || mb.is_valid()) {
			TreeItem *category = input_list_tree->get_root()->get_first_child();
			while (category) {
				TreeItem *input_item = category->get_first_child();

				if (input_item != nullptr) {
					// has_type this should be always true, unless the tree structure has been misconfigured.
					bool has_type = input_item->get_parent()->has_meta("__type");
					int input_type = input_item->get_parent()->get_meta("__type");
					if (!has_type) {
						return;
					}

					// If event type matches input types of this category.
					if ((k.is_valid() && input_type == INPUT_KEY) || (joyb.is_valid() && input_type == INPUT_JOY_BUTTON) || (joym.is_valid() && input_type == INPUT_JOY_MOTION) || (mb.is_valid() && input_type == INPUT_MOUSE_BUTTON)) {
						// Loop through all items of this category until one matches.
						while (input_item) {
							bool key_match = k.is_valid() && (Variant(k->get_keycode()) == input_item->get_meta("__keycode") || Variant(k->get_physical_keycode()) == input_item->get_meta("__keycode"));
							bool joyb_match = joyb.is_valid() && Variant(joyb->get_button_index()) == input_item->get_meta("__index");
							bool joym_match = joym.is_valid() && Variant(joym->get_axis()) == input_item->get_meta("__axis") && joym->get_axis_value() == (float)input_item->get_meta("__value");
							bool mb_match = mb.is_valid() && Variant(mb->get_button_index()) == input_item->get_meta("__index");
							if (key_match || joyb_match || joym_match || mb_match) {
								category->set_collapsed(false);
								input_item->select(0);
								input_list_tree->ensure_cursor_is_visible();
								return;
							}
							input_item = input_item->get_next();
						}
					}
				}

				category->set_collapsed(true); // Event not in this category, so collapse;
				category = category->get_next();
			}
		}
	} else {
		// Event is not valid, reset dialog
		event = p_event;
		Vector<String> strings;

		// Reset message, promp for input according to which input types are allowed.
		String text = TTR("Perform an Input (%s).");

		if (allowed_input_types & INPUT_KEY) {
			strings.append(TTR("Key"));
		}

		if (allowed_input_types & INPUT_JOY_BUTTON) {
			strings.append(TTR("Joypad Button"));
		}
		if (allowed_input_types & INPUT_JOY_MOTION) {
			strings.append(TTR("Joypad Axis"));
		}
		if (allowed_input_types & INPUT_MOUSE_BUTTON) {
			strings.append(TTR("Mouse Button in area below"));
		}
		if (strings.size() == 0) {
			text = TTR("Input Event dialog has been misconfigured: No input types are allowed.");
			event_as_text->set_text(text);
		} else {
			String insert_text = String(", ").join(strings);
			event_as_text->set_text(vformat(text, insert_text));
		}

		additional_options_container->hide();
		input_list_tree->deselect_all();
		_update_input_list();
	}
}

void InputEventConfigurationDialog::_tab_selected(int p_tab) {
	Callable signal_method = callable_mp(this, &InputEventConfigurationDialog::_listen_window_input);
	if (p_tab == 0) {
		// Start Listening.
		if (!is_connected("window_input", signal_method)) {
			connect("window_input", signal_method);
		}
	} else {
		// Stop Listening.
		if (is_connected("window_input", signal_method)) {
			disconnect("window_input", signal_method);
		}
		input_list_tree->call_deferred(SNAME("ensure_cursor_is_visible"));
		if (input_list_tree->get_selected() == nullptr) {
			// If nothing selected, scroll to top.
			input_list_tree->scroll_to_item(input_list_tree->get_root());
		}
	}
}

void InputEventConfigurationDialog::_listen_window_input(const Ref<InputEvent> &p_event) {
	// Ignore if echo or not pressed
	if (p_event->is_echo() || !p_event->is_pressed()) {
		return;
	}

	// Ignore mouse motion
	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid()) {
		return;
	}

	// Ignore mouse button if not in the detection rect
	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid()) {
		Rect2 r = mouse_detection_rect->get_rect();
		if (!r.has_point(mouse_detection_rect->get_local_mouse_position() + r.get_position())) {
			return;
		}
	}

	// Check what the type is and if it is allowed.
	Ref<InputEventKey> k = p_event;
	Ref<InputEventJoypadButton> joyb = p_event;
	Ref<InputEventJoypadMotion> joym = p_event;

	int type = 0;
	if (k.is_valid()) {
		type = INPUT_KEY;
	} else if (joyb.is_valid()) {
		type = INPUT_JOY_BUTTON;
	} else if (joym.is_valid()) {
		type = INPUT_JOY_MOTION;
	} else if (mb.is_valid()) {
		type = INPUT_MOUSE_BUTTON;
	}

	if (!(allowed_input_types & type)) {
		return;
	}

	if (joym.is_valid()) {
		float axis_value = joym->get_axis_value();
		if (ABS(axis_value) < 0.9) {
			// Ignore motion below 0.9 magnitude to avoid accidental touches
			return;
		} else {
			// Always make the value 1 or -1 for display consistency
			joym->set_axis_value(SIGN(axis_value));
		}
	}

	if (k.is_valid()) {
		k->set_pressed(false); // to avoid serialisation of 'pressed' property - doesn't matter for actions anyway.
		// Maintain physical keycode option state
		if (physical_key_checkbox->is_pressed()) {
			k->set_keycode(Key::NONE);
		} else {
			k->set_physical_keycode(Key::NONE);
		}
	}

	Ref<InputEventWithModifiers> mod = p_event;
	if (mod.is_valid()) {
		// Maintain store command option state
		mod->set_store_command(store_command_checkbox->is_pressed());

		mod->set_window_id(0);
	}

	_set_event(p_event);
	set_input_as_handled();
}

void InputEventConfigurationDialog::_search_term_updated(const String &) {
	_update_input_list();
}

void InputEventConfigurationDialog::_update_input_list() {
	input_list_tree->clear();

	TreeItem *root = input_list_tree->create_item();
	String search_term = input_list_search->get_text();

	bool collapse = input_list_search->get_text().is_empty();

	if (allowed_input_types & INPUT_KEY) {
		TreeItem *kb_root = input_list_tree->create_item(root);
		kb_root->set_text(0, TTR("Keyboard Keys"));
		kb_root->set_icon(0, icon_cache.keyboard);
		kb_root->set_collapsed(collapse);
		kb_root->set_meta("__type", INPUT_KEY);

		for (int i = 0; i < keycode_get_count(); i++) {
			String name = keycode_get_name_by_index(i);

			if (!search_term.is_empty() && name.findn(search_term) == -1) {
				continue;
			}

			TreeItem *item = input_list_tree->create_item(kb_root);
			item->set_text(0, name);
			item->set_meta("__keycode", keycode_get_value_by_index(i));
		}
	}

	if (allowed_input_types & INPUT_MOUSE_BUTTON) {
		TreeItem *mouse_root = input_list_tree->create_item(root);
		mouse_root->set_text(0, TTR("Mouse Buttons"));
		mouse_root->set_icon(0, icon_cache.mouse);
		mouse_root->set_collapsed(collapse);
		mouse_root->set_meta("__type", INPUT_MOUSE_BUTTON);

		MouseButton mouse_buttons[9] = { MouseButton::LEFT, MouseButton::RIGHT, MouseButton::MIDDLE, MouseButton::WHEEL_UP, MouseButton::WHEEL_DOWN, MouseButton::WHEEL_LEFT, MouseButton::WHEEL_RIGHT, MouseButton::MB_XBUTTON1, MouseButton::MB_XBUTTON2 };
		for (int i = 0; i < 9; i++) {
			Ref<InputEventMouseButton> mb;
			mb.instantiate();
			mb->set_button_index(mouse_buttons[i]);
			String desc = get_event_text(mb);

			if (!search_term.is_empty() && desc.findn(search_term) == -1) {
				continue;
			}

			TreeItem *item = input_list_tree->create_item(mouse_root);
			item->set_text(0, desc);
			item->set_meta("__index", mouse_buttons[i]);
		}
	}

	if (allowed_input_types & INPUT_JOY_BUTTON) {
		TreeItem *joyb_root = input_list_tree->create_item(root);
		joyb_root->set_text(0, TTR("Joypad Buttons"));
		joyb_root->set_icon(0, icon_cache.joypad_button);
		joyb_root->set_collapsed(collapse);
		joyb_root->set_meta("__type", INPUT_JOY_BUTTON);

		for (int i = 0; i < (int)JoyButton::MAX; i++) {
			Ref<InputEventJoypadButton> joyb;
			joyb.instantiate();
			joyb->set_button_index((JoyButton)i);
			String desc = get_event_text(joyb);

			if (!search_term.is_empty() && desc.findn(search_term) == -1) {
				continue;
			}

			TreeItem *item = input_list_tree->create_item(joyb_root);
			item->set_text(0, desc);
			item->set_meta("__index", i);
		}
	}

	if (allowed_input_types & INPUT_JOY_MOTION) {
		TreeItem *joya_root = input_list_tree->create_item(root);
		joya_root->set_text(0, TTR("Joypad Axes"));
		joya_root->set_icon(0, icon_cache.joypad_axis);
		joya_root->set_collapsed(collapse);
		joya_root->set_meta("__type", INPUT_JOY_MOTION);

		for (int i = 0; i < (int)JoyAxis::MAX * 2; i++) {
			int axis = i / 2;
			int direction = (i & 1) ? 1 : -1;
			Ref<InputEventJoypadMotion> joym;
			joym.instantiate();
			joym->set_axis((JoyAxis)axis);
			joym->set_axis_value(direction);
			String desc = get_event_text(joym);

			if (!search_term.is_empty() && desc.findn(search_term) == -1) {
				continue;
			}

			TreeItem *item = input_list_tree->create_item(joya_root);
			item->set_text(0, desc);
			item->set_meta("__axis", i >> 1);
			item->set_meta("__value", (i & 1) ? 1 : -1);
		}
	}
}

void InputEventConfigurationDialog::_mod_toggled(bool p_checked, int p_index) {
	Ref<InputEventWithModifiers> ie = event;

	// Not event with modifiers
	if (ie.is_null()) {
		return;
	}

	if (p_index == 0) {
		ie->set_alt_pressed(p_checked);
	} else if (p_index == 1) {
		ie->set_shift_pressed(p_checked);
	} else if (p_index == 2) {
		ie->set_command_pressed(p_checked);
	} else if (p_index == 3) {
		ie->set_ctrl_pressed(p_checked);
	} else if (p_index == 4) {
		ie->set_meta_pressed(p_checked);
	}

	_set_event(ie);
}

void InputEventConfigurationDialog::_store_command_toggled(bool p_checked) {
	Ref<InputEventWithModifiers> ie = event;
	if (ie.is_valid()) {
		ie->set_store_command(p_checked);
		_set_event(ie);
	}

	if (p_checked) {
		// If storing Command, show it's checkbox and hide Control (Win/Lin) or Meta (Mac)
#ifdef APPLE_STYLE_KEYS
		mod_checkboxes[MOD_META]->hide();

		mod_checkboxes[MOD_COMMAND]->show();
		mod_checkboxes[MOD_COMMAND]->set_text("Meta (Command)");
#else
		mod_checkboxes[MOD_CTRL]->hide();

		mod_checkboxes[MOD_COMMAND]->show();
		mod_checkboxes[MOD_COMMAND]->set_text("Control (Command)");
#endif
	} else {
		// If not, hide Command, show Control and Meta.
		mod_checkboxes[MOD_COMMAND]->hide();
		mod_checkboxes[MOD_CTRL]->show();
		mod_checkboxes[MOD_META]->show();
	}
}

void InputEventConfigurationDialog::_physical_keycode_toggled(bool p_checked) {
	Ref<InputEventKey> k = event;

	if (k.is_null()) {
		return;
	}

	if (p_checked) {
		k->set_physical_keycode(k->get_keycode());
		k->set_keycode(Key::NONE);
	} else {
		k->set_keycode((Key)k->get_physical_keycode());
		k->set_physical_keycode(Key::NONE);
	}

	_set_event(k);
}

void InputEventConfigurationDialog::_input_list_item_selected() {
	TreeItem *selected = input_list_tree->get_selected();

	// Invalid tree selection - type only exists on the "category" items, which are not a valid selection.
	if (selected->has_meta("__type")) {
		return;
	}

	InputEventConfigurationDialog::InputType input_type = (InputEventConfigurationDialog::InputType)(int)selected->get_parent()->get_meta("__type");

	switch (input_type) {
		case InputEventConfigurationDialog::INPUT_KEY: {
			Key keycode = (Key)(int)selected->get_meta("__keycode");
			Ref<InputEventKey> k;
			k.instantiate();

			if (physical_key_checkbox->is_pressed()) {
				k->set_physical_keycode(keycode);
				k->set_keycode(Key::NONE);
			} else {
				k->set_physical_keycode(Key::NONE);
				k->set_keycode(keycode);
			}

			// Maintain modifier state from checkboxes
			k->set_alt_pressed(mod_checkboxes[MOD_ALT]->is_pressed());
			k->set_shift_pressed(mod_checkboxes[MOD_SHIFT]->is_pressed());
			k->set_command_pressed(mod_checkboxes[MOD_COMMAND]->is_pressed());
			k->set_ctrl_pressed(mod_checkboxes[MOD_CTRL]->is_pressed());
			k->set_meta_pressed(mod_checkboxes[MOD_META]->is_pressed());
			k->set_store_command(store_command_checkbox->is_pressed());

			_set_event(k);
		} break;
		case InputEventConfigurationDialog::INPUT_MOUSE_BUTTON: {
			MouseButton idx = (MouseButton)(int)selected->get_meta("__index");
			Ref<InputEventMouseButton> mb;
			mb.instantiate();
			mb->set_button_index(idx);
			// Maintain modifier state from checkboxes
			mb->set_alt_pressed(mod_checkboxes[MOD_ALT]->is_pressed());
			mb->set_shift_pressed(mod_checkboxes[MOD_SHIFT]->is_pressed());
			mb->set_command_pressed(mod_checkboxes[MOD_COMMAND]->is_pressed());
			mb->set_ctrl_pressed(mod_checkboxes[MOD_CTRL]->is_pressed());
			mb->set_meta_pressed(mod_checkboxes[MOD_META]->is_pressed());
			mb->set_store_command(store_command_checkbox->is_pressed());

			_set_event(mb);
		} break;
		case InputEventConfigurationDialog::INPUT_JOY_BUTTON: {
			JoyButton idx = (JoyButton)(int)selected->get_meta("__index");
			Ref<InputEventJoypadButton> jb = InputEventJoypadButton::create_reference(idx);
			_set_event(jb);
		} break;
		case InputEventConfigurationDialog::INPUT_JOY_MOTION: {
			JoyAxis axis = (JoyAxis)(int)selected->get_meta("__axis");
			int value = selected->get_meta("__value");

			Ref<InputEventJoypadMotion> jm;
			jm.instantiate();
			jm->set_axis(axis);
			jm->set_axis_value(value);
			_set_event(jm);
		} break;
	}
}

void InputEventConfigurationDialog::_set_current_device(int i_device) {
	device_id_option->select(i_device + 1);
}

int InputEventConfigurationDialog::_get_current_device() const {
	return device_id_option->get_selected() - 1;
}

String InputEventConfigurationDialog::_get_device_string(int i_device) const {
	if (i_device == InputMap::ALL_DEVICES) {
		return TTR("All Devices");
	}
	return TTR("Device") + " " + itos(i_device);
}

void InputEventConfigurationDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			input_list_search->set_right_icon(input_list_search->get_theme_icon(SNAME("Search"), SNAME("EditorIcons")));

			physical_key_checkbox->set_icon(get_theme_icon(SNAME("KeyboardPhysical"), SNAME("EditorIcons")));

			icon_cache.keyboard = get_theme_icon(SNAME("Keyboard"), SNAME("EditorIcons"));
			icon_cache.mouse = get_theme_icon(SNAME("Mouse"), SNAME("EditorIcons"));
			icon_cache.joypad_button = get_theme_icon(SNAME("JoyButton"), SNAME("EditorIcons"));
			icon_cache.joypad_axis = get_theme_icon(SNAME("JoyAxis"), SNAME("EditorIcons"));

			mouse_detection_rect->set_color(get_theme_color(SNAME("dark_color_2"), SNAME("Editor")));

			_update_input_list();
		} break;
		default:
			break;
	}
}

void InputEventConfigurationDialog::popup_and_configure(const Ref<InputEvent> &p_event) {
	if (p_event.is_valid()) {
		_set_event(p_event);
	} else {
		// Clear Event
		_set_event(p_event);

		// Clear Checkbox Values
		for (int i = 0; i < MOD_MAX; i++) {
			mod_checkboxes[i]->set_pressed(false);
		}

		// Enable the Physical Key checkbox by default to encourage its use.
		// Physical Key should be used for most game inputs as it allows keys to work
		// on non-QWERTY layouts out of the box.
		// This is especially important for WASD movement layouts.
		physical_key_checkbox->set_pressed(true);

		store_command_checkbox->set_pressed(true);
		_set_current_device(0);

		// Switch to "Listen" tab
		tab_container->set_current_tab(0);
	}

	popup_centered();
}

Ref<InputEvent> InputEventConfigurationDialog::get_event() const {
	return event;
}

void InputEventConfigurationDialog::set_allowed_input_types(int p_type_masks) {
	allowed_input_types = p_type_masks;
}

InputEventConfigurationDialog::InputEventConfigurationDialog() {
	allowed_input_types = INPUT_KEY | INPUT_MOUSE_BUTTON | INPUT_JOY_BUTTON | INPUT_JOY_MOTION | INPUT_MOUSE_BUTTON;

	set_title(TTR("Event Configuration"));
	set_min_size(Size2i(550 * EDSCALE, 0)); // Min width

	VBoxContainer *main_vbox = memnew(VBoxContainer);
	add_child(main_vbox);

	tab_container = memnew(TabContainer);
	tab_container->set_tab_alignment(TabContainer::ALIGNMENT_LEFT);
	tab_container->set_use_hidden_tabs_for_min_size(true);
	tab_container->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	tab_container->connect("tab_selected", callable_mp(this, &InputEventConfigurationDialog::_tab_selected));
	main_vbox->add_child(tab_container);

	// Listen to input tab
	VBoxContainer *vb = memnew(VBoxContainer);
	vb->set_name(TTR("Listen for Input"));
	event_as_text = memnew(Label);
	event_as_text->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	vb->add_child(event_as_text);
	// Mouse button detection rect (Mouse button event outside this ColorRect will be ignored)
	mouse_detection_rect = memnew(ColorRect);
	mouse_detection_rect->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	vb->add_child(mouse_detection_rect);
	tab_container->add_child(vb);

	// List of all input options to manually select from.

	VBoxContainer *manual_vbox = memnew(VBoxContainer);
	manual_vbox->set_name(TTR("Manual Selection"));
	manual_vbox->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	tab_container->add_child(manual_vbox);

	input_list_search = memnew(LineEdit);
	input_list_search->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	input_list_search->set_placeholder(TTR("Filter Inputs"));
	input_list_search->set_clear_button_enabled(true);
	input_list_search->connect("text_changed", callable_mp(this, &InputEventConfigurationDialog::_search_term_updated));
	manual_vbox->add_child(input_list_search);

	input_list_tree = memnew(Tree);
	input_list_tree->set_custom_minimum_size(Size2(0, 100 * EDSCALE)); // Min height for tree
	input_list_tree->connect("item_selected", callable_mp(this, &InputEventConfigurationDialog::_input_list_item_selected));
	input_list_tree->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	manual_vbox->add_child(input_list_tree);

	input_list_tree->set_hide_root(true);
	input_list_tree->set_columns(1);

	_update_input_list();

	// Additional Options
	additional_options_container = memnew(VBoxContainer);
	additional_options_container->hide();

	Label *opts_label = memnew(Label);
	opts_label->set_theme_type_variation("HeaderSmall");
	opts_label->set_text(TTR("Additional Options"));
	additional_options_container->add_child(opts_label);

	// Device Selection
	device_container = memnew(HBoxContainer);
	device_container->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	Label *device_label = memnew(Label);
	device_label->set_theme_type_variation("HeaderSmall");
	device_label->set_text(TTR("Device:"));
	device_container->add_child(device_label);

	device_id_option = memnew(OptionButton);
	device_id_option->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	device_container->add_child(device_id_option);

	for (int i = -1; i < 8; i++) {
		device_id_option->add_item(_get_device_string(i));
	}
	_set_current_device(0);
	device_container->hide();
	additional_options_container->add_child(device_container);

	// Modifier Selection
	mod_container = memnew(HBoxContainer);
	for (int i = 0; i < MOD_MAX; i++) {
		String name = mods[i];
		mod_checkboxes[i] = memnew(CheckBox);
		mod_checkboxes[i]->connect("toggled", callable_mp(this, &InputEventConfigurationDialog::_mod_toggled), varray(i));
		mod_checkboxes[i]->set_text(name);
		mod_container->add_child(mod_checkboxes[i]);
	}

	mod_container->add_child(memnew(VSeparator));

	store_command_checkbox = memnew(CheckBox);
	store_command_checkbox->connect("toggled", callable_mp(this, &InputEventConfigurationDialog::_store_command_toggled));
	store_command_checkbox->set_pressed(true);
	store_command_checkbox->set_text(TTR("Store Command"));
#ifdef APPLE_STYLE_KEYS
	store_command_checkbox->set_tooltip(TTR("Toggles between serializing 'command' and 'meta'. Used for compatibility with Windows/Linux style keyboard."));
#else
	store_command_checkbox->set_tooltip(TTR("Toggles between serializing 'command' and 'control'. Used for compatibility with Apple Style keyboards."));
#endif
	mod_container->add_child(store_command_checkbox);

	mod_container->hide();
	additional_options_container->add_child(mod_container);

	// Physical Key Checkbox

	physical_key_checkbox = memnew(CheckBox);
	physical_key_checkbox->set_text(TTR("Use Physical Keycode"));
	physical_key_checkbox->set_tooltip(TTR("Stores the physical position of the key on the keyboard rather than the key's value. Used for compatibility with non-latin layouts.\nThis should generally be enabled for most game shortcuts, but not in non-game applications."));
	physical_key_checkbox->connect("toggled", callable_mp(this, &InputEventConfigurationDialog::_physical_keycode_toggled));
	physical_key_checkbox->hide();
	additional_options_container->add_child(physical_key_checkbox);

	main_vbox->add_child(additional_options_container);

	// Default to first tab
	tab_container->set_current_tab(0);
}

/////////////////////////////////////////

static bool _is_action_name_valid(const String &p_name) {
	const char32_t *cstr = p_name.get_data();
	for (int i = 0; cstr[i]; i++) {
		if (cstr[i] == '/' || cstr[i] == ':' || cstr[i] == '"' ||
				cstr[i] == '=' || cstr[i] == '\\' || cstr[i] < 32) {
			return false;
		}
	}
	return true;
}

void ActionMapEditor::_event_config_confirmed() {
	Ref<InputEvent> ev = event_config_dialog->get_event();

	Dictionary new_action = current_action.duplicate();
	Array events = new_action["events"];

	if (current_action_event_index == -1) {
		// Add new event
		events.push_back(ev);
	} else {
		// Edit existing event
		events[current_action_event_index] = ev;
	}

	new_action["events"] = events;
	emit_signal(SNAME("action_edited"), current_action_name, new_action);
}

void ActionMapEditor::_add_action_pressed() {
	_add_action(add_edit->get_text());
}

bool ActionMapEditor::_has_action(const String &p_name) const {
	for (const ActionInfo &action_info : actions_cache) {
		if (p_name == action_info.name) {
			return true;
		}
	}
	return false;
}

void ActionMapEditor::_add_action(const String &p_name) {
	if (p_name.is_empty() || !_is_action_name_valid(p_name)) {
		show_message(TTR("Invalid action name. It cannot be empty nor contain '/', ':', '=', '\\' or '\"'"));
		return;
	}

	if (_has_action(p_name)) {
		show_message(vformat(TTR("An action with the name '%s' already exists."), p_name));
		return;
	}

	add_edit->clear();
	emit_signal(SNAME("action_added"), p_name);
}

void ActionMapEditor::_action_edited() {
	TreeItem *ti = action_tree->get_edited();
	if (!ti) {
		return;
	}

	if (action_tree->get_selected_column() == 0) {
		// Name Edited
		String new_name = ti->get_text(0);
		String old_name = ti->get_meta("__name");

		if (new_name == old_name) {
			return;
		}

		if (new_name.is_empty() || !_is_action_name_valid(new_name)) {
			ti->set_text(0, old_name);
			show_message(TTR("Invalid action name. It cannot be empty nor contain '/', ':', '=', '\\' or '\"'"));
			return;
		}

		if (_has_action(new_name)) {
			ti->set_text(0, old_name);
			show_message(vformat(TTR("An action with the name '%s' already exists."), new_name));
			return;
		}

		emit_signal(SNAME("action_renamed"), old_name, new_name);
	} else if (action_tree->get_selected_column() == 1) {
		// Deadzone Edited
		String name = ti->get_meta("__name");
		Dictionary old_action = ti->get_meta("__action");
		Dictionary new_action = old_action.duplicate();
		new_action["deadzone"] = ti->get_range(1);

		// Call deferred so that input can finish propagating through tree, allowing re-making of tree to occur.
		call_deferred(SNAME("emit_signal"), "action_edited", name, new_action);
	}
}

void ActionMapEditor::_tree_button_pressed(Object *p_item, int p_column, int p_id) {
	ItemButton option = (ItemButton)p_id;

	TreeItem *item = Object::cast_to<TreeItem>(p_item);
	if (!item) {
		return;
	}

	switch (option) {
		case ActionMapEditor::BUTTON_ADD_EVENT: {
			current_action = item->get_meta("__action");
			current_action_name = item->get_meta("__name");
			current_action_event_index = -1;

			event_config_dialog->popup_and_configure();

		} break;
		case ActionMapEditor::BUTTON_EDIT_EVENT: {
			// Action and Action name is located on the parent of the event.
			current_action = item->get_parent()->get_meta("__action");
			current_action_name = item->get_parent()->get_meta("__name");

			current_action_event_index = item->get_meta("__index");

			Ref<InputEvent> ie = item->get_meta("__event");
			if (ie.is_valid()) {
				event_config_dialog->popup_and_configure(ie);
			}

		} break;
		case ActionMapEditor::BUTTON_REMOVE_ACTION: {
			// Send removed action name
			String name = item->get_meta("__name");
			emit_signal(SNAME("action_removed"), name);
		} break;
		case ActionMapEditor::BUTTON_REMOVE_EVENT: {
			// Remove event and send updated action
			Dictionary action = item->get_parent()->get_meta("__action");
			String action_name = item->get_parent()->get_meta("__name");

			int event_index = item->get_meta("__index");

			Array events = action["events"];
			events.remove_at(event_index);
			action["events"] = events;

			emit_signal(SNAME("action_edited"), action_name, action);
		} break;
		default:
			break;
	}
}

void ActionMapEditor::_tree_item_activated() {
	TreeItem *item = action_tree->get_selected();

	if (!item || !item->has_meta("__event")) {
		return;
	}

	_tree_button_pressed(item, 2, BUTTON_EDIT_EVENT);
}

void ActionMapEditor::set_show_builtin_actions(bool p_show) {
	show_builtin_actions = p_show;
	show_builtin_actions_checkbutton->set_pressed(p_show);

	// Prevent unnecessary updates of action list when cache is empty.
	if (!actions_cache.is_empty()) {
		update_action_list();
	}
}

void ActionMapEditor::_search_term_updated(const String &) {
	update_action_list();
}

Variant ActionMapEditor::get_drag_data_fw(const Point2 &p_point, Control *p_from) {
	TreeItem *selected = action_tree->get_selected();
	if (!selected) {
		return Variant();
	}

	String name = selected->get_text(0);
	Label *label = memnew(Label(name));
	label->set_theme_type_variation("HeaderSmall");
	label->set_modulate(Color(1, 1, 1, 1.0f));
	action_tree->set_drag_preview(label);

	Dictionary drag_data;

	if (selected->has_meta("__action")) {
		drag_data["input_type"] = "action";
	}

	if (selected->has_meta("__event")) {
		drag_data["input_type"] = "event";
	}

	action_tree->set_drop_mode_flags(Tree::DROP_MODE_INBETWEEN);

	return drag_data;
}

bool ActionMapEditor::can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const {
	Dictionary d = p_data;
	if (!d.has("input_type")) {
		return false;
	}

	TreeItem *selected = action_tree->get_selected();
	TreeItem *item = action_tree->get_item_at_position(p_point);
	if (!selected || !item || item == selected) {
		return false;
	}

	// Don't allow moving an action in-between events.
	if (d["input_type"] == "action" && item->has_meta("__event")) {
		return false;
	}

	// Don't allow moving an event to a different action.
	if (d["input_type"] == "event" && item->get_parent() != selected->get_parent()) {
		return false;
	}

	return true;
}

void ActionMapEditor::drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {
	if (!can_drop_data_fw(p_point, p_data, p_from)) {
		return;
	}

	TreeItem *selected = action_tree->get_selected();
	TreeItem *target = action_tree->get_item_at_position(p_point);
	bool drop_above = action_tree->get_drop_section_at_position(p_point) == -1;

	if (!target) {
		return;
	}

	Dictionary d = p_data;
	if (d["input_type"] == "action") {
		// Change action order.
		String relative_to = target->get_meta("__name");
		String action_name = selected->get_meta("__name");
		emit_signal(SNAME("action_reordered"), action_name, relative_to, drop_above);

	} else if (d["input_type"] == "event") {
		// Change event order
		int current_index = selected->get_meta("__index");
		int target_index = target->get_meta("__index");

		// Construct new events array.
		Dictionary new_action = selected->get_parent()->get_meta("__action");

		Array events = new_action["events"];
		Array new_events;

		// The following method was used to perform the array changes since `remove` followed by `insert` was not working properly at time of writing.
		// Loop thought existing events
		for (int i = 0; i < events.size(); i++) {
			// If you come across the current index, just skip it, as it has been moved.
			if (i == current_index) {
				continue;
			} else if (i == target_index) {
				// We are at the target index. If drop above, add selected event there first, then target, so moved event goes on top.
				if (drop_above) {
					new_events.push_back(events[current_index]);
					new_events.push_back(events[target_index]);
				} else {
					new_events.push_back(events[target_index]);
					new_events.push_back(events[current_index]);
				}
			} else {
				new_events.push_back(events[i]);
			}
		}

		new_action["events"] = new_events;
		emit_signal(SNAME("action_edited"), selected->get_parent()->get_meta("__name"), new_action);
	}
}

void ActionMapEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			action_list_search->set_right_icon(get_theme_icon(SNAME("Search"), SNAME("EditorIcons")));
		} break;
		default:
			break;
	}
}

void ActionMapEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_get_drag_data_fw"), &ActionMapEditor::get_drag_data_fw);
	ClassDB::bind_method(D_METHOD("_can_drop_data_fw"), &ActionMapEditor::can_drop_data_fw);
	ClassDB::bind_method(D_METHOD("_drop_data_fw"), &ActionMapEditor::drop_data_fw);

	ADD_SIGNAL(MethodInfo("action_added", PropertyInfo(Variant::STRING, "name")));
	ADD_SIGNAL(MethodInfo("action_edited", PropertyInfo(Variant::STRING, "name"), PropertyInfo(Variant::DICTIONARY, "new_action")));
	ADD_SIGNAL(MethodInfo("action_removed", PropertyInfo(Variant::STRING, "name")));
	ADD_SIGNAL(MethodInfo("action_renamed", PropertyInfo(Variant::STRING, "old_name"), PropertyInfo(Variant::STRING, "new_name")));
	ADD_SIGNAL(MethodInfo("action_reordered", PropertyInfo(Variant::STRING, "action_name"), PropertyInfo(Variant::STRING, "relative_to"), PropertyInfo(Variant::BOOL, "before")));
}

LineEdit *ActionMapEditor::get_search_box() const {
	return action_list_search;
}

InputEventConfigurationDialog *ActionMapEditor::get_configuration_dialog() {
	return event_config_dialog;
}

void ActionMapEditor::update_action_list(const Vector<ActionInfo> &p_action_infos) {
	if (!p_action_infos.is_empty()) {
		actions_cache = p_action_infos;
	}

	action_tree->clear();
	TreeItem *root = action_tree->create_item();

	int uneditable_count = 0;

	for (int i = 0; i < actions_cache.size(); i++) {
		ActionInfo action_info = actions_cache[i];

		if (!action_info.editable) {
			uneditable_count++;
		}

		String search_term = action_list_search->get_text();
		if (!search_term.is_empty() && action_info.name.findn(search_term) == -1) {
			continue;
		}

		if (!action_info.editable && !show_builtin_actions) {
			continue;
		}

		const Array events = action_info.action["events"];
		const Variant deadzone = action_info.action["deadzone"];

		// Update Tree...

		TreeItem *action_item = action_tree->create_item(root);
		action_item->set_meta("__action", action_info.action);
		action_item->set_meta("__name", action_info.name);

		// First Column - Action Name
		action_item->set_text(0, action_info.name);
		action_item->set_editable(0, action_info.editable);
		action_item->set_icon(0, action_info.icon);

		// Second Column - Deadzone
		action_item->set_editable(1, true);
		action_item->set_cell_mode(1, TreeItem::CELL_MODE_RANGE);
		action_item->set_range_config(1, 0.0, 1.0, 0.01);
		action_item->set_range(1, deadzone);

		// Third column - buttons
		action_item->add_button(2, action_tree->get_theme_icon(SNAME("Add"), SNAME("EditorIcons")), BUTTON_ADD_EVENT, false, TTR("Add Event"));
		action_item->add_button(2, action_tree->get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")), BUTTON_REMOVE_ACTION, !action_info.editable, action_info.editable ? TTR("Remove Action") : TTR("Cannot Remove Action"));

		action_item->set_custom_bg_color(0, action_tree->get_theme_color(SNAME("prop_subsection"), SNAME("Editor")));
		action_item->set_custom_bg_color(1, action_tree->get_theme_color(SNAME("prop_subsection"), SNAME("Editor")));

		for (int evnt_idx = 0; evnt_idx < events.size(); evnt_idx++) {
			Ref<InputEvent> event = events[evnt_idx];
			if (event.is_null()) {
				continue;
			}

			TreeItem *event_item = action_tree->create_item(action_item);

			// First Column - Text
			event_item->set_text(0, event_config_dialog->get_event_text(event)); // Need to us the special description for JoypadMotion here, so don't use as_text() directly.
			event_item->set_meta("__event", event);
			event_item->set_meta("__index", evnt_idx);

			// Third Column - Buttons
			event_item->add_button(2, action_tree->get_theme_icon(SNAME("Edit"), SNAME("EditorIcons")), BUTTON_EDIT_EVENT, false, TTR("Edit Event"));
			event_item->add_button(2, action_tree->get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")), BUTTON_REMOVE_EVENT, false, TTR("Remove Event"));
			event_item->set_button_color(2, 0, Color(1, 1, 1, 0.75));
			event_item->set_button_color(2, 1, Color(1, 1, 1, 0.75));
		}
	}
}

void ActionMapEditor::show_message(const String &p_message) {
	message->set_text(p_message);
	message->popup_centered(Size2(300, 100) * EDSCALE);
}

void ActionMapEditor::use_external_search_box(LineEdit *p_searchbox) {
	memdelete(action_list_search);
	action_list_search = p_searchbox;
	action_list_search->connect("text_changed", callable_mp(this, &ActionMapEditor::_search_term_updated));
}

ActionMapEditor::ActionMapEditor() {
	show_builtin_actions = false;

	// Main Vbox Container
	VBoxContainer *main_vbox = memnew(VBoxContainer);
	main_vbox->set_anchors_and_offsets_preset(PRESET_WIDE);
	add_child(main_vbox);

	HBoxContainer *top_hbox = memnew(HBoxContainer);
	main_vbox->add_child(top_hbox);

	action_list_search = memnew(LineEdit);
	action_list_search->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	action_list_search->set_placeholder(TTR("Filter Actions"));
	action_list_search->set_clear_button_enabled(true);
	action_list_search->connect("text_changed", callable_mp(this, &ActionMapEditor::_search_term_updated));
	top_hbox->add_child(action_list_search);

	show_builtin_actions_checkbutton = memnew(CheckButton);
	show_builtin_actions_checkbutton->set_pressed(false);
	show_builtin_actions_checkbutton->set_text(TTR("Show Built-in Actions"));
	show_builtin_actions_checkbutton->connect("toggled", callable_mp(this, &ActionMapEditor::set_show_builtin_actions));
	top_hbox->add_child(show_builtin_actions_checkbutton);

	// Adding Action line edit + button
	add_hbox = memnew(HBoxContainer);
	add_hbox->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	add_edit = memnew(LineEdit);
	add_edit->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	add_edit->set_placeholder(TTR("Add New Action"));
	add_edit->set_clear_button_enabled(true);
	add_edit->connect("text_submitted", callable_mp(this, &ActionMapEditor::_add_action));
	add_hbox->add_child(add_edit);

	Button *add_button = memnew(Button);
	add_button->set_text(TTR("Add"));
	add_button->connect("pressed", callable_mp(this, &ActionMapEditor::_add_action_pressed));
	add_hbox->add_child(add_button);

	main_vbox->add_child(add_hbox);

	// Action Editor Tree
	action_tree = memnew(Tree);
	action_tree->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	action_tree->set_columns(3);
	action_tree->set_hide_root(true);
	action_tree->set_column_titles_visible(true);
	action_tree->set_column_title(0, TTR("Action"));
	action_tree->set_column_clip_content(0, true);
	action_tree->set_column_title(1, TTR("Deadzone"));
	action_tree->set_column_expand(1, false);
	action_tree->set_column_custom_minimum_width(1, 80 * EDSCALE);
	action_tree->set_column_expand(2, false);
	action_tree->set_column_custom_minimum_width(2, 50 * EDSCALE);
	action_tree->connect("item_edited", callable_mp(this, &ActionMapEditor::_action_edited));
	action_tree->connect("item_activated", callable_mp(this, &ActionMapEditor::_tree_item_activated));
	action_tree->connect("button_pressed", callable_mp(this, &ActionMapEditor::_tree_button_pressed));
	main_vbox->add_child(action_tree);

	action_tree->set_drag_forwarding(this);

	// Adding event dialog
	event_config_dialog = memnew(InputEventConfigurationDialog);
	event_config_dialog->connect("confirmed", callable_mp(this, &ActionMapEditor::_event_config_confirmed));
	add_child(event_config_dialog);

	message = memnew(AcceptDialog);
	add_child(message);
}
