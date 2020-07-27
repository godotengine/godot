/*************************************************************************/
/*  input_map_editor.cpp                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "input_map_editor.h"

#include "core/input/input_map.h"
#include "core/os/keyboard.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"

static const char *_button_descriptions[JOY_SDL_BUTTONS] = {
	TTRC("Face Bottom, DualShock Cross, Xbox A, Nintendo B"),
	TTRC("Face Right, DualShock Circle, Xbox B, Nintendo A"),
	TTRC("Face Left, DualShock Square, Xbox X, Nintendo Y"),
	TTRC("Face Top, DualShock Triangle, Xbox Y, Nintendo X"),
	TTRC("DualShock Select, Xbox Back, Nintendo -"),
	TTRC("Home, DualShock PS, Guide"),
	TTRC("Start, Nintendo +"),
	TTRC("Left Stick, DualShock L3, Xbox L/LS"),
	TTRC("Right Stick, DualShock R3, Xbox R/RS"),
	TTRC("Left Shoulder, DualShock L1, Xbox LB"),
	TTRC("Right Shoulder, DualShock R1, Xbox RB"),
	TTRC("D-Pad Up"),
	TTRC("D-Pad Down"),
	TTRC("D-Pad Left"),
	TTRC("D-Pad Right")
};

static const char *_axis_descriptions[JOY_AXIS_MAX * 2] = {
	TTRC("Left Stick Left"),
	TTRC("Left Stick Right"),
	TTRC("Left Stick Up"),
	TTRC("Left Stick Down"),
	TTRC("Right Stick Left"),
	TTRC("Right Stick Right"),
	TTRC("Right Stick Up"),
	TTRC("Right Stick Down"),
	TTRC("Joystick 2 Left"),
	TTRC("Joystick 2 Right, Left Trigger, L2, LT"),
	TTRC("Joystick 2 Up"),
	TTRC("Joystick 2 Down, Right Trigger, R2, RT"),
	TTRC("Joystick 3 Left"),
	TTRC("Joystick 3 Right"),
	TTRC("Joystick 3 Up"),
	TTRC("Joystick 3 Down"),
	TTRC("Joystick 4 Left"),
	TTRC("Joystick 4 Right"),
	TTRC("Joystick 4 Up"),
	TTRC("Joystick 4 Down"),
};

void InputMapEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			action_add_error->add_theme_color_override("font_color", input_editor->get_theme_color("error_color", "Editor"));
			popup_add->add_icon_item(input_editor->get_theme_icon("Keyboard", "EditorIcons"), TTR("Key"), INPUT_KEY);
			popup_add->add_icon_item(input_editor->get_theme_icon("KeyboardPhysical", "EditorIcons"), TTR("Physical Key"), INPUT_KEY_PHYSICAL);
			popup_add->add_icon_item(input_editor->get_theme_icon("JoyButton", "EditorIcons"), TTR("Joy Button"), INPUT_JOY_BUTTON);
			popup_add->add_icon_item(input_editor->get_theme_icon("JoyAxis", "EditorIcons"), TTR("Joy Axis"), INPUT_JOY_MOTION);
			popup_add->add_icon_item(input_editor->get_theme_icon("Mouse", "EditorIcons"), TTR("Mouse Button"), INPUT_MOUSE_BUTTON);
			_update_actions();
		} break;
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			action_add_error->add_theme_color_override("font_color", input_editor->get_theme_color("error_color", "Editor"));
			popup_add->set_item_icon(popup_add->get_item_index(INPUT_KEY), input_editor->get_theme_icon("Keyboard", "EditorIcons"));
			popup_add->set_item_icon(popup_add->get_item_index(INPUT_KEY_PHYSICAL), input_editor->get_theme_icon("KeyboardPhysical", "EditorIcons"));
			popup_add->set_item_icon(popup_add->get_item_index(INPUT_JOY_BUTTON), input_editor->get_theme_icon("JoyButton", "EditorIcons"));
			popup_add->set_item_icon(popup_add->get_item_index(INPUT_JOY_MOTION), input_editor->get_theme_icon("JoyAxis", "EditorIcons"));
			popup_add->set_item_icon(popup_add->get_item_index(INPUT_MOUSE_BUTTON), input_editor->get_theme_icon("Mouse", "EditorIcons"));
			_update_actions();
		} break;
	}
}

static bool _validate_action_name(const String &p_name) {
	const char32_t *cstr = p_name.get_data();
	for (int i = 0; cstr[i]; i++) {
		if (cstr[i] == '/' || cstr[i] == ':' || cstr[i] == '"' ||
				cstr[i] == '=' || cstr[i] == '\\' || cstr[i] < 32) {
			return false;
		}
	}
	return true;
}

void InputMapEditor::_action_selected() {
	TreeItem *ti = input_editor->get_selected();
	if (!ti || !ti->is_editable(0)) {
		return;
	}

	add_at = "input/" + ti->get_text(0);
	edit_idx = -1;
}

void InputMapEditor::_action_edited() {
	TreeItem *ti = input_editor->get_selected();
	if (!ti) {
		return;
	}

	if (input_editor->get_selected_column() == 0) {
		String new_name = ti->get_text(0);
		String old_name = add_at.substr(add_at.find("/") + 1, add_at.length());

		if (new_name == old_name) {
			return;
		}

		if (new_name == "" || !_validate_action_name(new_name)) {
			ti->set_text(0, old_name);
			add_at = "input/" + old_name;

			message->set_text(TTR("Invalid action name. it cannot be empty nor contain '/', ':', '=', '\\' or '\"'"));
			message->popup_centered(Size2(300, 100) * EDSCALE);
			return;
		}

		String action_prop = "input/" + new_name;

		if (ProjectSettings::get_singleton()->has_setting(action_prop)) {
			ti->set_text(0, old_name);
			add_at = "input/" + old_name;

			message->set_text(vformat(TTR("An action with the name '%s' already exists."), new_name));
			message->popup_centered(Size2(300, 100) * EDSCALE);
			return;
		}

		int order = ProjectSettings::get_singleton()->get_order(add_at);
		Dictionary action = ProjectSettings::get_singleton()->get(add_at);

		setting = true;
		undo_redo->create_action(TTR("Rename Input Action Event"));
		undo_redo->add_do_method(ProjectSettings::get_singleton(), "clear", add_at);
		undo_redo->add_do_method(ProjectSettings::get_singleton(), "set", action_prop, action);
		undo_redo->add_do_method(ProjectSettings::get_singleton(), "set_order", action_prop, order);
		undo_redo->add_undo_method(ProjectSettings::get_singleton(), "clear", action_prop);
		undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set", add_at, action);
		undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set_order", add_at, order);
		undo_redo->add_do_method(this, "_update_actions");
		undo_redo->add_undo_method(this, "_update_actions");
		undo_redo->add_do_method(this, "emit_signal", inputmap_changed);
		undo_redo->add_undo_method(this, "emit_signal", inputmap_changed);
		undo_redo->commit_action();
		setting = false;

		add_at = action_prop;
	} else if (input_editor->get_selected_column() == 1) {
		String name = "input/" + ti->get_text(0);
		Dictionary old_action = ProjectSettings::get_singleton()->get(name);
		Dictionary new_action = old_action.duplicate();
		new_action["deadzone"] = ti->get_range(1);

		undo_redo->create_action(TTR("Change Action deadzone"));
		undo_redo->add_do_method(ProjectSettings::get_singleton(), "set", name, new_action);
		undo_redo->add_do_method(this, "emit_signal", inputmap_changed);
		undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set", name, old_action);
		undo_redo->add_undo_method(this, "emit_signal", inputmap_changed);
		undo_redo->commit_action();
	}
}

void InputMapEditor::_device_input_add() {
	Ref<InputEvent> ie;
	String name = add_at;
	int idx = edit_idx;
	Dictionary old_val = ProjectSettings::get_singleton()->get(name);
	Dictionary action = old_val.duplicate();
	Array events = action["events"];

	switch (add_type) {
		case INPUT_MOUSE_BUTTON: {
			Ref<InputEventMouseButton> mb;
			mb.instance();
			mb->set_button_index(device_index->get_selected() + 1);
			mb->set_device(_get_current_device());

			for (int i = 0; i < events.size(); i++) {
				Ref<InputEventMouseButton> aie = events[i];
				if (aie.is_null()) {
					continue;
				}
				if (aie->get_device() == mb->get_device() && aie->get_button_index() == mb->get_button_index()) {
					return;
				}
			}

			ie = mb;

		} break;
		case INPUT_JOY_MOTION: {
			Ref<InputEventJoypadMotion> jm;
			jm.instance();
			jm->set_axis(device_index->get_selected() >> 1);
			jm->set_axis_value((device_index->get_selected() & 1) ? 1 : -1);
			jm->set_device(_get_current_device());

			for (int i = 0; i < events.size(); i++) {
				Ref<InputEventJoypadMotion> aie = events[i];
				if (aie.is_null()) {
					continue;
				}

				if (aie->get_device() == jm->get_device() && aie->get_axis() == jm->get_axis() && aie->get_axis_value() == jm->get_axis_value()) {
					return;
				}
			}

			ie = jm;

		} break;
		case INPUT_JOY_BUTTON: {
			Ref<InputEventJoypadButton> jb;
			jb.instance();

			jb->set_button_index(device_index->get_selected());
			jb->set_device(_get_current_device());

			for (int i = 0; i < events.size(); i++) {
				Ref<InputEventJoypadButton> aie = events[i];
				if (aie.is_null()) {
					continue;
				}
				if (aie->get_device() == jb->get_device() && aie->get_button_index() == jb->get_button_index()) {
					return;
				}
			}
			ie = jb;

		} break;
		default: {
		}
	}

	if (idx < 0 || idx >= events.size()) {
		events.push_back(ie);
	} else {
		events[idx] = ie;
	}
	action["events"] = events;

	undo_redo->create_action(TTR("Add Input Action Event"));
	undo_redo->add_do_method(ProjectSettings::get_singleton(), "set", name, action);
	undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set", name, old_val);
	undo_redo->add_do_method(this, "_update_actions");
	undo_redo->add_undo_method(this, "_update_actions");
	undo_redo->add_do_method(this, "emit_signal", inputmap_changed);
	undo_redo->add_undo_method(this, "emit_signal", inputmap_changed);
	undo_redo->commit_action();

	_show_last_added(ie, name);
}

void InputMapEditor::_set_current_device(int i_device) {
	device_id->select(i_device + 1);
}

int InputMapEditor::_get_current_device() {
	return device_id->get_selected() - 1;
}

String InputMapEditor::_get_device_string(int i_device) {
	if (i_device == InputMap::ALL_DEVICES) {
		return TTR("All Devices");
	}
	return TTR("Device") + " " + itos(i_device);
}

void InputMapEditor::_press_a_key_confirm() {
	if (last_wait_for_key.is_null()) {
		return;
	}

	Ref<InputEventKey> ie;
	ie.instance();
	if (press_a_key_physical) {
		ie->set_physical_keycode(last_wait_for_key->get_physical_keycode());
		ie->set_keycode(0);
	} else {
		ie->set_physical_keycode(0);
		ie->set_keycode(last_wait_for_key->get_keycode());
	}
	ie->set_shift(last_wait_for_key->get_shift());
	ie->set_alt(last_wait_for_key->get_alt());
	ie->set_control(last_wait_for_key->get_control());
	ie->set_metakey(last_wait_for_key->get_metakey());

	String name = add_at;
	int idx = edit_idx;

	Dictionary old_val = ProjectSettings::get_singleton()->get(name);
	Dictionary action = old_val.duplicate();
	Array events = action["events"];

	for (int i = 0; i < events.size(); i++) {
		Ref<InputEventKey> aie = events[i];
		if (aie.is_null()) {
			continue;
		}
		if (!press_a_key_physical) {
			if (aie->get_keycode_with_modifiers() == ie->get_keycode_with_modifiers()) {
				return;
			}
		} else {
			if (aie->get_physical_keycode_with_modifiers() == ie->get_physical_keycode_with_modifiers()) {
				return;
			}
		}
	}

	if (idx < 0 || idx >= events.size()) {
		events.push_back(ie);
	} else {
		events[idx] = ie;
	}
	action["events"] = events;

	undo_redo->create_action(TTR("Add Input Action Event"));
	undo_redo->add_do_method(ProjectSettings::get_singleton(), "set", name, action);
	undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set", name, old_val);
	undo_redo->add_do_method(this, "_update_actions");
	undo_redo->add_undo_method(this, "_update_actions");
	undo_redo->add_do_method(this, "emit_signal", inputmap_changed);
	undo_redo->add_undo_method(this, "emit_signal", inputmap_changed);
	undo_redo->commit_action();

	_show_last_added(ie, name);
}

void InputMapEditor::_show_last_added(const Ref<InputEvent> &p_event, const String &p_name) {
	TreeItem *r = input_editor->get_root();

	String name = p_name;
	name.erase(0, 6);
	if (!r) {
		return;
	}
	r = r->get_children();
	if (!r) {
		return;
	}
	bool found = false;
	while (r) {
		if (r->get_text(0) != name) {
			r = r->get_next();
			continue;
		}
		TreeItem *child = r->get_children();
		while (child) {
			Variant input = child->get_meta("__input");
			if (p_event == input) {
				r->set_collapsed(false);
				child->select(0);
				found = true;
				break;
			}
			child = child->get_next();
		}
		if (found) {
			break;
		}
		r = r->get_next();
	}

	if (found) {
		input_editor->ensure_cursor_is_visible();
	}
}

void InputMapEditor::_wait_for_key(const Ref<InputEvent> &p_event) {
	Ref<InputEventKey> k = p_event;

	if (k.is_valid() && k->is_pressed() && k->get_keycode() != 0) {
		last_wait_for_key = p_event;
		const String str = (press_a_key_physical) ? keycode_get_string(k->get_physical_keycode_with_modifiers()) + TTR(" (Physical)") : keycode_get_string(k->get_keycode_with_modifiers());

		press_a_key_label->set_text(str);
		press_a_key->get_ok()->set_disabled(false);
		press_a_key->set_input_as_handled();
	}
}

void InputMapEditor::_edit_item(Ref<InputEvent> p_exiting_event) {
	InputType ie_type;

	if ((Ref<InputEventKey>(p_exiting_event)).is_valid()) {
		if ((Ref<InputEventKey>(p_exiting_event))->get_keycode() != 0) {
			ie_type = INPUT_KEY;
		} else {
			ie_type = INPUT_KEY_PHYSICAL;
		}
	} else if ((Ref<InputEventJoypadButton>(p_exiting_event)).is_valid()) {
		ie_type = INPUT_JOY_BUTTON;
	} else if ((Ref<InputEventMouseButton>(p_exiting_event)).is_valid()) {
		ie_type = INPUT_MOUSE_BUTTON;
	} else if ((Ref<InputEventJoypadMotion>(p_exiting_event)).is_valid()) {
		ie_type = INPUT_JOY_MOTION;
	} else {
		return;
	}

	_add_item(ie_type, p_exiting_event);
}

void InputMapEditor::_add_item(int p_item, Ref<InputEvent> p_exiting_event) {
	add_type = InputType(p_item);

	switch (add_type) {
		case INPUT_KEY: {
			press_a_key_physical = false;
			press_a_key_label->set_text(TTR("Press a Key..."));
			press_a_key->get_ok()->set_disabled(true);
			last_wait_for_key = Ref<InputEvent>();
			press_a_key->popup_centered(Size2(250, 80) * EDSCALE);
			//press_a_key->grab_focus();

		} break;
		case INPUT_KEY_PHYSICAL: {
			press_a_key_physical = true;
			press_a_key_label->set_text(TTR("Press a Key..."));

			last_wait_for_key = Ref<InputEvent>();
			press_a_key->popup_centered(Size2(250, 80) * EDSCALE);
			press_a_key->grab_focus();

		} break;
		case INPUT_MOUSE_BUTTON: {
			device_index_label->set_text(TTR("Mouse Button Index:"));
			device_index->clear();
			device_index->add_item(TTR("Left Button"));
			device_index->add_item(TTR("Right Button"));
			device_index->add_item(TTR("Middle Button"));
			device_index->add_item(TTR("Wheel Up Button"));
			device_index->add_item(TTR("Wheel Down Button"));
			device_index->add_item(TTR("Wheel Left Button"));
			device_index->add_item(TTR("Wheel Right Button"));
			device_index->add_item(TTR("X Button 1"));
			device_index->add_item(TTR("X Button 2"));
			device_input->popup_centered(Size2(350, 95) * EDSCALE);

			Ref<InputEventMouseButton> mb = p_exiting_event;
			if (mb.is_valid()) {
				device_index->select(mb->get_button_index() - 1);
				_set_current_device(mb->get_device());
				device_input->get_ok()->set_text(TTR("Change"));
			} else {
				_set_current_device(0);
				device_input->get_ok()->set_text(TTR("Add"));
			}

		} break;
		case INPUT_JOY_MOTION: {
			device_index_label->set_text(TTR("Joypad Axis Index:"));
			device_index->clear();
			for (int i = 0; i < JOY_AXIS_MAX * 2; i++) {
				String desc = TTR("Axis") + " " + itos(i / 2) + " " + ((i & 1) ? "+" : "-") +
							  " (" + TTR(_axis_descriptions[i]) + ")";
				device_index->add_item(desc);
			}
			device_input->popup_centered(Size2(350, 95) * EDSCALE);

			Ref<InputEventJoypadMotion> jm = p_exiting_event;
			if (jm.is_valid()) {
				device_index->select(jm->get_axis() * 2 + (jm->get_axis_value() > 0 ? 1 : 0));
				_set_current_device(jm->get_device());
				device_input->get_ok()->set_text(TTR("Change"));
			} else {
				_set_current_device(0);
				device_input->get_ok()->set_text(TTR("Add"));
			}

		} break;
		case INPUT_JOY_BUTTON: {
			device_index_label->set_text(TTR("Joypad Button Index:"));
			device_index->clear();
			for (int i = 0; i < JOY_BUTTON_MAX; i++) {
				String desc = TTR("Button") + " " + itos(i);
				if (i < JOY_SDL_BUTTONS) {
					desc += " (" + TTR(_button_descriptions[i]) + ")";
				}
				device_index->add_item(desc);
			}
			device_input->popup_centered(Size2(350, 95) * EDSCALE);

			Ref<InputEventJoypadButton> jb = p_exiting_event;
			if (jb.is_valid()) {
				device_index->select(jb->get_button_index());
				_set_current_device(jb->get_device());
				device_input->get_ok()->set_text(TTR("Change"));
			} else {
				_set_current_device(0);
				device_input->get_ok()->set_text(TTR("Add"));
			}

		} break;
		default: {
		}
	}
}

void InputMapEditor::_action_activated() {
	TreeItem *ti = input_editor->get_selected();

	if (!ti || ti->get_parent() == input_editor->get_root()) {
		return;
	}

	String name = "input/" + ti->get_parent()->get_text(0);
	Dictionary action = ProjectSettings::get_singleton()->get(name);
	Array events = action["events"];
	int idx = ti->get_metadata(0);

	ERR_FAIL_INDEX(idx, events.size());
	Ref<InputEvent> event = events[idx];
	if (event.is_null()) {
		return;
	}

	add_at = name;
	edit_idx = idx;
	_edit_item(event);
}

void InputMapEditor::_action_button_pressed(Object *p_obj, int p_column, int p_id) {
	TreeItem *ti = Object::cast_to<TreeItem>(p_obj);

	ERR_FAIL_COND(!ti);

	if (p_id == 1) {
		// Add action event
		Point2 ofs = input_editor->get_global_position();
		Rect2 ir = input_editor->get_item_rect(ti);
		ir.position.y -= input_editor->get_scroll().y;
		ofs += ir.position + ir.size;
		ofs.x -= 100;
		popup_add->set_position(ofs);
		popup_add->popup();
		add_at = "input/" + ti->get_text(0);
		edit_idx = -1;

	} else if (p_id == 2) {
		// Remove

		if (ti->get_parent() == input_editor->get_root()) {
			// Remove action
			String name = "input/" + ti->get_text(0);
			Dictionary old_val = ProjectSettings::get_singleton()->get(name);
			int order = ProjectSettings::get_singleton()->get_order(name);

			undo_redo->create_action(TTR("Erase Input Action"));
			undo_redo->add_do_method(ProjectSettings::get_singleton(), "clear", name);
			undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set", name, old_val);
			undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set_order", name, order);
			undo_redo->add_do_method(this, "_update_actions");
			undo_redo->add_undo_method(this, "_update_actions");
			undo_redo->add_do_method(this, "emit_signal", inputmap_changed);
			undo_redo->add_undo_method(this, "emit_signal", inputmap_changed);
			undo_redo->commit_action();

		} else {
			// Remove action event
			String name = "input/" + ti->get_parent()->get_text(0);
			Dictionary old_val = ProjectSettings::get_singleton()->get(name);
			Dictionary action = old_val.duplicate();
			int idx = ti->get_metadata(0);

			Array events = action["events"];
			ERR_FAIL_INDEX(idx, events.size());
			events.remove(idx);
			action["events"] = events;

			undo_redo->create_action(TTR("Erase Input Action Event"));
			undo_redo->add_do_method(ProjectSettings::get_singleton(), "set", name, action);
			undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set", name, old_val);
			undo_redo->add_do_method(this, "_update_actions");
			undo_redo->add_undo_method(this, "_update_actions");
			undo_redo->add_do_method(this, "emit_signal", inputmap_changed);
			undo_redo->add_undo_method(this, "emit_signal", inputmap_changed);
			undo_redo->commit_action();
		}
	} else if (p_id == 3) {
		// Edit

		if (ti->get_parent() == input_editor->get_root()) {
			// Edit action name
			ti->set_as_cursor(0);
			input_editor->edit_selected();

		} else {
			// Edit action event
			String name = "input/" + ti->get_parent()->get_text(0);
			int idx = ti->get_metadata(0);
			Dictionary action = ProjectSettings::get_singleton()->get(name);

			Array events = action["events"];
			ERR_FAIL_INDEX(idx, events.size());

			Ref<InputEvent> event = events[idx];

			if (event.is_null()) {
				return;
			}

			ti->set_as_cursor(0);
			add_at = name;
			edit_idx = idx;
			_edit_item(event);
		}
	}
}

void InputMapEditor::_update_actions() {
	if (setting) {
		return;
	}

	Map<String, bool> collapsed;

	if (input_editor->get_root() && input_editor->get_root()->get_children()) {
		for (TreeItem *item = input_editor->get_root()->get_children(); item; item = item->get_next()) {
			collapsed[item->get_text(0)] = item->is_collapsed();
		}
	}

	input_editor->clear();
	TreeItem *root = input_editor->create_item();
	input_editor->set_hide_root(true);

	List<PropertyInfo> props;
	ProjectSettings::get_singleton()->get_property_list(&props);
	for (List<PropertyInfo>::Element *E = props.front(); E; E = E->next()) {
		const String property_name = E->get().name;

		if (!property_name.begins_with("input/")) {
			continue;
		}

		const String name = property_name.get_slice("/", 1);

		TreeItem *item = input_editor->create_item(root);
		item->set_text(0, name);
		item->set_custom_bg_color(0, input_editor->get_theme_color("prop_subsection", "Editor"));
		if (collapsed.has(name)) {
			item->set_collapsed(collapsed[name]);
		}

		item->set_editable(1, true);
		item->set_cell_mode(1, TreeItem::CELL_MODE_RANGE);
		item->set_range_config(1, 0.0, 1.0, 0.01);

		item->set_custom_bg_color(1, input_editor->get_theme_color("prop_subsection", "Editor"));

		const bool is_builtin_input = ProjectSettings::get_singleton()->get_input_presets().find(property_name) != nullptr;
		const String tooltip_remove = is_builtin_input ? TTR("Built-in actions can't be removed as they're used for UI navigation.") : TTR("Remove");
		item->add_button(2, input_editor->get_theme_icon("Add", "EditorIcons"), 1, false, TTR("Add Event"));
		item->add_button(2, input_editor->get_theme_icon("Remove", "EditorIcons"), 2, false, tooltip_remove);

		if (is_builtin_input) {
			item->set_button_disabled(2, 1, true);
		} else {
			item->set_editable(0, true);
		}

		Dictionary action = ProjectSettings::get_singleton()->get(property_name);
		Array events = action["events"];
		item->set_range(1, action["deadzone"]);

		for (int i = 0; i < events.size(); i++) {
			Ref<InputEvent> event = events[i];
			if (event.is_null()) {
				continue;
			}

			TreeItem *action2 = input_editor->create_item(item);

			Ref<InputEventKey> k = event;
			if (k.is_valid()) {
				if (k->get_keycode() != 0) {
					action2->set_text(0, keycode_get_string(k->get_keycode_with_modifiers()));
					action2->set_icon(0, input_editor->get_theme_icon("Keyboard", "EditorIcons"));
				} else {
					action2->set_text(0, keycode_get_string(k->get_physical_keycode_with_modifiers()) + TTR(" (Physical)"));
					action2->set_icon(0, input_editor->get_theme_icon("KeyboardPhysical", "EditorIcons"));
				}
			}

			Ref<InputEventJoypadButton> jb = event;
			if (jb.is_valid()) {
				const int idx = jb->get_button_index();
				String str = _get_device_string(jb->get_device()) + ", " +
							 TTR("Button") + " " + itos(idx);
				if (idx >= 0 && idx < JOY_SDL_BUTTONS) {
					str += String() + " (" + TTR(_button_descriptions[jb->get_button_index()]) + ")";
				}

				action2->set_text(0, str);
				action2->set_icon(0, input_editor->get_theme_icon("JoyButton", "EditorIcons"));
			}

			Ref<InputEventMouseButton> mb = event;
			if (mb.is_valid()) {
				String str = _get_device_string(mb->get_device()) + ", ";
				switch (mb->get_button_index()) {
					case BUTTON_LEFT:
						str += TTR("Left Button");
						break;
					case BUTTON_RIGHT:
						str += TTR("Right Button");
						break;
					case BUTTON_MIDDLE:
						str += TTR("Middle Button");
						break;
					case BUTTON_WHEEL_UP:
						str += TTR("Wheel Up");
						break;
					case BUTTON_WHEEL_DOWN:
						str += TTR("Wheel Down");
						break;
					default:
						str += vformat(TTR("%d Button"), mb->get_button_index());
				}

				action2->set_text(0, str);
				action2->set_icon(0, input_editor->get_theme_icon("Mouse", "EditorIcons"));
			}

			Ref<InputEventJoypadMotion> jm = event;
			if (jm.is_valid()) {
				int ax = jm->get_axis();
				int n = 2 * ax + (jm->get_axis_value() < 0 ? 0 : 1);
				String str = _get_device_string(jm->get_device()) + ", " +
							 TTR("Axis") + " " + itos(ax) + " " + (jm->get_axis_value() < 0 ? "-" : "+") +
							 " (" + _axis_descriptions[n] + ")";
				action2->set_text(0, str);
				action2->set_icon(0, input_editor->get_theme_icon("JoyAxis", "EditorIcons"));
			}
			action2->set_metadata(0, i);
			action2->set_meta("__input", event);

			action2->add_button(2, input_editor->get_theme_icon("Edit", "EditorIcons"), 3, false, TTR("Edit"));
			action2->add_button(2, input_editor->get_theme_icon("Remove", "EditorIcons"), 2, false, TTR("Remove"));
			// Fade out the individual event buttons slightly to make the
			// Add/Remove buttons stand out more.
			action2->set_button_color(2, 0, Color(1, 1, 1, 0.75));
			action2->set_button_color(2, 1, Color(1, 1, 1, 0.75));
		}
	}

	_action_check(action_name->get_text());
}

void InputMapEditor::_action_check(String p_action) {
	if (p_action == "") {
		action_add->set_disabled(true);
	} else {
		if (!_validate_action_name(p_action)) {
			action_add_error->set_text(TTR("Invalid action name. It cannot be empty nor contain '/', ':', '=', '\\' or '\"'."));
			action_add_error->show();
			action_add->set_disabled(true);
			return;
		}
		if (ProjectSettings::get_singleton()->has_setting("input/" + p_action)) {
			action_add_error->set_text(vformat(TTR("An action with the name '%s' already exists."), p_action));
			action_add_error->show();
			action_add->set_disabled(true);
			return;
		}

		action_add->set_disabled(false);
	}

	action_add_error->hide();
}

void InputMapEditor::_action_adds(String) {
	if (!action_add->is_disabled()) {
		_action_add();
	}
}

void InputMapEditor::_action_add() {
	Dictionary action;
	action["events"] = Array();
	action["deadzone"] = 0.5f;
	String name = "input/" + action_name->get_text();
	undo_redo->create_action(TTR("Add Input Action"));
	undo_redo->add_do_method(ProjectSettings::get_singleton(), "set", name, action);
	undo_redo->add_undo_method(ProjectSettings::get_singleton(), "clear", name);
	undo_redo->add_do_method(this, "_update_actions");
	undo_redo->add_undo_method(this, "_update_actions");
	undo_redo->add_do_method(this, "emit_signal", inputmap_changed);
	undo_redo->add_undo_method(this, "emit_signal", inputmap_changed);
	undo_redo->commit_action();

	TreeItem *r = input_editor->get_root();

	if (!r) {
		return;
	}
	r = r->get_children();
	if (!r) {
		return;
	}
	while (r->get_next()) {
		r = r->get_next();
	}

	r->select(0);
	input_editor->ensure_cursor_is_visible();
	action_add_error->hide();
	action_name->clear();
}

Variant InputMapEditor::get_drag_data_fw(const Point2 &p_point, Control *p_from) {
	TreeItem *selected = input_editor->get_selected();
	if (!selected || selected->get_parent() != input_editor->get_root()) {
		return Variant();
	}

	String name = selected->get_text(0);
	VBoxContainer *vb = memnew(VBoxContainer);
	HBoxContainer *hb = memnew(HBoxContainer);
	Label *label = memnew(Label(name));
	hb->set_modulate(Color(1, 1, 1, 1.0f));
	hb->add_child(label);
	vb->add_child(hb);
	input_editor->set_drag_preview(vb);

	Dictionary drag_data;
	drag_data["type"] = "nodes";

	input_editor->set_drop_mode_flags(Tree::DROP_MODE_INBETWEEN);

	return drag_data;
}

bool InputMapEditor::can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const {
	Dictionary d = p_data;
	if (!d.has("type") || d["type"] != "nodes") {
		return false;
	}

	TreeItem *selected = input_editor->get_selected();
	TreeItem *item = input_editor->get_item_at_position(p_point);
	if (!selected || !item || item == selected || item->get_parent() == selected) {
		return false;
	}

	return true;
}

void InputMapEditor::drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {
	if (!can_drop_data_fw(p_point, p_data, p_from)) {
		return;
	}

	TreeItem *selected = input_editor->get_selected();
	TreeItem *item = input_editor->get_item_at_position(p_point);
	if (!item) {
		return;
	}
	TreeItem *target = item->get_parent() == input_editor->get_root() ? item : item->get_parent();

	String selected_name = "input/" + selected->get_text(0);
	int old_order = ProjectSettings::get_singleton()->get_order(selected_name);
	String target_name = "input/" + target->get_text(0);
	int target_order = ProjectSettings::get_singleton()->get_order(target_name);

	int order = old_order;
	bool is_below = target_order > old_order;
	TreeItem *iterator = is_below ? selected->get_next() : selected->get_prev();

	undo_redo->create_action(TTR("Moved Input Action Event"));
	while (iterator != target) {
		String iterator_name = "input/" + iterator->get_text(0);
		int iterator_order = ProjectSettings::get_singleton()->get_order(iterator_name);
		undo_redo->add_do_method(ProjectSettings::get_singleton(), "set_order", iterator_name, order);
		undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set_order", iterator_name, iterator_order);
		order = iterator_order;
		iterator = is_below ? iterator->get_next() : iterator->get_prev();
	}

	undo_redo->add_do_method(ProjectSettings::get_singleton(), "set_order", target_name, order);
	undo_redo->add_do_method(ProjectSettings::get_singleton(), "set_order", selected_name, target_order);
	undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set_order", target_name, target_order);
	undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set_order", selected_name, old_order);

	undo_redo->add_do_method(this, "_update_actions");
	undo_redo->add_undo_method(this, "_update_actions");
	undo_redo->add_do_method(this, "emit_signal", inputmap_changed);
	undo_redo->add_undo_method(this, "emit_signal", inputmap_changed);
	undo_redo->commit_action();
}

void InputMapEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_update_actions"), &InputMapEditor::_update_actions);

	ClassDB::bind_method(D_METHOD("get_drag_data_fw"), &InputMapEditor::get_drag_data_fw);
	ClassDB::bind_method(D_METHOD("can_drop_data_fw"), &InputMapEditor::can_drop_data_fw);
	ClassDB::bind_method(D_METHOD("drop_data_fw"), &InputMapEditor::drop_data_fw);

	ADD_SIGNAL(MethodInfo("inputmap_changed"));
}

InputMapEditor::InputMapEditor() {
	undo_redo = EditorNode::get_undo_redo();
	press_a_key_physical = false;
	inputmap_changed = "inputmap_changed";

	VBoxContainer *vbc = memnew(VBoxContainer);
	vbc->set_anchor_and_margin(MARGIN_TOP, Control::ANCHOR_BEGIN, 0);
	vbc->set_anchor_and_margin(MARGIN_BOTTOM, Control::ANCHOR_END, 0);
	vbc->set_anchor_and_margin(MARGIN_LEFT, Control::ANCHOR_BEGIN, 0);
	vbc->set_anchor_and_margin(MARGIN_RIGHT, Control::ANCHOR_END, 0);
	add_child(vbc);

	HBoxContainer *hbc = memnew(HBoxContainer);
	vbc->add_child(hbc);

	Label *l = memnew(Label);
	l->set_text(TTR("Action:"));
	hbc->add_child(l);

	action_name = memnew(LineEdit);
	action_name->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	action_name->connect("text_entered", callable_mp(this, &InputMapEditor::_action_adds));
	action_name->connect("text_changed", callable_mp(this, &InputMapEditor::_action_check));
	hbc->add_child(action_name);

	action_add_error = memnew(Label);
	action_add_error->hide();
	hbc->add_child(action_add_error);

	Button *add = memnew(Button);
	add->set_text(TTR("Add"));
	add->set_disabled(true);
	add->connect("pressed", callable_mp(this, &InputMapEditor::_action_add));
	hbc->add_child(add);
	action_add = add;

	input_editor = memnew(Tree);
	input_editor->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	input_editor->set_columns(3);
	input_editor->set_column_titles_visible(true);
	input_editor->set_column_title(0, TTR("Action"));
	input_editor->set_column_title(1, TTR("Deadzone"));
	input_editor->set_column_expand(1, false);
	input_editor->set_column_min_width(1, 80 * EDSCALE);
	input_editor->set_column_expand(2, false);
	input_editor->set_column_min_width(2, 50 * EDSCALE);
	input_editor->connect("item_edited", callable_mp(this, &InputMapEditor::_action_edited));
	input_editor->connect("item_activated", callable_mp(this, &InputMapEditor::_action_activated));
	input_editor->connect("cell_selected", callable_mp(this, &InputMapEditor::_action_selected));
	input_editor->connect("button_pressed", callable_mp(this, &InputMapEditor::_action_button_pressed));
#ifndef _MSC_VER
#warning need to make drag data forwarding to non controls happen
#endif
	//input_editor->set_drag_forwarding(this);
	vbc->add_child(input_editor);

	// Popups

	popup_add = memnew(PopupMenu);
	popup_add->connect("id_pressed", callable_mp(this, &InputMapEditor::_add_item), make_binds(Ref<InputEvent>()));
	add_child(popup_add);

	press_a_key = memnew(ConfirmationDialog);
	press_a_key->get_ok()->set_disabled(true);
	//press_a_key->set_focus_mode(Control::FOCUS_ALL);
	press_a_key->connect("window_input", callable_mp(this, &InputMapEditor::_wait_for_key));
	press_a_key->connect("confirmed", callable_mp(this, &InputMapEditor::_press_a_key_confirm));
	add_child(press_a_key);

	l = memnew(Label);
	l->set_text(TTR("Press a Key..."));
	l->set_anchors_and_margins_preset(Control::PRESET_WIDE);
	l->set_align(Label::ALIGN_CENTER);
	l->set_margin(MARGIN_TOP, 20);
	l->set_anchor_and_margin(MARGIN_BOTTOM, Control::ANCHOR_BEGIN, 30);
	press_a_key->add_child(l);
	press_a_key_label = l;

	device_input = memnew(ConfirmationDialog);
	device_input->get_ok()->set_text(TTR("Add"));
	device_input->connect("confirmed", callable_mp(this, &InputMapEditor::_device_input_add));
	add_child(device_input);

	hbc = memnew(HBoxContainer);
	device_input->add_child(hbc);

	VBoxContainer *vbc_left = memnew(VBoxContainer);
	hbc->add_child(vbc_left);

	l = memnew(Label);
	l->set_text(TTR("Device:"));
	vbc_left->add_child(l);

	device_id = memnew(OptionButton);
	for (int i = -1; i < 8; i++) {
		device_id->add_item(_get_device_string(i));
	}
	_set_current_device(0);
	vbc_left->add_child(device_id);

	VBoxContainer *vbc_right = memnew(VBoxContainer);
	vbc_right->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	hbc->add_child(vbc_right);

	l = memnew(Label);
	l->set_text(TTR("Index:"));
	vbc_right->add_child(l);

	device_index_label = l;
	device_index = memnew(OptionButton);
	device_index->set_clip_text(true);
	vbc_right->add_child(device_index);

	message = memnew(AcceptDialog);
	add_child(message);
}
