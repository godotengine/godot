/*************************************************************************/
/*  input_editor.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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

#include "input_editor.h"
#include "globals.h"
#include "editor_settings.h"
#include "os/keyboard.h"

static const char* _button_names[JOY_BUTTON_MAX] = {
	"PS X, XBox A, Nintendo B",
	"PS Circle, XBox B, Nintendo A",
	"PS Square, XBox X, Nintendo Y",
	"PS Triangle, XBox Y, Nintendo X",
	"L, L1",
	"R, R1",
	"L2",
	"R2",
	"L3",
	"R3",
	"Select, Nintendo -",
	"Start, Nintendo +",
	"D-Pad Up",
	"D-Pad Down",
	"D-Pad Left",
	"D-Pad Right"
};

void InputEditor::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {

		popup_add_key->add_icon_item(get_icon("Keyboard", "EditorIcons"), "Key", InputEvent::KEY);
		popup_add_key->add_icon_item(get_icon("JoyButton", "EditorIcons"), "Joy Button", InputEvent::JOYSTICK_BUTTON);
		popup_add_key->add_icon_item(get_icon("JoyAxis", "EditorIcons"), "Joy Axis", InputEvent::JOYSTICK_MOTION);
		popup_add_key->add_icon_item(get_icon("Mouse", "EditorIcons"), "Mouse Button", InputEvent::MOUSE_BUTTON);

		_update_actions();

	}
}

void InputEditor::_action_selected() {

	TreeItem *ti = input_tree->get_selected();

	if (!ti || !ti->is_editable(0))
		return;

	add_at = "input/" + ti->get_text(0);

}

void InputEditor::_action_edited() {

	TreeItem *ti = input_tree->get_selected();
	if (!ti)
		return;

	String new_name = ti->get_text(0);
	String old_name = add_at.substr(add_at.find("/") + 1, add_at.length());

	if (new_name == old_name) {

		return;

	}

	if (new_name.find("/") != -1 || new_name.find(":") != -1 || new_name == "") {

		ti->set_text(0, old_name);

		message->set_text("Invalid action (anything goes but / or :).");
		message->popup_centered(Size2(300, 100));

		return;

	}

	String action_prop = "input/" + new_name;

	if (use_editor_setttings) {

		if (EditorSettings::get_singleton()->has(action_prop)) {

			ti->set_text(0, old_name);

			message->set_text("Action '" + new_name + "' already exists!.");
			message->popup_centered(Size2(300, 100));

			return;

		}

		Array va = EditorSettings::get_singleton()->get(add_at);

		if (undo_redo) {
			setting = true;
			undo_redo->create_action("Rename Input Action Event");
			undo_redo->add_do_method(EditorSettings::get_singleton(), "erase", add_at);
			undo_redo->add_do_method(EditorSettings::get_singleton(), "set", action_prop, va);
			undo_redo->add_undo_method(EditorSettings::get_singleton(), "erase", action_prop);
			undo_redo->add_undo_method(EditorSettings::get_singleton(), "set", add_at, va);
			undo_redo->add_do_method(this, "_update_actions");
			undo_redo->add_undo_method(this, "_update_actions");
			undo_redo->add_do_method(this, "_settings_changed");
			undo_redo->add_undo_method(this, "_settings_changed");
			undo_redo->commit_action();
			setting = false;
		}

	}
	else {

		if (Globals::get_singleton()->has(action_prop)) {

			ti->set_text(0, old_name);

			message->set_text("Action '" + new_name + "' already exists!.");
			message->popup_centered(Size2(300, 100));

			return;
		}

		int order = Globals::get_singleton()->get_order(add_at);
		bool persisting = Globals::get_singleton()->is_persisting(add_at);
		Array va = Globals::get_singleton()->get(add_at);

		if (undo_redo) {
			setting = true;
			undo_redo->create_action("Rename Input Action Event");
			undo_redo->add_do_method(Globals::get_singleton(), "clear", add_at);
			undo_redo->add_do_method(Globals::get_singleton(), "set", action_prop, va);
			undo_redo->add_do_method(Globals::get_singleton(), "set_persisting", action_prop, persisting);
			undo_redo->add_do_method(Globals::get_singleton(), "set_order", action_prop, order);
			undo_redo->add_undo_method(Globals::get_singleton(), "clear", action_prop);
			undo_redo->add_undo_method(Globals::get_singleton(), "set", add_at, va);
			undo_redo->add_undo_method(Globals::get_singleton(), "set_persisting", add_at, persisting);
			undo_redo->add_undo_method(Globals::get_singleton(), "set_order", add_at, order);
			undo_redo->add_do_method(this, "_update_actions");
			undo_redo->add_undo_method(this, "_update_actions");
			undo_redo->add_do_method(this, "_settings_changed");
			undo_redo->add_undo_method(this, "_settings_changed");
			undo_redo->commit_action();
			setting = false;
		}

	}

	add_at = action_prop;

}


void InputEditor::_device_input_add() {

	InputEvent ie;
	String name = add_at;
	Variant old_val;
	ie.device = device_id->get_val();

	if (use_editor_setttings)
		old_val = EditorSettings::get_singleton()->get(name);
	else
		old_val = Globals::get_singleton()->get(name);

	Array arr = old_val;
	ie.type = add_type;

	switch (add_type) {

		case InputEvent::MOUSE_BUTTON:

			ie.mouse_button.button_index = device_index->get_selected() + 1;

			for (int i = 0;i<arr.size();i++) {

				InputEvent aie = arr[i];
				if (aie.device == ie.device && aie.type == InputEvent::MOUSE_BUTTON && aie.mouse_button.button_index == ie.mouse_button.button_index) {
					return;
				}
			}

			break;

		case InputEvent::JOYSTICK_MOTION:

			ie.joy_motion.axis = device_index->get_selected() >> 1;
			ie.joy_motion.axis_value = device_index->get_selected() & 1 ? 1 : -1;


			for (int i = 0;i<arr.size();i++) {

				InputEvent aie = arr[i];
				if (aie.device == ie.device && aie.type == InputEvent::JOYSTICK_MOTION && aie.joy_motion.axis == ie.joy_motion.axis && aie.joy_motion.axis_value == ie.joy_motion.axis_value) {
					return;
				}
			}

			break;

		case InputEvent::JOYSTICK_BUTTON:

			ie.joy_button.button_index = device_index->get_selected();

			for (int i = 0;i<arr.size();i++) {

				InputEvent aie = arr[i];
				if (aie.device == ie.device && aie.type == InputEvent::JOYSTICK_BUTTON && aie.joy_button.button_index == ie.joy_button.button_index) {
					return;
				}
			}

			break;

	}

	arr.push_back(ie);

	if (use_editor_setttings) {
		
		if (undo_redo) {
			undo_redo->create_action("Add Input Action Event");
			undo_redo->add_do_method(EditorSettings::get_singleton(), "set", name, arr);
			undo_redo->add_undo_method(EditorSettings::get_singleton(), "set", name, old_val);
			undo_redo->add_do_method(this, "_update_actions");
			undo_redo->add_undo_method(this, "_update_actions");
			undo_redo->add_do_method(this, "_settings_changed");
			undo_redo->add_undo_method(this, "_settings_changed");
			undo_redo->commit_action();
		}

	}
	else if (undo_redo) {

		undo_redo->create_action("Add Input Action Event");
		undo_redo->add_do_method(Globals::get_singleton(), "set", name, arr);
		undo_redo->add_do_method(Globals::get_singleton(), "set_persisting", name, true);
		undo_redo->add_undo_method(Globals::get_singleton(), "set", name, old_val);
		undo_redo->add_do_method(this, "_update_actions");
		undo_redo->add_undo_method(this, "_update_actions");
		undo_redo->add_do_method(this, "_settings_changed");
		undo_redo->add_undo_method(this, "_settings_changed");
		undo_redo->commit_action();

	}

}

void InputEditor::_press_a_key_confirm() {

	if (last_wait_for_key.type != InputEvent::KEY)
		return;

	InputEvent ie;
	ie.type = InputEvent::KEY;
	ie.key.scancode = last_wait_for_key.key.scancode;
	ie.key.mod = last_wait_for_key.key.mod;
	String name = add_at;

	Variant old_val;

	if (use_editor_setttings)
		old_val = EditorSettings::get_singleton()->get(name);
	else
		old_val = Globals::get_singleton()->get(name);

	Array arr = old_val;

	for (int i = 0;i<arr.size();i++) {

		InputEvent aie = arr[i];
		if (aie.type == InputEvent::KEY && aie.key.scancode == ie.key.scancode && aie.key.mod == ie.key.mod) {
			return;
		}

	}

	arr.push_back(ie);

	if (use_editor_setttings) {
		
		if (undo_redo) {

			undo_redo->create_action("Add Input Action Event");
			undo_redo->add_do_method(EditorSettings::get_singleton(), "set", name, arr);
			undo_redo->add_undo_method(EditorSettings::get_singleton(), "set", name, old_val);
			undo_redo->add_do_method(this, "_update_actions");
			undo_redo->add_undo_method(this, "_update_actions");
			undo_redo->add_do_method(this, "_settings_changed");
			undo_redo->add_undo_method(this, "_settings_changed");
			undo_redo->commit_action();

		}

	}
	else if (undo_redo) {

		undo_redo->create_action("Add Input Action Event");
		undo_redo->add_do_method(Globals::get_singleton(), "set", name, arr);
		undo_redo->add_do_method(Globals::get_singleton(), "set_persisting", name, true);
		undo_redo->add_undo_method(Globals::get_singleton(), "set", name, old_val);
		undo_redo->add_do_method(this, "_update_actions");
		undo_redo->add_undo_method(this, "_update_actions");
		undo_redo->add_do_method(this, "_settings_changed");
		undo_redo->add_undo_method(this, "_settings_changed");
		undo_redo->commit_action();

	}

}

void InputEditor::_wait_for_key(const InputEvent& p_event) {

	if (p_event.type == InputEvent::KEY && p_event.key.pressed && p_event.key.scancode != 0) {

		last_wait_for_key = p_event;
		String str = keycode_get_string(p_event.key.scancode).capitalize();
		if (p_event.key.mod.meta)
			str = "Meta+" + str;
		if (p_event.key.mod.shift)
			str = "Shift+" + str;
		if (p_event.key.mod.alt)
			str = "Alt+" + str;
		if (p_event.key.mod.control)
			str = "Control+" + str;


		press_a_key_label->set_text(str);
		press_a_key_dialog->accept_event();

	}

}


void InputEditor::_add_item(int p_item) {

	add_type = InputEvent::Type(p_item);

	switch (add_type) {

		case InputEvent::KEY:

			press_a_key_label->set_text("Press a Key..");
			last_wait_for_key = InputEvent();
			press_a_key_dialog->popup_centered(Size2(250, 80));
			press_a_key_dialog->grab_focus();

			break;

		case InputEvent::MOUSE_BUTTON:

			device_id->set_val(0);
			device_index_label->set_text("Mouse Button Index:");
			device_index->clear();
			device_index->add_item("Left Button");
			device_index->add_item("Right Button");
			device_index->add_item("Middle Button");
			device_index->add_item("Wheel Up Button");
			device_index->add_item("Wheel Down Button");
			device_index->add_item("Button 6");
			device_index->add_item("Button 7");
			device_index->add_item("Button 8");
			device_index->add_item("Button 9");
			device_input_dialog->popup_centered(Size2(350, 95));
		
			break;

		case InputEvent::JOYSTICK_MOTION:

			device_id->set_val(0);
			device_index_label->set_text("Joystick Axis Index:");
			device_index->clear();

			for (int i = 0; i<JOY_AXIS_MAX * 2; i++) {

				String desc;

				int ax = i / 2;
				if (ax == 0 || ax == 1)
					desc = " (Left Stick)";
				else if (ax == 2 || ax == 3)
					desc = " (Right Stick)";
				else if (ax == 6)
					desc = " (L2)";
				else if (ax == 7)
					desc = " (R2)";


				device_index->add_item("Axis " + itos(i / 2) + " " + (i & 1 ? "+" : "-") + desc);

			}

			device_input_dialog->popup_centered(Size2(350, 95));

			break;
			
		case InputEvent::JOYSTICK_BUTTON:

			device_id->set_val(0);
			device_index_label->set_text("Joystick Button Index:");
			device_index->clear();

			for (int i = 0; i<JOY_BUTTON_MAX; i++) {

				device_index->add_item(itos(i) + ": " + String(_button_names[i]));

			}

			device_input_dialog->popup_centered(Size2(350, 95));

			break;

	}

}

void InputEditor::_action_button_pressed(Object* p_obj, int p_column, int p_id) {

	TreeItem *ti = p_obj->cast_to<TreeItem>();

	ERR_FAIL_COND(!ti);

	if (p_id == 1) {

		// add
		Point2 ofs = input_tree->get_global_pos();
		Rect2 ir = input_tree->get_item_rect(ti);
		ir.pos.y -= input_tree->get_scroll().y;
		ofs += ir.pos + ir.size;
		ofs.x -= 100;

		popup_add_key->set_pos(ofs);
		popup_add_key->popup();

		add_at = "input/" + ti->get_text(0);

	}
	else if (p_id == 2) {

		// remove
		if (ti->get_parent() == input_tree->get_root()) {

			// remove whole action

			String name = "input/" + ti->get_text(0);

			if (use_editor_setttings) {

				Variant old_val = EditorSettings::get_singleton()->get(name);

				if (undo_redo) {
					undo_redo->create_action("Erase Input Action");
					undo_redo->add_do_method(EditorSettings::get_singleton(), "erase", name);
					undo_redo->add_undo_method(EditorSettings::get_singleton(), "set", name, old_val);
					undo_redo->add_do_method(this, "_update_actions");
					undo_redo->add_undo_method(this, "_update_actions");
					undo_redo->add_do_method(this, "_settings_changed");
					undo_redo->add_undo_method(this, "_settings_changed");
					undo_redo->commit_action();
				}

			}
			else {
				
				Variant old_val = Globals::get_singleton()->get(name);
				int order = Globals::get_singleton()->get_order(name);

				if (undo_redo) {
					undo_redo->create_action("Erase Input Action");
					undo_redo->add_do_method(Globals::get_singleton(), "clear", name);
					undo_redo->add_undo_method(Globals::get_singleton(), "set", name, old_val);
					undo_redo->add_undo_method(Globals::get_singleton(), "set_order", name, order);
					undo_redo->add_undo_method(Globals::get_singleton(), "set_persisting", name, Globals::get_singleton()->is_persisting(name));
					undo_redo->add_do_method(this, "_update_actions");
					undo_redo->add_undo_method(this, "_update_actions");
					undo_redo->add_do_method(this, "_settings_changed");
					undo_redo->add_undo_method(this, "_settings_changed");
					undo_redo->commit_action();
				}

			}

		}
		else {

			// remove action key

			String name = "input/" + ti->get_parent()->get_text(0);
			Variant old_val = use_editor_setttings ? EditorSettings::get_singleton()->get(name) : Globals::get_singleton()->get(name);
			int idx = ti->get_metadata(0);

			Array va = old_val;

			ERR_FAIL_INDEX(idx, va.size());

			for (int i = idx; i<va.size() - 1; i++) {

				va[i] = va[i + 1];

			}

			va.resize(va.size() - 1);

			if (use_editor_setttings) {
				
				if (undo_redo) {

					undo_redo->create_action("Erase Input Action Event");
					undo_redo->add_do_method(EditorSettings::get_singleton(), "set", name, va);
					undo_redo->add_undo_method(EditorSettings::get_singleton(), "set", name, old_val);
					undo_redo->add_do_method(this, "_update_actions");
					undo_redo->add_undo_method(this, "_update_actions");
					undo_redo->add_do_method(this, "_settings_changed");
					undo_redo->add_undo_method(this, "_settings_changed");
					undo_redo->commit_action();

				}

			}
			else if (undo_redo) {

				undo_redo->create_action("Erase Input Action Event");
				undo_redo->add_do_method(Globals::get_singleton(), "set", name, va);
				undo_redo->add_undo_method(Globals::get_singleton(), "set", name, old_val);
				undo_redo->add_do_method(this, "_update_actions");
				undo_redo->add_undo_method(this, "_update_actions");
				undo_redo->add_do_method(this, "_settings_changed");
				undo_redo->add_undo_method(this, "_settings_changed");
				undo_redo->commit_action();

			}

		}

	}

}


void InputEditor::_update_actions() const {

	if (setting)
		return;

	input_tree->clear();
	TreeItem *root = input_tree->create_item();
	input_tree->set_hide_root(true);

	List<PropertyInfo> props;
	if (use_editor_setttings)
		EditorSettings::get_singleton()->get_property_list(&props);
	else
		Globals::get_singleton()->get_property_list(&props);

	for (List<PropertyInfo>::Element *E = props.front();E;E = E->next()) {

		const PropertyInfo &pi = E->get();
		if (!pi.name.begins_with("input/"))
			continue;

		String name = pi.name.get_slice("/", 1);
		if (name == "")
			continue;

		TreeItem *item = input_tree->create_item(root);
		//item->set_cell_mode(0,TreeItem::CELL_MODE_CHECK);
		item->set_text(0, name);
		item->add_button(0, get_icon("Add", "EditorIcons"), 1);
		if (!use_editor_setttings && !Globals::get_singleton()->get_input_presets().find(pi.name)) {
			item->add_button(0, get_icon("Remove", "EditorIcons"), 2);
			item->set_editable(0, true);
		}
		item->set_custom_bg_color(0, get_color("prop_subsection", "Editor"));
		//item->set_checked(0,pi.usage&PROPERTY_USAGE_CHECKED);

		Array actions;

		if (use_editor_setttings)
			actions = EditorSettings::get_singleton()->get(pi.name);
		else
			actions = Globals::get_singleton()->get(pi.name);

		for (int i = 0;i<actions.size();i++) {

			if (actions[i].get_type() != Variant::INPUT_EVENT)
				continue;

			InputEvent ie = actions[i];

			TreeItem *action = input_tree->create_item(item);

			String str;
			switch (ie.type) {

				case InputEvent::KEY:

					str = keycode_get_string(ie.key.scancode).capitalize();
					if (ie.key.mod.meta)
						str = "Meta+" + str;
					if (ie.key.mod.shift)
						str = "Shift+" + str;
					if (ie.key.mod.alt)
						str = "Alt+" + str;
					if (ie.key.mod.control)
						str = "Control+" + str;

					action->set_text(0, str);
					action->set_icon(0, get_icon("Keyboard", "EditorIcons"));

					break;

				case InputEvent::JOYSTICK_BUTTON:

					str = "Device " + itos(ie.device) + ", Button " + itos(ie.joy_button.button_index);
					if (ie.joy_button.button_index >= 0 && ie.joy_button.button_index<JOY_BUTTON_MAX)
						str += String() + " (" + _button_names[ie.joy_button.button_index] + ").";
					else
						str += ".";

					action->set_text(0, str);
					action->set_icon(0, get_icon("JoyButton", "EditorIcons"));
				
					break;

				case InputEvent::MOUSE_BUTTON:

					str = "Device " + itos(ie.device) + ", ";

					switch (ie.mouse_button.button_index) {

						case BUTTON_LEFT: str += "Left Button."; break;
						case BUTTON_RIGHT: str += "Right Button."; break;
						case BUTTON_MIDDLE: str += "Middle Button."; break;
						case BUTTON_WHEEL_UP: str += "Wheel Up."; break;
						case BUTTON_WHEEL_DOWN: str += "Wheel Down."; break;
						default: str += "Button " + itos(ie.mouse_button.button_index) + ".";

					}

					action->set_text(0, str);
					action->set_icon(0, get_icon("Mouse", "EditorIcons"));

					break;

				case InputEvent::JOYSTICK_MOTION:

					String desc;
					int ax = ie.joy_motion.axis;

					if (ax == 0 || ax == 1)
						desc = " (Left Stick).";
					else if (ax == 2 || ax == 3)
						desc = " (Right Stick).";
					else if (ax == 6)
						desc = " (L2).";
					else if (ax == 7)
						desc = " (R2).";

					str = "Device " + itos(ie.device) + ", Axis " + itos(ie.joy_motion.axis) + " " + (ie.joy_motion.axis_value<0 ? "-" : "+") + desc;
					action->set_text(0, str);
					action->set_icon(0, get_icon("JoyAxis", "EditorIcons"));
				
					break;

			}

			action->add_button(0, get_icon("Remove", "EditorIcons"), 2);
			action->set_metadata(0, i);

		}

	}

}

void InputEditor::_action_adds(String) {

	_action_add();

}

void InputEditor::_action_add() {

	String action = action_name->get_text();

	if (action.find("/") != -1 || action.find(":") != -1 || action == "") {

		message->set_text("Invalid action (anything goes but / or :).");
		message->popup_centered(Size2(300, 100));

		return;

	}

	if (use_editor_setttings) {
		
		if (EditorSettings::get_singleton()->has("input/" + action)) {

			message->set_text("Action '" + action + "' already exists!.");
			message->popup_centered(Size2(300, 100));

			return;

		}

	}
	else if (Globals::get_singleton()->has("input/" + action)) {

		message->set_text("Action '" + action + "' already exists!.");
		message->popup_centered(Size2(300, 100));

		return;

	}

	Array va;
	String name = "input/" + action;

	if (use_editor_setttings) {
		
		if (undo_redo) {

			undo_redo->create_action("Add Input Action Event");
			undo_redo->add_do_method(EditorSettings::get_singleton(), "set", name, va);
			undo_redo->add_undo_method(EditorSettings::get_singleton(), "erase", name);
			undo_redo->add_do_method(this, "_update_actions");
			undo_redo->add_undo_method(this, "_update_actions");
			undo_redo->add_do_method(this, "_settings_changed");
			undo_redo->add_undo_method(this, "_settings_changed");
			undo_redo->commit_action();

		}

	}
	else if (undo_redo) {

		undo_redo->create_action("Add Input Action Event");
		undo_redo->add_do_method(Globals::get_singleton(), "set", name, va);
		undo_redo->add_do_method(Globals::get_singleton(), "set_persisting", name, true);
		undo_redo->add_undo_method(Globals::get_singleton(), "clear", name);
		undo_redo->add_do_method(this, "_update_actions");
		undo_redo->add_undo_method(this, "_update_actions");
		undo_redo->add_do_method(this, "_settings_changed");
		undo_redo->add_undo_method(this, "_settings_changed");
		undo_redo->commit_action();

	}

	TreeItem *r = input_tree->get_root();

	if (!r)
		return;

	r = r->get_children();

	if (!r)
		return;

	while (r->get_next())
		r = r->get_next();

	if (!r)
		return;

	r->select(0);
	input_tree->ensure_cursor_is_visible();

	action_name->set_text("");

}

void InputEditor::_settings_changed() {

	emit_signal("settings_changed");

}


void InputEditor::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("_action_add"), &InputEditor::_action_add);
	ObjectTypeDB::bind_method(_MD("_action_adds"), &InputEditor::_action_adds);
	ObjectTypeDB::bind_method(_MD("_action_selected"), &InputEditor::_action_selected);
	ObjectTypeDB::bind_method(_MD("_action_edited"), &InputEditor::_action_edited);
	ObjectTypeDB::bind_method(_MD("_action_button_pressed"), &InputEditor::_action_button_pressed);
	ObjectTypeDB::bind_method(_MD("_update_actions"), &InputEditor::_update_actions);
	ObjectTypeDB::bind_method(_MD("_wait_for_key"), &InputEditor::_wait_for_key);
	ObjectTypeDB::bind_method(_MD("_add_item"), &InputEditor::_add_item);
	ObjectTypeDB::bind_method(_MD("_device_input_add"), &InputEditor::_device_input_add);
	ObjectTypeDB::bind_method(_MD("_press_a_key_confirm"), &InputEditor::_press_a_key_confirm);
	ObjectTypeDB::bind_method(_MD("_settings_changed"), &InputEditor::_settings_changed);

	ADD_SIGNAL(MethodInfo("settings_changed"));

}

InputEditor::InputEditor(bool p_use_editor_settings, UndoRedo *p_undoredo) {

	use_editor_setttings = p_use_editor_settings; // otherwise use globals
	undo_redo = p_undoredo;

	Label *label = memnew(Label);
	add_child(label);
	label->set_pos(Point2(6, 5));
	label->set_text("Action:");

	action_name = memnew(LineEdit);
	add_child(action_name);
	action_name->set_anchor(MARGIN_RIGHT, ANCHOR_RATIO);
	action_name->set_begin(Point2(5, 25));
	action_name->set_end(Point2(0.85, 26));
	action_name->connect("text_entered", this, "_action_adds");

	Button *add_button = memnew(Button);
	add_child(add_button);
	add_button->set_anchor(MARGIN_LEFT, ANCHOR_RATIO);
	add_button->set_begin(Point2(0.86, 25));
	add_button->set_anchor(MARGIN_RIGHT, ANCHOR_END);
	add_button->set_end(Point2(5, 26));
	add_button->set_text("Add");
	add_button->connect("pressed", this, "_action_add");

	input_tree = memnew(Tree);
	add_child(input_tree);
	input_tree->set_area_as_parent_rect();
	input_tree->set_anchor_and_margin(MARGIN_TOP, ANCHOR_BEGIN, 55);
	input_tree->set_anchor_and_margin(MARGIN_BOTTOM, ANCHOR_END, 5);
	input_tree->set_anchor_and_margin(MARGIN_LEFT, ANCHOR_BEGIN, 5);
	input_tree->set_anchor_and_margin(MARGIN_RIGHT, ANCHOR_END, 5);
	input_tree->connect("item_edited", this, "_action_edited");
	input_tree->connect("cell_selected", this, "_action_selected");
	input_tree->connect("button_pressed", this, "_action_button_pressed");

	popup_add_key = memnew(PopupMenu);
	add_child(popup_add_key);
	popup_add_key->connect("item_pressed", this, "_add_item");

	press_a_key_dialog = memnew(ConfirmationDialog);
	add_child(press_a_key_dialog);
	press_a_key_dialog->set_focus_mode(FOCUS_ALL);

	label = memnew(Label);
	label->set_text("Press a Key..");
	label->set_area_as_parent_rect();
	label->set_align(Label::ALIGN_CENTER);
	label->set_margin(MARGIN_TOP, 20);
	label->set_anchor_and_margin(MARGIN_BOTTOM, ANCHOR_BEGIN, 30);

	press_a_key_label = label;
	press_a_key_dialog->add_child(label);
	press_a_key_dialog->connect("input_event", this, "_wait_for_key");
	press_a_key_dialog->connect("confirmed", this, "_press_a_key_confirm");

	device_input_dialog = memnew(ConfirmationDialog);
	add_child(device_input_dialog);
	device_input_dialog->get_ok()->set_text("Add");
	device_input_dialog->connect("confirmed", this, "_device_input_add");

	label = memnew(Label);
	device_input_dialog->add_child(label);
	label->set_text("Device:");
	label->set_pos(Point2(15, 10));

	label = memnew(Label);
	device_input_dialog->add_child(label);
	label->set_text("Index:");
	label->set_pos(Point2(90, 10));
	device_index_label = label;

	device_id = memnew(SpinBox);
	device_input_dialog->add_child(device_id);
	device_id->set_pos(Point2(20, 30));
	device_id->set_size(Size2(70, 10));
	device_id->set_val(0);

	device_index = memnew(OptionButton);
	device_input_dialog->add_child(device_index);
	device_index->set_pos(Point2(95, 30));
	device_index->set_size(Size2(300, 10));
	device_index->set_anchor_and_margin(MARGIN_RIGHT, ANCHOR_END, 10);

	message = memnew(ConfirmationDialog);
	add_child(message);
	message->get_cancel()->hide();
	message->set_hide_on_ok(true);

	setting = false;

}
