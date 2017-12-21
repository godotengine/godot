/*************************************************************************/
/*  base_button.cpp                                                      */
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
#include "base_button.h"

#include "os/keyboard.h"
#include "print_string.h"
#include "scene/main/viewport.h"
#include "scene/scene_string_names.h"

void BaseButton::_unpress_group() {

	if (!button_group.is_valid())
		return;

	if (toggle_mode) {
		status.pressed = true;
	}

	for (Set<BaseButton *>::Element *E = button_group->buttons.front(); E; E = E->next()) {
		if (E->get() == this)
			continue;

		E->get()->set_pressed(false);
	}
}

void BaseButton::_gui_input(Ref<InputEvent> p_event) {

	if (status.disabled) // no interaction with disabled button
		return;

	Ref<InputEventMouseButton> b = p_event;

	if (b.is_valid()) {
		if (status.disabled || b->get_button_index() != 1)
			return;

		if (status.pressing_button)
			return;

		if (action_mode == ACTION_MODE_BUTTON_PRESS) {

			if (b->is_pressed()) {

				emit_signal("button_down");

				if (!toggle_mode) { //mouse press attempt

					status.press_attempt = true;
					status.pressing_inside = true;

					pressed();
					if (get_script_instance()) {
						Variant::CallError ce;
						get_script_instance()->call(SceneStringNames::get_singleton()->_pressed, NULL, 0, ce);
					}

					emit_signal("pressed");
					_unpress_group();

				} else {

					status.pressed = !status.pressed;
					pressed();

					emit_signal("pressed");
					_unpress_group();

					toggled(status.pressed);
					if (get_script_instance()) {
						get_script_instance()->call(SceneStringNames::get_singleton()->_toggled, status.pressed);
					}
					emit_signal("toggled", status.pressed);
				}

			} else {

				emit_signal("button_up");

				/* this is pointless		if (status.press_attempt && status.pressing_inside) {
					//released();
					emit_signal("released");
				}
*/
				status.press_attempt = false;
			}
			update();
			return;
		}

		if (b->is_pressed()) {

			status.press_attempt = true;
			status.pressing_inside = true;
			emit_signal("button_down");

		} else {

			emit_signal("button_up");

			if (status.press_attempt && status.pressing_inside) {

				if (!toggle_mode) { //mouse press attempt

					pressed();
					if (get_script_instance()) {
						Variant::CallError ce;
						get_script_instance()->call(SceneStringNames::get_singleton()->_pressed, NULL, 0, ce);
					}

					emit_signal("pressed");

				} else {

					status.pressed = !status.pressed;

					pressed();
					emit_signal("pressed");

					toggled(status.pressed);
					if (get_script_instance()) {
						get_script_instance()->call(SceneStringNames::get_singleton()->_toggled, status.pressed);
					}
					emit_signal("toggled", status.pressed);
				}

				_unpress_group();
			}

			status.press_attempt = false;
		}

		update();
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid()) {
		if (status.press_attempt && status.pressing_button == 0) {
			bool last_press_inside = status.pressing_inside;
			status.pressing_inside = has_point(mm->get_position());
			if (last_press_inside != status.pressing_inside)
				update();
		}
	}

	if (!mm.is_valid() && !b.is_valid()) {

		if (p_event->is_echo()) {
			return;
		}

		if (status.disabled) {
			return;
		}

		if (status.press_attempt && status.pressing_button == 0) {
			return;
		}

		if (p_event->is_action("ui_accept")) {

			if (p_event->is_pressed()) {

				status.pressing_button++;
				status.press_attempt = true;
				status.pressing_inside = true;
				emit_signal("button_down");

			} else if (status.press_attempt) {

				if (status.pressing_button)
					status.pressing_button--;

				if (status.pressing_button)
					return;

				status.press_attempt = false;
				status.pressing_inside = false;

				emit_signal("button_up");

				if (!toggle_mode) { //mouse press attempt

					pressed();
					emit_signal("pressed");
				} else {

					status.pressed = !status.pressed;

					pressed();
					emit_signal("pressed");

					toggled(status.pressed);
					if (get_script_instance()) {
						get_script_instance()->call(SceneStringNames::get_singleton()->_toggled, status.pressed);
					}
					emit_signal("toggled", status.pressed);
				}

				_unpress_group();
			}

			accept_event();
			update();
		}
	}
}

void BaseButton::_notification(int p_what) {

	if (p_what == NOTIFICATION_MOUSE_ENTER) {

		status.hovering = true;
		update();
	}

	if (p_what == NOTIFICATION_MOUSE_EXIT) {
		status.hovering = false;
		update();
	}
	if (p_what == NOTIFICATION_DRAG_BEGIN) {

		if (status.press_attempt) {
			status.press_attempt = false;
			status.pressing_button = 0;
			update();
		}
	}

	if (p_what == NOTIFICATION_FOCUS_ENTER) {

		status.hovering = true;
		update();
	}

	if (p_what == NOTIFICATION_FOCUS_EXIT) {

		if (status.pressing_button && status.press_attempt) {
			status.press_attempt = false;
			status.pressing_button = 0;
			status.hovering = false;
			update();
		} else if (status.hovering) {
			status.hovering = false;
			update();
		}
	}

	if (p_what == NOTIFICATION_ENTER_TREE) {
	}

	if (p_what == NOTIFICATION_EXIT_TREE) {
	}

	if (p_what == NOTIFICATION_VISIBILITY_CHANGED && !is_visible_in_tree()) {

		if (!toggle_mode) {
			status.pressed = false;
		}
		status.hovering = false;
		status.press_attempt = false;
		status.pressing_inside = false;
		status.pressing_button = 0;
	}
}

void BaseButton::pressed() {

	if (get_script_instance())
		get_script_instance()->call("pressed");
}

void BaseButton::toggled(bool p_pressed) {

	if (get_script_instance()) {
		get_script_instance()->call("toggled", p_pressed);
	}
}

void BaseButton::set_disabled(bool p_disabled) {

	status.disabled = p_disabled;
	update();
	_change_notify("disabled");
	if (p_disabled)
		set_focus_mode(FOCUS_NONE);
	else
		set_focus_mode(enabled_focus_mode);
}

bool BaseButton::is_disabled() const {

	return status.disabled;
}

void BaseButton::set_pressed(bool p_pressed) {

	if (!toggle_mode)
		return;
	if (status.pressed == p_pressed)
		return;
	_change_notify("pressed");
	status.pressed = p_pressed;

	if (p_pressed) {
		_unpress_group();
	}
	update();
}

bool BaseButton::is_pressing() const {

	return status.press_attempt;
}

bool BaseButton::is_pressed() const {

	return toggle_mode ? status.pressed : status.press_attempt;
}

bool BaseButton::is_hovered() const {

	return status.hovering;
}

BaseButton::DrawMode BaseButton::get_draw_mode() const {

	if (status.disabled) {
		return DRAW_DISABLED;
	};

	//print_line("press attempt: "+itos(status.press_attempt)+" hover: "+itos(status.hovering)+" pressed: "+itos(status.pressed));
	if (status.press_attempt == false && status.hovering && !status.pressed) {

		return DRAW_HOVER;
	} else {
		/* determine if pressed or not */

		bool pressing;
		if (status.press_attempt) {

			pressing = status.pressing_inside;
			if (status.pressed)
				pressing = !pressing;
		} else {

			pressing = status.pressed;
		}

		if (pressing)
			return DRAW_PRESSED;
		else
			return DRAW_NORMAL;
	}

	return DRAW_NORMAL;
}

void BaseButton::set_toggle_mode(bool p_on) {

	toggle_mode = p_on;
}

bool BaseButton::is_toggle_mode() const {

	return toggle_mode;
}

void BaseButton::set_action_mode(ActionMode p_mode) {

	action_mode = p_mode;
}

BaseButton::ActionMode BaseButton::get_action_mode() const {

	return action_mode;
}

void BaseButton::set_enabled_focus_mode(FocusMode p_mode) {

	enabled_focus_mode = p_mode;
	if (!status.disabled) {
		set_focus_mode(p_mode);
	}
}

Control::FocusMode BaseButton::get_enabled_focus_mode() const {

	return enabled_focus_mode;
}

void BaseButton::set_shortcut(const Ref<ShortCut> &p_shortcut) {

	if (shortcut.is_null() == p_shortcut.is_null())
		return;

	shortcut = p_shortcut;
	set_process_unhandled_input(shortcut.is_valid());
}

Ref<ShortCut> BaseButton::get_shortcut() const {
	return shortcut;
}

void BaseButton::_unhandled_input(Ref<InputEvent> p_event) {

	if (!is_disabled() && is_visible_in_tree() && p_event->is_pressed() && !p_event->is_echo() && shortcut.is_valid() && shortcut->is_shortcut(p_event)) {

		if (get_viewport()->get_modal_stack_top() && !get_viewport()->get_modal_stack_top()->is_a_parent_of(this))
			return; //ignore because of modal window

		if (is_toggle_mode()) {
			set_pressed(!is_pressed());
			emit_signal("toggled", is_pressed());
		}

		emit_signal("pressed");
	}
}

String BaseButton::get_tooltip(const Point2 &p_pos) const {

	String tooltip = Control::get_tooltip(p_pos);
	if (shortcut.is_valid() && shortcut->is_valid()) {
		String text = shortcut->get_name() + " (" + shortcut->get_as_text() + ")";
		if (shortcut->get_name().nocasecmp_to(tooltip) != 0) {
			text += "\n" + tooltip;
		}
		tooltip = text;
	}
	return tooltip;
}

void BaseButton::set_button_group(const Ref<ButtonGroup> &p_group) {

	if (button_group.is_valid()) {
		button_group->buttons.erase(this);
	}

	button_group = p_group;

	if (button_group.is_valid()) {
		button_group->buttons.insert(this);
	}

	update(); //checkbox changes to radio if set a buttongroup
}

Ref<ButtonGroup> BaseButton::get_button_group() const {

	return button_group;
}

void BaseButton::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_gui_input"), &BaseButton::_gui_input);
	ClassDB::bind_method(D_METHOD("_unhandled_input"), &BaseButton::_unhandled_input);
	ClassDB::bind_method(D_METHOD("set_pressed", "pressed"), &BaseButton::set_pressed);
	ClassDB::bind_method(D_METHOD("is_pressed"), &BaseButton::is_pressed);
	ClassDB::bind_method(D_METHOD("is_hovered"), &BaseButton::is_hovered);
	ClassDB::bind_method(D_METHOD("set_toggle_mode", "enabled"), &BaseButton::set_toggle_mode);
	ClassDB::bind_method(D_METHOD("is_toggle_mode"), &BaseButton::is_toggle_mode);
	ClassDB::bind_method(D_METHOD("set_disabled", "disabled"), &BaseButton::set_disabled);
	ClassDB::bind_method(D_METHOD("is_disabled"), &BaseButton::is_disabled);
	ClassDB::bind_method(D_METHOD("set_action_mode", "mode"), &BaseButton::set_action_mode);
	ClassDB::bind_method(D_METHOD("get_action_mode"), &BaseButton::get_action_mode);
	ClassDB::bind_method(D_METHOD("get_draw_mode"), &BaseButton::get_draw_mode);
	ClassDB::bind_method(D_METHOD("set_enabled_focus_mode", "mode"), &BaseButton::set_enabled_focus_mode);
	ClassDB::bind_method(D_METHOD("get_enabled_focus_mode"), &BaseButton::get_enabled_focus_mode);

	ClassDB::bind_method(D_METHOD("set_shortcut", "shortcut"), &BaseButton::set_shortcut);
	ClassDB::bind_method(D_METHOD("get_shortcut"), &BaseButton::get_shortcut);

	ClassDB::bind_method(D_METHOD("set_button_group", "button_group"), &BaseButton::set_button_group);
	ClassDB::bind_method(D_METHOD("get_button_group"), &BaseButton::get_button_group);

	BIND_VMETHOD(MethodInfo("_pressed"));
	BIND_VMETHOD(MethodInfo("_toggled", PropertyInfo(Variant::BOOL, "pressed")));

	ADD_SIGNAL(MethodInfo("pressed"));
	ADD_SIGNAL(MethodInfo("button_up"));
	ADD_SIGNAL(MethodInfo("button_down"));
	ADD_SIGNAL(MethodInfo("toggled", PropertyInfo(Variant::BOOL, "pressed")));
	ADD_PROPERTYNZ(PropertyInfo(Variant::BOOL, "disabled"), "set_disabled", "is_disabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "toggle_mode"), "set_toggle_mode", "is_toggle_mode");
	ADD_PROPERTYNZ(PropertyInfo(Variant::BOOL, "pressed"), "set_pressed", "is_pressed");
	ADD_PROPERTYNO(PropertyInfo(Variant::INT, "action_mode", PROPERTY_HINT_ENUM, "Button Press,Button Release"), "set_action_mode", "get_action_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "enabled_focus_mode", PROPERTY_HINT_ENUM, "None,Click,All"), "set_enabled_focus_mode", "get_enabled_focus_mode");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "shortcut", PROPERTY_HINT_RESOURCE_TYPE, "ShortCut"), "set_shortcut", "get_shortcut");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "group", PROPERTY_HINT_RESOURCE_TYPE, "ButtonGroup"), "set_button_group", "get_button_group");

	BIND_ENUM_CONSTANT(DRAW_NORMAL);
	BIND_ENUM_CONSTANT(DRAW_PRESSED);
	BIND_ENUM_CONSTANT(DRAW_HOVER);
	BIND_ENUM_CONSTANT(DRAW_DISABLED);

	BIND_ENUM_CONSTANT(ACTION_MODE_BUTTON_PRESS);
	BIND_ENUM_CONSTANT(ACTION_MODE_BUTTON_RELEASE);
}

BaseButton::BaseButton() {

	toggle_mode = false;
	status.pressed = false;
	status.press_attempt = false;
	status.hovering = false;
	status.pressing_inside = false;
	status.disabled = false;
	status.pressing_button = 0;
	set_focus_mode(FOCUS_ALL);
	enabled_focus_mode = FOCUS_ALL;
	action_mode = ACTION_MODE_BUTTON_RELEASE;

	if (button_group.is_valid()) {
		button_group->buttons.erase(this);
	}
}

BaseButton::~BaseButton() {
}

void ButtonGroup::get_buttons(List<BaseButton *> *r_buttons) {

	for (Set<BaseButton *>::Element *E = buttons.front(); E; E = E->next()) {
		r_buttons->push_back(E->get());
	}
}

BaseButton *ButtonGroup::get_pressed_button() {

	for (Set<BaseButton *>::Element *E = buttons.front(); E; E = E->next()) {
		if (E->get()->is_pressed())
			return E->get();
	}

	return NULL;
}

void ButtonGroup::_bind_methods() {

	ClassDB::bind_method(D_METHOD("get_pressed_button"), &ButtonGroup::get_pressed_button);
}

ButtonGroup::ButtonGroup() {

	set_local_to_scene(true);
}
