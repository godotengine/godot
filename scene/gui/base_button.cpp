/*************************************************************************/
/*  base_button.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/os/keyboard.h"
#include "scene/main/window.h"
#include "scene/scene_string_names.h"

void BaseButton::_unpress_group() {
	if (!button_group.is_valid()) {
		return;
	}

	if (toggle_mode) {
		status.pressed = true;
	}

	for (Set<BaseButton *>::Element *E = button_group->buttons.front(); E; E = E->next()) {
		if (E->get() == this) {
			continue;
		}

		E->get()->set_pressed(false);
	}
}

void BaseButton::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	if (status.disabled) { // no interaction with disabled button
		return;
	}

	Ref<InputEventMouseButton> mouse_button = p_event;
	bool ui_accept = p_event->is_action("ui_accept") && !p_event->is_echo();

	bool button_masked = mouse_button.is_valid() && (mouse_button_to_mask(mouse_button->get_button_index()) & button_mask) != MouseButton::NONE;
	if (button_masked || ui_accept) {
		on_action_event(p_event);
		return;
	}

	Ref<InputEventMouseMotion> mouse_motion = p_event;
	if (mouse_motion.is_valid()) {
		if (status.press_attempt) {
			bool last_press_inside = status.pressing_inside;
			status.pressing_inside = has_point(mouse_motion->get_position());
			if (last_press_inside != status.pressing_inside) {
				update();
			}
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
	if (p_what == NOTIFICATION_DRAG_BEGIN || p_what == NOTIFICATION_SCROLL_BEGIN) {
		if (status.press_attempt) {
			status.press_attempt = false;
			update();
		}
	}

	if (p_what == NOTIFICATION_FOCUS_ENTER) {
		update();
	}

	if (p_what == NOTIFICATION_FOCUS_EXIT) {
		if (status.press_attempt) {
			status.press_attempt = false;
			update();
		} else if (status.hovering) {
			update();
		}
	}

	if (p_what == NOTIFICATION_EXIT_TREE || (p_what == NOTIFICATION_VISIBILITY_CHANGED && !is_visible_in_tree())) {
		if (!toggle_mode) {
			status.pressed = false;
		}
		status.hovering = false;
		status.press_attempt = false;
		status.pressing_inside = false;
	}
}

void BaseButton::_pressed() {
	GDVIRTUAL_CALL(_pressed);
	pressed();
	emit_signal(SNAME("pressed"));
}

void BaseButton::_toggled(bool p_pressed) {
	GDVIRTUAL_CALL(_toggled, p_pressed);
	toggled(p_pressed);
	emit_signal(SNAME("toggled"), p_pressed);
}

void BaseButton::on_action_event(Ref<InputEvent> p_event) {
	if (p_event->is_pressed()) {
		status.press_attempt = true;
		status.pressing_inside = true;
		emit_signal(SNAME("button_down"));
	}

	if (status.press_attempt && status.pressing_inside) {
		if (toggle_mode) {
			bool is_pressed = p_event->is_pressed();
			if (Object::cast_to<InputEventShortcut>(*p_event)) {
				is_pressed = false;
			}
			if ((is_pressed && action_mode == ACTION_MODE_BUTTON_PRESS) || (!is_pressed && action_mode == ACTION_MODE_BUTTON_RELEASE)) {
				if (action_mode == ACTION_MODE_BUTTON_PRESS) {
					status.press_attempt = false;
					status.pressing_inside = false;
				}
				status.pressed = !status.pressed;
				_unpress_group();
				if (button_group.is_valid()) {
					button_group->emit_signal(SNAME("pressed"), this);
				}
				_toggled(status.pressed);
				_pressed();
			}
		} else {
			if ((p_event->is_pressed() && action_mode == ACTION_MODE_BUTTON_PRESS) || (!p_event->is_pressed() && action_mode == ACTION_MODE_BUTTON_RELEASE)) {
				_pressed();
			}
		}
	}

	if (!p_event->is_pressed()) {
		Ref<InputEventMouseButton> mouse_button = p_event;
		if (mouse_button.is_valid()) {
			if (!has_point(mouse_button->get_position())) {
				status.hovering = false;
			}
		}
		status.press_attempt = false;
		status.pressing_inside = false;
		emit_signal(SNAME("button_up"));
	}

	update();
}

void BaseButton::pressed() {
}

void BaseButton::toggled(bool p_pressed) {
}

void BaseButton::set_disabled(bool p_disabled) {
	if (status.disabled == p_disabled) {
		return;
	}

	status.disabled = p_disabled;
	if (p_disabled) {
		if (!toggle_mode) {
			status.pressed = false;
		}
		status.press_attempt = false;
		status.pressing_inside = false;
	}
	update();
}

bool BaseButton::is_disabled() const {
	return status.disabled;
}

void BaseButton::set_pressed(bool p_pressed) {
	if (!toggle_mode) {
		return;
	}
	if (status.pressed == p_pressed) {
		return;
	}
	status.pressed = p_pressed;

	if (p_pressed) {
		_unpress_group();
		if (button_group.is_valid()) {
			button_group->emit_signal(SNAME("pressed"), this);
		}
	}
	_toggled(status.pressed);

	update();
}

void BaseButton::set_pressed_no_signal(bool p_pressed) {
	if (!toggle_mode) {
		return;
	}
	if (status.pressed == p_pressed) {
		return;
	}
	status.pressed = p_pressed;

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

	if (!status.press_attempt && status.hovering) {
		if (status.pressed) {
			return DRAW_HOVER_PRESSED;
		}

		return DRAW_HOVER;
	} else {
		/* determine if pressed or not */

		bool pressing;
		if (status.press_attempt) {
			pressing = (status.pressing_inside || keep_pressed_outside);
			if (status.pressed) {
				pressing = !pressing;
			}
		} else {
			pressing = status.pressed;
		}

		if (pressing) {
			return DRAW_PRESSED;
		} else {
			return DRAW_NORMAL;
		}
	}

	return DRAW_NORMAL;
}

void BaseButton::set_toggle_mode(bool p_on) {
	// Make sure to set 'pressed' to false if we are not in toggle mode
	if (!p_on) {
		set_pressed(false);
	}

	toggle_mode = p_on;
}

bool BaseButton::is_toggle_mode() const {
	return toggle_mode;
}

void BaseButton::set_shortcut_in_tooltip(bool p_on) {
	shortcut_in_tooltip = p_on;
}

bool BaseButton::is_shortcut_in_tooltip_enabled() const {
	return shortcut_in_tooltip;
}

void BaseButton::set_action_mode(ActionMode p_mode) {
	action_mode = p_mode;
}

BaseButton::ActionMode BaseButton::get_action_mode() const {
	return action_mode;
}

void BaseButton::set_button_mask(MouseButton p_mask) {
	button_mask = p_mask;
}

MouseButton BaseButton::get_button_mask() const {
	return button_mask;
}

void BaseButton::set_keep_pressed_outside(bool p_on) {
	keep_pressed_outside = p_on;
}

bool BaseButton::is_keep_pressed_outside() const {
	return keep_pressed_outside;
}

void BaseButton::set_shortcut(const Ref<Shortcut> &p_shortcut) {
	shortcut = p_shortcut;
	set_process_unhandled_key_input(shortcut.is_valid());
}

Ref<Shortcut> BaseButton::get_shortcut() const {
	return shortcut;
}

void BaseButton::unhandled_key_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	if (!_is_focus_owner_in_shorcut_context()) {
		return;
	}

	if (!is_disabled() && is_visible_in_tree() && !p_event->is_echo() && shortcut.is_valid() && shortcut->matches_event(p_event)) {
		on_action_event(p_event);
		accept_event();
	}
}

String BaseButton::get_tooltip(const Point2 &p_pos) const {
	String tooltip = Control::get_tooltip(p_pos);
	if (shortcut_in_tooltip && shortcut.is_valid() && shortcut->has_valid_event()) {
		String text = shortcut->get_name() + " (" + shortcut->get_as_text() + ")";
		if (!tooltip.is_empty() && shortcut->get_name().nocasecmp_to(tooltip) != 0) {
			text += "\n" + atr(tooltip);
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

void BaseButton::set_shortcut_context(Node *p_node) {
	ERR_FAIL_NULL_MSG(p_node, "Shortcut context node can't be null.");
	shortcut_context = p_node->get_instance_id();
}

Node *BaseButton::get_shortcut_context() const {
	Object *ctx_obj = ObjectDB::get_instance(shortcut_context);
	Node *ctx_node = Object::cast_to<Node>(ctx_obj);

	return ctx_node;
}

bool BaseButton::_is_focus_owner_in_shorcut_context() const {
	if (shortcut_context == ObjectID()) {
		// No context, therefore global - always "in" context.
		return true;
	}

	Node *ctx_node = get_shortcut_context();
	Control *vp_focus = get_focus_owner();

	// If the context is valid and the viewport focus is valid, check if the context is the focus or is a parent of it.
	return ctx_node && vp_focus && (ctx_node == vp_focus || ctx_node->is_ancestor_of(vp_focus));
}

void BaseButton::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_pressed", "pressed"), &BaseButton::set_pressed);
	ClassDB::bind_method(D_METHOD("is_pressed"), &BaseButton::is_pressed);
	ClassDB::bind_method(D_METHOD("set_pressed_no_signal", "pressed"), &BaseButton::set_pressed_no_signal);
	ClassDB::bind_method(D_METHOD("is_hovered"), &BaseButton::is_hovered);
	ClassDB::bind_method(D_METHOD("set_toggle_mode", "enabled"), &BaseButton::set_toggle_mode);
	ClassDB::bind_method(D_METHOD("is_toggle_mode"), &BaseButton::is_toggle_mode);
	ClassDB::bind_method(D_METHOD("set_shortcut_in_tooltip", "enabled"), &BaseButton::set_shortcut_in_tooltip);
	ClassDB::bind_method(D_METHOD("is_shortcut_in_tooltip_enabled"), &BaseButton::is_shortcut_in_tooltip_enabled);
	ClassDB::bind_method(D_METHOD("set_disabled", "disabled"), &BaseButton::set_disabled);
	ClassDB::bind_method(D_METHOD("is_disabled"), &BaseButton::is_disabled);
	ClassDB::bind_method(D_METHOD("set_action_mode", "mode"), &BaseButton::set_action_mode);
	ClassDB::bind_method(D_METHOD("get_action_mode"), &BaseButton::get_action_mode);
	ClassDB::bind_method(D_METHOD("set_button_mask", "mask"), &BaseButton::set_button_mask);
	ClassDB::bind_method(D_METHOD("get_button_mask"), &BaseButton::get_button_mask);
	ClassDB::bind_method(D_METHOD("get_draw_mode"), &BaseButton::get_draw_mode);
	ClassDB::bind_method(D_METHOD("set_keep_pressed_outside", "enabled"), &BaseButton::set_keep_pressed_outside);
	ClassDB::bind_method(D_METHOD("is_keep_pressed_outside"), &BaseButton::is_keep_pressed_outside);

	ClassDB::bind_method(D_METHOD("set_shortcut", "shortcut"), &BaseButton::set_shortcut);
	ClassDB::bind_method(D_METHOD("get_shortcut"), &BaseButton::get_shortcut);

	ClassDB::bind_method(D_METHOD("set_button_group", "button_group"), &BaseButton::set_button_group);
	ClassDB::bind_method(D_METHOD("get_button_group"), &BaseButton::get_button_group);

	ClassDB::bind_method(D_METHOD("set_shortcut_context", "node"), &BaseButton::set_shortcut_context);
	ClassDB::bind_method(D_METHOD("get_shortcut_context"), &BaseButton::get_shortcut_context);

	GDVIRTUAL_BIND(_pressed);
	GDVIRTUAL_BIND(_toggled, "button_pressed");

	ADD_SIGNAL(MethodInfo("pressed"));
	ADD_SIGNAL(MethodInfo("button_up"));
	ADD_SIGNAL(MethodInfo("button_down"));
	ADD_SIGNAL(MethodInfo("toggled", PropertyInfo(Variant::BOOL, "button_pressed")));
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "disabled"), "set_disabled", "is_disabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "toggle_mode"), "set_toggle_mode", "is_toggle_mode");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "shortcut_in_tooltip"), "set_shortcut_in_tooltip", "is_shortcut_in_tooltip_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "pressed"), "set_pressed", "is_pressed");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "action_mode", PROPERTY_HINT_ENUM, "Button Press,Button Release"), "set_action_mode", "get_action_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "button_mask", PROPERTY_HINT_FLAGS, "Mouse Left, Mouse Right, Mouse Middle"), "set_button_mask", "get_button_mask");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "keep_pressed_outside"), "set_keep_pressed_outside", "is_keep_pressed_outside");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "shortcut", PROPERTY_HINT_RESOURCE_TYPE, "Shortcut"), "set_shortcut", "get_shortcut");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "button_group", PROPERTY_HINT_RESOURCE_TYPE, "ButtonGroup"), "set_button_group", "get_button_group");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "shortcut_context", PROPERTY_HINT_RESOURCE_TYPE, "Node"), "set_shortcut_context", "get_shortcut_context");

	BIND_ENUM_CONSTANT(DRAW_NORMAL);
	BIND_ENUM_CONSTANT(DRAW_PRESSED);
	BIND_ENUM_CONSTANT(DRAW_HOVER);
	BIND_ENUM_CONSTANT(DRAW_DISABLED);
	BIND_ENUM_CONSTANT(DRAW_HOVER_PRESSED);

	BIND_ENUM_CONSTANT(ACTION_MODE_BUTTON_PRESS);
	BIND_ENUM_CONSTANT(ACTION_MODE_BUTTON_RELEASE);
}

BaseButton::BaseButton() {
	set_focus_mode(FOCUS_ALL);
}

BaseButton::~BaseButton() {
	if (button_group.is_valid()) {
		button_group->buttons.erase(this);
	}
}

void ButtonGroup::get_buttons(List<BaseButton *> *r_buttons) {
	for (Set<BaseButton *>::Element *E = buttons.front(); E; E = E->next()) {
		r_buttons->push_back(E->get());
	}
}

Array ButtonGroup::_get_buttons() {
	Array btns;
	for (Set<BaseButton *>::Element *E = buttons.front(); E; E = E->next()) {
		btns.push_back(E->get());
	}

	return btns;
}

BaseButton *ButtonGroup::get_pressed_button() {
	for (Set<BaseButton *>::Element *E = buttons.front(); E; E = E->next()) {
		if (E->get()->is_pressed()) {
			return E->get();
		}
	}

	return nullptr;
}

void ButtonGroup::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_pressed_button"), &ButtonGroup::get_pressed_button);
	ClassDB::bind_method(D_METHOD("get_buttons"), &ButtonGroup::_get_buttons);
	ADD_SIGNAL(MethodInfo("pressed", PropertyInfo(Variant::OBJECT, "button")));
}

ButtonGroup::ButtonGroup() {
	set_local_to_scene(true);
}
