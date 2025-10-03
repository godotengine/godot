/**************************************************************************/
/*  editor_scroll_box.cpp                                                 */
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

#include "editor_scroll_box.h"

#include "scene/gui/button.h"
#include "scene/gui/scroll_bar.h"
#include "scene/gui/scroll_container.h"
#include "scene/theme/theme_db.h"

void EditorScrollBox::ensure_control_visible(ObjectID p_id) {
	Control *c = ObjectDB::get_instance<Control>(p_id);
	if (!c) {
		return;
	}

	scroll_container->ensure_control_visible(c);
}

// Use set_control(nullptr) to remove the control used by the ScrollContainer.
void EditorScrollBox::set_control(Control *p_control) {
	if (p_control) {
		ERR_FAIL_COND_MSG(has_control(), "Container already has a control, use set_control(nullptr) to remove it.");
	}
	if (control) {
		control->disconnect(SceneStringName(resized), callable_mp(this, &EditorScrollBox::_update_buttons));
		scroll_container->remove_child(p_control);
		control = nullptr;
	}

	if (!p_control) {
		_update_buttons();
		return;
	}

	if (p_control->get_parent()) {
		p_control->get_parent()->remove_child(p_control);
	}

	control = p_control;
	control->connect(SceneStringName(resized), callable_mp(this, &EditorScrollBox::_update_buttons));
	scroll_container->add_child(p_control);

	_update_buttons();
}

bool EditorScrollBox::has_control() const {
	return control != nullptr;
}

Button *EditorScrollBox::get_right_button() const {
	return is_layout_rtl() ? left_button : right_button;
}

Button *EditorScrollBox::get_left_button() const {
	return is_layout_rtl() ? right_button : left_button;
}

void EditorScrollBox::_scroll(bool p_right) {
	if (is_layout_rtl()) {
		p_right = !p_right;
	}

	HScrollBar *hscroll_bar = scroll_container->get_h_scroll_bar();
	if (Input::get_singleton()->is_key_pressed(Key::CTRL)) {
		hscroll_bar->set_value(p_right ? hscroll_bar->get_max() : 0);
	} else if (Input::get_singleton()->is_key_pressed(Key::SHIFT)) {
		hscroll_bar->set_value(hscroll_bar->get_value() + hscroll_bar->get_page() * (p_right ? 1 : -1));
	} else {
		hscroll_bar->set_value(hscroll_bar->get_value() + (hscroll_bar->get_page() * 0.5) * (p_right ? 1 : -1));
	}
}

// Update the buttons visibility and disabled state.
void EditorScrollBox::_update_buttons() {
	bool show_arrows = control && control->get_size().width > scroll_container->get_size().width;
	left_button->set_visible(show_arrows);
	right_button->set_visible(show_arrows);

	if (show_arrows) {
		_update_disabled_buttons();
	}
}

void EditorScrollBox::_update_buttons_icon_and_tooltip() {
	Button *button = get_left_button();
	button->set_button_icon(theme_cache.arrow_left);
	button->set_accessibility_name(TTRC("Scroll Left"));
	button->set_tooltip_text(TTRC("Scroll Left\nHold Ctrl to scroll to the begin.\nHold Shift to scroll one page."));

	button = get_right_button();
	button->set_button_icon(theme_cache.arrow_right);
	button->set_accessibility_name(TTRC("Scroll Right"));
	button->set_tooltip_text(TTRC("Scroll Right\nHold Ctrl to scroll to the end.\nHold Shift to scroll one page."));
}

void EditorScrollBox::_update_disabled_buttons() {
	HScrollBar *scroll_bar = scroll_container->get_h_scroll_bar();

	get_left_button()->set_disabled(scroll_bar->get_value() == 0);
	get_right_button()->set_disabled(scroll_bar->get_value() + scroll_bar->get_page() == scroll_bar->get_max());
}

void EditorScrollBox::_accessibility_action_scroll_left(const Variant &p_data) {
	_scroll(false);
}

void EditorScrollBox::_accessibility_action_scroll_right(const Variant &p_data) {
	_scroll(true);
}

void EditorScrollBox::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ACCESSIBILITY_UPDATE: {
			RID ae = get_accessibility_element();
			ERR_FAIL_COND(ae.is_null());
			DisplayServer::get_singleton()->accessibility_update_set_role(ae, DisplayServer::AccessibilityRole::ROLE_SCROLL_VIEW);

			DisplayServer::get_singleton()->accessibility_update_add_action(ae, DisplayServer::AccessibilityAction::ACTION_SCROLL_LEFT, callable_mp(this, &EditorScrollBox::_accessibility_action_scroll_left));
			DisplayServer::get_singleton()->accessibility_update_add_action(ae, DisplayServer::AccessibilityAction::ACTION_SCROLL_RIGHT, callable_mp(this, &EditorScrollBox::_accessibility_action_scroll_right));
		} break;

		case NOTIFICATION_THEME_CHANGED:
		case NOTIFICATION_TRANSLATION_CHANGED:
		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED: {
			_update_buttons_icon_and_tooltip();
			_update_buttons();
		} break;

		case NOTIFICATION_RESIZED: {
			_update_buttons();
		} break;
	}
}

void EditorScrollBox::_bind_methods() {
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, EditorScrollBox, arrow_left);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, EditorScrollBox, arrow_right);
}

EditorScrollBox::EditorScrollBox() {
	set_h_size_flags(SIZE_EXPAND_FILL);

	left_button = memnew(Button);
	left_button->set_theme_type_variation("BottomPanelButton");
	left_button->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	left_button->connect(SceneStringName(pressed), callable_mp(this, &EditorScrollBox::_scroll).bind(false));
	add_child(left_button);
	left_button->hide();

	scroll_container = memnew(ScrollContainer);
	scroll_container->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	scroll_container->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_SHOW_NEVER);
	scroll_container->set_vertical_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
	scroll_container->set_force_pass_scroll_events(false);
	scroll_container->set_follow_focus(true);
	add_child(scroll_container);

	HScrollBar *hscroll = scroll_container->get_h_scroll_bar();
	hscroll->connect(CoreStringName(changed), callable_mp(this, &EditorScrollBox::_update_buttons), CONNECT_DEFERRED);
	hscroll->connect(SceneStringName(value_changed), callable_mp(this, &EditorScrollBox::_update_disabled_buttons).unbind(1), CONNECT_DEFERRED);

	right_button = memnew(Button);
	right_button->set_theme_type_variation("BottomPanelButton");
	right_button->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	right_button->connect(SceneStringName(pressed), callable_mp(this, &EditorScrollBox::_scroll).bind(true));
	add_child(right_button);
	right_button->hide();

	_update_buttons_icon_and_tooltip();
}
