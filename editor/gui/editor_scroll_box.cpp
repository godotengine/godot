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

#include "core/os/keyboard.h"
#include "scene/gui/button.h"
#include "scene/gui/scroll_bar.h"
#include "scene/gui/scroll_container.h"
#include "scene/theme/theme_db.h"

void EditorScrollBox::ensure_control_visible(Control *p_control) {
	ERR_FAIL_NULL(p_control);
	if (scroll_container->is_ancestor_of(p_control)) {
		callable_mp(scroll_container, &ScrollContainer::ensure_control_visible).call_deferred(p_control);
	}
}

void EditorScrollBox::set_control(Control *p_control) {
	if (p_control) {
		ERR_FAIL_COND_MSG(has_control(), "Container already has a control, use set_control(null) to remove it.");
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
	if (control) {
		return true;
	}
	return false;
}

void EditorScrollBox::_update_scroll_container() {
	ScrollBar *enabled_scroll = is_vertical() ? (ScrollBar *)scroll_container->get_v_scroll_bar() : (ScrollBar *)scroll_container->get_h_scroll_bar();
	ScrollBar *disabled_scroll = is_vertical() ? (ScrollBar *)scroll_container->get_h_scroll_bar() : (ScrollBar *)scroll_container->get_v_scroll_bar();

	Callable update_buttons = callable_mp(this, &EditorScrollBox::_update_buttons);
	Callable update_disabled = callable_mp(this, &EditorScrollBox::_update_disabled_buttons).unbind(1);

	if (!enabled_scroll->is_connected(CoreStringName(changed), update_buttons)) {
		enabled_scroll->connect(CoreStringName(changed), update_buttons, CONNECT_DEFERRED);
	}

	if (!enabled_scroll->is_connected(SceneStringName(value_changed), update_disabled)) {
		enabled_scroll->connect(SceneStringName(value_changed), update_disabled, CONNECT_DEFERRED);
	}

	if (disabled_scroll->is_connected(CoreStringName(changed), update_buttons)) {
		disabled_scroll->disconnect(CoreStringName(changed), update_buttons);
	}

	if (disabled_scroll->is_connected(SceneStringName(value_changed), update_disabled)) {
		disabled_scroll->disconnect(SceneStringName(value_changed), update_disabled);
	}

	if (is_vertical()) {
		scroll_container->set_v_size_flags(Control::SIZE_EXPAND_FILL);
		scroll_container->set_h_size_flags(Control::SIZE_FILL);
		scroll_container->set_vertical_scroll_mode(ScrollContainer::SCROLL_MODE_SHOW_NEVER);
		scroll_container->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
	} else {
		scroll_container->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		scroll_container->set_v_size_flags(Control::SIZE_FILL);
		scroll_container->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_SHOW_NEVER);
		scroll_container->set_vertical_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
	}
}

void EditorScrollBox::_scroll(bool p_right) {
	if (!is_vertical() && is_layout_rtl()) {
		p_right = !p_right;
	}

	ScrollBar *scroll_bar = is_vertical() ? (ScrollBar *)scroll_container->get_v_scroll_bar() : (ScrollBar *)scroll_container->get_h_scroll_bar();
	if (Input::get_singleton()->is_key_pressed(Key::CTRL)) {
		scroll_bar->set_value(p_right ? scroll_bar->get_max() : 0);
	} else if (Input::get_singleton()->is_key_pressed(Key::SHIFT)) {
		scroll_bar->set_value(scroll_bar->get_value() + scroll_bar->get_page() * (p_right ? 1 : -1));
	} else {
		scroll_bar->set_value(scroll_bar->get_value() + (scroll_bar->get_page() * 0.5) * (p_right ? 1 : -1));
	}
}

void EditorScrollBox::_update_buttons() {
	bool show_arrows = control && control->get_size()[is_vertical() ? 1 : 0] > scroll_container->get_size()[is_vertical() ? 1 : 0];
	first_button->set_visible(show_arrows);
	second_button->set_visible(show_arrows);

	if (show_arrows) {
		_update_disabled_buttons();
	}
}

void EditorScrollBox::_update_buttons_icon_and_tooltip() {
	if (!is_vertical()) {
		if (is_layout_rtl()) {
			second_button->set_button_icon(theme_cache.arrow_left);
			second_button->set_accessibility_name(TTRC("Scroll Left"));
			second_button->set_tooltip_text(TTRC("Scroll Left\nHold Ctrl to scroll to the begin.\nHold Shift to scroll one page."));
			first_button->set_button_icon(theme_cache.arrow_right);
			first_button->set_accessibility_name(TTRC("Scroll Right"));
			first_button->set_tooltip_text(TTRC("Scroll Right\nHold Ctrl to scroll to the end.\nHold Shift to scroll one page."));
		} else {
			first_button->set_button_icon(theme_cache.arrow_left);
			first_button->set_accessibility_name(TTRC("Scroll Left"));
			first_button->set_tooltip_text(TTRC("Scroll Left\nHold Ctrl to scroll to the begin.\nHold Shift to scroll one page."));
			second_button->set_button_icon(theme_cache.arrow_right);
			second_button->set_accessibility_name(TTRC("Scroll Right"));
			second_button->set_tooltip_text(TTRC("Scroll Right\nHold Ctrl to scroll to the end.\nHold Shift to scroll one page."));
		}
	} else {
		first_button->set_button_icon(theme_cache.arrow_up);
		first_button->set_accessibility_name(TTRC("Scroll Up"));
		first_button->set_tooltip_text(TTRC("Scroll Up\nHold Ctrl to scroll to the begin.\nHold Shift to scroll one page."));
		second_button->set_button_icon(theme_cache.arrow_down);
		second_button->set_accessibility_name(TTRC("Scroll Down"));
		second_button->set_tooltip_text(TTRC("Scroll Down\nHold Ctrl to scroll to the end.\nHold Shift to scroll one page."));
	}
}

void EditorScrollBox::_update_disabled_buttons() {
	ScrollBar *scroll_bar = is_vertical() ? (ScrollBar *)scroll_container->get_v_scroll_bar() : (ScrollBar *)scroll_container->get_h_scroll_bar();
	if (!is_vertical() && is_layout_rtl()) {
		first_button->set_disabled(scroll_bar->get_value() + scroll_bar->get_page() == scroll_bar->get_max());
		second_button->set_disabled(scroll_bar->get_value() == 0);
	} else {
		first_button->set_disabled(scroll_bar->get_value() == 0);
		second_button->set_disabled(scroll_bar->get_value() + scroll_bar->get_page() == scroll_bar->get_max());
	}
}

bool EditorScrollBox::_set(const StringName &p_name, const Variant &p_value) {
	if (p_name == "vertical") {
		if ((bool)p_value == is_vertical()) {
			return true;
		}
		set_vertical((bool)p_value);
		if (is_vertical() != (bool)p_value) {
			return true;
		}

		// switch the size flags
		if (is_vertical()) {
			set_v_size_flags(get_h_size_flags());
			set_h_size_flags(SIZE_FILL);
		} else {
			set_h_size_flags(get_v_size_flags());
			set_v_size_flags(SIZE_FILL);
		}

		_update_buttons_icon_and_tooltip();
		_update_buttons();
		_update_scroll_container();

		emit_signal(SNAME("vertical_changed"), p_value);
		return true;
	}
	return false;
}

void EditorScrollBox::_validate_property(PropertyInfo &p_property) const {
	if (is_fixed && p_property.name == "vertical") {
		p_property.usage = PROPERTY_USAGE_NONE;
	}
}

void EditorScrollBox::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_TRANSLATION_CHANGED:
		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED: {
			_update_scroll_container();
		}
			[[fallthrough]];

		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			_update_buttons_icon_and_tooltip();
		}
			[[fallthrough]];

		case NOTIFICATION_RESIZED: {
			_update_buttons();
		} break;
	}
}

void EditorScrollBox::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_control", "control"), &EditorScrollBox::set_control);
	ClassDB::bind_method(D_METHOD("get_control"), &EditorScrollBox::get_control);
	ClassDB::bind_method(D_METHOD("has_control"), &EditorScrollBox::has_control);
	ClassDB::bind_method(D_METHOD("ensure_control_visible", "control"), &EditorScrollBox::ensure_control_visible);
	ClassDB::bind_method(D_METHOD("get_first_button"), &EditorScrollBox::get_first_button);
	ClassDB::bind_method(D_METHOD("get_second_button"), &EditorScrollBox::get_second_button);
	ClassDB::bind_method(D_METHOD("get_scroll_container"), &EditorScrollBox::get_scroll_container);

	ADD_SIGNAL(MethodInfo(SNAME("vertical_changed"), PropertyInfo(Variant::BOOL, "is_vertical")));

	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, EditorScrollBox, arrow_left);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, EditorScrollBox, arrow_right);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, EditorScrollBox, arrow_up);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, EditorScrollBox, arrow_down);
}

EditorScrollBox::EditorScrollBox(bool p_vertical) {
	set_vertical(p_vertical);

	if (p_vertical) {
		set_v_size_flags(SIZE_EXPAND_FILL);
	} else {
		set_h_size_flags(SIZE_EXPAND_FILL);
	}

	first_button = memnew(Button);
	first_button->set_theme_type_variation("BottomPanelButton");
	first_button->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	first_button->connect(SceneStringName(pressed), callable_mp(this, &EditorScrollBox::_scroll).bind(false));
	add_child(first_button);
	first_button->hide();

	scroll_container = memnew(ScrollContainer);
	scroll_container->set_force_pass_scroll_events(false);
	scroll_container->set_follow_focus(true);
	add_child(scroll_container);

	second_button = memnew(Button);
	second_button->set_theme_type_variation("BottomPanelButton");
	second_button->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	second_button->connect(SceneStringName(pressed), callable_mp(this, &EditorScrollBox::_scroll).bind(true));
	add_child(second_button);
	second_button->hide();
}
