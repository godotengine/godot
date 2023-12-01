/**************************************************************************/
/*  spin_box.cpp                                                          */
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

#include "spin_box.h"

#include "core/input/input.h"
#include "core/math/expression.h"
#include "scene/theme/theme_db.h"

Size2 SpinBox::get_minimum_size() const {
	Size2 ms = line_edit->get_combined_minimum_size();
	ms.width += last_w;
	return ms;
}

void SpinBox::_update_text(bool p_keep_line_edit) {
	String value = String::num(get_value(), Math::range_step_decimals(get_step()));
	if (is_localizing_numeral_system()) {
		value = TS->format_number(value);
	}

	if (!line_edit->has_focus()) {
		if (!prefix.is_empty()) {
			value = prefix + " " + value;
		}
		if (!suffix.is_empty()) {
			value += " " + suffix;
		}
	}

	if (p_keep_line_edit && value == last_updated_text && value != line_edit->get_text()) {
		return;
	}

	line_edit->set_text_with_selection(value);
	last_updated_text = value;
}

void SpinBox::_text_submitted(const String &p_string) {
	Ref<Expression> expr;
	expr.instantiate();

	// Convert commas ',' to dots '.' for French/German etc. keyboard layouts.
	String text = p_string.replace(",", ".");
	text = text.replace(";", ",");
	text = TS->parse_number(text);
	// Ignore the prefix and suffix in the expression.
	text = text.trim_prefix(prefix + " ").trim_suffix(" " + suffix);

	Error err = expr->parse(text);
	if (err != OK) {
		// If the expression failed try without converting commas to dots - they might have been for parameter separation.
		text = p_string;
		text = TS->parse_number(text);
		text = text.trim_prefix(prefix + " ").trim_suffix(" " + suffix);

		err = expr->parse(text);
		if (err != OK) {
			return;
		}
	}

	Variant value = expr->execute(Array(), nullptr, false, true);
	if (value.get_type() != Variant::NIL) {
		set_value(value);
	}
	_update_text();
}

void SpinBox::_text_changed(const String &p_string) {
	int cursor_pos = line_edit->get_caret_column();

	_text_submitted(p_string);

	// Line edit 'set_text' method resets the cursor position so we need to undo that.
	line_edit->set_caret_column(cursor_pos);
}

LineEdit *SpinBox::get_line_edit() {
	return line_edit;
}

void SpinBox::_line_edit_input(const Ref<InputEvent> &p_event) {
}

void SpinBox::_range_click_timeout() {
	if (!drag.enabled && Input::get_singleton()->is_mouse_button_pressed(MouseButton::LEFT)) {
		bool up = get_local_mouse_position().y < (get_size().height / 2);
		double step = get_custom_arrow_step() != 0.0 ? get_custom_arrow_step() : get_step();
		set_value(get_value() + (up ? step : -step));

		if (range_click_timer->is_one_shot()) {
			range_click_timer->set_wait_time(0.075);
			range_click_timer->set_one_shot(false);
			range_click_timer->start();
		}

	} else {
		range_click_timer->stop();
	}
}

void SpinBox::_release_mouse() {
	if (drag.enabled) {
		drag.enabled = false;
		Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_HIDDEN);
		warp_mouse(drag.capture_pos);
		Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_VISIBLE);
	}
}

void SpinBox::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	if (!is_editable()) {
		return;
	}

	Ref<InputEventMouseButton> mb = p_event;

	double step = get_custom_arrow_step() != 0.0 ? get_custom_arrow_step() : get_step();

	if (mb.is_valid() && mb->is_pressed()) {
		bool up = mb->get_position().y < (get_size().height / 2);

		switch (mb->get_button_index()) {
			case MouseButton::LEFT: {
				line_edit->grab_focus();

				set_value(get_value() + (up ? step : -step));

				range_click_timer->set_wait_time(0.6);
				range_click_timer->set_one_shot(true);
				range_click_timer->start();

				drag.allowed = true;
				drag.capture_pos = mb->get_position();
			} break;
			case MouseButton::RIGHT: {
				line_edit->grab_focus();
				set_value((up ? get_max() : get_min()));
			} break;
			case MouseButton::WHEEL_UP: {
				if (line_edit->has_focus()) {
					set_value(get_value() + step * mb->get_factor());
					accept_event();
				}
			} break;
			case MouseButton::WHEEL_DOWN: {
				if (line_edit->has_focus()) {
					set_value(get_value() - step * mb->get_factor());
					accept_event();
				}
			} break;
			default:
				break;
		}
	}

	if (mb.is_valid() && !mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
		//set_default_cursor_shape(CURSOR_ARROW);
		range_click_timer->stop();
		_release_mouse();
		drag.allowed = false;
		line_edit->clear_pending_select_all_on_focus();
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid() && (mm->get_button_mask().has_flag(MouseButtonMask::LEFT))) {
		if (drag.enabled) {
			drag.diff_y += mm->get_relative().y;
			double diff_y = -0.01 * Math::pow(ABS(drag.diff_y), 1.8) * SIGN(drag.diff_y);
			set_value(CLAMP(drag.base_val + step * diff_y, get_min(), get_max()));
		} else if (drag.allowed && drag.capture_pos.distance_to(mm->get_position()) > 2) {
			Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_CAPTURED);
			drag.enabled = true;
			drag.base_val = get_value();
			drag.diff_y = 0;
		}
	}
}

void SpinBox::_line_edit_focus_enter() {
	int col = line_edit->get_caret_column();
	_update_text();
	line_edit->set_caret_column(col);

	// LineEdit text might change and it clears any selection. Have to re-select here.
	if (line_edit->is_select_all_on_focus() && !Input::get_singleton()->is_mouse_button_pressed(MouseButton::LEFT)) {
		line_edit->select_all();
	}
}

void SpinBox::_line_edit_focus_exit() {
	// Discontinue because the focus_exit was caused by left-clicking the arrows.
	const Viewport *viewport = get_viewport();
	if (!viewport || viewport->gui_get_focus_owner() == get_line_edit()) {
		return;
	}
	// Discontinue because the focus_exit was caused by right-click context menu.
	if (line_edit->is_menu_visible()) {
		return;
	}
	// Discontinue because the focus_exit was caused by canceling.
	if (Input::get_singleton()->is_action_pressed("ui_cancel")) {
		_update_text();
		return;
	}

	_text_submitted(line_edit->get_text());
}

inline void SpinBox::_adjust_width_for_icon(const Ref<Texture2D> &icon) {
	int w = icon->get_width();
	if ((w != last_w)) {
		line_edit->set_offset(SIDE_LEFT, 0);
		line_edit->set_offset(SIDE_RIGHT, -w);
		last_w = w;
	}
}

void SpinBox::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			_update_text(true);
			_adjust_width_for_icon(theme_cache.updown_icon);

			RID ci = get_canvas_item();
			Size2i size = get_size();

			if (is_layout_rtl()) {
				theme_cache.updown_icon->draw(ci, Point2i(0, (size.height - theme_cache.updown_icon->get_height()) / 2));
			} else {
				theme_cache.updown_icon->draw(ci, Point2i(size.width - theme_cache.updown_icon->get_width(), (size.height - theme_cache.updown_icon->get_height()) / 2));
			}
		} break;

		case NOTIFICATION_ENTER_TREE: {
			_adjust_width_for_icon(theme_cache.updown_icon);
			_update_text();
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED:
			drag.allowed = false;
			[[fallthrough]];
		case NOTIFICATION_EXIT_TREE: {
			_release_mouse();
		} break;

		case NOTIFICATION_TRANSLATION_CHANGED: {
			queue_redraw();
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			call_deferred(SNAME("update_minimum_size"));
			get_line_edit()->call_deferred(SNAME("update_minimum_size"));
		} break;

		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED: {
			queue_redraw();
		} break;
	}
}

void SpinBox::set_horizontal_alignment(HorizontalAlignment p_alignment) {
	line_edit->set_horizontal_alignment(p_alignment);
}

HorizontalAlignment SpinBox::get_horizontal_alignment() const {
	return line_edit->get_horizontal_alignment();
}

void SpinBox::set_suffix(const String &p_suffix) {
	if (suffix == p_suffix) {
		return;
	}

	suffix = p_suffix;
	_update_text();
}

String SpinBox::get_suffix() const {
	return suffix;
}

void SpinBox::set_prefix(const String &p_prefix) {
	if (prefix == p_prefix) {
		return;
	}

	prefix = p_prefix;
	_update_text();
}

String SpinBox::get_prefix() const {
	return prefix;
}

void SpinBox::set_update_on_text_changed(bool p_enabled) {
	if (update_on_text_changed == p_enabled) {
		return;
	}

	update_on_text_changed = p_enabled;

	if (p_enabled) {
		line_edit->connect("text_changed", callable_mp(this, &SpinBox::_text_changed), CONNECT_DEFERRED);
	} else {
		line_edit->disconnect("text_changed", callable_mp(this, &SpinBox::_text_changed));
	}
}

bool SpinBox::get_update_on_text_changed() const {
	return update_on_text_changed;
}

void SpinBox::set_select_all_on_focus(bool p_enabled) {
	line_edit->set_select_all_on_focus(p_enabled);
}

bool SpinBox::is_select_all_on_focus() const {
	return line_edit->is_select_all_on_focus();
}

void SpinBox::set_editable(bool p_enabled) {
	line_edit->set_editable(p_enabled);
}

bool SpinBox::is_editable() const {
	return line_edit->is_editable();
}

void SpinBox::apply() {
	_text_submitted(line_edit->get_text());
}

void SpinBox::set_custom_arrow_step(double p_custom_arrow_step) {
	custom_arrow_step = p_custom_arrow_step;
}

double SpinBox::get_custom_arrow_step() const {
	return custom_arrow_step;
}

void SpinBox::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_horizontal_alignment", "alignment"), &SpinBox::set_horizontal_alignment);
	ClassDB::bind_method(D_METHOD("get_horizontal_alignment"), &SpinBox::get_horizontal_alignment);
	ClassDB::bind_method(D_METHOD("set_suffix", "suffix"), &SpinBox::set_suffix);
	ClassDB::bind_method(D_METHOD("get_suffix"), &SpinBox::get_suffix);
	ClassDB::bind_method(D_METHOD("set_prefix", "prefix"), &SpinBox::set_prefix);
	ClassDB::bind_method(D_METHOD("get_prefix"), &SpinBox::get_prefix);
	ClassDB::bind_method(D_METHOD("set_editable", "enabled"), &SpinBox::set_editable);
	ClassDB::bind_method(D_METHOD("set_custom_arrow_step", "arrow_step"), &SpinBox::set_custom_arrow_step);
	ClassDB::bind_method(D_METHOD("get_custom_arrow_step"), &SpinBox::get_custom_arrow_step);
	ClassDB::bind_method(D_METHOD("is_editable"), &SpinBox::is_editable);
	ClassDB::bind_method(D_METHOD("set_update_on_text_changed", "enabled"), &SpinBox::set_update_on_text_changed);
	ClassDB::bind_method(D_METHOD("get_update_on_text_changed"), &SpinBox::get_update_on_text_changed);
	ClassDB::bind_method(D_METHOD("set_select_all_on_focus", "enabled"), &SpinBox::set_select_all_on_focus);
	ClassDB::bind_method(D_METHOD("is_select_all_on_focus"), &SpinBox::is_select_all_on_focus);
	ClassDB::bind_method(D_METHOD("apply"), &SpinBox::apply);
	ClassDB::bind_method(D_METHOD("get_line_edit"), &SpinBox::get_line_edit);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "alignment", PROPERTY_HINT_ENUM, "Left,Center,Right,Fill"), "set_horizontal_alignment", "get_horizontal_alignment");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "editable"), "set_editable", "is_editable");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "update_on_text_changed"), "set_update_on_text_changed", "get_update_on_text_changed");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "prefix"), "set_prefix", "get_prefix");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "suffix"), "set_suffix", "get_suffix");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "custom_arrow_step", PROPERTY_HINT_RANGE, "0,10000,0.0001,or_greater"), "set_custom_arrow_step", "get_custom_arrow_step");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "select_all_on_focus"), "set_select_all_on_focus", "is_select_all_on_focus");

	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, SpinBox, updown_icon, "updown");
}

SpinBox::SpinBox() {
	line_edit = memnew(LineEdit);
	add_child(line_edit, false, INTERNAL_MODE_FRONT);

	line_edit->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	line_edit->set_mouse_filter(MOUSE_FILTER_PASS);
	line_edit->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_LEFT);

	line_edit->connect("text_submitted", callable_mp(this, &SpinBox::_text_submitted), CONNECT_DEFERRED);
	line_edit->connect("focus_entered", callable_mp(this, &SpinBox::_line_edit_focus_enter), CONNECT_DEFERRED);
	line_edit->connect("focus_exited", callable_mp(this, &SpinBox::_line_edit_focus_exit), CONNECT_DEFERRED);
	line_edit->connect("gui_input", callable_mp(this, &SpinBox::_line_edit_input));

	range_click_timer = memnew(Timer);
	range_click_timer->connect("timeout", callable_mp(this, &SpinBox::_range_click_timeout));
	add_child(range_click_timer, false, INTERNAL_MODE_FRONT);
}
