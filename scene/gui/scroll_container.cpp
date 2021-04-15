/**************************************************************************/
/*  scroll_container.cpp                                                  */
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

#include "scroll_container.h"

#include "core/config/project_settings.h"
#include "core/os/os.h"
#include "scene/main/window.h"

Size2 ScrollContainer::get_minimum_size() const {
	Size2 min_size;

	// Calculated in this function, as it needs to traverse all child controls once to calculate;
	// and needs to be calculated before being used by update_scrollbars().
	largest_child_min_size = Size2();

	for (int i = 0; i < get_child_count(); i++) {
		Control *c = Object::cast_to<Control>(get_child(i));
		if (!c || !c->is_visible()) {
			continue;
		}
		if (c->is_set_as_top_level()) {
			continue;
		}
		if (c == h_scroll || c == v_scroll) {
			continue;
		}

		Size2 child_min_size = c->get_combined_minimum_size();

		largest_child_min_size.x = MAX(largest_child_min_size.x, child_min_size.x);
		largest_child_min_size.y = MAX(largest_child_min_size.y, child_min_size.y);
	}

	if (horizontal_scroll_mode == SCROLL_MODE_DISABLED) {
		min_size.x = MAX(min_size.x, largest_child_min_size.x);
	}
	if (vertical_scroll_mode == SCROLL_MODE_DISABLED) {
		min_size.y = MAX(min_size.y, largest_child_min_size.y);
	}

	bool h_scroll_show = horizontal_scroll_mode == SCROLL_MODE_SHOW_ALWAYS || (horizontal_scroll_mode == SCROLL_MODE_AUTO && largest_child_min_size.x > min_size.x);
	bool v_scroll_show = vertical_scroll_mode == SCROLL_MODE_SHOW_ALWAYS || (vertical_scroll_mode == SCROLL_MODE_AUTO && largest_child_min_size.y > min_size.y);

	if (h_scroll_show && h_scroll->get_parent() == this) {
		min_size.y += h_scroll->get_minimum_size().y;
	}
	if (v_scroll_show && v_scroll->get_parent() == this) {
		min_size.x += v_scroll->get_minimum_size().x;
	}

	min_size += theme_cache.panel_style->get_minimum_size();
	return min_size;
}

void ScrollContainer::_update_theme_item_cache() {
	Container::_update_theme_item_cache();

	theme_cache.panel_style = get_theme_stylebox(SNAME("panel"));
}

void ScrollContainer::_cancel_drag() {
	set_physics_process_internal(false);
	animating = false;
	drag_touching = false;
	drag_speed = Vector2();
	drag_accum = Vector2();
	drag_from = Vector2();

	if (beyond_deadzone) {
		emit_signal(SNAME("scroll_ended"));
		propagate_notification(NOTIFICATION_SCROLL_END);
		beyond_deadzone = false;
	}
}

void ScrollContainer::_start_inertial_scroll() {
	inertial_scroll_duration_current = inertial_time_left;
	inertial_start = Vector2(h_scroll->get_value(), v_scroll->get_value());
	animating = true;
	set_physics_process_internal(true);
}

void ScrollContainer::_button_scroll(bool p_horizontal, float p_amount) {
	// Cap the inertial target
	if (inertial_target.x < 0) {
		inertial_target.x = 0;
	}
	if (inertial_target.x > (h_scroll->get_max() - h_scroll->get_page())) {
		inertial_target.x = h_scroll->get_max() - h_scroll->get_page();
	}
	if (inertial_target.y < 0) {
		inertial_target.y = 0;
	}
	if (inertial_target.y > (v_scroll->get_max() - v_scroll->get_page())) {
		inertial_target.y = v_scroll->get_max() - v_scroll->get_page();
	}
	_cancel_drag();

	// Multiply the amount by the step (1/8th of a page)
	if (p_horizontal) {
		p_amount *= h_scroll->get_page() * scroll_step;
	} else {
		p_amount *= v_scroll->get_page() * scroll_step;
	}

	// Do scroll
	if (always_smoothed) {
		// Vector2 inertial_target = Vector2(h_scroll->get_value(), v_scroll->get_value());
		if (p_horizontal) {
			inertial_target.x += p_amount;
		} else {
			inertial_target.y += p_amount;
		}
		inertial_time_left = smooth_scroll_duration_button;
		_start_inertial_scroll();
		emit_signal("scroll_started");
		propagate_notification(NOTIFICATION_SCROLL_BEGIN);
	} else {
		if (p_horizontal) {
			h_scroll->set_value(h_scroll->get_value() + p_amount);
		} else {
			v_scroll->set_value(v_scroll->get_value() + p_amount);
		}
	}
}

void ScrollContainer::gui_input(const Ref<InputEvent> &p_gui_input) {
	ERR_FAIL_COND(p_gui_input.is_null());
	_check_expected_scroll();

	double prev_v_scroll = v_scroll->get_value();
	double prev_h_scroll = h_scroll->get_value();
	bool h_scroll_enabled = horizontal_scroll_mode != SCROLL_MODE_DISABLED;
	bool v_scroll_enabled = vertical_scroll_mode != SCROLL_MODE_DISABLED;

	Ref<InputEventMouseButton> mb = p_gui_input;

	if (mb.is_valid()) {
		if (mb->is_pressed()) {
			bool scroll_value_modified = false;

			bool v_scroll_hidden = !v_scroll->is_visible() && vertical_scroll_mode != SCROLL_MODE_SHOW_NEVER;
			if (mb->get_button_index() == MouseButton::WHEEL_UP) {
				// By default, the vertical orientation takes precedence. This is an exception.
				if ((h_scroll_enabled && mb->is_shift_pressed()) || v_scroll_hidden) {
					_button_scroll(true, -mb->get_factor());
					scroll_value_modified = true;
				} else if (v_scroll_enabled) {
					_button_scroll(false, -mb->get_factor());
					scroll_value_modified = true;
				}
			}
			if (mb->get_button_index() == MouseButton::WHEEL_DOWN) {
				if ((h_scroll_enabled && mb->is_shift_pressed()) || v_scroll_hidden) {
					_button_scroll(true, mb->get_factor());
					scroll_value_modified = true;
				} else if (v_scroll_enabled) {
					_button_scroll(false, mb->get_factor());
					scroll_value_modified = true;
				}
			}

			bool h_scroll_hidden = !h_scroll->is_visible() && horizontal_scroll_mode != SCROLL_MODE_SHOW_NEVER;
			if (mb->get_button_index() == MouseButton::WHEEL_LEFT) {
				// By default, the horizontal orientation takes precedence. This is an exception.
				if ((v_scroll_enabled && mb->is_shift_pressed()) || h_scroll_hidden) {
					_button_scroll(false, -mb->get_factor());
					scroll_value_modified = true;
				} else if (h_scroll_enabled) {
					_button_scroll(true, -mb->get_factor());
					scroll_value_modified = true;
				}
			}
			if (mb->get_button_index() == MouseButton::WHEEL_RIGHT) {
				if ((v_scroll_enabled && mb->is_shift_pressed()) || h_scroll_hidden) {
					_button_scroll(false, mb->get_factor());
					scroll_value_modified = true;
				} else if (h_scroll_enabled) {
					_button_scroll(true, mb->get_factor());
					scroll_value_modified = true;
				}
			}

			if (scroll_value_modified && (v_scroll->get_value() != prev_v_scroll || h_scroll->get_value() != prev_h_scroll)) {
				accept_event(); // Accept event if scroll changed.
				return;
			}
		}

		_update_expected_scroll();

		bool is_touchscreen_available = DisplayServer::get_singleton()->is_touchscreen_available();
		if (!is_touchscreen_available) {
			return;
		}

		if (mb->get_button_index() != MouseButton::LEFT) {
			return;
		}

		if (mb->is_pressed()) {
			if (drag_touching) {
				_cancel_drag();
			}

			drag_speed = Vector2();
			drag_accum = Vector2();
			drag_from = Vector2(prev_h_scroll, prev_v_scroll);
			drag_touching = true;
			animating = false;
			beyond_deadzone = false;
			set_physics_process_internal(true);

		} else {
			if (drag_touching) {
				drag_touching = false;
				inertial_time_left = inertial_scroll_duration_touch;
				_start_inertial_scroll();
			}
		}
		return;
	}

	Ref<InputEventMouseMotion> mm = p_gui_input;

	if (mm.is_valid()) {
		if (drag_touching && !animating) {
			Vector2 motion = mm->get_relative();
			drag_accum -= motion;

			if (beyond_deadzone || (h_scroll_enabled && Math::abs(drag_accum.x) > deadzone) || (v_scroll_enabled && Math::abs(drag_accum.y) > deadzone)) {
				if (!beyond_deadzone) {
					propagate_notification(NOTIFICATION_SCROLL_BEGIN);
					emit_signal(SNAME("scroll_started"));

					beyond_deadzone = true;
					// Resetting drag_accum here ensures smooth scrolling after reaching deadzone.
					drag_accum = -motion;
				}
				Vector2 diff = drag_from + drag_accum;
				if (h_scroll_enabled) {
					h_scroll->set_value(diff.x);
				} else {
					drag_accum.x = 0;
				}
				if (v_scroll_enabled) {
					v_scroll->set_value(diff.y);
				} else {
					drag_accum.y = 0;
				}
				drag_speed -= -motion;
			}
		}
	}

	// If a touch event is already being processed, ignore pan events (which can also be valid when using touch)
	if (!drag_touching) {
		Ref<InputEventPanGesture> pan_gesture = p_gui_input;
		if (pan_gesture.is_valid()) {
			if (h_scroll->is_visible_in_tree()) {
				h_scroll->set_value(prev_h_scroll + h_scroll->get_page() * pan_gesture->get_delta().x / 8);
			}
			if (v_scroll->is_visible_in_tree()) {
				v_scroll->set_value(prev_v_scroll + v_scroll->get_page() * pan_gesture->get_delta().y / 8);
			}
		}

		if (v_scroll->get_value() != prev_v_scroll || h_scroll->get_value() != prev_h_scroll) {
			accept_event(); // Accept event if scroll changed.
		}
		return;
	}

	// If the event is not a mouse event, try processing it as page up or page down (this requires focus!)
	if (!mm.is_valid() && !mb.is_valid()) {
		if (p_gui_input->is_action_pressed("ui_page_up", true)) {
			if (v_scroll->is_visible_in_tree()) {
				_button_scroll(false, -1 / scroll_step);
				accept_event();
			}
		}
		if (p_gui_input->is_action_pressed("ui_page_down", true)) {
			if (v_scroll->is_visible_in_tree()) {
				_button_scroll(false, 1 / scroll_step);
				accept_event();
			}
		}

		if (v_scroll->get_value() != prev_v_scroll || h_scroll->get_value() != prev_h_scroll) {
			accept_event(); // Accept event if scroll changed.
		}
	}

	_update_expected_scroll();
}

void ScrollContainer::_update_scrollbar_position() {
	if (!_updating_scrollbars) {
		return;
	}

	Size2 hmin = h_scroll->get_combined_minimum_size();
	Size2 vmin = v_scroll->get_combined_minimum_size();

	h_scroll->set_anchor_and_offset(SIDE_LEFT, ANCHOR_BEGIN, 0);
	h_scroll->set_anchor_and_offset(SIDE_RIGHT, ANCHOR_END, 0);
	h_scroll->set_anchor_and_offset(SIDE_TOP, ANCHOR_END, -hmin.height);
	h_scroll->set_anchor_and_offset(SIDE_BOTTOM, ANCHOR_END, 0);

	v_scroll->set_anchor_and_offset(SIDE_LEFT, ANCHOR_END, -vmin.width);
	v_scroll->set_anchor_and_offset(SIDE_RIGHT, ANCHOR_END, 0);
	v_scroll->set_anchor_and_offset(SIDE_TOP, ANCHOR_BEGIN, 0);
	v_scroll->set_anchor_and_offset(SIDE_BOTTOM, ANCHOR_END, 0);

	_updating_scrollbars = false;
}

void ScrollContainer::_gui_focus_changed(Control *p_control) {
	if (follow_focus && is_ancestor_of(p_control)) {
		ensure_control_visible(p_control);
	}
}

void ScrollContainer::ensure_control_visible(Control *p_control) {
	ERR_FAIL_COND_MSG(!is_ancestor_of(p_control), "Must be an ancestor of the control.");

	_check_expected_scroll();

	Rect2 global_rect = get_global_rect();
	Rect2 other_rect = p_control->get_global_rect();
	float right_margin = v_scroll->is_visible() ? v_scroll->get_size().x : 0.0f;
	float bottom_margin = h_scroll->is_visible() ? h_scroll->get_size().y : 0.0f;

	float h_diff = MAX(MIN(other_rect.position.x, global_rect.position.x), other_rect.position.x + other_rect.size.x - global_rect.size.x + (is_layout_rtl() ? right_margin : 0.0)) - global_rect.position.x;
	float v_diff = MAX(MIN(other_rect.position.y, global_rect.position.y), other_rect.position.y + other_rect.size.y - global_rect.size.y + bottom_margin) - global_rect.position.y;

	if (always_smoothed) {
		if (h_diff != 0 || v_diff != 0) {
			_cancel_drag();
			inertial_target = Vector2(h_scroll->get_value() + h_diff, v_scroll->get_value() + v_diff);
			inertial_time_left = smooth_scroll_duration_button;
			_start_inertial_scroll();
			emit_signal("scroll_started");
			propagate_notification(NOTIFICATION_SCROLL_BEGIN);
		}
	} else {
		h_scroll->set_value(h_scroll->get_value() + h_diff);
		v_scroll->set_value(v_scroll->get_value() + v_diff);
	}

	_update_expected_scroll();
}

void ScrollContainer::_reposition_children() {
	update_scrollbars();
	Size2 size = get_size();
	Point2 ofs;

	size -= theme_cache.panel_style->get_minimum_size();
	ofs += theme_cache.panel_style->get_offset();
	bool rtl = is_layout_rtl();

	if (h_scroll->is_visible_in_tree() && h_scroll->get_parent() == this) { //scrolls may have been moved out for reasons
		size.y -= h_scroll->get_minimum_size().y;
	}

	if (v_scroll->is_visible_in_tree() && v_scroll->get_parent() == this) { //scrolls may have been moved out for reasons
		size.x -= v_scroll->get_minimum_size().x;
	}

	for (int i = 0; i < get_child_count(); i++) {
		Control *c = Object::cast_to<Control>(get_child(i));
		if (!c || !c->is_visible()) {
			continue;
		}
		if (c->is_set_as_top_level()) {
			continue;
		}
		if (c == h_scroll || c == v_scroll) {
			continue;
		}
		Size2 minsize = c->get_combined_minimum_size();

		Rect2 r = Rect2(-Size2(get_h_scroll(), get_v_scroll()), minsize);
		if (c->get_h_size_flags().has_flag(SIZE_EXPAND)) {
			r.size.width = MAX(size.width, minsize.width);
		}
		if (c->get_v_size_flags().has_flag(SIZE_EXPAND)) {
			r.size.height = MAX(size.height, minsize.height);
		}
		r.position += ofs;
		if (rtl && v_scroll->is_visible_in_tree() && v_scroll->get_parent() == this) {
			r.position.x += v_scroll->get_minimum_size().x;
		}
		r.position = r.position.floor();
		fit_child_in_rect(c, r);
	}

	queue_redraw();
}

void ScrollContainer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED:
		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED:
		case NOTIFICATION_TRANSLATION_CHANGED: {
			_updating_scrollbars = true;
			call_deferred(SNAME("_update_scrollbar_position"));
		} break;

		case NOTIFICATION_READY: {
			Viewport *viewport = get_viewport();
			ERR_FAIL_COND(!viewport);
			viewport->connect("gui_focus_changed", callable_mp(this, &ScrollContainer::_gui_focus_changed));
			_reposition_children();
		} break;

		case NOTIFICATION_SORT_CHILDREN: {
			_reposition_children();
		} break;

		case NOTIFICATION_DRAW: {
			draw_style_box(theme_cache.panel_style, Rect2(Vector2(), get_size()));
		} break;

		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			if (animating) {
				_check_expected_scroll();

				inertial_time_left -= get_physics_process_delta_time();
				if (inertial_time_left <= 0) {
					inertial_time_left = 0;
				}

				float normalized_time = inertial_time_left / inertial_scroll_duration_current;
				Vector2 pos = inertial_target.lerp(inertial_start, normalized_time * normalized_time);

				bool turnoff_h = horizontal_scroll_mode == SCROLL_MODE_DISABLED;
				bool turnoff_v = vertical_scroll_mode == SCROLL_MODE_DISABLED;

				// Stop and cap scroll value if reaching the end or beginning
				if (pos.x < 0) {
					pos.x = 0;
					turnoff_h = true;
				}
				if (pos.x > (h_scroll->get_max() - h_scroll->get_page())) {
					pos.x = h_scroll->get_max() - h_scroll->get_page();
					turnoff_h = true;
				}
				if (pos.y < 0) {
					pos.y = 0;
					turnoff_v = true;
				}
				if (pos.y > (v_scroll->get_max() - v_scroll->get_page())) {
					pos.y = v_scroll->get_max() - v_scroll->get_page();
					turnoff_v = true;
				}

				if (horizontal_scroll_mode != SCROLL_MODE_DISABLED) {
					h_scroll->set_value(pos.x);
				}
				if (vertical_scroll_mode != SCROLL_MODE_DISABLED) {
					v_scroll->set_value(pos.y);
				}

				// If the animation is over, or if scrolling has stopped due to reaching the end or beginning, stop animating
				if ((turnoff_h && turnoff_v) || inertial_time_left <= 0) {
					_cancel_drag();
				}
				_update_expected_scroll();
			}
			if (drag_touching) {
				// Set the target to where the scroll will be if it continues for inertial_scroll_duration_touch seconds.
				Vector2 pos = Vector2(h_scroll->get_value(), v_scroll->get_value());
				inertial_target = pos + drag_speed * inertial_scroll_duration_touch * 2000 * get_physics_process_delta_time();
				// Reset drag_speed
				drag_speed = Vector2();
			}
		} break;
	}
}

void ScrollContainer::_check_expected_scroll() {
	Vector2 pos = Vector2(h_scroll->get_value(), v_scroll->get_value());
	if (expected_scroll_value != pos) {
		_cancel_drag();
		inertial_target = pos;
	}
}
void ScrollContainer::_update_expected_scroll() {
	expected_scroll_value = Vector2(h_scroll->get_value(), v_scroll->get_value());
}

void ScrollContainer::update_scrollbars() {
	Size2 size = get_size();
	size -= theme_cache.panel_style->get_minimum_size();

	Size2 hmin = h_scroll->get_combined_minimum_size();
	Size2 vmin = v_scroll->get_combined_minimum_size();

	h_scroll->set_visible(horizontal_scroll_mode == SCROLL_MODE_SHOW_ALWAYS || (horizontal_scroll_mode == SCROLL_MODE_AUTO && largest_child_min_size.width > size.width));
	v_scroll->set_visible(vertical_scroll_mode == SCROLL_MODE_SHOW_ALWAYS || (vertical_scroll_mode == SCROLL_MODE_AUTO && largest_child_min_size.height > size.height));

	h_scroll->set_max(largest_child_min_size.width);
	h_scroll->set_page((v_scroll->is_visible() && v_scroll->get_parent() == this) ? size.width - vmin.width : size.width);

	v_scroll->set_max(largest_child_min_size.height);
	v_scroll->set_page((h_scroll->is_visible() && h_scroll->get_parent() == this) ? size.height - hmin.height : size.height);

	// Avoid scrollbar overlapping.
	h_scroll->set_anchor_and_offset(SIDE_RIGHT, ANCHOR_END, (v_scroll->is_visible() && v_scroll->get_parent() == this) ? -vmin.width : 0);
	v_scroll->set_anchor_and_offset(SIDE_BOTTOM, ANCHOR_END, (h_scroll->is_visible() && h_scroll->get_parent() == this) ? -hmin.height : 0);
}

void ScrollContainer::_scroll_moved(float) {
	queue_sort();
};

void ScrollContainer::set_h_scroll(int p_pos) {
	h_scroll->set_value(p_pos);
	_cancel_drag();
}

int ScrollContainer::get_h_scroll() const {
	return h_scroll->get_value();
}

void ScrollContainer::set_v_scroll(int p_pos) {
	v_scroll->set_value(p_pos);
	_cancel_drag();
}

int ScrollContainer::get_v_scroll() const {
	return v_scroll->get_value();
}

void ScrollContainer::set_horizontal_scroll_mode(ScrollMode p_mode) {
	if (horizontal_scroll_mode == p_mode) {
		return;
	}

	horizontal_scroll_mode = p_mode;
	update_minimum_size();
	queue_sort();
}

ScrollContainer::ScrollMode ScrollContainer::get_horizontal_scroll_mode() const {
	return horizontal_scroll_mode;
}

void ScrollContainer::set_vertical_scroll_mode(ScrollMode p_mode) {
	if (vertical_scroll_mode == p_mode) {
		return;
	}

	vertical_scroll_mode = p_mode;
	update_minimum_size();
	queue_sort();
}

ScrollContainer::ScrollMode ScrollContainer::get_vertical_scroll_mode() const {
	return vertical_scroll_mode;
}

int ScrollContainer::get_deadzone() const {
	return deadzone;
}

void ScrollContainer::set_deadzone(int p_deadzone) {
	deadzone = p_deadzone;
}

bool ScrollContainer::is_following_focus() const {
	return follow_focus;
}

void ScrollContainer::set_follow_focus(bool p_follow) {
	follow_focus = p_follow;
}

bool ScrollContainer::is_always_smoothed() const {
	return always_smoothed;
}

void ScrollContainer::set_always_smoothed(bool p_enabled) {
	always_smoothed = p_enabled;
}

float ScrollContainer::get_scroll_step() const {
	return scroll_step;
}

void ScrollContainer::set_scroll_step(float p_value) {
	scroll_step = p_value;
}

PackedStringArray ScrollContainer::get_configuration_warnings() const {
	PackedStringArray warnings = Container::get_configuration_warnings();

	int found = 0;

	for (int i = 0; i < get_child_count(); i++) {
		Control *c = Object::cast_to<Control>(get_child(i));
		if (!c) {
			continue;
		}
		if (c->is_set_as_top_level()) {
			continue;
		}
		if (c == h_scroll || c == v_scroll) {
			continue;
		}

		found++;
	}

	if (found != 1) {
		warnings.push_back(RTR("ScrollContainer is intended to work with a single child control.\nUse a container as child (VBox, HBox, etc.), or a Control and set the custom minimum size manually."));
	}

	return warnings;
}

HScrollBar *ScrollContainer::get_h_scroll_bar() {
	return h_scroll;
}

VScrollBar *ScrollContainer::get_v_scroll_bar() {
	return v_scroll;
}

void ScrollContainer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_update_scrollbar_position"), &ScrollContainer::_update_scrollbar_position);

	ClassDB::bind_method(D_METHOD("set_h_scroll", "value"), &ScrollContainer::set_h_scroll);
	ClassDB::bind_method(D_METHOD("get_h_scroll"), &ScrollContainer::get_h_scroll);

	ClassDB::bind_method(D_METHOD("set_v_scroll", "value"), &ScrollContainer::set_v_scroll);
	ClassDB::bind_method(D_METHOD("get_v_scroll"), &ScrollContainer::get_v_scroll);

	ClassDB::bind_method(D_METHOD("set_horizontal_scroll_mode", "enable"), &ScrollContainer::set_horizontal_scroll_mode);
	ClassDB::bind_method(D_METHOD("get_horizontal_scroll_mode"), &ScrollContainer::get_horizontal_scroll_mode);

	ClassDB::bind_method(D_METHOD("set_vertical_scroll_mode", "enable"), &ScrollContainer::set_vertical_scroll_mode);
	ClassDB::bind_method(D_METHOD("get_vertical_scroll_mode"), &ScrollContainer::get_vertical_scroll_mode);

	ClassDB::bind_method(D_METHOD("set_deadzone", "deadzone"), &ScrollContainer::set_deadzone);
	ClassDB::bind_method(D_METHOD("get_deadzone"), &ScrollContainer::get_deadzone);

	ClassDB::bind_method(D_METHOD("set_follow_focus", "enabled"), &ScrollContainer::set_follow_focus);
	ClassDB::bind_method(D_METHOD("is_following_focus"), &ScrollContainer::is_following_focus);
	ClassDB::bind_method(D_METHOD("set_always_smoothed", "enabled"), &ScrollContainer::set_always_smoothed);
	ClassDB::bind_method(D_METHOD("is_always_smoothed"), &ScrollContainer::is_always_smoothed);
	ClassDB::bind_method(D_METHOD("set_scroll_step", "value"), &ScrollContainer::set_scroll_step);
	ClassDB::bind_method(D_METHOD("get_scroll_step"), &ScrollContainer::get_scroll_step);

	ClassDB::bind_method(D_METHOD("get_h_scroll_bar"), &ScrollContainer::get_h_scroll_bar);
	ClassDB::bind_method(D_METHOD("get_v_scroll_bar"), &ScrollContainer::get_v_scroll_bar);
	ClassDB::bind_method(D_METHOD("ensure_control_visible", "control"), &ScrollContainer::ensure_control_visible);

	ADD_SIGNAL(MethodInfo("scroll_started"));
	ADD_SIGNAL(MethodInfo("scroll_ended"));

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "follow_focus"), "set_follow_focus", "is_following_focus");

	ADD_GROUP("Scroll", "scroll_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "scroll_horizontal", PROPERTY_HINT_NONE, "suffix:px"), "set_h_scroll", "get_h_scroll");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "scroll_vertical", PROPERTY_HINT_NONE, "suffix:px"), "set_v_scroll", "get_v_scroll");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "horizontal_scroll_mode", PROPERTY_HINT_ENUM, "Disabled,Auto,Always Show,Never Show"), "set_horizontal_scroll_mode", "get_horizontal_scroll_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "vertical_scroll_mode", PROPERTY_HINT_ENUM, "Disabled,Auto,Always Show,Never Show"), "set_vertical_scroll_mode", "get_vertical_scroll_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "scroll_deadzone"), "set_deadzone", "get_deadzone");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "scroll_smoothed"), "set_always_smoothed", "is_always_smoothed");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "scroll_step"), "set_scroll_step", "get_scroll_step");

	BIND_ENUM_CONSTANT(SCROLL_MODE_DISABLED);
	BIND_ENUM_CONSTANT(SCROLL_MODE_AUTO);
	BIND_ENUM_CONSTANT(SCROLL_MODE_SHOW_ALWAYS);
	BIND_ENUM_CONSTANT(SCROLL_MODE_SHOW_NEVER);

	GLOBAL_DEF("gui/scroll/default_scroll_step", 0.125);
	GLOBAL_DEF("gui/scroll/default_scroll_deadzone", 0);
	GLOBAL_DEF("gui/scroll/default_scroll_smoothed", false);
	GLOBAL_DEF("gui/scroll/smooth_scroll_duration_button", 0.15);
	GLOBAL_DEF("gui/scroll/inertial_scroll_duration_touch", 1.5);
};

ScrollContainer::ScrollContainer() {
	h_scroll = memnew(HScrollBar);
	h_scroll->set_name("_h_scroll");
	add_child(h_scroll, false, INTERNAL_MODE_BACK);
	h_scroll->connect("value_changed", callable_mp(this, &ScrollContainer::_scroll_moved));

	v_scroll = memnew(VScrollBar);
	v_scroll->set_name("_v_scroll");
	add_child(v_scroll, false, INTERNAL_MODE_BACK);
	v_scroll->connect("value_changed", callable_mp(this, &ScrollContainer::_scroll_moved));

	scroll_step = GLOBAL_GET("gui/scroll/default_scroll_step");
	deadzone = GLOBAL_GET("gui/scroll/default_scroll_deadzone");
	always_smoothed = GLOBAL_GET("gui/scroll/default_scroll_smoothed");
	inertial_scroll_duration_touch = GLOBAL_GET("gui/scroll/inertial_scroll_duration_touch");
	smooth_scroll_duration_button = GLOBAL_GET("gui/scroll/smooth_scroll_duration_button");

	set_focus_mode(FOCUS_CLICK);
	set_clip_contents(true);
};
