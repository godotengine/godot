/*************************************************************************/
/*  scroll_container.cpp                                                 */
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

#include "scroll_container.h"
#include "core/os/os.h"
#include "scene/main/window.h"

Size2 ScrollContainer::get_minimum_size() const {
	Ref<StyleBox> sb = get_theme_stylebox(SNAME("bg"));
	Size2 min_size;

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
		Size2 minsize = c->get_combined_minimum_size();

		if (!scroll_h) {
			min_size.x = MAX(min_size.x, minsize.x);
		}
		if (!scroll_v) {
			min_size.y = MAX(min_size.y, minsize.y);
		}
	}

	if (h_scroll->is_visible_in_tree()) {
		min_size.y += h_scroll->get_minimum_size().y;
	}
	if (v_scroll->is_visible_in_tree()) {
		min_size.x += v_scroll->get_minimum_size().x;
	}
	min_size += sb->get_minimum_size();
	return min_size;
}

void ScrollContainer::_cancel_drag() {
	set_physics_process_internal(false);
	drag_touching_deaccel = false;
	drag_touching = false;
	drag_speed = Vector2();
	drag_accum = Vector2();
	last_drag_accum = Vector2();
	drag_from = Vector2();

	if (beyond_deadzone) {
		emit_signal(SNAME("scroll_ended"));
		propagate_notification(NOTIFICATION_SCROLL_END);
		beyond_deadzone = false;
	}
}

void ScrollContainer::gui_input(const Ref<InputEvent> &p_gui_input) {
	ERR_FAIL_COND(p_gui_input.is_null());

	double prev_v_scroll = v_scroll->get_value();
	double prev_h_scroll = h_scroll->get_value();

	Ref<InputEventMouseButton> mb = p_gui_input;

	if (mb.is_valid()) {
		if (mb->get_button_index() == MouseButton::WHEEL_UP && mb->is_pressed()) {
			// only horizontal is enabled, scroll horizontally
			if (h_scroll->is_visible() && (!v_scroll->is_visible() || mb->is_shift_pressed())) {
				h_scroll->set_value(h_scroll->get_value() - h_scroll->get_page() / 8 * mb->get_factor());
			} else if (v_scroll->is_visible_in_tree()) {
				v_scroll->set_value(v_scroll->get_value() - v_scroll->get_page() / 8 * mb->get_factor());
			}
		}

		if (mb->get_button_index() == MouseButton::WHEEL_DOWN && mb->is_pressed()) {
			// only horizontal is enabled, scroll horizontally
			if (h_scroll->is_visible() && (!v_scroll->is_visible() || mb->is_shift_pressed())) {
				h_scroll->set_value(h_scroll->get_value() + h_scroll->get_page() / 8 * mb->get_factor());
			} else if (v_scroll->is_visible()) {
				v_scroll->set_value(v_scroll->get_value() + v_scroll->get_page() / 8 * mb->get_factor());
			}
		}

		if (mb->get_button_index() == MouseButton::WHEEL_LEFT && mb->is_pressed()) {
			if (h_scroll->is_visible_in_tree()) {
				h_scroll->set_value(h_scroll->get_value() - h_scroll->get_page() * mb->get_factor() / 8);
			}
		}

		if (mb->get_button_index() == MouseButton::WHEEL_RIGHT && mb->is_pressed()) {
			if (h_scroll->is_visible_in_tree()) {
				h_scroll->set_value(h_scroll->get_value() + h_scroll->get_page() * mb->get_factor() / 8);
			}
		}

		if (v_scroll->get_value() != prev_v_scroll || h_scroll->get_value() != prev_h_scroll) {
			accept_event(); //accept event if scroll changed
		}

		if (!DisplayServer::get_singleton()->screen_is_touchscreen(DisplayServer::get_singleton()->window_get_current_screen(get_viewport()->get_window_id()))) {
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
			last_drag_accum = Vector2();
			drag_from = Vector2(h_scroll->get_value(), v_scroll->get_value());
			drag_touching = !DisplayServer::get_singleton()->screen_is_touchscreen(DisplayServer::get_singleton()->window_get_current_screen(get_viewport()->get_window_id()));
			drag_touching_deaccel = false;
			beyond_deadzone = false;
			time_since_motion = 0;
			if (drag_touching) {
				set_physics_process_internal(true);
				time_since_motion = 0;
			}

		} else {
			if (drag_touching) {
				if (drag_speed == Vector2()) {
					_cancel_drag();
				} else {
					drag_touching_deaccel = true;
				}
			}
		}
	}

	Ref<InputEventMouseMotion> mm = p_gui_input;

	if (mm.is_valid()) {
		if (drag_touching && !drag_touching_deaccel) {
			Vector2 motion = mm->get_relative();
			drag_accum -= motion;

			if (beyond_deadzone || (scroll_h && Math::abs(drag_accum.x) > deadzone) || (scroll_v && Math::abs(drag_accum.y) > deadzone)) {
				if (!beyond_deadzone) {
					propagate_notification(NOTIFICATION_SCROLL_BEGIN);
					emit_signal(SNAME("scroll_started"));

					beyond_deadzone = true;
					// resetting drag_accum here ensures smooth scrolling after reaching deadzone
					drag_accum = -motion;
				}
				Vector2 diff = drag_from + drag_accum;
				if (scroll_h) {
					h_scroll->set_value(diff.x);
				} else {
					drag_accum.x = 0;
				}
				if (scroll_v) {
					v_scroll->set_value(diff.y);
				} else {
					drag_accum.y = 0;
				}
				time_since_motion = 0;
			}
		}
	}

	Ref<InputEventPanGesture> pan_gesture = p_gui_input;
	if (pan_gesture.is_valid()) {
		if (h_scroll->is_visible_in_tree()) {
			h_scroll->set_value(h_scroll->get_value() + h_scroll->get_page() * pan_gesture->get_delta().x / 8);
		}
		if (v_scroll->is_visible_in_tree()) {
			v_scroll->set_value(v_scroll->get_value() + v_scroll->get_page() * pan_gesture->get_delta().y / 8);
		}
	}

	if (v_scroll->get_value() != prev_v_scroll || h_scroll->get_value() != prev_h_scroll) {
		accept_event(); //accept event if scroll changed
	}
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

	Rect2 global_rect = get_global_rect();
	Rect2 other_rect = p_control->get_global_rect();
	float right_margin = v_scroll->is_visible() ? v_scroll->get_size().x : 0.0f;
	float bottom_margin = h_scroll->is_visible() ? h_scroll->get_size().y : 0.0f;

	Vector2 diff = Vector2(MAX(MIN(other_rect.position.x, global_rect.position.x), other_rect.position.x + other_rect.size.x - global_rect.size.x + (!is_layout_rtl() ? right_margin : 0.0f)),
			MAX(MIN(other_rect.position.y, global_rect.position.y), other_rect.position.y + other_rect.size.y - global_rect.size.y + bottom_margin));

	set_h_scroll(get_h_scroll() + (diff.x - global_rect.position.x));
	set_v_scroll(get_v_scroll() + (diff.y - global_rect.position.y));
}

void ScrollContainer::_update_dimensions() {
	child_max_size = Size2(0, 0);
	Size2 size = get_size();
	Point2 ofs;

	Ref<StyleBox> sb = get_theme_stylebox(SNAME("bg"));
	size -= sb->get_minimum_size();
	ofs += sb->get_offset();
	bool rtl = is_layout_rtl();

	if (h_scroll->is_visible_in_tree() && h_scroll->get_parent() == this) { //scrolls may have been moved out for reasons
		size.y -= h_scroll->get_minimum_size().y;
	}

	if (v_scroll->is_visible_in_tree() && v_scroll->get_parent() == this) { //scrolls may have been moved out for reasons
		size.x -= v_scroll->get_minimum_size().x;
	}

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
		Size2 minsize = c->get_combined_minimum_size();
		child_max_size.x = MAX(child_max_size.x, minsize.x);
		child_max_size.y = MAX(child_max_size.y, minsize.y);

		Rect2 r = Rect2(-Size2(get_h_scroll(), get_v_scroll()), minsize);
		if (!scroll_h || (!h_scroll->is_visible_in_tree() && c->get_h_size_flags() & SIZE_EXPAND)) {
			r.position.x = 0;
			if (c->get_h_size_flags() & SIZE_EXPAND) {
				r.size.width = MAX(size.width, minsize.width);
			} else {
				r.size.width = minsize.width;
			}
		}
		if (!scroll_v || (!v_scroll->is_visible_in_tree() && c->get_v_size_flags() & SIZE_EXPAND)) {
			r.position.y = 0;
			if (c->get_v_size_flags() & SIZE_EXPAND) {
				r.size.height = MAX(size.height, minsize.height);
			} else {
				r.size.height = minsize.height;
			}
		}
		r.position += ofs;
		if (rtl && v_scroll->is_visible_in_tree() && v_scroll->get_parent() == this) {
			r.position.x += v_scroll->get_minimum_size().x;
		}
		r.position = r.position.floor();
		fit_child_in_rect(c, r);
	}

	update();
}

void ScrollContainer::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE || p_what == NOTIFICATION_THEME_CHANGED || p_what == NOTIFICATION_LAYOUT_DIRECTION_CHANGED || p_what == NOTIFICATION_TRANSLATION_CHANGED) {
		_updating_scrollbars = true;
		call_deferred(SNAME("_update_scrollbar_position"));
	};

	if (p_what == NOTIFICATION_READY) {
		Viewport *viewport = get_viewport();
		ERR_FAIL_COND(!viewport);
		viewport->connect("gui_focus_changed", callable_mp(this, &ScrollContainer::_gui_focus_changed));
		_update_dimensions();
	}

	if (p_what == NOTIFICATION_SORT_CHILDREN) {
		_update_dimensions();
	};

	if (p_what == NOTIFICATION_DRAW) {
		Ref<StyleBox> sb = get_theme_stylebox(SNAME("bg"));
		draw_style_box(sb, Rect2(Vector2(), get_size()));

		update_scrollbars();
	}

	if (p_what == NOTIFICATION_INTERNAL_PHYSICS_PROCESS) {
		if (drag_touching) {
			if (drag_touching_deaccel) {
				Vector2 pos = Vector2(h_scroll->get_value(), v_scroll->get_value());
				pos += drag_speed * get_physics_process_delta_time();

				bool turnoff_h = false;
				bool turnoff_v = false;

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

				if (scroll_h) {
					h_scroll->set_value(pos.x);
				}
				if (scroll_v) {
					v_scroll->set_value(pos.y);
				}

				float sgn_x = drag_speed.x < 0 ? -1 : 1;
				float val_x = Math::abs(drag_speed.x);
				val_x -= 1000 * get_physics_process_delta_time();

				if (val_x < 0) {
					turnoff_h = true;
				}

				float sgn_y = drag_speed.y < 0 ? -1 : 1;
				float val_y = Math::abs(drag_speed.y);
				val_y -= 1000 * get_physics_process_delta_time();

				if (val_y < 0) {
					turnoff_v = true;
				}

				drag_speed = Vector2(sgn_x * val_x, sgn_y * val_y);

				if (turnoff_h && turnoff_v) {
					_cancel_drag();
				}

			} else {
				if (time_since_motion == 0 || time_since_motion > 0.1) {
					Vector2 diff = drag_accum - last_drag_accum;
					last_drag_accum = drag_accum;
					drag_speed = diff / get_physics_process_delta_time();
				}

				time_since_motion += get_physics_process_delta_time();
			}
		}
	}
};

void ScrollContainer::update_scrollbars() {
	Size2 size = get_size();
	Ref<StyleBox> sb = get_theme_stylebox(SNAME("bg"));
	size -= sb->get_minimum_size();

	Size2 hmin;
	Size2 vmin;
	if (scroll_h) {
		hmin = h_scroll->get_combined_minimum_size();
	}
	if (scroll_v) {
		vmin = v_scroll->get_combined_minimum_size();
	}

	Size2 min = child_max_size;

	bool hide_scroll_h = !scroll_h || min.width <= size.width || !h_scroll_visible;
	bool hide_scroll_v = !scroll_v || min.height <= size.height || !v_scroll_visible;

	h_scroll->set_max(min.width);
	h_scroll->set_page(size.width - (hide_scroll_v ? 0 : vmin.width));
	h_scroll->set_visible(!hide_scroll_h);

	v_scroll->set_max(min.height);
	v_scroll->set_page(size.height - (hide_scroll_h ? 0 : hmin.height));
	v_scroll->set_visible(!hide_scroll_v);

	// Avoid scrollbar overlapping.
	h_scroll->set_anchor_and_offset(SIDE_RIGHT, ANCHOR_END, hide_scroll_v ? 0 : -vmin.width);
	v_scroll->set_anchor_and_offset(SIDE_BOTTOM, ANCHOR_END, hide_scroll_h ? 0 : -hmin.height);
}

void ScrollContainer::_scroll_moved(float) {
	queue_sort();
	update();
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

void ScrollContainer::set_enable_h_scroll(bool p_enable) {
	if (scroll_h == p_enable) {
		return;
	}

	scroll_h = p_enable;
	minimum_size_changed();
	queue_sort();
}

bool ScrollContainer::is_h_scroll_enabled() const {
	return scroll_h;
}

void ScrollContainer::set_enable_v_scroll(bool p_enable) {
	if (scroll_v == p_enable) {
		return;
	}

	scroll_v = p_enable;
	minimum_size_changed();
	queue_sort();
}

bool ScrollContainer::is_v_scroll_enabled() const {
	return scroll_v;
}

void ScrollContainer::set_h_scroll_visible(bool p_visible) {
	if (h_scroll_visible == p_visible) {
		return;
	}

	h_scroll_visible = p_visible;
	update_scrollbars();
}

bool ScrollContainer::is_h_scroll_visible() const {
	return h_scroll_visible;
}

void ScrollContainer::set_v_scroll_visible(bool p_visible) {
	if (v_scroll_visible == p_visible) {
		return;
	}

	v_scroll_visible = p_visible;
	update_scrollbars();
}

bool ScrollContainer::is_v_scroll_visible() const {
	return v_scroll_visible;
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

TypedArray<String> ScrollContainer::get_configuration_warnings() const {
	TypedArray<String> warnings = Container::get_configuration_warnings();

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
		warnings.push_back(TTR("ScrollContainer is intended to work with a single child control.\nUse a container as child (VBox, HBox, etc.), or a Control and set the custom minimum size manually."));
	}

	return warnings;
}

HScrollBar *ScrollContainer::get_h_scrollbar() {
	return h_scroll;
}

VScrollBar *ScrollContainer::get_v_scrollbar() {
	return v_scroll;
}

void ScrollContainer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_update_scrollbar_position"), &ScrollContainer::_update_scrollbar_position);

	ClassDB::bind_method(D_METHOD("set_h_scroll", "value"), &ScrollContainer::set_h_scroll);
	ClassDB::bind_method(D_METHOD("get_h_scroll"), &ScrollContainer::get_h_scroll);

	ClassDB::bind_method(D_METHOD("set_v_scroll", "value"), &ScrollContainer::set_v_scroll);
	ClassDB::bind_method(D_METHOD("get_v_scroll"), &ScrollContainer::get_v_scroll);

	ClassDB::bind_method(D_METHOD("set_enable_h_scroll", "enable"), &ScrollContainer::set_enable_h_scroll);
	ClassDB::bind_method(D_METHOD("is_h_scroll_enabled"), &ScrollContainer::is_h_scroll_enabled);

	ClassDB::bind_method(D_METHOD("set_enable_v_scroll", "enable"), &ScrollContainer::set_enable_v_scroll);
	ClassDB::bind_method(D_METHOD("is_v_scroll_enabled"), &ScrollContainer::is_v_scroll_enabled);

	ClassDB::bind_method(D_METHOD("set_h_scroll_visible", "visible"), &ScrollContainer::set_h_scroll_visible);
	ClassDB::bind_method(D_METHOD("is_h_scroll_visible"), &ScrollContainer::is_h_scroll_visible);

	ClassDB::bind_method(D_METHOD("set_v_scroll_visible", "visible"), &ScrollContainer::set_v_scroll_visible);
	ClassDB::bind_method(D_METHOD("is_v_scroll_visible"), &ScrollContainer::is_v_scroll_visible);

	ClassDB::bind_method(D_METHOD("set_deadzone", "deadzone"), &ScrollContainer::set_deadzone);
	ClassDB::bind_method(D_METHOD("get_deadzone"), &ScrollContainer::get_deadzone);

	ClassDB::bind_method(D_METHOD("set_follow_focus", "enabled"), &ScrollContainer::set_follow_focus);
	ClassDB::bind_method(D_METHOD("is_following_focus"), &ScrollContainer::is_following_focus);

	ClassDB::bind_method(D_METHOD("get_h_scrollbar"), &ScrollContainer::get_h_scrollbar);
	ClassDB::bind_method(D_METHOD("get_v_scrollbar"), &ScrollContainer::get_v_scrollbar);
	ClassDB::bind_method(D_METHOD("ensure_control_visible", "control"), &ScrollContainer::ensure_control_visible);

	ADD_SIGNAL(MethodInfo("scroll_started"));
	ADD_SIGNAL(MethodInfo("scroll_ended"));

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "follow_focus"), "set_follow_focus", "is_following_focus");

	ADD_GROUP("Scroll", "scroll_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "scroll_horizontal"), "set_h_scroll", "get_h_scroll");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "scroll_vertical"), "set_v_scroll", "get_v_scroll");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "scroll_horizontal_enabled"), "set_enable_h_scroll", "is_h_scroll_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "scroll_vertical_enabled"), "set_enable_v_scroll", "is_v_scroll_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "scroll_horizontal_visible"), "set_h_scroll_visible", "is_h_scroll_visible");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "scroll_vertical_visible"), "set_v_scroll_visible", "is_v_scroll_visible");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "scroll_deadzone"), "set_deadzone", "get_deadzone");

	GLOBAL_DEF("gui/common/default_scroll_deadzone", 0);
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

	deadzone = GLOBAL_GET("gui/common/default_scroll_deadzone");

	set_clip_contents(true);
};
