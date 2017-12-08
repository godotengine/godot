/*************************************************************************/
/*  scroll_container.cpp                                                 */
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
#include "scroll_container.h"
#include "os/os.h"
bool ScrollContainer::clips_input() const {

	return true;
}

Size2 ScrollContainer::get_minimum_size() const {

	Size2 min_size;

	for (int i = 0; i < get_child_count(); i++) {

		Control *c = Object::cast_to<Control>(get_child(i));
		if (!c)
			continue;
		if (c->is_set_as_toplevel())
			continue;
		if (c == h_scroll || c == v_scroll)
			continue;
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
	return min_size;
};

void ScrollContainer::_cancel_drag() {
	set_physics_process(false);
	drag_touching_deaccel = false;
	drag_touching = false;
	drag_speed = Vector2();
	drag_accum = Vector2();
	last_drag_accum = Vector2();
	drag_from = Vector2();
}

void ScrollContainer::_gui_input(const Ref<InputEvent> &p_gui_input) {

	Ref<InputEventMouseButton> mb = p_gui_input;

	if (mb.is_valid()) {

		if (mb->get_button_index() == BUTTON_WHEEL_UP && mb->is_pressed()) {
			// only horizontal is enabled, scroll horizontally
			if (h_scroll->is_visible() && !v_scroll->is_visible()) {
				h_scroll->set_value(h_scroll->get_value() - h_scroll->get_page() / 8 * mb->get_factor());
			} else if (v_scroll->is_visible_in_tree()) {
				v_scroll->set_value(v_scroll->get_value() - v_scroll->get_page() / 8 * mb->get_factor());
			}
		}

		if (mb->get_button_index() == BUTTON_WHEEL_DOWN && mb->is_pressed()) {
			// only horizontal is enabled, scroll horizontally
			if (h_scroll->is_visible() && !v_scroll->is_visible()) {
				h_scroll->set_value(h_scroll->get_value() + h_scroll->get_page() / 8 * mb->get_factor());
			} else if (v_scroll->is_visible()) {
				v_scroll->set_value(v_scroll->get_value() + v_scroll->get_page() / 8 * mb->get_factor());
			}
		}

		if (mb->get_button_index() == BUTTON_WHEEL_LEFT && mb->is_pressed()) {
			if (h_scroll->is_visible_in_tree()) {
				h_scroll->set_value(h_scroll->get_value() - h_scroll->get_page() * mb->get_factor() / 8);
			}
		}

		if (mb->get_button_index() == BUTTON_WHEEL_RIGHT && mb->is_pressed()) {
			if (h_scroll->is_visible_in_tree()) {
				h_scroll->set_value(h_scroll->get_value() + h_scroll->get_page() * mb->get_factor() / 8);
			}
		}

		if (!OS::get_singleton()->has_touchscreen_ui_hint())
			return;

		if (mb->get_button_index() != BUTTON_LEFT)
			return;

		if (mb->is_pressed()) {

			if (drag_touching) {
				set_physics_process(false);
				drag_touching_deaccel = false;
				drag_touching = false;
				drag_speed = Vector2();
				drag_accum = Vector2();
				last_drag_accum = Vector2();
				drag_from = Vector2();
			}

			if (true) {
				drag_speed = Vector2();
				drag_accum = Vector2();
				last_drag_accum = Vector2();
				drag_from = Vector2(h_scroll->get_value(), v_scroll->get_value());
				drag_touching = OS::get_singleton()->has_touchscreen_ui_hint();
				drag_touching_deaccel = false;
				time_since_motion = 0;
				if (drag_touching) {
					set_physics_process(true);
					time_since_motion = 0;
				}
			}

		} else {
			if (drag_touching) {

				if (drag_speed == Vector2()) {
					drag_touching_deaccel = false;
					drag_touching = false;
					set_physics_process(false);
				} else {

					drag_touching_deaccel = true;
				}
			}
		}
	}

	Ref<InputEventMouseMotion> mm = p_gui_input;

	if (mm.is_valid()) {

		if (drag_touching && !drag_touching_deaccel) {

			Vector2 motion = Vector2(mm->get_relative().x, mm->get_relative().y);
			drag_accum -= motion;
			Vector2 diff = drag_from + drag_accum;

			if (scroll_h)
				h_scroll->set_value(diff.x);
			else
				drag_accum.x = 0;
			if (scroll_v)
				v_scroll->set_value(diff.y);
			else
				drag_accum.y = 0;
			time_since_motion = 0;
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
}

void ScrollContainer::_update_scrollbar_position() {

	Size2 hmin = h_scroll->get_combined_minimum_size();
	Size2 vmin = v_scroll->get_combined_minimum_size();

	v_scroll->set_anchor_and_margin(MARGIN_LEFT, ANCHOR_END, -vmin.width);
	v_scroll->set_anchor_and_margin(MARGIN_RIGHT, ANCHOR_END, 0);
	v_scroll->set_anchor_and_margin(MARGIN_TOP, ANCHOR_BEGIN, 0);
	v_scroll->set_anchor_and_margin(MARGIN_BOTTOM, ANCHOR_END, 0);

	h_scroll->set_anchor_and_margin(MARGIN_LEFT, ANCHOR_BEGIN, 0);
	h_scroll->set_anchor_and_margin(MARGIN_RIGHT, ANCHOR_END, 0);
	h_scroll->set_anchor_and_margin(MARGIN_TOP, ANCHOR_END, -hmin.height);
	h_scroll->set_anchor_and_margin(MARGIN_BOTTOM, ANCHOR_END, 0);

	h_scroll->raise();
	v_scroll->raise();
}

void ScrollContainer::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE || p_what == NOTIFICATION_THEME_CHANGED) {

		call_deferred("_update_scrollbar_position");
	};

	if (p_what == NOTIFICATION_SORT_CHILDREN) {

		child_max_size = Size2(0, 0);
		Size2 size = get_size();
		if (h_scroll->is_visible_in_tree())
			size.y -= h_scroll->get_minimum_size().y;

		if (v_scroll->is_visible_in_tree())
			size.x -= h_scroll->get_minimum_size().x;

		for (int i = 0; i < get_child_count(); i++) {

			Control *c = Object::cast_to<Control>(get_child(i));
			if (!c)
				continue;
			if (c->is_set_as_toplevel())
				continue;
			if (c == h_scroll || c == v_scroll)
				continue;
			Size2 minsize = c->get_combined_minimum_size();
			child_max_size.x = MAX(child_max_size.x, minsize.x);
			child_max_size.y = MAX(child_max_size.y, minsize.y);

			Rect2 r = Rect2(-scroll, minsize);
			if (!scroll_h || (!h_scroll->is_visible_in_tree() && c->get_h_size_flags() & SIZE_EXPAND)) {
				r.position.x = 0;
				if (c->get_h_size_flags() & SIZE_EXPAND)
					r.size.width = MAX(size.width, minsize.width);
				else
					r.size.width = minsize.width;
			}
			if (!scroll_v || (!v_scroll->is_visible_in_tree() && c->get_v_size_flags() & SIZE_EXPAND)) {
				r.position.y = 0;
				r.size.height = size.height;
				if (c->get_v_size_flags() & SIZE_EXPAND)
					r.size.height = MAX(size.height, minsize.height);
				else
					r.size.height = minsize.height;
			}
			fit_child_in_rect(c, r);
		}
		update();
	};

	if (p_what == NOTIFICATION_DRAW) {

		update_scrollbars();
	}

	if (p_what == NOTIFICATION_PHYSICS_PROCESS) {

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

				if (scroll_h)
					h_scroll->set_value(pos.x);
				if (scroll_v)
					v_scroll->set_value(pos.y);

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
					set_physics_process(false);
					drag_touching = false;
					drag_touching_deaccel = false;
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

	Size2 hmin = h_scroll->get_combined_minimum_size();
	Size2 vmin = v_scroll->get_combined_minimum_size();

	Size2 min = child_max_size;

	if (!scroll_v || min.height <= size.height - hmin.height) {

		v_scroll->hide();
		v_scroll->set_max(0);
		scroll.y = 0;
	} else {

		v_scroll->show();
		v_scroll->set_max(min.height);
		v_scroll->set_page(size.height - hmin.height);
		scroll.y = v_scroll->get_value();
	}

	if (!scroll_h || min.width <= size.width - vmin.width) {

		h_scroll->hide();
		h_scroll->set_max(0);
		scroll.x = 0;
	} else {

		h_scroll->show();
		h_scroll->set_max(min.width);
		h_scroll->set_page(size.width - vmin.width);
		scroll.x = h_scroll->get_value();
	}
}

void ScrollContainer::_scroll_moved(float) {

	scroll.x = h_scroll->get_value();
	scroll.y = v_scroll->get_value();
	queue_sort();

	update();
};

void ScrollContainer::set_enable_h_scroll(bool p_enable) {

	scroll_h = p_enable;
	queue_sort();
}

bool ScrollContainer::is_h_scroll_enabled() const {

	return scroll_h;
}

void ScrollContainer::set_enable_v_scroll(bool p_enable) {

	scroll_v = p_enable;
	queue_sort();
}

bool ScrollContainer::is_v_scroll_enabled() const {

	return scroll_v;
}

int ScrollContainer::get_v_scroll() const {

	return v_scroll->get_value();
}
void ScrollContainer::set_v_scroll(int p_pos) {

	v_scroll->set_value(p_pos);
	_cancel_drag();
}

int ScrollContainer::get_h_scroll() const {

	return h_scroll->get_value();
}
void ScrollContainer::set_h_scroll(int p_pos) {

	h_scroll->set_value(p_pos);
	_cancel_drag();
}

String ScrollContainer::get_configuration_warning() const {

	int found = 0;

	for (int i = 0; i < get_child_count(); i++) {

		Control *c = Object::cast_to<Control>(get_child(i));
		if (!c)
			continue;
		if (c->is_set_as_toplevel())
			continue;
		if (c == h_scroll || c == v_scroll)
			continue;

		found++;
	}

	if (found != 1)
		return TTR("ScrollContainer is intended to work with a single child control.\nUse a container as child (VBox,HBox,etc), or a Control and set the custom minimum size manually.");
	else
		return "";
}

void ScrollContainer::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_scroll_moved"), &ScrollContainer::_scroll_moved);
	ClassDB::bind_method(D_METHOD("_gui_input"), &ScrollContainer::_gui_input);
	ClassDB::bind_method(D_METHOD("set_enable_h_scroll", "enable"), &ScrollContainer::set_enable_h_scroll);
	ClassDB::bind_method(D_METHOD("is_h_scroll_enabled"), &ScrollContainer::is_h_scroll_enabled);
	ClassDB::bind_method(D_METHOD("set_enable_v_scroll", "enable"), &ScrollContainer::set_enable_v_scroll);
	ClassDB::bind_method(D_METHOD("is_v_scroll_enabled"), &ScrollContainer::is_v_scroll_enabled);
	ClassDB::bind_method(D_METHOD("_update_scrollbar_position"), &ScrollContainer::_update_scrollbar_position);
	ClassDB::bind_method(D_METHOD("set_h_scroll", "val"), &ScrollContainer::set_h_scroll);
	ClassDB::bind_method(D_METHOD("get_h_scroll"), &ScrollContainer::get_h_scroll);
	ClassDB::bind_method(D_METHOD("set_v_scroll", "val"), &ScrollContainer::set_v_scroll);
	ClassDB::bind_method(D_METHOD("get_v_scroll"), &ScrollContainer::get_v_scroll);

	ADD_GROUP("Scroll", "scroll_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "scroll_horizontal"), "set_enable_h_scroll", "is_h_scroll_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "scroll_vertical"), "set_enable_v_scroll", "is_v_scroll_enabled");
};

ScrollContainer::ScrollContainer() {

	h_scroll = memnew(HScrollBar);
	h_scroll->set_name("_h_scroll");
	add_child(h_scroll);

	v_scroll = memnew(VScrollBar);
	v_scroll->set_name("_v_scroll");
	add_child(v_scroll);

	h_scroll->connect("value_changed", this, "_scroll_moved");
	v_scroll->connect("value_changed", this, "_scroll_moved");

	drag_speed = Vector2();
	drag_touching = false;
	drag_touching_deaccel = false;
	scroll_h = true;
	scroll_v = true;

	set_clip_contents(true);
};
