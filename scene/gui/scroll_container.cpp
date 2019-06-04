/*************************************************************************/
/*  scroll_container.cpp                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

		Control *c = get_child(i)->cast_to<Control>();
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

	if (h_scroll->is_visible()) {
		min_size.y += h_scroll->get_minimum_size().y;
	}
	if (v_scroll->is_visible()) {
		min_size.x += v_scroll->get_minimum_size().x;
	}
	return min_size;
};

void ScrollContainer::_cancel_drag() {
	set_fixed_process(false);
	drag_touching_deaccel = false;
	drag_touching = false;
	drag_speed = Vector2();
	drag_accum = Vector2();
	last_drag_accum = Vector2();
	drag_from = Vector2();

	if (beyond_deadzone) {
		emit_signal("scroll_ended");
		propagate_notification(NOTIFICATION_SCROLL_END);
		beyond_deadzone = false;
	}
}

void ScrollContainer::_input_event(const InputEvent &p_input_event) {

	switch (p_input_event.type) {

		case InputEvent::MOUSE_BUTTON: {

			const InputEventMouseButton &mb = p_input_event.mouse_button;

			if (mb.button_index == BUTTON_WHEEL_UP && mb.pressed) {
				// only horizontal is enabled, scroll horizontally
				if (h_scroll->is_visible() && !v_scroll->is_visible()) {
					h_scroll->set_val(h_scroll->get_val() - h_scroll->get_page() / 8 * mb.factor);
				} else if (v_scroll->is_visible()) {
					v_scroll->set_val(v_scroll->get_val() - v_scroll->get_page() / 8 * mb.factor);
				}
			}

			if (mb.button_index == BUTTON_WHEEL_DOWN && mb.pressed) {
				// only horizontal is enabled, scroll horizontally
				if (h_scroll->is_visible() && !v_scroll->is_visible()) {
					h_scroll->set_val(h_scroll->get_val() + h_scroll->get_page() / 8 * mb.factor);
				} else if (v_scroll->is_visible()) {
					v_scroll->set_val(v_scroll->get_val() + v_scroll->get_page() / 8 * mb.factor);
				}
			}

			if (mb.button_index == BUTTON_WHEEL_LEFT && mb.pressed) {
				if (h_scroll->is_visible()) {
					h_scroll->set_val(h_scroll->get_val() - h_scroll->get_page() * mb.factor / 8);
				}
			}

			if (mb.button_index == BUTTON_WHEEL_RIGHT && mb.pressed) {
				if (h_scroll->is_visible()) {
					h_scroll->set_val(h_scroll->get_val() + h_scroll->get_page() * mb.factor / 8);
				}
			}

			if (!OS::get_singleton()->has_touchscreen_ui_hint())
				return;

			if (mb.button_index != BUTTON_LEFT)
				break;

			if (mb.pressed) {

				if (drag_touching) {
					_cancel_drag();
				}

				if (true) {
					drag_speed = Vector2();
					drag_accum = Vector2();
					last_drag_accum = Vector2();
					drag_from = Vector2(h_scroll->get_val(), v_scroll->get_val());
					drag_touching = OS::get_singleton()->has_touchscreen_ui_hint();
					drag_touching_deaccel = false;
					beyond_deadzone = false;
					time_since_motion = 0;
					if (drag_touching) {
						set_fixed_process(true);
						time_since_motion = 0;
					}
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

		} break;
		case InputEvent::MOUSE_MOTION: {

			const InputEventMouseMotion &mm = p_input_event.mouse_motion;

			if (drag_touching && !drag_touching_deaccel) {

				Vector2 motion = Vector2(mm.relative_x, mm.relative_y);
				drag_accum -= motion;
				if (beyond_deadzone || scroll_h && Math::abs(drag_accum.x) > deadzone || scroll_v && Math::abs(drag_accum.y) > deadzone) {
					if (!beyond_deadzone) {
						propagate_notification(NOTIFICATION_SCROLL_BEGIN);
						emit_signal("scroll_started");

						beyond_deadzone = true;
						// resetting drag_accum here ensures smooth scrolling after reaching deadzone
						drag_accum = -motion;
					}
					Vector2 diff = drag_from + drag_accum;
					if (scroll_h)
						h_scroll->set_val(diff.x);
					else
						drag_accum.x = 0;
					if (scroll_v)
						v_scroll->set_val(diff.y);
					else
						drag_accum.y = 0;
					time_since_motion = 0;
				}
			}

		} break;
	}
}

void ScrollContainer::_update_scrollbar_pos() {

	Size2 hmin = h_scroll->get_combined_minimum_size();
	Size2 vmin = v_scroll->get_combined_minimum_size();

	v_scroll->set_anchor_and_margin(MARGIN_LEFT, ANCHOR_END, vmin.width);
	v_scroll->set_anchor_and_margin(MARGIN_RIGHT, ANCHOR_END, 0);
	v_scroll->set_anchor_and_margin(MARGIN_TOP, ANCHOR_BEGIN, 0);
	v_scroll->set_anchor_and_margin(MARGIN_BOTTOM, ANCHOR_END, 0);

	h_scroll->set_anchor_and_margin(MARGIN_LEFT, ANCHOR_BEGIN, 0);
	h_scroll->set_anchor_and_margin(MARGIN_RIGHT, ANCHOR_END, 0);
	h_scroll->set_anchor_and_margin(MARGIN_TOP, ANCHOR_END, hmin.height);
	h_scroll->set_anchor_and_margin(MARGIN_BOTTOM, ANCHOR_END, 0);

	h_scroll->raise();
	v_scroll->raise();
}

void ScrollContainer::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE || p_what == NOTIFICATION_THEME_CHANGED) {

		call_deferred("_update_scrollbar_pos");
	};

	if (p_what == NOTIFICATION_SORT_CHILDREN) {

		child_max_size = Size2(0, 0);
		Size2 size = get_size();
		if (h_scroll->is_visible())
			size.y -= h_scroll->get_minimum_size().y;

		if (v_scroll->is_visible())
			size.x -= h_scroll->get_minimum_size().x;

		for (int i = 0; i < get_child_count(); i++) {

			Control *c = get_child(i)->cast_to<Control>();
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
			if (!scroll_h || (!h_scroll->is_visible() && c->get_h_size_flags() & SIZE_EXPAND)) {
				r.pos.x = 0;
				if (c->get_h_size_flags() & SIZE_EXPAND)
					r.size.width = MAX(size.width, minsize.width);
				else
					r.size.width = minsize.width;
			}
			if (!scroll_v || (!v_scroll->is_visible() && c->get_v_size_flags() & SIZE_EXPAND)) {
				r.pos.y = 0;
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

		VisualServer::get_singleton()->canvas_item_set_clip(get_canvas_item(), true);
	}

	if (p_what == NOTIFICATION_FIXED_PROCESS) {

		if (drag_touching) {

			if (drag_touching_deaccel) {

				Vector2 pos = Vector2(h_scroll->get_val(), v_scroll->get_val());
				pos += drag_speed * get_fixed_process_delta_time();

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
					h_scroll->set_val(pos.x);
				if (scroll_v)
					v_scroll->set_val(pos.y);

				float sgn_x = drag_speed.x < 0 ? -1 : 1;
				float val_x = Math::abs(drag_speed.x);
				val_x -= 1000 * get_fixed_process_delta_time();

				if (val_x < 0) {
					turnoff_h = true;
				}

				float sgn_y = drag_speed.y < 0 ? -1 : 1;
				float val_y = Math::abs(drag_speed.y);
				val_y -= 1000 * get_fixed_process_delta_time();

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
					drag_speed = diff / get_fixed_process_delta_time();
				}

				time_since_motion += get_fixed_process_delta_time();
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
		scroll.y = 0;
	} else {

		v_scroll->show();
		scroll.y = v_scroll->get_val();
	}

	v_scroll->set_max(min.height);
	v_scroll->set_page(size.height - hmin.height);

	if (!scroll_h || min.width <= size.width - vmin.width) {

		h_scroll->hide();
		scroll.x = 0;
	} else {

		h_scroll->show();
		h_scroll->set_max(min.width);
		h_scroll->set_page(size.width - vmin.width);
		scroll.x = h_scroll->get_val();
	}
}

void ScrollContainer::_scroll_moved(float) {

	scroll.x = h_scroll->get_val();
	scroll.y = v_scroll->get_val();
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

	return v_scroll->get_val();
}

void ScrollContainer::set_v_scroll(int p_pos) {

	v_scroll->set_val(p_pos);
	_cancel_drag();
}

int ScrollContainer::get_h_scroll() const {

	return h_scroll->get_val();
}

void ScrollContainer::set_h_scroll(int p_pos) {

	h_scroll->set_val(p_pos);
	_cancel_drag();
}

int ScrollContainer::get_deadzone() const {
	return deadzone;
}

void ScrollContainer::set_deadzone(int p_deadzone) {
	deadzone = p_deadzone;
}

void ScrollContainer::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("_scroll_moved"), &ScrollContainer::_scroll_moved);
	ObjectTypeDB::bind_method(_MD("_input_event"), &ScrollContainer::_input_event);
	ObjectTypeDB::bind_method(_MD("set_enable_h_scroll", "enable"), &ScrollContainer::set_enable_h_scroll);
	ObjectTypeDB::bind_method(_MD("is_h_scroll_enabled"), &ScrollContainer::is_h_scroll_enabled);
	ObjectTypeDB::bind_method(_MD("set_enable_v_scroll", "enable"), &ScrollContainer::set_enable_v_scroll);
	ObjectTypeDB::bind_method(_MD("is_v_scroll_enabled"), &ScrollContainer::is_v_scroll_enabled);
	ObjectTypeDB::bind_method(_MD("_update_scrollbar_pos"), &ScrollContainer::_update_scrollbar_pos);
	ObjectTypeDB::bind_method(_MD("set_h_scroll", "val"), &ScrollContainer::set_h_scroll);
	ObjectTypeDB::bind_method(_MD("get_h_scroll"), &ScrollContainer::get_h_scroll);
	ObjectTypeDB::bind_method(_MD("set_v_scroll", "val"), &ScrollContainer::set_v_scroll);
	ObjectTypeDB::bind_method(_MD("get_v_scroll"), &ScrollContainer::get_v_scroll);
	ObjectTypeDB::bind_method(_MD("set_deadzone", "deadzone"), &ScrollContainer::set_deadzone);
	ObjectTypeDB::bind_method(_MD("get_deadzone"), &ScrollContainer::get_deadzone);

	ADD_SIGNAL(MethodInfo("scroll_started"));
	ADD_SIGNAL(MethodInfo("scroll_ended"));

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "scroll/horizontal"), _SCS("set_enable_h_scroll"), _SCS("is_h_scroll_enabled"));
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "scroll/vertical"), _SCS("set_enable_v_scroll"), _SCS("is_v_scroll_enabled"));
	ADD_PROPERTY(PropertyInfo(Variant::INT, "scroll/deadzone"), _SCS("set_deadzone"), _SCS("get_deadzone"));
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
	beyond_deadzone = false;
	scroll_h = true;
	scroll_v = true;

	deadzone = GLOBAL_DEF("gui/common/default_scroll_deadzone", 0);
};
