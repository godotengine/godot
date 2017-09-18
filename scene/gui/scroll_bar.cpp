/*************************************************************************/
/*  scroll_bar.cpp                                                       */
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
#include "scroll_bar.h"

#include "os/keyboard.h"
#include "os/os.h"
#include "print_string.h"

bool ScrollBar::focus_by_default = false;

void ScrollBar::set_can_focus_by_default(bool p_can_focus) {

	focus_by_default = p_can_focus;
}

void ScrollBar::_gui_input(Ref<InputEvent> p_event) {

	Ref<InputEventMouseMotion> m = p_event;
	if (!m.is_valid() || drag.active) {
		emit_signal("scrolling");
	}

	Ref<InputEventMouseButton> b = p_event;

	if (b.is_valid()) {
		accept_event();

		if (b->get_button_index() == 5 && b->is_pressed()) {

			/*
			if (orientation==VERTICAL)
				set_val( get_val() + get_page() / 4.0 );
			else
			*/
			set_value(get_value() + get_page() / 4.0);
			accept_event();
		}

		if (b->get_button_index() == 4 && b->is_pressed()) {

			/*
			if (orientation==HORIZONTAL)
				set_val( get_val() - get_page() / 4.0 );
			else
			*/
			set_value(get_value() - get_page() / 4.0);
			accept_event();
		}

		if (b->get_button_index() != 1)
			return;

		if (b->is_pressed()) {

			double ofs = orientation == VERTICAL ? b->get_position().y : b->get_position().x;
			Ref<Texture> decr = get_icon("decrement");
			Ref<Texture> incr = get_icon("increment");

			double decr_size = orientation == VERTICAL ? decr->get_height() : decr->get_width();
			double incr_size = orientation == VERTICAL ? incr->get_height() : incr->get_width();
			double grabber_ofs = get_grabber_offset();
			double grabber_size = get_grabber_size();
			double total = orientation == VERTICAL ? get_size().height : get_size().width;

			if (ofs < decr_size) {

				set_value(get_value() - (custom_step >= 0 ? custom_step : get_step()));
				return;
			}

			if (ofs > total - incr_size) {

				set_value(get_value() + (custom_step >= 0 ? custom_step : get_step()));
				return;
			}

			ofs -= decr_size;

			if (ofs < grabber_ofs) {

				if (scrolling) {
					target_scroll = CLAMP(target_scroll - get_page(), get_min(), get_max() - get_page());
				} else {
					target_scroll = CLAMP(get_value() - get_page(), get_min(), get_max() - get_page());
				}

				if (smooth_scroll_enabled) {
					scrolling = true;
					set_fixed_process(true);
				} else {
					set_value(target_scroll);
				}
				return;
			}

			ofs -= grabber_ofs;

			if (ofs < grabber_size) {

				drag.active = true;
				drag.pos_at_click = grabber_ofs + ofs;
				drag.value_at_click = get_as_ratio();
				update();
			} else {
				if (scrolling) {
					target_scroll = CLAMP(target_scroll + get_page(), get_min(), get_max() - get_page());
				} else {
					target_scroll = CLAMP(get_value() + get_page(), get_min(), get_max() - get_page());
				}

				if (smooth_scroll_enabled) {
					scrolling = true;
					set_fixed_process(true);
				} else {
					set_value(target_scroll);
				}
			}

		} else {

			drag.active = false;
			update();
		}
	}

	if (m.is_valid()) {

		accept_event();

		if (drag.active) {

			double ofs = orientation == VERTICAL ? m->get_position().y : m->get_position().x;
			Ref<Texture> decr = get_icon("decrement");

			double decr_size = orientation == VERTICAL ? decr->get_height() : decr->get_width();
			ofs -= decr_size;

			double diff = (ofs - drag.pos_at_click) / get_area_size();

			set_as_ratio(drag.value_at_click + diff);
		} else {

			double ofs = orientation == VERTICAL ? m->get_position().y : m->get_position().x;
			Ref<Texture> decr = get_icon("decrement");
			Ref<Texture> incr = get_icon("increment");

			double decr_size = orientation == VERTICAL ? decr->get_height() : decr->get_width();
			double incr_size = orientation == VERTICAL ? incr->get_height() : incr->get_width();
			double total = orientation == VERTICAL ? get_size().height : get_size().width;

			HighlightStatus new_hilite;

			if (ofs < decr_size) {

				new_hilite = HIGHLIGHT_DECR;

			} else if (ofs > total - incr_size) {

				new_hilite = HIGHLIGHT_INCR;

			} else {

				new_hilite = HIGHLIGHT_RANGE;
			}

			if (new_hilite != highlight) {

				highlight = new_hilite;
				update();
			}
		}
	}

	Ref<InputEventKey> k = p_event;

	if (k.is_valid()) {

		if (!k->is_pressed())
			return;

		switch (k->get_scancode()) {

			case KEY_LEFT: {

				if (orientation != HORIZONTAL)
					return;
				set_value(get_value() - (custom_step >= 0 ? custom_step : get_step()));

			} break;
			case KEY_RIGHT: {

				if (orientation != HORIZONTAL)
					return;
				set_value(get_value() + (custom_step >= 0 ? custom_step : get_step()));

			} break;
			case KEY_UP: {

				if (orientation != VERTICAL)
					return;

				set_value(get_value() - (custom_step >= 0 ? custom_step : get_step()));

			} break;
			case KEY_DOWN: {

				if (orientation != VERTICAL)
					return;
				set_value(get_value() + (custom_step >= 0 ? custom_step : get_step()));

			} break;
			case KEY_HOME: {

				set_value(get_min());

			} break;
			case KEY_END: {

				set_value(get_max());

			} break;
		}
	}
}

void ScrollBar::_notification(int p_what) {

	if (p_what == NOTIFICATION_DRAW) {

		RID ci = get_canvas_item();

		Ref<Texture> decr = highlight == HIGHLIGHT_DECR ? get_icon("decrement_highlight") : get_icon("decrement");
		Ref<Texture> incr = highlight == HIGHLIGHT_INCR ? get_icon("increment_highlight") : get_icon("increment");
		Ref<StyleBox> bg = has_focus() ? get_stylebox("scroll_focus") : get_stylebox("scroll");

		Ref<StyleBox> grabber;
		if (drag.active)
			grabber = get_stylebox("grabber_pressed");
		else if (highlight == HIGHLIGHT_RANGE)
			grabber = get_stylebox("grabber_highlight");
		else
			grabber = get_stylebox("grabber");

		Point2 ofs;

		VisualServer *vs = VisualServer::get_singleton();

		vs->canvas_item_add_texture_rect(ci, Rect2(Point2(), decr->get_size()), decr->get_rid());

		if (orientation == HORIZONTAL)
			ofs.x += decr->get_width();
		else
			ofs.y += decr->get_height();

		Size2 area = get_size();

		if (orientation == HORIZONTAL)
			area.width -= incr->get_width() + decr->get_width();
		else
			area.height -= incr->get_height() + decr->get_height();

		bg->draw(ci, Rect2(ofs, area));

		if (orientation == HORIZONTAL)
			ofs.width += area.width;
		else
			ofs.height += area.height;

		vs->canvas_item_add_texture_rect(ci, Rect2(ofs, decr->get_size()), incr->get_rid());
		Rect2 grabber_rect;

		if (orientation == HORIZONTAL) {

			grabber_rect.size.width = get_grabber_size();
			grabber_rect.size.height = get_size().height;
			grabber_rect.position.y = 0;
			grabber_rect.position.x = get_grabber_offset() + decr->get_width() + bg->get_margin(MARGIN_LEFT);
		} else {

			grabber_rect.size.width = get_size().width;
			grabber_rect.size.height = get_grabber_size();
			grabber_rect.position.y = get_grabber_offset() + decr->get_height() + bg->get_margin(MARGIN_TOP);
			grabber_rect.position.x = 0;
		}

		grabber->draw(ci, grabber_rect);
	}

	if (p_what == NOTIFICATION_ENTER_TREE) {

		if (has_node(drag_slave_path)) {
			Node *n = get_node(drag_slave_path);
			drag_slave = Object::cast_to<Control>(n);
		}

		if (drag_slave) {
			drag_slave->connect("gui_input", this, "_drag_slave_input");
			drag_slave->connect("tree_exited", this, "_drag_slave_exit", varray(), CONNECT_ONESHOT);
		}
	}
	if (p_what == NOTIFICATION_EXIT_TREE) {

		if (drag_slave) {
			drag_slave->disconnect("gui_input", this, "_drag_slave_input");
			drag_slave->disconnect("tree_exited", this, "_drag_slave_exit");
		}

		drag_slave = NULL;
	}

	if (p_what == NOTIFICATION_FIXED_PROCESS) {

		if (scrolling) {
			if (get_value() != target_scroll) {
				double target = target_scroll - get_value();
				double dist = sqrt(target * target);
				double vel = ((target / dist) * 500) * get_fixed_process_delta_time();

				if (vel >= dist) {
					set_value(target_scroll);
				} else {
					set_value(get_value() + vel);
				}
			} else {
				scrolling = false;
				set_fixed_process(false);
			}
		} else if (drag_slave_touching) {

			if (drag_slave_touching_deaccel) {

				Vector2 pos = Vector2(orientation == HORIZONTAL ? get_value() : 0, orientation == VERTICAL ? get_value() : 0);
				pos += drag_slave_speed * get_fixed_process_delta_time();

				bool turnoff = false;

				if (orientation == HORIZONTAL) {

					if (pos.x < 0) {
						pos.x = 0;
						turnoff = true;
					}

					if (pos.x > (get_max() - get_page())) {
						pos.x = get_max() - get_page();
						turnoff = true;
					}

					set_value(pos.x);

					float sgn_x = drag_slave_speed.x < 0 ? -1 : 1;
					float val_x = Math::abs(drag_slave_speed.x);
					val_x -= 1000 * get_fixed_process_delta_time();

					if (val_x < 0) {
						turnoff = true;
					}

					drag_slave_speed.x = sgn_x * val_x;

				} else {

					if (pos.y < 0) {
						pos.y = 0;
						turnoff = true;
					}

					if (pos.y > (get_max() - get_page())) {
						pos.y = get_max() - get_page();
						turnoff = true;
					}

					set_value(pos.y);

					float sgn_y = drag_slave_speed.y < 0 ? -1 : 1;
					float val_y = Math::abs(drag_slave_speed.y);
					val_y -= 1000 * get_fixed_process_delta_time();

					if (val_y < 0) {
						turnoff = true;
					}
					drag_slave_speed.y = sgn_y * val_y;
				}

				if (turnoff) {
					set_fixed_process(false);
					drag_slave_touching = false;
					drag_slave_touching_deaccel = false;
				}

			} else {

				if (time_since_motion == 0 || time_since_motion > 0.1) {

					Vector2 diff = drag_slave_accum - last_drag_slave_accum;
					last_drag_slave_accum = drag_slave_accum;
					drag_slave_speed = diff / get_fixed_process_delta_time();
				}

				time_since_motion += get_fixed_process_delta_time();
			}
		}
	}

	if (p_what == NOTIFICATION_MOUSE_EXIT) {

		highlight = HIGHLIGHT_NONE;
		update();
	}
}

double ScrollBar::get_grabber_min_size() const {

	Ref<StyleBox> grabber = get_stylebox("grabber");
	Size2 gminsize = grabber->get_minimum_size() + grabber->get_center_size();
	return (orientation == VERTICAL) ? gminsize.height : gminsize.width;
}

double ScrollBar::get_grabber_size() const {

	float range = get_max() - get_min();
	if (range <= 0)
		return 0;

	float page = (get_page() > 0) ? get_page() : 0;
	/*
	if (grabber_range < get_step())
		grabber_range=get_step();
	*/

	double area_size = get_area_size();
	double grabber_size = page / range * area_size;
	return grabber_size + get_grabber_min_size();
}

double ScrollBar::get_area_size() const {

	if (orientation == VERTICAL) {

		double area = get_size().height;
		area -= get_stylebox("scroll")->get_minimum_size().height;
		area -= get_icon("increment")->get_height();
		area -= get_icon("decrement")->get_height();
		area -= get_grabber_min_size();
		return area;

	} else if (orientation == HORIZONTAL) {

		double area = get_size().width;
		area -= get_stylebox("scroll")->get_minimum_size().width;
		area -= get_icon("increment")->get_width();
		area -= get_icon("decrement")->get_width();
		area -= get_grabber_min_size();
		return area;
	} else {

		return 0;
	}
}

double ScrollBar::get_area_offset() const {

	double ofs = 0;

	if (orientation == VERTICAL) {

		ofs += get_stylebox("hscroll")->get_margin(MARGIN_TOP);
		ofs += get_icon("decrement")->get_height();
	}

	if (orientation == HORIZONTAL) {

		ofs += get_stylebox("hscroll")->get_margin(MARGIN_LEFT);
		ofs += get_icon("decrement")->get_width();
	}

	return ofs;
}

double ScrollBar::get_click_pos(const Point2 &p_pos) const {

	float pos = (orientation == VERTICAL) ? p_pos.y : p_pos.x;
	pos -= get_area_offset();

	float area = get_area_size();
	if (area == 0)
		return 0;
	else
		return pos / area;
}

double ScrollBar::get_grabber_offset() const {

	return (get_area_size()) * get_as_ratio();
}

Size2 ScrollBar::get_minimum_size() const {

	Ref<Texture> incr = get_icon("increment");
	Ref<Texture> decr = get_icon("decrement");
	Ref<StyleBox> bg = get_stylebox("scroll");
	Size2 minsize;

	if (orientation == VERTICAL) {

		minsize.width = MAX(incr->get_size().width, (bg->get_minimum_size() + bg->get_center_size()).width);
		minsize.height += incr->get_size().height;
		minsize.height += decr->get_size().height;
		minsize.height += bg->get_minimum_size().height;
		minsize.height += get_grabber_min_size();
	}

	if (orientation == HORIZONTAL) {

		minsize.height = MAX(incr->get_size().height, (bg->get_center_size() + bg->get_minimum_size()).height);
		minsize.width += incr->get_size().width;
		minsize.width += decr->get_size().width;
		minsize.width += bg->get_minimum_size().width;
		minsize.width += get_grabber_min_size();
	}

	return minsize;
}

void ScrollBar::set_custom_step(float p_custom_step) {

	custom_step = p_custom_step;
}

float ScrollBar::get_custom_step() const {

	return custom_step;
}

void ScrollBar::_drag_slave_exit() {

	if (drag_slave) {
		drag_slave->disconnect("gui_input", this, "_drag_slave_input");
	}
	drag_slave = NULL;
}

void ScrollBar::_drag_slave_input(const Ref<InputEvent> &p_input) {

	Ref<InputEventMouseButton> mb = p_input;

	if (mb.is_valid()) {

		if (mb->get_button_index() != 1)
			return;

		if (mb->is_pressed()) {

			if (drag_slave_touching) {
				set_fixed_process(false);
				drag_slave_touching_deaccel = false;
				drag_slave_touching = false;
				drag_slave_speed = Vector2();
				drag_slave_accum = Vector2();
				last_drag_slave_accum = Vector2();
				drag_slave_from = Vector2();
			}

			if (true) {
				drag_slave_speed = Vector2();
				drag_slave_accum = Vector2();
				last_drag_slave_accum = Vector2();
				//drag_slave_from=Vector2(h_scroll->get_val(),v_scroll->get_val());
				drag_slave_from = Vector2(orientation == HORIZONTAL ? get_value() : 0, orientation == VERTICAL ? get_value() : 0);

				drag_slave_touching = OS::get_singleton()->has_touchscreen_ui_hint();
				drag_slave_touching_deaccel = false;
				time_since_motion = 0;
				if (drag_slave_touching) {
					set_fixed_process(true);
					time_since_motion = 0;
				}
			}

		} else {

			if (drag_slave_touching) {

				if (drag_slave_speed == Vector2()) {
					drag_slave_touching_deaccel = false;
					drag_slave_touching = false;
					set_fixed_process(false);
				} else {

					drag_slave_touching_deaccel = true;
				}
			}
		}
	}

	Ref<InputEventMouseMotion> mm = p_input;

	if (mm.is_valid()) {

		if (drag_slave_touching && !drag_slave_touching_deaccel) {

			Vector2 motion = Vector2(mm->get_relative().x, mm->get_relative().y);

			drag_slave_accum -= motion;
			Vector2 diff = drag_slave_from + drag_slave_accum;

			if (orientation == HORIZONTAL)
				set_value(diff.x);
			/*
			else
				drag_slave_accum.x=0;
			*/
			if (orientation == VERTICAL)
				set_value(diff.y);
			/*
			else
				drag_slave_accum.y=0;
			*/
			time_since_motion = 0;
		}
	}
}

void ScrollBar::set_drag_slave(const NodePath &p_path) {

	if (is_inside_tree()) {

		if (drag_slave) {
			drag_slave->disconnect("gui_input", this, "_drag_slave_input");
			drag_slave->disconnect("tree_exited", this, "_drag_slave_exit");
		}
	}

	drag_slave = NULL;
	drag_slave_path = p_path;

	if (is_inside_tree()) {

		if (has_node(p_path)) {
			Node *n = get_node(p_path);
			drag_slave = Object::cast_to<Control>(n);
		}

		if (drag_slave) {
			drag_slave->connect("gui_input", this, "_drag_slave_input");
			drag_slave->connect("tree_exited", this, "_drag_slave_exit", varray(), CONNECT_ONESHOT);
		}
	}
}

NodePath ScrollBar::get_drag_slave() const {

	return drag_slave_path;
}

void ScrollBar::set_smooth_scroll_enabled(bool p_enable) {
	smooth_scroll_enabled = p_enable;
}

bool ScrollBar::is_smooth_scroll_enabled() const {
	return smooth_scroll_enabled;
}

void ScrollBar::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_gui_input"), &ScrollBar::_gui_input);
	ClassDB::bind_method(D_METHOD("set_custom_step", "step"), &ScrollBar::set_custom_step);
	ClassDB::bind_method(D_METHOD("get_custom_step"), &ScrollBar::get_custom_step);
	ClassDB::bind_method(D_METHOD("_drag_slave_input"), &ScrollBar::_drag_slave_input);
	ClassDB::bind_method(D_METHOD("_drag_slave_exit"), &ScrollBar::_drag_slave_exit);

	ADD_SIGNAL(MethodInfo("scrolling"));

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "custom_step", PROPERTY_HINT_RANGE, "-1,4096"), "set_custom_step", "get_custom_step");
}

ScrollBar::ScrollBar(Orientation p_orientation) {

	orientation = p_orientation;
	highlight = HIGHLIGHT_NONE;
	custom_step = -1;
	drag_slave = NULL;

	drag.active = false;

	drag_slave_speed = Vector2();
	drag_slave_touching = false;
	drag_slave_touching_deaccel = false;

	scrolling = false;
	target_scroll = 0;
	smooth_scroll_enabled = false;

	if (focus_by_default)
		set_focus_mode(FOCUS_ALL);
	set_step(0);
}

ScrollBar::~ScrollBar() {
}
