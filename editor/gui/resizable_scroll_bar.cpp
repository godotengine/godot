/**************************************************************************/
/*  resizable_scroll_bar.cpp                                              */
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

#include "resizable_scroll_bar.h"

#include "scene/theme/theme_db.h"

void ResizableScrollBar::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	Ref<InputEventMouseButton> b = p_event;
	Ref<InputEventMouseMotion> m = p_event;

	const double min_resize_handle_left = get_grabber_offset();
	const double min_resize_handle_right = min_resize_handle_left + get_resize_handle_size();

	const double max_resize_handle_left = get_grabber_offset() + get_grabber_size() - get_resize_handle_size();
	const double max_resize_handle_right = max_resize_handle_left + get_resize_handle_size();

	if (b.is_valid()) {
		if (b->is_pressed()) {
			double ofs = orientation == VERTICAL ? b->get_position().y : b->get_position().x;
			if (min_resize_handle_right < ofs && ofs < max_resize_handle_left) {
				ScrollBar::gui_input(p_event);
			} else {
				start_page_at_click = get_value();
				end_page_at_click = get_value() + get_page();

				if (min_resize_handle_left < ofs && ofs < min_resize_handle_right) {
					min_handle_being_dragged = true;
					handle_offset = ofs - min_resize_handle_left;
				} else if (max_resize_handle_left < ofs && ofs < max_resize_handle_right) {
					max_handle_being_dragged = true;
					handle_offset = ofs - max_resize_handle_left;
				}
				queue_redraw();
				accept_event();
			}
		} else {
			min_handle_being_dragged = false;
			max_handle_being_dragged = false;
			ScrollBar::gui_input(p_event);
		}
	}

	if (m.is_valid()) {
		if (drag.active) {
			ScrollBar::gui_input(p_event);
			return;
		}

		accept_event();

		// Offset is distance from the right edge of the decrement button
		// to the event's location.
		double ofs = orientation == VERTICAL ? m->get_position().y : m->get_position().x;
		Ref<Texture2D> decr = theme_cache.decrement_icon;
		double decr_size = orientation == VERTICAL ? decr->get_height() : decr->get_width();
		ofs -= decr_size + theme_cache.scroll_style->get_margin(orientation == VERTICAL ? SIDE_TOP : SIDE_LEFT);

		if (min_handle_being_dragged) {
			if (get_max() < end_page_at_click) {
				end_page_at_click = get_max();
			}

			double min_value = get_min();
			double max_value = end_page_at_click - ratio_to_value((get_resize_handle_size()) / get_area_size());
			start_page_at_drag = CLAMP(ratio_to_value((ofs - handle_offset) / get_area_size()), min_value, max_value);
			emit_signal(SNAME("zoom_changed"));
			return;
		} else if (max_handle_being_dragged) {
			double min_value = start_page_at_click + ratio_to_value((get_resize_handle_size()) / get_area_size());
			double max_value = get_max();
			end_page_at_drag = CLAMP(ratio_to_value((ofs - handle_offset) / get_area_size()), min_value, max_value);
			emit_signal(SNAME("zoom_changed"));
			return;
		} else {
			int new_highlight = HIGHLIGHT_RANGE;
			if (min_resize_handle_left < ofs && ofs < min_resize_handle_right) {
				new_highlight = HIGHLIGHT_MIN_HANDLE;
			} else if (max_resize_handle_left < ofs && ofs < max_resize_handle_right) {
				new_highlight = HIGHLIGHT_MAX_HANDLE;
			}

			if (new_highlight != highlight) {
				highlight = new_highlight;
				queue_redraw();
			}
		}
	}
}

void ResizableScrollBar::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			RID ci = get_canvas_item();

			Ref<Texture2D> decr;

			if (decr_active) {
				decr = theme_cache.decrement_pressed_icon;
			} else if (highlight == HIGHLIGHT_DECR) {
				decr = theme_cache.decrement_hl_icon;
			} else {
				decr = theme_cache.decrement_icon;
			}

			Ref<StyleBox> min_handle, max_handle;
			if (min_handle_being_dragged) {
				min_handle = theme_cache.grabber_pressed_style;
			} else if (highlight == HIGHLIGHT_MIN_HANDLE) {
				min_handle = theme_cache.grabber_hl_style;
			} else {
				min_handle = theme_cache.grabber_style;
			}

			if (max_handle_being_dragged) {
				max_handle = theme_cache.grabber_pressed_style;
			} else if (highlight == HIGHLIGHT_MAX_HANDLE) {
				max_handle = theme_cache.grabber_hl_style;
			} else {
				max_handle = theme_cache.grabber_style;
			}

			Rect2 min_handle_rect, max_handle_rect;

			// Only valid for HORIZONTAL.
			// If/when you add a vertical version, make sure the widths and heights
			// here get switched appropriately!
			min_handle_rect.size.width = get_resize_handle_size();
			min_handle_rect.size.height = get_resize_handle_size();
			min_handle_rect.position.y = 0;
			min_handle_rect.position.x = get_grabber_offset() + decr->get_width() + theme_cache.scroll_style->get_margin(SIDE_LEFT);

			max_handle_rect.size.width = get_resize_handle_size();
			max_handle_rect.size.height = get_resize_handle_size();
			max_handle_rect.position.y = 0;
			max_handle_rect.position.x = get_grabber_offset() + decr->get_width() + theme_cache.scroll_style->get_margin(SIDE_LEFT) + get_grabber_size() - get_resize_handle_size();

			if (get_grabber_size() >= get_resize_handle_size() * 2.0) {
				min_handle->draw(ci, min_handle_rect);
				max_handle->draw(ci, max_handle_rect);
			}
		} break;
	}
}

bool ResizableScrollBar::is_min_handle_being_dragged() {
	return min_handle_being_dragged;
}

bool ResizableScrollBar::is_max_handle_being_dragged() {
	return max_handle_being_dragged;
}

double ResizableScrollBar::get_start_page() {
	return start_page_at_click;
}

double ResizableScrollBar::get_end_page() {
	return end_page_at_click;
}

double ResizableScrollBar::get_start_page_at_drag() {
	return start_page_at_drag;
}

double ResizableScrollBar::get_end_page_at_drag() {
	return end_page_at_drag;
}

double ResizableScrollBar::get_resize_handle_size() {
	return get_size().height;
}

double ResizableScrollBar::ratio_to_value(double p_value) {
	double v;
	double percent = (get_max() - get_min()) * p_value;
	v = percent + get_min();
	v = CLAMP(v, get_min(), get_max());
	return v;
}

void ResizableScrollBar::_bind_methods() {
	ADD_SIGNAL(MethodInfo("zoom_changed"));
}

ResizableScrollBar::ResizableScrollBar(Orientation p_orientation) :
		ScrollBar(p_orientation) {}
