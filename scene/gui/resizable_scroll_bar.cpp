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

	Ref<InputEventMouseMotion> m = p_event;

	Ref<InputEventMouseButton> b = p_event;

	if (b.is_valid()) {
		accept_event();

		if (b->get_button_index() == MouseButton::WHEEL_DOWN && b->is_pressed()) {
			double change = ((get_page() != 0.0) ? get_page() / PAGE_DIVISOR : (get_max() - get_min()) / 16.0) * b->get_factor();
			scroll(MAX(change, get_step()));
			accept_event();
		}

		if (b->get_button_index() == MouseButton::WHEEL_UP && b->is_pressed()) {
			double change = ((get_page() != 0.0) ? get_page() / PAGE_DIVISOR : (get_max() - get_min()) / 16.0) * b->get_factor();
			scroll(-MAX(change, get_step()));
			accept_event();
		}

		if (b->get_button_index() != MouseButton::LEFT) {
			return;
		}

		if (b->is_pressed()) {
			double ofs = orientation == VERTICAL ? b->get_position().y : b->get_position().x;
			Ref<Texture2D> decr = theme_cache.decrement_icon;
			Ref<Texture2D> incr = theme_cache.increment_icon;

			double decr_size = orientation == VERTICAL ? decr->get_height() : decr->get_width();
			double incr_size = orientation == VERTICAL ? incr->get_height() : incr->get_width();
			double grabber_ofs = get_grabber_offset();
			double grabber_size = get_grabber_size();
			double total = orientation == VERTICAL ? get_size().height : get_size().width;

			if (ofs < decr_size) {
				decr_active = true;
				scroll(-(custom_step >= 0 ? custom_step : get_step()));
				queue_redraw();
				return;
			}

			if (ofs > total - incr_size) {
				incr_active = true;
				scroll(custom_step >= 0 ? custom_step : get_step());
				queue_redraw();
				return;
			}

			ofs -= decr_size + theme_cache.scroll_style->get_margin(orientation == VERTICAL ? SIDE_TOP : SIDE_LEFT);

			if (ofs < grabber_ofs) {
				if (scrolling) {
					target_scroll = CLAMP(target_scroll - get_page(), get_min(), get_max() - get_page());
				} else {
					double change = get_page() != 0.0 ? get_page() : (get_max() - get_min()) / 16.0;
					target_scroll = CLAMP(get_value() - change, get_min(), get_max() - get_page());
				}

				if (smooth_scroll_enabled) {
					scrolling = true;
					set_process_internal(true);
				} else {
					scroll_to(target_scroll);
				}
				return;
			}

			ofs -= grabber_ofs;
			if (get_grabber_size() >= get_handle_min_size()) {
				if (ofs < get_handle_size()) {
					drag.min_handle_being_dragged = true;
					drag.pos_at_click = grabber_ofs + ofs;
					drag.value_at_click = get_as_ratio();
					drag.start_page_at_click = get_value();
					drag.end_page_at_click = get_value() + get_page();
					drag.handle_offset = ofs;
					queue_redraw();
				} else if (ofs > get_handle_size() && ofs < grabber_size - get_handle_size()) {
					drag.grabber_being_dragged = true;
					drag.pos_at_click = grabber_ofs + ofs;
					drag.value_at_click = get_as_ratio();
					queue_redraw();
				} else if (ofs > grabber_size - get_handle_size() && ofs < grabber_size) {
					drag.max_handle_being_dragged = true;
					drag.pos_at_click = grabber_ofs + ofs;
					drag.value_at_click = get_as_ratio();
					drag.start_page_at_click = get_value();
					drag.end_page_at_click = get_value() + get_page();
					drag.handle_offset = grabber_size - ofs;
					queue_redraw();
				} else {
					if (scrolling) {
						target_scroll = CLAMP(target_scroll + get_page(), get_min(), get_max() - get_page());
					} else {
						double change = get_page() != 0.0 ? get_page() : (get_max() - get_min()) / 16.0;
						target_scroll = CLAMP(get_value() + change, get_min(), get_max() - get_page());
					}

					if (smooth_scroll_enabled) {
						scrolling = true;
						set_process_internal(true);
					} else {
						scroll_to(target_scroll);
					}
				}
			} else {
				if (ofs < grabber_size) {
					drag.grabber_being_dragged = true;
					drag.pos_at_click = grabber_ofs + ofs;
					drag.value_at_click = get_as_ratio();
					queue_redraw();
				} else {
					if (scrolling) {
						target_scroll = CLAMP(target_scroll + get_page(), get_min(), get_max() - get_page());
					} else {
						double change = get_page() != 0.0 ? get_page() : (get_max() - get_min()) / 16.0;
						target_scroll = CLAMP(get_value() + change, get_min(), get_max() - get_page());
					}

					if (smooth_scroll_enabled) {
						scrolling = true;
						set_process_internal(true);
					} else {
						scroll_to(target_scroll);
					}
				}
			}
		} else {
			incr_active = false;
			decr_active = false;
			drag.grabber_being_dragged = false;
			drag.min_handle_being_dragged = false;
			drag.max_handle_being_dragged = false;
			queue_redraw();
		}
	}

	if (m.is_valid()) {
		accept_event();

		if (drag.grabber_being_dragged) {
			double ofs = orientation == VERTICAL ? m->get_position().y : m->get_position().x;
			Ref<Texture2D> decr = theme_cache.decrement_icon;

			double decr_size = orientation == VERTICAL ? decr->get_height() : decr->get_width();
			ofs -= decr_size + theme_cache.scroll_style->get_margin(orientation == VERTICAL ? SIDE_TOP : SIDE_LEFT);

			double diff = (ofs - drag.pos_at_click) / get_area_size();

			double prev_scroll = get_value();

			set_as_ratio(drag.value_at_click + diff);

			if (!Math::is_equal_approx(prev_scroll, get_value())) {
				emit_signal(SNAME("scrolling"));
			}
		} else if (drag.min_handle_being_dragged) {
			double ofs = orientation == VERTICAL ? m->get_position().y : m->get_position().x;
			Ref<Texture2D> decr = theme_cache.decrement_icon;

			double decr_size = orientation == VERTICAL ? decr->get_height() : decr->get_width();
			ofs -= decr_size + theme_cache.scroll_style->get_margin(orientation == VERTICAL ? SIDE_TOP : SIDE_LEFT);

			if (get_max() < drag.end_page_at_click) {
				drag.end_page_at_click = get_max();
			}

			double min_value = get_min();
			double max_value = drag.end_page_at_click - ratio_to_value((get_handle_min_size()) / get_area_size());

			drag.start_page_at_drag = CLAMP(ratio_to_value((ofs - drag.handle_offset) / get_area_size()), min_value, max_value);

			emit_signal(SNAME("change_zoom"));
		} else if (drag.max_handle_being_dragged) {
			double ofs = orientation == VERTICAL ? m->get_position().y : m->get_position().x;
			Ref<Texture2D> decr = theme_cache.decrement_icon;

			double decr_size = orientation == VERTICAL ? decr->get_height() : decr->get_width();
			ofs -= decr_size + theme_cache.scroll_style->get_margin(orientation == VERTICAL ? SIDE_TOP : SIDE_LEFT);

			double min_value = drag.start_page_at_click + ratio_to_value((get_handle_min_size()) / get_area_size());
			double max_value = get_max();

			drag.end_page_at_drag = CLAMP(ratio_to_value((ofs + drag.handle_offset - get_grabber_min_size()) / get_area_size()), min_value, max_value);

			emit_signal(SNAME("change_zoom"));
		} else {
			double ofs = orientation == VERTICAL ? m->get_position().y : m->get_position().x;
			Ref<Texture2D> decr = theme_cache.decrement_icon;
			Ref<Texture2D> incr = theme_cache.increment_icon;

			double decr_size = orientation == VERTICAL ? decr->get_height() : decr->get_width();
			double incr_size = orientation == VERTICAL ? incr->get_height() : incr->get_width();
			double total = orientation == VERTICAL ? get_size().height : get_size().width;
			double grabber_ofs = get_grabber_offset();
			double grabber_size = get_grabber_size();

			HighlightStatus new_hilite;

			if (ofs < decr_size) {
				new_hilite = HIGHLIGHT_DECR;

			} else if (ofs > grabber_ofs && ofs < grabber_ofs + get_handle_size()) {
				new_hilite = HIGHLIGHT_MIN_HANDLE;

			} else if (ofs > grabber_ofs + grabber_size - get_handle_size() && ofs < grabber_ofs + grabber_size) {
				new_hilite = HIGHLIGHT_MAX_HANDLE;

			} else if (ofs > total - incr_size) {
				new_hilite = HIGHLIGHT_INCR;

			} else {
				new_hilite = HIGHLIGHT_RANGE;
			}

			if (new_hilite != highlight) {
				highlight = new_hilite;
				queue_redraw();
			}
		}
	}

	if (p_event->is_pressed()) {
		if (p_event->is_action("ui_left", true)) {
			if (orientation != HORIZONTAL) {
				return;
			}
			scroll(-(custom_step >= 0 ? custom_step : get_step()));

		} else if (p_event->is_action("ui_right", true)) {
			if (orientation != HORIZONTAL) {
				return;
			}
			scroll(custom_step >= 0 ? custom_step : get_step());

		} else if (p_event->is_action("ui_up", true)) {
			if (orientation != VERTICAL) {
				return;
			}

			scroll(-(custom_step >= 0 ? custom_step : get_step()));

		} else if (p_event->is_action("ui_down", true)) {
			if (orientation != VERTICAL) {
				return;
			}
			scroll(custom_step >= 0 ? custom_step : get_step());

		} else if (p_event->is_action("ui_home", true)) {
			scroll_to(get_min());

		} else if (p_event->is_action("ui_end", true)) {
			scroll_to(get_max());
		}
	}
}

void ResizableScrollBar::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			RID ci = get_canvas_item();

			Ref<Texture2D> decr, incr;

			if (decr_active) {
				decr = theme_cache.decrement_pressed_icon;
			} else if (highlight == HIGHLIGHT_DECR) {
				decr = theme_cache.decrement_hl_icon;
			} else {
				decr = theme_cache.decrement_icon;
			}

			if (incr_active) {
				incr = theme_cache.increment_pressed_icon;
			} else if (highlight == HIGHLIGHT_INCR) {
				incr = theme_cache.increment_hl_icon;
			} else {
				incr = theme_cache.increment_icon;
			}

			Ref<StyleBox> grabber, min_handle, max_handle;
			if (drag.grabber_being_dragged) {
				grabber = theme_cache.grabber_pressed_style;
			} else if (highlight == HIGHLIGHT_RANGE) {
				grabber = theme_cache.grabber_hl_style;
			} else {
				grabber = theme_cache.grabber_style;
			}

			if (drag.min_handle_being_dragged) {
				min_handle = theme_cache.grabber_pressed_style;
			} else if (highlight == HIGHLIGHT_MIN_HANDLE) {
				min_handle = theme_cache.grabber_hl_style;
			} else {
				min_handle = theme_cache.grabber_style;
			}

			if (drag.max_handle_being_dragged) {
				max_handle = theme_cache.grabber_pressed_style;
			} else if (highlight == HIGHLIGHT_MAX_HANDLE) {
				max_handle = theme_cache.grabber_hl_style;
			} else {
				max_handle = theme_cache.grabber_style;
			}

			Point2 ofs;

			decr->draw(ci, Point2());

			if (orientation == HORIZONTAL) {
				ofs.x += decr->get_width();
			} else {
				ofs.y += decr->get_height();
			}

			Size2 area = get_size();

			if (orientation == HORIZONTAL) {
				area.width -= incr->get_width() + decr->get_width();
			} else {
				area.height -= incr->get_height() + decr->get_height();
			}

			if (has_focus()) {
				theme_cache.scroll_focus_style->draw(ci, Rect2(ofs, area));
			} else {
				theme_cache.scroll_style->draw(ci, Rect2(ofs, area));
			}

			if (orientation == HORIZONTAL) {
				ofs.width += area.width;
			} else {
				ofs.height += area.height;
			}

			incr->draw(ci, ofs);
			Rect2 grabber_rect, min_handle_rect, max_handle_rect;

			if (orientation == HORIZONTAL) {
				grabber_rect.size.width = get_grabber_size();
				grabber_rect.size.height = get_size().height;
				grabber_rect.position.y = 0;
				grabber_rect.position.x = get_grabber_offset() + decr->get_width() + theme_cache.scroll_style->get_margin(SIDE_LEFT);

				min_handle_rect.size.width = get_handle_size();
				min_handle_rect.size.height = get_size().height;
				min_handle_rect.position.y = 0;
				min_handle_rect.position.x = get_grabber_offset() + decr->get_width() + theme_cache.scroll_style->get_margin(SIDE_LEFT);

				max_handle_rect.size.width = get_handle_size();
				max_handle_rect.size.height = get_size().height;
				max_handle_rect.position.y = 0;
				max_handle_rect.position.x = get_grabber_offset() + decr->get_width() + theme_cache.scroll_style->get_margin(SIDE_LEFT) + get_grabber_size() - get_handle_size();
			} else {
				// TODO: Implement
				grabber_rect.size.width = get_size().width;
				grabber_rect.size.height = get_grabber_size();
				grabber_rect.position.y = get_grabber_offset() + decr->get_height() + theme_cache.scroll_style->get_margin(SIDE_TOP);
				grabber_rect.position.x = 0;
			}

			grabber->draw(ci, grabber_rect);
			if (get_grabber_size() >= get_handle_min_size()) {
				min_handle->draw(ci, min_handle_rect);
				max_handle->draw(ci, max_handle_rect);
			}
		} break;

		case NOTIFICATION_ACCESSIBILITY_UPDATE:
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_EXIT_TREE:
		case NOTIFICATION_VISIBILITY_CHANGED: {
			ScrollBar::_notification(p_what);
		} break;
		case NOTIFICATION_MOUSE_EXIT: {
			highlight = HIGHLIGHT_NONE;
			queue_redraw();
		} break;
	}
}

bool ResizableScrollBar::is_min_handle_being_dragged() {
	return drag.min_handle_being_dragged;
}

bool ResizableScrollBar::is_max_handle_being_dragged() {
	return drag.max_handle_being_dragged;
}

double ResizableScrollBar::get_end_page() {
	return drag.end_page_at_click;
}

double ResizableScrollBar::get_start_page() {
	return drag.start_page_at_click;
}

double ResizableScrollBar::get_end_page_at_drag() {
	return drag.end_page_at_drag;
}

double ResizableScrollBar::get_start_page_at_drag() {
	return drag.start_page_at_drag;
}

double ResizableScrollBar::get_handle_size() {
	return 2.5 * get_grabber_min_size();
}

double ResizableScrollBar::get_handle_min_size() {
	return 2.5 * get_handle_size();
}

double ResizableScrollBar::ratio_to_value(double p_value) {
	double v;
	double percent = (get_max() - get_min()) * p_value;
	v = percent + get_min();
	v = CLAMP(v, get_min(), get_max());
	return v;
}

void ResizableScrollBar::_bind_methods() {
	ADD_SIGNAL(MethodInfo("change_zoom"));
}

ResizableScrollBar::ResizableScrollBar(Orientation p_orientation) :
		ScrollBar(p_orientation) {}

ResizableScrollBar::~ResizableScrollBar() {}
