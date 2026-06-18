/**************************************************************************/
/*  caption_button_overlay.cpp                                            */
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

#include "caption_button_overlay.h"

#include "core/object/class_db.h"
#include "scene/gui/control.h"
#include "scene/main/window.h"
#include "servers/display/display_server.h"

// -- Geometry helpers ------------------------------------------------

// Maps logical zone indices to left-to-right column order.
// LTR: minimize(0), maximize(1), close(2)
// RTL: close(0),    maximize(1), minimize(2)  — cluster moves to top-left
static CaptionButtonOverlay::ButtonZone _ltr_zone(int col, bool rtl) {
	if (rtl) {
		// Columns are: 0=close, 1=maximize, 2=minimize
		switch (col) {
			case 0:
				return CaptionButtonOverlay::ZONE_CLOSE;
			case 1:
				return CaptionButtonOverlay::ZONE_MAXIMIZE;
			default:
				return CaptionButtonOverlay::ZONE_MINIMIZE;
		}
	} else {
		return (CaptionButtonOverlay::ButtonZone)col;
	}
}

Rect2 CaptionButtonOverlay::_zone_rect(ButtonZone p_zone) const {
	bool rtl = is_layout_rtl();
	int col = 0;
	if (rtl) {
		switch (p_zone) {
			case ZONE_CLOSE:
				col = 0;
				break;
			case ZONE_MAXIMIZE:
				col = 1;
				break;
			default:
				col = 2;
				break;
		}
	} else {
		col = (int)p_zone;
	}
	float bw = get_size().x / 3.0f;
	float bh = get_size().y;
	return Rect2(col * bw, 0, bw, bh);
}

CaptionButtonOverlay::ButtonZone CaptionButtonOverlay::_zone_at(const Vector2 &p_pos) const {
	bool rtl = is_layout_rtl();
	float bw = get_size().x / 3.0f;
	float bh = get_size().y;
	for (int col = 0; col < 3; col++) {
		Rect2 r(col * bw, 0, bw, bh);
		if (r.has_point(p_pos)) {
			return _ltr_zone(col, rtl);
		}
	}
	return ZONE_NONE;
}

// -- Icon drawing ------------------------------------------------

void CaptionButtonOverlay::_draw_icon_minimize(const Rect2 &p_rect, const Color &p_color) {
	// Horizontal line at vertical center.
	float scale = _dpi_scale;
	float cx = p_rect.get_center().x;
	float cy = p_rect.get_center().y;
	float half = 5.0f * scale;
	bool antiAliased = std::abs(scale - std::round(scale)) > 1e-6f;
	draw_line(Vector2(cx - half, cy), Vector2(cx + half, cy), p_color, scale, antiAliased);
}

void CaptionButtonOverlay::_draw_icon_maximize(const Rect2 &p_rect, const Color &p_color) {
	// Square outline centered in the button.
	float scale = _dpi_scale;
	float cx = p_rect.get_center().x;
	float cy = p_rect.get_center().y;
	float half = 4.5f * scale;
	bool antiAliased = std::abs(scale - std::round(scale)) > 1e-6f;
	Rect2 sq(cx - half, cy - half, half * 2.0f, half * 2.0f);
	draw_rect(sq, p_color, false, scale, antiAliased);
}

void CaptionButtonOverlay::_draw_icon_restore(const Rect2 &p_rect, const Color &p_color) {
	// Two overlapping square outlines — front (bottom-left, full) and back (top-right, L-shaped).
	float scale = _dpi_scale;
	float cx = p_rect.get_center().x;
	float cy = p_rect.get_center().y;
	float sz = 7.0f * scale;
	float offset = 2.0f * scale;

	bool antiAliased = std::abs(scale - std::round(scale)) > 1e-6f;

	// Front square — shifted down by half the offset so the combined icon is vertically centered.
	Rect2 front(cx - sz * 0.5f, cy - sz * 0.5f + offset * 0.5f, sz, sz);
	draw_rect(front, p_color, false, scale, antiAliased);

	// Back square — derived directly from front (off px right, off px up).
	Rect2 back(front.position + Vector2(offset, -offset), Size2(sz, sz));

	float r = 1.5f * scale;
	float bx = back.position.x;
	float by = back.position.y;

	// Top edge — stop short of the corner by radius.
	draw_line(Point2(bx, by), Point2(bx + sz - r, by), p_color, scale, antiAliased);
	// Right edge — start below the corner by radius.
	draw_line(Point2(bx + sz, by + r), Point2(bx + sz, by + sz), p_color, scale, antiAliased);
	// Quarter-circle arc joining them at the top-right corner.
	draw_arc(Vector2(bx + sz - r, by + r), r, -Math::PI * 0.5f, 0.0f, 4, p_color, scale, antiAliased);
}

void CaptionButtonOverlay::_draw_icon_close(const Rect2 &p_rect, const Color &p_color) {
	// × — two diagonal lines.
	float scale = _dpi_scale;
	float cx = p_rect.get_center().x;
	float cy = p_rect.get_center().y;
	float half = 5.0f * scale;
	draw_line(Vector2(cx - half, cy - half), Vector2(cx + half, cy + half), p_color, scale, true);
	draw_line(Vector2(cx + half, cy - half), Vector2(cx - half, cy + half), p_color, scale, true);
}

// -- Control overrides -----------------------------------------------

CaptionButtonOverlay::CaptionButtonOverlay() {
	set_mouse_filter(MOUSE_FILTER_STOP);
}

void CaptionButtonOverlay::set_window_focused(bool p_focused) {
	if (window_focused == p_focused) {
		return;
	}
	window_focused = p_focused;
	queue_redraw();
}

void CaptionButtonOverlay::set_window_maximized(bool p_maximized) {
	if (window_maximized == p_maximized) {
		return;
	}
	window_maximized = p_maximized;
	queue_redraw();
}

void CaptionButtonOverlay::set_minimize_disabled(bool p_disabled) {
	if (minimize_disabled == p_disabled) {
		return;
	}
	minimize_disabled = p_disabled;
	queue_redraw();
}

void CaptionButtonOverlay::set_maximize_disabled(bool p_disabled) {
	if (maximize_disabled == p_disabled) {
		return;
	}
	maximize_disabled = p_disabled;
	queue_redraw();
}

void CaptionButtonOverlay::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			// Background colour for hovered / pressed button
			const Color close_hover_bg(0.769f, 0.169f, 0.110f, 1.0f); // Windows 11 red
			const Color normal_hover_bg(1.0f, 1.0f, 1.0f, 0.08f);
			const Color normal_press_bg(1.0f, 1.0f, 1.0f, 0.04f);

			for (int z = 0; z < 3; z++) {
				ButtonZone zone = (ButtonZone)z;
				Rect2 r = _zone_rect(zone);
				if (zone == hover_zone) {
					Color bg = (zone == ZONE_CLOSE) ? close_hover_bg : (pressing ? normal_press_bg : normal_hover_bg);
					draw_rect(r, bg);
				}
			}

			Color icon_color = get_theme_color(SNAME("title_color"), SNAME("Window"));
			Color disabled_color = icon_color * Color(1, 1, 1, 0.35f);

			// When unfocused, dim icons that are not currently hovered.
			auto icon_alpha = [&](ButtonZone p_zone) -> float {
				return (!window_focused && hover_zone != p_zone) ? 0.4f : 1.0f;
			};

			// Close: white when hovered (always full opacity); otherwise dimmed when unfocused.
			Color close_color = (hover_zone == ZONE_CLOSE)
					? Color(1, 1, 1)
					: Color(icon_color.r, icon_color.g, icon_color.b, icon_color.a * icon_alpha(ZONE_CLOSE));
			_draw_icon_close(_zone_rect(ZONE_CLOSE), close_color);

			Color min_color = minimize_disabled ? disabled_color : icon_color;
			min_color.a *= icon_alpha(ZONE_MINIMIZE);
			_draw_icon_minimize(_zone_rect(ZONE_MINIMIZE), min_color);

			Color max_color = maximize_disabled ? disabled_color : icon_color;
			max_color.a *= icon_alpha(ZONE_MAXIMIZE);
			if (window_maximized) {
				_draw_icon_restore(_zone_rect(ZONE_MAXIMIZE), max_color);
			} else {
				_draw_icon_maximize(_zone_rect(ZONE_MAXIMIZE), max_color);
			}
		} break;

		case NOTIFICATION_MOUSE_EXIT: {
			hover_zone = ZONE_NONE;
			pressing = false;
			queue_redraw();
		} break;
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_WM_DPI_CHANGE: {
			_dpi_scale = DisplayServer::get_singleton()->screen_get_dpi(get_window()->get_current_screen()) / 96.0f;
			queue_redraw();
		} break;
	}
}

void CaptionButtonOverlay::gui_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid()) {
		ButtonZone new_zone = _zone_at(mm->get_position());
		if (new_zone != hover_zone) {
			hover_zone = new_zone;
			queue_redraw();
		}
		return;
	}

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->get_button_index() == MouseButton::LEFT) {
		if (mb->is_pressed()) {
			pressing = true;
			queue_redraw();
		} else if (pressing) {
			pressing = false;
			queue_redraw();
			ButtonZone zone = _zone_at(mb->get_position());
			if (zone == ZONE_CLOSE) {
				emit_signal(SNAME("close_pressed"));
			} else if (zone == ZONE_MINIMIZE && !minimize_disabled) {
				emit_signal(SNAME("minimize_pressed"));
			} else if (zone == ZONE_MAXIMIZE && !maximize_disabled) {
				emit_signal(SNAME("maximize_pressed"));
			}
		}
		accept_event();
	}
}

void CaptionButtonOverlay::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_window_focused", "focused"), &CaptionButtonOverlay::set_window_focused);
	ClassDB::bind_method(D_METHOD("set_window_maximized", "maximized"), &CaptionButtonOverlay::set_window_maximized);
	ClassDB::bind_method(D_METHOD("get_window_maximized"), &CaptionButtonOverlay::get_window_maximized);
	ClassDB::bind_method(D_METHOD("set_minimize_disabled", "disabled"), &CaptionButtonOverlay::set_minimize_disabled);
	ClassDB::bind_method(D_METHOD("set_maximize_disabled", "disabled"), &CaptionButtonOverlay::set_maximize_disabled);

	ADD_SIGNAL(MethodInfo("close_pressed"));
	ADD_SIGNAL(MethodInfo("minimize_pressed"));
	ADD_SIGNAL(MethodInfo("maximize_pressed"));
}
