/**************************************************************************/
/*  editor_title_bar.cpp                                                  */
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

#include "editor_title_bar.h"

void EditorTitleBar::gui_input(const Ref<InputEvent> &p_event) {
	if (!can_move) {
		return;
	}

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid() && moving) {
		if (mm->get_button_mask().has_flag(MouseButtonMask::LEFT)) {
			Window *w = Object::cast_to<Window>(get_viewport());
			if (w) {
				Point2 mouse = DisplayServer::get_singleton()->mouse_get_position();
				w->set_position(mouse - click_pos);
			}
		} else {
			moving = false;
		}
	}

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && has_point(mb->get_position())) {
		Window *w = Object::cast_to<Window>(get_viewport());
		if (w) {
			if (mb->get_button_index() == MouseButton::LEFT) {
				if (mb->is_pressed()) {
					if (DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_WINDOW_DRAG)) {
						DisplayServer::get_singleton()->window_start_drag(w->get_window_id());
					} else {
						click_pos = DisplayServer::get_singleton()->mouse_get_position() - w->get_position();
						moving = true;
					}
				} else {
					moving = false;
				}
			}
			if (mb->get_button_index() == MouseButton::LEFT && mb->is_double_click() && mb->is_pressed()) {
				if (DisplayServer::get_singleton()->window_maximize_on_title_dbl_click()) {
					if (w->get_mode() == Window::MODE_WINDOWED) {
						w->set_mode(Window::MODE_MAXIMIZED);
					} else if (w->get_mode() == Window::MODE_MAXIMIZED) {
						w->set_mode(Window::MODE_WINDOWED);
					}
				} else if (DisplayServer::get_singleton()->window_minimize_on_title_dbl_click()) {
					w->set_mode(Window::MODE_MINIMIZED);
				}
				moving = false;
			}
		}
	}
}

void EditorTitleBar::set_center_control(Control *p_center_control) {
	center_control = p_center_control;
}

Control *EditorTitleBar::get_center_control() const {
	return center_control;
}

void EditorTitleBar::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_EXIT_TREE: {
			SceneTree::get_singleton()->get_root()->disconnect(SceneStringName(nonclient_window_input), callable_mp(this, &EditorTitleBar::gui_input));
			get_window()->set_nonclient_area(Rect2i());
		} break;
		case NOTIFICATION_ENTER_TREE: {
			SceneTree::get_singleton()->get_root()->connect(SceneStringName(nonclient_window_input), callable_mp(this, &EditorTitleBar::gui_input));
			[[fallthrough]];
		}
		case NOTIFICATION_RESIZED: {
			get_window()->set_nonclient_area(get_global_transform().xform(Rect2i(get_position(), get_size())));
		} break;
		case NOTIFICATION_SORT_CHILDREN: {
			if (!center_control) {
				break;
			}
			Control *prev = nullptr;
			Control *base = nullptr;
			Control *next = nullptr;

			bool rtl = is_layout_rtl();

			int start;
			int end;
			int delta;
			if (rtl) {
				start = get_child_count() - 1;
				end = -1;
				delta = -1;
			} else {
				start = 0;
				end = get_child_count();
				delta = +1;
			}

			for (int i = start; i != end; i += delta) {
				Control *c = as_sortable_control(get_child(i));
				if (!c) {
					continue;
				}
				if (base) {
					next = c;
					break;
				}
				if (c != center_control) {
					prev = c;
					continue;
				}
				base = c;
			}
			if (base && prev && next) {
				Size2i title_size = get_size();
				Size2i c_size = base->get_combined_minimum_size();

				int min_offset = prev->get_position().x + prev->get_combined_minimum_size().x;
				int max_offset = next->get_position().x + next->get_size().x - next->get_combined_minimum_size().x - c_size.x;

				int offset = (title_size.width - c_size.width) / 2;
				offset = CLAMP(offset, min_offset, max_offset);

				fit_child_in_rect(prev, Rect2i(prev->get_position().x, 0, offset - prev->get_position().x, title_size.height));
				fit_child_in_rect(base, Rect2i(offset, 0, c_size.width, title_size.height));
				fit_child_in_rect(next, Rect2i(offset + c_size.width, 0, next->get_position().x + next->get_size().x - (offset + c_size.width), title_size.height));
			}
		} break;
	}
}

void EditorTitleBar::set_can_move_window(bool p_enabled) {
	can_move = p_enabled;
	set_process_input(can_move);
}

bool EditorTitleBar::get_can_move_window() const {
	return can_move;
}
