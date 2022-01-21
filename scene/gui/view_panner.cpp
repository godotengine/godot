/*************************************************************************/
/*  view_panner.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "view_panner.h"

#include "core/input/input.h"
#include "core/os/keyboard.h"

bool ViewPanner::gui_input(const Ref<InputEvent> &p_event, Rect2 p_canvas_rect) {
	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid()) {
		// Alt modifier is unused, so ignore such events.
		if (mb->is_alt_pressed()) {
			return false;
		}

		Vector2i scroll_vec = Vector2((mb->get_button_index() == MouseButton::WHEEL_RIGHT) - (mb->get_button_index() == MouseButton::WHEEL_LEFT), (mb->get_button_index() == MouseButton::WHEEL_DOWN) - (mb->get_button_index() == MouseButton::WHEEL_UP));
		if (scroll_vec != Vector2()) {
			if (control_scheme == SCROLL_PANS) {
				if (mb->is_ctrl_pressed()) {
					scroll_vec.y *= mb->get_factor();
					callback_helper(zoom_callback, scroll_vec, mb->get_position());
					return true;
				} else {
					Vector2 panning;
					if (mb->is_shift_pressed()) {
						panning.x += mb->get_factor() * scroll_vec.y;
						panning.y += mb->get_factor() * scroll_vec.x;
					} else {
						panning.y += mb->get_factor() * scroll_vec.y;
						panning.x += mb->get_factor() * scroll_vec.x;
					}
					callback_helper(scroll_callback, panning);
					return true;
				}
			} else {
				if (mb->is_ctrl_pressed()) {
					Vector2 panning;
					if (mb->is_shift_pressed()) {
						panning.x += mb->get_factor() * scroll_vec.y;
						panning.y += mb->get_factor() * scroll_vec.x;
					} else {
						panning.y += mb->get_factor() * scroll_vec.y;
						panning.x += mb->get_factor() * scroll_vec.x;
					}
					callback_helper(scroll_callback, panning);
					return true;
				} else if (!mb->is_shift_pressed()) {
					scroll_vec.y *= mb->get_factor();
					callback_helper(zoom_callback, scroll_vec, mb->get_position());
					return true;
				}
			}
		}

		if (mb->get_button_index() == MouseButton::MIDDLE || (mb->get_button_index() == MouseButton::RIGHT && !disable_rmb) || (mb->get_button_index() == MouseButton::LEFT && (Input::get_singleton()->is_key_pressed(Key::SPACE) || (is_dragging && !mb->is_pressed())))) {
			if (mb->is_pressed()) {
				is_dragging = true;
			} else {
				is_dragging = false;
			}
			return true;
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid()) {
		if (is_dragging) {
			if (p_canvas_rect != Rect2()) {
				callback_helper(pan_callback, Input::get_singleton()->warp_mouse_motion(mm, p_canvas_rect));
			} else {
				callback_helper(pan_callback, mm->get_relative());
			}
			return true;
		}
	}

	return false;
}

void ViewPanner::callback_helper(Callable p_callback, Vector2 p_arg1, Vector2 p_arg2) {
	if (p_callback == zoom_callback) {
		const Variant **argptr = (const Variant **)alloca(sizeof(Variant *) * 2);
		Variant var1 = p_arg1;
		argptr[0] = &var1;
		Variant var2 = p_arg2;
		argptr[1] = &var2;

		Variant result;
		Callable::CallError ce;
		p_callback.call(argptr, 2, result, ce);
	} else {
		const Variant **argptr = (const Variant **)alloca(sizeof(Variant *));
		Variant var = p_arg1;
		argptr[0] = &var;

		Variant result;
		Callable::CallError ce;
		p_callback.call(argptr, 1, result, ce);
	}
}

void ViewPanner::set_callbacks(Callable p_scroll_callback, Callable p_pan_callback, Callable p_zoom_callback) {
	scroll_callback = p_scroll_callback;
	pan_callback = p_pan_callback;
	zoom_callback = p_zoom_callback;
}

void ViewPanner::set_control_scheme(ControlScheme p_scheme) {
	control_scheme = p_scheme;
}

void ViewPanner::set_disable_rmb(bool p_disable) {
	disable_rmb = p_disable;
}
