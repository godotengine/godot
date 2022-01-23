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
#include "core/input/shortcut.h"
#include "core/os/keyboard.h"

bool ViewPanner::gui_input(const Ref<InputEvent> &p_event, Rect2 p_canvas_rect) {
	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid()) {
		Vector2i scroll_vec = Vector2((mb->get_button_index() == MouseButton::WHEEL_RIGHT) - (mb->get_button_index() == MouseButton::WHEEL_LEFT), (mb->get_button_index() == MouseButton::WHEEL_DOWN) - (mb->get_button_index() == MouseButton::WHEEL_UP));
		if (scroll_vec != Vector2()) {
			if (control_scheme == SCROLL_PANS) {
				if (mb->is_ctrl_pressed()) {
					scroll_vec.y *= mb->get_factor();
					callback_helper(zoom_callback, varray(scroll_vec, mb->get_position(), mb->is_alt_pressed()));
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
					callback_helper(scroll_callback, varray(panning, mb->is_alt_pressed()));
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
					callback_helper(scroll_callback, varray(panning, mb->is_alt_pressed()));
					return true;
				} else if (!mb->is_shift_pressed()) {
					scroll_vec.y *= mb->get_factor();
					callback_helper(zoom_callback, varray(scroll_vec, mb->get_position(), mb->is_alt_pressed()));
					return true;
				}
			}
		}

		// Alt is not used for button presses, so ignore it.
		if (mb->is_alt_pressed()) {
			return false;
		}

		if (mb->get_button_index() == MouseButton::MIDDLE || (enable_rmb && mb->get_button_index() == MouseButton::RIGHT) || (!simple_panning_enabled && mb->get_button_index() == MouseButton::LEFT && is_panning())) {
			if (mb->is_pressed()) {
				is_dragging = true;
			} else {
				is_dragging = false;
			}
			return mb->get_button_index() != MouseButton::LEFT || mb->is_pressed(); // Don't consume LMB release events (it fixes some selection problems).
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid()) {
		if (is_dragging) {
			if (p_canvas_rect != Rect2()) {
				callback_helper(pan_callback, varray(Input::get_singleton()->warp_mouse_motion(mm, p_canvas_rect)));
			} else {
				callback_helper(pan_callback, varray(mm->get_relative()));
			}
			return true;
		}
	}

	Ref<InputEventKey> k = p_event;
	if (k.is_valid()) {
		if (pan_view_shortcut.is_valid() && pan_view_shortcut->matches_event(k)) {
			pan_key_pressed = k->is_pressed();
			if (simple_panning_enabled || (Input::get_singleton()->get_mouse_button_mask() & MouseButton::LEFT) != MouseButton::NONE) {
				is_dragging = pan_key_pressed;
			}
			return true;
		}
	}

	return false;
}

void ViewPanner::release_pan_key() {
	pan_key_pressed = false;
	is_dragging = false;
}

void ViewPanner::callback_helper(Callable p_callback, Vector<Variant> p_args) {
	const Variant **argptr = (const Variant **)alloca(sizeof(Variant *) * p_args.size());
	for (int i = 0; i < p_args.size(); i++) {
		argptr[i] = &p_args[i];
	}

	Variant result;
	Callable::CallError ce;
	p_callback.call(argptr, p_args.size(), result, ce);
}

void ViewPanner::set_callbacks(Callable p_scroll_callback, Callable p_pan_callback, Callable p_zoom_callback) {
	scroll_callback = p_scroll_callback;
	pan_callback = p_pan_callback;
	zoom_callback = p_zoom_callback;
}

void ViewPanner::set_control_scheme(ControlScheme p_scheme) {
	control_scheme = p_scheme;
}

void ViewPanner::set_enable_rmb(bool p_enable) {
	enable_rmb = p_enable;
}

void ViewPanner::set_pan_shortcut(Ref<Shortcut> p_shortcut) {
	pan_view_shortcut = p_shortcut;
	pan_key_pressed = false;
}

void ViewPanner::set_simple_panning_enabled(bool p_enabled) {
	simple_panning_enabled = p_enabled;
}

void ViewPanner::setup(ControlScheme p_scheme, Ref<Shortcut> p_shortcut, bool p_simple_panning) {
	set_control_scheme(p_scheme);
	set_pan_shortcut(p_shortcut);
	set_simple_panning_enabled(p_simple_panning);
}

bool ViewPanner::is_panning() const {
	return is_dragging || pan_key_pressed;
}

ViewPanner::ViewPanner() {
	Array inputs;
	inputs.append(InputEventKey::create_reference(Key::SPACE));

	pan_view_shortcut.instantiate();
	pan_view_shortcut->set_events(inputs);
}
