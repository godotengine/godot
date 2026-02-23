/**************************************************************************/
/*  display_server_mock.cpp                                               */
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

#include "display_server_mock.h"

#include "core/input/input.h"
#include "core/input/input_event.h"
#include "servers/rendering/dummy/rasterizer_dummy.h"

Vector<String> DisplayServerMock::get_rendering_drivers_func() {
	Vector<String> drivers;
	drivers.push_back("dummy");
	return drivers;
}

void DisplayServerMock::_set_mouse_position(const Point2i &p_position) {
	if (mouse_position == p_position) {
		return;
	}
	mouse_position = p_position;
	_set_window_over(Rect2i(Point2i(0, 0), window_get_size()).has_point(p_position));
}
void DisplayServerMock::_set_window_over(bool p_over) {
	if (p_over == window_over) {
		return;
	}
	window_over = p_over;
	_send_window_event(p_over ? WINDOW_EVENT_MOUSE_ENTER : WINDOW_EVENT_MOUSE_EXIT);
}
void DisplayServerMock::_send_window_event(WindowEvent p_event) {
	if (event_callback.is_valid()) {
		Variant event = int(p_event);
		event_callback.call(event);
	}
}

bool DisplayServerMock::has_feature(Feature p_feature) const {
	switch (p_feature) {
		case FEATURE_MOUSE:
		case FEATURE_CURSOR_SHAPE:
		case FEATURE_CLIPBOARD:
		case FEATURE_CLIPBOARD_PRIMARY:
			return true;
		default: {
		}
	}
	return false;
}

DisplayServer *DisplayServerMock::create_func(const String &p_rendering_driver, DisplayServer::WindowMode p_mode, DisplayServer::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, Context p_context, int64_t p_parent_window, Error &r_error) {
	r_error = OK;
	RasterizerDummy::make_current();
	return memnew(DisplayServerMock());
}

void DisplayServerMock::simulate_event(Ref<InputEvent> p_event) {
	Ref<InputEvent> event = p_event;
	Ref<InputEventMouse> me = p_event;
	if (me.is_valid()) {
		Ref<InputEventMouseMotion> mm = p_event;
		if (mm.is_valid()) {
			mm->set_relative(mm->get_position() - mouse_position);
			event = mm;
		}
		_set_mouse_position(me->get_position());
	}
	Input::get_singleton()->parse_input_event(event);
}
