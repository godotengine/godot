/**************************************************************************/
/*  test_button.h                                                         */
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

#ifndef TEST_BUTTON_H
#define TEST_BUTTON_H

#include "scene/gui/button.h"
#include "scene/main/window.h"

#include "tests/test_macros.h"

namespace TestButton {
TEST_CASE("[SceneTree][Button] Button is_hovered() test") {
	Window *root = SceneTree::get_singleton()->get_root();

	Window *window = memnew(Window);
	root->add_child(w);
	window->set_size(Size2i(500, 500));
	window->set_position(Size2i(0, 0));
	window->set_content_scale_size(Size2i(500, 500));
	window->set_content_scale_mode(Window::CONTENT_SCALE_MODE_CANVAS_ITEMS);
	window->set_content_scale_aspect(Window::CONTENT_SCALE_ASPECT_KEEP);

	// Create new button instance
	Button *button = memnew(Button);
	CHECK(button != nullptr);

	// Set up button's size and position
	window->add_child(button);
	button->set_size(Size2i(50, 50));
	button->set_position(Size2i(10, 10));

	// Button should initially be not hovered
	CHECK(button->is_hovered() == false);

	// Simulate mouse entering the button
	SEND_GUI_MOUSE_MOTION_EVENT(Point2i(25, 25), MouseButtonMask::NONE, Key::NONE);
	CHECK(button->is_hovered() == true);

	// Simulate mouse exiting the button
	SEND_GUI_MOUSE_MOTION_EVENT(Point2i(150, 150), MouseButtonMask::NONE, Key::NONE);
	CHECK(button->is_hovered() == false);

	memdelete(button);
	memdelete(window);
}

} //namespace TestButton
#endif // TEST_BUTTON_H
