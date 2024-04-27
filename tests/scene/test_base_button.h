/**************************************************************************/
/*  test_base_button.h                                                    */
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

#ifndef TEST_BASE_BUTTON_H
#define TEST_BASE_BUTTON_H

#include "scene/gui/base_button.h"
#include "scene/main/window.h"

#include "tests/test_macros.h"

namespace TestBaseButton {

// https://github.com/godotengine/godot/issues/43440

TEST_CASE("[SceneTree][BaseButton]") {
	BaseButton *base_button = memnew(BaseButton);
	base_button->set_size(Size2i(50, 50));
	base_button->set_position(Point2i(5, 5));
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(base_button);

	SUBCASE("[BaseButton][Hovering] is_hovered is updated when mouse enters and exits") {
		CHECK_FALSE(base_button->is_hovered());
		SEND_GUI_MOUSE_MOTION_EVENT(Point2i(0, 0), MouseButtonMask::NONE, Key::NONE);
		CHECK_FALSE(base_button->is_hovered());
		SEND_GUI_MOUSE_MOTION_EVENT(Point2i(5, 5), MouseButtonMask::NONE, Key::NONE);
		CHECK(base_button->is_hovered());
		SEND_GUI_MOUSE_MOTION_EVENT(Point2i(54, 54), MouseButtonMask::NONE, Key::NONE);
		CHECK(base_button->is_hovered());
		SEND_GUI_MOUSE_MOTION_EVENT(Point2i(55, 55), MouseButtonMask::NONE, Key::NONE);
		CHECK_FALSE(base_button->is_hovered());
	}

	SUBCASE("[BaseButton][Pressing] is_pressing is updated when mouse clicks and releases") {}
	SUBCASE("[BaseButton][Pressed] is_pressed is updated when mouse clicks and releases") {}
	SUBCASE("[BaseButton][Set Pressed] set_pressed works") {}
}

} // namespace TestBaseButton

#endif // TEST_BASE_BUTTON_H
