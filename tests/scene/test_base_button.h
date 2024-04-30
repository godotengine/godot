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

	Point2i onButtonTopLeft = Point2i(5, 5);
	Point2i onButtonBottomRight = Point2i(54, 54);
	Point2i offButtonTopLeft = Point2i(4, 4);
	Point2i offButtonBottomRight = Point2i(55, 55);

	Array args1;
	Array empty_args;
	empty_args.push_back(args1);

	SIGNAL_WATCH(base_button, "pressed");
	SIGNAL_WATCH(base_button, "button_down");
	SIGNAL_WATCH(base_button, "button_up");
	SIGNAL_WATCH(base_button, "toggled");

	SUBCASE("[BaseButton][Hovering] is_hovered is updated when mouse enters and exits") {
		CHECK_FALSE(base_button->is_hovered());
		SEND_GUI_MOUSE_MOTION_EVENT(offButtonTopLeft, MouseButtonMask::NONE, Key::NONE);
		CHECK_FALSE(base_button->is_hovered());
		SEND_GUI_MOUSE_MOTION_EVENT(onButtonTopLeft, MouseButtonMask::NONE, Key::NONE);
		CHECK(base_button->is_hovered());
		SEND_GUI_MOUSE_MOTION_EVENT(onButtonBottomRight, MouseButtonMask::NONE, Key::NONE);
		CHECK(base_button->is_hovered());
		SEND_GUI_MOUSE_MOTION_EVENT(offButtonBottomRight, MouseButtonMask::NONE, Key::NONE);
		CHECK_FALSE(base_button->is_hovered());
	}

	SUBCASE("[BaseButton][Pressing] is_pressing is updated when mouse clicks and releases") {
		CHECK_FALSE(base_button->is_pressing());
		SIGNAL_CHECK_FALSE("pressed");
		SIGNAL_CHECK_FALSE("button_down");
		SIGNAL_CHECK_FALSE("button_up");
		
		SEND_GUI_MOUSE_BUTTON_EVENT(onButtonTopLeft, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		
		CHECK(base_button->is_pressing());
		SIGNAL_CHECK_FALSE("pressed");
		SIGNAL_CHECK("button_down", empty_args);
		SIGNAL_CHECK_FALSE("button_up")
		
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(onButtonTopLeft, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		
		CHECK_FALSE(base_button->is_pressing());
		SIGNAL_CHECK("pressed", empty_args);
		SIGNAL_CHECK_FALSE("button_down");
		SIGNAL_CHECK("button_up", empty_args)
	}

	SUBCASE("[BaseButton][Pressed] is_pressed is updated when mouse clicks and releases") {
		// TODO signal checks
		CHECK_FALSE(base_button->is_pressed());
		SEND_GUI_MOUSE_BUTTON_EVENT(onButtonTopLeft, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK(base_button->is_pressed());
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(onButtonTopLeft, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK_FALSE(base_button->is_pressed());

		SUBCASE("is_pressed is updated when mouse releases for toggle_mode") {
			// TODO signal checks
			base_button->set_toggle_mode(true);
			CHECK_FALSE(base_button->is_pressed());
			SEND_GUI_MOUSE_BUTTON_EVENT(onButtonTopLeft, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
			CHECK_FALSE(base_button->is_pressed());
			SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(onButtonTopLeft, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
			CHECK(base_button->is_pressed());
			SEND_GUI_MOUSE_BUTTON_EVENT(onButtonTopLeft, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
			CHECK(base_button->is_pressed());
			SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(onButtonTopLeft, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
			CHECK_FALSE(base_button->is_pressed());
			base_button->set_toggle_mode(false);
		}
	}

	SUBCASE("[BaseButton][Set Pressed] set_pressed works") {}

	// get_draw_mode
	SIGNAL_UNWATCH(base_button, "pressed");
	SIGNAL_UNWATCH(base_button, "button_down");
	SIGNAL_UNWATCH(base_button, "button_up");
	SIGNAL_WATCH(base_button, "toggled");
}

} // namespace TestBaseButton

#endif // TEST_BASE_BUTTON_H
