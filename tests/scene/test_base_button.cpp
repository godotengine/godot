/**************************************************************************/
/*  test_base_button.cpp                                                  */
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

#include "tests/test_macros.h"

TEST_FORCE_LINK(test_base_button)

#include "scene/gui/base_button.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"
#include "tests/display_server_mock.h"
#include "tests/signal_watcher.h"

namespace TestBaseButton {

TEST_CASE("[SceneTree][BaseButton] is_hovered()") {
	// Create new button instance.
	BaseButton *button = memnew(BaseButton);
	CHECK(button != nullptr);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(button);

	// Set up button's size and position.
	button->set_size(Size2i(50, 50));
	button->set_position(Size2i(10, 10));

	// Button should initially be not hovered.
	CHECK(button->is_hovered() == false);

	// Simulate mouse entering the button.
	SEND_GUI_MOUSE_MOTION_EVENT(Point2i(25, 25), MouseButtonMask::NONE, Key::NONE);
	CHECK(button->is_hovered() == true);

	// Simulate mouse exiting the button.
	SEND_GUI_MOUSE_MOTION_EVENT(Point2i(150, 150), MouseButtonMask::NONE, Key::NONE);
	CHECK(button->is_hovered() == false);

	memdelete(button);
}

TEST_CASE("[SceneTree][BaseButton] get_draw_mode()") {
	// Create new button instance.
	BaseButton *button = memnew(BaseButton);
	CHECK(button != nullptr);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(button);

	// Set up button's size and position.
	button->set_size(Size2i(50, 50));
	button->set_position(Size2i(10, 10));

	// Button should not be initially pressed nor disabled.
	CHECK(button->is_pressed() == false);
	CHECK(button->is_disabled() == false);

	SUBCASE("[SceneTree][BaseButton] Draw Normal") {
		CHECK(button->is_hovered() == false);

		// Draw normal when button is not hovered nor disabled nor pressed.
		CHECK(button->get_draw_mode() == BaseButton::DrawMode::DRAW_NORMAL);
	}

	SUBCASE("[SceneTree][BaseButton] Draw Pressed") {
		CHECK(button->is_hovered() == false);
		button->set_toggle_mode(true);
		button->set_pressed_no_signal(true);

		// Draw pressed when button is pressed and not hovered.
		CHECK(button->get_draw_mode() == BaseButton::DrawMode::DRAW_PRESSED);
	}

	SUBCASE("[SceneTree][BaseButton] Draw Hover") {
		SEND_GUI_MOUSE_MOTION_EVENT(Point2i(25, 25), MouseButtonMask::NONE, Key::NONE);
		CHECK(button->is_hovered() == true);

		// Draw hover when button is hovered and not disabled nor pressed.
		CHECK(button->get_draw_mode() == BaseButton::DrawMode::DRAW_HOVER);
	}

	SUBCASE("[SceneTree][BaseButton] Draw Disabled") {
		button->set_disabled(true);

		// Draw disabled when button is disabled and not hovered.
		CHECK(button->get_draw_mode() == BaseButton::DrawMode::DRAW_DISABLED);

		SEND_GUI_MOUSE_MOTION_EVENT(Point2i(25, 25), MouseButtonMask::NONE, Key::NONE);
		CHECK(button->is_hovered() == true);

		// Draw disabled when button is disabled and hovered.
		CHECK(button->get_draw_mode() == BaseButton::DrawMode::DRAW_DISABLED);
	}

	SUBCASE("[SceneTree][BaseButton] Draw Hover and Pressed") {
		SEND_GUI_MOUSE_MOTION_EVENT(Point2i(25, 25), MouseButtonMask::NONE, Key::NONE);
		CHECK(button->is_hovered() == true);
		button->set_toggle_mode(true);
		button->set_pressed_no_signal(true);

		// Draw pressed and hovered when button is pressed and hovered.
		CHECK(button->get_draw_mode() == BaseButton::DrawMode::DRAW_HOVER_PRESSED);
	}

	memdelete(button);
}

TEST_CASE("[SceneTree][BaseButton] set_pressed_no_signal()") {
	// Create new button instance.
	BaseButton *button = memnew(BaseButton);
	CHECK(button != nullptr);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(button);

	button->set_toggle_mode(true);
	CHECK(button->is_pressed() == false);

	SIGNAL_WATCH(button, "pressed");

	button->set_pressed_no_signal(true);
	CHECK(button->is_pressed() == true);
	SIGNAL_CHECK_FALSE("pressed");

	button->set_pressed_no_signal(false);
	CHECK(button->is_pressed() == false);
	SIGNAL_CHECK_FALSE("pressed");

	SIGNAL_UNWATCH(button, "pressed");
	memdelete(button);
}

} // namespace TestBaseButton
