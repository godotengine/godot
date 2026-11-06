/**************************************************************************/
/*  test_button.cpp                                                       */
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

TEST_FORCE_LINK(test_button)

#include "scene/gui/button.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"
#include "tests/display_server_mock.h"
#include "tests/signal_watcher.h"

namespace TestButton {

TEST_CASE("[SceneTree][Button] is_hovered()") {
	// Create new button instance.
	Button *button = memnew(Button);
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

TEST_CASE("[SceneTree][Button] Touch toggle signals on press") {
	Button *button = memnew(Button);
	SceneTree::get_singleton()->get_root()->add_child(button);
	button->set_size(Size2(100, 100));
	button->set_toggle_mode(true);
	button->set_action_mode(BaseButton::ACTION_MODE_BUTTON_PRESS);

	SIGNAL_WATCH(button, "button_down");
	SIGNAL_WATCH(button, "button_up");
	SIGNAL_WATCH(button, "toggled");
	SIGNAL_WATCH(button, "pressed");
	Array no_args;
	no_args.push_back(Array());

	Ref<InputEventScreenTouch> touch;
	touch.instantiate();
	touch->set_position(Point2(20, 20));

	SUBCASE("Each tap emits down and up without toggling again on release") {
		for (int i = 0; i < 3; i++) {
			touch->set_pressed(true);
			static_cast<Control *>(button)->gui_input(touch);
			SIGNAL_CHECK("button_down", no_args);
			SIGNAL_CHECK_FALSE("button_up");
			Array toggled_args = { { i % 2 == 0 } };
			SIGNAL_CHECK("toggled", toggled_args);
			SIGNAL_CHECK("pressed", no_args);

			// A different finger must not release the active touch.
			touch->set_index(1);
			touch->set_pressed(false);
			static_cast<Control *>(button)->gui_input(touch);
			SIGNAL_CHECK_FALSE("button_up");

			touch->set_index(0);
			static_cast<Control *>(button)->gui_input(touch);
			SIGNAL_CHECK("button_up", no_args);
			SIGNAL_CHECK_FALSE("button_down");
			SIGNAL_CHECK_FALSE("toggled");
			SIGNAL_CHECK_FALSE("pressed");
			CHECK(button->is_pressed() == (i % 2 == 0));
		}
	}

	SUBCASE("A missing release does not prevent subsequent activations") {
		for (int i = 0; i < 3; i++) {
			// Use a new finger index as well as reusing the previous index.
			touch->set_index(i / 2);
			touch->set_pressed(true);
			static_cast<Control *>(button)->gui_input(touch);
			Array toggled_args = { { i % 2 == 0 } };
			SIGNAL_CHECK("toggled", toggled_args);
			SIGNAL_CHECK("pressed", no_args);
			CHECK(button->is_pressed() == (i % 2 == 0));
		}
		touch->set_pressed(false);
		static_cast<Control *>(button)->gui_input(touch);
		SIGNAL_CHECK("button_up", no_args);
	}

	SIGNAL_UNWATCH(button, "button_down");
	SIGNAL_UNWATCH(button, "button_up");
	SIGNAL_UNWATCH(button, "toggled");
	SIGNAL_UNWATCH(button, "pressed");
	memdelete(button);
}

} // namespace TestButton
