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

TEST_CASE("[SceneTree][Button] Long press activates only once") {
	Button *button = memnew(Button);
	SceneTree::get_singleton()->get_root()->add_child(button);
	button->set_size(Size2(100, 100));
	button->set_toggle_mode(true);
	button->set_action_mode(BaseButton::ACTION_MODE_BUTTON_PRESS);
	bool cancel_before_long_press = true;

	SUBCASE("Android toggle on press") {}
	SUBCASE("Android button on press") {
		button->set_toggle_mode(false);
	}
	SUBCASE("Android toggle on release") {
		button->set_action_mode(BaseButton::ACTION_MODE_BUTTON_RELEASE);
	}
	SUBCASE("Android button on release") {
		button->set_toggle_mode(false);
		button->set_action_mode(BaseButton::ACTION_MODE_BUTTON_RELEASE);
	}
	SUBCASE("Long press without cancellation") {
		cancel_before_long_press = false;
	}

	SIGNAL_WATCH(button, "pressed");
	SIGNAL_WATCH(button, "toggled");
	Array no_args;
	no_args.push_back(Array());
	Array toggled_args = { { true } };
	const bool activate_on_press = button->get_action_mode() == BaseButton::ACTION_MODE_BUTTON_PRESS;

	Ref<InputEventScreenTouch> touch;
	touch.instantiate();
	touch->set_position(Point2(20, 20));
	touch->set_pressed(true);
	static_cast<Control *>(button)->gui_input(touch);
	if (activate_on_press) {
		SIGNAL_CHECK("pressed", no_args);
		if (button->is_toggle_mode()) {
			SIGNAL_CHECK("toggled", toggled_args);
		}
	} else {
		SIGNAL_CHECK_FALSE("pressed");
	}

	if (cancel_before_long_press) {
		// Android cancels the original touch before reporting the long press.
		touch->set_pressed(false);
		touch->set_canceled(true);
		static_cast<Control *>(button)->gui_input(touch);
	}
	touch->set_canceled(false);
	touch->set_pressed(true);
	touch->set_long_press(true);
	static_cast<Control *>(button)->gui_input(touch);
	SIGNAL_CHECK_FALSE("pressed");
	SIGNAL_CHECK_FALSE("toggled");

	touch->set_pressed(false);
	static_cast<Control *>(button)->gui_input(touch);
	if (activate_on_press) {
		SIGNAL_CHECK_FALSE("pressed");
		SIGNAL_CHECK_FALSE("toggled");
	} else {
		SIGNAL_CHECK("pressed", no_args);
		if (button->is_toggle_mode()) {
			SIGNAL_CHECK("toggled", toggled_args);
		}
	}
	if (button->is_toggle_mode()) {
		CHECK(button->is_pressed());
	}

	// The next ordinary tap must still activate the button.
	touch->set_long_press(false);
	touch->set_pressed(true);
	static_cast<Control *>(button)->gui_input(touch);
	touch->set_pressed(false);
	static_cast<Control *>(button)->gui_input(touch);
	SIGNAL_CHECK("pressed", no_args);
	if (button->is_toggle_mode()) {
		Array untoggled_args = { { false } };
		SIGNAL_CHECK("toggled", untoggled_args);
		CHECK_FALSE(button->is_pressed());
	}

	SIGNAL_UNWATCH(button, "pressed");
	SIGNAL_UNWATCH(button, "toggled");
	memdelete(button);
}

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

TEST_CASE("[SceneTree][Button] Click emits the pressed signal") {
	// Create new button instance.
	Button *button = memnew(Button);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(button);

	// Set up button's size and position.
	button->set_size(Size2i(50, 50));
	button->set_position(Size2i(10, 10));

	// Watch the button's interaction signals.
	SIGNAL_WATCH(button, "pressed");
	SIGNAL_WATCH(button, "button_down");
	SIGNAL_WATCH(button, "button_up");

	// These signals carry no arguments.
	Array empty_args = { {} };

	// Simulate mouse entering the button.
	SEND_GUI_MOUSE_MOTION_EVENT(Point2i(35, 35), MouseButtonMask::NONE, Key::NONE);

	// Pressing the mouse should fire "button_down" but not "pressed".
	SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(35, 35), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
	SIGNAL_CHECK("button_down", empty_args);
	SIGNAL_CHECK_FALSE("pressed");

	// Releasing the mouse should fire "pressed" and "button_up".
	SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(Point2i(35, 35), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
	SIGNAL_CHECK("pressed", empty_args);
	SIGNAL_CHECK("button_up", empty_args);

	SIGNAL_UNWATCH(button, "pressed");
	SIGNAL_UNWATCH(button, "button_down");
	SIGNAL_UNWATCH(button, "button_up");

	memdelete(button);
}

TEST_CASE("[SceneTree][Button] Disabled button ignores clicks") {
	// Create new button instance.
	Button *button = memnew(Button);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(button);

	// Set up button's size and position.
	button->set_size(Size2i(50, 50));
	button->set_position(Size2i(10, 10));

	// Disable the button.
	button->set_disabled(true);
	CHECK(button->is_disabled());

	// Watch the button's pressed signal.
	SIGNAL_WATCH(button, "pressed");

	// Simulate a full click on the button.
	SEND_GUI_MOUSE_MOTION_EVENT(Point2i(35, 35), MouseButtonMask::NONE, Key::NONE);
	SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(35, 35), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
	SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(Point2i(35, 35), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);

	// A disabled button should not emit "pressed".
	SIGNAL_CHECK_FALSE("pressed");

	SIGNAL_UNWATCH(button, "pressed");

	memdelete(button);
}

TEST_CASE("[SceneTree][Button] Toggle mode latches the pressed state") {
	// Create new button instance.
	Button *button = memnew(Button);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(button);

	// Watch the button's toggled signal.
	SIGNAL_WATCH(button, "toggled");

	// Without toggle mode, set_pressed() should be ignored.
	button->set_pressed(true);
	CHECK_FALSE(button->is_pressed());
	SIGNAL_CHECK_FALSE("toggled");

	// Enable toggle mode.
	button->set_toggle_mode(true);

	// Button should now stay pressed and emit "toggled".
	button->set_pressed(true);
	CHECK(button->is_pressed());
	SIGNAL_CHECK("toggled", { { true } });

	// Button should stay unpressed and emit "toggled" again.
	button->set_pressed(false);
	CHECK_FALSE(button->is_pressed());
	SIGNAL_CHECK("toggled", { { false } });

	SIGNAL_UNWATCH(button, "toggled");

	memdelete(button);
}

} // namespace TestButton
