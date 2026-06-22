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

#include "core/object/callable_mp.h"
#include "scene/gui/button.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"
#include "tests/display_server_mock.h"

namespace TestButton {
// Tests related to `BaseButton` using `Button` as a concrete implementation.
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

TEST_CASE("[SceneTree][BaseButton] is_pressed()") {
	// Create new button instance.
	Button *button = memnew(Button);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(button);

	// Set up button's size and position.
	button->set_size(Size2i(50, 50));
	button->set_position(Size2i(10, 10));
	SEND_GUI_MOUSE_MOTION_EVENT(Point2i(25, 25), MouseButtonMask::NONE, Key::NONE);

	SUBCASE("Mouse Not pressed") {
		// Button should initially be not pressed.
		CHECK(button->is_pressed() == false);
	}

	SUBCASE("Mouse Pressed") {
		// Simulate mouse press on the button.
		SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		CHECK(button->is_pressed() == true);
	}

	SUBCASE("Mouse Released") {
		// Simulate mouse release on the button.
		SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK(button->is_pressed() == false);
	}

	memdelete(button);
}

// Utilitary class to listen to button pressed and toggled signals.
class ButtonCompleteInteractionListener : public Object {
	GDCLASS(ButtonCompleteInteractionListener, Object);

public:
	int pressed_count = 0;
	int toggled_count = 0;
	bool last_toggled_state = false;

	void on_pressed() {
		pressed_count++;
	}

	void on_toggled(bool p_state) {
		toggled_count++;
		last_toggled_state = p_state;
	}
};

TEST_CASE("[SceneTree][BaseButton] Action Mode (Press vs Release)") {
	// Button configuration.
	Button *button = memnew(Button);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(button);
	button->set_size(Size2i(50, 50));
	button->set_position(Size2i(10, 10));

	// Ensure mouse is over the button.
	SEND_GUI_MOUSE_MOTION_EVENT(Point2i(25, 25), MouseButtonMask::NONE, Key::NONE);

	// Listener configuration.
	ButtonCompleteInteractionListener *listener = memnew(ButtonCompleteInteractionListener);
	button->connect("pressed", callable_mp(listener, &ButtonCompleteInteractionListener::on_pressed));
	button->connect("toggled", callable_mp(listener, &ButtonCompleteInteractionListener::on_toggled));

	SUBCASE("Default Action Mode: ACTION_MODE_BUTTON_RELEASE") {
		// Ensure the default mode is to trigger on release.
		CHECK(button->get_action_mode() == BaseButton::ACTION_MODE_BUTTON_RELEASE);

		// Simulate pressing the button.
		SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		CHECK(button->is_pressed() == true); // Button should be in pressed state immediately on press, regardless of action mode.
		CHECK(listener->pressed_count == 0);
		CHECK(listener->toggled_count == 0); // Should not emit toggle signal since toggle mode is off

		// Simulate releasing the button.
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK(button->is_pressed() == false); // Button should return to not pressed state on release.
		CHECK(listener->pressed_count == 1);
		CHECK(listener->toggled_count == 0); // Should not emit toggled signal since toggle mode is off
	}

	SUBCASE("Modified Action Mode: ACTION_MODE_BUTTON_PRESS") {
		// Change action mode to trigger on press instead of release.
		button->set_action_mode(BaseButton::ACTION_MODE_BUTTON_PRESS);

		// Simulate pressing the button.
		SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		CHECK(button->is_pressed() == true); // Button should be in pressed state immediately on press.
		CHECK(listener->pressed_count == 1);
		CHECK(listener->toggled_count == 0); // Should not emit toggled signal since toggle mode is off

		// Simulate releasing the button.
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK(button->is_pressed() == false); // Button should return to not pressed state on release.
		CHECK(listener->pressed_count == 1); // Should not increment on release since we're in press mode.
		CHECK(listener->toggled_count == 0); // Should not emit toggled signal since toggle mode is off
	}

	memdelete(listener);
	memdelete(button);
}

TEST_CASE("[SceneTree][BaseButton] Toggle Mode") {
	// Button configuration.
	Button *button = memnew(Button);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(button);
	button->set_size(Size2i(50, 50));
	button->set_position(Size2i(10, 10));

	// Listener configuration: connect to both pressed and toggled signals to verify their behavior in toggle mode.
	ButtonCompleteInteractionListener *listener = memnew(ButtonCompleteInteractionListener);
	button->connect("pressed", callable_mp(listener, &ButtonCompleteInteractionListener::on_pressed));
	button->connect("toggled", callable_mp(listener, &ButtonCompleteInteractionListener::on_toggled));

	// Ensure mouse is over the button.
	SEND_GUI_MOUSE_MOTION_EVENT(Point2i(25, 25), MouseButtonMask::NONE, Key::NONE);

	SUBCASE("Toggle Mode with Release Action Mode") {
		button->set_toggle_mode(true);

		// Mouse pressed
		SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		CHECK(button->is_pressed() == false); // With toggle mode on, is_pressed represents the toggled state, which should not change until the click is released.
		CHECK(listener->pressed_count == 0); // Not emit until release
		CHECK(listener->toggled_count == 0); // Not emit until release

		// Mouse released
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK(button->is_pressed() == true); // Retain state
		CHECK(listener->pressed_count == 1); // Emit on release
		CHECK(listener->toggled_count == 1); // Emit on release
		CHECK(listener->last_toggled_state == true);

		// Mouse pressed again to untoggle
		SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		CHECK(listener->pressed_count == 1); // Not emit until release
		CHECK(listener->toggled_count == 1); // Not emit until release

		// Mouse released
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK(button->is_pressed() == false); // Retain state
		CHECK(listener->pressed_count == 2); // Emit on release
		CHECK(listener->toggled_count == 2); // Emit on release
		CHECK(listener->last_toggled_state == false);
	}

	SUBCASE("Toggle Mode with Press Action Mode") {
		button->set_toggle_mode(true);
		button->set_action_mode(BaseButton::ACTION_MODE_BUTTON_PRESS);

		// Mouse pressed
		SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		CHECK(button->is_pressed() == true); // With toggle mode on and action mode set to press, is_pressed should reflect the toggled state immediately on press.
		CHECK(listener->pressed_count == 1); // Emit on press
		CHECK(listener->toggled_count == 1); // Emit on press
		CHECK(listener->last_toggled_state == true);

		// Mouse released
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK(button->is_pressed() == true); // State should remain the same after release since it toggled on press.
		CHECK(listener->pressed_count == 1); // Should not emit on release since we're in press mode.
		CHECK(listener->toggled_count == 1); // Should not emit on release since we're in press mode.

		// Mouse pressed again to untoggle
		SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		CHECK(button->is_pressed() == false); // Should toggle immediately on press.
		CHECK(listener->pressed_count == 2); // Emit on press
		CHECK(listener->toggled_count == 2); // Emit on press
		CHECK(listener->last_toggled_state == false);

		// Mouse released
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK(button->is_pressed() == false); // State should remain the same after release since it toggled on press.
		CHECK(listener->pressed_count == 2); // Should not emit on release since we're in press mode.
		CHECK(listener->toggled_count == 2); // Should not emit on release since we're in press mode.
	}

	memdelete(listener);
	memdelete(button);
}

TEST_CASE("[SceneTree][BaseButton] set_pressed_no_signal Behavior") {
	// Button configuration.
	Button *button = memnew(Button);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(button);

	// Listener configuration
	ButtonCompleteInteractionListener *listener = memnew(ButtonCompleteInteractionListener);
	button->connect("pressed", callable_mp(listener, &ButtonCompleteInteractionListener::on_pressed));
	button->connect("toggled", callable_mp(listener, &ButtonCompleteInteractionListener::on_toggled));

	SUBCASE("With toggle_mode = false (Should be ignored)") {
		CHECK(button->is_toggle_mode() == false);

		// Attempt to set the button as pressed using set_pressed_no_signal.
		button->set_pressed_no_signal(true);

		// Only works if toggle_mode is true
		// The state should not change, and no signals should be emitted.
		CHECK(button->is_pressed() == false);
		CHECK(listener->pressed_count == 0);
		CHECK(listener->toggled_count == 0);
	}

	SUBCASE("With toggle_mode = true (Changes state silently)") {
		button->set_toggle_mode(true);

		// Set as pressed using set_pressed_no_signal.
		button->set_pressed_no_signal(true);

		// The state should change to true
		CHECK(button->is_pressed() == true);
		// Signals should not be emitted.
		CHECK(listener->pressed_count == 0);
		CHECK(listener->toggled_count == 0);

		// Set as not pressed using set_pressed_no_signal.
		button->set_pressed_no_signal(false);

		// The state should change to false
		CHECK(button->is_pressed() == false);
		// Signals should not be emitted.
		CHECK(listener->pressed_count == 0);
		CHECK(listener->toggled_count == 0);
	}

	memdelete(listener);
	memdelete(button);
}

TEST_CASE("[SceneTree][BaseButton] Disabled State Behavior") {
	// Button configuration.
	Button *button = memnew(Button);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(button);
	button->set_size(Size2i(50, 50));
	button->set_position(Size2i(10, 10));

	// Listener configuration
	ButtonCompleteInteractionListener *listener = memnew(ButtonCompleteInteractionListener);
	button->connect("pressed", callable_mp(listener, &ButtonCompleteInteractionListener::on_pressed));
	button->connect("toggled", callable_mp(listener, &ButtonCompleteInteractionListener::on_toggled));

	// Disable the button
	button->set_disabled(true);

	SUBCASE("Should ignore mouse hover") {
		// Move the mouse over the button
		SEND_GUI_MOUSE_MOTION_EVENT(Point2i(25, 25), MouseButtonMask::NONE, Key::NONE);

		// Since it's disabled, it should not enter the hovered state
		CHECK(button->is_hovered() == false);
	}

	SUBCASE("Should ignore mouse clicks and emit no signals") {
		button->set_toggle_mode(true); // Ativa toggle só para garantir o teste mais rigoroso

		// Move the mouse over the button and try to click
		SEND_GUI_MOUSE_MOTION_EVENT(Point2i(25, 25), MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);

		// State should remain unchanged (not pressed) since the button is disabled
		CHECK(button->is_pressed() == false);
		// Simulate mouse release
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);

		// No signals should be emitted when interacting with a disabled button
		CHECK(listener->pressed_count == 0);
		CHECK(listener->toggled_count == 0);
	}

	SUBCASE("Re-enabling button restores normal behavior") {
		// Re-enable the button
		button->set_disabled(false);
		CHECK(button->is_disabled() == false);

		// Move the mouse over the button
		SEND_GUI_MOUSE_MOTION_EVENT(Point2i(25, 25), MouseButtonMask::NONE, Key::NONE);

		// Now it should enter the hovered state
		CHECK(button->is_hovered() == true);

		// Try clicking the button
		SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		CHECK(button->is_pressed() == true); // State should change to pressed since the button is now enabled
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);

		// Signals should be emitted as normal when the button is enabled
		CHECK(listener->pressed_count == 1);
	}

	memdelete(listener);
	memdelete(button);
}

TEST_CASE("[SceneTree][BaseButton] get_draw_mode() State Machine") {
	// Button configuration.
	Button *button = memnew(Button);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(button);
	button->set_size(Size2i(50, 50));
	button->set_position(Size2i(10, 10));

	SUBCASE("DRAW_NORMAL State") {
		// Initially, the button should be in the normal state.
		CHECK(button->get_draw_mode() == BaseButton::DRAW_NORMAL);
	}

	SUBCASE("DRAW_HOVER State") {
		// Move the mouse over the button to trigger hover state.
		SEND_GUI_MOUSE_MOTION_EVENT(Point2i(25, 25), MouseButtonMask::NONE, Key::NONE);
		CHECK(button->get_draw_mode() == BaseButton::DRAW_HOVER);
	}

	SUBCASE("DRAW_PRESSED State with toggle_mode") {
		// To test the pure DRAW_PRESSED state (without hover), the mouse needs to click
		// and then move away from the button while holding the click (or using toggle_mode)
		button->set_toggle_mode(true);

		// Click and release on the button to activate the toggle
		SEND_GUI_MOUSE_MOTION_EVENT(Point2i(25, 25), MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		// Move the mouse away while the button is toggled (pressed)
		SEND_GUI_MOUSE_MOTION_EVENT(Point2i(150, 150), MouseButtonMask::NONE, Key::NONE);

		// Now the button should be in the pressed state without hover
		CHECK(button->get_draw_mode() == BaseButton::DRAW_PRESSED);
	}

	SUBCASE("DRAW_PRESSED State without toggle_mode") {
		// Click and hold on the button to trigger pressed state without toggle mode
		SEND_GUI_MOUSE_MOTION_EVENT(Point2i(25, 25), MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_MOTION_EVENT(Point2i(100, 100), MouseButtonMask::NONE, Key::NONE);

		// The button should be in the pressed state even if the mouse moves away while holding the click
		CHECK(button->get_draw_mode() == BaseButton::DRAW_PRESSED);

		// Release the click to return to normal state
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(Point2i(100, 100), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK(button->get_draw_mode() == BaseButton::DRAW_NORMAL);
	}
	SUBCASE("DRAW_HOVER_PRESSED State with toggle_mode") {
		button->set_toggle_mode(true);

		// Click and release on the button to activate the toggle
		SEND_GUI_MOUSE_MOTION_EVENT(Point2i(25, 25), MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);

		// Button should be in pressed + hovered state since it's toggled on and mouse is still over it
		CHECK(button->get_draw_mode() == BaseButton::DRAW_HOVER_PRESSED);
	}

	SUBCASE("DRAW_HOVER_PRESSED State without toggle_mode") {
		// Click and hold on the button to trigger hover+pressed state without toggle mode
		SEND_GUI_MOUSE_MOTION_EVENT(Point2i(25, 25), MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);

		// The button should be in the hover+pressed state while the click is held and mouse is over it
		CHECK(button->get_draw_mode() == BaseButton::DRAW_HOVER_PRESSED);

		// Release the click to return to normal state
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK(button->get_draw_mode() == BaseButton::DRAW_HOVER);
	}

	SUBCASE("DRAW_DISABLED State") {
		// Disable the button to trigger the disabled state.
		button->set_disabled(true);

		// Move the mouse over the button to verify it remains in the disabled state.
		SEND_GUI_MOUSE_MOTION_EVENT(Point2i(25, 25), MouseButtonMask::NONE, Key::NONE);

		CHECK(button->get_draw_mode() == BaseButton::DRAW_DISABLED);
	}

	memdelete(button);
}
} // namespace TestButton
