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
#include "scene/resources/placeholder_textures.h"
#include "servers/display/accessibility_server_dummy.h"
#include "servers/display/accessibility_server_enums.h"
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

// Utility class to listen to button pressed and toggled signals.
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

	SUBCASE("Should receive mouse hover") {
		// Move the mouse over the button
		SEND_GUI_MOUSE_MOTION_EVENT(Point2i(25, 25), MouseButtonMask::NONE, Key::NONE);

		// Even when disabled, the button should still detect hover state changes (but not pressed state).
		CHECK(button->is_hovered() == true);
	}

	SUBCASE("Should ignore mouse clicks and emit no signals") {
		button->set_toggle_mode(true);

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

		// The button should be
		CHECK(button->get_draw_mode() == BaseButton::DRAW_PRESSED);

		// Release the click to return to hover state
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK(button->get_draw_mode() == BaseButton::DRAW_HOVER); // Should return to hover state since mouse is still over it
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

	SUBCASE("DRAW_DISABLED State") {
		// Disable the button to trigger the disabled state.
		button->set_disabled(true);

		// Move the mouse over the button to verify it remains in the disabled state.
		SEND_GUI_MOUSE_MOTION_EVENT(Point2i(25, 25), MouseButtonMask::NONE, Key::NONE);

		CHECK(button->get_draw_mode() == BaseButton::DRAW_DISABLED);
	}

	memdelete(button);
}

// Tests related to `ButtonGroup` class and its interaction with `BaseButton`, using `Button` as a concrete implementation.
TEST_CASE("[SceneTree][ButtonGroup] Click interactions and allow_unpress") {
	// Scenario and Group Creation
	Window *root = SceneTree::get_singleton()->get_root();
	Ref<ButtonGroup> group = memnew(ButtonGroup);
	// Creating two toggle buttons and adding them to the same ButtonGroup.
	Button *btn1 = memnew(Button);
	btn1->set_toggle_mode(true);
	btn1->set_button_group(group);
	btn1->set_size(Size2i(40, 40));
	btn1->set_position(Size2i(10, 10)); // Occupies the area from (10, 10) to (50, 50)
	root->add_child(btn1);
	Button *btn2 = memnew(Button);
	btn2->set_toggle_mode(true);
	btn2->set_button_group(group);
	btn2->set_size(Size2i(40, 40));
	btn2->set_position(Size2i(60, 10)); // Occupies the area from (60, 10) to (100, 50)
	root->add_child(btn2);

	// Ensure the initial state is clean (no button pressed).
	CHECK(group->get_pressed_button() == nullptr);

	SUBCASE("Mutual exclusion between buttons") {
		group->set_allow_unpress(false);

		SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(30, 30), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(Point2i(30, 30), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);

		CHECK(btn1->is_pressed() == true);
		CHECK(group->get_pressed_button() == btn1);

		SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(80, 30), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(Point2i(80, 30), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);

		CHECK(btn1->is_pressed() == false);
		CHECK(btn2->is_pressed() == true);
		CHECK(group->get_pressed_button() == btn2);
	}

	SUBCASE("Attempt to uncheck with allow_unpress DISABLED") {
		group->set_allow_unpress(false);

		SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(30, 30), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(Point2i(30, 30), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK(btn1->is_pressed() == true);

		SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(30, 30), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(Point2i(30, 30), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);

		CHECK(btn1->is_pressed() == true);
		CHECK(group->get_pressed_button() == btn1);
	}

	SUBCASE("Allow unchecking with allow_unpress ENABLED") {
		group->set_allow_unpress(true);

		SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(30, 30), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(Point2i(30, 30), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK(btn1->is_pressed() == true);

		SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(30, 30), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(Point2i(30, 30), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);

		CHECK(btn1->is_pressed() == false);
		CHECK(group->get_pressed_button() == nullptr);
	}

	root->remove_child(btn1);
	root->remove_child(btn2);
	memdelete(btn1);
	memdelete(btn2);
}

TEST_CASE("[SceneTree][ButtonGroup] Management and Lifecycle of Buttons") {
	Window *root = SceneTree::get_singleton()->get_root();
	Ref<ButtonGroup> group = memnew(ButtonGroup);

	SUBCASE("Addition of buttons and exposure of lists") {
		Button *btn1 = memnew(Button);
		Button *btn2 = memnew(Button);
		root->add_child(btn1);
		root->add_child(btn2);

		// Adding buttons to the group
		btn1->set_button_group(group);
		btn2->set_button_group(group);

		// 1. Testing get_buttons method (used internally in C++)
		List<BaseButton *> button_list;
		group->get_buttons(&button_list);
		CHECK(button_list.size() == 2);
		CHECK(button_list.find(btn1) != nullptr);
		CHECK(button_list.find(btn2) != nullptr);

		// 2. Testing the _get_buttons method (the binder exposed for GDScript)
		TypedArray<BaseButton> typed_array = group->_get_buttons();
		CHECK(typed_array.size() == 2);
		CHECK(typed_array.find(btn1) != -1);
		CHECK(typed_array.find(btn2) != -1);

		memdelete(btn1);
		memdelete(btn2);
	}

	SUBCASE("Automatic removal of button when deleted (Lifecycle)") {
		Button *btn = memnew(Button);
		root->add_child(btn);
		btn->set_button_group(group);

		// Ensure the button was added to the group
		TypedArray<BaseButton> list_antes = group->_get_buttons();
		CHECK(list_antes.size() == 1);

		// Delete the button from memory, BaseButton has a destructor that notifies ButtonGroup to clean up.
		root->remove_child(btn);
		memdelete(btn);

		// The internal list of the ButtonGroup MUST be empty now
		TypedArray<BaseButton> list_depois = group->_get_buttons();
		CHECK(list_depois.size() == 0);
	}
}

// =========================================================================
// FOURTH WEEK: NEW TEST CASES BASED ON COVERAGE AND SPECIFICATION
// =========================================================================
TEST_CASE("[SceneTree][BaseButton] Input de Toque e Gestos") {
	Button *button = memnew(Button);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(button);
	button->set_size(Size2i(50, 50));
	button->set_position(Size2i(10, 10));

	// Global setup to prepare the button for the subcases.
	button->show();
	button->grab_focus();

	// Create a local listener to validate signal emissions.
	ButtonCompleteInteractionListener *gestures_listener = memnew(ButtonCompleteInteractionListener);
	button->connect("pressed", callable_mp(gestures_listener, &ButtonCompleteInteractionListener::on_pressed));

	SUBCASE("InputEventScreenTouch - Pressionar e Soltar na tela usando macros") {
		SEND_GUI_TOUCH_EVENT(Point2i(25, 25), true, false);
		CHECK(button->is_pressed() == true);
		CHECK(gestures_listener->pressed_count == 0);

		SEND_GUI_TOUCH_EVENT(Point2i(25, 25), false, false);
		CHECK(button->is_pressed() == false);
		CHECK(gestures_listener->pressed_count == 1);
	}

	SUBCASE("InputEventScreenDrag - Mover o dedo para fora do botão") {
		button->set_keep_pressed_outside(false);

		SEND_GUI_TOUCH_EVENT(Point2i(25, 25), true, false);
		CHECK(button->is_pressed() == true);

		Ref<InputEventScreenDrag> touch_drag_out;
		touch_drag_out.instantiate();
		touch_drag_out->set_index(0);
		touch_drag_out->set_position(Point2i(-200, -200));

		_SEND_DISPLAYSERVER_EVENT(touch_drag_out);
		MessageQueue::get_singleton()->flush();

		CHECK(button->get_draw_mode() == BaseButton::DRAW_NORMAL);

		SEND_GUI_TOUCH_EVENT(Point2i(-200, -200), false, false);
		CHECK(gestures_listener->pressed_count == 0);

		button->set_keep_pressed_outside(true);
		CHECK(button->is_keep_pressed_outside() == true);

		SEND_GUI_TOUCH_EVENT(Point2i(25, 25), true, false);
		CHECK(button->is_pressed() == true);

		_SEND_DISPLAYSERVER_EVENT(touch_drag_out->duplicate());
		MessageQueue::get_singleton()->flush();

		CHECK(button->is_pressed() == true);
		CHECK(button->get_draw_mode() == BaseButton::DRAW_PRESSED);

		SEND_GUI_TOUCH_EVENT(Point2i(-200, -200), false, false);
		CHECK(gestures_listener->pressed_count == 0);
	}

	SUBCASE("InputEventMouseMotion - Mover mouse para fora segurando clique") {
		button->set_keep_pressed_outside(false);

		SEND_GUI_MOUSE_MOTION_EVENT(Point2i(25, 25), MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		CHECK(button->is_pressed() == true);

		SEND_GUI_MOUSE_MOTION_EVENT(Point2i(-150, -150), MouseButtonMask::LEFT, Key::NONE);

		CHECK(button->get_draw_mode() == BaseButton::DRAW_NORMAL);

		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(Point2i(-150, -150), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK(gestures_listener->pressed_count == 0);

		button->set_keep_pressed_outside(true);

		SEND_GUI_MOUSE_MOTION_EVENT(Point2i(25, 25), MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		CHECK(button->is_pressed() == true);

		SEND_GUI_MOUSE_MOTION_EVENT(Point2i(-150, -150), MouseButtonMask::LEFT, Key::NONE);

		CHECK(button->is_pressed() == true);
		CHECK(button->get_draw_mode() == BaseButton::DRAW_PRESSED);

		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(Point2i(-150, -150), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK(gestures_listener->pressed_count == 0);
	}

	button->disconnect("pressed", callable_mp(gestures_listener, &ButtonCompleteInteractionListener::on_pressed));
	memdelete(gestures_listener);
	root->remove_child(button);
	memdelete(button);
}

TEST_CASE("[SceneTree][BaseButton] Notifications and System Lifecycle (_notification)") {
	Button *button = memnew(Button);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(button);
	button->set_size(Size2i(50, 50));
	button->set_position(Size2i(10, 10));

	// Listener used to catch any unintended signal emission.
	ButtonCompleteInteractionListener *listener = memnew(ButtonCompleteInteractionListener);
	button->connect("pressed", callable_mp(listener, &ButtonCompleteInteractionListener::on_pressed));

	SUBCASE("Cancellation by NOTIFICATION_DRAG_BEGIN and NOTIFICATION_SCROLL_BEGIN") {
		SEND_GUI_MOUSE_MOTION_EVENT(Point2i(25, 25), MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);

		button->notification(Control::NOTIFICATION_DRAG_BEGIN);

		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);

		CHECK(listener->pressed_count == 0);

		SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		button->notification(Control::NOTIFICATION_SCROLL_BEGIN);
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK(listener->pressed_count == 0);
	}

	SUBCASE("Losing keyboard/gamepad focus cancels the click in progress") {
		button->grab_focus();

		SEND_GUI_KEY_EVENT(Key::SPACE);

		button->notification(Control::NOTIFICATION_FOCUS_EXIT);

		SEND_GUI_KEY_UP_EVENT(Key::SPACE);

		CHECK(listener->pressed_count == 0);
	}

	root->remove_child(button);
	memdelete(listener);
	memdelete(button);
}

TEST_CASE("[SceneTree][BaseButton] Dynamic Transition State Changes") {
	Button *button = memnew(Button);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(button);
	button->set_size(Size2i(50, 50));
	button->set_position(Size2i(10, 10));

	ButtonCompleteInteractionListener *listener = memnew(ButtonCompleteInteractionListener);
	button->connect("toggled", callable_mp(listener, &ButtonCompleteInteractionListener::on_toggled));

	SUBCASE("set_toggle_mode(false) clears retained presses") {
		button->set_toggle_mode(true);

		button->set_pressed_no_signal(true);
		CHECK(button->is_pressed() == true);

		button->set_toggle_mode(false);
		CHECK(button->is_pressed() == false);
	}

	SUBCASE("set_button_group replaces and removes old references") {
		Ref<ButtonGroup> group_a = memnew(ButtonGroup);
		Ref<ButtonGroup> group_b = memnew(ButtonGroup);

		button->set_button_group(group_a);
		TypedArray<BaseButton> lista_a = group_a->_get_buttons();
		CHECK(lista_a.size() == 1);

		button->set_button_group(group_b);

		TypedArray<BaseButton> lista_a_pos = group_a->_get_buttons();
		TypedArray<BaseButton> lista_b_pos = group_b->_get_buttons();
		CHECK(lista_a_pos.size() == 0);
		CHECK(lista_b_pos.size() == 1);

		// Remover passando nulo
		button->set_button_group(Ref<ButtonGroup>());
	}

	root->remove_child(button);
	memdelete(listener);
	memdelete(button);
}

TEST_CASE("[SceneTree][BaseButton] Máscaras de Botão e Atalhos (Shortcut)") {
	Button *button = memnew(Button);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(button);
	button->set_size(Size2i(50, 50));
	button->set_position(Size2i(10, 10));

	ButtonCompleteInteractionListener *listener = memnew(ButtonCompleteInteractionListener);
	button->connect("pressed", callable_mp(listener, &ButtonCompleteInteractionListener::on_pressed));

	SUBCASE("Button mask API and behavior") {
		SEND_GUI_MOUSE_MOTION_EVENT(Point2i(25, 25), MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK(listener->pressed_count == 1);
		listener->pressed_count = 0;

		button->set_button_mask(MouseButtonMask::RIGHT);
		CHECK(button->get_button_mask() == MouseButtonMask::RIGHT);

		SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK(listener->pressed_count == 0);

		SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(25, 25), MouseButton::RIGHT, MouseButtonMask::RIGHT, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(Point2i(25, 25), MouseButton::RIGHT, MouseButtonMask::NONE, Key::NONE);
		CHECK(listener->pressed_count == 1);
		listener->pressed_count = 0;

		button->set_button_mask(MouseButtonMask::LEFT);
		CHECK(button->get_button_mask() == MouseButtonMask::LEFT);

		SEND_GUI_MOUSE_BUTTON_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(Point2i(25, 25), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK(listener->pressed_count == 1);
		listener->pressed_count = 0;

		button->disconnect("pressed", callable_mp(listener, &ButtonCompleteInteractionListener::on_pressed));
	}

	SUBCASE("Shortcut association and in_shortcut_feedback validation") {
		Ref<Shortcut> shortcut;
		shortcut.instantiate();

		Ref<InputEventKey> key_event;
		key_event.instantiate();
		key_event->set_keycode(Key::ENTER);

		TypedArray<InputEvent> events;
		events.append(key_event);
		shortcut->set_events(events);

		button->set_shortcut(shortcut);
		CHECK(button->get_shortcut() == shortcut);

		button->set_shortcut_feedback(false);
		CHECK(button->is_shortcut_feedback() == false);

		button->set_shortcut_in_tooltip(true);
		CHECK(button->is_shortcut_in_tooltip_enabled() == true);

		key_event->set_pressed(true);
		_SEND_DISPLAYSERVER_EVENT(key_event);
		MessageQueue::get_singleton()->flush();

		CHECK(listener->pressed_count == 1);
	}

	root->remove_child(button);
	memdelete(listener);
	memdelete(button);
}
} // namespace TestButton
