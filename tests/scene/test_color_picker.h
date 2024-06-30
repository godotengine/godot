/**************************************************************************/
/*  test_color_picker.h                                                   */
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

#ifndef TEST_COLOR_PICKER_H
#define TEST_COLOR_PICKER_H

#include "scene/gui/color_picker.h"

#include "tests/test_macros.h"

namespace TestColorPicker {

TEST_CASE("[SceneTree][ColorPicker]") {
	ColorPicker *cp = memnew(ColorPicker);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(cp);

	SUBCASE("[COLOR_PICKER] Mouse movement after Slider release") {
		Point2i pos_left = Point2i(50, 340); // On the left side of the red slider.
		Point2i pos_right = Point2i(200, 340); // On the right side of the red slider.
		SEND_GUI_MOUSE_MOTION_EVENT(pos_left, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(pos_left, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		CHECK(cp->get_pick_color().r < 0.5);
		SEND_GUI_MOUSE_MOTION_EVENT(pos_right, MouseButtonMask::LEFT, Key::NONE);
		CHECK(cp->get_pick_color().r > 0.5);
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(pos_right, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_MOTION_EVENT(pos_left, MouseButtonMask::NONE, Key::NONE);
		CHECK(cp->get_pick_color().r > 0.5); // Issue GH-77773.
	}
}

// Tests for opening the color picker, created to fix the Issue #91813
TEST_CASE("[SceneTree][ColorPicker][ColorPickerButton]") {
	ColorPickerButton *cpb = memnew(ColorPickerButton);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(cpb);

	CHECK(cpb != nullptr);
	CHECK(cpb->get_popup_panel() != nullptr);
	CHECK(!cpb->get_popup_panel()->is_visible());

	SUBCASE("[COLOR_PICKER_BUTTON] When pressed, picker pops up.") {
		// Press the button
		Point2i button_position = cpb->get_global_position();
		SEND_GUI_MOUSE_BUTTON_EVENT(button_position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);

		// Pop up shows
		CHECK(cpb->get_popup_panel() != nullptr);
		CHECK(cpb->get_popup_panel()->is_visible());
	}

	SUBCASE("[COLOR_PICKER_BUTTON] Pressing, picker pops and clicking outside the picker it closes.") {
		// Press the button
		Point2i button_position = cpb->get_global_position();
		SEND_GUI_MOUSE_BUTTON_EVENT(button_position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);

		// Pop up shows
		CHECK(cpb->get_popup_panel() != nullptr);
		CHECK(cpb->get_popup_panel()->is_visible());

		// Press the button again
		SEND_GUI_MOUSE_BUTTON_EVENT(button_position - Point2i(10, 10), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);

		// Pop up closes
		CHECK(!cpb->get_popup_panel()->is_visible());
	}

	SUBCASE("[COLOR_PICKER_BUTTON] Pressing, picker pops and clicking on the same spot it closes. (toggle_mode = true)") {
		// Press the button
		Point2i button_position = cpb->get_global_position();
		SEND_GUI_MOUSE_BUTTON_EVENT(button_position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);

		// Pop up shows
		CHECK(cpb->get_popup_panel() != nullptr);
		CHECK(cpb->get_popup_panel()->is_visible());

		// Press the button again
		SEND_GUI_MOUSE_BUTTON_EVENT(button_position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);

		// Pop up closes
		CHECK(!cpb->get_popup_panel()->is_visible());
	}

	SUBCASE("[COLOR_PICKER_BUTTON] Clicking outside the color picker closes it and doesn't reopen it") {
		// Open the picker
		Point2i button_position = cpb->get_global_position();
		SEND_GUI_MOUSE_BUTTON_EVENT(button_position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		CHECK(cpb->get_popup_panel()->is_visible());

		// Click outside the picker
		Point2i outside_position = cpb->get_global_position() - Point2i(100, 100); // Position outside the picker
		SEND_GUI_MOUSE_BUTTON_EVENT(outside_position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);

		// Check the picker is closed
		CHECK(!cpb->get_popup_panel()->is_visible());

		// Click outside again to verify it doesn't reopen
		SEND_GUI_MOUSE_BUTTON_EVENT(outside_position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		CHECK(!cpb->get_popup_panel()->is_visible());
	}

	SUBCASE("[COLOR_PICKER_BUTTON][COLOR_PICKER] Opening the picker, changing its color, and then closing saves the color.") {
		// Initial state checks
		CHECK(!cpb->get_popup_panel()->is_visible());

		// Open the picker
		Point2i button_position = cpb->get_global_position();
		SEND_GUI_MOUSE_BUTTON_EVENT(button_position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		CHECK(cpb->get_popup_panel()->is_visible());

		// Change the color
		ColorPicker *cp = cpb->get_picker();
		Color initial_color = cp->get_pick_color();
		Point2i slider_position = cp->get_global_position() + Point2i(50, 50); // Adjusted example position for slider
		SEND_GUI_MOUSE_BUTTON_EVENT(slider_position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_MOTION_EVENT(slider_position + Point2i(150, 0), MouseButtonMask::LEFT, Key::NONE); // Move slider
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(slider_position + Point2i(150, 0), MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		Color new_color = cp->get_pick_color();
		CHECK(new_color != initial_color); // Check color has changed

		// Close the picker by pressing the button again
		SEND_GUI_MOUSE_BUTTON_EVENT(button_position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		CHECK(!cpb->get_popup_panel()->is_visible());

		// Verify the color is saved
		CHECK(cp->get_pick_color() == new_color);
	}

	SUBCASE("[COLOR_PICKER_BUTTON][COLOR_PICKER] Picker button.") {
		// Initial state checks
		CHECK(!cpb->get_popup_panel()->is_visible());

		// Open the picker
		Point2i button_position = cpb->get_global_position();
		SEND_GUI_MOUSE_BUTTON_EVENT(button_position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		CHECK(cpb->get_popup_panel()->is_visible());

		// Press pick button
		ColorPicker *cp = cpb->get_picker();
		Color initial_color = cp->get_pick_color();
		Point2i pick_btn_pos = cp->get_pick_btn()->get_global_position() + Point2i(8, 8);
		SEND_GUI_MOUSE_BUTTON_EVENT(pick_btn_pos, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);

		// Picks the color (different from the original) and the color picker closes
		SEND_GUI_MOUSE_BUTTON_EVENT(button_position - Point2i(10, 10), MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		CHECK(cp->get_pick_color() != initial_color);
		CHECK(!cpb->get_popup_panel()->is_visible());
	}
}

} // namespace TestColorPicker

#endif // TEST_COLOR_PICKER_H
