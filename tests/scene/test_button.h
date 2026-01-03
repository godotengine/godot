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

#pragma once

#include "core/input/input_event.h"
#include "core/input/shortcut.h"
#include "core/object/ref_counted.h"
#include "core/os/keyboard.h"
#include "scene/gui/button.h"
#include "scene/main/window.h"
#include "tests/test_macros.h"

namespace TestButton {
TEST_CASE("[SceneTree][Button] Getters and setters") {
	Button *button = memnew(Button);

	SUBCASE("Pressed") {
		CHECK_FALSE(button->is_pressed());

		REQUIRE_FALSE(button->is_toggle_mode());
		button->set_pressed(true);
		CHECK_FALSE_MESSAGE(button->is_pressed(), "Button can be set as pressed only in toggle mode.");
		button->set_pressed_no_signal(true);
		CHECK_FALSE_MESSAGE(button->is_pressed(), "Button can be set as pressed only in toggle mode.");

		// Enable toggle mode for set_pressed* to take effect
		button->set_toggle_mode(true);
		REQUIRE(button->is_toggle_mode());

		button->set_pressed(true);
		CHECK(button->is_pressed());
		button->set_pressed(false);
		CHECK_FALSE(button->is_pressed());
		button->set_pressed_no_signal(true);
		CHECK(button->is_pressed());
		button->set_pressed_no_signal(false);
		CHECK_FALSE(button->is_pressed());
	}

	SUBCASE("Toggle mode") {
		CHECK_FALSE(button->is_toggle_mode());

		button->set_toggle_mode(true);
		CHECK(button->is_toggle_mode());

		button->set_pressed(true);
		CHECK(button->is_pressed());
		button->set_toggle_mode(false);
		CHECK_FALSE(button->is_toggle_mode());
		CHECK_FALSE_MESSAGE(button->is_pressed(), "Button should be set as not pressed when disabling toggle mode.");
	}

	SUBCASE("Shortcut in tooltip") {
		CHECK(button->is_shortcut_in_tooltip_enabled());

		button->set_shortcut_in_tooltip(false);
		CHECK_FALSE(button->is_shortcut_in_tooltip_enabled());
		button->set_shortcut_in_tooltip(true);
		CHECK(button->is_shortcut_in_tooltip_enabled());
	}

	SUBCASE("Disabled") {
		CHECK_FALSE(button->is_disabled());

		button->set_disabled(true);
		CHECK(button->is_disabled());
		button->set_disabled(false);
		CHECK_FALSE(button->is_disabled());
	}

	SUBCASE("Action mode") {
		CHECK_EQ(button->get_action_mode(), Button::ActionMode::ACTION_MODE_BUTTON_RELEASE);

		button->set_action_mode(Button::ActionMode::ACTION_MODE_BUTTON_PRESS);
		CHECK_EQ(button->get_action_mode(), Button::ActionMode::ACTION_MODE_BUTTON_PRESS);
		button->set_action_mode(Button::ActionMode::ACTION_MODE_BUTTON_RELEASE);
		CHECK_EQ(button->get_action_mode(), Button::ActionMode::ACTION_MODE_BUTTON_RELEASE);
	}

	SUBCASE("Keep pressed outside") {
		CHECK_FALSE(button->is_keep_pressed_outside());

		button->set_keep_pressed_outside(true);
		CHECK(button->is_keep_pressed_outside());
		button->set_keep_pressed_outside(false);
		CHECK_FALSE(button->is_keep_pressed_outside());
	}

	SUBCASE("Shortcut feedback") {
		CHECK(button->is_shortcut_feedback());

		button->set_shortcut_feedback(false);
		CHECK_FALSE(button->is_shortcut_feedback());
		button->set_shortcut_feedback(true);
		CHECK(button->is_shortcut_feedback());
	}

	SUBCASE("Button mask") {
		CHECK_EQ(button->get_button_mask(), static_cast<int64_t>(MouseButtonMask::LEFT));

		button->set_button_mask(MouseButtonMask::RIGHT);
		CHECK_EQ(button->get_button_mask(), static_cast<int64_t>(MouseButtonMask::RIGHT));
		button->set_button_mask(MouseButtonMask::MIDDLE);
		CHECK_EQ(button->get_button_mask(), static_cast<int64_t>(MouseButtonMask::MIDDLE));
		button->set_button_mask(MouseButtonMask::MB_XBUTTON1);
		CHECK_EQ(button->get_button_mask(), static_cast<int64_t>(MouseButtonMask::MB_XBUTTON1));
		button->set_button_mask(MouseButtonMask::MB_XBUTTON2);
		CHECK_EQ(button->get_button_mask(), static_cast<int64_t>(MouseButtonMask::MB_XBUTTON2));
		button->set_button_mask(MouseButtonMask::NONE);
		CHECK_EQ(button->get_button_mask(), static_cast<int64_t>(MouseButtonMask::NONE));
	}

	SUBCASE("Shortcut") {
		CHECK(button->get_shortcut().is_null());

		Ref<Shortcut> shortcut1;
		shortcut1.instantiate();
		Ref<Shortcut> shortcut2;
		shortcut2.instantiate();

		button->set_shortcut(shortcut1);
		CHECK_EQ(button->get_shortcut(), shortcut1);
		button->set_shortcut(shortcut2);
		CHECK_EQ(button->get_shortcut(), shortcut2);
	}

	SUBCASE("Button group") {
		CHECK(button->get_button_group().is_null());

		Ref<ButtonGroup> button_group1;
		button_group1.instantiate();
		Ref<ButtonGroup> button_group2;
		button_group2.instantiate();

		button->set_button_group(button_group1);
		CHECK_EQ(button->get_button_group(), button_group1);
		button->set_button_group(button_group2);
		CHECK_EQ(button->get_button_group(), button_group2);
	}

	memdelete(button);
}

TEST_CASE("[SceneTree][Button][Signal] button_down") {
	Button *button = memnew(Button);

	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(button);

	button->set_size(Size2i(50, 50));
	button->set_position(Point2i(10, 10));

	SIGNAL_WATCH(button, "button_down");
	Array empty_signal_args;
	empty_signal_args.push_back(Array());

	const Point2i inside_button_position(25, 25);

	SEND_GUI_MOUSE_MOTION_EVENT(inside_button_position, MouseButtonMask::NONE, Key::NONE);
	SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);

	SIGNAL_CHECK("button_down", empty_signal_args);

	SIGNAL_UNWATCH(button, "button_down");
	memdelete(button);
}

TEST_CASE("[SceneTree][Button][Signal] button_up") {
	Button *button = memnew(Button);

	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(button);

	button->set_size(Size2i(50, 50));
	button->set_position(Point2i(10, 10));

	Array empty_signal_args;
	empty_signal_args.push_back(Array());

	const Point2i inside_button_position(25, 25);

	SUBCASE("Mouse button is released") {
		SIGNAL_WATCH(button, "button_up");

		SEND_GUI_MOUSE_MOTION_EVENT(inside_button_position, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);

		SIGNAL_CHECK("button_up", empty_signal_args);

		SIGNAL_UNWATCH(button, "button_up");
	}

	SUBCASE("Lost focus") {
		SIGNAL_WATCH(button, "button_up");

		SEND_GUI_MOUSE_MOTION_EVENT(inside_button_position, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		button->release_focus();

		SIGNAL_CHECK("button_up", empty_signal_args);

		SIGNAL_UNWATCH(button, "button_up");
	}

	memdelete(button);
}

TEST_CASE("[SceneTree][Button][Signal] pressed") {
	Button *button = memnew(Button);

	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(button);

	button->set_size(Size2i(50, 50));
	button->set_position(Point2i(10, 10));

	Array empty_signal_args;
	empty_signal_args.push_back(Array());

	const Point2i inside_button_position(25, 25);

	SUBCASE("Mouse button pressed, action mode - release") {
		SIGNAL_WATCH(button, SceneStringName(pressed));

		REQUIRE_EQ(button->get_action_mode(), Button::ActionMode::ACTION_MODE_BUTTON_RELEASE);

		SEND_GUI_MOUSE_MOTION_EVENT(inside_button_position, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);

		SIGNAL_CHECK_FALSE(SceneStringName(pressed));

		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);

		SIGNAL_CHECK(SceneStringName(pressed), empty_signal_args);

		SIGNAL_UNWATCH(button, SceneStringName(pressed));
	}

	SUBCASE("Mouse button pressed, action mode - press") {
		SIGNAL_WATCH(button, SceneStringName(pressed));

		button->set_action_mode(Button::ActionMode::ACTION_MODE_BUTTON_PRESS);
		REQUIRE_EQ(button->get_action_mode(), Button::ActionMode::ACTION_MODE_BUTTON_PRESS);

		SEND_GUI_MOUSE_MOTION_EVENT(inside_button_position, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);

		SIGNAL_CHECK(SceneStringName(pressed), empty_signal_args);

		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);

		SIGNAL_CHECK_FALSE(SceneStringName(pressed));

		SIGNAL_UNWATCH(button, SceneStringName(pressed));
	}

	SUBCASE("Mouse button pressed, action mode - release, toggle mode on") {
		SIGNAL_WATCH(button, SceneStringName(pressed));

		button->set_toggle_mode(true);
		REQUIRE(button->is_toggle_mode());
		REQUIRE_EQ(button->get_action_mode(), Button::ActionMode::ACTION_MODE_BUTTON_RELEASE);

		SEND_GUI_MOUSE_MOTION_EVENT(inside_button_position, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);

		SIGNAL_CHECK_FALSE(SceneStringName(pressed));

		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);

		SIGNAL_CHECK(SceneStringName(pressed), empty_signal_args);

		SIGNAL_UNWATCH(button, SceneStringName(pressed));
	}

	SUBCASE("Shortcut") {
		SIGNAL_WATCH(button, SceneStringName(pressed));

		Ref<InputEventKey> k;
		k.instantiate();
		k->set_keycode(Key::J);

		Array input_array;
		input_array.append(k);

		Ref<Shortcut> shortcut;
		shortcut.instantiate();
		shortcut->set_events(input_array);

		button->set_shortcut(shortcut);
		REQUIRE_EQ(button->get_shortcut(), shortcut);

		SEND_GUI_KEY_EVENT(Key::J);

		SIGNAL_CHECK(SceneStringName(pressed), empty_signal_args);

		SIGNAL_UNWATCH(button, SceneStringName(pressed));
	}

	SUBCASE("Shortcut, toggle mode is on") {
		SIGNAL_WATCH(button, SceneStringName(pressed));

		Ref<InputEventKey> k;
		k.instantiate();
		k->set_keycode(Key::J);

		Array input_array;
		input_array.append(k);

		Ref<Shortcut> shortcut;
		shortcut.instantiate();
		shortcut->set_events(input_array);

		button->set_shortcut(shortcut);
		REQUIRE_EQ(button->get_shortcut(), shortcut);

		button->set_toggle_mode(true);
		REQUIRE(button->is_toggle_mode());

		SEND_GUI_KEY_EVENT(Key::J);

		SIGNAL_CHECK(SceneStringName(pressed), empty_signal_args);

		SIGNAL_UNWATCH(button, SceneStringName(pressed));
	}

	SUBCASE("set_pressed_no_signal()") {
		SIGNAL_WATCH(button, SceneStringName(pressed));

		button->set_toggle_mode(true);
		REQUIRE(button->is_toggle_mode());
		REQUIRE_FALSE(button->is_pressed());

		button->set_pressed_no_signal(true);
		REQUIRE(button->is_pressed());

		SIGNAL_CHECK_FALSE(SceneStringName(pressed));

		SIGNAL_UNWATCH(button, SceneStringName(pressed));
	}

	memdelete(button);
}

TEST_CASE("[SceneTree][Button][Signal] toggled") {
	Button *button = memnew(Button);

	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(button);

	button->set_size(Size2i(50, 50));
	button->set_position(Point2i(10, 10));

	const Point2i inside_button_position(25, 25);

	SUBCASE("Mouse button") {
		SIGNAL_WATCH(button, SceneStringName(toggled));
		Array signal_args;
		Array pressed;
		pressed.push_back(true);
		signal_args.push_back(pressed);

		button->set_toggle_mode(true);
		REQUIRE(button->is_toggle_mode());

		SEND_GUI_MOUSE_MOTION_EVENT(inside_button_position, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);

		SIGNAL_CHECK(SceneStringName(toggled), signal_args);

		signal_args[0].set(0, false);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);

		SIGNAL_CHECK(SceneStringName(toggled), signal_args);

		SIGNAL_UNWATCH(button, SceneStringName(toggled));
	}

	SUBCASE("Mouse button, toggle mode is disabled") {
		SIGNAL_WATCH(button, SceneStringName(toggled));

		REQUIRE_FALSE(button->is_toggle_mode());

		SEND_GUI_MOUSE_MOTION_EVENT(inside_button_position, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);

		SIGNAL_CHECK_FALSE(SceneStringName(toggled));

		SIGNAL_UNWATCH(button, SceneStringName(toggled));
	}

	SUBCASE("set_pressed()") {
		SIGNAL_WATCH(button, SceneStringName(toggled));
		Array signal_args;
		Array pressed;
		pressed.push_back(true);
		signal_args.push_back(pressed);

		button->set_toggle_mode(true);
		REQUIRE(button->is_toggle_mode());
		REQUIRE_FALSE(button->is_pressed());

		button->set_pressed(true);

		SIGNAL_CHECK(SceneStringName(toggled), signal_args);

		SIGNAL_UNWATCH(button, SceneStringName(toggled));
	}

	SUBCASE("set_pressed(), same value") {
		SIGNAL_WATCH(button, SceneStringName(toggled));

		button->set_toggle_mode(true);
		REQUIRE(button->is_toggle_mode());
		REQUIRE_FALSE(button->is_pressed());

		button->set_pressed(false);

		SIGNAL_CHECK_FALSE(SceneStringName(toggled));

		SIGNAL_UNWATCH(button, SceneStringName(toggled));
	}

	SUBCASE("set_pressed_no_signal()") {
		SIGNAL_WATCH(button, SceneStringName(toggled));

		button->set_toggle_mode(true);
		REQUIRE(button->is_toggle_mode());
		REQUIRE_FALSE(button->is_pressed());

		button->set_pressed_no_signal(true);
		REQUIRE(button->is_pressed());

		SIGNAL_CHECK_FALSE(SceneStringName(toggled));

		SIGNAL_UNWATCH(button, SceneStringName(toggled));
	}

	SUBCASE("Disable toggle mode when button is pressed") {
		SIGNAL_WATCH(button, SceneStringName(toggled));
		Array signal_args;
		Array pressed;
		pressed.push_back(false);
		signal_args.push_back(pressed);

		button->set_toggle_mode(true);
		REQUIRE(button->is_toggle_mode());
		button->set_pressed(true);
		REQUIRE(button->is_pressed());

		// Discard the signal emitted when pressed is set
		SIGNAL_DISCARD(SceneStringName(toggled));

		button->set_toggle_mode(false);

		SIGNAL_CHECK(SceneStringName(toggled), signal_args);

		SIGNAL_UNWATCH(button, SceneStringName(toggled));
	}

	SUBCASE("Shortcut") {
		SIGNAL_WATCH(button, SceneStringName(toggled));
		Array signal_args;
		Array pressed;
		pressed.push_back(true);
		signal_args.push_back(pressed);

		Ref<InputEventKey> k1;
		Ref<InputEventKey> k2;
		k1.instantiate();
		k2.instantiate();
		k1->set_keycode(Key::J);
		k2->set_keycode(Key::K);

		Array input_array;
		input_array.append(k1);
		input_array.append(k2);

		Ref<Shortcut> shortcut;
		shortcut.instantiate();
		shortcut->set_events(input_array);

		button->set_shortcut(shortcut);
		REQUIRE_EQ(button->get_shortcut(), shortcut);

		button->set_toggle_mode(true);
		REQUIRE(button->is_toggle_mode());

		SEND_GUI_KEY_EVENT(Key::J);

		SIGNAL_CHECK(SceneStringName(toggled), signal_args);

		signal_args[0].set(0, false);
		SEND_GUI_KEY_EVENT(Key::K);

		SIGNAL_CHECK(SceneStringName(toggled), signal_args);

		SIGNAL_UNWATCH(button, SceneStringName(toggled));
	}

	SUBCASE("Shortcut, toggle mode is disabled") {
		SIGNAL_WATCH(button, SceneStringName(toggled));

		Ref<InputEventKey> k;
		k.instantiate();
		k->set_keycode(Key::J);

		Array input_array;
		input_array.append(k);

		Ref<Shortcut> shortcut;
		shortcut.instantiate();
		shortcut->set_events(input_array);

		button->set_shortcut(shortcut);
		REQUIRE_EQ(button->get_shortcut(), shortcut);
		REQUIRE_FALSE(button->is_toggle_mode());

		SEND_GUI_KEY_EVENT(Key::J);

		SIGNAL_CHECK_FALSE(SceneStringName(toggled));

		SIGNAL_UNWATCH(button, SceneStringName(toggled));
	}

	memdelete(button);
}

TEST_CASE("[SceneTree][Button] Draw mode") {
	Button *button = memnew(Button);

	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(button);

	button->set_size(Size2i(50, 50));
	button->set_position(Point2i(10, 10));

	const Point2i inside_button_position(25, 25);
	const Point2i outside_button_position(150, 150);

	SUBCASE("Normal") {
		CHECK_EQ(button->get_draw_mode(), Button::DrawMode::DRAW_NORMAL);
	}

	SUBCASE("Disabled") {
		button->set_disabled(true);
		CHECK_EQ(button->get_draw_mode(), Button::DrawMode::DRAW_DISABLED);
	}

	SUBCASE("Hover") {
		SEND_GUI_MOUSE_MOTION_EVENT(inside_button_position, MouseButtonMask::NONE, Key::NONE);
		CHECK_EQ(button->get_draw_mode(), Button::DrawMode::DRAW_HOVER);
	}

	SUBCASE("Draw pressed") {
		SEND_GUI_MOUSE_MOTION_EVENT(inside_button_position, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		CHECK_EQ(button->get_draw_mode(), Button::DrawMode::DRAW_PRESSED);
	}

	SUBCASE("Draw hover pressed") {
		button->set_toggle_mode(true);
		REQUIRE(button->is_toggle_mode());

		SEND_GUI_MOUSE_MOTION_EVENT(inside_button_position, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK_EQ(button->get_draw_mode(), Button::DrawMode::DRAW_HOVER_PRESSED);
	}

	SUBCASE("Draw hover pressed, shortcut feedback") {
		Ref<InputEventKey> k;
		k.instantiate();
		k->set_keycode(Key::J);

		Array input_array;
		input_array.append(k);

		Ref<Shortcut> shortcut;
		shortcut.instantiate();
		shortcut->set_events(input_array);

		button->set_shortcut(shortcut);
		REQUIRE_EQ(button->get_shortcut(), shortcut);
		REQUIRE(button->is_shortcut_feedback());

		SEND_GUI_KEY_EVENT(Key::J);
		CHECK_EQ(button->get_draw_mode(), Button::DrawMode::DRAW_HOVER_PRESSED);
	}

	memdelete(button);
}

TEST_CASE("[SceneTree][Button] Disabling button while attempting to press") {
	Button *button = memnew(Button);

	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(button);

	button->set_size(Size2i(50, 50));
	button->set_position(Point2i(10, 10));

	const Point2i inside_button_position(25, 25);

	SEND_GUI_MOUSE_MOTION_EVENT(inside_button_position, MouseButtonMask::NONE, Key::NONE);
	SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
	REQUIRE(button->is_pressing());

	button->set_disabled(true);
	CHECK(button->is_disabled());
	CHECK_FALSE_MESSAGE(button->is_pressing(), "Button should be set as not pressing when disabled.");

	memdelete(button);
}

TEST_CASE("[SceneTree][Button] Button interactions") {
	Button *button = memnew(Button);

	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(button);

	button->set_size(Size2i(50, 50));
	button->set_position(Point2i(10, 10));

	const Point2i inside_button_position(25, 25);

	SUBCASE("Pressed") {
		SUBCASE("Toggle mode off") {
			REQUIRE_FALSE(button->is_toggle_mode());
			CHECK_FALSE(button->is_pressed());

			SEND_GUI_MOUSE_MOTION_EVENT(inside_button_position, MouseButtonMask::NONE, Key::NONE);
			SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
			CHECK(button->is_pressed());

			SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
			CHECK_FALSE(button->is_pressed());
		}

		SUBCASE("Toggle mode on") {
			button->set_toggle_mode(true);
			REQUIRE(button->is_toggle_mode());
			CHECK_FALSE(button->is_pressed());

			SEND_GUI_MOUSE_MOTION_EVENT(inside_button_position, MouseButtonMask::NONE, Key::NONE);
			SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
			SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
			CHECK(button->is_pressed());
		}

		SUBCASE("Toggle mode on, shortcut") {
			Ref<InputEventKey> k;
			k.instantiate();
			k->set_keycode(Key::J);

			Array input_array;
			input_array.append(k);

			Ref<Shortcut> shortcut;
			shortcut.instantiate();
			shortcut->set_events(input_array);

			button->set_shortcut(shortcut);
			REQUIRE_EQ(button->get_shortcut(), shortcut);
			button->set_toggle_mode(true);
			REQUIRE(button->is_toggle_mode());
			CHECK_FALSE(button->is_pressed());

			SEND_GUI_KEY_EVENT(Key::J);
			CHECK(button->is_pressed());
		}
	}

	SUBCASE("Pressing") {
		CHECK_FALSE(button->is_pressing());

		SEND_GUI_MOUSE_MOTION_EVENT(inside_button_position, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		CHECK(button->is_pressing());

		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		CHECK_FALSE(button->is_pressing());
	}

	SUBCASE("Hovered") {
		CHECK_FALSE(button->is_hovered());

		const Point2i outside_button_position(150, 150);

		SEND_GUI_MOUSE_MOTION_EVENT(inside_button_position, MouseButtonMask::NONE, Key::NONE);
		CHECK(button->is_hovered());

		SEND_GUI_MOUSE_MOTION_EVENT(outside_button_position, MouseButtonMask::NONE, Key::NONE);
		CHECK_FALSE(button->is_hovered());
	}

	SUBCASE("Hovered, visibility changed") {
		CHECK_FALSE(button->is_hovered());

		SEND_GUI_MOUSE_MOTION_EVENT(inside_button_position, MouseButtonMask::NONE, Key::NONE);
		REQUIRE(button->is_hovered());

		button->hide();
		CHECK_FALSE(button->is_hovered());
	}

	memdelete(button);
}

TEST_CASE("[SceneTree][Button] Button mask") {
	Button *button = memnew(Button);

	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(button);

	button->set_size(Size2i(50, 50));
	button->set_position(Point2i(10, 10));

	const Point2i inside_button_position(25, 25);

	SUBCASE("Button mask: left") {
		REQUIRE_EQ(button->get_button_mask(), static_cast<int64_t>(MouseButtonMask::LEFT));

		SEND_GUI_MOUSE_MOTION_EVENT(inside_button_position, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		CHECK(button->is_pressed());

		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::RIGHT, MouseButtonMask::RIGHT, Key::NONE);
		CHECK_FALSE(button->is_pressed());

		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(inside_button_position, MouseButton::RIGHT, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::MIDDLE, MouseButtonMask::MIDDLE, Key::NONE);
		CHECK_FALSE(button->is_pressed());

		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(inside_button_position, MouseButton::MIDDLE, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::MB_XBUTTON1, MouseButtonMask::MB_XBUTTON1, Key::NONE);
		CHECK_FALSE(button->is_pressed());

		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(inside_button_position, MouseButton::MB_XBUTTON1, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::MB_XBUTTON2, MouseButtonMask::MB_XBUTTON2, Key::NONE);
		CHECK_FALSE(button->is_pressed());
	}

	SUBCASE("Button mask: right") {
		button->set_button_mask(MouseButtonMask::RIGHT);
		REQUIRE_EQ(button->get_button_mask(), static_cast<int64_t>(MouseButtonMask::RIGHT));

		SEND_GUI_MOUSE_MOTION_EVENT(inside_button_position, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		CHECK_FALSE(button->is_pressed());

		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::RIGHT, MouseButtonMask::RIGHT, Key::NONE);
		CHECK(button->is_pressed());

		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(inside_button_position, MouseButton::RIGHT, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::MIDDLE, MouseButtonMask::MIDDLE, Key::NONE);
		CHECK_FALSE(button->is_pressed());

		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(inside_button_position, MouseButton::MIDDLE, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::MB_XBUTTON1, MouseButtonMask::MB_XBUTTON1, Key::NONE);
		CHECK_FALSE(button->is_pressed());

		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(inside_button_position, MouseButton::MB_XBUTTON1, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::MB_XBUTTON2, MouseButtonMask::MB_XBUTTON2, Key::NONE);
		CHECK_FALSE(button->is_pressed());
	}

	SUBCASE("Button mask: middle") {
		button->set_button_mask(MouseButtonMask::MIDDLE);
		REQUIRE_EQ(button->get_button_mask(), static_cast<int64_t>(MouseButtonMask::MIDDLE));

		SEND_GUI_MOUSE_MOTION_EVENT(inside_button_position, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		CHECK_FALSE(button->is_pressed());

		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::RIGHT, MouseButtonMask::RIGHT, Key::NONE);
		CHECK_FALSE(button->is_pressed());

		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(inside_button_position, MouseButton::RIGHT, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::MIDDLE, MouseButtonMask::MIDDLE, Key::NONE);
		CHECK(button->is_pressed());

		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(inside_button_position, MouseButton::MIDDLE, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::MB_XBUTTON1, MouseButtonMask::MB_XBUTTON1, Key::NONE);
		CHECK_FALSE(button->is_pressed());

		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(inside_button_position, MouseButton::MB_XBUTTON1, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::MB_XBUTTON2, MouseButtonMask::MB_XBUTTON2, Key::NONE);
		CHECK_FALSE(button->is_pressed());
	}

	SUBCASE("Button mask: xbutton1") {
		button->set_button_mask(MouseButtonMask::MB_XBUTTON1);
		REQUIRE_EQ(button->get_button_mask(), static_cast<int64_t>(MouseButtonMask::MB_XBUTTON1));

		SEND_GUI_MOUSE_MOTION_EVENT(inside_button_position, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		CHECK_FALSE(button->is_pressed());

		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::RIGHT, MouseButtonMask::RIGHT, Key::NONE);
		CHECK_FALSE(button->is_pressed());

		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(inside_button_position, MouseButton::RIGHT, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::MIDDLE, MouseButtonMask::MIDDLE, Key::NONE);
		CHECK_FALSE(button->is_pressed());

		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(inside_button_position, MouseButton::MIDDLE, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::MB_XBUTTON1, MouseButtonMask::MB_XBUTTON1, Key::NONE);
		CHECK(button->is_pressed());

		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(inside_button_position, MouseButton::MB_XBUTTON1, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::MB_XBUTTON2, MouseButtonMask::MB_XBUTTON2, Key::NONE);
		CHECK_FALSE(button->is_pressed());
	}

	SUBCASE("Button mask: xbutton2") {
		button->set_button_mask(MouseButtonMask::MB_XBUTTON2);
		REQUIRE_EQ(button->get_button_mask(), static_cast<int64_t>(MouseButtonMask::MB_XBUTTON2));

		SEND_GUI_MOUSE_MOTION_EVENT(inside_button_position, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::LEFT, Key::NONE);
		CHECK_FALSE(button->is_pressed());

		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(inside_button_position, MouseButton::LEFT, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::RIGHT, MouseButtonMask::RIGHT, Key::NONE);
		CHECK_FALSE(button->is_pressed());

		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(inside_button_position, MouseButton::RIGHT, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::MIDDLE, MouseButtonMask::MIDDLE, Key::NONE);
		CHECK_FALSE(button->is_pressed());

		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(inside_button_position, MouseButton::MIDDLE, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::MB_XBUTTON1, MouseButtonMask::MB_XBUTTON1, Key::NONE);
		CHECK_FALSE(button->is_pressed());

		SEND_GUI_MOUSE_BUTTON_RELEASED_EVENT(inside_button_position, MouseButton::MB_XBUTTON1, MouseButtonMask::NONE, Key::NONE);
		SEND_GUI_MOUSE_BUTTON_EVENT(inside_button_position, MouseButton::MB_XBUTTON2, MouseButtonMask::MB_XBUTTON2, Key::NONE);
		CHECK(button->is_pressed());
	}

	memdelete(button);
}

} //namespace TestButton
