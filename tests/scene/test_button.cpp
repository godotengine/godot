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

#include "core/input/input.h"
#include "core/input/input_map.h"
#include "scene/gui/button.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"
#include "tests/display_server_mock.h"
#include "tests/signal_watcher.h"

namespace TestButton {

TEST_CASE("[SceneTree][Button] is_hovered()") {
	// Create new button instance.
	Button *button = memnew(Button);
	REQUIRE(button != nullptr);
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

TEST_CASE("[SceneTree][Button] touch event with emulated mouse triggers pressed once") {
	Button *button = memnew(Button);
	REQUIRE(button != nullptr);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(button);

	button->set_size(Size2i(50, 50));
	button->set_position(Size2i(10, 10));
	button->set_action_mode(BaseButton::ACTION_MODE_BUTTON_PRESS);

	const bool was_emulating_mouse_from_touch = Input::get_singleton()->is_emulating_mouse_from_touch();
	Input::get_singleton()->set_emulate_mouse_from_touch(true);

	SEND_GUI_MOUSE_MOTION_EVENT(Point2i(25, 25), MouseButtonMask::NONE, Key::NONE);
	CHECK(button->is_hovered());

	SIGNAL_WATCH(button, SNAME("pressed"));

	SEND_GUI_TOUCH_EVENT(Point2i(25, 25), true, false);
	SIGNAL_CHECK(SNAME("pressed"), { {} });

	SEND_GUI_TOUCH_EVENT(Point2i(25, 25), false, false);
	SIGNAL_CHECK_FALSE(SNAME("pressed"));

	SIGNAL_UNWATCH(button, SNAME("pressed"));
	Input::get_singleton()->set_emulate_mouse_from_touch(was_emulating_mouse_from_touch);
	memdelete(button);
}

TEST_CASE("[SceneTree][Button] touch event with emulated mouse ui_accept triggers pressed once") {
	Button *button = memnew(Button);
	REQUIRE(button != nullptr);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(button);

	button->set_size(Size2i(50, 50));
	button->set_position(Size2i(10, 10));
	button->set_action_mode(BaseButton::ACTION_MODE_BUTTON_PRESS);

	Ref<InputEventMouseButton> ui_accept_event;
	ui_accept_event.instantiate();
	ui_accept_event->set_device(InputMap::ALL_DEVICES);
	ui_accept_event->set_button_index(MouseButton::LEFT);
	ui_accept_event->set_button_mask(MouseButtonMask::LEFT);
	InputMap::get_singleton()->action_add_event(SNAME("ui_accept"), ui_accept_event);

	const bool was_emulating_mouse_from_touch = Input::get_singleton()->is_emulating_mouse_from_touch();
	Input::get_singleton()->set_emulate_mouse_from_touch(true);

	SIGNAL_WATCH(button, SNAME("pressed"));

	SEND_GUI_TOUCH_EVENT(Point2i(25, 25), true, false);
	SIGNAL_CHECK(SNAME("pressed"), { {} });

	SEND_GUI_TOUCH_EVENT(Point2i(25, 25), false, false);
	SIGNAL_CHECK_FALSE(SNAME("pressed"));

	SIGNAL_UNWATCH(button, SNAME("pressed"));
	Input::get_singleton()->set_emulate_mouse_from_touch(was_emulating_mouse_from_touch);
	InputMap::get_singleton()->action_erase_event(SNAME("ui_accept"), ui_accept_event);
	memdelete(button);
}

TEST_CASE("[SceneTree][Button] emulated mouse ui_accept event does not trigger pressed") {
	Button *button = memnew(Button);
	REQUIRE(button != nullptr);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(button);

	button->set_size(Size2i(50, 50));
	button->set_position(Size2i(10, 10));
	button->set_action_mode(BaseButton::ACTION_MODE_BUTTON_PRESS);

	Ref<InputEventMouseButton> ui_accept_event;
	ui_accept_event.instantiate();
	ui_accept_event->set_device(InputMap::ALL_DEVICES);
	ui_accept_event->set_button_index(MouseButton::LEFT);
	ui_accept_event->set_button_mask(MouseButtonMask::LEFT);
	InputMap::get_singleton()->action_add_event(SNAME("ui_accept"), ui_accept_event);

	SEND_GUI_MOUSE_MOTION_EVENT(Point2i(25, 25), MouseButtonMask::NONE, Key::NONE);
	CHECK(button->is_hovered());

	SIGNAL_WATCH(button, SNAME("pressed"));

	Ref<InputEventMouseButton> event;
	event.instantiate();
	event->set_device(InputEvent::DEVICE_ID_EMULATION);
	event->set_position(Point2i(25, 25));
	event->set_button_index(MouseButton::LEFT);
	event->set_button_mask(MouseButtonMask::LEFT);
	event->set_factor(1);
	event->set_pressed(true);
	CHECK(event->is_action(SNAME("ui_accept"), true));

	_SEND_DISPLAYSERVER_EVENT(event);
	MessageQueue::get_singleton()->flush();

	SIGNAL_CHECK_FALSE(SNAME("pressed"));

	SIGNAL_UNWATCH(button, SNAME("pressed"));
	InputMap::get_singleton()->action_erase_event(SNAME("ui_accept"), ui_accept_event);
	memdelete(button);
}

} // namespace TestButton
