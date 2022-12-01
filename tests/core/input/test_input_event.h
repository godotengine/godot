/*************************************************************************/
/*  test_input_event.h                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef TEST_INPUT_EVENT_H
#define TEST_INPUT_EVENT_H

#include "core/input/input_event.h"
#include "core/input/input_map.h"

#include "tests/test_macros.h"

#define INPUT_EVENT_TYPES InputEventWithModifiers, \
						  InputEventKey,           \
						  InputEventMouse,         \
						  InputEventMouseButton,   \
						  InputEventMouseMotion,   \
						  InputEventJoypadMotion,  \
						  InputEventJoypadButton,  \
						  InputEventScreenTouch,   \
						  InputEventScreenDrag,    \
						  InputEventAction,        \
						  InputEventGesture,       \
						  InputEventPanGesture,    \
						  InputEventMIDI,          \
						  InputEventShortcut

namespace TestInputEvent {

TEST_CASE_TEMPLATE_DEFINE("[InputEvent] Event correctly changes and returns device", T, test_device) {
	T event = T();

	event.InputEvent::set_device(InputEvent::DEVICE_ID_TOUCH_MOUSE);
	CHECK(event.InputEvent::get_device() == InputEvent::DEVICE_ID_TOUCH_MOUSE);
	CHECK(event.InputEvent::get_device() != InputEvent::DEVICE_ID_INTERNAL);

	event.set_device(InputEvent::DEVICE_ID_INTERNAL);
	CHECK(event.InputEvent::get_device() == InputEvent::DEVICE_ID_INTERNAL);
	CHECK(event.InputEvent::get_device() != InputEvent::DEVICE_ID_TOUCH_MOUSE);
}
TEST_CASE_TEMPLATE_INVOKE(test_device, INPUT_EVENT_TYPES);

TEST_CASE_TEMPLATE_DEFINE("[InputEvent] Test action properties through InputEvent", T, test_actions) {
	T event = T();
	InputMap inputMap = InputMap();

	StringName testEventName("TestInputEvent");
	InputMap::get_singleton()->add_action(testEventName);

	SUBCASE("[InputEvent] Action is present") {
		CHECK(event.InputEvent::is_action(testEventName) == true);
		CHECK(event.InputEvent::is_action(testEventName, true) == true);
	}

	SUBCASE("[InputEvent] Action has default pressed value") {
		//default Action value is false
		CHECK(event.InputEvent::is_action_pressed(testEventName) == false);
		CHECK(event.InputEvent::is_action_pressed(testEventName, true) == false);
		CHECK(event.InputEvent::is_action_pressed(testEventName, true, true) == false);
	}

	SUBCASE("[InputEvent] Action has default released value") {
		//default Action value is true
		CHECK(event.InputEvent::is_action_released(testEventName) == true);
		CHECK(event.InputEvent::is_action_released(testEventName, true) == true);
	}

	SUBCASE("[InputEvent] Action has default strength value") {
		//default Action value is 0.5f
		CHECK(event.InputEvent::get_action_strength(testEventName) == 0.5f);
		CHECK(event.InputEvent::get_action_strength(testEventName, true) == 0.5f);
	}

	SUBCASE("[InputEvent] Action has default strength value") {
		//default Action value is 0.5f
		CHECK(event.InputEvent::get_action_raw_strength(testEventName) == 0.5f);
		CHECK(event.InputEvent::get_action_raw_strength(testEventName, true) == 0.5f);
	}

	InputMap::get_singleton()->erase_action(testEventName);
}
TEST_CASE_TEMPLATE_INVOKE(test_actions, INPUT_EVENT_TYPES);

} // namespace TestInputEvent

#endif // TEST_INPUT_EVENT_H
