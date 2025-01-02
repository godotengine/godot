/**************************************************************************/
/*  test_input.h                                                          */
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

#ifndef TEST_INPUT_H
#define TEST_INPUT_H

#include "tests/test_macros.h"

namespace TestInput {
TEST_CASE("[Input] Correctly removes keys from keyset when modifier pushed first") {
	InputMap *map = memnew(InputMap);
	Input *input = memnew(Input);

	input->set_use_accumulated_input(false);

	// Press "shift"
	// modifier is not set when pressed
	InputEventKey *shift = memnew(InputEventKey());
	shift->set_keycode(Key::SHIFT);
	shift->set_pressed(true);
	input->parse_input_event(shift);

	// Press "_"
	InputEventKey *underscore = memnew(InputEventKey());
	underscore->set_pressed(true);
	underscore->set_shift_pressed(true);
	underscore->set_keycode(Key::UNDERSCORE);
	input->parse_input_event(underscore);
	CHECK(input->is_anything_pressed());

	// Release "shift"
	// Modifier is set when released
	InputEventKey *shift_release = memnew(InputEventKey());
	shift_release->set_keycode(Key::SHIFT);
	shift_release->set_shift_pressed(true);
	shift_release->set_pressed(false);
	input->parse_input_event(shift_release);
	CHECK_FALSE(input->is_anything_pressed());
	memdelete(input);
	memdelete(map);
}

TEST_CASE("[Input] Correctly removes keys from keyset when modifier pushed second") {
	InputMap *map = memnew(InputMap);
	Input *input = memnew(Input);

	input->set_use_accumulated_input(false);

	// Press "-"
	InputEventKey *minus = memnew(InputEventKey());
	minus->set_pressed(true);
	minus->set_keycode(Key::MINUS);
	input->parse_input_event(minus);

	// Press "shift"
	// modifier is not set when pressed
	InputEventKey *shift = memnew(InputEventKey());
	shift->set_keycode(Key::SHIFT);
	shift->set_pressed(true);
	input->parse_input_event(shift);

	// Release "_"
	InputEventKey *underscore = memnew(InputEventKey());
	underscore->set_pressed(false);
	underscore->set_shift_pressed(true);
	underscore->set_keycode(Key::UNDERSCORE);
	input->parse_input_event(underscore);

	CHECK(input->is_anything_pressed());

	// Release "shift"
	// Modifier is set when released
	InputEventKey *shift_release = memnew(InputEventKey());
	shift_release->set_keycode(Key::SHIFT);
	shift_release->set_shift_pressed(true);
	shift_release->set_pressed(false);
	input->parse_input_event(shift_release);
	CHECK_FALSE(input->is_anything_pressed());
	memdelete(input);
	memdelete(map);
}

TEST_CASE("[Input] Correctly removes modifier from keyset") {
	InputMap *map = memnew(InputMap);
	Input *input = memnew(Input);

	input->set_use_accumulated_input(false);
	InputEventKey *shift = memnew(InputEventKey);

	// Press "shift"
	shift->set_keycode(Key::SHIFT);
	shift->set_pressed(true);
	input->parse_input_event(shift);

	CHECK(input->is_anything_pressed());

	// Release "shift"
	InputEventKey *shift_release = memnew(InputEventKey());
	shift_release->set_keycode(Key::SHIFT);
	shift_release->set_shift_pressed(true);
	shift_release->set_pressed(false);
	input->parse_input_event(shift_release);
	CHECK_FALSE(input->is_anything_pressed());
	memdelete(input);
	memdelete(map);
}

} // namespace TestInput

#endif // TEST_INPUT_H
