/**************************************************************************/
/*  test_line_edit.h                                                      */
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

#include "core/input/input.h"
#include "scene/gui/line_edit.h"
#include "scene/main/window.h"

#include "tests/test_macros.h"

namespace TestLineEdit {

void assert_extend_left(LineEdit *line_edit) {
	// Select character '5' (columns 4-5), caret at left.
	line_edit->select(4, 5);
	line_edit->set_caret_column(4);

	// Press Shift+Left to extend selection leftward.
	SEND_GUI_KEY_EVENT(Key::LEFT | KeyModifierMask::SHIFT);
	SEND_GUI_KEY_UP_EVENT(Key::LEFT | KeyModifierMask::SHIFT);

	CHECK_MESSAGE(line_edit->get_caret_column() == 3, "caret should move from 4 to 3");
	CHECK(line_edit->get_selected_text() == "45");
}

void assert_shrink_left(LineEdit *line_edit) {
	// Select "234" (columns 1-4), caret at left.
	line_edit->select(1, 4);
	line_edit->set_caret_column(1);

	// Press Shift+Right to shrink selection from left.
	SEND_GUI_KEY_EVENT(Key::RIGHT | KeyModifierMask::SHIFT);
	SEND_GUI_KEY_UP_EVENT(Key::RIGHT | KeyModifierMask::SHIFT);

	CHECK_MESSAGE(line_edit->get_caret_column() == 2, "caret should move from 1 to 2");
	CHECK(line_edit->get_selected_text() == "34");
}

void assert_shrink_right(LineEdit *line_edit) {
	// Select "234" (columns 1-4), caret at right.
	line_edit->select(1, 4);
	line_edit->set_caret_column(4);

	// Press Shift+Left to shrink selection from right.
	SEND_GUI_KEY_EVENT(Key::LEFT | KeyModifierMask::SHIFT);
	SEND_GUI_KEY_UP_EVENT(Key::LEFT | KeyModifierMask::SHIFT);

	CHECK_MESSAGE(line_edit->get_caret_column() == 3, "caret should move from 4 to 3");
	CHECK(line_edit->get_selected_text() == "23");
}

void assert_extend_right(LineEdit *line_edit) {
	// Select character '3' (columns 2-3), caret at right.
	line_edit->select(2, 3);
	line_edit->set_caret_column(3);

	// Press Shift+Right to extend selection rightward.
	SEND_GUI_KEY_EVENT(Key::RIGHT | KeyModifierMask::SHIFT);
	SEND_GUI_KEY_UP_EVENT(Key::RIGHT | KeyModifierMask::SHIFT);

	CHECK_MESSAGE(line_edit->get_caret_column() == 4, "caret should move from 3 to 4");
	CHECK(line_edit->get_selected_text() == "34");
}

void assert_all(LineEdit *line_edit) {
	assert_extend_left(line_edit);
	assert_extend_right(line_edit);
	assert_shrink_left(line_edit);
	assert_shrink_right(line_edit);
}

TEST_CASE("[SceneTree][LineEdit] Selection modification with arrow keys") {
	LineEdit *line_edit = memnew(LineEdit);
	SceneTree::get_singleton()->get_root()->add_child(line_edit);
	line_edit->set_text("12345");
	line_edit->set_caret_mid_grapheme_enabled(true);
	line_edit->grab_focus();

	SUBCASE("Left-to-right") {
		line_edit->set_text_direction(Control::TEXT_DIRECTION_LTR);
		assert_all(line_edit);
	}

	SUBCASE("Right-to-left") {
		line_edit->set_text_direction(Control::TEXT_DIRECTION_RTL);
		assert_all(line_edit);
	}

	memdelete(line_edit);
}

} // namespace TestLineEdit
