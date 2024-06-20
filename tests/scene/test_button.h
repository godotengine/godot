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

#ifndef TEST_BUTTON_H
#define TEST_BUTTON_H

#include "core/input/input_event.h"
#include "scene/gui/button.h"
#include "scene/gui/control.h"
#include "tests/test_macros.h"

namespace TestButton {
TEST_CASE("[ScenceTree][Button] Button is_hovered()") {
	// Create a new button instance
	Button *button = memnew(Button);
	CHECK(button != nullptr);

	Control *parent = memnew(Control);
	parent->add_child(button);

	button->set_position(Vector2(10, 10));
	button->set_size(Vector2(100, 50));

	Node *root = memnew(Node);
	root->add_child(parent);

	// Simulate mouse hover
	Ref<InputEventMouseMotion> mouse_motion = InputEventMouseMotion::_new();
	mouse_motion->set_position(Vector2(50, 25)); // Position inside the button
	button->_input(mouse_motion);
	CHECK(button->is_hovered() == true);

	// Mouse not hovered
	mouse_motion->set_position(Vector2(200, 200)); // Position outside the button
	button->_input(mouse_motion);
	CHECK(button->is_hovered() == false);

	memdelete(button);
	memdelete(parent);
	memdelete(root);
}
} //namespace TestButton
#endif // TEST_BUTTON_H
