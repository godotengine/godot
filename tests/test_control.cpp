/**************************************************************************/
/*  test_control.cpp                                                       */
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

#include "scene/gui/control.h"
#include "scene/main/window.h"
#include "core/input/input_event_key.h"
#include "scene/main/scene_tree.h"
#include "tests/test_macros.h"

namespace TestControl {

TEST_CASE("[SceneTree][Control] Container layout and focus behavior") {
	Window *root = SceneTree::get_singleton()->get_root();

	Control *parent = memnew(Control);
	parent->set_size(Size2(300, 200));
	root->add_child(parent);

	Control *child1 = memnew(Control);
	Control *child2 = memnew(Control);
	child1->set_size(Size2(100, 100));
	child2->set_size(Size2(100, 100));

	child1->set_position(Vector2(0, 0));
	child2->set_position(Vector2(0, 120));

	child1->set_focus_mode(Control::FOCUS_ALL);
	child2->set_focus_mode(Control::FOCUS_ALL);

	child1->set_focus_neighbor(MARGIN_BOTTOM, child2);
	child2->set_focus_neighbor(MARGIN_TOP, child1);

	parent->add_child(child1);
	parent->add_child(child2);

	child1->notification(NOTIFICATION_RESIZED);
	child2->notification(NOTIFICATION_RESIZED);

	CHECK_MESSAGE(child1->get_position().is_equal_approx(Vector2(0, 0)), "Child1 position should be (0, 0)");
	CHECK_MESSAGE(child2->get_position().is_equal_approx(Vector2(0, 120)), "Child2 position should be (0, 120)");

	SceneTree::get_singleton()->set_focus_owner(child1);
	CHECK(SceneTree::get_singleton()->get_focus_owner() == child1);

	Ref<InputEventKey> focus_next_event;
	focus_next_event.instantiate();
	focus_next_event->set_pressed(true);
	focus_next_event->set_action("ui_focus_next");

	child1->emit_signal("gui_input", focus_next_event);

	CHECK_MESSAGE(SceneTree::get_singleton()->get_focus_owner() == child2,
		"Focus should move from child1 to child2 using ui_focus_next.");

	memdelete(parent);
}

} // namespace TestControl
