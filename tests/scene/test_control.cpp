/**************************************************************************/
/*  test_control.cpp                                                      */
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
#include "scene/main/scene_tree.h"
#include "tests/test_macros.h"

namespace TestControl {

TEST_CASE("[SceneTree][Control] Layout and Focus Behavior") {
	Window *root = SceneTree::get_singleton()->get_root();

	// Create parent control
	Control *parent = memnew(Control);
	parent->set_size(Size2(400, 400));
	root->add_child(parent);

	// Create child control
	Control *child = memnew(Control);
	parent->add_child(child);

	// Set anchors
	child->set_anchor(SIDE_LEFT, 0.25);
	child->set_anchor(SIDE_TOP, 0.25);
	child->set_anchor(SIDE_RIGHT, 0.75);
	child->set_anchor(SIDE_BOTTOM, 0.75);

	// Process scene to apply layout
	SceneTree::get_singleton()->process(0);

	// Check position and size
	CHECK_MESSAGE(
		child->get_position().is_equal_approx(Vector2(100, 100)),
		"Child should be positioned at 25% of parent size.");
	CHECK_MESSAGE(
		child->get_size().is_equal_approx(Vector2(200, 200)),
		"Child should be sized to 50% of parent width and height.");

	// Set focus neighbors
	child->set_focus_neighbour(SIDE_LEFT, NodePath("LeftControl"));
	child->set_focus_neighbour(SIDE_RIGHT, NodePath("RightControl"));

	// Check focus neighbors
	CHECK_MESSAGE(
		child->get_focus_neighbour(SIDE_LEFT) == NodePath("LeftControl"),
		"Left focus neighbor should be correctly set.");
	CHECK_MESSAGE(
		child->get_focus_neighbour(SIDE_RIGHT) == NodePath("RightControl"),
		"Right focus neighbor should be correctly set.");

	// Cleanup
	memdelete(child);
	memdelete(parent);
}

} // namespace TestControl