/**************************************************************************/
/*  test_box_container.cpp                                                */
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

TEST_FORCE_LINK(test_box_container)

#include "scene/gui/box_container.h"
#include "scene/gui/control.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"
#include "tests/scene/mock_control.h"

namespace TestBoxContainer {

TEST_CASE("[SceneTree][BoxContainer] HBoxContainer") {
	HBoxContainer *hbox_container = memnew(HBoxContainer);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(hbox_container);

	hbox_container->set_size(Size2(100, 100));
	SceneTree::get_singleton()->process(0);

	SUBCASE("Default behavior") {
		Control *child_control = memnew(Control);
		hbox_container->add_child(child_control);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				hbox_container->get_alignment() == BoxContainer::ALIGNMENT_BEGIN,
				"HBoxContainer alignment is set to ALIGNMENT_BEGIN by default.");

		CHECK_MESSAGE(
				!hbox_container->is_vertical(),
				"HBoxContainer orientation is horizontal.");

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(0, 0)),
				"Child control is aligned to the left in ALIGNMENT_BEGIN mode.");

		CHECK_MESSAGE(
				child_control->get_h_size_flags() == Control::SIZE_FILL,
				"Child control horizontal size flags are set to SIZE_FILL by default.");

		CHECK_MESSAGE(
				child_control->get_v_size_flags() == Control::SIZE_FILL,
				"Child control vertical size flags are set to SIZE_FILL by default.");

		CHECK_MESSAGE(
				child_control->get_size().is_equal_approx(Size2(0, 100)),
				"Child control takes up minimum horizontal space and matches container height by default.");

		memdelete(child_control);
	}

	SUBCASE("Child Horizontal Size Flags") {
		Control *child_control = memnew(Control);
		hbox_container->add_child(child_control);
		child_control->set_custom_minimum_size(Size2(20, 20));
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(0, 0)),
				"Child control is aligned to the left by default with SIZE_FILL.");

		child_control->set_h_size_flags(Control::SIZE_SHRINK_BEGIN);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(0, 0)),
				"Child control is aligned to the left of in SIZE_SHRINK_BEGIN horizontal size flag mode.");

		child_control->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(0, 0)),
				"Child control is aligned to the left of in SIZE_SHRINK_CENTER horizontal size flag mode.");

		child_control->set_h_size_flags(Control::SIZE_SHRINK_END);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(0, 0)),
				"Child control is aligned to the left of in SIZE_SHRINK_END horizontal size flag mode.");

		child_control->set_h_size_flags(Control::SIZE_EXPAND | Control::SIZE_SHRINK_BEGIN);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(0, 0)),
				"Child control is aligned to the left in SIZE_EXPAND | SIZE_SHRINK_BEGIN horizontal size flag mode.");

		child_control->set_h_size_flags(Control::SIZE_EXPAND | Control::SIZE_SHRINK_CENTER);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(40, 0)),
				"Child control is aligned to the center in SIZE_EXPAND | SIZE_SHRINK_CENTER horizontal size flag mode.");

		child_control->set_h_size_flags(Control::SIZE_EXPAND | Control::SIZE_SHRINK_END);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(80, 0)),
				"Child control is aligned to the right in SIZE_EXPAND | SIZE_SHRINK_END horizontal size flag mode.");

		memdelete(child_control);
	}

	SUBCASE("Child Vertical Size Flags") {
		Control *child_control = memnew(Control);
		hbox_container->add_child(child_control);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(0, 0)),
				"Child control is aligned to the top by default with SIZE_FILL.");

		child_control->set_v_size_flags(Control::SIZE_SHRINK_BEGIN);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(0, 0)),
				"Child control is aligned to the top of in SIZE_SHRINK_BEGIN vertical size flag mode.");

		child_control->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(0, 50)),
				"Child control is aligned to the center of in SIZE_SHRINK_CENTER vertical size flag mode.");

		child_control->set_v_size_flags(Control::SIZE_SHRINK_END);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(0, 100)),
				"Child control is aligned to the bottom of in SIZE_SHRINK_END vertical size flag mode.");

		memdelete(child_control);
	}

	SUBCASE("Child Alignment") {
		Control *child_control = memnew(Control);
		hbox_container->add_child(child_control);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(0, 0)),
				"Child control is aligned to the left in ALIGNMENT_BEGIN mode by default.");

		hbox_container->set_alignment(BoxContainer::ALIGNMENT_CENTER);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(50, 0)),
				"Child control is aligned to the center in ALIGNMENT_CENTER mode.");

		hbox_container->set_alignment(BoxContainer::ALIGNMENT_END);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(100, 0)),
				"Child control is aligned to the right in ALIGNMENT_END mode.");

		memdelete(child_control);
	}

	SUBCASE("Child Alignment RTL") {
		Control *child_control = memnew(Control);
		hbox_container->add_child(child_control);
		hbox_container->set_layout_direction(Control::LAYOUT_DIRECTION_RTL);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(100, 0)),
				"Child control is aligned to the right in RTL ALIGNMENT_BEGIN mode by default.");

		hbox_container->set_alignment(BoxContainer::ALIGNMENT_CENTER);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(50, 0)),
				"Child control is aligned to the center in RTL ALIGNMENT_CENTER mode.");

		hbox_container->set_alignment(BoxContainer::ALIGNMENT_END);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(0, 0)),
				"Child control is aligned to the right in RTL ALIGNMENT_END mode.");

		memdelete(child_control);
	}

	SUBCASE("Child Separation") {
		Control *child_control_1 = memnew(Control);
		Control *child_control_2 = memnew(Control);
		hbox_container->add_child(child_control_1);
		hbox_container->add_child(child_control_2);

		hbox_container->add_theme_constant_override("separation", 10);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control_2->get_position().is_equal_approx(Point2(10, 0)),
				"Child controls are separated by the specified separation value.");

		memdelete(child_control_1);
		memdelete(child_control_2);
	}

	SUBCASE("Spacers") {
		Control *child_control_1 = memnew(Control);
		Control *child_control_2 = memnew(Control);
		hbox_container->add_child(child_control_1);
		hbox_container->add_spacer();
		hbox_container->add_child(child_control_2);
		hbox_container->add_theme_constant_override("separation", 0);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control_2->get_position().is_equal_approx(Point2(100, 0)),
				"Spacer pushes child controls apart.");

		memdelete(child_control_1);
		memdelete(child_control_2);
	}

	SUBCASE("Expanding children") {
		Control *child_control_1 = memnew(Control);
		Control *child_control_2 = memnew(Control);
		hbox_container->add_child(child_control_1);
		hbox_container->add_child(child_control_2);
		hbox_container->add_theme_constant_override("separation", 0);

		child_control_1->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control_1->get_size().is_equal_approx(Size2(100, 100)),
				"One child control expands to fill available space.");

		CHECK_MESSAGE(
				child_control_2->get_size().is_equal_approx(Size2(0, 100)),
				"The non-expanding child control remains at minimum size.");

		CHECK_MESSAGE(
				child_control_1->get_position().is_equal_approx(Point2(0, 0)),
				"Expanding child control is aligned to the left.");

		CHECK_MESSAGE(
				child_control_2->get_position().is_equal_approx(Point2(100, 0)),
				"Non-expanding child control is positioned after the expanding child control.");

		child_control_2->set_custom_minimum_size(Size2(20, 0));
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control_1->get_size().is_equal_approx(Size2(80, 100)),
				"Expanding child control shrinks to accommodate the custom minimum size of the other child.");

		CHECK_MESSAGE(
				child_control_2->get_size().is_equal_approx(Size2(20, 100)),
				"The non-expanding child control respects its custom minimum size.");

		CHECK_MESSAGE(
				child_control_1->get_position().is_equal_approx(Point2(0, 0)),
				"Expanding child control is aligned to the left.");

		CHECK_MESSAGE(
				child_control_2->get_position().is_equal_approx(Point2(80, 0)),
				"Non-expanding child control is positioned after the expanding child control.");

		child_control_2->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control_1->get_size().is_equal_approx(Size2(50, 100)),
				"Both child controls expand equally to fill available space.");

		CHECK_MESSAGE(
				child_control_2->get_size().is_equal_approx(Size2(50, 100)),
				"Both child controls expand equally to fill available space.");

		CHECK_MESSAGE(
				child_control_1->get_position().is_equal_approx(Point2(0, 0)),
				"First expanding child control is aligned to the left.");

		CHECK_MESSAGE(
				child_control_2->get_position().is_equal_approx(Point2(50, 0)),
				"Second expanding child control is positioned after the first expanding child control.");

		child_control_2->set_stretch_ratio(3);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control_1->get_size().is_equal_approx(Size2(25, 100)),
				"Child control with lower stretch ratio takes less space.");

		CHECK_MESSAGE(
				child_control_2->get_size().is_equal_approx(Size2(75, 100)),
				"Child control with higher stretch ratio takes more space.");

		CHECK_MESSAGE(
				child_control_1->get_position().is_equal_approx(Point2(0, 0)),
				"First expanding child control is aligned to the left.");

		CHECK_MESSAGE(
				child_control_2->get_position().is_equal_approx(Point2(25, 0)),
				"Second expanding child control is positioned after the first expanding child control.");

		memdelete(child_control_1);
		memdelete(child_control_2);
	}

	SUBCASE("Desired size children") {
		MockControl *child_control_1 = memnew(MockControl);
		MockControl *child_control_2 = memnew(MockControl);
		hbox_container->add_child(child_control_1);
		hbox_container->add_child(child_control_2);

		child_control_1->set_desired_size(Size2(50, 0));
		child_control_2->set_desired_size(Size2(50, 0));

		// Set a custom maximum size to ensure the container does not expand to accommodate children in "a + b > c" case at first.
		// It is also necessary to set max size propagation to false to prevent children's bound desired sizes from being capped in "a + b > c" case.
		hbox_container->set_size(Size2());
		hbox_container->set_custom_maximum_size(Size2(100, 100));
		hbox_container->set_propagate_maximum_size(false);
		hbox_container->add_theme_constant_override("separation", 0);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				hbox_container->get_size().is_equal_approx(Size2(100, 0)),
				"Container expands to accommodate children up to its maximum size. Case: a + b = c");

		CHECK_MESSAGE(
				(child_control_1->get_size().is_equal_approx(Size2(50, 0)) && child_control_2->get_size().is_equal_approx(Size2(50, 0))),
				"Children control with equal desired sizes take equal space. Case: a + b = c");

		hbox_container->set_size(Size2());
		child_control_1->set_desired_size(Size2(200, 0));
		child_control_2->set_desired_size(Size2(200, 0));
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				hbox_container->get_size().is_equal_approx(Size2(100, 0)),
				"Container expands to accommodate children up to its maximum size. Case: a + b > c");

		CHECK_MESSAGE(
				(child_control_1->get_size().is_equal_approx(Size2(50, 0)) && child_control_2->get_size().is_equal_approx(Size2(50, 0))),
				"Children control with equal desired sizes take equal space. Case: a + b > c");

		hbox_container->set_size(Size2());
		child_control_1->set_desired_size(Size2(10, 0));
		child_control_2->set_desired_size(Size2(10, 0));
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				hbox_container->get_size().is_equal_approx(Size2(20, 0)),
				"Container only expands as much as needed to accommodate children. Case: a + b < c");

		CHECK_MESSAGE(
				(child_control_1->get_size().is_equal_approx(Size2(10, 0)) && child_control_2->get_size().is_equal_approx(Size2(10, 0))),
				"Children control with equal desired sizes take equal space. Case: a + b < c");

		hbox_container->set_size(Size2());
		child_control_1->set_desired_size(Size2(75, 0));
		child_control_2->set_desired_size(Size2(25, 0));
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				hbox_container->get_size().is_equal_approx(Size2(100, 0)),
				"Container expands to accommodate children up to its maximum size. Case: a + b = c");

		CHECK_MESSAGE(
				(child_control_1->get_size().is_equal_approx(Size2(75, 0)) && child_control_2->get_size().is_equal_approx(Size2(25, 0))),
				"Children control with different desired sizes take different space. Case: a + b = c");

		hbox_container->set_size(Size2());
		child_control_1->set_desired_size(Size2(300, 0));
		child_control_2->set_desired_size(Size2(100, 0));
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				hbox_container->get_size().is_equal_approx(Size2(100, 0)),
				"Container expands to accommodate children up to its maximum size. Case: a + b > c");

		CHECK_MESSAGE(
				(child_control_1->get_size().is_equal_approx(Size2(75, 0)) && child_control_2->get_size().is_equal_approx(Size2(25, 0))),
				"Children control with different desired sizes take different space. Case: a + b > c");

		hbox_container->set_size(Size2());
		child_control_1->set_desired_size(Size2(40, 0));
		child_control_2->set_desired_size(Size2(20, 0));
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				hbox_container->get_size().is_equal_approx(Size2(60, 0)),
				"Container only expands as much as needed to accommodate children. Case: a + b < c");

		CHECK_MESSAGE(
				(child_control_1->get_size().is_equal_approx(Size2(40, 0)) && child_control_2->get_size().is_equal_approx(Size2(20, 0))),
				"Children control with different desired sizes take different space. Case: a + b < c");

		memdelete(child_control_1);
		memdelete(child_control_2);
	}

	SUBCASE("Desired size takes priority over expanding children") {
		MockControl *child_control_1 = memnew(MockControl);
		MockControl *child_control_2 = memnew(MockControl);
		Control *child_control_3 = memnew(Control);
		hbox_container->add_child(child_control_1);
		hbox_container->add_child(child_control_2);
		hbox_container->add_child(child_control_3);

		child_control_1->set_desired_size(Size2(50, 0));
		child_control_2->set_desired_size(Size2(50, 0));
		child_control_3->set_h_size_flags(Control::SIZE_EXPAND_FILL);

		hbox_container->set_size(Size2());
		hbox_container->set_custom_maximum_size(Size2(100, 100));
		hbox_container->set_propagate_maximum_size(false);
		hbox_container->add_theme_constant_override("separation", 0);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				hbox_container->get_size().is_equal_approx(Size2(100, 0)),
				"Container desired sizes take priority over expanding children.");
		CHECK_MESSAGE(
				(child_control_1->get_size().is_equal_approx(Size2(50, 0)) &&
						child_control_2->get_size().is_equal_approx(Size2(50, 0)) &&
						child_control_3->get_size().is_equal_approx(Size2(0, 0))),
				"Expanding children do not take space from children with desired sizes.");

		memdelete(child_control_3);
		memdelete(child_control_2);
		memdelete(child_control_1);
	}

	memdelete(hbox_container);
}

TEST_CASE("[SceneTree][BoxContainer] VBoxContainer") {
	VBoxContainer *vbox_container = memnew(VBoxContainer);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(vbox_container);

	vbox_container->set_size(Size2(100, 100));
	SceneTree::get_singleton()->process(0);

	SUBCASE("Default behavior") {
		Control *child_control = memnew(Control);
		vbox_container->add_child(child_control);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				vbox_container->get_alignment() == BoxContainer::ALIGNMENT_BEGIN,
				"HBoxContainer alignment is set to ALIGNMENT_BEGIN by default.");

		CHECK_MESSAGE(
				vbox_container->is_vertical(),
				"VBoxContainer orientation is vertical.");

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(0, 0)),
				"Child control is aligned to the left in ALIGNMENT_BEGIN mode.");

		CHECK_MESSAGE(
				child_control->get_h_size_flags() == Control::SIZE_FILL,
				"Child control horizontal size flags are set to SIZE_FILL by default.");

		CHECK_MESSAGE(
				child_control->get_v_size_flags() == Control::SIZE_FILL,
				"Child control vertical size flags are set to SIZE_FILL by default.");

		CHECK_MESSAGE(
				child_control->get_size().is_equal_approx(Size2(100, 0)),
				"Child control takes up minimum vertical space and matches container width by default.");

		memdelete(child_control);
	}

	SUBCASE("Child Size Flags") {
		Control *child_control = memnew(Control);
		vbox_container->add_child(child_control);
		child_control->set_custom_minimum_size(Size2(20, 20));
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(0, 0)),
				"Child control is aligned to the top by default with SIZE_FILL.");

		child_control->set_v_size_flags(Control::SIZE_SHRINK_BEGIN);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(0, 0)),
				"Child control is aligned to the top of in SIZE_SHRINK_BEGIN vertical size flag mode.");

		child_control->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(0, 0)),
				"Child control is aligned to the top of in SIZE_SHRINK_CENTER vertical size flag mode.");

		child_control->set_v_size_flags(Control::SIZE_SHRINK_END);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(0, 0)),
				"Child control is aligned to the top of in SIZE_SHRINK_END vertical size flag mode.");

		child_control->set_v_size_flags(Control::SIZE_EXPAND | Control::SIZE_SHRINK_BEGIN);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(0, 0)),
				"Child control is aligned to the top in SIZE_EXPAND | SIZE_SHRINK_BEGIN vertical size flag mode.");

		child_control->set_v_size_flags(Control::SIZE_EXPAND | Control::SIZE_SHRINK_CENTER);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(0, 40)),
				"Child control is aligned to the center in SIZE_EXPAND | SIZE_SHRINK_CENTER vertical size flag mode.");

		child_control->set_v_size_flags(Control::SIZE_EXPAND | Control::SIZE_SHRINK_END);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(0, 80)),
				"Child control is aligned to the bottom in SIZE_EXPAND | SIZE_SHRINK_END vertical size flag mode.");

		memdelete(child_control);
	}

	SUBCASE("Child Horizontal Size Flags") {
		Control *child_control = memnew(Control);
		vbox_container->add_child(child_control);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(0, 0)),
				"Child control is aligned to the left by default with SIZE_FILL.");

		child_control->set_h_size_flags(Control::SIZE_SHRINK_BEGIN);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(0, 0)),
				"Child control is aligned to the left of in SIZE_SHRINK_BEGIN horizontal size flag mode.");

		child_control->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(50, 0)),
				"Child control is aligned to the center of in SIZE_SHRINK_CENTER horizontal size flag mode.");

		child_control->set_h_size_flags(Control::SIZE_SHRINK_END);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(100, 0)),
				"Child control is aligned to the right of in SIZE_SHRINK_END horizontal size flag mode.");

		memdelete(child_control);
	}

	SUBCASE("Child Alignment") {
		Control *child_control = memnew(Control);
		vbox_container->add_child(child_control);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(0, 0)),
				"Child control is aligned to the top in ALIGNMENT_BEGIN mode by default.");

		vbox_container->set_alignment(BoxContainer::ALIGNMENT_CENTER);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(0, 50)),
				"Child control is aligned to the center in ALIGNMENT_CENTER mode.");

		vbox_container->set_alignment(BoxContainer::ALIGNMENT_END);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(0, 100)),
				"Child control is aligned to the bottom in ALIGNMENT_END mode.");

		memdelete(child_control);
	}

	SUBCASE("Child Separation") {
		Control *child_control_1 = memnew(Control);
		Control *child_control_2 = memnew(Control);
		vbox_container->add_child(child_control_1);
		vbox_container->add_child(child_control_2);

		vbox_container->add_theme_constant_override("separation", 10);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control_2->get_position().is_equal_approx(Point2(0, 10)),
				"Child controls are separated by the specified separation value.");

		memdelete(child_control_1);
		memdelete(child_control_2);
	}

	SUBCASE("Spacers") {
		Control *child_control_1 = memnew(Control);
		Control *child_control_2 = memnew(Control);
		vbox_container->add_child(child_control_1);
		vbox_container->add_spacer();
		vbox_container->add_child(child_control_2);
		vbox_container->add_theme_constant_override("separation", 0);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control_2->get_position().is_equal_approx(Point2(0, 100)),
				"Spacer pushes child controls apart.");

		memdelete(child_control_1);
		memdelete(child_control_2);
	}

	SUBCASE("Expanding children") {
		Control *child_control_1 = memnew(Control);
		Control *child_control_2 = memnew(Control);
		vbox_container->add_child(child_control_1);
		vbox_container->add_child(child_control_2);
		vbox_container->add_theme_constant_override("separation", 0);

		child_control_1->set_v_size_flags(Control::SIZE_EXPAND_FILL);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control_1->get_size().is_equal_approx(Size2(100, 100)),
				"One child control expands to fill available space.");

		CHECK_MESSAGE(
				child_control_2->get_size().is_equal_approx(Size2(100, 0)),
				"The non-expanding child control remains at minimum size.");

		CHECK_MESSAGE(
				child_control_1->get_position().is_equal_approx(Point2(0, 0)),
				"Expanding child control is aligned to the left.");

		CHECK_MESSAGE(
				child_control_2->get_position().is_equal_approx(Point2(0, 100)),
				"Non-expanding child control is positioned after the expanding child control.");

		child_control_2->set_custom_minimum_size(Size2(0, 20));
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control_1->get_size().is_equal_approx(Size2(100, 80)),
				"Expanding child control shrinks to accommodate the custom minimum size of the other child.");

		CHECK_MESSAGE(
				child_control_2->get_size().is_equal_approx(Size2(100, 20)),
				"The non-expanding child control respects its custom minimum size.");

		CHECK_MESSAGE(
				child_control_1->get_position().is_equal_approx(Point2(0, 0)),
				"Expanding child control is aligned to the left.");

		CHECK_MESSAGE(
				child_control_2->get_position().is_equal_approx(Point2(0, 80)),
				"Non-expanding child control is positioned after the expanding child control.");

		child_control_2->set_v_size_flags(Control::SIZE_EXPAND_FILL);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control_1->get_size().is_equal_approx(Size2(100, 50)),
				"Both child controls expand equally to fill available space.");

		CHECK_MESSAGE(
				child_control_2->get_size().is_equal_approx(Size2(100, 50)),
				"Both child controls expand equally to fill available space.");

		CHECK_MESSAGE(
				child_control_1->get_position().is_equal_approx(Point2(0, 0)),
				"First expanding child control is aligned to the left.");

		CHECK_MESSAGE(
				child_control_2->get_position().is_equal_approx(Point2(0, 50)),
				"Second expanding child control is positioned after the first expanding child control.");

		child_control_2->set_stretch_ratio(3);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control_1->get_size().is_equal_approx(Size2(100, 25)),
				"Child control with lower stretch ratio takes less space.");

		CHECK_MESSAGE(
				child_control_2->get_size().is_equal_approx(Size2(100, 75)),
				"Child control with higher stretch ratio takes more space.");

		CHECK_MESSAGE(
				child_control_1->get_position().is_equal_approx(Point2(0, 0)),
				"First expanding child control is aligned to the left.");

		CHECK_MESSAGE(
				child_control_2->get_position().is_equal_approx(Point2(0, 25)),
				"Second expanding child control is positioned after the first expanding child control.");

		memdelete(child_control_1);
		memdelete(child_control_2);
	}

	SUBCASE("Desired size children") {
		MockControl *child_control_1 = memnew(MockControl);
		MockControl *child_control_2 = memnew(MockControl);
		vbox_container->add_child(child_control_1);
		vbox_container->add_child(child_control_2);

		child_control_1->set_desired_size(Size2(0, 50));
		child_control_2->set_desired_size(Size2(0, 50));

		// Set a custom maximum size to ensure the container does not expand to accommodate children in "a + b > c" case at first.
		// It is also necessary to set max size propagation to false to prevent children's bound desired sizes from being capped in "a + b > c" case.
		vbox_container->set_size(Size2());
		vbox_container->set_custom_maximum_size(Size2(100, 100));
		vbox_container->set_propagate_maximum_size(false);
		vbox_container->add_theme_constant_override("separation", 0);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				vbox_container->get_size().is_equal_approx(Size2(0, 100)),
				"Container expands to accommodate children up to its maximum size. Case: a + b = c");

		CHECK_MESSAGE(
				(child_control_1->get_size().is_equal_approx(Size2(0, 50)) && child_control_2->get_size().is_equal_approx(Size2(0, 50))),
				"Children control with equal desired sizes take equal space. Case: a + b = c");

		vbox_container->set_size(Size2());
		child_control_1->set_desired_size(Size2(0, 200));
		child_control_2->set_desired_size(Size2(0, 200));
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				vbox_container->get_size().is_equal_approx(Size2(0, 100)),
				"Container expands to accommodate children up to its maximum size. Case: a + b > c");

		CHECK_MESSAGE(
				(child_control_1->get_size().is_equal_approx(Size2(0, 50)) && child_control_2->get_size().is_equal_approx(Size2(0, 50))),
				"Children control with equal desired sizes take equal space. Case: a + b > c");

		vbox_container->set_size(Size2());
		child_control_1->set_desired_size(Size2(0, 10));
		child_control_2->set_desired_size(Size2(0, 10));
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				vbox_container->get_size().is_equal_approx(Size2(0, 20)),
				"Container only expands as much as needed to accommodate children. Case: a + b < c");

		CHECK_MESSAGE(
				(child_control_1->get_size().is_equal_approx(Size2(0, 10)) && child_control_2->get_size().is_equal_approx(Size2(0, 10))),
				"Children control with equal desired sizes take equal space. Case: a + b < c");

		vbox_container->set_size(Size2());
		child_control_1->set_desired_size(Size2(0, 75));
		child_control_2->set_desired_size(Size2(0, 25));
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				vbox_container->get_size().is_equal_approx(Size2(0, 100)),
				"Container expands to accommodate children up to its maximum size. Case: a + b = c");

		CHECK_MESSAGE(
				(child_control_1->get_size().is_equal_approx(Size2(0, 75)) && child_control_2->get_size().is_equal_approx(Size2(0, 25))),
				"Children control with different desired sizes take different space. Case: a + b = c");

		vbox_container->set_size(Size2());
		child_control_1->set_desired_size(Size2(0, 300));
		child_control_2->set_desired_size(Size2(0, 100));
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				vbox_container->get_size().is_equal_approx(Size2(0, 100)),
				"Container expands to accommodate children up to its maximum size. Case: a + b > c");

		CHECK_MESSAGE(
				(child_control_1->get_size().is_equal_approx(Size2(0, 75)) && child_control_2->get_size().is_equal_approx(Size2(0, 25))),
				"Children control with different desired sizes take different space. Case: a + b > c");

		vbox_container->set_size(Size2());
		child_control_1->set_desired_size(Size2(0, 40));
		child_control_2->set_desired_size(Size2(0, 20));
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				vbox_container->get_size().is_equal_approx(Size2(0, 60)),
				"Container only expands as much as needed to accommodate children. Case: a + b < c");

		CHECK_MESSAGE(
				(child_control_1->get_size().is_equal_approx(Size2(0, 40)) && child_control_2->get_size().is_equal_approx(Size2(0, 20))),
				"Children control with different desired sizes take different space. Case: a + b < c");

		memdelete(child_control_1);
		memdelete(child_control_2);
	}

	SUBCASE("Desired size takes priority over expanding children") {
		MockControl *child_control_1 = memnew(MockControl);
		MockControl *child_control_2 = memnew(MockControl);
		Control *child_control_3 = memnew(Control);
		vbox_container->add_child(child_control_1);
		vbox_container->add_child(child_control_2);
		vbox_container->add_child(child_control_3);

		child_control_1->set_desired_size(Size2(0, 50));
		child_control_2->set_desired_size(Size2(0, 50));
		child_control_3->set_v_size_flags(Control::SIZE_EXPAND_FILL);

		vbox_container->set_size(Size2());
		vbox_container->set_custom_maximum_size(Size2(100, 100));
		vbox_container->set_propagate_maximum_size(false);
		vbox_container->add_theme_constant_override("separation", 0);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				vbox_container->get_size().is_equal_approx(Size2(0, 100)),
				"Container desired sizes take priority over expanding children.");
		CHECK_MESSAGE(
				(child_control_1->get_size().is_equal_approx(Size2(0, 50)) &&
						child_control_2->get_size().is_equal_approx(Size2(0, 50)) &&
						child_control_3->get_size().is_equal_approx(Size2(0, 0))),
				"Expanding children do not take space from children with desired sizes.");

		memdelete(child_control_3);
		memdelete(child_control_2);
		memdelete(child_control_1);
	}

	memdelete(vbox_container);
}

} // namespace TestBoxContainer
