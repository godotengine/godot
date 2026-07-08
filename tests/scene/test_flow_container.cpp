/**************************************************************************/
/*  test_flow_container.cpp                                               */
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

TEST_FORCE_LINK(test_flow_container)

#include "scene/gui/control.h"
#include "scene/gui/flow_container.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"

namespace TestFlowContainer {

TEST_CASE("[SceneTree][FlowContainer] HFlowContainer") {
	HFlowContainer *hflow_container = memnew(HFlowContainer);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(hflow_container);

	hflow_container->add_theme_constant_override("h_separation", 0);
	hflow_container->add_theme_constant_override("v_separation", 0);
	hflow_container->set_size(Size2(100, 100));
	SceneTree::get_singleton()->process(0);

	SUBCASE("Default behavior") {
		Control *child_control_1 = memnew(Control);
		Control *child_control_2 = memnew(Control);
		Control *child_control_3 = memnew(Control);
		child_control_1->set_custom_minimum_size(Size2(20, 10));
		child_control_2->set_custom_minimum_size(Size2(20, 10));
		child_control_3->set_custom_minimum_size(Size2(20, 10));
		hflow_container->add_child(child_control_1);
		hflow_container->add_child(child_control_2);
		hflow_container->add_child(child_control_3);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				hflow_container->get_alignment() == FlowContainer::ALIGNMENT_BEGIN,
				"HFlowContainer alignment is set to ALIGNMENT_BEGIN by default.");

		CHECK_MESSAGE(
				hflow_container->get_last_wrap_alignment() == FlowContainer::LAST_WRAP_ALIGNMENT_INHERIT,
				"HFlowContainer last wrap alignment is set to LAST_WRAP_ALIGNMENT_INHERIT by default.");

		CHECK_MESSAGE(
				!hflow_container->is_vertical(),
				"HFlowContainer is horizontal.");

		CHECK_MESSAGE(
				!hflow_container->is_reverse_fill(),
				"HFlowContainer reverse fill is disabled by default.");

		CHECK_MESSAGE(
				hflow_container->get_line_count() == 1,
				"HFlowContainer keeps all children in one line when there is enough horizontal space.");

		CHECK_MESSAGE(
				hflow_container->get_line_max_child_count() == 3,
				"HFlowContainer reports the child count of the first line.");

		CHECK_MESSAGE(
				child_control_1->get_position().is_equal_approx(Point2(0, 0)),
				"First child control is positioned at the beginning of the line.");

		CHECK_MESSAGE(
				child_control_2->get_position().is_equal_approx(Point2(20, 0)),
				"Second child control is placed after the first child control.");

		CHECK_MESSAGE(
				child_control_3->get_position().is_equal_approx(Point2(40, 0)),
				"Third child control is placed after the second child control.");

		memdelete(child_control_3);
		memdelete(child_control_2);
		memdelete(child_control_1);
	}

	SUBCASE("Wrapping and last wrap alignment") {
		Control *child_control_1 = memnew(Control);
		Control *child_control_2 = memnew(Control);
		Control *child_control_3 = memnew(Control);
		child_control_1->set_custom_minimum_size(Size2(20, 10));
		child_control_2->set_custom_minimum_size(Size2(20, 10));
		child_control_3->set_custom_minimum_size(Size2(20, 10));
		hflow_container->set_size(Size2(50, 100));
		hflow_container->add_child(child_control_1);
		hflow_container->add_child(child_control_2);
		hflow_container->add_child(child_control_3);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				hflow_container->get_line_count() == 2,
				"HFlowContainer wraps children to a second line when they no longer fit.");

		CHECK_MESSAGE(
				hflow_container->get_line_max_child_count() == 2,
				"HFlowContainer reports the first line child count as the line max child count.");

		CHECK_MESSAGE(
				child_control_1->get_position().is_equal_approx(Point2(0, 0)),
				"First child control starts at the first line origin.");

		CHECK_MESSAGE(
				child_control_2->get_position().is_equal_approx(Point2(20, 0)),
				"Second child control remains on the first line.");

		CHECK_MESSAGE(
				child_control_3->get_position().is_equal_approx(Point2(0, 10)),
				"Third child control wraps to the second line.");

		hflow_container->set_last_wrap_alignment(FlowContainer::LAST_WRAP_ALIGNMENT_CENTER);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control_3->get_position().is_equal_approx(Point2(10, 10)),
				"Last wrapped line is centered relative to the previous line when LAST_WRAP_ALIGNMENT_CENTER is used.");

		hflow_container->set_last_wrap_alignment(FlowContainer::LAST_WRAP_ALIGNMENT_END);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control_3->get_position().is_equal_approx(Point2(20, 10)),
				"Last wrapped line is aligned to the end relative to the previous line when LAST_WRAP_ALIGNMENT_END is used.");

		memdelete(child_control_3);
		memdelete(child_control_2);
		memdelete(child_control_1);
	}

	SUBCASE("Alignment and RTL") {
		Control *child_control_1 = memnew(Control);
		Control *child_control_2 = memnew(Control);
		child_control_1->set_custom_minimum_size(Size2(20, 10));
		child_control_2->set_custom_minimum_size(Size2(20, 10));
		hflow_container->set_size(Size2(100, 100));
		hflow_container->add_child(child_control_1);
		hflow_container->add_child(child_control_2);
		SceneTree::get_singleton()->process(0);

		hflow_container->set_alignment(FlowContainer::ALIGNMENT_CENTER);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control_1->get_position().is_equal_approx(Point2(30, 0)),
				"First child control is centered when ALIGNMENT_CENTER is used.");

		CHECK_MESSAGE(
				child_control_2->get_position().is_equal_approx(Point2(50, 0)),
				"Second child control follows centered alignment when ALIGNMENT_CENTER is used.");

		hflow_container->set_alignment(FlowContainer::ALIGNMENT_END);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control_1->get_position().is_equal_approx(Point2(60, 0)),
				"First child control is aligned to the end when ALIGNMENT_END is used.");

		CHECK_MESSAGE(
				child_control_2->get_position().is_equal_approx(Point2(80, 0)),
				"Second child control is aligned to the end when ALIGNMENT_END is used.");

		hflow_container->set_alignment(FlowContainer::ALIGNMENT_BEGIN);
		hflow_container->set_layout_direction(Control::LAYOUT_DIRECTION_RTL);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control_1->get_position().is_equal_approx(Point2(80, 0)),
				"First child control starts at the right edge in RTL mode with ALIGNMENT_BEGIN.");

		CHECK_MESSAGE(
				child_control_2->get_position().is_equal_approx(Point2(60, 0)),
				"Second child control follows right-to-left ordering in RTL mode with ALIGNMENT_BEGIN.");

		memdelete(child_control_2);
		memdelete(child_control_1);
	}

	SUBCASE("Expanding children and stretch ratio") {
		Control *child_control_1 = memnew(Control);
		Control *child_control_2 = memnew(Control);
		child_control_1->set_custom_minimum_size(Size2(20, 10));
		child_control_2->set_custom_minimum_size(Size2(20, 10));
		child_control_1->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		child_control_2->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		hflow_container->set_size(Size2(100, 100));
		hflow_container->add_child(child_control_1);
		hflow_container->add_child(child_control_2);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control_1->get_size().is_equal_approx(Size2(50, 10)),
				"First child control expands to half of the available width when both children have equal stretch ratio.");

		CHECK_MESSAGE(
				child_control_2->get_size().is_equal_approx(Size2(50, 10)),
				"Second child control expands to half of the available width when both children have equal stretch ratio.");

		CHECK_MESSAGE(
				child_control_2->get_position().is_equal_approx(Point2(50, 0)),
				"Second child control is placed immediately after the first expanded child control.");

		child_control_2->set_stretch_ratio(3.0f);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control_1->get_size().is_equal_approx(Size2(35, 10)),
				"First child control receives less width when its stretch ratio is lower.");

		CHECK_MESSAGE(
				child_control_2->get_size().is_equal_approx(Size2(65, 10)),
				"Second child control receives more width when its stretch ratio is higher.");

		CHECK_MESSAGE(
				child_control_2->get_position().is_equal_approx(Point2(35, 0)),
				"Second child control starts after the first child control's stretched width.");

		memdelete(child_control_2);
		memdelete(child_control_1);
	}

	SUBCASE("Reverse fill") {
		Control *child_control_1 = memnew(Control);
		Control *child_control_2 = memnew(Control);
		Control *child_control_3 = memnew(Control);
		child_control_1->set_custom_minimum_size(Size2(20, 10));
		child_control_2->set_custom_minimum_size(Size2(20, 10));
		child_control_3->set_custom_minimum_size(Size2(20, 10));
		hflow_container->set_size(Size2(50, 100));
		hflow_container->add_child(child_control_1);
		hflow_container->add_child(child_control_2);
		hflow_container->add_child(child_control_3);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control_1->get_position().is_equal_approx(Point2(0, 0)),
				"First child control starts at the top row by default.");

		CHECK_MESSAGE(
				child_control_3->get_position().is_equal_approx(Point2(0, 10)),
				"Third child control wraps to the next row by default.");

		hflow_container->set_reverse_fill(true);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control_1->get_position().is_equal_approx(Point2(0, 90)),
				"First child control moves to the bottom row when reverse fill is enabled.");

		CHECK_MESSAGE(
				child_control_2->get_position().is_equal_approx(Point2(20, 90)),
				"Second child control remains in the same row as the first child control when reverse fill is enabled.");

		CHECK_MESSAGE(
				child_control_3->get_position().is_equal_approx(Point2(0, 80)),
				"Wrapped row order is inverted from bottom to top when reverse fill is enabled.");

		hflow_container->set_layout_direction(Control::LAYOUT_DIRECTION_RTL);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control_1->get_position().is_equal_approx(Point2(30, 90)),
				"RTL mirrors horizontal placement while reverse fill keeps rows bottom to top.");

		CHECK_MESSAGE(
				child_control_2->get_position().is_equal_approx(Point2(10, 90)),
				"Second child control follows mirrored RTL order while reverse fill remains enabled.");

		CHECK_MESSAGE(
				child_control_3->get_position().is_equal_approx(Point2(30, 80)),
				"Wrapped child control is mirrored in RTL while keeping reverse-filled row order.");

		memdelete(child_control_3);
		memdelete(child_control_2);
		memdelete(child_control_1);
	}

	memdelete(hflow_container);
}

TEST_CASE("[SceneTree][FlowContainer] VFlowContainer") {
	VFlowContainer *vflow_container = memnew(VFlowContainer);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(vflow_container);

	vflow_container->add_theme_constant_override("h_separation", 0);
	vflow_container->add_theme_constant_override("v_separation", 0);
	vflow_container->set_size(Size2(100, 100));
	SceneTree::get_singleton()->process(0);

	SUBCASE("Default behavior") {
		Control *child_control_1 = memnew(Control);
		Control *child_control_2 = memnew(Control);
		Control *child_control_3 = memnew(Control);
		child_control_1->set_custom_minimum_size(Size2(20, 10));
		child_control_2->set_custom_minimum_size(Size2(20, 10));
		child_control_3->set_custom_minimum_size(Size2(20, 10));
		vflow_container->add_child(child_control_1);
		vflow_container->add_child(child_control_2);
		vflow_container->add_child(child_control_3);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				vflow_container->get_alignment() == FlowContainer::ALIGNMENT_BEGIN,
				"VFlowContainer alignment is set to ALIGNMENT_BEGIN by default.");

		CHECK_MESSAGE(
				vflow_container->get_last_wrap_alignment() == FlowContainer::LAST_WRAP_ALIGNMENT_INHERIT,
				"VFlowContainer last wrap alignment is set to LAST_WRAP_ALIGNMENT_INHERIT by default.");

		CHECK_MESSAGE(
				vflow_container->is_vertical(),
				"VFlowContainer is vertical.");

		CHECK_MESSAGE(
				!vflow_container->is_reverse_fill(),
				"VFlowContainer reverse fill is disabled by default.");

		CHECK_MESSAGE(
				vflow_container->get_line_count() == 1,
				"VFlowContainer keeps all children in one column when there is enough vertical space.");

		CHECK_MESSAGE(
				vflow_container->get_line_max_child_count() == 3,
				"VFlowContainer reports the child count of the first column.");

		CHECK_MESSAGE(
				child_control_1->get_position().is_equal_approx(Point2(0, 0)),
				"First child control is positioned at the top of the first column.");

		CHECK_MESSAGE(
				child_control_2->get_position().is_equal_approx(Point2(0, 10)),
				"Second child control is placed below the first child control.");

		CHECK_MESSAGE(
				child_control_3->get_position().is_equal_approx(Point2(0, 20)),
				"Third child control is placed below the second child control.");

		memdelete(child_control_3);
		memdelete(child_control_2);
		memdelete(child_control_1);
	}

	SUBCASE("Wrapping and last wrap alignment") {
		Control *child_control_1 = memnew(Control);
		Control *child_control_2 = memnew(Control);
		Control *child_control_3 = memnew(Control);
		child_control_1->set_custom_minimum_size(Size2(20, 20));
		child_control_2->set_custom_minimum_size(Size2(20, 20));
		child_control_3->set_custom_minimum_size(Size2(20, 20));
		vflow_container->set_size(Size2(100, 50));
		vflow_container->add_child(child_control_1);
		vflow_container->add_child(child_control_2);
		vflow_container->add_child(child_control_3);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				vflow_container->get_line_count() == 2,
				"VFlowContainer wraps children to a second column when they no longer fit vertically.");

		CHECK_MESSAGE(
				vflow_container->get_line_max_child_count() == 2,
				"VFlowContainer reports the first column child count as the line max child count.");

		CHECK_MESSAGE(
				child_control_1->get_position().is_equal_approx(Point2(0, 0)),
				"First child control starts at the first column origin.");

		CHECK_MESSAGE(
				child_control_2->get_position().is_equal_approx(Point2(0, 20)),
				"Second child control remains in the first column.");

		CHECK_MESSAGE(
				child_control_3->get_position().is_equal_approx(Point2(20, 0)),
				"Third child control wraps to the second column.");

		vflow_container->set_last_wrap_alignment(FlowContainer::LAST_WRAP_ALIGNMENT_CENTER);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control_3->get_position().is_equal_approx(Point2(20, 10)),
				"Last wrapped column is centered relative to the previous column when LAST_WRAP_ALIGNMENT_CENTER is used.");

		vflow_container->set_last_wrap_alignment(FlowContainer::LAST_WRAP_ALIGNMENT_END);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control_3->get_position().is_equal_approx(Point2(20, 20)),
				"Last wrapped column is aligned to the end relative to the previous column when LAST_WRAP_ALIGNMENT_END is used.");

		memdelete(child_control_3);
		memdelete(child_control_2);
		memdelete(child_control_1);
	}

	SUBCASE("Alignment and RTL") {
		Control *child_control_1 = memnew(Control);
		Control *child_control_2 = memnew(Control);
		child_control_1->set_custom_minimum_size(Size2(20, 20));
		child_control_2->set_custom_minimum_size(Size2(20, 20));
		vflow_container->set_size(Size2(100, 100));
		vflow_container->add_child(child_control_1);
		vflow_container->add_child(child_control_2);
		SceneTree::get_singleton()->process(0);

		vflow_container->set_alignment(FlowContainer::ALIGNMENT_CENTER);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control_1->get_position().is_equal_approx(Point2(0, 30)),
				"First child control is vertically centered when ALIGNMENT_CENTER is used.");

		CHECK_MESSAGE(
				child_control_2->get_position().is_equal_approx(Point2(0, 50)),
				"Second child control follows centered vertical alignment when ALIGNMENT_CENTER is used.");

		vflow_container->set_alignment(FlowContainer::ALIGNMENT_END);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control_1->get_position().is_equal_approx(Point2(0, 60)),
				"First child control is aligned to the bottom when ALIGNMENT_END is used.");

		CHECK_MESSAGE(
				child_control_2->get_position().is_equal_approx(Point2(0, 80)),
				"Second child control is aligned to the bottom when ALIGNMENT_END is used.");

		vflow_container->set_alignment(FlowContainer::ALIGNMENT_BEGIN);
		vflow_container->set_layout_direction(Control::LAYOUT_DIRECTION_RTL);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control_1->get_position().is_equal_approx(Point2(80, 0)),
				"First child control moves to the right edge in RTL mode with vertical flow.");

		CHECK_MESSAGE(
				child_control_2->get_position().is_equal_approx(Point2(80, 20)),
				"Second child control follows RTL horizontal mirroring in vertical flow.");

		memdelete(child_control_2);
		memdelete(child_control_1);
	}

	SUBCASE("Expanding children and stretch ratio") {
		Control *child_control_1 = memnew(Control);
		Control *child_control_2 = memnew(Control);
		child_control_1->set_custom_minimum_size(Size2(20, 20));
		child_control_2->set_custom_minimum_size(Size2(20, 20));
		child_control_1->set_v_size_flags(Control::SIZE_EXPAND_FILL);
		child_control_2->set_v_size_flags(Control::SIZE_EXPAND_FILL);
		vflow_container->set_size(Size2(100, 100));
		vflow_container->add_child(child_control_1);
		vflow_container->add_child(child_control_2);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control_1->get_size().is_equal_approx(Size2(20, 50)),
				"First child control expands to half of the available height when both children have equal stretch ratio.");

		CHECK_MESSAGE(
				child_control_2->get_size().is_equal_approx(Size2(20, 50)),
				"Second child control expands to half of the available height when both children have equal stretch ratio.");

		CHECK_MESSAGE(
				child_control_2->get_position().is_equal_approx(Point2(0, 50)),
				"Second child control is placed immediately after the first expanded child control.");

		child_control_2->set_stretch_ratio(3.0f);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control_1->get_size().is_equal_approx(Size2(20, 35)),
				"First child control receives less height when its stretch ratio is lower.");

		CHECK_MESSAGE(
				child_control_2->get_size().is_equal_approx(Size2(20, 65)),
				"Second child control receives more height when its stretch ratio is higher.");

		CHECK_MESSAGE(
				child_control_2->get_position().is_equal_approx(Point2(0, 35)),
				"Second child control starts after the first child control's stretched height.");

		memdelete(child_control_2);
		memdelete(child_control_1);
	}

	SUBCASE("Reverse fill") {
		Control *child_control_1 = memnew(Control);
		Control *child_control_2 = memnew(Control);
		Control *child_control_3 = memnew(Control);
		child_control_1->set_custom_minimum_size(Size2(20, 20));
		child_control_2->set_custom_minimum_size(Size2(20, 20));
		child_control_3->set_custom_minimum_size(Size2(20, 20));
		vflow_container->set_size(Size2(100, 30));
		vflow_container->add_child(child_control_1);
		vflow_container->add_child(child_control_2);
		vflow_container->add_child(child_control_3);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control_1->get_position().is_equal_approx(Point2(0, 0)),
				"First child control starts in the first column by default.");

		CHECK_MESSAGE(
				child_control_2->get_position().is_equal_approx(Point2(20, 0)),
				"Second child control starts in the second column by default.");

		CHECK_MESSAGE(
				child_control_3->get_position().is_equal_approx(Point2(40, 0)),
				"Third child control starts in the third column by default.");

		vflow_container->set_reverse_fill(true);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control_1->get_position().is_equal_approx(Point2(80, 0)),
				"First child control starts from the right when reverse fill is enabled in vertical mode.");

		CHECK_MESSAGE(
				child_control_2->get_position().is_equal_approx(Point2(60, 0)),
				"Second child control follows right-to-left column filling when reverse fill is enabled.");

		CHECK_MESSAGE(
				child_control_3->get_position().is_equal_approx(Point2(40, 0)),
				"Third child control continues right-to-left column filling when reverse fill is enabled.");

		vflow_container->set_layout_direction(Control::LAYOUT_DIRECTION_RTL);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control_1->get_position().is_equal_approx(Point2(0, 0)),
				"Vertical flow with RTL and reverse fill fills columns from left to right.");

		CHECK_MESSAGE(
				child_control_2->get_position().is_equal_approx(Point2(20, 0)),
				"Second child control follows left-to-right order in vertical mode with RTL and reverse fill.");

		CHECK_MESSAGE(
				child_control_3->get_position().is_equal_approx(Point2(40, 0)),
				"Third child control follows left-to-right order in vertical mode with RTL and reverse fill.");

		memdelete(child_control_3);
		memdelete(child_control_2);
		memdelete(child_control_1);
	}

	memdelete(vflow_container);
}

} // namespace TestFlowContainer
