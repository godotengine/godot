/**************************************************************************/
/*  test_box_container.h                                                  */
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

#include "scene/gui/box_container.h"
#include "scene/main/window.h"

#include "tests/test_macros.h"

namespace TestBoxContainer {
static inline void check_sizes(Container *container, Vector<float> sizes, int sep, float offset = 0, bool horizontal = true) {
	CHECK(container->get_child_count() == sizes.size());
	float last_pos = offset;
	for (int i = 0; i < container->get_child_count(); i++) {
		// Assuming there is no invalid children.
		Control *c = Object::cast_to<Control>(container->get_child(i));
		const float size = sizes[i];
		Rect2 rect;
		if (horizontal) {
			rect.size.y = container->get_size().y;
			rect.position.x = last_pos;
			rect.size.x = size;
		} else {
			rect.size.x = container->get_size().x;
			rect.position.y = last_pos;
			rect.size.y = size;
		}
		CHECK_MESSAGE(c->get_rect().is_equal_approx(rect), vformat("Child %s is the wrong size. Expected %s. Actual %s.", i, rect, c->get_rect()));
		last_pos = last_pos + size + sep;
	}
}

static inline void check_sizes_rtl(Container *container, Vector<float> sizes, int sep, float offset = 0) {
	CHECK(container->get_child_count() == sizes.size());
	float last_pos = container->get_size().x - offset;
	for (int i = 0; i < container->get_child_count(); i++) {
		// Assuming there is no invalid children.
		Control *c = Object::cast_to<Control>(container->get_child(i));
		const float size = sizes[i];
		Rect2 rect;
		rect.size.y = container->get_size().y;
		rect.position.x = last_pos - size;
		rect.size.x = size;
		CHECK_MESSAGE(c->get_rect().is_equal_approx(rect), vformat("Child %s is the wrong size. Expected %s. Actual %s.", i, rect, c->get_rect()));
		last_pos = last_pos - size - sep;
	}
}

static inline void set_size_flags(Container *container, Vector<float> expand_ratios, bool horizontal = true) {
	for (int i = 0; i < container->get_child_count(); i++) {
		Control *c = Object::cast_to<Control>(container->get_child(i));
		const float ratio = expand_ratios[i];
		if (horizontal) {
			c->set_h_size_flags(ratio > 0 ? Control::SIZE_EXPAND_FILL : Control::SIZE_FILL);
		} else {
			c->set_v_size_flags(ratio > 0 ? Control::SIZE_EXPAND_FILL : Control::SIZE_FILL);
		}
		if (ratio > 0) {
			c->set_stretch_ratio(ratio);
		}
	}
	MessageQueue::get_singleton()->flush();
}

TEST_CASE("[SceneTree][BoxContainer] Two children") {
	BoxContainer *box_container = memnew(BoxContainer);
	box_container->set_size(Size2(500, 500));
	SceneTree::get_singleton()->get_root()->add_child(box_container);

	Control *child_a = memnew(Control);
	Control *child_b = memnew(Control);
	box_container->add_child(child_a);
	box_container->add_child(child_b);

	const Size2 min_size = Size2(10, 10);
	child_a->set_custom_minimum_size(min_size);
	child_b->set_custom_minimum_size(min_size);
	MessageQueue::get_singleton()->flush();

	const int separation = box_container->get_theme_constant("separation");

	SUBCASE("[BoxContainer] Horizontal default sizes") {
		CHECK_FALSE(box_container->is_vertical());

		set_size_flags(box_container, { -1, -1 }); // None expanded.
		check_sizes(box_container, { min_size.x, min_size.x }, separation);

		set_size_flags(box_container, { 1, -1 }); // First expanded.
		float child_1_expanded_size = box_container->get_size().x - separation - min_size.x;
		check_sizes(box_container, { child_1_expanded_size, min_size.x }, separation);

		set_size_flags(box_container, { -1, 1 }); // Second expanded.
		check_sizes(box_container, { min_size.x, child_1_expanded_size }, separation);

		set_size_flags(box_container, { 1, 1 }); // Both expanded.
		float child_2_expanded_size = (box_container->get_size().x - separation) / 2;
		check_sizes(box_container, { child_2_expanded_size, child_2_expanded_size }, separation);
	}

	SUBCASE("[BoxContainer] Vertical default sizes") {
		box_container->set_vertical(true);
		CHECK(box_container->is_vertical());

		set_size_flags(box_container, { -1, -1 }, false); // None expanded.
		check_sizes(box_container, { min_size.y, min_size.y }, separation, 0, false);

		set_size_flags(box_container, { 1, -1 }, false); // First expanded.
		float child_1_expanded_size = box_container->get_size().y - separation - min_size.y;
		check_sizes(box_container, { child_1_expanded_size, min_size.y }, separation, 0, false);

		set_size_flags(box_container, { -1, 1 }, false); // Second expanded.
		check_sizes(box_container, { min_size.y, child_1_expanded_size }, separation, 0, false);

		set_size_flags(box_container, { 1, 1 }, false); // Both expanded.
		float child_2_expanded_size = (box_container->get_size().y - separation) / 2;
		check_sizes(box_container, { child_2_expanded_size, child_2_expanded_size }, separation, 0, false);
	}

	SUBCASE("[BoxContainer] Right to left") {
		box_container->set_layout_direction(Control::LAYOUT_DIRECTION_RTL);

		set_size_flags(box_container, { -1, -1 }); // None expanded.
		check_sizes_rtl(box_container, { min_size.x, min_size.x }, separation);

		set_size_flags(box_container, { 1, -1 }); // First expanded.
		float child_1_expanded_size = box_container->get_size().x - separation - min_size.x;
		check_sizes_rtl(box_container, { child_1_expanded_size, min_size.x }, separation);

		set_size_flags(box_container, { -1, 1 }); // Second expanded.
		check_sizes_rtl(box_container, { min_size.x, child_1_expanded_size }, separation);

		set_size_flags(box_container, { 1, 1 }); // Both expanded.
		float child_2_expanded_size = (box_container->get_size().x - separation) / 2;
		check_sizes_rtl(box_container, { child_2_expanded_size, child_2_expanded_size }, separation);
	}

	SUBCASE("[BoxContainer] Horizontal alignment") {
		set_size_flags(box_container, { -1, -1 }); // None expanded.

		// Alignment Begin.
		box_container->set_alignment(BoxContainer::ALIGNMENT_BEGIN);
		MessageQueue::get_singleton()->flush();
		check_sizes(box_container, { min_size.x, min_size.x }, separation, 0);

		// Alignment End.
		box_container->set_alignment(BoxContainer::ALIGNMENT_END);
		MessageQueue::get_singleton()->flush();
		check_sizes(box_container, { min_size.x, min_size.x }, separation, box_container->get_size().x - min_size.x * 2 - separation);

		// Alignment Center.
		box_container->set_alignment(BoxContainer::ALIGNMENT_CENTER);
		MessageQueue::get_singleton()->flush();
		check_sizes(box_container, { min_size.x, min_size.x }, separation, (box_container->get_size().x - min_size.x * 2 - separation) / 2);
	}

	SUBCASE("[BoxContainer] Vertical alignment") {
		box_container->set_vertical(true);
		set_size_flags(box_container, { -1, -1 }, false); // None expanded.

		// Alignment Begin.
		box_container->set_alignment(BoxContainer::ALIGNMENT_BEGIN);
		MessageQueue::get_singleton()->flush();
		check_sizes(box_container, { min_size.y, min_size.y }, separation, 0, false);

		// Alignment End.
		box_container->set_alignment(BoxContainer::ALIGNMENT_END);
		MessageQueue::get_singleton()->flush();
		check_sizes(box_container, { min_size.y, min_size.y }, separation, box_container->get_size().y - min_size.y * 2 - separation, false);

		// Alignment Center.
		box_container->set_alignment(BoxContainer::ALIGNMENT_CENTER);
		MessageQueue::get_singleton()->flush();
		check_sizes(box_container, { min_size.y, min_size.y }, separation, (box_container->get_size().y - min_size.y * 2 - separation) / 2, false);
	}

	SUBCASE("[BoxContainer] Right to left alignment") {
		box_container->set_layout_direction(Control::LAYOUT_DIRECTION_RTL);
		set_size_flags(box_container, { -1, -1 }); // None expanded.

		// Alignment Begin.
		box_container->set_alignment(BoxContainer::ALIGNMENT_BEGIN);
		MessageQueue::get_singleton()->flush();
		check_sizes_rtl(box_container, { min_size.x, min_size.x }, separation, 0);

		// Alignment End.
		box_container->set_alignment(BoxContainer::ALIGNMENT_END);
		MessageQueue::get_singleton()->flush();
		check_sizes_rtl(box_container, { min_size.x, min_size.x }, separation, box_container->get_size().x - min_size.x * 2 - separation);

		// Alignment Center.
		box_container->set_alignment(BoxContainer::ALIGNMENT_CENTER);
		MessageQueue::get_singleton()->flush();
		check_sizes_rtl(box_container, { min_size.x, min_size.x }, separation, (box_container->get_size().x - min_size.x * 2 - separation) / 2);
	}

	SUBCASE("[BoxContainer] Alignment with size flags") {
		// Alignments do not affect anything if any children expand.
		// Alignment Begin.
		box_container->set_alignment(BoxContainer::ALIGNMENT_BEGIN);

		set_size_flags(box_container, { 1, -1 }); // First expanded.
		float child_1_expanded_size = box_container->get_size().x - separation - min_size.x;
		check_sizes(box_container, { child_1_expanded_size, min_size.x }, separation);

		set_size_flags(box_container, { -1, 1 }); // Second expanded.
		check_sizes(box_container, { min_size.x, child_1_expanded_size }, separation);

		set_size_flags(box_container, { 1, 1 }); // Both expanded.
		float child_2_expanded_size = (box_container->get_size().x - separation) / 2;
		check_sizes(box_container, { child_2_expanded_size, child_2_expanded_size }, separation);

		// Alignment End.
		box_container->set_alignment(BoxContainer::ALIGNMENT_END);

		set_size_flags(box_container, { 1, -1 }); // First expanded.
		check_sizes(box_container, { child_1_expanded_size, min_size.x }, separation);

		set_size_flags(box_container, { -1, 1 }); // Second expanded.
		check_sizes(box_container, { min_size.x, child_1_expanded_size }, separation);

		set_size_flags(box_container, { 1, 1 }); // Both expanded.
		check_sizes(box_container, { child_2_expanded_size, child_2_expanded_size }, separation);

		// Alignment Center.
		box_container->set_alignment(BoxContainer::ALIGNMENT_CENTER);

		set_size_flags(box_container, { 1, -1 }); // First expanded.
		check_sizes(box_container, { child_1_expanded_size, min_size.x }, separation);

		set_size_flags(box_container, { -1, 1 }); // Second expanded.
		check_sizes(box_container, { min_size.x, child_1_expanded_size }, separation);

		set_size_flags(box_container, { 1, 1 }); // Both expanded.
		check_sizes(box_container, { child_2_expanded_size, child_2_expanded_size }, separation);
	}

	SUBCASE("[BoxContainer] No minimum sizes") {
		child_a->set_custom_minimum_size(Size2(0, 0));
		child_b->set_custom_minimum_size(Size2(0, 0));

		set_size_flags(box_container, { -1, -1 }); // None expanded.
		check_sizes(box_container, { 0, 0 }, separation);

		set_size_flags(box_container, { 1, -1 }); // First expanded.
		float child_1_expanded_size = box_container->get_size().x - separation;
		check_sizes(box_container, { child_1_expanded_size, 0 }, separation);

		set_size_flags(box_container, { -1, 1 }); // Second expanded.
		check_sizes(box_container, { 0, child_1_expanded_size }, separation);

		set_size_flags(box_container, { 1, 1 }); // Both expanded.
		float child_2_expanded_size = (box_container->get_size().x - separation) / 2;
		check_sizes(box_container, { child_2_expanded_size, child_2_expanded_size }, separation);
	}

	memdelete(child_b);
	memdelete(child_a);
	memdelete(box_container);
}

TEST_CASE("[SceneTree][BoxContainer] Three children") {
	BoxContainer *box_container = memnew(BoxContainer);
	box_container->set_size(Size2(500.5, 500.5));
	SceneTree::get_singleton()->get_root()->add_child(box_container);

	Control *child_a = memnew(Control);
	Control *child_b = memnew(Control);
	Control *child_c = memnew(Control);
	box_container->add_child(child_a);
	box_container->add_child(child_b);
	box_container->add_child(child_c);

	const Size2 min_size = Size2(10, 10);
	child_a->set_custom_minimum_size(min_size);
	child_b->set_custom_minimum_size(min_size);
	child_c->set_custom_minimum_size(min_size);
	MessageQueue::get_singleton()->flush();

	const int separation = box_container->get_theme_constant("separation");

	SUBCASE("[BoxContainer] Default sizes") {
		set_size_flags(box_container, { -1, -1, -1 }); // None expanded.
		check_sizes(box_container, { min_size.x, min_size.x, min_size.x }, separation);

		set_size_flags(box_container, { 1, -1, -1 }); // First expanded.
		float child_1_expanded_size = box_container->get_size().x - separation * 2 - min_size.x * 2;
		check_sizes(box_container, { child_1_expanded_size, min_size.x, min_size.x }, separation);

		set_size_flags(box_container, { -1, 1, -1 }); // Second expanded.
		check_sizes(box_container, { min_size.x, child_1_expanded_size, min_size.x }, separation);

		set_size_flags(box_container, { -1, -1, 1 }); // Third expanded.
		check_sizes(box_container, { min_size.x, min_size.x, child_1_expanded_size }, separation);

		set_size_flags(box_container, { 1, 1, -1 }); // First and second expanded.
		float child_2_expanded_size = (box_container->get_size().x - min_size.x - separation * 2) / 2;
		check_sizes(box_container, { child_2_expanded_size, child_2_expanded_size, min_size.x }, separation);

		set_size_flags(box_container, { 1, -1, 1 }); // First and third expanded.
		check_sizes(box_container, { child_2_expanded_size, min_size.x, child_2_expanded_size }, separation);

		set_size_flags(box_container, { -1, 1, 1 }); // Second and third expanded.
		check_sizes(box_container, { min_size.x, child_2_expanded_size, child_2_expanded_size }, separation);

		set_size_flags(box_container, { 1, 1, 1 }); // All expanded.
		float child_3_expanded_size = (box_container->get_size().x - separation * 2) / 3;
		check_sizes(box_container, { child_3_expanded_size, child_3_expanded_size, child_3_expanded_size }, separation);

		set_size_flags(box_container, { 1, 2, 3 }); // All expanded, different ratios.
		float child_6_expanded_size = (box_container->get_size().x - separation * 2) / 6;
		check_sizes(box_container, { child_6_expanded_size, child_6_expanded_size * 2, child_6_expanded_size * 3 }, separation);
	}

	SUBCASE("[BoxContainer] Resize and alignment") {
		set_size_flags(box_container, { -1, -1, -1 }); // None expanded.

		// Alignment Begin.
		box_container->set_alignment(BoxContainer::ALIGNMENT_BEGIN);
		box_container->set_size(Size2(500, 500));
		MessageQueue::get_singleton()->flush();
		check_sizes(box_container, { min_size.x, min_size.x, min_size.x }, separation);

		// Decrease size.
		box_container->set_size(Size2(200, 200));
		MessageQueue::get_singleton()->flush();
		check_sizes(box_container, { min_size.x, min_size.x, min_size.x }, separation);

		// Increase size.
		box_container->set_size(Size2(600, 600));
		MessageQueue::get_singleton()->flush();
		check_sizes(box_container, { min_size.x, min_size.x, min_size.x }, separation);

		// Alignment End.
		box_container->set_alignment(BoxContainer::ALIGNMENT_END);
		box_container->set_size(Size2(500, 500));
		MessageQueue::get_singleton()->flush();
		check_sizes(box_container, { min_size.x, min_size.x, min_size.x }, separation, box_container->get_size().x - min_size.x * 3 - separation * 2);

		// Decrease size.
		box_container->set_size(Size2(200, 200));
		MessageQueue::get_singleton()->flush();
		check_sizes(box_container, { min_size.x, min_size.x, min_size.x }, separation, box_container->get_size().x - min_size.x * 3 - separation * 2);

		// Increase size.
		box_container->set_size(Size2(600, 600));
		MessageQueue::get_singleton()->flush();
		check_sizes(box_container, { min_size.x, min_size.x, min_size.x }, separation, box_container->get_size().x - min_size.x * 3 - separation * 2);

		// Alignment Center.
		box_container->set_alignment(BoxContainer::ALIGNMENT_CENTER);
		box_container->set_size(Size2(500, 500));
		MessageQueue::get_singleton()->flush();
		check_sizes(box_container, { min_size.x, min_size.x, min_size.x }, separation, (box_container->get_size().x - min_size.x * 3 - separation * 2) / 2);

		// Decrease size.
		box_container->set_size(Size2(200, 200));
		MessageQueue::get_singleton()->flush();
		check_sizes(box_container, { min_size.x, min_size.x, min_size.x }, separation, (box_container->get_size().x - min_size.x * 3 - separation * 2) / 2);

		// Increase size.
		box_container->set_size(Size2(600, 600));
		MessageQueue::get_singleton()->flush();
		check_sizes(box_container, { min_size.x, min_size.x, min_size.x }, separation, (box_container->get_size().x - min_size.x * 3 - separation * 2) / 2);
	}

	SUBCASE("[BoxContainer] Large minimum size") {
		// Minimum size makes the child larger than it would be from its expand flag.
		Size2 large_min_size = Size2(250, 250);
		child_a->set_custom_minimum_size(large_min_size);

		// Minimum size clamps size, with only 1 other expanding child.
		set_size_flags(box_container, { 1, 1, -1 }); // First and second expanded.
		check_sizes(box_container, { large_min_size.x, box_container->get_size().x - large_min_size.x - min_size.x - separation * 2, min_size.x }, separation);

		// Minimum size clamps size, with 2 other expanding children.
		set_size_flags(box_container, { 1, 1, 1 }); // All expanded.
		float rem_size = (box_container->get_size().x - large_min_size.x - separation * 2) / 2;
		check_sizes(box_container, { large_min_size.x, rem_size, rem_size }, separation);

		// Minimum size clamps on last child.
		child_a->set_custom_minimum_size(min_size);
		child_c->set_custom_minimum_size(large_min_size);
		set_size_flags(box_container, { 1, 1, 1 }); // All expanded.
		check_sizes(box_container, { rem_size, rem_size, large_min_size.x }, separation);
	}

	memdelete(box_container);
}

TEST_CASE("[SceneTree][BoxContainer] Many children") {
	BoxContainer *box_container = memnew(BoxContainer);
	box_container->set_size(Size2(500, 500));
	SceneTree::get_singleton()->get_root()->add_child(box_container);
	const int separation = box_container->get_theme_constant("separation");

	// Having many children set to expand will size them all evenly.
	const int num_children = 30;
	for (int i = 0; i < num_children; i++) {
		Control *new_child = memnew(Control);
		new_child->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		box_container->add_child(new_child);
	}

	MessageQueue::get_singleton()->flush();

	PackedFloat32Array sizes;
	sizes.resize(num_children);
	float *sizes_w = sizes.ptrw();
	float child_size = (box_container->get_size().x - separation * (num_children - 1)) / num_children;
	for (int i = 0; i < num_children; i++) {
		sizes_w[i] = child_size;
	}

	check_sizes(box_container, sizes, separation);

	memdelete(box_container);
}

} // namespace TestBoxContainer
