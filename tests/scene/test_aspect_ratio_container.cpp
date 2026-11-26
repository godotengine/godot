/**************************************************************************/
/*  test_aspect_ratio_container.cpp                                       */
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

TEST_FORCE_LINK(test_aspect_ratio_container)

#include "scene/gui/aspect_ratio_container.h"
#include "scene/gui/control.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"

namespace TestAspectRatioContainer {

TEST_CASE("[SceneTree][AspectRatioContainer] Default Properties") {
	AspectRatioContainer *aspect_ratio_container = memnew(AspectRatioContainer);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(aspect_ratio_container);
	Control *child_control = memnew(Control);
	aspect_ratio_container->add_child(child_control);

	CHECK_MESSAGE(
			aspect_ratio_container->get_stretch_mode() == AspectRatioContainer::STRETCH_FIT,
			"AspectRatioContainer stretch mode is set to STRETCH_FIT by default.");

	CHECK_MESSAGE(
			Math::is_equal_approx(aspect_ratio_container->get_ratio(), 1.0f),
			"AspectRatioContainer ratio is set to 1.0 by default.");

	CHECK_MESSAGE(
			child_control->get_position().is_equal_approx(Point2(0, 0)),
			"Child control is positioned at the top-left corner of the AspectRatioContainer by default.");

	CHECK_MESSAGE(
			child_control->get_h_size_flags() == Control::SIZE_FILL,
			"Child control horizontal size flags are set to SIZE_FILL by default.");

	CHECK_MESSAGE(
			child_control->get_v_size_flags() == Control::SIZE_FILL,
			"Child control vertical size flags are set to SIZE_FILL by default.");

	memdelete(child_control);
	memdelete(aspect_ratio_container);
}

TEST_CASE("[SceneTree][AspectRatioContainer] Aspect Ratio") {
	AspectRatioContainer *aspect_ratio_container = memnew(AspectRatioContainer);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(aspect_ratio_container);
	Control *child_control = memnew(Control);
	aspect_ratio_container->add_child(child_control);

	aspect_ratio_container->set_size(Size2(200, 100));
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			child_control->get_size().is_equal_approx(Size2(100, 100)),
			"Child control is resized to maintain aspect ratio when AspectRatioContainer is resized.");

	aspect_ratio_container->set_size(Size2(100, 200));
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			child_control->get_size().is_equal_approx(Size2(100, 100)),
			"Child control is resized to maintain aspect ratio when AspectRatioContainer is resized.");

	aspect_ratio_container->set_ratio(2.0f);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			child_control->get_size().is_equal_approx(Size2(100, 50)),
			"Child control is resized to maintain aspect ratio when AspectRatioContainer ratio is changed.");

	aspect_ratio_container->set_ratio(0.5f);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			child_control->get_size().is_equal_approx(Size2(100, 200)),
			"Child control is resized to maintain aspect ratio when AspectRatioContainer ratio is changed.");

	aspect_ratio_container->set_size(Size2(200, 100));
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			child_control->get_size().is_equal_approx(Size2(50, 100)),
			"Child control is resized to maintain aspect ratio when AspectRatioContainer is resized after ratio change.");

	memdelete(child_control);
	memdelete(aspect_ratio_container);
}

TEST_CASE("[SceneTree][AspectRatioContainer] Stretch Modes") {
	AspectRatioContainer *aspect_ratio_container = memnew(AspectRatioContainer);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(aspect_ratio_container);
	Control *child_control = memnew(Control);
	aspect_ratio_container->add_child(child_control);

	aspect_ratio_container->set_size(Size2(200, 100));
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			child_control->get_size().is_equal_approx(Size2(100, 100)),
			"Child control is resized to maintain aspect ratio in STRETCH_FIT mode.");

	aspect_ratio_container->set_stretch_mode(AspectRatioContainer::STRETCH_WIDTH_CONTROLS_HEIGHT);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			child_control->get_size().is_equal_approx(Size2(200, 200)),
			"Child control height is adjusted to maintain aspect ratio in STRETCH_WIDTH_CONTROLS_HEIGHT mode.");

	aspect_ratio_container->set_stretch_mode(AspectRatioContainer::STRETCH_HEIGHT_CONTROLS_WIDTH);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			child_control->get_size().is_equal_approx(Size2(100, 100)),
			"Child control width is adjusted to maintain aspect ratio in STRETCH_HEIGHT_CONTROLS_WIDTH mode.");

	aspect_ratio_container->set_stretch_mode(AspectRatioContainer::STRETCH_COVER);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			child_control->get_size().is_equal_approx(Size2(200, 200)),
			"Child control is resized to cover the entire AspectRatioContainer in STRETCH_COVER mode.");

	aspect_ratio_container->set_size(Size2(100, 200));
	aspect_ratio_container->set_stretch_mode(AspectRatioContainer::STRETCH_FIT);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			aspect_ratio_container->get_stretch_mode() == AspectRatioContainer::STRETCH_FIT,
			"AspectRatioContainer stretch mode is set to STRETCH_FIT by default.");

	CHECK_MESSAGE(
			child_control->get_size().is_equal_approx(Size2(100, 100)),
			"Child control is resized to maintain aspect ratio in STRETCH_FIT mode.");

	aspect_ratio_container->set_stretch_mode(AspectRatioContainer::STRETCH_WIDTH_CONTROLS_HEIGHT);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			child_control->get_size().is_equal_approx(Size2(100, 100)),
			"Child control height is adjusted to maintain aspect ratio in STRETCH_WIDTH_CONTROLS_HEIGHT mode.");

	aspect_ratio_container->set_stretch_mode(AspectRatioContainer::STRETCH_HEIGHT_CONTROLS_WIDTH);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			child_control->get_size().is_equal_approx(Size2(200, 200)),
			"Child control width is adjusted to maintain aspect ratio in STRETCH_HEIGHT_CONTROLS_WIDTH mode.");

	aspect_ratio_container->set_stretch_mode(AspectRatioContainer::STRETCH_COVER);
	SceneTree::get_singleton()->process(0);

	CHECK_MESSAGE(
			child_control->get_size().is_equal_approx(Size2(200, 200)),
			"Child control is resized to cover the entire AspectRatioContainer in STRETCH_COVER mode.");

	memdelete(child_control);
	memdelete(aspect_ratio_container);
}

TEST_CASE("[SceneTree][AspectRatioContainer] Alignment modes") {
	AspectRatioContainer *aspect_ratio_container = memnew(AspectRatioContainer);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(aspect_ratio_container);
	Control *child_control = memnew(Control);
	aspect_ratio_container->add_child(child_control);

	SUBCASE("Horizontal Alignment") {
		aspect_ratio_container->set_size(Size2(200, 100));
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				aspect_ratio_container->get_alignment_horizontal() == AspectRatioContainer::ALIGNMENT_CENTER,
				"AspectRatioContainer horizontal alignment is set to ALIGNMENT_CENTER by default.");

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(50, 0)),
				"Child control is aligned to the center in ALIGNMENT_CENTER horizontal alignment mode.");

		aspect_ratio_container->set_alignment_horizontal(AspectRatioContainer::ALIGNMENT_BEGIN);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(0, 0)),
				"Child control is aligned to the left in ALIGNMENT_BEGIN horizontal alignment mode.");

		aspect_ratio_container->set_alignment_horizontal(AspectRatioContainer::ALIGNMENT_END);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(100, 0)),
				"Child control is aligned to the right in ALIGNMENT_END horizontal alignment mode.");
	}

	SUBCASE("Vertical Alignment") {
		aspect_ratio_container->set_size(Size2(100, 200));
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				aspect_ratio_container->get_alignment_vertical() == AspectRatioContainer::ALIGNMENT_CENTER,
				"AspectRatioContainer vertical alignment is set to ALIGNMENT_CENTER by default.");

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(0, 50)),
				"Child control is aligned to the center in ALIGNMENT_CENTER vertical alignment mode.");

		aspect_ratio_container->set_alignment_vertical(AspectRatioContainer::ALIGNMENT_BEGIN);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(0, 0)),
				"Child control is aligned to the top in ALIGNMENT_BEGIN vertical alignment mode.");

		aspect_ratio_container->set_alignment_vertical(AspectRatioContainer::ALIGNMENT_END);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(0, 100)),
				"Child control is aligned to the bottom in ALIGNMENT_END vertical alignment mode.");
	}

	memdelete(child_control);
	memdelete(aspect_ratio_container);
}

TEST_CASE("[SceneTree][AspectRatioContainer] Container Sizing Flags") {
	AspectRatioContainer *aspect_ratio_container = memnew(AspectRatioContainer);
	Window *root = SceneTree::get_singleton()->get_root();
	root->add_child(aspect_ratio_container);
	Control *child_control = memnew(Control);
	aspect_ratio_container->add_child(child_control);

	SUBCASE("Horizontal Flags") {
		aspect_ratio_container->set_size(Size2(200, 100));
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(50, 0)),
				"Child control is vertically centered by default with SIZE_FILL.");

		child_control->set_h_size_flags(Control::SIZE_SHRINK_BEGIN);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(50, 0)),
				"Child control is aligned to the left of its would-be area with SIZE_FILL in SIZE_SHRINK_BEGIN horizontal size flag mode.");

		child_control->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(100, 0)),
				"Child control is centered in its would-be area with SIZE_FILL in SIZE_SHRINK_CENTER horizontal size flag mode.");

		child_control->set_h_size_flags(Control::SIZE_SHRINK_END);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(150, 0)),
				"Child control is aligned to the right of its would-be area with SIZE_FILL in SIZE_SHRINK_END horizontal size flag mode.");
	}

	SUBCASE("Horizontal Flags RTL") {
		aspect_ratio_container->set_size(Size2(200, 100));
		aspect_ratio_container->set_layout_direction(Control::LAYOUT_DIRECTION_RTL);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(50, 0)),
				"Child control is vertically centered by default with SIZE_FILL.");

		child_control->set_h_size_flags(Control::SIZE_SHRINK_BEGIN);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(150, 0)),
				"Child control is aligned to the right of its would-be area with SIZE_FILL in RTL SIZE_SHRINK_BEGIN horizontal size flag mode.");

		child_control->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(100, 0)),
				"Child control is centered in its would-be area with SIZE_FILL in SIZE_SHRINK_CENTER horizontal size flag mode.");

		child_control->set_h_size_flags(Control::SIZE_SHRINK_END);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(50, 0)),
				"Child control is aligned to the left of its would-be area with SIZE_FILL in RTL SIZE_SHRINK_END horizontal size flag mode.");
	}

	SUBCASE("Vertical Flags") {
		aspect_ratio_container->set_size(Size2(100, 200));
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(0, 50)),
				"Child control is horizontally centered by default with SIZE_FILL.");

		child_control->set_v_size_flags(Control::SIZE_SHRINK_BEGIN);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(0, 50)),
				"Child control is aligned to the top of its would-be area with SIZE_FILL in SIZE_SHRINK_BEGIN vertical size flag mode.");

		child_control->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(0, 100)),
				"Child control is centered in its would-be area with SIZE_FILL in SIZE_SHRINK_CENTER vertical size flag mode.");

		child_control->set_v_size_flags(Control::SIZE_SHRINK_END);
		SceneTree::get_singleton()->process(0);

		CHECK_MESSAGE(
				child_control->get_position().is_equal_approx(Point2(0, 150)),
				"Child control is aligned to the bottom of its would-be area with SIZE_FILL in SIZE_SHRINK_END vertical size flag mode.");
	}

	memdelete(child_control);
	memdelete(aspect_ratio_container);
}

} // namespace TestAspectRatioContainer
